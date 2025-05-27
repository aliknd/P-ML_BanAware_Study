#!/usr/bin/env python3
"""
compare_pipelines.py
====================

Now:
  - One deterministic train/test split per user (seeded at top).
  - Global-SSL & Global-Supervised each train only on train-day windows.
  - One shared CNN per (fruit,scenario) saved under `global_cnns/...`
    (like encoders under `_global_encoders/...`).
  - Shared CNN and encoders get copied into each user’s folder.

Optional class balancing (--sample-mode):
  - Modes: original, undersample, oversample (default=original).
  - Undersample: “round_robin_undersample” drops excess negatives
    across users in a round‐robin fashion until total_neg == total_pos.
  - Oversample: “round_robin_oversample” duplicates positive windows
    (with small Gaussian jitter on their hr_seq/st_seq) in turn
    until total_pos == total_neg.
  - All original vs. new counts (per‐user and global) plus number of
    added/removed samples are logged in each pipeline’s split_details.txt:
      – Global pipelines show a “GLOBAL …” summary and per‐user breakdown.
      – Personal‐SSL appends a “POST-SAMPLING SUMMARY” block.
  - To use: pass `--sample-mode undersample` or `--sample-mode oversample`
    when invoking the script; omit or use `original` to leave data untouched.
"""

import argparse, warnings, os, shutil, sys, random
from pathlib import Path

import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D,
                                     GlobalAveragePooling1D,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model

# ─── Seed everything for split reproducibility ─────────────────────────────
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ─── project helpers ───────────────────────────────────────────────────────
from src.classifier_utils import (
    BASE_DATA_DIR, ALLOWED_SCENARIOS,
    load_signal_data, load_label_data, process_label_window
)
from src.signal_utils import (
    WINDOW_SIZE, STEP_SIZE, create_windows,
    build_simclr_encoder, create_projection_head, train_simclr
)
from src.chart_utils import (
    bootstrap_threshold_metrics, plot_thresholds, plot_ssl_losses
)

# ─── Hyperparameters ──────────────────────────────────────────────────────
GS_EPOCHS, GS_BATCH, GS_LR, GS_PATIENCE = 200, 32, 1e-3, 15
BATCH_SSL, SSL_EPOCHS                   = 32, 100
CLF_EPOCHS, CLF_PATIENCE                = 200, 15
WINDOW_LEN                              = WINDOW_SIZE


# ─── Plot helpers ─────────────────────────────────────────────────────────
def plot_clf_losses(train, val, out_dir, fname):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train, label="Train")
    plt.plot(val,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Binary CE")
    plt.title(fname.replace('_', ' ').title())
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}.png")
    plt.close()

# New helper: sample timestamps where no event occurs
def derive_negative_labels(hr_df, pos_df, n_samples,
                          window_hours=1, random_state=42):
    """
    Sample up to n_samples timestamps from hr_df.index such that
    no pos_df event occurs within ±window_hours of any sampled time.
    """
    # round to minute
    all_times   = pd.DatetimeIndex(hr_df.index.round('T').unique())
    event_times = pos_df['hawaii_createdat_time'].dt.round('T').unique()

    # Build mask of valid times
    valid = []
    w = pd.Timedelta(hours=window_hours)
    for t in all_times:
        # check no event within ±w
        if not ((event_times >= (t - w)) & (event_times <= (t + w))).any():
            valid.append(t)

    if not valid:
        return pd.DataFrame(columns=['hawaii_createdat_time'])

    k       = min(len(valid), n_samples)
    sampled = pd.to_datetime(np.random.default_rng(random_state).choice(valid, size=k, replace=False))
    return pd.DataFrame({'hawaii_createdat_time': sampled})

# ─── Split / window helpers ────────────────────────────────────────────────
def _train_test_days_by_samples(pos_df, neg_df, hr_df, st_df):
    events = pd.concat([pos_df, neg_df])
    if events.empty:
        return np.array([]), np.array([])

    days = list(np.sort(events['hawaii_createdat_time'].dt.date.unique()))
    counts = []
    for d in days:
        p = pos_df[pos_df['hawaii_createdat_time'].dt.date == d]
        n = neg_df[neg_df['hawaii_createdat_time'].dt.date == d]
        rows = pd.concat([
            process_label_window(p, hr_df, st_df, 1),
            process_label_window(n, hr_df, st_df, 0)
        ])
        counts.append(len(rows))

    total = sum(counts)
    if total == 0:
        random.shuffle(days)
        cut = int(round(0.75 * len(days)))
        return np.array(days[:cut]), np.array(days[cut:])

    day_counts = list(zip(days, counts))
    random.shuffle(day_counts)

    target = 0.25 * total
    te_days, cum = [], 0
    for day, cnt in day_counts:
        te_days.append(day)
        cum += cnt
        if cum >= target:
            break

    tr_days = [d for d, _ in day_counts if d not in te_days]
    tr_days.sort(); te_days.sort()
    return np.array(tr_days), np.array(te_days)

# ─── New day‐split helper: 60/20/20 stratified days ────────────────────────
from sklearn.model_selection import train_test_split

# ─── Hyperparameters ──────────────────────────────────────────────────────
MIN_TEST_WINDOWS      = 2       # your existing guard
MIN_SAMPLES_PER_CLASS = 1       # new: require ≥1 pos & ≥1 neg in TEST

import numpy as np
import pandas as pd

def undersample_negatives(X, y, random_state=42):
    """
    Down-sample negatives to match positives.
    Returns (X_balanced, y_balanced).
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos   = len(pos_idx)
    if n_pos == 0 or len(neg_idx) <= n_pos:
        return X, y
    rng = np.random.default_rng(random_state)
    neg_keep = rng.choice(neg_idx, size=n_pos, replace=False)
    keep = np.concatenate([pos_idx, neg_keep])
    rng.shuffle(keep)
    return X[keep], y[keep]

def oversample_positives(X, y, random_state=42, noise_scale=0.05):
    """
    Up-sample positives with Gaussian jitter to match negatives.
    Returns (X_balanced, y_balanced, n_added).
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos   = len(pos_idx)
    n_neg   = len(neg_idx)
    if n_pos == 0 or n_pos >= n_neg:
        return X, y, 0

    rng      = np.random.default_rng(random_state)
    picks    = rng.choice(pos_idx, size=(n_neg - n_pos), replace=True)
    X_pos    = X[picks]
    std_dev  = X_pos.std(axis=0, ddof=1)
    noise    = rng.normal(0, std_dev * noise_scale, size=X_pos.shape)
    X_new    = X_pos + noise
    y_new    = np.ones(len(X_new), dtype=y.dtype)

    Xb = np.vstack([X, X_new])
    yb = np.concatenate([y, y_new])
    return Xb, yb, len(X_new)

def round_robin_undersample(train_info, random_state=42):
    # train_info[u] is a dict {'days':…, 'df': DataFrame}
    total_pos = sum(int(info["df"]["state_val"].sum()) for info in train_info.values())
    total_neg = sum(int((info["df"]["state_val"]==0).sum()) for info in train_info.values())
    to_remove = total_neg - total_pos
    if to_remove <= 0:
        return
    rng   = np.random.default_rng(random_state)
    users = list(train_info.keys())
    removed = 0
    idx = 0
    while removed < to_remove and users:
        u = users[idx % len(users)]
        df = train_info[u]["df"]
        negs = df.index[df["state_val"] == 0].tolist()
        if not negs:
            users.remove(u)
        else:
            drop = rng.choice(negs)
            train_info[u]["df"] = df.drop(drop).reset_index(drop=True)
            removed += 1
        idx += 1

def round_robin_oversample(train_info, random_state=42, noise_scale=0.05):
    """
    Evenly add synthetic positives across users until pos == neg globally.
    Modifies train_info[u]["df"] in place.
    Returns total added.
    """
    rng = np.random.default_rng(random_state)

    # 1) Compute how many to add
    total_pos = sum(int(info["df"]["state_val"].sum()) for info in train_info.values())
    total_neg = sum(int((info["df"]["state_val"] == 0).sum()) for info in train_info.values())
    to_add = total_neg - total_pos
    if to_add <= 0:
        return 0

    # Helper to ensure seq → 1D float array
    def _flatten_to_float(seq):
        return np.asarray(seq, dtype=float)

    users = list(train_info.keys())
    added = 0
    idx = 0

    # 2) Round-robin over users
    while added < to_add and users:
        u = users[idx % len(users)]
        df = train_info[u]["df"]

        # pick by positional index
        pos_idx = np.flatnonzero(df["state_val"].values == 1)
        if pos_idx.size == 0:
            users.remove(u)
        else:
            # sample one positive
            pick = rng.choice(pos_idx)
            row = df.iloc[pick].copy()  # always a Series

            # flatten and jitter
            hr = _flatten_to_float(row["hr_seq"])
            st = _flatten_to_float(row["st_seq"])
            hr_std, st_std = hr.std(ddof=1), st.std(ddof=1)
            hr += rng.normal(0, hr_std * noise_scale, size=hr.shape)
            st += rng.normal(0, st_std * noise_scale, size=st.shape)

            # write back
            row["hr_seq"] = hr.tolist()
            row["st_seq"] = st.tolist()

            # append the synthetic row
            train_info[u]["df"] = pd.concat([df, row.to_frame().T], ignore_index=True)
            added += 1

        idx += 1

    return added

def sample_train_info(train_info, mode, random_state=42):
    """
    Apply sampling to train_info and return:
      {
        "global": {orig_pos, orig_neg, new_pos, new_neg, added, removed},
        "per_user": {
           uid: {orig_pos, orig_neg, new_pos, new_neg}, ...
        }
      }
    """
    # capture originals
    per_user = {}
    for u, info in train_info.items():
        df = info["df"]
        per_user[u] = {
            "orig_pos": int(df["state_val"].sum()),
            "orig_neg": int((df["state_val"]==0).sum())
        }
    g = per_user  # alias

    orig_pos = sum(u["orig_pos"] for u in g.values())
    orig_neg = sum(u["orig_neg"] for u in g.values())

    added = removed = 0
    if mode == "undersample":
        round_robin_undersample(train_info, random_state)
    elif mode == "oversample":
        added = round_robin_oversample(train_info, random_state)
    # recompute per-user new
    for u, info in train_info.items():
        df = info["df"]
        per_user[u].update({
            "new_pos": int(df["state_val"].sum()),
            "new_neg": int((df["state_val"]==0).sum())
        })

    new_pos = sum(u["new_pos"] for u in per_user.values())
    new_neg = sum(u["new_neg"] for u in per_user.values())
    removed = orig_neg - new_neg

    return {
        "global": {
          "orig_pos": orig_pos, "orig_neg": orig_neg,
          "new_pos":  new_pos,  "new_neg":  new_neg,
          "added":    added,    "removed": removed
        },
        "per_user": per_user
    }

def ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df):
    """
    1) Split off ~20% of days as TEST (no strat).
    2) From remaining ~80%, stratify into 75/25 → TRAIN (~60%) / VAL (~20%).
    3) Retry until TEST has ≥MIN_TEST_WINDOWS total windows AND
       at least MIN_SAMPLES_PER_CLASS positives AND negatives.
    """
    days = np.array(sorted(
        pd.concat([pos_df, neg_df])['hawaii_createdat_time']
          .dt.date.unique()
    ))

    # Precompute window counts per day for stratification
    day_counts = {d: _count_windows(pos_df, neg_df, hr_df, st_df, [d])
                  for d in days}

    for attempt in range(10):
        # 1) carve off TEST
        trval_days, test_days = train_test_split(
            days, test_size=0.2, random_state=42 + attempt
        )
        # check total windows
        n_test = sum(day_counts[d] for d in test_days)
        if n_test < MIN_TEST_WINDOWS:
            continue

        # check per-class windows in TEST
        df_te   = collect_windows(pos_df, neg_df, hr_df, st_df, test_days)
        pos_te  = df_te['state_val'].sum()
        neg_te  = len(df_te) - pos_te
        if pos_te < MIN_SAMPLES_PER_CLASS or neg_te < MIN_SAMPLES_PER_CLASS:
            continue

        # 2) stratify TRAIN/VAL on remaining days
        trval_counts = [day_counts[d] for d in trval_days]
        try:
            train_days, val_days = train_test_split(
                trval_days,
                test_size=0.25,               # 0.25 of the 80% → 20% overall
                stratify=trval_counts,
                random_state=42 + attempt
            )
        except ValueError:
            train_days, val_days = train_test_split(
                trval_days,
                test_size=0.25,
                random_state=42 + attempt
            )

        # final sanity check: VAL must have at least MIN_TEST_WINDOWS as well
        n_val = sum(day_counts[d] for d in val_days)
        if n_val < MIN_TEST_WINDOWS:
            continue

        return np.array(train_days), np.array(val_days), np.array(test_days)

    raise RuntimeError(
        f"Unable to find a TEST split with ≥{MIN_TEST_WINDOWS} windows "
        f"and ≥{MIN_SAMPLES_PER_CLASS} positives & negatives after 10 tries"
    )

def collect_windows(df_p, df_n, hr_df, st_df, days):
    p = df_p[df_p['hawaii_createdat_time'].dt.date.isin(days)]
    n = df_n[df_n['hawaii_createdat_time'].dt.date.isin(days)]
    return pd.concat([
        process_label_window(p, hr_df, st_df, 1),
        process_label_window(n, hr_df, st_df, 0)
    ])


def _count_windows(df_p, df_n, hr_df, st_df, days):
    return len(collect_windows(df_p, df_n, hr_df, st_df, days))


def _write_skip_file(root: Path, train_days, test_days, n_tr, n_te):
    root.mkdir(parents=True, exist_ok=True)
    msg = [
        "SKIPPED – Not enough windows for benchmarking.\n",
        f"Train days ({len(train_days)}): {list(train_days)}",
        f"Test  days ({len(test_days)}): {list(test_days)}",
        f"Train windows: {n_tr}",
        f"Test  windows: {n_te}",
        "",
        "Minimum required: 2 train windows AND 2 test windows."
    ]
    (root / "not_enough_data.txt").write_text("\n".join(msg))
    print(">> SKIPPED – see not_enough_data.txt for details")


def write_split_details(
    results_dir, pipeline, train_info, val_info, test_info,
    sample_mode="original", sample_summary=None
):
    """
    Writes split_details.txt with sampling mode, global & per-user stats,
    then the TRAIN/VAL/TEST breakdown.
    """
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    uid, te_days, df_te = test_info
    df_val = val_info[uid]["df"]; val_days = val_info[uid]["days"]

    # helper
    def cnt(df):
        p = int(df["state_val"].sum())
        n = len(df)-p
        return p, n

    with open(results_dir/"split_details.txt","w") as f:
        f.write(f"Pipeline: {pipeline}\n")
        f.write(f"Sampling mode: {sample_mode}\n")
        if sample_summary:
            G = sample_summary["global"]
            f.write(
              f"GLOBAL  ORIG +{G['orig_pos']}/-{G['orig_neg']}  "
              f"→ USED +{G['new_pos']}/-{G['new_neg']}\n"
            )
            if G["added"]:   f.write(f"Synthetic positives added: {G['added']}\n")
            if G["removed"]: f.write(f"Negatives removed: {G['removed']}\n")
            f.write("\nUSER-LEVEL SAMPLING:\n")
            for u, stats in sample_summary["per_user"].items():
                f.write(
                  f"  {u}: ORIG +{stats['orig_pos']}/-{stats['orig_neg']}  "
                  f"→ USED +{stats['new_pos']}/-{stats['new_neg']}\n"
                )
            f.write("\n")

        # TRAIN
        f.write("=== TRAINING DAYS (used for model fitting) ===\n")
        for u, info in train_info.items():
            p,n = cnt(info["df"])
            f.write(f"User {u} TRAIN days: {info['days']}\n")
            f.write(f"   windows={len(info['df'])}  (+={p}, -={n})\n\n")

        # VAL
        p_va,n_va = cnt(df_val)
        f.write("=== VALIDATION DAYS (only target user) ===\n")
        f.write(f"User {uid} VAL days: {val_days}\n")
        f.write(f"   windows={len(df_val)}  (+={p_va}, -={n_va})\n\n")

        # TEST
        p_te,n_te = cnt(df_te)
        f.write("=== TEST DAYS (only target user) ===\n")
        f.write(f"User {uid} TEST days: {te_days}\n")
        f.write(f"   windows={len(df_te)}  (+={p_te}, -={n_te})\n")

# ─── Helper to train/load per-user SSL encoder ─────────────────────────────
def _train_or_load_encoder(path, dtype, df, train_days, results_dir):
    if path.exists():
        enc = load_model(path)
        enc.trainable = False
        return enc

    mask = np.isin(df.index.date, train_days)
    vals = StandardScaler().fit_transform(df.loc[mask, 'value'].values.reshape(-1,1))
    segs = create_windows(vals, WINDOW_SIZE, STEP_SIZE).astype('float32')
    if not len(segs):
        raise RuntimeError(f"No {dtype} segments to train encoder.")

    n, idx = len(segs), np.random.permutation(len(segs))
    tr, va = segs[idx[:int(0.8*n)]], segs[idx[int(0.8*n):]]

    enc = build_simclr_encoder(WINDOW_SIZE)
    head = create_projection_head()
    tr_l, va_l = train_simclr(enc, head, tr, va,
                              batch_size=BATCH_SSL, epochs=SSL_EPOCHS)
    enc.save(path)
    enc.trainable = False
    plot_ssl_losses(tr_l, va_l, results_dir, encoder_name=f"{dtype}_ssl")
    return enc

# ─── Shared SSL encoders (train only on train-days) ───────────────────────
def _ensure_global_encoders(shared_root, fruit, scenario, all_splits):
    sdir = Path(shared_root) / f"{fruit}_{scenario}"
    sdir.mkdir(parents=True, exist_ok=True)
    paths = {
        'hr':    sdir / 'hr_encoder.keras',
        'steps': sdir / 'steps_encoder.keras'
    }
    if all(p.exists() for p in paths.values()):
        hr = load_model(paths['hr'])
        hr.trainable = False
        st = load_model(paths['steps'])
        st.trainable = False
        return hr, st, sdir

    losses = {}
    for dtype in ['hr', 'steps']:
        bank = []
        for u, pairs in ALLOWED_SCENARIOS.items():
            if (fruit, scenario) not in pairs:
                continue

            # now unpack train/val/test days
            tr_days_u, _val_days_u, _te_days_u = all_splits.get(u, ([], [], []))
            if len(tr_days_u) == 0:
                continue

            hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
            df = hr_df if dtype == 'hr' else st_df

            # only use train days here
            mask = np.isin(df.index.date, tr_days_u)
            vals = StandardScaler()\
                   .fit_transform(df.loc[mask, 'value'].values.reshape(-1, 1))

            if len(vals) < WINDOW_SIZE:
                continue

            bank.append(create_windows(vals, WINDOW_SIZE, STEP_SIZE).astype('float32'))

        if not bank:
            raise RuntimeError(f"No train-day segments for global {dtype} SSL!")

        segs = np.concatenate(bank, axis=0)
        n, idx = len(segs), np.random.permutation(len(segs))
        tr, va = segs[idx[: int(0.8 * n)]], segs[idx[int(0.8 * n) :]]

        enc = build_simclr_encoder(WINDOW_SIZE)
        head = create_projection_head()
        tr_l, va_l = train_simclr(enc, head, tr, va,
                                  batch_size=BATCH_SSL, epochs=SSL_EPOCHS)

        enc.save(paths[dtype])
        enc.trainable = False
        losses[dtype] = (tr_l, va_l)

    # plot losses for both modalities
    plot_ssl_losses(*losses['hr'],    sdir, encoder_name="global_hr")
    plot_ssl_losses(*losses['steps'], sdir, encoder_name="global_steps")

    hr = load_model(paths['hr']);    hr.trainable = False
    st = load_model(paths['steps']); st.trainable = False
    return hr, st, sdir

# ─── Shared CNN for Global-Supervised ─────────────────────────────────────
def ensure_global_supervised(shared_cnn_root, fruit, scenario, all_splits, uid):
    """
    Train (or load) a single global CNN on ALL users' train-day windows,
    validating only on the target user's validation windows.

    Returns:
        m     : the trained Keras model
        sdir  : the directory where the model was saved/loaded
    """
    sdir = Path(shared_cnn_root) / f"{fruit}_{scenario}"
    sdir.mkdir(parents=True, exist_ok=True)
    model_path = sdir / 'cnn_classifier.keras'
    if model_path.exists():
        m = load_model(model_path)
        return m, sdir

    # 1) Gather all users' TRAIN windows
    X_list, y_list = [], []
    for u, pairs in ALLOWED_SCENARIOS.items():
        if (fruit, scenario) not in pairs:
            continue
        tr_days_u, _, _ = all_splits.get(u, ([], [], []))
        if len(tr_days_u) == 0:
            continue

        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
        if len(orig_neg) < len(pos_df):
            extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg

        df_u = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
        for h_seq, s_seq, label in zip(df_u['hr_seq'], df_u['st_seq'], df_u['state_val']):
            X_list.append(np.vstack([h_seq, s_seq]).T)
            y_list.append(label)

    if not X_list:
        raise RuntimeError("No train windows for global-supervised!")

    X = np.stack(X_list)
    y = np.array(y_list)

    # 2) Build the target user's VAL set
    tr_days_u, val_days_u, _ = all_splits[uid]
    hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df_u = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_df_u = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, 'None')
    if neg_df_u.empty or len(neg_df_u) < len(pos_df_u):
        neg_df_u = derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u))

    df_val_u = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, val_days_u)

    def build_XY(df):
        X = np.stack([np.vstack([h,s]).T for h,s in zip(df['hr_seq'], df['st_seq'])])
        return X, df['state_val'].values

    X_val_u, y_val_u = build_XY(df_val_u)

    # 3) Build & compile the model
    inp = layers.Input(shape=(WINDOW_SIZE, 2))
    x = layers.Conv1D(64, 8, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(4, activation='relu')(se)
    se = layers.Dense(64, activation='sigmoid')(se)
    se = layers.Reshape((1, 64))(se)
    x = layers.Multiply()([x, se])

    x = layers.Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(8, activation='relu')(se)
    se = layers.Dense(128, activation='sigmoid')(se)
    se = layers.Reshape((1, 128))(se)
    x = layers.Multiply()([x, se])

    x = layers.Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    cnn_out = layers.GlobalAveragePooling1D()(x)

    lstm_out = layers.LSTM(128)(inp)
    lstm_out = layers.Dropout(0.5)(lstm_out)

    combined = layers.concatenate([cnn_out, lstm_out])
    out = layers.Dense(1, activation='sigmoid')(combined)

    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    # 4) Compute class weights
    classes = np.unique(y)
    if len(classes) == 2:
        cw_vals = compute_class_weight('balanced', classes=classes, y=y)
        class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}
    else:
        class_weight = {0:1, 1:1}

    # 5) Callbacks
    es = EarlyStopping(monitor='val_loss', patience=GS_PATIENCE, restore_best_weights=True, verbose=1)
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # 6) SINGLE TRAINING pass
    hist = m.fit(
        X, y,
        validation_data=(X_val_u, y_val_u),
        batch_size=GS_BATCH,
        epochs=GS_EPOCHS,
        class_weight=class_weight,
        callbacks=[es, lr_cb],
        verbose=2
    )

    # 7) Save & plot
    plot_clf_losses(hist.history['loss'], hist.history['val_loss'], sdir, 'global_cnn_lstm_loss')
    m.save(model_path)
    return m, sdir

# ─── Pipeline #1: Global-Supervised (60/20/20 + explicit val) ─────────────
def run_global_supervised(
    fruit: str,
    scenario: str,
    uid: str,
    user_root: Path,
    all_splits: dict,
    shared_cnn_root: Path,
    neg_df_u: pd.DataFrame,        # ← NEW argument
    sample_mode: str = "original",
):
    """
    Pipeline #1: Global-Supervised

    1) Train or load a single CNN on ALL users' train-day windows,
       validating only on this user's validation windows.
    2) Optionally undersample or oversample TRAIN windows across users.
    3) Copy the model & plots into the user's folder.
    4) Build train/val/test splits, write split_details.txt with sampling info,
       then threshold & bootstrap on TEST.
    """
    print(f"\n>> Global-Supervised ({fruit}_{scenario})")

    # Directories
    out_dir   = user_root / 'global_supervised'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    # 1) Train or load the shared CNN (validating on this user's VAL only)
    model, src_dir = ensure_global_supervised(
        shared_cnn_root, fruit, scenario, all_splits, uid
    )
    # Copy model files and plots
    for fpath in Path(src_dir).glob('*'):
        if fpath.suffix == '.keras':
            shutil.copy2(fpath, models_d / fpath.name)
        elif fpath.suffix == '.png':
            shutil.copy2(fpath, results_d / fpath.name)

    # 2) Build per-user train_info and val_info
    train_info = {}
    val_info   = {}
    for u, (tr_days, val_days, _) in all_splits.items():
        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
        if len(orig_neg) < len(pos_df):
            extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg

        df_tr  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
        df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)

        train_info[u] = {"days": tr_days.tolist(),  "df": df_tr}
        val_info[u]   = {"days": val_days.tolist(), "df": df_val}

    # 3) Apply sampling (original, undersample, or oversample)
    sample_summary = sample_train_info(train_info, mode=sample_mode, random_state=42)

    # 4) Prepare this user's TEST windows
    tr_days_u, val_days_u, te_days_u = all_splits[uid]
    hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df_u = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)

    # <— use the precomputed neg_df_u, do NOT call derive_negative_labels here
    df_te = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)

    # 5) Write split_details.txt (with sampling info and original vs used counts)
    write_split_details(
        results_d,
        "global_supervised",
        train_info,
        val_info,
        (uid, te_days_u.tolist(), df_te),
        sample_mode=sample_mode,
        sample_summary=sample_summary
    )

    # 6) Build X/y arrays for training, validation, and test
    def build_XY(df_list):
        X = np.stack([np.vstack([h, s]).T
                      for h, s in zip(df_list['hr_seq'], df_list['st_seq'])])
        y = df_list['state_val'].values
        return X, y

    X_tr, y_tr   = build_XY(pd.concat([v["df"] for v in train_info.values()]))
    X_val, y_val = build_XY(val_info[uid]["df"])
    X_te, y_te   = build_XY(df_te)

    # 7) Threshold selection on VAL & bootstrap on TEST
    cw_vals = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}

    val_preds  = model.predict(X_val, verbose=0).flatten()
    thresholds = np.arange(0.0, 1.0001, 0.01)
    scores = []
    for thr in thresholds:
        pbin = (val_preds >= thr).astype(int)
        tp = ((pbin == 1) & (y_val == 1)).sum()
        fn = ((pbin == 0) & (y_val == 1)).sum()
        fp = ((pbin == 1) & (y_val == 0)).sum()
        tn = ((pbin == 0) & (y_val == 0)).sum()
        tpr  = tp / (tp + fn) if tp + fn > 0 else 0.0
        spec = tn / (tn + fp) if tn + fp > 0 else 0.0
        scores.append(0.7 * tpr + 0.3 * spec)

    best_threshold = float(thresholds[np.argmax(scores)])
    (results_d / "selected_threshold.txt").write_text(f"{best_threshold:.4f}\n")

    df_boot, auc_mean, auc_std = bootstrap_threshold_metrics(
        y_te,
        model.predict(X_te, verbose=0).flatten(),
        thresholds=np.array([best_threshold]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)
    plot_thresholds(
        y_te,
        model.predict(X_te, verbose=0).flatten(),
        str(results_d),
        f"{uid} {fruit}_{scenario} (global_supervised)",
        thresholds=np.array([best_threshold]),
        sample_frac=0.7,
        n_iters=1000
    )

    return df_boot, auc_mean, auc_std

def run_personal_ssl(
    uid: str,
    fruit: str,
    scenario: str,
    user_root: Path,
    tr_days_u: np.ndarray,
    val_days_u: np.ndarray,
    te_days_u: np.ndarray,
    neg_df_u: pd.DataFrame,
    sample_mode: str = "original"
):
    """
    Pipeline #2: Personal-SSL

    1) Load this user's signals & labels.
    2) Train (or load) SSL encoders on TRAIN days.
    3) Window & label for TRAIN/VAL/TEST and write split_details.txt.
    4) Encode into features, then optionally undersample/oversample.
    5) Log pre- and post-sampling summaries.
    6) Train a small classifier, threshold & bootstrap on TEST.
    """
    print(f"\n>> Personal-SSL ({fruit}_{scenario})")

    out_dir   = user_root / 'personal_ssl'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    # 1) Load signals & labels
    hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df       = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_df       = neg_df_u  # use precomputed negatives

    # 2) Train or load SSL encoders
    enc_hr = _train_or_load_encoder(models_d / 'hr_encoder.keras',
                                    'hr', hr_df, tr_days_u, results_d)
    enc_st = _train_or_load_encoder(models_d / 'steps_encoder.keras',
                                    'steps', st_df, tr_days_u, results_d)

    # 3) Window & label for TRAIN/VAL/TEST
    df_tr  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
    df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days_u)
    df_te  = collect_windows(pos_df, neg_df, hr_df, st_df, te_days_u)

    # 4a) Build train_info for sampling summary
    train_info = {
        uid: {
            "days": tr_days_u.tolist(),
            "df": df_tr.copy()
        }
    }

    # 4b) Apply sampling and capture summary
    sample_summary = sample_train_info(train_info, mode=sample_mode, random_state=42)

    # 5) Write split details with sampling summary
    write_split_details(
        results_d,
        "personal_ssl",
        train_info,
        {uid: {"days": val_days_u.tolist(), "df": df_val}},
        (uid, te_days_u.tolist(), df_te),
        sample_mode=sample_mode,
        sample_summary=sample_summary
    )

    # 6) Encode into feature vectors
    def encode(df):
        hr_seq = np.stack(df['hr_seq'])[..., None]
        st_seq = np.stack(df['st_seq'])[..., None]
        return enc_hr.predict(hr_seq, verbose=0), enc_st.predict(st_seq, verbose=0)

    H_tr,  S_tr  = encode(df_tr)
    H_val, S_val = encode(df_val)
    H_te,  S_te  = encode(df_te)

    X_tr = np.concatenate([H_tr, S_tr], axis=1)
    y_tr = df_tr['state_val'].values
    X_val = np.concatenate([H_val, S_val], axis=1)
    y_val = df_val['state_val'].values
    X_te  = np.concatenate([H_te, S_te], axis=1)
    y_te  = df_te['state_val'].values

    # 7) Build & train classifier
    clf = Sequential([
        Dense(64, activation='relu', input_shape=(X_tr.shape[1],), kernel_regularizer=l2(0.01)),
        BatchNormalization(), Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    clf.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=CLF_PATIENCE, restore_best_weights=True, verbose=1)
    cw_vals = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}

    clf.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=CLF_EPOCHS,
        batch_size=16,
        class_weight=class_weight,
        callbacks=[es],
        verbose=2
    )

    # 8) Threshold & bootstrap on TEST
    val_preds  = clf.predict(X_val, verbose=0).flatten()
    thresholds = np.arange(0.0, 1.0001, 0.01)
    scores     = []
    for thr in thresholds:
        pbin = (val_preds >= thr).astype(int)
        tp = ((pbin == 1) & (y_val == 1)).sum()
        fn = ((pbin == 0) & (y_val == 1)).sum()
        fp = ((pbin == 1) & (y_val == 0)).sum()
        tn = ((pbin == 0) & (y_val == 0)).sum()
        tpr  = tp / (tp + fn) if tp + fn > 0 else 0.0
        spec = tn / (tn + fp) if tn + fp > 0 else 0.0
        scores.append(0.7 * tpr + 0.3 * spec)

    best_threshold = float(thresholds[np.argmax(scores)])
    (results_d / "selected_threshold.txt").write_text(f"{best_threshold:.4f}\n")

    df_boot, auc_mean, auc_std = bootstrap_threshold_metrics(
        y_te,
        clf.predict(X_te, verbose=0).flatten(),
        thresholds=np.array([best_threshold]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)
    plot_thresholds(
        y_te,
        clf.predict(X_te, verbose=0).flatten(),
        str(results_d),
        f"{uid} {fruit}_{scenario} (personal_ssl)",
        thresholds=np.array([best_threshold]),
        sample_frac=0.7,
        n_iters=1000
    )

    return df_boot, auc_mean, auc_std

# ─── Pipeline #3: Global-SSL ───────────────────────────────────────────────
def run_global_ssl(
    uid: str,
    fruit: str,
    scenario: str,
    user_root: Path,
    shared_enc_root: Path,
    all_splits: dict,
    neg_df_u: pd.DataFrame,        # ← NEW argument
    sample_mode: str = "original"
):
    """
    Pipeline #3: Global-SSL

    1) Train or load shared SSL encoders on all users' train-day windows.
    2) Optionally undersample or oversample those windows across users.
    3) Validate and test only on this user's days.
    """
    print(f"\n>> Global-SSL ({fruit}_{scenario})")

    out_dir   = user_root / 'global_ssl'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    # 1) Load or train shared encoders
    enc_hr, enc_st, enc_src = _ensure_global_encoders(
        shared_enc_root, fruit, scenario, all_splits
    )
    for fpath in Path(enc_src).glob('*'):
        if fpath.suffix == '.keras':
            shutil.copy2(fpath, models_d / fpath.name)
        elif fpath.suffix == '.png':
            shutil.copy2(fpath, results_d / fpath.name)

    # 2) Build per-user train_info and val_info
    train_info = {}
    val_info   = {}
    for u, (tr_days, val_days, _) in all_splits.items():
        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
        if len(orig_neg) < len(pos_df):
            extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg

        df_tr  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
        df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)

        train_info[u] = {"days": tr_days.tolist(),  "df": df_tr}
        val_info[u]   = {"days": val_days.tolist(), "df": df_val}

    #  3) Apply sampling across users
    sample_summary = sample_train_info(train_info, mode=sample_mode, random_state=42)

    # 4) Build this user's test windows
    tr_days_u, val_days_u, te_days_u = all_splits[uid]
    hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df_u        = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)

    # <— use the precomputed neg_df_u, do NOT call derive_negative_labels here
    df_te = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)

    # 5) Write split_details.txt with sampling info
    write_split_details(
        results_d,
        "global_ssl",
        train_info,
        val_info,
        (uid, te_days_u.tolist(), df_te),
        sample_mode=sample_mode,
        sample_summary=sample_summary
    )

    # 6) Encode and build X/y
    def encode(df):
        hr_seq = np.stack(df['hr_seq'])[..., None]
        st_seq = np.stack(df['st_seq'])[..., None]
        return enc_hr.predict(hr_seq, verbose=0), enc_st.predict(st_seq, verbose=0)

    H_tr,  S_tr  = encode(pd.concat([v["df"] for v in train_info.values()]))
    df_val_u      = val_info[uid]["df"]
    H_val, S_val = encode(df_val_u)
    H_te,  S_te  = encode(df_te)

    X_tr = np.concatenate([H_tr, S_tr], axis=1)
    y_tr = pd.concat([v["df"] for v in train_info.values()])['state_val'].values
    X_val = np.concatenate([H_val, S_val], axis=1)
    y_val = df_val_u['state_val'].values
    X_te  = np.concatenate([H_te, S_te], axis=1)
    y_te  = df_te['state_val'].values

    # ─── **NEW**: ensure everything is float32 ───────────────────────────────
    X_tr  = X_tr.astype('float32')
    y_tr  = y_tr.astype('float32')
    X_val = X_val.astype('float32')
    y_val = y_val.astype('float32')
    X_te  = X_te.astype('float32')
    y_te  = y_te.astype('float32')

    # 7) Train classifier and threshold/bootstrap
    clf = Sequential([
        Dense(64, activation='relu', input_shape=(X_tr.shape[1],), kernel_regularizer=l2(0.01)),
        BatchNormalization(), Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=CLF_PATIENCE, restore_best_weights=True, verbose=1)
    cw_vals = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}

    clf.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=CLF_EPOCHS,
        batch_size=16,
        class_weight=class_weight,
        callbacks=[es],
        verbose=2
    )

    # 8) Evaluate on TEST
    val_preds  = clf.predict(X_val, verbose=0).flatten()
    thresholds = np.arange(0.0, 1.0001, 0.01)
    scores     = []
    for thr in thresholds:
        pbin = (val_preds >= thr).astype(int)
        tp = ((pbin == 1) & (y_val == 1)).sum()
        fn = ((pbin == 0) & (y_val == 1)).sum()
        fp = ((pbin == 1) & (y_val == 0)).sum()
        tn = ((pbin == 0) & (y_val == 0)).sum()
        tpr  = tp / (tp + fn) if tp + fn > 0 else 0.0
        spec = tn / (tn + fp) if tn + fp > 0 else 0.0
        scores.append(0.7 * tpr + 0.3 * spec)

    best_threshold = float(thresholds[np.argmax(scores)])
    (results_d / "selected_threshold.txt").write_text(f"{best_threshold:.4f}\n")

    df_boot, auc_mean, auc_std = bootstrap_threshold_metrics(
        y_te,
        clf.predict(X_te, verbose=0).flatten(),
        thresholds=np.array([best_threshold]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)
    plot_thresholds(
        y_te,
        clf.predict(X_te, verbose=0).flatten(),
        str(results_d),
        f"{uid} {fruit}_{scenario} (global_ssl)",
        thresholds=np.array([best_threshold]),
        sample_frac=0.7,
        n_iters=1000
    )

    return df_boot, auc_mean, auc_std

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    pa = argparse.ArgumentParser()
    pa.add_argument("--user",       required=True)
    pa.add_argument("--fruit",      required=True)
    pa.add_argument("--scenario",   required=True)
    pa.add_argument("--output-dir", default="results")
    pa.add_argument(
        "--sample-mode",
        choices=["original", "undersample", "oversample"],
        default="original",
        help="How to balance classes in TRAIN/VAL: keep original, undersample negs, or oversample pos."
    )
    pa.add_argument(
        "--results-subdir",
        default="results",
        help="Name of the subdirectory under each pipeline where results (CSVs, plots, split_details) go."
    )
    args = pa.parse_args()

    # override the global default
    global RESULTS_SUBDIR
    RESULTS_SUBDIR = args.results_subdir

    # prepare top‐level and per‐user directories
    top_out         = Path(args.output_dir)
    user_root       = top_out / args.user / f"{args.fruit}_{args.scenario}"
    shared_enc_root = top_out / "_global_encoders"
    shared_cnn_root = top_out / "global_cnns"
    user_root.mkdir(parents=True, exist_ok=True)

    # seed everything
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # ─── build day‐level splits and negatives for all users ─────────────────
    all_splits   = {}
    all_negatives = {}     # ← NEW: store per-user neg_df

    for u, pairs in ALLOWED_SCENARIOS.items():
        if (args.fruit, args.scenario) not in pairs:
            continue

        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df       = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
        orig_neg     = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, 'None')

        if len(orig_neg) < len(pos_df):
            extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg

        all_negatives[u] = neg_df

        try:
            tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
        except RuntimeError as e:
            print(f"Skipping user {u}: {e}")
            continue

        all_splits[u] = (tr_u, val_u, te_u)

    # ensure the target user has splits
    if args.user not in all_splits:
        print(f"Skipping user {args.user}: no data for {args.fruit}/{args.scenario}.")
        sys.exit(0)

    # retrieve the one and only neg_df for test
    neg_df_u = all_negatives[args.user]
    tr_days_u, val_days_u, te_days_u = all_splits[args.user]

    # run pipelines, passing neg_df_u into each
    df_gs, auc_gs_m, auc_gs_s = run_global_supervised(
        args.fruit, args.scenario, args.user,
        user_root, all_splits, shared_cnn_root,
        sample_mode=args.sample_mode,
        neg_df_u=neg_df_u
    )
    df_ps, auc_ps_m, auc_ps_s = run_personal_ssl(
        args.user, args.fruit, args.scenario,
        user_root, tr_days_u, val_days_u, te_days_u,
        neg_df_u=neg_df_u,
        sample_mode=args.sample_mode
    )
    df_gl, auc_gl_m, auc_gl_s = run_global_ssl(
        args.user, args.fruit, args.scenario,
        user_root, shared_enc_root, all_splits,
        neg_df_u=neg_df_u,
        sample_mode=args.sample_mode
    )

    # produce comparison summary
    rows = []
    for name, (df, auc_m, auc_s) in [
        ("global_supervised", (df_gs, auc_gs_m, auc_gs_s)),
        ("personal_ssl",     (df_ps, auc_ps_m, auc_ps_s)),
        ("global_ssl",       (df_gl, auc_gl_m, auc_gl_s))
    ]:
        tmp = df.copy()
        tmp["Balance"] = tmp[["Sensitivity_Mean", "Specificity_Mean"]].min(axis=1)
        best = tmp.sort_values(
            ["Balance", "Sensitivity_Mean"],
            ascending=[False, False]
        ).iloc[0]
        rows.append({
            "Pipeline":         name,
            "Best_Threshold":   best["Threshold"],
            "Accuracy_Mean":    best["Accuracy_Mean"],
            "Accuracy_STD":     best["Accuracy_STD"],
            "Sensitivity_Mean": best["Sensitivity_Mean"],
            "Sensitivity_STD":  best["Sensitivity_STD"],
            "Specificity_Mean": best["Specificity_Mean"],
            "Specificity_STD":  best["Specificity_STD"],
            "AUC_Mean":         auc_m,
            "AUC_STD":          auc_s
        })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(user_root / "comparison_summary.csv", index=False)

    print("\n--- Comparison Summary ---")
    print(df_summary.to_markdown(index=False))