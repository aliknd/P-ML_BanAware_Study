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

def undersample_negatives(X, y, random_state=42):
    """
    If negatives >> positives, down-sample negatives to match #positives.
    Returns balanced (X, y).
    """
    # indices of each class
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos   = len(pos_idx)

    # only undersample if negatives exceed
    if len(neg_idx) > n_pos > 0:
        rng = np.random.default_rng(random_state)
        neg_idx = rng.choice(neg_idx, size=n_pos, replace=False)

    keep = np.concatenate([pos_idx, neg_idx])
    # shuffle so model sees mixed examples
    rng = np.random.default_rng(random_state)
    rng.shuffle(keep)

    return X[keep], y[keep]

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


def write_split_details(results_dir, pipeline, train_info, val_info, test_info):
    """
    Write a split_details.txt with three clear sections:

      1) Training days: all users whose data were used to fit the model
      2) Validation days: only the target user, used for threshold selection
      3) Test days: only the target user, held out for final evaluation

    Parameters
    ----------
    results_dir : str or Path
        Directory to write split_details.txt into.
    pipeline : str
        Name of the pipeline (e.g. "global_supervised").
    train_info : dict[user] -> {"days": list of dates, "df": DataFrame of windows}
        Per‐user train splits.
    val_info : dict[user] -> {"days": list of dates, "df": DataFrame of windows}
        Per‐user validation splits.
    test_info : tuple (user, test_days_list, test_df)
        The target user’s test split.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    uid, te_days, df_te = test_info
    df_val = val_info[uid]["df"]
    val_days = val_info[uid]["days"]

    # Count positives/negatives
    def counts(df):
        pos = int(df["state_val"].sum())
        neg = len(df) - pos
        return pos, neg

    pos_te, neg_te = counts(df_te)
    pos_va, neg_va = counts(df_val)

    with open(results_dir / "split_details.txt", "w") as f:
        f.write(f"Pipeline: {pipeline}\n\n")

        # ─── Section 1: Training days ───────────────────────────────────
        f.write("=== TRAINING DAYS (used for model fitting) ===\n")
        for user, info in train_info.items():
            days = info["days"]
            df   = info["df"]
            pos, neg = counts(df)
            f.write(f"User {user} TRAIN days: {days}\n")
            f.write(f"   windows={len(df)}  (+={pos}, -={neg})\n\n")

        # ─── Section 2: Validation days ────────────────────────────────
        f.write("=== VALIDATION DAYS (only target user; used for threshold selection) ===\n")
        f.write(f"User {uid}   VAL days: {val_days}\n")
        f.write(f"   windows={len(df_val)}  (+={pos_va}, -={neg_va})\n\n")

        # ─── Section 3: Test days ───────────────────────────────────────
        f.write("=== TEST DAYS (only target user; final evaluation) ===\n")
        f.write(f"User {uid}  TEST days: {te_days}\n")
        f.write(f"   windows={len(df_te)}  (+={pos_te}, -={neg_te})\n")

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
        neg_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
        if neg_df.empty or len(neg_df) < len(pos_df):
            neg_df = derive_negative_labels(hr_df, pos_df, len(pos_df))

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
def run_global_supervised(fruit, scenario, uid, user_root,
                          all_splits, shared_cnn_root):
    """
    1) Load (or train) the global CNN once using ensure_global_supervised
       (which uses all users' train windows + only this user's VAL).
    2) Copy the model & plots into the user's folder.
    3) Build train/val/test splits and perform thresholding & bootstrapping.
    """
    print(f"\n>> Global-Supervised ({fruit}_{scenario})")
    out_dir   = Path(user_root) / 'global_supervised'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    # 1) Train or load the global CNN (validating on this user's VAL only)
    model, src_dir = ensure_global_supervised(
        shared_cnn_root, fruit, scenario, all_splits, uid
    )
    for f in Path(src_dir).glob('*'):
        if f.suffix in ('.keras',):
            shutil.copy2(f, models_d / f.name)
        elif f.suffix in ('.png',):
            shutil.copy2(f, results_d / f.name)

    # 2) Unpack the splits
    tr_days_u, val_days_u, te_days_u = all_splits[uid]

    # 3) Build train_info & val_info dictionaries
    train_info, val_info = {}, {}
    for u, (tr_days, val_days, _) in all_splits.items():
        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
        neg_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
        if neg_df.empty or len(neg_df) < len(pos_df):
            neg_df = derive_negative_labels(hr_df, pos_df, len(pos_df))

        df_tr  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
        df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)
        train_info[u] = {"days": tr_days.tolist(),  "df": df_tr}
        val_info[u]   = {"days": val_days.tolist(), "df": df_val}

    # 4) Prepare test windows for this user
    hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df       = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_df       = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, 'None')
    if neg_df.empty or len(neg_df) < len(pos_df):
        neg_df = derive_negative_labels(hr_df, pos_df, len(pos_df))
    df_te = collect_windows(pos_df, neg_df, hr_df, st_df, te_days_u)

    # 5) Log split details
    write_split_details(results_d, "global_supervised", train_info, val_info,
                        (uid, te_days_u.tolist(), df_te))

    # 6) Build X/y arrays
    def build_XY(df_list):
        X = np.stack([np.vstack([h,s]).T for h,s in zip(df_list['hr_seq'], df_list['st_seq'])])
        return X, df_list['state_val'].values

    X_tr, y_tr   = build_XY(pd.concat([v["df"] for v in train_info.values()]))
    X_val, y_val = build_XY(val_info[uid]["df"])
    X_te, y_te   = build_XY(df_te)

    # 7) Threshold selection on VAL & bootstrap on TEST
    # Compute class weights
    cw_vals      = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}

    # Pick threshold on VAL
    val_preds  = model.predict(X_val, verbose=0).flatten()
    thresholds = np.arange(0.0, 1.0001, 0.01)
    scores = []
    for thr in thresholds:
        pbin = (val_preds >= thr).astype(int)
        tp = ((pbin==1)&(y_val==1)).sum()
        fn = ((pbin==0)&(y_val==1)).sum()
        fp = ((pbin==1)&(y_val==0)).sum()
        tn = ((pbin==0)&(y_val==0)).sum()
        tpr  = tp/(tp+fn) if tp+fn>0 else 0.0
        spec = tn/(tn+fp) if tn+fp>0 else 0.0
        scores.append(0.7*tpr + 0.3*spec)
    best_threshold = float(thresholds[np.argmax(scores)])
    (Path(results_d)/"selected_threshold.txt").write_text(f"{best_threshold:.4f}\n")

    # Bootstrap on TEST
    df_boot, auc_mean, auc_std = bootstrap_threshold_metrics(
        y_te,
        model.predict(X_te, verbose=0).flatten(),
        thresholds=np.array([best_threshold]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )
    df_boot.to_csv(Path(results_d)/"bootstrap_metrics.csv", index=False)
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

# ─── Pipeline #2: Personal-SSL (60/20/20 days + explicit val windows) ─────
def run_personal_ssl(uid, fruit, scenario, user_root,
                     tr_days_u, val_days_u, te_days_u):
    print(f"\n>> Personal-SSL ({fruit}_{scenario})")
    out_dir   = user_root / 'personal_ssl'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    # 1) load signals & labels for this user
    hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df       = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_df       = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, 'None')
    if neg_df.empty or len(neg_df) < len(pos_df):
        neg_df = derive_negative_labels(hr_df, pos_df, len(pos_df))

    # 2) train or load SSL‐encoders on TRAIN days
    enc_hr = _train_or_load_encoder(models_d / 'hr_encoder.keras',
                                    'hr', hr_df, tr_days_u, results_d)
    enc_st = _train_or_load_encoder(models_d / 'steps_encoder.keras',
                                    'steps', st_df, tr_days_u, results_d)

    # 3) window & label for TRAIN/VAL/TEST
    df_tr  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
    df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days_u)
    df_te  = collect_windows(pos_df, neg_df, hr_df, st_df, te_days_u)

    write_split_details(
        results_d,
        "personal_ssl",
        {uid: {"days": list(tr_days_u),  "df": df_tr }},
        {uid: {"days": list(val_days_u), "df": df_val}},
        (uid, list(te_days_u), df_te)
    )

    # 4) encode all splits
    def encode(df):
        hr_seq = np.stack(df['hr_seq'])[..., None]
        st_seq = np.stack(df['st_seq'])[..., None]
        return enc_hr.predict(hr_seq, verbose=0), \
               enc_st.predict(st_seq, verbose=0)

    H_tr, S_tr   = encode(df_tr)
    H_val, S_val = encode(df_val)
    H_te, S_te   = encode(df_te)

    X_tr, y_tr = np.concatenate([H_tr,   S_tr], axis=1), df_tr['state_val'].values
    X_val, y_val = np.concatenate([H_val,  S_val], axis=1), df_val['state_val'].values
    X_te, y_te   = np.concatenate([H_te,   S_te], axis=1), df_te['state_val'].values

    # 5) build & compile classifier
    clf = Sequential([
        Dense(64, activation='relu',  input_shape=(X_tr.shape[1],),
              kernel_regularizer=l2(0.01)),
        BatchNormalization(), Dropout(0.5),
        Dense(32, activation='relu',  kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(16, activation='relu',  kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    clf.compile(optimizer=Adam(1e-3),
                loss="binary_crossentropy",
                metrics=["accuracy"])

    # 6) train on TRAIN with explicit VAL
    es = EarlyStopping(monitor='val_loss',
                       patience=CLF_PATIENCE,
                       restore_best_weights=True,
                       verbose=1)

    cw_vals      = compute_class_weight("balanced",
                                        classes=np.unique(y_tr),
                                        y=y_tr)
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

    # 7) pick threshold on VAL
    val_preds  = clf.predict(X_val, verbose=0).flatten()
    thresholds = np.arange(0.0, 1.0001, 0.01)
    scores     = []
    for thr in thresholds:
        pbin = (val_preds >= thr).astype(int)
        tp = ((pbin==1)&(y_val==1)).sum()
        fn = ((pbin==0)&(y_val==1)).sum()
        fp = ((pbin==1)&(y_val==0)).sum()
        tn = ((pbin==0)&(y_val==0)).sum()
        tpr  = tp/(tp+fn) if tp+fn>0 else 0.0
        spec = tn/(tn+fp) if tn+fp>0 else 0.0
        scores.append(0.7*tpr + 0.3*spec)
    best_threshold = float(thresholds[np.argmax(scores)])
    (results_d/"selected_threshold.txt").write_text(f"{best_threshold:.4f}\n")

    # 8) final TEST bootstrap
    df_boot, auc_mean, auc_std = bootstrap_threshold_metrics(
        y_te,
        clf.predict(X_te, verbose=0).flatten(),
        thresholds=np.array([best_threshold]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )
    df_boot.to_csv(results_d/"bootstrap_metrics.csv", index=False)

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

def run_global_ssl(uid, fruit, scenario, user_root,
                   shared_enc_root, all_splits):
    """
    1) Load (or train) shared SSL encoders on all users' train days.
    2) Validate and test only on the target user's windows.
    """
    print(f"\n>> Global-SSL ({fruit}_{scenario})")
    out_dir   = Path(user_root) / 'global_ssl'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    # 1) Load or train SSL encoders
    enc_hr, enc_st, enc_src = _ensure_global_encoders(
        shared_enc_root, fruit, scenario, all_splits
    )
    for f in Path(enc_src).glob('*'):
        if f.suffix == '.keras':
            shutil.copy2(f, models_d / f.name)
        elif f.suffix == '.png':
            shutil.copy2(f, results_d / f.name)

    # 2) Unpack this user's split
    tr_days_u, val_days_u, te_days_u = all_splits[uid]

    # 3) Build train_info & val_info
    train_info, val_info = {}, {}
    for u, (tr_days, val_days, _) in all_splits.items():
        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df       = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
        neg_df       = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
        if neg_df.empty or len(neg_df) < len(pos_df):
            neg_df = derive_negative_labels(hr_df, pos_df, len(pos_df))

        df_tr  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
        df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)
        train_info[u] = {"days": tr_days.tolist(),  "df": df_tr}
        val_info[u]   = {"days": val_days.tolist(), "df": df_val}

    # 4) This user's TEST windows
    hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df       = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_df       = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, 'None')
    if neg_df.empty or len(neg_df) < len(pos_df):
        neg_df = derive_negative_labels(hr_df, pos_df, len(pos_df))
    df_te = collect_windows(pos_df, neg_df, hr_df, st_df, te_days_u)

    # 5) Log split details
    write_split_details(results_d, "global_ssl", train_info, val_info,
                        (uid, te_days_u.tolist(), df_te))

    # 6) Encode sequences
    def encode(df):
        hr_seq = np.stack(df['hr_seq'])[..., None]
        st_seq = np.stack(df['st_seq'])[..., None]
        return enc_hr.predict(hr_seq, verbose=0), enc_st.predict(st_seq, verbose=0)

    H_tr, S_tr = encode(pd.concat([v["df"] for v in train_info.values()]))
    df_val_u   = val_info[uid]["df"]
    H_val, S_val = encode(df_val_u)
    H_te, S_te = encode(df_te)

    X_tr, y_tr = np.concatenate([H_tr, S_tr], axis=1), pd.concat([v["df"] for v in train_info.values()])['state_val'].values
    X_val, y_val = np.concatenate([H_val, S_val], axis=1), df_val_u['state_val'].values
    X_te, y_te   = np.concatenate([H_te, S_te], axis=1), df_te['state_val'].values

    # 7) Build & train classifier
    clf = Sequential([
        Dense(64, activation='relu',  input_shape=(X_tr.shape[1],), kernel_regularizer=l2(0.01)),
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

    # 8) Threshold selection on VAL
    val_preds  = clf.predict(X_val, verbose=0).flatten()
    thresholds = np.arange(0.0, 1.0001, 0.01)
    scores     = []
    for thr in thresholds:
        pbin = (val_preds >= thr).astype(int)
        tp = ((pbin==1)&(y_val==1)).sum()
        fn = ((pbin==0)&(y_val==1)).sum()
        fp = ((pbin==1)&(y_val==0)).sum()
        tn = ((pbin==0)&(y_val==0)).sum()
        tpr  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
        scores.append(0.7*tpr + 0.3*spec)
    best_threshold = float(thresholds[np.argmax(scores)])
    (Path(results_d)/"selected_threshold.txt").write_text(f"{best_threshold:.4f}\n")

    # 9) Final TEST‐time bootstrap & ROC
    df_boot, auc_mean, auc_std = bootstrap_threshold_metrics(
        y_te,
        clf.predict(X_te, verbose=0).flatten(),
        thresholds=np.array([best_threshold]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )
    df_boot.to_csv(Path(results_d)/"bootstrap_metrics.csv", index=False)
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

# ─── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # ─── Parse arguments ────────────────────────────────────────────────────
    pa = argparse.ArgumentParser()
    pa.add_argument("--user",      required=True)
    pa.add_argument("--fruit",     required=True)
    pa.add_argument("--scenario",  required=True)
    pa.add_argument("--output-dir", default="results")
    args = pa.parse_args()

    # ─── Prepare output directories ─────────────────────────────────────────
    top_out         = Path(args.output_dir)
    user_root       = top_out / args.user / f"{args.fruit}_{args.scenario}"
    shared_enc_root = top_out / "_global_encoders"
    shared_cnn_root = top_out / "global_cnns"
    user_root.mkdir(parents=True, exist_ok=True)

    # ─── Seed everything ─────────────────────────────────────────────────────
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # ─── Build day‐level splits for all users ───────────────────────────────
    all_splits = {}
    for u, pairs in ALLOWED_SCENARIOS.items():
        if (args.fruit, args.scenario) not in pairs:
            continue

        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df       = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
        neg_df       = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, "None")
        if neg_df.empty or len(neg_df) < len(pos_df):
            neg_df = derive_negative_labels(hr_df, pos_df, len(pos_df))

        try:
            tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
        except RuntimeError as e:
            print(f"Skipping user {u}: {e}")
            continue

        all_splits[u] = (tr_u, val_u, te_u)

    # ─── Ensure target user has splits ──────────────────────────────────────
    if args.user not in all_splits:
        print(f"Skipping user {args.user}: no data for {args.fruit}/{args.scenario}.")
        sys.exit(0)
    tr_days_u, val_days_u, te_days_u = all_splits[args.user]

    # ─── Tiny‐data guard (train/test) ───────────────────────────────────────
    hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / args.user)
    pos_u = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, args.scenario)
    neg_u = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, "None")
    if neg_u.empty or len(neg_u) < len(pos_u):
        neg_u = derive_negative_labels(hr_df_u, pos_u, len(pos_u))

    n_tr = _count_windows(pos_u, neg_u, hr_df_u, st_df_u, tr_days_u)
    n_te = _count_windows(pos_u, neg_u, hr_df_u, st_df_u, te_days_u)
    if n_tr < 2 or n_te < 2:
        _write_skip_file(user_root, tr_days_u, te_days_u, n_tr, n_te)
        sys.exit(0)

    # ─── Run Global-Supervised ──────────────────────────────────────────────
    df_gs, auc_gs_m, auc_gs_s = run_global_supervised(
        args.fruit, args.scenario, args.user, user_root,
        all_splits, shared_cnn_root
    )

    # ─── Run Personal-SSL ──────────────────────────────────────────────────
    df_ps, auc_ps_m, auc_ps_s = run_personal_ssl(
        args.user, args.fruit, args.scenario, user_root,
        tr_days_u, val_days_u, te_days_u
    )

    # ─── Run Global-SSL ────────────────────────────────────────────────────
    df_gl, auc_gl_m, auc_gl_s = run_global_ssl(
        args.user, args.fruit, args.scenario, user_root,
        shared_enc_root, all_splits
    )

    # ─── Comparison Summary ────────────────────────────────────────────────
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
            "Pipeline":          name,
            "Best_Threshold":    best["Threshold"],
            "Accuracy_Mean":     best["Accuracy_Mean"],
            "Accuracy_STD":      best["Accuracy_STD"],
            "Sensitivity_Mean":  best["Sensitivity_Mean"],
            "Sensitivity_STD":   best["Sensitivity_STD"],
            "Specificity_Mean":  best["Specificity_Mean"],
            "Specificity_STD":   best["Specificity_STD"],
            "AUC_Mean":          auc_m,
            "AUC_STD":           auc_s
        })
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(user_root / "comparison_summary.csv", index=False)

    print("\n--- Comparison Summary ---")
    print(df_summary.to_markdown(index=False))
