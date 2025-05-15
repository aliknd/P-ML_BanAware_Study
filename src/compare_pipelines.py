#!/usr/bin/env python3
"""
compare_pipelines.py
====================

End-to-end benchmarking of three pipelines—Global-Supervised CNN,
Personal-SSL, and Global-SSL—unless the target user/scenario has too few
windows.  If < 2 train *or* < 2 test windows, the script skips heavy work,
creates a folder, and writes not_enough_data.txt.

Each pipeline now:
  - Splits days so that ~75% of windows are in train, ≥25% in test,
    by whole‐day grouping.
  - Ensures the test set contains at least one positive and one negative.
  - Writes a `split_details.txt` in its results/ folder, showing per‐user
    TRAIN/TEST days, window counts, and class distributions.
"""

import argparse, warnings, os, shutil, sys
from pathlib import Path

import random
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

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ---- project helpers ---------------------------------------------------- #
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

# ------------------------------------------------------------------------- #
# Hyper-parameters
# ------------------------------------------------------------------------- #
GS_EPOCHS, GS_BATCH, GS_LR, GS_PATIENCE = 200, 32, 1e-3, 15
BATCH_SSL, SSL_EPOCHS                   = 32, 100
CLF_EPOCHS, CLF_PATIENCE                = 200, 15
WINDOW_LEN                              = WINDOW_SIZE


# ------------------------------------------------------------------------- #
# Helper: generic classifier-loss plot
# ------------------------------------------------------------------------- #
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


def _count_windows(df_p, df_n, hr_df, st_df, days):
    """
    Count how many label‐windows fall on the given `days`.
    """
    p = df_p[df_p['hawaii_createdat_time'].dt.date.isin(days)]
    n = df_n[df_n['hawaii_createdat_time'].dt.date.isin(days)]
    rows = pd.concat([
        process_label_window(p, hr_df, st_df, 1),
        process_label_window(n, hr_df, st_df, 0),
    ])
    return len(rows)


def _write_skip_file(root: Path,
                     train_days, test_days,
                     n_train, n_test):
    """
    When there aren’t enough windows (<2 train or <2 test), write out a
    not_enough_data.txt with summary and exit.
    """
    root.mkdir(parents=True, exist_ok=True)
    msg = [
        "SKIPPED – Not enough windows for benchmarking.\n",
        f"Train days ({len(train_days)}): {list(train_days)}",
        f"Test  days ({len(test_days)}): {list(test_days)}",
        f"Train windows: {n_train}",
        f"Test  windows: {n_test}",
        "",
        "Minimum required: 2 train windows AND 2 test windows."
    ]
    (root / "not_enough_data.txt").write_text("\n".join(msg))
    print(">> SKIPPED – see not_enough_data.txt for details")

# ------------------------------------------------------------------------- #
# Model builders
# ------------------------------------------------------------------------- #
def build_cnn():
    m = Sequential([
        Conv1D(32, 3, activation="relu", input_shape=(WINDOW_LEN, 2)),
        MaxPooling1D(2),
        Conv1D(64, 3, activation="relu"),
        GlobalAveragePooling1D(),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=Adam(GS_LR), loss="binary_crossentropy",
              metrics=["accuracy"])
    return m

def build_clf(dim):
    m = Sequential([
        Dense(64, activation='relu', input_shape=(dim,),
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(), Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy"])
    return m


# ------------------------------------------------------------------------- #
# Day-based 75%/25% random split helper
# ------------------------------------------------------------------------- #
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


# ------------------------------------------------------------------------- #
# Enhanced split: ensure both classes in test
# ------------------------------------------------------------------------- #
def ensure_train_test_days(pos_df, neg_df, hr_df, st_df):
    tr_days, te_days = _train_test_days_by_samples(pos_df, neg_df, hr_df, st_df)
    tr_days, te_days = list(tr_days), list(te_days)

    def count_labels(days):
        p = pos_df[pos_df['hawaii_createdat_time'].dt.date.isin(days)]
        n = neg_df[neg_df['hawaii_createdat_time'].dt.date.isin(days)]
        rows = pd.concat([
            process_label_window(p, hr_df, st_df, 1),
            process_label_window(n, hr_df, st_df, 0)
        ])
        if rows.empty:
            return 0, 0
        pos = int(rows['state_val'].sum())
        neg = len(rows) - pos
        return pos, neg

    pos_te, neg_te = count_labels(te_days)
    if pos_te == 0 or neg_te == 0:
        for d in tr_days[:]:
            p, n = count_labels([d])
            if (pos_te == 0 and p > 0) or (neg_te == 0 and n > 0):
                tr_days.remove(d)
                te_days.append(d)
                break

    return np.array(tr_days), np.array(te_days)


# ------------------------------------------------------------------------- #
# Window collection helper
# ------------------------------------------------------------------------- #
def collect_windows(df_p, df_n, hr_df, st_df, days):
    p = df_p[df_p['hawaii_createdat_time'].dt.date.isin(days)]
    n = df_n[df_n['hawaii_createdat_time'].dt.date.isin(days)]
    return pd.concat([
        process_label_window(p, hr_df, st_df, 1),
        process_label_window(n, hr_df, st_df, 0)
    ])


# ------------------------------------------------------------------------- #
# Write split_details.txt
# ------------------------------------------------------------------------- #
def write_split_details(results_dir, pipeline, train_info, test_info):
    results_dir = Path(results_dir)
    with open(results_dir / "split_details.txt", "w") as f:
        f.write(f"Pipeline: {pipeline}\n\n")
        for user, info in train_info.items():
            df_i = info["df"]
            pos_i = int(df_i['state_val'].sum())
            neg_i = len(df_i) - pos_i
            f.write(f"User {user} TRAIN days: {info['days']}\n")
            f.write(f"   windows={len(df_i)}  (+={pos_i}, -={neg_i})\n\n")
        uid, te_days, df_te = test_info
        pos_te = int(df_te['state_val'].sum())
        neg_te = len(df_te) - pos_te
        f.write(f"User {uid} TEST  days: {te_days}\n")
        f.write(f"   windows={len(df_te)}  (+={pos_te}, -={neg_te})\n")


# ------------------------------------------------------------------------- #
# Shared encoder loader/trainer (Global-SSL)
# ------------------------------------------------------------------------- #
def _ensure_global_encoders(shared_root, fruit, scenario):
    sdir = Path(shared_root) / f"{fruit}_{scenario}"
    sdir.mkdir(parents=True, exist_ok=True)

    paths = {'hr': sdir / 'hr_encoder.keras',
             'steps': sdir / 'steps_encoder.keras'}
    if all(p.exists() for p in paths.values()):
        hr = load_model(paths['hr']); hr.trainable = False
        st = load_model(paths['steps']); st.trainable = False
        return hr, st, sdir

    losses = {}
    for dtype in ['hr', 'steps']:
        bank = []
        for u, pairs in ALLOWED_SCENARIOS.items():
            if (fruit, scenario) not in pairs:
                continue
            hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
            pos_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
            neg_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
            tr_days, _ = _train_test_days_by_samples(pos_df, neg_df, hr_df, st_df)
            if not tr_days.size:
                continue
            df = hr_df if dtype == 'hr' else st_df
            mask = np.isin(df.index.date, tr_days)
            vals = StandardScaler().fit_transform(df.loc[mask,'value'].values.reshape(-1,1))
            if len(vals) < WINDOW_SIZE:
                continue
            segs = create_windows(vals, WINDOW_SIZE, STEP_SIZE).astype('float32')
            bank.append(segs)
        if not bank:
            raise RuntimeError(f"No train-day segments found for global {dtype} SSL!")
        segs = np.concatenate(bank, axis=0)
        n, idx = len(segs), np.random.permutation(len(segs))
        tr, va = segs[idx[:int(0.8*n)]], segs[idx[int(0.8*n):]]
        enc = build_simclr_encoder(WINDOW_SIZE)
        head = create_projection_head()
        tr_l, va_l = train_simclr(enc, head, tr, va,
                                  batch_size=BATCH_SSL, epochs=SSL_EPOCHS)
        enc.save(paths[dtype]); enc.trainable = False
        losses[dtype] = (tr_l, va_l)

    plot_ssl_losses(*losses['hr'],    sdir, encoder_name="global_hr")
    plot_ssl_losses(*losses['steps'], sdir, encoder_name="global_steps")
    hr = load_model(paths['hr']); hr.trainable = False
    st = load_model(paths['steps']); st.trainable = False
    return hr, st, sdir


# ------------------------------------------------------------------------- #
# Load or train per-user encoder (Personal-SSL)
# ------------------------------------------------------------------------- #
def _train_or_load_encoder(path, dtype, df, train_days, results_dir):
    if path.exists():
        enc = load_model(path); enc.trainable = False
        return enc

    mask = np.isin(df.index.date, train_days)
    vals = StandardScaler().fit_transform(df.loc[mask,'value'].values.reshape(-1,1))
    segs = create_windows(vals, WINDOW_SIZE, STEP_SIZE).astype('float32')
    if not len(segs):
        raise RuntimeError(f"No {dtype} segments to train encoder.")
    n, idx = len(segs), np.random.permutation(len(segs))
    tr, va = segs[idx[:int(0.8*n)]], segs[idx[int(0.8*n):]]
    enc = build_simclr_encoder(WINDOW_SIZE)
    head = create_projection_head()
    tr_l, va_l = train_simclr(enc, head, tr, va,
                              batch_size=BATCH_SSL, epochs=SSL_EPOCHS)
    enc.save(path); enc.trainable = False
    plot_ssl_losses(tr_l, va_l, results_dir, encoder_name=f"{dtype}_ssl")
    return enc


def _copy_shared_to_user(src_dir, models_d, results_d):
    for f in Path(src_dir).glob("*"):
        dst = models_d / f.name if f.suffix==".keras" else results_d / f.name
        if not dst.exists():
            shutil.copy2(f, dst)


# ------------------------------------------------------------------------- #
# Pipeline #1 : Global-Supervised CNN
# ------------------------------------------------------------------------- #
def run_global_supervised(fruit, scenario, uid, user_root, tr_days_u, te_days_u):
    """
    Global-Supervised CNN:
      - Builds a 1D-CNN classifier.
      - Trains on ALL other users’ train-days + the target user’s tr_days_u.
      - Tests on the target user’s te_days_u only.
    """
    print(f"\n>> Global-Supervised ({fruit}_{scenario})")
    out_dir   = Path(user_root) / 'global_supervised'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    # --- 1) Load target-user signals + labels
    hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_u = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_u = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, 'None')

    # --- 2) Collect other-users’ training windows
    train_info = {}
    for other_uid, pairs in ALLOWED_SCENARIOS.items():
        if other_uid == uid or (fruit, scenario) not in pairs:
            continue
        hr_o, st_o = load_signal_data(Path(BASE_DATA_DIR) / other_uid)
        pos_o = load_label_data(Path(BASE_DATA_DIR) / other_uid, fruit, scenario)
        neg_o = load_label_data(Path(BASE_DATA_DIR) / other_uid, fruit, 'None')

        # Use the same _train_test_days_by_samples logic for other users,
        # but we only care about their train days here:
        tr_o, _ = _train_test_days_by_samples(pos_o, neg_o, hr_o, st_o)
        if len(tr_o) == 0:
            continue

        df_o = collect_windows(pos_o, neg_o, hr_o, st_o, tr_o)
        train_info[other_uid] = {
            "days": list(tr_o),
            "df":   df_o
        }

    # --- 3) Collect target-user’s train + test windows
    df_tr_u = collect_windows(pos_u, neg_u, hr_df_u, st_df_u, tr_days_u)
    df_te_u = collect_windows(pos_u, neg_u, hr_df_u, st_df_u, te_days_u)
    train_info[uid] = {
        "days": list(tr_days_u),
        "df":   df_tr_u
    }

    # --- 4) Write split_details.txt
    write_split_details(
        results_d,
        "global_supervised",
        train_info,
        (uid, list(te_days_u), df_te_u)
    )

    # --- 5) Stack up features & labels
    #   - each window is (hr_seq, st_seq) → shape (WINDOW_LEN,2)
    df_all_tr = pd.concat([info["df"] for info in train_info.values()], ignore_index=True)
    tr_X = np.stack([
        np.vstack([h, s]).T
        for h,s in zip(df_all_tr['hr_seq'], df_all_tr['st_seq'])
    ])
    tr_y = df_all_tr['state_val'].values

    te_X = np.stack([
        np.vstack([h, s]).T
        for h,s in zip(df_te_u['hr_seq'], df_te_u['st_seq'])
    ])
    te_y = df_te_u['state_val'].values

    # --- 6) Build, compile, and train the CNN
    model = build_cnn()

    classes = np.unique(tr_y)
    if len(classes) == 2:
        cw_arr = compute_class_weight('balanced', classes=classes, y=tr_y)
        class_weight = {i: cw_arr[i] for i in range(len(cw_arr))}
    else:
        class_weight = {0: 1, 1: 1}

    es = EarlyStopping(monitor='val_loss',
                       patience=GS_PATIENCE,
                       restore_best_weights=True,
                       verbose=1)

    val_split = 0.1 if len(tr_X) >= 10 else 0.0

    hist = model.fit(
        tr_X, tr_y,
        validation_split=val_split,
        epochs=GS_EPOCHS,
        batch_size=GS_BATCH,
        class_weight=class_weight,
        callbacks=[es],
        verbose=2
    )

    # save loss curves + model
    plot_clf_losses(hist.history['loss'],
                    hist.history.get('val_loss', hist.history['loss']),
                    results_d, 'cnn_loss')
    model.save(models_d / 'cnn_classifier.keras')

    # --- 7) Evaluate on test set via bootstrap metrics
    preds = model.predict(te_X, verbose=0).flatten()
    df_boot, auc_mean, auc_std = bootstrap_threshold_metrics(te_y, preds)
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)
    plot_thresholds(te_y, preds, results_d,
                    f"{uid} {fruit}_{scenario} (global_supervised)")

    return df_boot, auc_mean, auc_std

# ------------------------------------------------------------------------- #
# Pipeline #2 : Personal-SSL
# ------------------------------------------------------------------------- #
def run_personal_ssl(uid, fruit, scenario, user_root, tr_days_u, te_days_u):
    """
    Personal-SSL:
      - Trains (or loads) a SimCLR encoder *only* on the target user’s tr_days_u.
      - Builds a classifier on those learned features.
      - Tests on the target user’s te_days_u only.
    """
    print(f">> Personal-SSL       ({fruit}_{scenario})")
    out_dir   = Path(user_root) / 'personal_ssl'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    # 1) Load this user’s signals and labels
    hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df       = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_df       = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, 'None')

    # 2) Train or load per-user encoders on tr_days_u
    paths = {
        'hr':    models_d / 'hr_encoder.keras',
        'steps': models_d / 'steps_encoder.keras'
    }
    enc_hr = _train_or_load_encoder(paths['hr'], 'hr',   hr_df, tr_days_u, results_d)
    enc_st = _train_or_load_encoder(paths['steps'], 'steps', st_df, tr_days_u, results_d)

    # 3) Collect this user’s train + test windows
    df_tr = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
    df_te = collect_windows(pos_df, neg_df, hr_df, st_df, te_days_u)

    # 4) Write split_details.txt
    write_split_details(
        results_d,
        "personal_ssl",
        { uid: {"days": list(tr_days_u), "df": df_tr} },
        (uid, list(te_days_u), df_te)
    )

    # 5) Encode windows to features
    #    - each encoder expects input shape (batch, WINDOW_LEN, 1)
    h_tr = np.stack(df_tr['hr_seq'])   # shape: (n_tr, WINDOW_LEN)
    s_tr = np.stack(df_tr['st_seq'])
    H_tr = enc_hr.predict(np.expand_dims(h_tr, -1), verbose=0)
    S_tr = enc_st.predict(np.expand_dims(s_tr, -1), verbose=0)
    Xtr  = np.concatenate([H_tr, S_tr], axis=1)
    ytr  = df_tr['state_val'].values

    h_te = np.stack(df_te['hr_seq'])
    s_te = np.stack(df_te['st_seq'])
    H_te = enc_hr.predict(np.expand_dims(h_te, -1), verbose=0)
    S_te = enc_st.predict(np.expand_dims(s_te, -1), verbose=0)
    Xte  = np.concatenate([H_te, S_te], axis=1)
    yte  = df_te['state_val'].values

    # 6) Build and train the classifier
    clf = build_clf(Xtr.shape[1])
    es  = EarlyStopping(monitor='val_loss',
                        patience=CLF_PATIENCE,
                        restore_best_weights=True,
                        verbose=1)
    val_split = 0.1 if len(Xtr) >= 10 else 0.0

    classes = np.unique(ytr)
    if len(classes) == 2:
        cw_arr = compute_class_weight('balanced', classes=classes, y=ytr)
        class_weight = {i: cw_arr[i] for i in range(len(cw_arr))}
    else:
        class_weight = {0: 1, 1: 1}

    hist = clf.fit(
        Xtr, ytr,
        validation_split=val_split,
        epochs=CLF_EPOCHS,
        batch_size=16,
        class_weight=class_weight,
        callbacks=[es],
        verbose=2
    )

    # 7) Save loss curves and model
    plot_clf_losses(hist.history['loss'],
                    hist.history.get('val_loss', hist.history['loss']),
                    results_d, 'clf_loss')
    clf.save(models_d / 'classifier.keras')

    # 8) Evaluate on test set
    preds, = clf.predict(Xte, verbose=0).flatten(), 
    df_boot, auc_mean, auc_std = bootstrap_threshold_metrics(yte, preds)
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)
    plot_thresholds(yte, preds, results_d,
                    f"{uid} {fruit}_{scenario} (personal_ssl)")

    return df_boot, auc_mean, auc_std

# ------------------------------------------------------------------------- #
# Pipeline #3 : Global-SSL
# ------------------------------------------------------------------------- #
def run_global_ssl(uid, fruit, scenario, user_root, shared_root,
                   tr_days_u, te_days_u):
    """
    Global-SSL:
      - Ensures or trains a shared SimCLR encoder on all users’ data.
      - Copies the trained encoders into this user’s folder.
      - Builds train set from all other users’ *all* windows plus this user’s tr_days_u.
      - Tests on this user’s te_days_u only.
    """
    print(f">> Global-SSL         ({fruit}_{scenario})")
    out_dir   = Path(user_root) / 'global_ssl'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    # 1) Ensure or train shared encoders
    enc_hr, enc_st, enc_src = _ensure_global_encoders(shared_root, fruit, scenario)
    _copy_shared_to_user(enc_src, models_d, results_d)

    # 2) Load target‐user data
    hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_u = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_u = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, 'None')

    # 3) Build train_info for ALL users
    train_info = {}

    # 3a) Other users
    for other_uid, pairs in ALLOWED_SCENARIOS.items():
        if other_uid == uid or (fruit, scenario) not in pairs:
            continue

        # load their raw signals and labels
        hr_o, st_o = load_signal_data(Path(BASE_DATA_DIR) / other_uid)
        pos_o      = load_label_data(Path(BASE_DATA_DIR) / other_uid, fruit, scenario)
        neg_o      = load_label_data(Path(BASE_DATA_DIR) / other_uid, fruit, 'None')

        # a) collect **all** their windows for training
        rows_o = pd.concat([
            process_label_window(pos_o, hr_o, st_o, 1),
            process_label_window(neg_o, hr_o, st_o, 0)
        ])

        # b) derive their list of event-days from the original labels
        days_o = np.sort(
            pd.concat([pos_o, neg_o])['hawaii_createdat_time']
              .dt.date
              .unique()
        ).tolist()

        train_info[other_uid] = {
            "days": days_o,
            "df":   rows_o
        }

    # 3b) Target user’s training windows (only tr_days_u)
    rows_tr_u = collect_windows(pos_u, neg_u, hr_df_u, st_df_u, tr_days_u)
    train_info[uid] = {
        "days": list(tr_days_u),
        "df":   rows_tr_u
    }

    # 4) Collect test windows for the target user
    df_te = collect_windows(pos_u, neg_u, hr_df_u, st_df_u, te_days_u)

    # 5) Write split_details.txt with the full train_info
    write_split_details(
        results_d,
        "global_ssl",
        train_info,
        (uid, list(te_days_u), df_te)
    )

    # 6) Prepare training data for the classifier
    hr_train, st_train, y_train = [], [], []
    for info in train_info.values():
        for h, s, lbl in zip(info["df"]['hr_seq'],
                             info["df"]['st_seq'],
                             info["df"]['state_val']):
            hr_train.append(h)
            st_train.append(s)
            y_train.append(lbl)

    # 7) Featurize with the shared encoders
    Xtr_hr = enc_hr.predict(np.expand_dims(np.stack(hr_train), -1), verbose=0)
    Xtr_st = enc_st.predict(np.expand_dims(np.stack(st_train), -1), verbose=0)
    Xtr    = np.concatenate([Xtr_hr, Xtr_st], axis=1)
    ytr    = np.array(y_train)

    Xte_hr = enc_hr.predict(np.expand_dims(np.stack(df_te['hr_seq']), -1), verbose=0)
    Xte_st = enc_st.predict(np.expand_dims(np.stack(df_te['st_seq']), -1), verbose=0)
    Xte    = np.concatenate([Xte_hr, Xte_st], axis=1)
    yte    = df_te['state_val'].values

    # 8) Train the dense classifier as before…
    clf = build_clf(Xtr.shape[1])
    es  = EarlyStopping(monitor='val_loss',
                        patience=CLF_PATIENCE,
                        restore_best_weights=True,
                        verbose=1)
    val_split = 0.1 if len(Xtr) >= 10 else 0.0

    classes = np.unique(ytr)
    if len(classes) == 2:
        cw_arr = compute_class_weight('balanced', classes=classes, y=ytr)
        class_weight = {i: cw_arr[i] for i in range(len(cw_arr))}
    else:
        class_weight = {0: 1, 1: 1}

    hist = clf.fit(
        Xtr, ytr,
        validation_split=val_split,
        epochs=CLF_EPOCHS,
        batch_size=16,
        class_weight=class_weight,
        callbacks=[es],
        verbose=2
    )

    # 9) Save losses & model, then bootstrap/test
    plot_clf_losses(hist.history['loss'],
                    hist.history.get('val_loss', hist.history['loss']),
                    results_d, 'clf_loss')
    clf.save(models_d / 'classifier.keras')

    preds = clf.predict(Xte, verbose=0).flatten()
    df_boot, auc_mean, auc_std = bootstrap_threshold_metrics(yte, preds)
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)
    plot_thresholds(yte, preds, results_d,
                    f"{uid} {fruit}_{scenario} (global_ssl)")

    return df_boot, auc_mean, auc_std

# ------------------------------------------------------------------------- #
# Main entrypoint
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    pa = argparse.ArgumentParser()
    pa.add_argument("--user",     required=True)
    pa.add_argument("--fruit",    required=True)
    pa.add_argument("--scenario", required=True)
    pa.add_argument("--output-dir", default="results")
    args = pa.parse_args()

    # create output folders
    top_out   = Path(args.output_dir)
    user_root = top_out / args.user / f"{args.fruit}_{args.scenario}"
    shared_enc_root = top_out / '_global_encoders'
    user_root.mkdir(parents=True, exist_ok=True)

    # one-time split for the *target* user
    hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR)/args.user)
    pos_u = load_label_data(Path(BASE_DATA_DIR)/args.user, args.fruit, args.scenario)
    neg_u = load_label_data(Path(BASE_DATA_DIR)/args.user, args.fruit, 'None')
    tr_days_u, te_days_u = ensure_train_test_days(pos_u, neg_u, hr_df_u, st_df_u)

    # tiny-data guard
    n_tr = _count_windows(pos_u, neg_u, hr_df_u, st_df_u, tr_days_u)
    n_te = _count_windows(pos_u, neg_u, hr_df_u, st_df_u, te_days_u)
    if n_tr < 2 or n_te < 2:
        _write_skip_file(user_root, tr_days_u, te_days_u, n_tr, n_te)
        sys.exit(0)

    # now run each pipeline with the same split
    df_gs, auc_gs_m, auc_gs_s = run_global_supervised(
        args.fruit, args.scenario, args.user, user_root,
        tr_days_u, te_days_u
    )
    df_ps, auc_ps_m, auc_ps_s = run_personal_ssl(
        args.user, args.fruit, args.scenario, user_root,
        tr_days_u, te_days_u
    )
    df_gl, auc_gl_m, auc_gl_s = run_global_ssl(
        args.user, args.fruit, args.scenario, user_root,
        shared_enc_root, tr_days_u, te_days_u
    )

    # comparison summary (unchanged)
    rows = []
    for name, (df, auc_m, auc_s) in [
        ("global_supervised", (df_gs, auc_gs_m, auc_gs_s)),
        ("personal_ssl",     (df_ps, auc_ps_m, auc_ps_s)),
        ("global_ssl",       (df_gl, auc_gl_m, auc_gl_s))
    ]:
        tmp = df.copy()
        tmp['Balance'] = tmp[['Sensitivity_Mean','Specificity_Mean']].min(axis=1)
        best = tmp.sort_values(['Balance','Sensitivity_Mean'], ascending=[False,False]).iloc[0]
        rows.append({
            "Pipeline":       name,
            "Best_Threshold": best["Threshold"],
            "Accuracy_Mean":  best["Accuracy_Mean"],
            "Accuracy_STD":   best["Accuracy_STD"],
            "Sensitivity_Mean":  best["Sensitivity_Mean"],
            "Sensitivity_STD":   best["Sensitivity_STD"],
            "Specificity_Mean": best["Specificity_Mean"],
            "Specificity_STD":  best["Specificity_STD"],
            "AUC_Mean": auc_m,
            "AUC_STD":  auc_s
        })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(user_root/"comparison_summary.csv", index=False)
    print("\n--- Comparison Summary ---")
    print(df_summary.to_markdown(index=False))