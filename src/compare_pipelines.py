#!/usr/bin/env python3
"""
compare_pipelines.py
====================

End‑to‑end benchmarking of three pipelines—Global‑Supervised CNN,
Personal‑SSL, and Global‑SSL—unless the target user/scenario has too few
windows.  If < 2 train *or* < 2 test windows, the script skips heavy work,
creates a folder, and writes not_enough_data.txt.

Directory layout (when data are sufficient):

{output_dir}/
  _global_encoders/                # shared SimCLR encoders (global SSL)
  {USER}/{FRUIT}_{SCENARIO}/
      global_supervised/
        models_saved/   results/
      personal_ssl/
        models_saved/   results/
      global_ssl/
        models_saved/   results/
      comparison_summary.csv
"""

import argparse, warnings, os, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D,
                                     GlobalAveragePooling1D,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

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
# Hyper‑parameters
# ------------------------------------------------------------------------- #
GS_EPOCHS, GS_BATCH, GS_LR, GS_PATIENCE = 200, 256, 1e-3, 15
BATCH_SSL, SSL_EPOCHS                   = 128, 80
CLF_EPOCHS, CLF_PATIENCE                = 200, 15
WINDOW_LEN                              = WINDOW_SIZE


# ------------------------------------------------------------------------- #
# Helper: generic classifier‑loss plot
# ------------------------------------------------------------------------- #
def plot_clf_losses(train, val, out_dir, fname):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train, label="Train")
    plt.plot(val,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Binary CE")
    plt.title(fname.replace('_', ' ').title())
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}.png")
    plt.close()


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
    return Sequential([
        Dense(64, activation='relu', input_shape=(dim,),
              kernel_regularizer=tf.keras.regularizers.l2(1e-2)),
        BatchNormalization(), Dropout(0.5),
        Dense(32, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-2)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])


# ------------------------------------------------------------------------- #
# Day‑based 75 % / 25 % split helper
# ------------------------------------------------------------------------- #
def _train_test_days(pos_df, neg_df):
    events = pd.concat([pos_df, neg_df])
    if events.empty:
        return np.array([]), np.array([])
    days = np.sort(events['hawaii_createdat_time'].dt.date.unique())
    if len(days) == 1:
        return days, np.array([])
    cut = int(round(0.75 * len(days)))
    return days[:cut], days[cut:]


# ------------------------------------------------------------------------- #
# -------- Tiny‑data guard helpers ---------------------------------------- #
# ------------------------------------------------------------------------- #
def _count_windows(df_p, df_n, hr_df, st_df, days):
    p = df_p[df_p['hawaii_createdat_time'].dt.date.isin(days)]
    n = df_n[df_n['hawaii_createdat_time'].dt.date.isin(days)]
    rows = pd.concat([
        process_label_window(p, hr_df, st_df, 1),
        process_label_window(n, hr_df, st_df, 0)
    ])
    return len(rows)


def _write_skip_file(root: Path,
                     train_days, test_days,
                     n_train, n_test):
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
# -------- Shared encoder loader/trainer ---------------------------------- #
# ------------------------------------------------------------------------- #
def _train_or_load_encoder(path, dtype, df, train_days, results_dir):
    if path.exists():
        enc = load_model(path); enc.trainable = False
        return enc

    mask = np.isin(df.index.date, train_days)
    vals = StandardScaler().fit_transform(df.loc[mask, 'value']
                                          .values.reshape(-1, 1))
    segs = create_windows(vals, WINDOW_SIZE, STEP_SIZE).astype('float32')
    if len(segs) == 0:
        raise RuntimeError(f"No {dtype} segments to train encoder.")
    n, idx = len(segs), np.random.permutation(len(segs))
    tr, va = segs[idx[:int(0.8 * n)]], segs[idx[int(0.8 * n):]]

    enc, head = build_simclr_encoder(WINDOW_SIZE), create_projection_head()
    tr_l, va_l = train_simclr(enc, head, tr, va,
                              batch_size=BATCH_SSL, epochs=SSL_EPOCHS)
    enc.save(path); enc.trainable = False
    plot_ssl_losses(tr_l, va_l, results_dir, encoder_name=f"{dtype}_ssl")
    return enc


# ------------------------------------------------------------------------- #
# -------- Pipeline #1 : Global‑Supervised CNN ---------------------------- #
# ------------------------------------------------------------------------- #
def run_global_supervised(fruit, scenario, uid, user_root):
    print(f"\n>> Global‑Supervised ({fruit}_{scenario})")
    out_dir   = user_root / 'global_supervised'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(exist_ok=True)

    tr_X, tr_y, te_X, te_y = [], [], [], []

    for other_uid, pairs in ALLOWED_SCENARIOS.items():
        if (fruit, scenario) not in pairs:
            continue

        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / other_uid)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / other_uid, fruit, scenario)
        neg_df = load_label_data(Path(BASE_DATA_DIR) / other_uid, fruit, 'None')

        def _rows(df_p, df_n, days):
            p = df_p[df_p['hawaii_createdat_time'].dt.date.isin(days)] \
                if days is not None else df_p
            n = df_n[df_n['hawaii_createdat_time'].dt.date.isin(days)] \
                if days is not None else df_n
            return pd.concat([
                process_label_window(p, hr_df, st_df, 1),
                process_label_window(n, hr_df, st_df, 0)
            ])

        if other_uid == uid:
            tr_d, te_d = _train_test_days(pos_df, neg_df)
            rows_tr = _rows(pos_df, neg_df, tr_d)
            rows_te = _rows(pos_df, neg_df, te_d)
            for h, s in zip(rows_tr['hr_seq'], rows_tr['st_seq']):
                tr_X.append(np.vstack([h, s]).T)
            tr_y += rows_tr['state_val'].tolist()
            for h, s in zip(rows_te['hr_seq'], rows_te['st_seq']):
                te_X.append(np.vstack([h, s]).T)
            te_y += rows_te['state_val'].tolist()
        else:
            rows = _rows(pos_df, neg_df, None)
            for h, s in zip(rows['hr_seq'], rows['st_seq']):
                tr_X.append(np.vstack([h, s]).T)
            tr_y += rows['state_val'].tolist()

    tr_X, tr_y = np.stack(tr_X), np.array(tr_y)
    te_X, te_y = np.stack(te_X), np.array(te_y)

    # Build & train CNN
    model = build_cnn()
    try:
        cw_arr = compute_class_weight('balanced',
                                      classes=np.unique(tr_y), y=tr_y)
        cw = {i: cw_arr[i] for i in range(len(cw_arr))}
    except ValueError:
        cw = None
    val_split = 0.1 if len(tr_X) >= 10 else 0.0
    es = EarlyStopping(monitor='val_loss', patience=GS_PATIENCE,
                       restore_best_weights=True, verbose=1)
    hist = model.fit(tr_X, tr_y,
                     validation_split=val_split,
                     epochs=GS_EPOCHS, batch_size=GS_BATCH,
                     class_weight=cw, callbacks=[es], verbose=2)
    plot_clf_losses(hist.history['loss'],
                    hist.history.get('val_loss', hist.history['loss']),
                    results_d, 'cnn_loss')
    model.save(models_d / 'cnn_classifier.keras')

    preds = model.predict(te_X, verbose=0).flatten()
    df_boot, auc_m, auc_s = bootstrap_threshold_metrics(te_y, preds)
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)
    plot_thresholds(te_y, preds, results_d,
                    f"{uid} {fruit}_{scenario} (global_supervised)")
    return df_boot, auc_m, auc_s


# ------------------------------------------------------------------------- #
# -------- Pipeline #2 : Personal‑SSL ------------------------------------- #
# ------------------------------------------------------------------------- #
def run_personal_ssl(uid, fruit, scenario, user_root):
    print(f">> Personal‑SSL       ({fruit}_{scenario})")
    out_dir   = user_root / 'personal_ssl'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(exist_ok=True)

    hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_df = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, 'None')
    tr_days, te_days = _train_test_days(pos_df, neg_df)
    if tr_days.size == 0 or te_days.size == 0:
        raise RuntimeError("Personal‑SSL needs events in both train & test.")

    paths = {'hr': models_d / 'hr_encoder.keras',
             'steps': models_d / 'steps_encoder.keras'}
    enc_hr = _train_or_load_encoder(paths['hr'], 'hr', hr_df, tr_days, results_d)
    enc_st = _train_or_load_encoder(paths['steps'], 'steps', st_df, tr_days, results_d)

    def _rows(df_p, df_n, days):
        p = df_p[df_p['hawaii_createdat_time'].dt.date.isin(days)]
        n = df_n[df_n['hawaii_createdat_time'].dt.date.isin(days)]
        return pd.concat([
            process_label_window(p, hr_df, st_df, 1),
            process_label_window(n, hr_df, st_df, 0)
        ])

    df_tr = _rows(pos_df, neg_df, tr_days)
    df_te = _rows(pos_df, neg_df, te_days)

    Xtr = np.concatenate([
        enc_hr.predict(np.expand_dims(np.stack(df_tr['hr_seq']), -1), verbose=0),
        enc_st.predict(np.expand_dims(np.stack(df_tr['st_seq']), -1), verbose=0)
    ], axis=1)
    ytr = df_tr['state_val'].values

    Xte = np.concatenate([
        enc_hr.predict(np.expand_dims(np.stack(df_te['hr_seq']), -1), verbose=0),
        enc_st.predict(np.expand_dims(np.stack(df_te['st_seq']), -1), verbose=0)
    ], axis=1)
    yte = df_te['state_val'].values

    clf = build_clf(Xtr.shape[1])
    try:
        cw_arr = compute_class_weight('balanced',
                                      classes=np.unique(ytr), y=ytr)
        cw = {i: cw_arr[i] for i in range(len(cw_arr))}
    except ValueError:
        cw = None

    clf.compile(optimizer=Adam(1e-3),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    val_split = 0.1 if len(Xtr) >= 10 else 0.0
    es = EarlyStopping(monitor='val_loss', patience=CLF_PATIENCE,
                       restore_best_weights=True, verbose=1)

    hist = clf.fit(Xtr, ytr,
                   validation_split=val_split,
                   epochs=CLF_EPOCHS, batch_size=16,
                   class_weight=cw, callbacks=[es], verbose=2)
    plot_clf_losses(hist.history['loss'],
                    hist.history.get('val_loss', hist.history['loss']),
                    results_d, 'clf_loss')
    clf.save(models_d / 'classifier.keras')

    preds = clf.predict(Xte, verbose=0).flatten()
    df_boot, auc_m, auc_s = bootstrap_threshold_metrics(yte, preds)
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)
    plot_thresholds(yte, preds, results_d,
                    f"{uid} {fruit}_{scenario} (personal_ssl)")
    return df_boot, auc_m, auc_s


# ------------------------------------------------------------------------- #
# -------- Pipeline #3 : Global‑SSL --------------------------------------- #
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

    print("   ↳ training global encoders (train‑day segments from all users)")
    losses = {}
    for dtype in ['hr', 'steps']:
        bank = []
        for u in ALLOWED_SCENARIOS:
            hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
            pos_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
            neg_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
            tr_days, _ = _train_test_days(pos_df, neg_df)
            if tr_days.size == 0:
                continue
            df = hr_df if dtype == 'hr' else st_df
            mask = np.isin(df.index.date, tr_days)
            vals = StandardScaler().fit_transform(
                df.loc[mask, 'value'].values.reshape(-1, 1))
            if len(vals) < WINDOW_SIZE:
                continue
            bank.append(
                create_windows(vals, WINDOW_SIZE, STEP_SIZE).astype('float32')
            )

        segs = np.concatenate(bank, axis=0)
        n, idx = len(segs), np.random.permutation(len(segs))
        tr, va = segs[idx[:int(0.8 * n)]], segs[idx[int(0.8 * n):]]

        enc, head = build_simclr_encoder(WINDOW_SIZE), create_projection_head()
        tr_l, va_l = train_simclr(enc, head, tr, va,
                                  batch_size=BATCH_SSL, epochs=SSL_EPOCHS)
        enc.save(paths[dtype]); enc.trainable = False
        losses[dtype] = (tr_l, va_l)

    plot_ssl_losses(*losses['hr'],    sdir, encoder_name="global_hr")
    plot_ssl_losses(*losses['steps'], sdir, encoder_name="global_steps")

    hr = load_model(paths['hr']); hr.trainable = False
    st = load_model(paths['steps']); st.trainable = False
    return hr, st, sdir


def _copy_shared_to_user(src_dir, models_d, results_d):
    for f in src_dir.glob("*"):
        dst = models_d / f.name if f.suffix == ".keras" else results_d / f.name
        if not dst.exists():
            shutil.copy2(f, dst)


def run_global_ssl(uid, fruit, scenario, user_root, shared_root):
    print(f">> Global‑SSL         ({fruit}_{scenario})")
    out_dir   = user_root / 'global_ssl'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(exist_ok=True)

    enc_hr, enc_st, enc_src = _ensure_global_encoders(shared_root,
                                                      fruit, scenario)
    _copy_shared_to_user(enc_src, models_d, results_d)

    hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_df = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, 'None')
    tr_days, te_days = _train_test_days(pos_df, neg_df)
    if tr_days.size == 0 or te_days.size == 0:
        raise RuntimeError("Global‑SSL needs events in both train & test.")

    def _rows(df_p, df_n, days):
        p = df_p[df_p['hawaii_createdat_time'].dt.date.isin(days)]
        n = df_n[df_n['hawaii_createdat_time'].dt.date.isin(days)]
        return pd.concat([
            process_label_window(p, hr_df, st_df, 1),
            process_label_window(n, hr_df, st_df, 0)
        ])

    df_tr = _rows(pos_df, neg_df, tr_days)
    df_te = _rows(pos_df, neg_df, te_days)

    Xtr = np.concatenate([
        enc_hr.predict(np.expand_dims(np.stack(df_tr['hr_seq']), -1), verbose=0),
        enc_st.predict(np.expand_dims(np.stack(df_tr['st_seq']), -1), verbose=0)
    ], axis=1)
    ytr = df_tr['state_val'].values

    Xte = np.concatenate([
        enc_hr.predict(np.expand_dims(np.stack(df_te['hr_seq']), -1), verbose=0),
        enc_st.predict(np.expand_dims(np.stack(df_te['st_seq']), -1), verbose=0)
    ], axis=1)
    yte = df_te['state_val'].values

    clf = build_clf(Xtr.shape[1])
    try:
        cw_arr = compute_class_weight('balanced',
                                      classes=np.unique(ytr), y=ytr)
        cw = {i: cw_arr[i] for i in range(len(cw_arr))}
    except ValueError:
        cw = None

    clf.compile(optimizer=Adam(1e-3),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    val_split = 0.1 if len(Xtr) >= 10 else 0.0
    es = EarlyStopping(monitor='val_loss', patience=CLF_PATIENCE,
                       restore_best_weights=True, verbose=1)
    hist = clf.fit(Xtr, ytr,
                   validation_split=val_split,
                   epochs=CLF_EPOCHS, batch_size=16,
                   class_weight=cw, callbacks=[es], verbose=2)
    plot_clf_losses(hist.history['loss'],
                    hist.history.get('val_loss', hist.history['loss']),
                    results_d, 'clf_loss')
    clf.save(models_d / 'classifier.keras')

    preds = clf.predict(Xte, verbose=0).flatten()
    df_boot, auc_m, auc_s = bootstrap_threshold_metrics(yte, preds)
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)
    plot_thresholds(yte, preds, results_d,
                    f"{uid} {fruit}_{scenario} (global_ssl)")
    return df_boot, auc_m, auc_s


# ------------------------------------------------------------------------- #
# -----------------------------   MAIN   ---------------------------------- #
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    pa = argparse.ArgumentParser()
    pa.add_argument("--user",     required=True)
    pa.add_argument("--fruit",    required=True)
    pa.add_argument("--scenario", required=True)
    pa.add_argument("--output-dir", default="results")
    args = pa.parse_args()

    top_out   = Path(args.output_dir)
    user_root = top_out / args.user / f"{args.fruit}_{args.scenario}"

    # 1) tiny‑data guard
    hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / args.user)
    pos_df = load_label_data(Path(BASE_DATA_DIR) / args.user,
                             args.fruit, args.scenario)
    neg_df = load_label_data(Path(BASE_DATA_DIR) / args.user,
                             args.fruit, 'None')
    tr_days, te_days = _train_test_days(pos_df, neg_df)
    n_tr = _count_windows(pos_df, neg_df, hr_df, st_df, tr_days)
    n_te = _count_windows(pos_df, neg_df, hr_df, st_df, te_days)

    if n_tr < 2 or n_te < 2:
        _write_skip_file(user_root, tr_days, te_days, n_tr, n_te)
        sys.exit(0)

    # 2) full benchmark
    shared_enc_root = top_out / '_global_encoders'
    if user_root.exists():
        shutil.rmtree(user_root)
    user_root.mkdir(parents=True, exist_ok=True)

    df_gs, auc_gs_m, auc_gs_s = run_global_supervised(
        args.fruit, args.scenario, args.user, user_root)
    df_ps, auc_ps_m, auc_ps_s = run_personal_ssl(
        args.user, args.fruit, args.scenario, user_root)
    df_gl, auc_gl_m, auc_gl_s = run_global_ssl(
        args.user, args.fruit, args.scenario, user_root, shared_enc_root)

    rows = []
    for name, (df, auc_m, auc_s) in [
        ("global_supervised", (df_gs, auc_gs_m, auc_gs_s)),
        ("personal_ssl",     (df_ps, auc_ps_m, auc_ps_s)),
        ("global_ssl",       (df_gl, auc_gl_m, auc_gl_s))
    ]:
        best = df.loc[df['Accuracy_Mean'].idxmax()]
        rows.append({
            "Pipeline":          name,
            "Best_Threshold":    best["Threshold"],
            "Accuracy_Mean":     best["Accuracy_Mean"],
            "Accuracy_STD":      best["Accuracy_STD"],
            "Sensitivity_Mean":  best["Sensitivity_Mean"],
            "Sensitivity_STD":   best["Sensitivity_STD"],
            "Specificity_Mean":  best["Specificity_Mean"],
            "Specificity_STD":   best["Specificity_STD"],
            "AUC_Mean": auc_m,
            "AUC_STD":  auc_s
        })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(user_root / "comparison_summary.csv", index=False)
    print("\n--- Comparison Summary ---")
    print(df_summary.to_markdown(index=False))