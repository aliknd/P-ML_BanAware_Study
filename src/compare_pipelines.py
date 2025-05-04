#!/usr/bin/env python3
"""
compare_pipelines.py
===================

Orchestrate and compare three classification pipelines for a given user and scenario.
Produces a new output directory structured as:

{output_dir}/
  {user_id}/
    {fruit}_{scenario}/
      global_supervised/
        models_saved/
        results/
      personal_ssl/
        models_saved/
        results/
      global_ssl/
        models_saved/
        results/

All original models/encoders are copied into models_saved, even if already present. New models are trained only if originals are missing.

Usage:
    python3 -m src.compare_pipelines --user ID10 --fruit Nectarine --scenario Use [--output-dir results_compare]
"""
import argparse
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Shared utilities
from src.classifier_utils import (
    BASE_DATA_DIR,
    load_signal_data,
    load_label_data,
    process_label_window,
)
from src.chart_utils import bootstrap_threshold_metrics, plot_thresholds, plot_ssl_losses
from src.signal_utils import (
    WINDOW_SIZE,
    STEP_SIZE,
    create_windows,
    build_simclr_encoder,
    train_simclr,
)
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Pipeline original roots
from src.global_supervised_pipeline import (
    EXP_ROOT as GS_ROOT,
    build_cnn,
    collect_scenario,
    PATIENCE,
    EPOCHS,
    BATCH,
)
from src.personal_ssl_pipeline import (
    EXP_ROOT as PS_ROOT,
    build_clf,
    BATCH_SSL,
    CLF_EPOCHS,
    create_projection_head,
)
from src.global_ssl_pipeline import (
    EXP_ROOT as GSSL_ROOT,
    ensure_encoders,
    CLF_EPOCHS as GSSL_CLF_EPOCHS,
)

# --------------------------------------------------------------------------- #
# Pipeline wrappers
# --------------------------------------------------------------------------- #

def run_global_supervised(fruit, scenario, user, base_dir):
    pipeline = base_dir / 'global_supervised'
    models_dir = pipeline / 'models_saved'
    results_dir = pipeline / 'results'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    old_model = GS_ROOT / f"{fruit}_{scenario}" / f"{fruit}_{scenario}_cnn.keras"
    new_model = models_dir / f"global_{fruit}_{scenario}.keras"

    if old_model.exists():
        shutil.copy(old_model, new_model)
        model = load_model(new_model)
    else:
        X, y = collect_scenario(fruit, scenario)
        train_idx, _ = train_test_split(
            np.arange(len(y)), test_size=0.1, random_state=42, stratify=y
        )
        tr_idx, val_idx = train_test_split(
            train_idx, test_size=0.111111, random_state=42, stratify=y[train_idx]
        )
        model = build_cnn()
        cw_vals = compute_class_weight('balanced', classes=np.unique(y), y=y)
        cw = dict(enumerate(cw_vals))
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE, restore_best_weights=True
        )
        model.fit(
            X[tr_idx], y[tr_idx],
            validation_data=(X[val_idx], y[val_idx]),
            epochs=EPOCHS, batch_size=BATCH,
            class_weight=cw, callbacks=[es], verbose=2
        )
        model.save(new_model)

    # Threshold analysis
    u_dir = results_dir / user
    u_dir.mkdir(exist_ok=True)
    hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR)/user)
    pos = load_label_data(Path(BASE_DATA_DIR)/user, fruit, scenario)
    neg = load_label_data(Path(BASE_DATA_DIR)/user, fruit, 'None')
    rows = pd.concat([
        process_label_window(pos, hr_df, st_df, 1),
        process_label_window(neg, hr_df, st_df, 0),
    ], ignore_index=True)
    X_u = np.stack([np.vstack([h, s]).T for h, s in zip(rows['hr_seq'], rows['st_seq'])])
    y_u = rows['state_val'].values
    preds = model.predict(X_u, verbose=0).flatten()
    df, auc_mean, auc_std = bootstrap_threshold_metrics(y_u, preds)
    df.to_csv(u_dir / 'bootstrap_metrics.csv', index=False)
    plot_thresholds(y_u, preds, u_dir, f"{user} {fruit}_{scenario} (global_supervised)")
    return df, auc_mean, auc_std


def run_personal_ssl(user, fruit, scenario, base_dir):
    pipeline = base_dir / 'personal_ssl'
    models_dir = pipeline / 'models_saved'
    results_dir = pipeline / 'results'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR)/user)
    pos = load_label_data(Path(BASE_DATA_DIR)/user, fruit, scenario)
    neg = load_label_data(Path(BASE_DATA_DIR)/user, fruit, 'None')
    dfp = process_label_window(pos, hr_df, st_df, 1)
    dfn = process_label_window(neg, hr_df, st_df, 0)
    df_all = pd.concat([dfp, dfn], ignore_index=True)
    X_hr = np.expand_dims(np.stack(df_all['hr_seq'].tolist()), -1)
    X_st = np.expand_dims(np.stack(df_all['st_seq'].tolist()), -1)

    # HR encoder
    old_hr = PS_ROOT / user / 'hr_encoder.keras'
    new_hr = models_dir / 'hr_encoder.keras'
    if old_hr.exists():
        shutil.copy(old_hr, new_hr)
        enc_hr = load_model(new_hr)
    else:
        vals = np.vstack(df_all['hr_seq'].tolist())
        segs = create_windows(vals, WINDOW_SIZE, STEP_SIZE).astype('float32')
        enc_hr = build_simclr_encoder(WINDOW_SIZE)
        head = create_projection_head()
        tr, va = train_simclr(
            enc_hr, head,
            segs[:int(0.8*len(segs))], segs[int(0.8*len(segs)):],
            batch_size=BATCH_SSL
        )
        enc_hr.save(new_hr)
        plot_ssl_losses(tr, va, results_dir, encoder_name='hr')

    # Steps encoder
    old_st = PS_ROOT / user / 'steps_encoder.keras'
    new_st = models_dir / 'steps_encoder.keras'
    if old_st.exists():
        shutil.copy(old_st, new_st)
        enc_st = load_model(new_st)
    else:
        vals = np.vstack(df_all['st_seq'].tolist())
        segs = create_windows(vals, WINDOW_SIZE, STEP_SIZE).astype('float32')
        enc_st = build_simclr_encoder(WINDOW_SIZE)
        head = create_projection_head()
        tr, va = train_simclr(
            enc_st, head,
            segs[:int(0.8*len(segs))], segs[int(0.8*len(segs)):],
            batch_size=BATCH_SSL
        )
        enc_st.save(new_st)
        plot_ssl_losses(tr, va, results_dir, encoder_name='steps')

    enc_hr.trainable = enc_st.trainable = False
    H = enc_hr.predict(X_hr)
    S = enc_st.predict(X_st)
    X_feat = np.concatenate([H, S], axis=1)
    y = df_all['state_val'].values

    # Safe stratified split
    classes, counts = np.unique(y, return_counts=True)
    strat = y if min(counts) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=strat
    )

    # Classifier
    old_clf = PS_ROOT / user / f"{fruit}_{scenario}" / 'classifier.keras'
    new_clf = models_dir / 'classifier.keras'
    if old_clf.exists():
        shutil.copy(old_clf, new_clf)
        clf = load_model(new_clf)
    else:
        cw_vals = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
        cw = dict(enumerate(cw_vals))
        clf = build_clf(X_feat.shape[1])
        clf.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        clf.fit(
            Xtr, ytr,
            validation_split=0.1,
            epochs=CLF_EPOCHS,
            batch_size=16,
            class_weight=cw,
            verbose=2
        )
        clf.save(new_clf)
        plot_thresholds(yte, clf.predict(Xte).flatten(), results_dir, f"{user} {fruit}_{scenario} (personal_ssl)")

    preds = clf.predict(Xte).flatten()
    df, auc_mean, auc_std = bootstrap_threshold_metrics(yte, preds)
    df.to_csv(results_dir / 'bootstrap_metrics.csv', index=False)
    return df, auc_mean, auc_std


def run_global_ssl(user, fruit, scenario, base_dir):
    pipeline = base_dir / 'global_ssl'
    models_dir = pipeline / 'models_saved'
    results_dir = pipeline / 'results'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Copy encoders
    old_enc = GSSL_ROOT / '_encoders'
    for fname in ['hr_encoder.keras', 'steps_encoder.keras']:
        src = old_enc / fname; dst = models_dir / fname
        if src.exists(): shutil.copy(src, dst)
    enc_hr = load_model(models_dir / 'hr_encoder.keras')
    enc_st = load_model(models_dir / 'steps_encoder.keras')
    enc_hr.trainable = enc_st.trainable = False

    # Data prep
    u_base = Path(BASE_DATA_DIR) / user
    hr_df, st_df = load_signal_data(u_base)
    pos = load_label_data(u_base, fruit, scenario)
    neg = load_label_data(u_base, fruit, 'None')
    dfp = process_label_window(pos, hr_df, st_df, 1)
    dfn = process_label_window(neg, hr_df, st_df, 0)
    df_all = pd.concat([dfp, dfn], ignore_index=True)
    X_hr = np.expand_dims(np.stack(df_all['hr_seq'].tolist()), -1)
    X_st = np.expand_dims(np.stack(df_all['st_seq'].tolist()), -1)
    H, S = enc_hr.predict(X_hr), enc_st.predict(X_st)
    X_feat = np.concatenate([H, S], axis=1)
    y = df_all['state_val'].values

    # Safe stratified split
    classes, counts = np.unique(y, return_counts=True)
    strat = y if min(counts) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=strat
    )

    # Classifier
    old_clf = GSSL_ROOT / user / f"{fruit}_{scenario}" / 'classifier.keras'
    new_clf = models_dir / 'classifier.keras'
    if old_clf.exists(): shutil.copy(old_clf, new_clf); clf = load_model(new_clf)
    else:
        cw_vals = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
        cw = dict(enumerate(cw_vals))
        clf = Sequential([
            Dense(64, activation='relu', input_shape=(X_feat.shape[1],)),
            BatchNormalization(), Dropout(0.5),
            Dense(32, activation='relu'), Dropout(0.5),
            Dense(1, activation='sigmoid'),
        ])
        clf.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        clf.fit(
            Xtr, ytr,
            validation_split=0.1,
            epochs=GSSL_CLF_EPOCHS,
            batch_size=16,
            class_weight=cw,
            verbose=2
        )
        clf.save(new_clf)
        plot_thresholds(yte, clf.predict(Xte).flatten(), results_dir, f"{user} {fruit}_{scenario} (global_ssl)")

    preds = clf.predict(Xte).flatten()
    df, auc_mean, auc_std = bootstrap_threshold_metrics(yte, preds)
    df.to_csv(results_dir / 'bootstrap_metrics.csv', index=False)
    return df, auc_mean, auc_std

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', required=True)
    parser.add_argument('--fruit', required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--output-dir', default='results_compare')
    args = parser.parse_args()

    root = Path(args.output_dir) / args.user / f"{args.fruit}_{args.scenario}"
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    df_gs, auc_gs_m, auc_gs_s = run_global_supervised(args.fruit, args.scenario, args.user, root)
    df_ps, auc_ps_m, auc_ps_s = run_personal_ssl(args.user, args.fruit, args.scenario, root)
    df_gl, auc_gl_m, auc_gl_s = run_global_ssl(args.user, args.fruit, args.scenario, root)

    # Summary
    summary = []
    for name, (df, auc_m, auc_s) in [
        ('global_supervised', (df_gs, auc_gs_m, auc_gs_s)),
        ('global_ssl', (df_gl, auc_gl_m, auc_gl_s)),
        ('personal_ssl', (df_ps, auc_ps_m, auc_ps_s))
    ]:
        best = df.loc[df['Accuracy_Mean'].idxmax()]
        summary.append({
            'Pipeline': name,
            'Threshold': best['Threshold'],
            'Accuracy_Mean': best['Accuracy_Mean'],
            'Accuracy_STD': best['Accuracy_STD'],
            'Sensitivity_Mean': best['Sensitivity_Mean'],
            'Sensitivity_STD': best['Sensitivity_STD'],
            'Specificity_Mean': best['Specificity_Mean'],
            'Specificity_STD': best['Specificity_STD'],
            'AUC_Mean': auc_m,
            'AUC_Std': auc_s,
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(root / 'comparison_summary.csv', index=False)
    print(summary_df.to_markdown(index=False))
