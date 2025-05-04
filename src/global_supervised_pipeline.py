#!/usr/bin/env python3
"""
global_supervised_pipeline.py
=============================

For each (fruit,scenario) across users:
 1) Collect windows pooled across users.
 2) Train a 1‑D‑CNN on 80% train, 10% val, 10% test.
 3) Save model, train/val plots, scenario-level threshold plot, test metrics,
    and per-user threshold + metrics under results/global_supervised/<fruit>_<scenario>/<UID>/
"""
import warnings, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.classifier_utils import (
    BASE_DATA_DIR,
    ALLOWED_SCENARIOS,
    load_signal_data,
    load_label_data,
    process_label_window
)

# Experiment settings
EXP_ROOT   = Path("results")/"global_supervised"
WINDOW_LEN = 30
BATCH      = 256
EPOCHS     = 60
LR         = 1e-3
PATIENCE   = 8

def build_cnn():
    m = Sequential([
        Conv1D(32,3,activation='relu',input_shape=(WINDOW_LEN,2)),
        MaxPooling1D(2),
        Conv1D(64,3,activation='relu'),
        GlobalAveragePooling1D(),
        Dense(128,activation='relu'),
        Dense(1,activation='sigmoid')
    ])
    m.compile(optimizer=Adam(LR),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return m

def collect_scenario(fruit: str, scen: str):
    """Pool X,y for one scenario across all users."""
    Xs, ys = [], []
    for uid,pairs in ALLOWED_SCENARIOS.items():
        if (fruit,scen) not in pairs: continue
        udir = Path(BASE_DATA_DIR)/uid
        hr_df,st_df = load_signal_data(udir)
        pos = load_label_data(udir,fruit,scen)
        neg = load_label_data(udir,fruit,'None')
        for df_lab,label in [(pos,1),(neg,0)]:
            if df_lab.empty: continue
            rows = process_label_window(df_lab,hr_df,st_df,label)
            if rows.empty or 'hr_seq' not in rows: continue
            for h,s in zip(rows['hr_seq'], rows['st_seq']):
                Xs.append(np.vstack([h,s]).T)
            ys += rows['state_val'].tolist()
    if not Xs:
        raise RuntimeError(f"No data for {fruit}_{scen}")
    return np.stack(Xs), np.array(ys)

def analyze_user_thresholds(model, fruit, scen, base_out: Path):
    """Per-user threshold plots + default metrics."""
    thresholds = np.arange(0,1.01,0.01)
    for uid,pairs in ALLOWED_SCENARIOS.items():
        if (fruit,scen) not in pairs: continue
        udir = Path(BASE_DATA_DIR)/uid
        hr_df,st_df = load_signal_data(udir)
        pos = load_label_data(udir,fruit,scen)
        neg = load_label_data(udir,fruit,'None')
        rows = pd.concat([
            process_label_window(pos,hr_df,st_df,1),
            process_label_window(neg,hr_df,st_df,0)
        ],ignore_index=True)
        if rows.empty or 'hr_seq' not in rows: continue

        Xu = np.stack([np.vstack([h,s]).T for h,s in zip(rows['hr_seq'],rows['st_seq'])])
        yu = rows['state_val'].values
        probs = model.predict(Xu,verbose=0).flatten()

        # metrics at 0.5
        preds0 = (probs>=0.5).astype(int)
        cm0 = confusion_matrix(yu,preds0)
        report0 = classification_report(yu,preds0)

        user_dir = base_out/uid
        user_dir.mkdir(parents=True,exist_ok=True)
        with open(user_dir/'metrics.txt','w') as f:
            f.write("Confusion Matrix:\n"+str(cm0)+"\n\n")
            f.write("Classification Report:\n"+report0)

        # threshold curves
        sens, spec, acc = [],[],[]
        for t in thresholds:
            p = (probs>=t).astype(int)
            cm = confusion_matrix(yu,p)
            if cm.shape==(1,1):
                # only one class
                val = cm[0,0]
                if np.unique(yu)[0]==0:
                    tn,fp,fn,tp = val,0,0,0
                else:
                    tn,fp,fn,tp = 0,0,0,val
            else:
                tn,fp,fn,tp = cm.ravel()
            sens.append(tp/(tp+fn) if tp+fn else 0)
            spec.append(tn/(tn+fp) if tn+fp else 0)
            acc.append((tp+tn)/len(yu))

        plt.figure(figsize=(12,6))
        plt.plot(thresholds,sens,  label='Sensitivity')
        plt.plot(thresholds,spec,  label='Specificity')
        plt.plot(thresholds,acc,  label='Accuracy')
        plt.xlabel('Threshold'); plt.ylabel('Score')
        plt.title(f'{uid} – {fruit}_{scen}')
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(user_dir/'threshold_analysis.png'); plt.close()

def train_and_save(fruit:str, scen:str):
    """Train scenario model, save everything, run per-user analysis."""
    try:
        X,y = collect_scenario(fruit,scen)
    except RuntimeError as e:
        print(f"Skipping {fruit}_{scen}: {e}")
        return

    # 80/10/10 split
    idx_rest,idx_test = train_test_split(np.arange(len(y)), test_size=0.1,
                                         random_state=42, stratify=y)
    idx_train,idx_val = train_test_split(idx_rest, test_size=0.1111111,
                                         random_state=42, stratify=y[idx_rest])

    model = build_cnn()
    cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight = {i:cw[i] for i in range(len(cw))}
    es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    hist = model.fit(
        X[idx_train], y[idx_train],
        validation_data=(X[idx_val], y[idx_val]),
        epochs=EPOCHS, batch_size=BATCH,
        class_weight=class_weight, callbacks=[es], verbose=2
    )

    out_dir = EXP_ROOT/f"{fruit}_{scen}"
    out_dir.mkdir(parents=True,exist_ok=True)

    # save model
    model.save(out_dir/f"{fruit}_{scen}_cnn.keras")

    # train/val plots
    for metric in ('loss','accuracy'):
        plt.figure(figsize=(8,4))
        plt.plot(hist.history[metric],      label=f'train_{metric}')
        plt.plot(hist.history[f'val_{metric}'], label=f'val_{metric}')
        plt.xlabel('Epoch'); plt.ylabel(metric.capitalize())
        plt.title(f'{fruit}_{scen} {metric}')
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(out_dir/f"{metric}.png"); plt.close()

    # scenario-level threshold analysis
    probs_tr = model.predict(X[idx_train],verbose=0).flatten()
    probs_te = model.predict(X[idx_test], verbose=0).flatten()
    thr = np.arange(0,1.01,0.01)
    tr_res, te_res = [],[]
    for t in thr:
        # train
        p_tr = (probs_tr>=t).astype(int)
        cm = confusion_matrix(y[idx_train], p_tr)
        if cm.shape==(1,1):
            val=cm[0,0]; tn,fp,fn,tp = (val,0,0,0) if np.unique(y[idx_train])[0]==0 else (0,0,0,val)
        else:
            tn,fp,fn,tp = cm.ravel()
        tr_res.append((t, tp/(tp+fn) if tp+fn else 0,
                          tn/(tn+fp) if tn+fp else 0,
                          (tp+tn)/len(idx_train)))
        # test
        p_te = (probs_te>=t).astype(int)
        cm = confusion_matrix(y[idx_test], p_te)
        if cm.shape==(1,1):
            val=cm[0,0]; tn,fp,fn,tp = (val,0,0,0) if np.unique(y[idx_test])[0]==0 else (0,0,0,val)
        else:
            tn,fp,fn,tp = cm.ravel()
        te_res.append((t, tp/(tp+fn) if tp+fn else 0,
                          tn/(tn+fp) if tn+fp else 0,
                          (tp+tn)/len(idx_test)))
    df_tr = pd.DataFrame(tr_res, columns=['Threshold','Sensitivity','Specificity','Accuracy'])
    df_te = pd.DataFrame(te_res, columns=['Threshold','Sensitivity','Specificity','Accuracy'])

    plt.figure(figsize=(12,6))
    plt.plot(df_tr['Threshold'], df_tr['Sensitivity'], label='Sensitivity (Train)')
    plt.plot(df_tr['Threshold'], df_tr['Specificity'], label='Specificity (Train)')
    plt.plot(df_tr['Threshold'], df_tr['Accuracy'],    label='Accuracy    (Train)')
    plt.plot(df_te['Threshold'], df_te['Sensitivity'], '--', label='Sensitivity (Test)')
    plt.plot(df_te['Threshold'], df_te['Specificity'], '--', label='Specificity (Test)')
    plt.plot(df_te['Threshold'], df_te['Accuracy'],    '--', label='Accuracy    (Test)')
    plt.xlabel('Threshold'); plt.ylabel('Score')
    plt.title(f'{fruit}_{scen} Threshold Analysis')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir/'threshold_analysis.png'); plt.close()

    # per-user analysis
    analyze_user_thresholds(model, fruit, scen, out_dir)
    print(f"✓ completed {fruit}_{scen}")

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    scenarios = set(pair for pairs in ALLOWED_SCENARIOS.values() for pair in pairs)
    for fruit,scen in scenarios:
        train_and_save(fruit,scen)
