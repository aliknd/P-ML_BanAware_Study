#!/usr/bin/env python3
"""
global_supervised_pipeline.py
=============================
1) Train a single 1‑D‑CNN on *all* participants’ training‑split windows.
2) Produce per‑UID threshold charts.
3) Save:
     results/global_supervised/global_supervised_cnn.keras
     results/global_supervised/<UID>/<fruit>_<scenario>_threshold.png
"""
import os, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.encoder_ssl import (
    BASE_DATA_DIR, ALLOWED_SCENARIOS,
    load_signal_data, load_label_data, process_label_window
)

EXP_ROOT   = Path("results")/"global_supervised"
WINDOW_LEN = 30; BATCH=256; EPOCHS=60; LR=1e-3; PATIENCE=8

def build_cnn():
    return Sequential([
        Conv1D(32,3,activation='relu',input_shape=(WINDOW_LEN,2)),
        MaxPooling1D(2),
        Conv1D(64,3,activation='relu'),
        GlobalAveragePooling1D(),
        Dense(128,activation='relu'),
        Dense(1,activation='sigmoid')
    ])

def collect_all():
    Xs, ys = [], []
    for uid in ALLOWED_SCENARIOS:
        udir = Path(BASE_DATA_DIR)/uid
        hr_df, st_df = load_signal_data(udir)
        for fruit, scen in ALLOWED_SCENARIOS[uid]:
            for df_lab,val in [(load_label_data(udir,fruit,scen),1),
                               (load_label_data(udir,fruit,'None'),0)]:
                if df_lab.empty: continue
                rows = process_label_window(df_lab, hr_df, st_df, val)
                for h,s in zip(rows['hr_seq'], rows['st_seq']):
                    Xs.append(np.vstack([h,s]).T)
                ys += list(rows['state_val'])
    return np.stack(Xs), np.array(ys)

def plot_thr(uid,clf,hr_df,st_df,fruit,scen):
    pos = load_label_data(Path(BASE_DATA_DIR)/uid, fruit, scen)
    neg = load_label_data(Path(BASE_DATA_DIR)/uid, fruit, 'None')
    if pos.empty or neg.empty: return
    rows = pd.concat([
        process_label_window(pos, hr_df, st_df, 1),
        process_label_window(neg, hr_df, st_df, 0)
    ], ignore_index=True)
    X = np.stack([np.vstack([h,s]).T for h,s in zip(rows['hr_seq'], rows['st_seq'])])
    y = rows['state_val'].values
    probs = clf.predict(X, verbose=0)
    thr = np.arange(0,1.01,0.01)
    S,T,A = [],[],[]
    for t in thr:
        tn,fp,fn,tp = confusion_matrix(y,(probs>t).astype(int)).ravel()
        S.append(tp/(tp+fn) if tp+fn else 0)
        T.append(tn/(tn+fp) if tn+fp else 0)
        A.append((tp+tn)/len(y))
    plt.figure(figsize=(10,4))
    plt.plot(thr,S,label='Sensitivity')
    plt.plot(thr,T,label='Specificity')
    plt.plot(thr,A,label='Accuracy')
    plt.xlabel('Threshold'); plt.ylabel('Score'); plt.legend()
    plt.grid(True); plt.tight_layout()
    out = EXP_ROOT/uid
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out/f"{fruit}_{scen}_threshold.png")
    plt.close()

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    # Train global supervised CNN
    X,y = collect_all()
    idx_tr,idx_va = train_test_split(np.arange(len(y)), test_size=0.1,
                                     random_state=42, stratify=y)
    model = build_cnn()
    cw = compute_class_weight('balanced', classes=np.unique(y), y=y)
    cw = {i:cw[i] for i in range(len(cw))}
    model.compile(optimizer=Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model.fit(X[idx_tr], y[idx_tr], epochs=EPOCHS, batch_size=BATCH,
              validation_data=(X[idx_va], y[idx_va]),
              class_weight=cw, callbacks=[es], verbose=2)
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    model.save(EXP_ROOT/"global_supervised_cnn.keras")
    print("✓ saved global_supervised_cnn.keras")

    # Per‑UID threshold charts
    for uid in ALLOWED_SCENARIOS:
        udir = Path(BASE_DATA_DIR)/uid
        hr_df, st_df = load_signal_data(udir)
        for fruit,scen in ALLOWED_SCENARIOS[uid]:
            plot_thr(uid, model, hr_df, st_df, fruit, scen)
            print(f"✓ plotted {uid} {fruit}_{scen}")