#!/usr/bin/env python3
"""
personal_ssl_pipeline.py
========================
1) Train **per‑UID** HR & Steps SimCLR encoders.
2) Freeze & train a dense classifier on that UID’s labels.
3) Save encoders, classifier, threshold chart, and SSL loss chart under:
     results/personal_ssl/<UID>/<fruit>_<scenario>/
"""
import os, warnings, random
import numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from src.encoder_ssl import (
    BASE_DATA_DIR, ALLOWED_SCENARIOS,
    load_signal_data, load_label_data, process_label_window
)
from src.classify import (
    WINDOW_SIZE, STEP_SIZE, create_windows,
    apply_augmentations, create_projection_head
)
from src.pipeline_utils import (
    build_simclr_encoder, train_simclr, plot_ssl_losses
)
from src.chart_utils import plot_thresholds

EXP_ROOT   = Path("results") / "personal_ssl"
BATCH_SSL  = 128
CLF_EPOCHS = 40

def build_clf(dim):
    return Sequential([
        Dense(64, activation='relu', input_shape=(dim,), kernel_regularizer=l2(1e-2)),
        BatchNormalization(), Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(1e-2)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    for uid in ALLOWED_SCENARIOS:
        udir = Path(BASE_DATA_DIR)/uid
        if not udir.is_dir(): continue
        print(f"\n→ {uid}")
        hr_df, st_df = load_signal_data(udir)

        # SimCLR pretraining & loss curves
        for dtype in ['hr','steps']:
            # build windows
            csv = udir/f"{uid[2:]}.csv"
            df  = pd.read_csv(csv).query("data_type==@dtype")
            df['time'] = pd.to_datetime(df['time'])
            df.sort_values('time', inplace=True)
            days = np.sort(df['time'].dt.date.unique())
            tr_days = days[:int(0.75*len(days))]

            vals = StandardScaler().fit_transform(
                df[df['time'].dt.date.isin(tr_days)]['value']
                  .values.reshape(-1,1)
            )
            segs = create_windows(vals, WINDOW_SIZE, STEP_SIZE).astype('float32')

            # split train/val
            n = len(segs)
            idx = np.random.permutation(n)
            split = int(0.8*n)
            tr_segs = segs[idx[:split]]
            va_segs = segs[idx[split:]]

            enc  = build_simclr_encoder(WINDOW_SIZE)
            head = create_projection_head()
            tr_losses, va_losses = train_simclr(enc, head, tr_segs, va_segs)

            out_dir = EXP_ROOT/uid
            out_dir.mkdir(parents=True, exist_ok=True)   
            enc.save(out_dir/f"{dtype}_encoder.keras")

            # SSL loss chart
            plot_ssl_losses(tr_losses, va_losses, out_dir, encoder_name=dtype)

            print(f"✓ {dtype} encoder + SSL loss → {out_dir}")

        # load encoders for classification
        enc_hr = tf.keras.models.load_model(EXP_ROOT/uid/"hr_encoder.keras")
        enc_st = tf.keras.models.load_model(EXP_ROOT/uid/"steps_encoder.keras")

        # per‑scenario classifier + threshold chart
        for fruit,scenario in ALLOWED_SCENARIOS[uid]:
            pos = load_label_data(udir, fruit, scenario)
            neg = load_label_data(udir, fruit, 'None')
            if pos.empty or neg.empty:
                print(f"  — skip {fruit}_{scenario}"); continue

            dfp = process_label_window(pos, hr_df, st_df, 1)
            dfn = process_label_window(neg, hr_df, st_df, 0)
            df_all = pd.concat([dfp, dfn], ignore_index=True)
            if df_all.empty: continue

            # embeddings
            hr_arr = np.stack(df_all['hr_seq'].tolist())[...,None]
            st_arr = np.stack(df_all['st_seq'].tolist())[...,None]
            X = np.concatenate([enc_hr.predict(hr_arr),
                                enc_st.predict(st_arr)], axis=1)
            y = df_all['state_val'].values

            Xtr,Xte,ytr,yte = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
            cw = {i:cw[i] for i in range(len(cw))}

            clf = build_clf(X.shape[1])
            clf.compile(optimizer=Adam(1e-3),
                        loss='binary_crossentropy', metrics=['accuracy'])
            clf.fit(Xtr, ytr, epochs=CLF_EPOCHS, batch_size=16,
                    validation_split=0.1, class_weight=cw, verbose=2)

            out = EXP_ROOT/uid/f"{fruit}_{scenario}"
            out.mkdir(parents=True, exist_ok=True)    # ← ensure folder exists
            clf.save(out/"classifier.keras")

            p_tr = clf.predict(Xtr)
            p_te = clf.predict(Xte)
            plot_thresholds(ytr, p_tr, yte, p_te, out,
                            f"{uid} {fruit}_{scenario} (personal_ssl)")
            print(f"✓ classifier & threshold → {out}")