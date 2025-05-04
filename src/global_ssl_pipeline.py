#!/usr/bin/env python3
"""
global_ssl_pipeline.py
======================
1) Train **global** HR & Steps SimCLR encoders.
2) Freeze & per‑UID dense classifiers + threshold charts.
3) Save under:
     results/global_ssl/_encoders/
     results/global_ssl/<UID>/<fruit>_<scenario>/
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

EXP_ROOT   = Path("results")/"global_ssl"
ENC_DIR    = EXP_ROOT/"_encoders"
CLF_EPOCHS = 40

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    print("→ training global SSL encoders …")

    # 1) aggregate and train
    losses = {}
    for dtype in ['hr','steps']:
        segs_list = []
        for uid in ALLOWED_SCENARIOS:
            csv = Path(BASE_DATA_DIR)/uid/f"{uid[2:]}.csv"
            if not csv.exists(): continue
            df = pd.read_csv(csv).query("data_type==@dtype")
            df['time'] = pd.to_datetime(df['time'])
            df.sort_values('time', inplace=True)
            days = np.sort(df['time'].dt.date.unique())
            tr_df = df[df['time'].dt.date.isin(days[:int(0.75*len(days))])]
            vals = StandardScaler().fit_transform(tr_df['value'].values.reshape(-1,1))
            segs_list.append(create_windows(vals, WINDOW_SIZE, STEP_SIZE).astype('float32'))

        G = np.concatenate(segs_list, axis=0)
        # split train/val
        n = len(G); idx = np.random.permutation(n); sp = int(0.8*n)
        tr_segs, va_segs = G[idx[:sp]], G[idx[sp:]]

        enc  = build_simclr_encoder(WINDOW_SIZE)
        head = create_projection_head()
        tr_losses, va_losses = train_simclr(enc, head, tr_segs, va_segs)

        ENC_DIR.mkdir(parents=True, exist_ok=True)
        enc.save(ENC_DIR/f"{dtype}_encoder.keras")
        losses[dtype] = (tr_losses, va_losses)
        print(f"✓ saved global {dtype} encoder")

    # SSL loss charts
    for dtype in ['hr','steps']:
        plot_ssl_losses(*losses[dtype], ENC_DIR, encoder_name=f"global_{dtype}")

    # 2) per‑UID classification
    enc_hr = tf.keras.models.load_model(ENC_DIR/"hr_encoder.keras")
    enc_st = tf.keras.models.load_model(ENC_DIR/"steps_encoder.keras")
    enc_hr.trainable = enc_st.trainable = False

    print("\n→ per‑UID classification …")
    for uid in ALLOWED_SCENARIOS:
        udir = Path(BASE_DATA_DIR)/uid
        hr_df, st_df = load_signal_data(udir)
        for fruit, scen in ALLOWED_SCENARIOS[uid]:
            pos = load_label_data(udir, fruit, scen)
            neg = load_label_data(udir, fruit, 'None')
            if pos.empty or neg.empty:
                print(f"  — skip {uid} {fruit}_{scen}"); continue

            dfp = process_label_window(pos, hr_df, st_df, 1)
            dfn = process_label_window(neg, hr_df, st_df, 0)
            df_all = pd.concat([dfp, dfn], ignore_index=True)
            if df_all.empty: continue

            a = np.stack(df_all['hr_seq'].tolist())[...,None]
            b = np.stack(df_all['st_seq'].tolist())[...,None]
            X = np.concatenate([enc_hr.predict(a), enc_st.predict(b)], axis=1)
            y = df_all['state_val'].values

            Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
            cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
            cw = {i:cw[i] for i in range(len(cw))}

            clf = Sequential([
                Dense(64,activation='relu',input_shape=(X.shape[1],),kernel_regularizer=l2(1e-2)),
                BatchNormalization(),Dropout(0.5),
                Dense(32,activation='relu',kernel_regularizer=l2(1e-2)),
                Dropout(0.5),
                Dense(1,activation='sigmoid')
            ])
            clf.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
            clf.fit(Xtr,ytr,epochs=CLF_EPOCHS,batch_size=16,
                    validation_split=0.1,class_weight=cw,verbose=2)

            out = EXP_ROOT/uid/f"{fruit}_{scen}"
            out.mkdir(parents=True, exist_ok=True)
            clf.save(out/"classifier.keras")

            p_tr = clf.predict(Xtr); p_te = clf.predict(Xte)
            plot_thresholds(ytr,p_tr,yte,p_te,out,
                            f"{uid} {fruit}_{scen} (global_ssl)")
            print(f"✓ [{uid}] {fruit}_{scen}")