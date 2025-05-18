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

def ensure_train_test_days(pos_df, neg_df, hr_df, st_df,
                           min_pos: int = 1, min_neg: int = 1):
    # 1) Build per-day counts
    days = sorted(pd.concat([pos_df, neg_df])['hawaii_createdat_time']
                  .dt.date.unique())
    daily = []
    for d in days:
        p = pos_df[pos_df['hawaii_createdat_time'].dt.date == d]
        n = neg_df[neg_df['hawaii_createdat_time'].dt.date == d]
        win = pd.concat([
            process_label_window(p, hr_df, st_df, 1),
            process_label_window(n, hr_df, st_df, 0)
        ])
        if win.empty or 'state_val' not in win.columns:
            pos_count = neg_count = 0
        else:
            pos_count = int(win['state_val'].sum())
            neg_count = len(win) - pos_count

        daily.append({
            'day':         d,
            'pos_count':   pos_count,
            'neg_count':   neg_count,
            'win_count':   pos_count + neg_count
        })
    df_days = pd.DataFrame(daily)

    # 2) Figure out how many windows = 25% of total
    total_windows  = df_days['win_count'].sum()
    target_windows = math.ceil(0.25 * total_windows)

    # 3) Greedy test-day selection
    days_shuffled = df_days['day'].sample(frac=1, random_state=42).tolist()
    test_days = []
    tp = tn = tw = 0

    for d in days_shuffled:
        row = df_days[df_days['day'] == d].iloc[0]
        test_days.append(d)
        tp += row['pos_count']
        tn += row['neg_count']
        tw += row['win_count']
        if tp >= min_pos and tn >= min_neg and tw >= target_windows:
            break

    # If we satisfied all three criteria, done:
    if tp >= min_pos and tn >= min_neg and tw >= target_windows:
        train_days = [d for d in days if d not in test_days]
        return np.array(train_days), np.array(test_days)

    # 4) Fallback: classic 75/25 split
    return _train_test_days_by_samples(pos_df, neg_df, hr_df, st_df)

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
    paths = {'hr': sdir / 'hr_encoder.keras', 'steps': sdir / 'steps_encoder.keras'}
    if all(p.exists() for p in paths.values()):
        hr = load_model(paths['hr']); hr.trainable = False
        st = load_model(paths['steps']); st.trainable = False
        return hr, st, sdir

    losses = {}
    for dtype in ['hr','steps']:
        bank = []
        for u,pairs in ALLOWED_SCENARIOS.items():
            if (fruit,scenario) not in pairs: continue
            tr_days_u, _ = all_splits.get(u,([],[]))
            if len(tr_days_u)==0: continue
            hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR)/u)
            df = hr_df if dtype=='hr' else st_df
            mask = np.isin(df.index.date, tr_days_u)
            vals = StandardScaler()\
                   .fit_transform(df.loc[mask,'value'].values.reshape(-1,1))
            if len(vals)<WINDOW_SIZE: continue
            bank.append(create_windows(vals,WINDOW_SIZE,STEP_SIZE).astype('float32'))
        if not bank:
            raise RuntimeError(f"No train-day segments for global {dtype} SSL!")
        segs = np.concatenate(bank,axis=0)
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


# ─── Shared CNN for Global-Supervised ─────────────────────────────────────
def ensure_global_supervised(shared_cnn_root, fruit, scenario, all_splits):
    sdir = Path(shared_cnn_root) / f"{fruit}_{scenario}"
    sdir.mkdir(parents=True, exist_ok=True)
    model_path = sdir / 'cnn_classifier.keras'
    if model_path.exists():
        return load_model(model_path), sdir

    X_list,y_list = [],[]
    for u,pairs in ALLOWED_SCENARIOS.items():
        if (fruit,scenario) not in pairs: continue
        tr_days_u,_ = all_splits.get(u,([],[]))
        if not len(tr_days_u): continue
        hr_df,st_df = load_signal_data(Path(BASE_DATA_DIR)/u)
        pos_df = load_label_data(Path(BASE_DATA_DIR)/u,fruit,scenario)
        neg_df = load_label_data(Path(BASE_DATA_DIR)/u,fruit,'None')
        df_u = collect_windows(pos_df,neg_df,hr_df,st_df,tr_days_u)
        for h,s,l in zip(df_u['hr_seq'],df_u['st_seq'],df_u['state_val']):
            X_list.append(np.vstack([h,s]).T)
            y_list.append(l)

    if not X_list: raise RuntimeError("No train windows for global-supervised!")
    X = np.stack(X_list); y = np.array(y_list)

    m = Sequential([
        Conv1D(32,3,activation="relu",input_shape=(WINDOW_LEN,2)),
        MaxPooling1D(2),
        Conv1D(64,3,activation="relu"),
        GlobalAveragePooling1D(),
        Dense(128,activation="relu"),
        Dense(1,activation="sigmoid"),
    ])
    m.compile(optimizer=Adam(GS_LR),loss="binary_crossentropy",metrics=["accuracy"])

    classes = np.unique(y)
    if len(classes)==2:
        cw=compute_class_weight('balanced',classes=classes,y=y)
        class_weight={i:cw[i] for i in range(len(cw))}
    else:
        class_weight={0:1,1:1}

    es=EarlyStopping(monitor='val_loss',patience=GS_PATIENCE,
                     restore_best_weights=True,verbose=1)
    val_split=0.1 if len(X)>=10 else 0.0
    hist=m.fit(X,y,
               validation_split=val_split,
               batch_size=GS_BATCH,
               epochs=GS_EPOCHS,
               class_weight=class_weight,
               callbacks=[es],
               verbose=2)

    plot_clf_losses(hist.history['loss'],
                    hist.history.get('val_loss',hist.history['loss']),
                    sdir,'cnn_loss')
    m.save(model_path)
    return m,sdir


# ─── Pipeline #1: Global-Supervised ───────────────────────────────────────
def run_global_supervised(fruit,scenario,uid,user_root,
                          all_splits,shared_cnn_root):
    print(f"\n>> Global-Supervised ({fruit}_{scenario})")
    out_dir=user_root/'global_supervised'
    models_d=out_dir/'models_saved'
    results_d=out_dir/'results'
    models_d.mkdir(parents=True,exist_ok=True)
    results_d.mkdir(parents=True,exist_ok=True)

    model,src_dir=ensure_global_supervised(shared_cnn_root,fruit,scenario,all_splits)
    # copy model+loss
    for f in Path(src_dir).glob('*'):
        if f.suffix=='.keras':
            shutil.copy2(f,models_d/f.name)
        elif f.suffix=='.png':
            shutil.copy2(f,results_d/f.name)

    train_info={}
    for u,(tr_days_o,_) in all_splits.items():
        if (fruit,scenario) not in ALLOWED_SCENARIOS.get(u,[]): continue
        hr_o,st_o=load_signal_data(Path(BASE_DATA_DIR)/u)
        pos_o=load_label_data(Path(BASE_DATA_DIR)/u,fruit,scenario)
        neg_o=load_label_data(Path(BASE_DATA_DIR)/u,fruit,'None')
        df_o=collect_windows(pos_o,neg_o,hr_o,st_o,tr_days_o)
        train_info[u]={"days":list(tr_days_o),"df":df_o}

    tr_days_u,te_days_u=all_splits[uid]
    hr_u,st_u=load_signal_data(Path(BASE_DATA_DIR)/uid)
    pos_u=load_label_data(Path(BASE_DATA_DIR)/uid,fruit,scenario)
    neg_u=load_label_data(Path(BASE_DATA_DIR)/uid,fruit,'None')
    df_te_u=collect_windows(pos_u,neg_u,hr_u,st_u,te_days_u)

    write_split_details(results_d,"global_supervised",train_info,(uid,list(te_days_u),df_te_u))

    te_X=np.stack([np.vstack([h,s]).T for h,s in zip(df_te_u['hr_seq'],df_te_u['st_seq'])])
    te_y=df_te_u['state_val'].values

    preds=model.predict(te_X,verbose=0).flatten()
    df_boot,auc_m,auc_s=bootstrap_threshold_metrics(te_y,preds)
    df_boot.to_csv(results_d/"bootstrap_metrics.csv",index=False)
    plot_thresholds(te_y,preds,results_d,
                    f"{uid} {fruit}_{scenario} (global_supervised)")
    return df_boot,auc_m,auc_s


# ─── Pipeline #2: Personal-SSL (unchanged, just unpacked) ────────────────
def run_personal_ssl(uid,fruit,scenario,user_root,tr_days_u,te_days_u):
    print(f">> Personal-SSL ({fruit}_{scenario})")
    out_dir=user_root/'personal_ssl'
    models_d=out_dir/'models_saved'
    results_d=out_dir/'results'
    models_d.mkdir(parents=True,exist_ok=True)
    results_d.mkdir(parents=True,exist_ok=True)

    hr_df,st_df=load_signal_data(Path(BASE_DATA_DIR)/uid)
    pos_df=load_label_data(Path(BASE_DATA_DIR)/uid,fruit,scenario)
    neg_df=load_label_data(Path(BASE_DATA_DIR)/uid,fruit,'None')

    # train/load encoders
    paths={'hr':models_d/'hr_encoder.keras','steps':models_d/'steps_encoder.keras'}
    enc_hr=_train_or_load_encoder(paths['hr'],'hr',hr_df,tr_days_u,results_d)
    enc_st=_train_or_load_encoder(paths['steps'],'steps',st_df,tr_days_u,results_d)

    # collect windows & write split_details
    df_tr=collect_windows(pos_df,neg_df,hr_df,st_df,tr_days_u)
    df_te=collect_windows(pos_df,neg_df,hr_df,st_df,te_days_u)
    write_split_details(results_d,"personal_ssl",
                        {uid:{"days":list(tr_days_u),"df":df_tr}},
                        (uid,list(te_days_u),df_te))

    # encode & train classifier
    h_tr=np.stack(df_tr['hr_seq']); s_tr=np.stack(df_tr['st_seq'])
    H_tr=enc_hr.predict(h_tr[...,None],verbose=0)
    S_tr=enc_st.predict(s_tr[...,None],verbose=0)
    Xtr=np.concatenate([H_tr,S_tr],axis=1); ytr=df_tr['state_val'].values

    h_te=np.stack(df_te['hr_seq']); s_te=np.stack(df_te['st_seq'])
    H_te=enc_hr.predict(h_te[...,None],verbose=0)
    S_te=enc_st.predict(s_te[...,None],verbose=0)
    Xte=np.concatenate([H_te,S_te],axis=1); yte=df_te['state_val'].values

    clf=Sequential([
        Dense(64,activation='relu',input_shape=(Xtr.shape[1],),
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),Dropout(0.5),
        Dense(32,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(16,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(1,activation='sigmoid')
    ])
    clf.compile(optimizer=Adam(1e-3),loss="binary_crossentropy",metrics=["accuracy"])

    es=EarlyStopping(monitor='val_loss',patience=CLF_PATIENCE,
                     restore_best_weights=True,verbose=1)
    val_split=0.1 if len(Xtr)>=10 else 0.0

    classes=np.unique(ytr)
    if len(classes)==2:
        cw=compute_class_weight('balanced',classes=classes,y=ytr)
        class_weight={i:cw[i] for i in range(len(cw))}
    else:
        class_weight={0:1,1:1}

    hist=clf.fit(Xtr,ytr,
                 validation_split=val_split,
                 epochs=CLF_EPOCHS,
                 batch_size=16,
                 class_weight=class_weight,
                 callbacks=[es],
                 verbose=2)

    plot_clf_losses(hist.history['loss'],
                    hist.history.get('val_loss',hist.history['loss']),
                    results_d,'clf_loss')
    clf.save(models_d/'classifier.keras')

    preds=clf.predict(Xte,verbose=0).flatten()
    df_boot,auc_m,auc_s=bootstrap_threshold_metrics(yte,preds)
    df_boot.to_csv(results_d/"bootstrap_metrics.csv",index=False)
    plot_thresholds(yte,preds,results_d,
                    f"{uid} {fruit}_{scenario} (personal_ssl)")

    return df_boot,auc_m,auc_s


# ─── Pipeline #3: Global-SSL (unchanged except unpack) ────────────────────
def run_global_ssl(uid,fruit,scenario,user_root,
                   shared_enc_root,all_splits):
    print(f">> Global-SSL ({fruit}_{scenario})")
    out_dir=user_root/'global_ssl'
    models_d=out_dir/'models_saved'
    results_d=out_dir/'results'
    models_d.mkdir(parents=True,exist_ok=True)
    results_d.mkdir(parents=True,exist_ok=True)

    enc_hr,enc_st,enc_src=_ensure_global_encoders(shared_enc_root,fruit,scenario,all_splits)
    for f in Path(enc_src).glob('*'):
        if f.suffix=='.keras':
            shutil.copy2(f,models_d/f.name)
        elif f.suffix=='.png':
            shutil.copy2(f,results_d/f.name)

    train_info={}
    for u,(tr_days_o,_) in all_splits.items():
        if (fruit,scenario) not in ALLOWED_SCENARIOS.get(u,[]): continue
        hr_o,st_o=load_signal_data(Path(BASE_DATA_DIR)/u)
        pos_o=load_label_data(Path(BASE_DATA_DIR)/u,fruit,scenario)
        neg_o=load_label_data(Path(BASE_DATA_DIR)/u,fruit,'None')
        df_o=collect_windows(pos_o,neg_o,hr_o,st_o,tr_days_o)
        train_info[u]={"days":list(tr_days_o),"df":df_o}

    tr_days_u,te_days_u=all_splits[uid]
    hr_u,st_u=load_signal_data(Path(BASE_DATA_DIR)/uid)
    pos_u=load_label_data(Path(BASE_DATA_DIR)/uid,fruit,scenario)
    neg_u=load_label_data(Path(BASE_DATA_DIR)/uid,fruit,'None')
    rows_tr_u=collect_windows(pos_u,neg_u,hr_u,st_u,tr_days_u)
    train_info[uid]={"days":list(tr_days_u),"df":rows_tr_u}

    df_te=collect_windows(pos_u,neg_u,hr_u,st_u,te_days_u)

    write_split_details(results_d,"global_ssl",train_info,(uid,list(te_days_u),df_te))

    hr_tr,st_tr,y_tr=[],[],[]
    for info in train_info.values():
        hr_tr.extend(info['df']['hr_seq'])
        st_tr.extend(info['df']['st_seq'])
        y_tr.extend(info['df']['state_val'])

    Xtr_hr=enc_hr.predict(np.expand_dims(np.stack(hr_tr),-1),verbose=0)
    Xtr_st=enc_st.predict(np.expand_dims(np.stack(st_tr),-1),verbose=0)
    Xtr=np.concatenate([Xtr_hr,Xtr_st],axis=1); ytr=np.array(y_tr)

    Xte_hr=enc_hr.predict(np.expand_dims(np.stack(df_te['hr_seq']),-1),verbose=0)
    Xte_st=enc_st.predict(np.expand_dims(np.stack(df_te['st_seq']),-1),verbose=0)
    Xte=np.concatenate([Xte_hr,Xte_st],axis=1); yte=df_te['state_val'].values

    clf=Sequential([
        Dense(64,activation='relu',input_shape=(Xtr.shape[1],),
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),Dropout(0.5),
        Dense(32,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(16,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(1,activation='sigmoid')
    ])
    clf.compile(optimizer=Adam(1e-3),loss="binary_crossentropy",metrics=["accuracy"])

    es=EarlyStopping(monitor='val_loss',patience=CLF_PATIENCE,
                     restore_best_weights=True,verbose=1)
    val_split=0.1 if len(Xtr)>=10 else 0.0

    classes=np.unique(ytr)
    if len(classes)==2:
        cw=compute_class_weight('balanced',classes=classes,y=ytr)
        class_weight={i:cw[i] for i in range(len(cw))}
    else:
        class_weight={0:1,1:1}

    hist=clf.fit(Xtr,ytr,
                 validation_split=val_split,
                 epochs=CLF_EPOCHS,
                 batch_size=16,
                 class_weight=class_weight,
                 callbacks=[es],
                 verbose=2)

    plot_clf_losses(hist.history['loss'],
                    hist.history.get('val_loss',hist.history['loss']),
                    results_d,'clf_loss')
    clf.save(models_d/'classifier.keras')

    preds=clf.predict(Xte,verbose=0).flatten()
    df_boot,auc_m,auc_s=bootstrap_threshold_metrics(yte,preds)
    df_boot.to_csv(results_d/"bootstrap_metrics.csv",index=False)
    plot_thresholds(yte,preds,results_d,
                    f"{uid} {fruit}_{scenario} (global_ssl)")

    return df_boot,auc_m,auc_s


# ─── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    pa=argparse.ArgumentParser()
    pa.add_argument("--user",    required=True)
    pa.add_argument("--fruit",   required=True)
    pa.add_argument("--scenario",required=True)
    pa.add_argument("--output-dir",default="results")
    args=pa.parse_args()

    top_out         =Path(args.output_dir)
    user_root       =top_out/args.user/f"{args.fruit}_{args.scenario}"
    shared_enc_root =top_out/'_global_encoders'
    shared_cnn_root =top_out/'global_cnns'
    user_root.mkdir(parents=True,exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # build all_splits once for every user:
    all_splits = {}
    for u, pairs in ALLOWED_SCENARIOS.items():
        if (args.fruit, args.scenario) not in pairs:
            continue

        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df       = load_label_data(Path(BASE_DATA_DIR) / u,
                                    args.fruit, args.scenario)
        neg_df       = load_label_data(Path(BASE_DATA_DIR) / u,
                                    args.fruit, 'None')

        tr_u, te_u = ensure_train_test_days(pos_df, neg_df,
                                            hr_df, st_df,
                                            min_pos=1, min_neg=1)
        all_splits[u] = (tr_u, te_u)

    if args.user not in all_splits:
        raise ValueError(f"No data for {args.user} in {args.fruit}/{args.scenario}")
    tr_days_u,te_days_u = all_splits[args.user]

    # 2) Tiny-data guard
    hr_df_u,st_df_u=load_signal_data(Path(BASE_DATA_DIR)/args.user)
    pos_u=load_label_data(Path(BASE_DATA_DIR)/args.user,args.fruit,args.scenario)
    neg_u=load_label_data(Path(BASE_DATA_DIR)/args.user,args.fruit,'None')
    n_tr=_count_windows(pos_u,neg_u,hr_df_u,st_df_u,tr_days_u)
    n_te=_count_windows(pos_u,neg_u,hr_df_u,st_df_u,te_days_u)
    if n_tr<2 or n_te<2:
        _write_skip_file(user_root,tr_days_u,te_days_u,n_tr,n_te)
        sys.exit(0)

    # 3) Run pipelines
    df_gs,auc_gs_m,auc_gs_s = run_global_supervised(
        args.fruit,args.scenario,args.user,user_root,
        all_splits,shared_cnn_root
    )
    df_ps,auc_ps_m,auc_ps_s = run_personal_ssl(
        args.user,args.fruit,args.scenario,user_root,
        tr_days_u,te_days_u
    )
    df_gl,auc_gl_m,auc_gl_s = run_global_ssl(
        args.user,args.fruit,args.scenario,user_root,
        shared_enc_root,all_splits
    )

    # 4) Summary (unchanged)
    rows=[]
    for name,(df,auc_m,auc_s) in [
        ("global_supervised",(df_gs,auc_gs_m,auc_gs_s)),
        ("personal_ssl",    (df_ps,auc_ps_m,auc_ps_s)),
        ("global_ssl",      (df_gl,auc_gl_m,auc_gl_s))
    ]:
        tmp=df.copy()
        tmp['Balance']=tmp[['Sensitivity_Mean','Specificity_Mean']].min(axis=1)
        best=tmp.sort_values(['Balance','Sensitivity_Mean'],ascending=[False,False]).iloc[0]
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
    df_summary=pd.DataFrame(rows)
    df_summary.to_csv(user_root/"comparison_summary.csv",index=False)
    print("\n--- Comparison Summary ---")
    print(df_summary.to_markdown(index=False))
