import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------------------------------------------------------- #
BASE_DATA_DIR = '.'
RESULTS_DIR   = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

ALLOWED_SCENARIOS = {
    'ID5':  [('Melon', 'Crave')],
    'ID10': [('Nectarine','Use'), ('Carrot','Use'),
             ('Carrot','Crave'), ('Nectarine','Crave')],
    'ID12': [('Melon','Use'), ('Nectarine','Use'),
             ('Melon','Crave'), ('Nectarine','Crave')],
    'ID13': [('Nectarine','Use'), ('Carrot','Crave')],
    'ID18': [('Carrot','Use'), ('Carrot','Crave')],
    'ID19': [('Melon','Use'), ('Melon','Crave')],
    'ID25': [('Almond','Use')],
    'ID27': [('Melon','Use'), ('Nectarine','Use'),
             ('Melon','Crave'), ('Nectarine','Crave')]
}

WINDOW_HOURS   = 1            # before/after event
RESAMPLE_MIN   = '4min'       # 30 points/hour
FEATURE_POINTS = 30

# ----------------------------------------------------------------------------- #
def _safe_div(num, denom):
    return num / denom if denom else 0.0


# ----------------------------------------------------------------------------- #
# Data loaders
# ----------------------------------------------------------------------------- #
def load_signal_data(user_dir):
    """
    Read '<numeric>.csv' inside user_dir and return two pre‑processed
    pandas.Series indexed by 1‑minute timestamps:
        hr_df['value']  and  st_df['value']
    """
    uid = os.path.basename(user_dir)
    numeric = uid.split('ID')[-1]
    path = os.path.join(user_dir, f"{numeric}.csv")
    data = pd.read_csv(path)

    hr_df = data.loc[data['data_type'] == 'hr',    ['time', 'value']].copy()
    st_df = data.loc[data['data_type'] == 'steps', ['time', 'value']].copy()

    hr_df['value'] = pd.to_numeric(hr_df['value'], errors='coerce')
    st_df['value'] = pd.to_numeric(st_df['value'], errors='coerce')

    hr_df['time'] = pd.to_datetime(hr_df['time']).dt.tz_localize(None)
    st_df['time'] = pd.to_datetime(st_df['time']).dt.tz_localize(None)
    hr_df.sort_values('time', inplace=True)
    st_df.sort_values('time', inplace=True)

    # aggregate duplicate timestamps
    hr_df = hr_df.groupby('time', as_index=False)['value'].mean()
    st_df = st_df.groupby('time', as_index=False)['value'].mean()

    hr_df['value'].ffill(inplace=True)
    st_df['value'].ffill(inplace=True)

    hr_df.set_index('time', inplace=True)
    st_df.set_index('time', inplace=True)

    idx = pd.date_range(start=hr_df.index.min(),
                        end=hr_df.index.max(), freq='T')
    hr_df = hr_df.reindex(idx, method='ffill')
    st_df = st_df.reindex(idx, method='ffill').fillna(0)

    return hr_df, st_df


def load_label_data(user_dir, fruit, scenario):
    """Read '<UID>_<Scenario>.csv' and filter to given fruit code."""
    uid = os.path.basename(user_dir)
    fname = f"{uid}_{scenario}.csv"
    path  = os.path.join(user_dir, fname)
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df['hawaii_createdat_time'] = (
        pd.to_datetime(df['hawaii_createdat_time']).dt.tz_localize(None)
    )
    return df[df['substance_fruit_label'] == fruit]


def process_label_window(df_label, hr_df, st_df, val):
    """
    Extract a fixed‑length window of HR and step means around each label time:
    returns rows with lists hr_seq, st_seq, state_val.
    """
    scaler = StandardScaler()
    records = []
    half = pd.Timedelta(hours=WINDOW_HOURS)

    for _, row in df_label.iterrows():
        t0 = row['hawaii_createdat_time']
        hr_win = hr_df.loc[t0 - half:t0 + half]
        st_win = st_df.loc[t0 - half:t0 + half]

        if len(hr_win) < FEATURE_POINTS or len(st_win) < FEATURE_POINTS:
            continue

        hr_means = (hr_win['value']
                    .resample(RESAMPLE_MIN).mean()
                    .iloc[:FEATURE_POINTS].values.reshape(-1, 1))
        st_means = (st_win['value']
                    .resample(RESAMPLE_MIN).mean()
                    .iloc[:FEATURE_POINTS].values.reshape(-1, 1))

        hr_scaled = scaler.fit_transform(hr_means).flatten().tolist()
        st_scaled = scaler.fit_transform(st_means).flatten().tolist()
        records.append({'hr_seq': hr_scaled,
                        'st_seq': st_scaled,
                        'state_val': val})
    return pd.DataFrame(records)


def generate_embeddings(df_feat, enc_hr, enc_st):
    """Use pretrained encoders to obtain concatenated embeddings."""
    hr_arr = np.stack(df_feat['hr_seq'].values)[:, :, None]
    st_arr = np.stack(df_feat['st_seq'].values)[:, :, None]
    hr_emb = enc_hr.predict(hr_arr, verbose=0)
    st_emb = enc_st.predict(st_arr, verbose=0)
    return np.concatenate([hr_emb, st_emb], axis=1)


# ----------------------------------------------------------------------------- #
# Down‑stream classifier training helper
# ----------------------------------------------------------------------------- #
def train_and_eval(X_train, y_train, X_test, y_test, out_dir):
    """
    Train a new MLP‑classifier and produce threshold curves.
    Returns the fitted Keras model.
    """
    os.makedirs(out_dir, exist_ok=True)

    cw_arr = compute_class_weight('balanced',
                                  classes=np.unique(y_train), y=y_train)
    class_weight = {i: cw_arr[i] for i in range(len(cw_arr))}

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],),
              kernel_regularizer=l2(0.01)),
        BatchNormalization(), Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])

    es = EarlyStopping(monitor='val_loss', patience=10,
                       restore_best_weights=True)
    model.fit(X_train, y_train,
              validation_split=0.1, epochs=50, batch_size=16,
              class_weight=class_weight, callbacks=[es], verbose=1)

    # metric helper
    def _metrics(y_true, preds):
        tn, fp, fn, tp = confusion_matrix(
            y_true, preds, labels=[0, 1]).ravel()
        sens = _safe_div(tp, tp + fn)
        spec = _safe_div(tn, tn + fp)
        acc  = (tp + tn) / len(y_true)
        return sens, spec, acc

    thresholds = np.arange(0, 1.01, 0.01)

    # -------- TRAIN curve --------
    probs_tr = model.predict(X_train, verbose=0)
    tr_rows  = []
    for thr in thresholds:
        preds = (probs_tr > thr).astype(int)
        tr_rows.append((thr, *_metrics(y_train, preds)))
    df_tr = pd.DataFrame(tr_rows,
                         columns=['Threshold','Sensitivity','Specificity',
                                  'Accuracy'])

    mask = (df_tr['Sensitivity'] > 0.9) & (df_tr['Specificity'] > 0.5)
    best_thr = df_tr.loc[mask, 'Threshold'].iloc[0] if mask.any() else 0.5
    print(f"\nBest threshold (train rule) = {best_thr:.2f}")

    # -------- TEST curve --------
    probs_te = model.predict(X_test, verbose=0)
    te_rows  = []
    for thr in thresholds:
        preds = (probs_te > thr).astype(int)
        te_rows.append((thr, *_metrics(y_test, preds)))
    df_te = pd.DataFrame(te_rows,
                         columns=['Threshold','Sensitivity','Specificity',
                                  'Accuracy'])

    sens, spec, acc = _metrics(y_test, (probs_te > best_thr).astype(int))
    print(f"Test @best_thr: Sens={sens:.2f} Spec={spec:.2f} Acc={acc:.2f}")

    # -------- plot --------
    plt.figure(figsize=(12, 6))
    plt.plot(df_tr['Threshold'], df_tr['Sensitivity'], label='Sens (Train)')
    plt.plot(df_tr['Threshold'], df_tr['Specificity'], label='Spec (Train)')
    plt.plot(df_tr['Threshold'], df_tr['Accuracy'],   label='Acc  (Train)')
    plt.plot(df_te['Threshold'], df_te['Sensitivity'], '--',
             label='Sens (Test)')
    plt.plot(df_te['Threshold'], df_te['Specificity'], '--',
             label='Spec (Test)')
    plt.plot(df_te['Threshold'], df_te['Accuracy'], '--',
             label='Acc  (Test)')
    plt.xlabel('Threshold'); plt.ylabel('Score')
    plt.title('Sensitivity / Specificity / Accuracy vs Threshold')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'threshold_analysis.png'))
    plt.close()

    model.save(os.path.join(out_dir, 'classifier.keras'))
    return model
