import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DATA_DIR = '.'            # root folder containing user subfolders
RESULTS_DIR = 'results'        # where to save outputs
# Only build models for scenarios listed in the paper per participant
ALLOWED_SCENARIOS = {
    'ID5': [('Melon', 'Crave')],
    #'ID10': [('Nectarine','Use'), ('Carrot','Use'), ('Carrot','Crave'), ('Nectarine','Crave')],
    #'ID12': [('Melon','Use'), ('Nectarine','Use'), ('Melon','Crave'), ('Nectarine','Crave')],
    #'ID13': [('Nectarine','Use'), ('Carrot','Crave')],
    #'ID18': [('Carrot','Use'), ('Carrot','Crave')],
    #'ID19': [('Melon','Use'), ('Melon','Crave')],
    #'ID25': [('Almond','Use')],
    #'ID27': [('Melon','Use'), ('Nectarine','Use'), ('Melon','Crave'), ('Nectarine','Crave')]
}
WINDOW_HOURS = 1               # hours before/after event to extract
RESAMPLE_MIN = '4min'          # resample frequency for 30 points/hour
FEATURE_POINTS = 30
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 10

# Ensure results dir exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def load_signal_data(user_dir):
    """Load and preprocess heart rate & steps from '<numeric>.csv' in user_dir."""
    uid = os.path.basename(user_dir)
    numeric = uid.split('ID')[-1]
    path = os.path.join(user_dir, f"{numeric}.csv")
    data = pd.read_csv(path)
    hr_df = data[data['data_type']=='hr'][['time','value']].copy()
    st_df = data[data['data_type']=='steps'][['time','value']].copy()
    # Ensure numeric values
    hr_df['value'] = pd.to_numeric(hr_df['value'], errors='coerce')
    st_df['value'] = pd.to_numeric(st_df['value'], errors='coerce')
    # Convert times and sort
    hr_df['time'] = pd.to_datetime(hr_df['time']).dt.tz_localize(None)
    st_df['time'] = pd.to_datetime(st_df['time']).dt.tz_localize(None)
    hr_df.sort_values('time', inplace=True)
    st_df.sort_values('time', inplace=True)
    # Forward-fill
    hr_df['value'].ffill(inplace=True)
    st_df['value'].ffill(inplace=True)
    # Set index and reindex per-minute
    hr_df.set_index('time', inplace=True)
    st_df.set_index('time', inplace=True)
    idx = pd.date_range(hr_df.index.min(), hr_df.index.max(), freq='min')
    hr_df = hr_df.reindex(idx, method='ffill')
    st_df = st_df.reindex(idx, method='ffill').fillna(0)
    return hr_df, st_df


def load_label_data(user_dir, fruit, scenario):
    """Read '<UID>_<Scenario>.csv', filter by fruit code."""
    uid = os.path.basename(user_dir)
    fname = f"{uid}_{scenario}.csv"
    path = os.path.join(user_dir, fname)
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df['hawaii_createdat_time'] = pd.to_datetime(df['hawaii_createdat_time']).dt.tz_localize(None)
    return df[df['substance_fruit_label']==fruit]


def process_label_window(df_label, hr_df, st_df, val):
    """Convert each label timestamp to fixed-length HR and steps windows."""
    scaler = StandardScaler()
    records = []
    half = pd.Timedelta(hours=WINDOW_HOURS)
    for _, row in df_label.iterrows():
        t0 = row['hawaii_createdat_time']
        hr_win = hr_df.loc[t0-half:t0+half]
        st_win = st_df.loc[t0-half:t0+half]
        if len(hr_win) < FEATURE_POINTS or len(st_win) < FEATURE_POINTS:
            continue
        # Resample and mean
        hr_means = hr_win['value'].resample(RESAMPLE_MIN).mean().iloc[:FEATURE_POINTS].values.reshape(-1,1)
        st_means = st_win['value'].resample(RESAMPLE_MIN).mean().iloc[:FEATURE_POINTS].values.reshape(-1,1)
        # Scale
        hr_scaled = scaler.fit_transform(hr_means).flatten().tolist()
        st_scaled = scaler.fit_transform(st_means).flatten().tolist()
        records.append({'hr_seq': hr_scaled, 'st_seq': st_scaled, 'state_val': val})
    return pd.DataFrame(records)


def generate_embeddings(df_feat, enc_hr, enc_st):
    """Use pretrained encoders to get embeddings."""
    hr_arr = np.stack(df_feat['hr_seq'].values)[:, :, None]
    st_arr = np.stack(df_feat['st_seq'].values)[:, :, None]
    hr_emb = enc_hr.predict(hr_arr, verbose=0)
    st_emb = enc_st.predict(st_arr, verbose=0)
    return np.concatenate([hr_emb, st_emb], axis=1)


def train_and_eval(X_train, y_train, X_test, y_test, out_dir):
    """Train model and perform threshold analysis identical to original code."""
    os.makedirs(out_dir, exist_ok=True)
    # Class weights
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: cw[i] for i in range(len(cw))}
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1,
              class_weight=class_weight_dict, callbacks=[es], verbose=1)
    # Threshold analysis on train
    probs_train = model.predict(X_train)
    thresholds = np.arange(0,1.01,0.01)
    training_results = []
    print("\nTraining Set Evaluation:")
    for thr in thresholds:
        preds = (probs_train > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, preds).ravel()
        recall = tp/(tp+fn)
        specificity = tn/(tn+fp)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        training_results.append((thr, recall, specificity, accuracy))
        print(f"Threshold: {thr:.2f}, Sensitivity: {recall:.2f}, Specificity: {specificity:.2f}, Accuracy: {accuracy:.2f}")
    df_tr = pd.DataFrame(training_results, columns=['Threshold','Sensitivity','Specificity','Accuracy'])
    # Best threshold
    best_thr_row = df_tr[(df_tr['Sensitivity']>0.9)&(df_tr['Specificity']>0.5)]
    best_threshold = best_thr_row.iloc[0]['Threshold'] if not best_thr_row.empty else 0.5
    print(f"\nBest Threshold from Training Set: {best_threshold}")
    # Test evaluation at best threshold
    probs_test = model.predict(X_test)
    preds_test_best = (probs_test > best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_test_best).ravel()
    recall_test = tp/(tp+fn)
    specificity_test = tn/(tn+fp)
    accuracy_test = (tp+tn)/(tp+tn+fp+fn)
    print(f"\nTest Set Evaluation at Best Threshold ({best_threshold}):")
    print(f"Sensitivity: {recall_test:.2f}")
    print(f"Specificity: {specificity_test:.2f}")
    print(f"Accuracy: {accuracy_test:.2f}")
    # Test evaluation for each threshold
    print("\nTest Set Evaluation for Each Threshold:")
    test_results = []
    for thr in thresholds:
        preds_t = (probs_test > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_t).ravel()
        rec = tp/(tp+fn)
        spec = tn/(tn+fp)
        acc = (tp+tn)/(tp+tn+fp+fn)
        test_results.append((thr, rec, spec, acc))
        print(f"Threshold: {thr:.2f}, Sensitivity: {rec:.2f}, Specificity: {spec:.2f}, Accuracy: {acc:.2f}")
    df_te = pd.DataFrame(test_results, columns=['Threshold','Sensitivity','Specificity','Accuracy'])
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(df_tr['Threshold'], df_tr['Sensitivity'], label='Sensitivity (Train)')
    plt.plot(df_tr['Threshold'], df_tr['Specificity'], label='Specificity (Train)')
    plt.plot(df_tr['Threshold'], df_tr['Accuracy'], label='Accuracy (Train)')
    plt.plot(df_te['Threshold'], df_te['Sensitivity'], linestyle='--', label='Sensitivity (Test)')
    plt.plot(df_te['Threshold'], df_te['Specificity'], linestyle='--', label='Specificity (Test)')
    plt.plot(df_te['Threshold'], df_te['Accuracy'], linestyle='--', label='Accuracy (Test)')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Sensitivity, Specificity, and Accuracy vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'threshold_analysis.png'))
    plt.close()
    model.save(os.path.join(out_dir, 'classifier.keras'))
    return model