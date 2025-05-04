# charts_utils.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_thresholds(y_train, prob_train, y_test, prob_test, out_dir, title):
    """
    Compute sensitivity, specificity, accuracy over thresholds 0-1
    and save the threshold_analysis.png under out_dir.
    """
    thresholds = np.arange(0, 1.01, 0.01)
    train_rows, test_rows = [], []
    for t in thresholds:
        # train
        preds_tr = (prob_train > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, preds_tr).ravel()
        sens_t = tp / (tp + fn) if (tp + fn) else 0
        spec_t = tn / (tn + fp) if (tn + fp) else 0
        acc_t  = (tp + tn) / len(y_train)
        train_rows.append((t, sens_t, spec_t, acc_t))
        # test
        preds_te = (prob_test > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_te).ravel()
        sens_e = tp / (tp + fn) if (tp + fn) else 0
        spec_e = tn / (tn + fp) if (tn + fp) else 0
        acc_e  = (tp + tn) / len(y_test)
        test_rows.append((t, sens_e, spec_e, acc_e))

    df_tr = pd.DataFrame(train_rows, columns=['Threshold','Sensitivity','Specificity','Accuracy'])
    df_te = pd.DataFrame(test_rows, columns=['Threshold','Sensitivity','Specificity','Accuracy'])

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.plot(df_tr['Threshold'], df_tr['Sensitivity'], label='Sensitivity (Train)')
    plt.plot(df_tr['Threshold'], df_tr['Specificity'], label='Specificity (Train)')
    plt.plot(df_tr['Threshold'], df_tr['Accuracy'],    label='Accuracy    (Train)')
    plt.plot(df_te['Threshold'], df_te['Sensitivity'], '--', label='Sensitivity (Test)')
    plt.plot(df_te['Threshold'], df_te['Specificity'], '--', label='Specificity (Test)')
    plt.plot(df_te['Threshold'], df_te['Accuracy'],    '--', label='Accuracy    (Test)')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'threshold_analysis.png'))
    plt.close()
