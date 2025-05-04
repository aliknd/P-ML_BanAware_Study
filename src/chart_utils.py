import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_thresholds(y_train, probs_train, y_test, probs_test, out_dir, title):
    """
    Plot sensitivity, specificity, and accuracy vs threshold for both
    train and test sets. Safely handle cases where confusion_matrix
    returns a 1×1 array (only one class present).
    """
    os.makedirs(out_dir, exist_ok=True)
    thresholds = np.arange(0, 1.01, 0.01)
    train_results = []
    test_results  = []

    for t in thresholds:
        # TRAIN
        preds_tr = (probs_train >= t).astype(int)
        cm_tr = confusion_matrix(y_train, preds_tr)

        if cm_tr.shape == (1,1):
            # only one class in y_train
            val = cm_tr[0,0]
            cls = np.unique(y_train)[0]
            if cls == 0:
                tn, fp, fn, tp = val, 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, val
        else:
            tn, fp, fn, tp = cm_tr.ravel()

        sens_tr = tp / (tp + fn) if (tp + fn) else 0
        spec_tr = tn / (tn + fp) if (tn + fp) else 0
        acc_tr  = (tp + tn) / (tp + tn + fp + fn)
        train_results.append((t, sens_tr, spec_tr, acc_tr))

        # TEST
        preds_te = (probs_test >= t).astype(int)
        cm_te = confusion_matrix(y_test, preds_te)

        if cm_te.shape == (1,1):
            # only one class in y_test
            val = cm_te[0,0]
            cls = np.unique(y_test)[0]
            if cls == 0:
                tn, fp, fn, tp = val, 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, val
        else:
            tn, fp, fn, tp = cm_te.ravel()

        sens_te = tp / (tp + fn) if (tp + fn) else 0
        spec_te = tn / (tn + fp) if (tn + fp) else 0
        acc_te  = (tp + tn) / (tp + tn + fp + fn)
        test_results.append((t, sens_te, spec_te, acc_te))

    df_tr = pd.DataFrame(train_results, columns=['Threshold','Sensitivity','Specificity','Accuracy'])
    df_te = pd.DataFrame(test_results,  columns=['Threshold','Sensitivity','Specificity','Accuracy'])

    # now plot exactly as before
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

def plot_ssl_losses(train_losses, val_losses, out_dir, encoder_name="encoder"):
    """
    Plot & save train vs. validation SSL loss for one encoder.
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8,4))
    epochs = range(1, len(train_losses)+1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{encoder_name} SSL Pretraining Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{encoder_name}_ssl_pretrain_loss.png"))
    plt.close()
