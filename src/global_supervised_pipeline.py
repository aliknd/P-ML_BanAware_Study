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
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.classifier_utils import (
    BASE_DATA_DIR, ALLOWED_SCENARIOS,
    load_signal_data, load_label_data, process_label_window
)
from src.chart_utils import (
    plot_thresholds, bootstrap_threshold_metrics
)

# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
EXP_ROOT   = Path("results") / "global_supervised"
WINDOW_LEN = 30
BATCH      = 256
EPOCHS     = 60
LR         = 1e-3
PATIENCE   = 8

def build_cnn():
    m = Sequential([
        Conv1D(32, 3, activation="relu", input_shape=(WINDOW_LEN, 2)),
        MaxPooling1D(2),
        Conv1D(64, 3, activation="relu"),
        GlobalAveragePooling1D(),
        Dense(128, activation="relu"),
        Dense(1,   activation="sigmoid"),
    ])
    m.compile(optimizer=Adam(LR), loss="binary_crossentropy", metrics=["accuracy"])
    return m

# --------------------------------------------------------------------------- #
# data aggregation helper
# --------------------------------------------------------------------------- #
def collect_scenario(fruit: str, scen: str):
    Xs, ys = [], []
    for uid, pairs in ALLOWED_SCENARIOS.items():
        if (fruit, scen) not in pairs:
            continue
        udir          = Path(BASE_DATA_DIR) / uid
        hr_df, st_df  = load_signal_data(udir)
        pos           = load_label_data(udir, fruit, scen)
        neg           = load_label_data(udir, fruit, "None")
        for df_lab, label in [(pos, 1), (neg, 0)]:
            if df_lab.empty:
                continue
            rows = process_label_window(df_lab, hr_df, st_df, label)
            if rows.empty or "hr_seq" not in rows:
                continue
            for h, s in zip(rows["hr_seq"], rows["st_seq"]):
                Xs.append(np.vstack([h, s]).T)
            ys += rows["state_val"].tolist()
    if not Xs:
        raise RuntimeError(f"No data for {fruit}_{scen}")
    return np.stack(Xs), np.array(ys)

# --------------------------------------------------------------------------- #
# per‑user analysis (boot‑strapped)
# --------------------------------------------------------------------------- #
def analyze_user_thresholds(model, fruit, scen, base_out: Path):
    thresholds = np.arange(0.0, 1.01, 0.01)
    for uid, pairs in ALLOWED_SCENARIOS.items():
        if (fruit, scen) not in pairs:
            continue
        udir          = Path(BASE_DATA_DIR) / uid
        hr_df, st_df  = load_signal_data(udir)
        pos           = load_label_data(udir, fruit, scen)
        neg           = load_label_data(udir, fruit, "None")
        rows = pd.concat(
            [process_label_window(pos, hr_df, st_df, 1),
             process_label_window(neg, hr_df, st_df, 0)],
            ignore_index=True,
        )
        if rows.empty or "hr_seq" not in rows:
            continue

        X_u = np.stack(
            [np.vstack([h, s]).T for h, s in zip(rows["hr_seq"], rows["st_seq"])]
        )
        y_u     = rows["state_val"].values
        probs_u = model.predict(X_u, verbose=0).flatten()

        user_dir = base_out / uid
        user_dir.mkdir(parents=True, exist_ok=True)

        # boot‑strap stats + plots
        df_u, auc_m, auc_s = bootstrap_threshold_metrics(
            y_u, probs_u, thresholds)
        df_u.to_csv(user_dir / "bootstrap_metrics.csv", index=False)

        # save passing thresholds
        mask = (df_u["Sensitivity_Mean"] > 0.9) & (df_u["Specificity_Mean"] > 0.5)
        df_u[mask].to_csv(user_dir / "passing_thresholds.csv", index=False)

        # plot (single‑set error bars)
        plt.figure(figsize=(10, 5))
        for metric, col in [
            ("Sensitivity", "tab:orange"),
            ("Specificity", "tab:green"),
            ("Accuracy",    "tab:blue"),
        ]:
            plt.errorbar(
                thresholds,
                df_u[f"{metric}_Mean"],
                yerr=df_u[f"{metric}_STD"],
                fmt="-o",
                capsize=2,
                label=metric,
            )
        plt.xlabel("Threshold"); plt.ylabel("Score"); plt.grid(True)
        plt.title(f"{uid} – {fruit}_{scen}\nBoot AUC = {auc_m:.3f} ± {auc_s:.3f}")
        plt.legend(); plt.tight_layout()
        plt.savefig(user_dir / "threshold_analysis.png"); plt.close()

# --------------------------------------------------------------------------- #
# main training routine
# --------------------------------------------------------------------------- #
def train_and_save(fruit: str, scen: str):
    try:
        X, y = collect_scenario(fruit, scen)
    except RuntimeError as e:
        print(f"Skipping {fruit}_{scen}: {e}")
        return

    idx_rest, idx_test = train_test_split(
        np.arange(len(y)),
        test_size=0.1,
        random_state=42,
        stratify=y,
    )
    idx_train, idx_val = train_test_split(
        idx_rest,
        test_size=0.111111,
        random_state=42,
        stratify=y[idx_rest],
    )

    model   = build_cnn()
    cw_vals = compute_class_weight("balanced", classes=np.unique(y), y=y)
    cw      = {i: cw_vals[i] for i in range(len(cw_vals))}
    es      = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )
    hist    = model.fit(
        X[idx_train], y[idx_train],
        validation_data=(X[idx_val], y[idx_val]),
        epochs=EPOCHS, batch_size=BATCH,
        class_weight=cw, callbacks=[es], verbose=2,
    )

    out_dir = EXP_ROOT / f"{fruit}_{scen}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / f"{fruit}_{scen}_cnn.keras")

    # training curves
    for m in ("loss", "accuracy"):
        plt.figure(figsize=(8, 4))
        plt.plot(hist.history[m], label=f"train_{m}")
        plt.plot(hist.history[f"val_{m}"], label=f"val_{m}")
        plt.xlabel("Epoch"); plt.ylabel(m.capitalize()); plt.grid(True)
        plt.title(f"{fruit}_{scen} – {m}")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"{m}.png"); plt.close()

    # -------- boot‑strapped threshold plot + ROC
    p_tr = model.predict(X[idx_train], verbose=0)
    p_te = model.predict(X[idx_test],  verbose=0)
    # -------- boot‑strapped threshold plot + ROC (test only)
    p_te = model.predict(X[idx_test], verbose=0)
    plot_thresholds(
        y[idx_test],
        p_te,
        out_dir,
        f"{fruit}_{scen} (global_supervised)",
    )

    # -------- per‑user analysis
    analyze_user_thresholds(model, fruit, scen, out_dir)
    print(f"✓ completed {fruit}_{scen}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    scenarios = {pair for pairs in ALLOWED_SCENARIOS.values() for pair in pairs}
    for fruit, scen in sorted(scenarios):
        train_and_save(fruit, scen)
