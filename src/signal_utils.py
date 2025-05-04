import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from scipy.interpolate import interp1d  # for stretch–crop

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
BASE_DATA_DIR = '.'  # Directory containing participant subfolders
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)
WINDOW_SIZE = 30
STEP_SIZE = 15
BATCH_SIZE = 32
EPOCHS = 100

# -------------------------------------------------------------------------
# Data utilities
# -------------------------------------------------------------------------
def load_and_prepare(data_path, data_type):
    df = pd.read_csv(data_path)
    df = df[df['data_type'] == data_type].copy()
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)
    return df

def split_data_based_on_days(df, train_frac=0.75):
    unique_days = np.sort(df['time'].dt.date.unique())
    num_train = int(len(unique_days) * train_frac)
    train_days = unique_days[:num_train]
    test_days  = unique_days[num_train:]
    train_df = df[df['time'].dt.date.isin(train_days)]
    test_df  = df[df['time'].dt.date.isin(test_days)]
    return train_df, test_df

def create_windows(values, window_size, step_size):
    segments = []
    for start in range(0, len(values) - window_size + 1, step_size):
        segments.append(values[start:start + window_size])
    return np.array(segments)

# -------------------------------------------------------------------------
# Augmentations (DeepMind et al., “Training Augmentations”)
# -------------------------------------------------------------------------
def jitter(data, noise_level=0.05):
    return data + np.random.normal(0, noise_level, size=data.shape)

def flip(data):
    return data[::-1]

def stretch_crop(data, scale_range=(0.8, 1.2)):
    # flatten any (L,1) into (L,)
    flat = data.flatten()
    orig_len = len(flat)
    orig_idx = np.arange(orig_len)

    # pick random stretch factor
    scale = np.random.uniform(scale_range[0], scale_range[1])
    new_len = int(np.round(orig_len * scale))

    # interpolate onto new timeline
    new_idx = np.linspace(0, orig_len - 1, new_len)
    f = interp1d(orig_idx, flat, kind='linear')
    stretched = f(new_idx)

    # crop or pad back to orig_len
    if new_len > orig_len:
        start = np.random.randint(0, new_len - orig_len + 1)
        cropped = stretched[start:start + orig_len]
    elif new_len < orig_len:
        pad = orig_len - new_len
        left = pad // 2
        right = pad - left
        cropped = np.pad(stretched, (left, right), mode='edge')
    else:
        cropped = stretched

    # restore shape
    if data.ndim > 1:
        return cropped.reshape(orig_len, 1)
    return cropped

def apply_augmentations(windows):
    aug = np.zeros_like(windows)
    for i, w in enumerate(windows):
        sample = w.copy()
        if np.random.rand() < 0.5:
            sample = jitter(sample)
        if np.random.rand() < 0.5:
            sample = flip(sample)
        if np.random.rand() < 0.5:
            sample = stretch_crop(sample)
        aug[i] = sample
    return aug

# -------------------------------------------------------------------------
# Model setup
# -------------------------------------------------------------------------
def create_encoder(window_size, num_features):
    return Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(window_size, num_features)),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu')
    ])

def create_projection_head():
    return Sequential([
        Dense(64, activation='relu'),
        Dense(32)
    ])

def contrastive_loss(z_i, z_j, temperature=0.1):
    z = tf.concat([z_i, z_j], axis=0)
    z_norm = tf.math.l2_normalize(z, axis=1)
    sim = tf.matmul(z_norm, z_norm, transpose_b=True) / temperature
    batch = tf.shape(z_i)[0]
    labels = tf.concat([tf.range(batch), tf.range(batch)], axis=0)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, sim, from_logits=True)
    return tf.reduce_mean(loss)

def train_step(x, x_aug, enc, head, opt):
    with tf.GradientTape() as tape:
        z_i = enc(x, training=True)
        z_j = enc(x_aug, training=True)
        p_i = head(z_i, training=True)
        p_j = head(z_j, training=True)
        loss = contrastive_loss(p_i, p_j)
    grads = tape.gradient(loss, enc.trainable_variables + head.trainable_variables)
    opt.apply_gradients(zip(grads, enc.trainable_variables + head.trainable_variables))
    return loss

def train_model(train_ds, val_ds, enc, head, opt, dtype, results_dir):
    train_losses, val_losses = [], []
    for e in range(EPOCHS):
        m = tf.keras.metrics.Mean()
        for x, x_aug in train_ds:
            m.update_state(train_step(x, x_aug, enc, head, opt))
        train_losses.append(m.result().numpy())

        vm = tf.keras.metrics.Mean()
        for batch in val_ds:
            data = batch[0] if isinstance(batch, tuple) else batch
            z = enc(data, training=False)
            p = head(z, training=False)
            vm.update_state(contrastive_loss(p, p))
        val_losses.append(vm.result().numpy())

        print(f"Epoch {e+1}: Train={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}")

    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig_path = os.path.join(results_dir, f"{dtype}_loss.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved loss plot to {fig_path}")

def build_simclr_encoder(window_size):
    """Wrap the SimCLR encoder to accept single-channel input."""
    return create_encoder(window_size, 1)

def train_simclr(encoder, head, train_segments, val_segments,
                 batch_size=128, epochs=80):
    """
    Train SimCLR encoder+head; returns train_losses, val_losses.
    """
    opt = tf.keras.optimizers.Adam(1e-3)
    n_train = len(train_segments)
    train_losses, val_losses = [], []
    # prepare val dataset
    val_ds = tf.data.Dataset.from_tensor_slices(val_segments).batch(batch_size)

    for ep in range(1, epochs+1):
        # train
        idx = np.random.permutation(n_train)
        epoch_train_loss = 0.0
        for i in range(0, n_train, batch_size):
            b = idx[i:i+batch_size]
            x = train_segments[b]
            x_aug = apply_augmentations(x.copy())
            with tf.GradientTape() as tape:
                z1 = encoder(x, training=True)
                z2 = encoder(x_aug, training=True)
                p1 = head(z1, training=True)
                p2 = head(z2, training=True)
                loss = contrastive_loss(p1, p2)
            grads = tape.gradient(
                loss, encoder.trainable_weights + head.trainable_weights
            )
            opt.apply_gradients(
                zip(grads, encoder.trainable_weights + head.trainable_weights)
            )
            epoch_train_loss += loss.numpy() * len(b)
        avg_train = epoch_train_loss / n_train
        train_losses.append(avg_train)

        # val
        vm = tf.keras.metrics.Mean()
        for x_val in val_ds:
            z = encoder(x_val, training=False)
            p = head(z, training=False)
            vm.update_state(contrastive_loss(p, p))
        avg_val = vm.result().numpy()
        val_losses.append(avg_val)

        print(f"[SimCLR ep {ep}/{epochs}] train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

    return train_losses, val_losses