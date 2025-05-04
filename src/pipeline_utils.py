# pipeline_utils.py
import os
import numpy as np
import tensorflow as tf
from src.classify import create_encoder, apply_augmentations, contrastive_loss
import matplotlib.pyplot as plt

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
