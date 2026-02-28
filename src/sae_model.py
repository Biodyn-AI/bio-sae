"""
TopK Sparse Autoencoder for biological foundation model interpretability.

Architecture follows Gao et al. (2024) "Scaling and evaluating sparse autoencoders"
with modifications for biological data (continuous activations vs. discrete tokens).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder.

    Encoder maps d_model -> n_features, then keeps only top-k activations.
    Decoder maps n_features -> d_model with unit-norm columns.
    """

    def __init__(self, d_model, n_features, k, dtype=torch.float32):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.k = k

        # Encoder: d_model -> n_features
        self.W_enc = nn.Linear(d_model, n_features, bias=True, dtype=dtype)

        # Decoder: n_features -> d_model (no bias — use separate pre-encoder bias)
        self.W_dec = nn.Linear(n_features, d_model, bias=True, dtype=dtype)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Kaiming init for encoder, unit-norm columns for decoder."""
        nn.init.kaiming_uniform_(self.W_enc.weight, nonlinearity='relu')
        nn.init.zeros_(self.W_enc.bias)

        nn.init.kaiming_uniform_(self.W_dec.weight, nonlinearity='linear')
        # Normalize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=1)

        nn.init.zeros_(self.W_dec.bias)

    def encode(self, x):
        """Encode input to sparse feature activations.

        Args:
            x: (batch, d_model) input activations

        Returns:
            h_sparse: (batch, n_features) sparse activations (only top-k nonzero)
            topk_indices: (batch, k) indices of active features
        """
        # Pre-activation
        h = self.W_enc(x)  # (batch, n_features)

        # TopK: keep only top-k activations, zero the rest
        topk_values, topk_indices = torch.topk(h, self.k, dim=-1)
        topk_values = F.relu(topk_values)  # ReLU on top-k values

        # Scatter into sparse tensor
        h_sparse = torch.zeros_like(h)
        h_sparse.scatter_(-1, topk_indices, topk_values)

        return h_sparse, topk_indices

    def decode(self, h_sparse):
        """Decode sparse features back to d_model space.

        Args:
            h_sparse: (batch, n_features) sparse activations

        Returns:
            x_hat: (batch, d_model) reconstruction
        """
        return self.W_dec(h_sparse)

    def forward(self, x):
        """Full forward pass: encode -> decode.

        Args:
            x: (batch, d_model) input activations

        Returns:
            x_hat: (batch, d_model) reconstruction
            h_sparse: (batch, n_features) sparse feature activations
            topk_indices: (batch, k) indices of active features
        """
        h_sparse, topk_indices = self.encode(x)
        x_hat = self.decode(h_sparse)
        return x_hat, h_sparse, topk_indices

    def loss(self, x, x_hat):
        """MSE reconstruction loss (no L1 — sparsity enforced by TopK)."""
        return F.mse_loss(x, x_hat)

    @torch.no_grad()
    def normalize_decoder(self):
        """Project decoder columns back to unit norm (call after optimizer step)."""
        self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=1)

    @torch.no_grad()
    def get_feature_stats(self, h_sparse):
        """Compute per-feature statistics from a batch.

        Returns dict with:
            - mean_activation: mean activation per feature (over batch)
            - activation_freq: fraction of batch where each feature fires
            - l0_norm: mean number of active features per input
        """
        active = (h_sparse > 0).float()
        return {
            'mean_activation': h_sparse.float().mean(dim=0).cpu().numpy(),
            'activation_freq': active.mean(dim=0).cpu().numpy(),
            'l0_norm': active.sum(dim=1).float().mean().item(),
        }

    def save(self, path):
        """Save model state + hyperparameters."""
        state = {
            'model_state_dict': self.state_dict(),
            'd_model': self.d_model,
            'n_features': self.n_features,
            'k': self.k,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path, device='cpu'):
        """Load model from checkpoint."""
        state = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            d_model=state['d_model'],
            n_features=state['n_features'],
            k=state['k'],
        )
        model.load_state_dict(state['model_state_dict'])
        return model.to(device)


class SAETrainer:
    """Training loop for TopK SAE with logging and checkpointing."""

    def __init__(self, sae, lr=3e-4, device='cpu'):
        self.sae = sae.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
        self.step = 0
        self.log = []

    def train_step(self, batch):
        """Single training step.

        Args:
            batch: (batch_size, d_model) tensor of activations

        Returns:
            loss value (float)
        """
        self.sae.train()
        batch = batch.to(self.device)

        x_hat, h_sparse, topk_indices = self.sae(batch)
        loss = self.sae.loss(batch, x_hat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Project decoder back to unit norm
        self.sae.normalize_decoder()

        self.step += 1
        return loss.item()

    def train_epoch(self, activations, batch_size=4096, log_every=1000,
                    checkpoint_dir=None, checkpoint_every=10000):
        """Train one epoch over activations array.

        Args:
            activations: numpy array or memmap (N, d_model)
            batch_size: training batch size
            log_every: log metrics every N steps
            checkpoint_dir: directory to save checkpoints (None = no checkpoints)
            checkpoint_every: checkpoint every N steps
        """
        n = len(activations)
        indices = np.random.permutation(n)

        epoch_losses = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            batch = torch.tensor(activations[batch_idx], dtype=torch.float32)

            loss = self.train_step(batch)
            epoch_losses.append(loss)

            if self.step % log_every == 0:
                # Compute validation metrics on the batch
                with torch.no_grad():
                    self.sae.eval()
                    batch_gpu = batch.to(self.device)
                    x_hat, h_sparse, _ = self.sae(batch_gpu)

                    mse = F.mse_loss(batch_gpu, x_hat).item()
                    # Variance explained
                    total_var = batch_gpu.var(dim=0).sum().item()
                    resid_var = (batch_gpu - x_hat).var(dim=0).sum().item()
                    var_explained = 1.0 - resid_var / max(total_var, 1e-10)

                    stats = self.sae.get_feature_stats(h_sparse)
                    dead_features = (stats['activation_freq'] == 0).sum()

                entry = {
                    'step': self.step,
                    'loss': mse,
                    'var_explained': var_explained,
                    'l0_norm': stats['l0_norm'],
                    'dead_features': int(dead_features),
                    'mean_loss_recent': float(np.mean(epoch_losses[-log_every:])),
                }
                self.log.append(entry)
                print(f"  Step {self.step:>6d} | Loss {mse:.6f} | "
                      f"VarExpl {var_explained:.4f} | "
                      f"L0 {stats['l0_norm']:.1f} | "
                      f"Dead {dead_features}/{self.sae.n_features}")

            if checkpoint_dir and self.step % checkpoint_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"sae_step_{self.step}.pt")
                self.sae.save(ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

        return float(np.mean(epoch_losses))

    def save_log(self, path):
        """Save training log to JSON."""
        with open(path, 'w') as f:
            json.dump(self.log, f, indent=2)
