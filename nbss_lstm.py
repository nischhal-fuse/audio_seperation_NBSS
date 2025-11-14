#!/usr/bin/env python
"""
True NBSS (8-channel WSJ0-2mix 8 kHz) – Bi-LSTM separator + full-band PIT
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# Args
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='mc_wsj0_2mix_8ch_robust')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--log_dir', type=str, default='nbss_checkpoints')
parser.add_argument('--fast_dev_run', action='store_true')
args = parser.parse_args()

# -------------------------------
# Dataset
# -------------------------------
class NBSSDataset(Dataset):
    def __init__(self, meta_list, max_len=32000):
        self.items = meta_list
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        mix_path = item['mixture']
        s1_path = item['sources'][0]
        s2_path = item['sources'][1]

        # Load: (T, 8) for mix, (T,) for sources
        mix, _ = sf.read(mix_path)          # (T, 8)
        s1, _ = sf.read(s1_path)            # (T,)
        s2, _ = sf.read(s2_path)            # (T,)

        mix = mix.T                         # (8, T)
        T = mix.shape[1]

        # Fixed-length cropping / padding
        if T < self.max_len:
            pad = self.max_len - T
            mix = np.pad(mix, ((0, 0), (0, pad)))
            s1 = np.pad(s1, (0, pad))
            s2 = np.pad(s2, (0, pad))
        else:
            start = np.random.randint(0, T - self.max_len + 1)
            mix = mix[:, start:start + self.max_len]
            s1 = s1[start:start + self.max_len]
            s2 = s2[start:start + self.max_len]

        mix = torch.from_numpy(mix).float()           # (8, T)
        s1 = torch.from_numpy(s1).float()
        s2 = torch.from_numpy(s2).float()
        sources = torch.stack([s1, s2], dim=0)         # (2, T)

        return mix, sources


# -------------------------------
# LayerNorm for Conv1d output
# -------------------------------
class ConvLayerNorm(nn.Module):
    """LayerNorm for (B,C,T) → transpose → LayerNorm → transpose back"""
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        # x: (B, C, T)
        x = x.transpose(1, 2)      # (B, T, C)
        x = self.norm(x)
        x = x.transpose(1, 2)      # (B, C, T)
        return x


# -------------------------------
# True NBSS Model (Bi-LSTM separator)
# -------------------------------
class NBSS(pl.LightningModule):
    def __init__(self,
                 N=256,          # encoder output channels (bottleneck)
                 L=20,           # encoder kernel / stride
                 hidden=512,     # Bi-LSTM hidden size
                 num_layers=2):  # Bi-LSTM layers
        super().__init__()
        self.save_hyperparameters()

        # ---------------------
        # Encoder: (8, T) → (N, T')
        # ---------------------
        self.enc = nn.Conv1d(8, N, kernel_size=L, stride=L // 2, bias=False)
        self.enc_norm = ConvLayerNorm(N)

        # ---------------------
        # Separator: Bi-LSTM
        # ---------------------
        self.lstm = nn.LSTM(
            input_size=N,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        # Output: 2 masks × N channels
        self.mask_fc = nn.Linear(hidden * 2, N * 2)

        # ---------------------
        # Decoder
        # ---------------------
        self.dec = nn.ConvTranspose1d(N, 8, kernel_size=L, stride=L // 2, bias=False)

        # Loss
        self.criterion = nn.L1Loss()

    def forward(self, x):
        """
        x: (B, 8, T)
        returns: (B, 8, T, 2)
        """
        B, C, T = x.shape

        # ----- Encoder -----
        e = F.relu(self.enc(x))          # (B, N, T')
        e = self.enc_norm(e)             # (B, N, T')

        # ----- Bi-LSTM -----
        e_t = e.transpose(1, 2)          # (B, T', N)
        lstm_out, _ = self.lstm(e_t)     # (B, T', 2*hidden)

        # ----- Mask generation -----
        mask_logits = self.mask_fc(lstm_out)        # (B, T', 2*N)
        masks = torch.sigmoid(mask_logits)
        masks = masks.view(B, -1, 2, self.hparams.N) # (B, T', 2, N)
        masks = masks.permute(0, 2, 3, 1)            # (B, 2, N, T')

        # ----- Apply masks -----
        est_enc = e.unsqueeze(1) * masks             # (B, 2, N, T')

        # ----- Decoder -----
        est_list = []
        for i in range(2):
            dec_out = self.dec(est_enc[:, i])       # (B, 8, T)
            est_list.append(dec_out)
        est = torch.stack(est_list, dim=1)          # (B, 2, 8, T)

        # Output shape: (B, 8, T, 2)
        return est.permute(0, 2, 3, 1)

    # ---------------------
    # PIT Loss (full-band)
    # ---------------------
    def pit_loss(self, est, ref):
        """
        est: (B, 8, T, 2)
        ref: (B, 2, T) → expand to (B, 2, 8, T)
        """
        B, _, T, _ = est.shape
        ref_exp = ref.unsqueeze(2).expand(-1, -1, 8, -1)  # (B, 2, 8, T)

        # Two possible assignments
        loss1 = self.criterion(est[..., 0], ref_exp[:, 0])
        loss2 = self.criterion(est[..., 1], ref_exp[:, 1])
        loss_perm1 = loss1 + loss2

        loss1_swap = self.criterion(est[..., 0], ref_exp[:, 1])
        loss2_swap = self.criterion(est[..., 1], ref_exp[:, 0])
        loss_perm2 = loss1_swap + loss2_swap

        return torch.min(loss_perm1, loss_perm2)

    def training_step(self, batch, batch_idx):
        mix, sources = batch
        est = self(mix)                    # (B, 8, T, 2)
        loss = self.pit_loss(est, sources)
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mix, sources = batch
        est = self(mix)
        loss = self.pit_loss(est, sources)
        self.log('val_loss', loss, prog_bar=True)

        # SI-SDR on first microphone (for monitoring)
        s1_ref = sources[0, 0].cpu().numpy()
        s2_ref = sources[0, 1].cpu().numpy()
        s1_est = est[0, :, :, 0].cpu().numpy()[0]  # first mic
        s2_est = est[0, :, :, 1].cpu().numpy()[0]

        sdr1 = self.si_sdr(s1_ref, s1_est)
        sdr2 = self.si_sdr(s2_ref, s2_est)
        sdr = max(sdr1, sdr2)
        self.log('val_sisdr', sdr, prog_bar=True)

    @staticmethod
    def si_sdr(ref, est):
        ref = ref - ref.mean()
        est = est - est.mean()
        alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-8)
        s_target = alpha * ref
        e_noise = est - s_target
        return 10 * np.log10(
            np.sum(s_target**2) / (np.sum(e_noise**2) + 1e-8) + 1e-8
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


# -------------------------------
# Load Metadata
# -------------------------------
def load_split(data_root, split):
    path = Path(data_root) / f"{split}_metadata.json"
    with open(path) as f:
        return json.load(f)


# -------------------------------
# Main
# -------------------------------
def main():
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    train_meta = load_split(args.data_root, 'tr')
    val_meta = load_split(args.data_root, 'cv')

    train_set = NBSSDataset(train_meta)
    val_set = NBSSDataset(val_meta)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = NBSS()

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.log_dir,
            filename='nbss-bilstm-{epoch:02d}-{val_sisdr:.2f}',
            monitor='val_sisdr',
            mode='max',
            save_top_k=3
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        precision=16,
        gradient_clip_val=args.grad_clip,
        log_every_n_steps=10,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        default_root_dir=args.log_dir
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"\nTraining complete! Best model saved in: {args.log_dir}")


if __name__ == '__main__':
    main()