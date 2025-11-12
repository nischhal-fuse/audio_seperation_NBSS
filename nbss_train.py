#!/usr/bin/env python
"""
NBSS Training on 8-channel WSJ0-2mix (8 kHz) — FULLY FIXED
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
parser.add_argument('--max_epochs', type=int, default=100)
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
        mix = sf.read(mix_path)[0].T  # (8, T)
        s1 = sf.read(s1_path)[0]
        s2 = sf.read(s2_path)[0]

        T = mix.shape[1]
        if T < self.max_len:
            pad = self.max_len - T
            mix = np.pad(mix, ((0,0), (0,pad)))
            s1 = np.pad(s1, (0,pad))
            s2 = np.pad(s2, (0,pad))
        else:
            start = np.random.randint(0, T - self.max_len + 1)
            mix = mix[:, start:start+self.max_len]
            s1 = s1[start:start+self.max_len]
            s2 = s2[start:start+self.max_len]

        mix = torch.from_numpy(mix).float()      # (8, T)
        s1 = torch.from_numpy(s1).float()
        s2 = torch.from_numpy(s2).float()
        sources = torch.stack([s1, s2], dim=0)    # (2, T)

        return mix, sources

# -------------------------------
# LayerNorm wrapper for Conv1d
# -------------------------------
class ConvLayerNorm(nn.Module):
    """LayerNorm for Conv1d: handles (B, C, T) -> (B, T, C) -> LayerNorm -> (B, C, T)"""
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
    
    def forward(self, x):
        # x: (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)       # (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        return x

# -------------------------------
# Model: NBSS (8-channel Conv-TasNet)
# -------------------------------
class NBSS(pl.LightningModule):
    def __init__(self, N=256, L=20, B=256, P=3, X=8, R=3):
        super().__init__()
        self.save_hyperparameters()

        # Encoder: (8, T) → (N, T')
        self.enc = nn.Conv1d(8, N, kernel_size=L, stride=L//2)
        self.enc_norm = ConvLayerNorm(N)

        # Separator blocks
        self.sep = nn.ModuleList()
        for _ in range(R):
            blocks = []
            for x in range(X):
                in_c = N if x == 0 else B
                dilation = 2**x
                blocks += [
                    nn.Conv1d(in_c, B, 1),
                    nn.ReLU(),
                    ConvLayerNorm(B),
                    nn.Conv1d(B, B, P, dilation=dilation, padding='same'),
                    nn.ReLU(),
                    ConvLayerNorm(B),
                ]
            self.sep.append(nn.Sequential(*blocks))

        # Mask networks
        self.mask_nets = nn.ModuleList([nn.Conv1d(B, N, 1) for _ in range(2)])

        # Decoder
        self.dec = nn.ConvTranspose1d(N, 8, kernel_size=L, stride=L//2)

        self.criterion = nn.L1Loss()

    def forward(self, x):
        # x: (B, 8, T)
        B, C, T = x.shape

        # Encoder
        w = F.relu(self.enc(x))  # (B, N, T')
        w = self.enc_norm(w)     # (B, N, T')

        # Separator
        for layer in self.sep:
            w = layer(w) + w

        # Masks
        masks = torch.stack([torch.sigmoid(net(w)) for net in self.mask_nets], dim=1)
        est = w.unsqueeze(1) * masks  # (B, 2, N, T')

        # Decoder
        est = torch.stack([self.dec(est[:, i]) for i in range(2)], dim=1)  # (B, 2, 8, T)
        return est.permute(0, 2, 3, 1)  # (B, 8, T, 2)

    def training_step(self, batch, batch_idx):
        mix, sources = batch  # (B, 8, T), (B, 2, T)
        est = self(mix)       # (B, 8, T, 2)

        # Expand sources to (B, 2, 8, T)
        s1 = sources[:, 0:1].unsqueeze(2).expand(-1, -1, 8, -1)
        s2 = sources[:, 1:2].unsqueeze(2).expand(-1, -1, 8, -1)

        loss1 = self.criterion(est[..., 0], s1.squeeze(1))
        loss2 = self.criterion(est[..., 1], s2.squeeze(1))
        loss = torch.min(loss1, loss2)

        self.log('train_loss', loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mix, sources = batch
        est = self(mix)

        s1 = sources[:, 0:1].unsqueeze(2).expand(-1, -1, 8, -1)
        s2 = sources[:, 1:2].unsqueeze(2).expand(-1, -1, 8, -1)

        loss1 = F.l1_loss(est[..., 0], s1.squeeze(1), reduction='mean')
        loss2 = F.l1_loss(est[..., 1], s2.squeeze(1), reduction='mean')
        loss = torch.min(loss1, loss2)
        self.log('val_loss', loss, prog_bar=True)

        # SI-SDR on first mic
        s1_ref = sources[0, 0].cpu().numpy()
        s1_est = est[0, 0, :, 0].cpu().numpy()
        s2_ref = sources[0, 1].cpu().numpy()
        s2_est = est[0, 0, :, 1].cpu().numpy()

        sdr = max(self.si_sdr(s1_ref, s1_est), self.si_sdr(s2_ref, s2_est))
        self.log('val_sisdr', sdr, prog_bar=True)
        return loss

    def si_sdr(self, ref, est):
        ref = ref - ref.mean()
        est = est - est.mean()
        alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-8)
        s_target = alpha * ref
        e_noise = est - s_target
        return 10 * np.log10(np.sum(s_target**2) / (np.sum(e_noise**2) + 1e-8) + 1e-8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

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
    torch.set_float32_matmul_precision('medium')  # Use Tensor Cores

    train_meta = load_split(args.data_root, 'tr')
    val_meta = load_split(args.data_root, 'cv')

    train_set = NBSSDataset(train_meta)
    val_set = NBSSDataset(val_meta)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = NBSS()

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.log_dir,
            filename='nbss-{epoch:02d}-{val_sisdr:.2f}',
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