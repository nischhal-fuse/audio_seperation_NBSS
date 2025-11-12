#!/usr/bin/env python
"""
8-CHANNEL WSJ0-2MIX — ROBUST VERSION WITH FALLBACKS
"""
import os
import json
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch
import torch.nn.functional as F
from pathlib import Path
import librosa
import time
import traceback

# Try to import gpuRIR but have fallback
try:
    import gpuRIR
    GPU_RIR_AVAILABLE = True
    print("✅ gpuRIR available")
except ImportError:
    GPU_RIR_AVAILABLE = False
    print("⚠️  gpuRIR not available, using fallback RIR generation")

# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--wsj0_root', type=str, default='/home/ness/FuseMachines/audio/speech_seperation_pytorch/wsj0_2mix')
parser.add_argument('--output_root', type=str, default='mc_wsj0_2mix_8ch_robust')
parser.add_argument('--n_mics', type=int, default=8)
parser.add_argument('--array_radius', type=float, default=0.08)
parser.add_argument('--room_sz_min', type=float, nargs=3, default=[3.0, 3.0, 2.0])
parser.add_argument('--room_sz_max', type=float, nargs=3, default=[10.0, 8.0, 3.5])
parser.add_argument('--rt60_min', type=float, default=0.2)
parser.add_argument('--rt60_max', type=float, default=0.7)
parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--target_sr', type=int, default=8000)
parser.add_argument('--max_files', type=int, default=None)
parser.add_argument('--max_length', type=float, default=4.0)
parser.add_argument('--use_fallback', action='store_true', help='Use fallback RIR generation instead of gpuRIR')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------------------------------------------------
def downsample(sig, orig_sr, target_sr):
    if orig_sr == target_sr: return sig
    if len(sig.shape) > 1: sig = np.mean(sig, axis=1)
    return librosa.resample(sig, orig_sr=orig_sr, target_sr=target_sr)

def make_circular_array(center, radius, n_mics):
    angles = np.linspace(0, 2*np.pi, n_mics, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    z = np.full(n_mics, center[2])
    return np.column_stack([x, y, z])

def simulate_rirs_fallback(n_mics=8, n_sources=2, fs=8000, duration=0.2):
    """Fast fallback RIR generation when gpuRIR fails"""
    T_rir = int(duration * fs)
    rirs = np.zeros((n_mics, T_rir, n_sources))
    
    for mic_idx in range(n_mics):
        angle = 2 * np.pi * mic_idx / n_mics
        for src_idx in range(n_sources):
            # Spatial delays based on microphone position
            base_delay = 0.01 + 0.005 * src_idx
            spatial_delay = 0.003 * np.cos(angle + src_idx * np.pi)
            total_delay = base_delay + spatial_delay
            
            delay_samples = min(int(total_delay * fs), T_rir - 1)
            
            # Direct path
            rirs[mic_idx, delay_samples, src_idx] = 1.0
            
            # Early reflections
            for refl in range(1, 4):
                refl_delay = delay_samples + refl * int(0.004 * fs)
                if refl_delay < T_rir:
                    rirs[mic_idx, refl_delay, src_idx] = 0.4 / refl
    
    return rirs

def simulate_rirs_gpu_safe(room_sz, src_pos, mic_pos, fs, rt60, timeout=5):
    """Safe gpuRIR wrapper with timeout"""
    if args.use_fallback or not GPU_RIR_AVAILABLE:
        return simulate_rirs_fallback(len(mic_pos), len(src_pos), fs)
    
    try:
        import math
        L, W, H = room_sz
        V = L * W * H
        S = 2 * (L*W + L*H + W*H)
        alpha = math.exp(-0.1611 * V / (rt60 * S)) if rt60 > 0 else 0.0
        beta = [alpha] * 6
        
        # Use simpler settings for stability
        rirs = gpuRIR.simulateRIR(
            room_sz=room_sz.astype(np.float32).tolist(),
            beta=beta,
            pos_src=src_pos.astype(np.float32),
            pos_rcv=mic_pos.astype(np.float32),
            nb_img=[6, 6, 3],  # Reduced for stability
            Tmax=0.2,          # Shorter for stability
            fs=fs
        )
        return rirs.transpose(1, 2, 0)  # (n_src, n_mic, T) → (n_mic, T, n_src)
    
    except Exception as e:
        print(f"⚠️  gpuRIR failed: {e}, using fallback")
        return simulate_rirs_fallback(len(mic_pos), len(src_pos), fs)

def convolve_cpu_fast(sources, rirs):
    """CPU convolution that's faster and more reliable"""
    n_src, T_sig = sources.shape
    n_mic, T_rir, _ = rirs.shape
    
    mixture = np.zeros((n_mic, T_sig))
    
    for m in range(n_mic):
        for s in range(n_src):
            # Use numpy convolution (often faster than torch for CPU)
            conv_result = np.convolve(sources[s], rirs[m, :, s], mode='full')[:T_sig]
            mixture[m] += conv_result
    
    return mixture

def safe_load_audio(file_path, target_sr, max_length=None):
    try:
        sig, sr = sf.read(file_path)
        if len(sig.shape) > 1: sig = np.mean(sig, axis=1)
        sig = downsample(sig, sr, target_sr)
        if max_length is not None:
            max_s = int(max_length * target_sr)
            if len(sig) > max_s:
                start = np.random.randint(0, len(sig) - max_s)
                sig = sig[start:start + max_s]
        return sig, sr
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None, None

def normalize_signals(mix, s1, s2):
    mx = np.max(np.abs(mix))
    if mx > 1e-6:
        scale = 0.95 / mx
        mix *= scale; s1 *= scale; s2 *= scale
    return mix, s1, s2

# ----------------------------------------------------------------------
def main():
    s1_dir = Path(args.wsj0_root) / 's1'
    s2_dir = Path(args.wsj0_root) / 's2'
    
    if not s1_dir.exists() or not s2_dir.exists():
        raise FileNotFoundError("WSJ0 directories not found")
    
    # Find pairs more efficiently
    s1_files = list(s1_dir.glob("*.wav"))
    pairs = []
    for p1 in s1_files:
        p2 = s2_dir / p1.name
        if p2.exists():
            pairs.append((p1.stem, p1, p2))
    
    print(f"Found {len(pairs)} pairs")
    
    if args.max_files: 
        pairs = pairs[:args.max_files]
        print(f"Limited to {args.max_files} files")

    # Split data
    idx = np.random.permutation(len(pairs))
    n_tr = int(len(idx) * args.split_ratio[0])
    n_cv = int(len(idx) * args.split_ratio[1])
    splits = {
        'tr': idx[:n_tr], 
        'cv': idx[n_tr:n_tr+n_cv], 
        'tt': idx[n_tr+n_cv:]
    }

    os.makedirs(args.output_root, exist_ok=True)
    
    for split, ids in splits.items():
        out_dir = Path(args.output_root) / split
        out_dir.mkdir(exist_ok=True)
        meta = []
        error_count = 0
        
        print(f"\n📁 Processing {split} split ({len(ids)} files)")
        
        pbar = tqdm(ids, desc=f"Generating {split}", position=0, leave=True)
        for i in pbar:
            uid, p1, p2 = pairs[i]
            
            try:
                # Load audio with error handling
                s1, sr1 = safe_load_audio(p1, args.target_sr, args.max_length)
                s2, sr2 = safe_load_audio(p2, args.target_sr, args.max_length)
                
                if s1 is None or s2 is None:
                    error_count += 1
                    pbar.set_postfix({"errors": error_count, "current": uid})
                    continue

                # Trim to same length
                T = min(len(s1), len(s2))
                if T == 0:
                    error_count += 1
                    continue
                    
                s1, s2 = s1[:T], s2[:T]
                sources = np.stack([s1, s2])

                # Generate room configuration
                room_sz = np.random.uniform(args.room_sz_min, args.room_sz_max)
                rt60 = np.random.uniform(args.rt60_min, args.rt60_max)
                
                # Source positions
                src1 = np.random.uniform([1, 1, 1], room_sz - [1, 1, 1])
                src2 = src1.copy()
                attempts = 0
                while np.linalg.norm(src1 - src2) < 1.0 and attempts < 10:
                    src2 = np.random.uniform([1, 1, 1], room_sz - [1, 1, 1])
                    attempts += 1
                
                src_pos = np.stack([src1, src2])
                
                # Microphone array
                center = room_sz / 2
                center[2] = 1.2
                mic_pos = make_circular_array(center, args.array_radius, args.n_mics)

                # Generate RIRs safely
                rirs = simulate_rirs_gpu_safe(room_sz, src_pos, mic_pos, args.target_sr, rt60)
                
                # Verify RIR shape
                if rirs.shape[0] != args.n_mics or rirs.shape[2] != 2:
                    print(f"❌ Invalid RIR shape {rirs.shape} for {uid}")
                    error_count += 1
                    continue

                # Convolve on CPU for stability
                mixture = convolve_cpu_fast(sources, rirs)
                
                if mixture.shape[0] != args.n_mics:
                    print(f"❌ Invalid mixture shape {mixture.shape} for {uid}")
                    error_count += 1
                    continue

                # Normalize and save
                mixture, s1, s2 = normalize_signals(mixture, s1, s2)

                sf.write(out_dir / f"{uid}_mix.wav", mixture.T, args.target_sr, subtype='PCM_16')
                sf.write(out_dir / f"{uid}_s1.wav", s1, args.target_sr, subtype='PCM_16')
                sf.write(out_dir / f"{uid}_s2.wav", s2, args.target_sr, subtype='PCM_16')

                meta.append({
                    "uid": uid, 
                    "mixture": str(out_dir / f"{uid}_mix.wav"), 
                    "sources": [str(out_dir / f"{uid}_s1.wav"), str(out_dir / f"{uid}_s2.wav")],
                    "room": room_sz.tolist(),
                    "rt60": float(rt60),
                    "duration_sec": float(T) / args.target_sr
                })
                
                pbar.set_postfix({
                    "success": len(meta), 
                    "errors": error_count, 
                    "current": uid
                })
                
                # Small delay to prevent resource contention
                if len(meta) % 100 == 0:
                    time.sleep(0.1)

            except Exception as e:
                error_count += 1
                print(f"❌ Error processing {uid}: {e}")
                # Don't print full traceback for common errors
                if "CUDA" not in str(e) and "memory" not in str(e).lower():
                    traceback.print_exc()
                continue

        # Save metadata
        (Path(args.output_root) / f"{split}_metadata.json").write_text(
            json.dumps(meta, indent=2))
        print(f"✅ {split}: {len(meta)} samples, {error_count} errors")

    print(f"\n🎉 Dataset generation complete at: {args.output_root}")

if __name__ == '__main__':
    main()