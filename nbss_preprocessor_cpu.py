#!/usr/bin/env python
"""
Generate 8-channel WSJ0-2mix (32 kHz → 8 kHz) - SIMPLIFIED VERSION
"""
import os
import json
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
import librosa

# ----------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--wsj0_root', type=str,
                    default='/home/ness/FuseMachines/audio/speech_seperation_pytorch/wsj0_2mix')
parser.add_argument('--output_root', type=str, default='mc_wsj0_2mix_8ch_fixed')
parser.add_argument('--n_mics', type=int, default=8)
parser.add_argument('--target_sr', type=int, default=8000)
parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_files', type=int, default=None)
args = parser.parse_args()

np.random.seed(args.seed)

# ----------------------------------------------------------------------
# SIMPLIFIED RIR SIMULATION
# ----------------------------------------------------------------------
def create_simple_rirs(n_mics=8, n_sources=2, fs=8000, max_delay=0.1):
    """
    Create simple but realistic RIRs that will produce 8-channel output
    """
    max_delay_samples = int(max_delay * fs)
    rirs = np.zeros((n_mics, max_delay_samples, n_sources))
    
    # Create different delays for each microphone and source
    for mic_idx in range(n_mics):
        for src_idx in range(n_sources):
            # Different base delay for each source
            base_delay = 0.01 + 0.02 * src_idx  # 10ms + source-specific offset
            
            # Microphone-specific variation (circular array)
            mic_angle = 2 * np.pi * mic_idx / n_mics
            mic_delay_variation = 0.005 * np.cos(mic_angle)  # ±5ms variation
            
            # Source position variation
            src_angle = np.pi * src_idx  # Sources on opposite sides
            src_delay_variation = 0.01 * np.cos(src_angle + mic_angle)
            
            total_delay = base_delay + mic_delay_variation + src_delay_variation
            delay_samples = int(total_delay * fs)
            
            if delay_samples < max_delay_samples:
                # Create realistic RIR with direct path and some reflections
                rirs[mic_idx, delay_samples, src_idx] = 1.0  # Direct path
                
                # Add some early reflections
                for reflection in range(1, 4):
                    reflection_delay = delay_samples + reflection * int(0.005 * fs)
                    if reflection_delay < max_delay_samples:
                        rirs[mic_idx, reflection_delay, src_idx] = 0.3 / reflection
    
    return rirs

def convolve_multichannel(sources, rirs):
    """
    sources: (2, T) - two source signals
    rirs: (8, T_rir, 2) - RIRs for 8 mics and 2 sources
    returns: (8, T) - 8-channel mixture
    """
    n_mics, T_rir, n_sources = rirs.shape
    T_sig = sources.shape[1]
    
    # Initialize output
    mixture = np.zeros((n_mics, T_sig + T_rir - 1))
    
    # Convolve each source with each microphone's RIR
    for mic_idx in range(n_mics):
        for src_idx in range(n_sources):
            source_sig = sources[src_idx]
            rir = rirs[mic_idx, :, src_idx]
            conv_result = np.convolve(source_sig, rir)
            mixture[mic_idx, :len(conv_result)] += conv_result
    
    # Trim to original signal length
    mixture = mixture[:, :T_sig]
    
    return mixture

def downsample(sig, orig_sr, target_sr):
    """High-quality resample with librosa"""
    if orig_sr == target_sr:
        return sig
    if len(sig.shape) > 1:
        sig = np.mean(sig, axis=1)
    return librosa.resample(sig, orig_sr=orig_sr, target_sr=target_sr)

def safe_load_audio(file_path, target_sr):
    """Safely load and preprocess audio"""
    try:
        sig, orig_sr = sf.read(file_path)
        if len(sig.shape) > 1:
            sig = np.mean(sig, axis=1)
        if not np.isfinite(sig).all():
            return None, None
        sig = downsample(sig, orig_sr, target_sr)
        return sig, orig_sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def normalize_signals(mixture, s1, s2):
    """Normalize to prevent clipping"""
    mx = np.max(np.abs(mixture))
    if mx > 1e-6:
        scale = 0.95 / mx
        mixture = mixture * scale
        s1 = s1 * scale
        s2 = s2 * scale
    return mixture, s1, s2

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    s1_dir = Path(args.wsj0_root) / 's1'
    s2_dir = Path(args.wsj0_root) / 's2'
    
    if not (s1_dir.exists() and s2_dir.exists()):
        raise FileNotFoundError(f"s1/ or s2/ missing in {args.wsj0_root}")

    # Find all pairs
    s1_files = sorted([p for p in s1_dir.iterdir() if p.suffix == '.wav'])
    uid_map = {p.stem: {'s1': p, 's2': None} for p in s1_files}
    
    for p in s2_dir.iterdir():
        if p.suffix == '.wav' and p.stem in uid_map:
            uid_map[p.stem]['s2'] = p

    pairs = [(uid, info['s1'], info['s2']) for uid, info in uid_map.items() if info['s2']]
    print(f"Complete pairs: {len(pairs)}")

    if args.max_files:
        pairs = pairs[:args.max_files]
        print(f"Limited to {args.max_files} files")

    # Split data
    idx = np.arange(len(pairs))
    np.random.shuffle(idx)
    n = len(idx)
    n_tr = int(n * args.split_ratio[0])
    n_cv = int(n * args.split_ratio[1])
    
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

        print(f"\nProcessing {split} split with {len(ids)} files...")
        
        for i in tqdm(ids, desc=f"Generating {split}"):
            uid, p1, p2 = pairs[i]
            
            try:
                # Load and downsample sources
                s1, orig_sr1 = safe_load_audio(p1, args.target_sr)
                s2, orig_sr2 = safe_load_audio(p2, args.target_sr)
                
                if s1 is None or s2 is None:
                    error_count += 1
                    continue

                # Trim to same length
                T = min(len(s1), len(s2))
                if T == 0:
                    error_count += 1
                    continue
                    
                s1, s2 = s1[:T], s2[:T]
                sources = np.stack([s1, s2])  # (2, T)

                # Generate RIRs - THIS IS THE KEY PART
                rirs = create_simple_rirs(n_mics=args.n_mics, n_sources=2, fs=args.target_sr)
                
                # Verify RIR shape
                assert rirs.shape == (args.n_mics, int(0.1 * args.target_sr), 2), \
                    f"RIR shape incorrect: {rirs.shape}"
                
                # Create multichannel mixture
                mixture = convolve_multichannel(sources, rirs)  # (8, T)
                
                # Verify output shape
                assert mixture.shape[0] == args.n_mics, \
                    f"Mixture has {mixture.shape[0]} channels, expected {args.n_mics}"
                
                # Normalize
                mixture, s1, s2 = normalize_signals(mixture, s1, s2)

                # Save files
                sf.write(out_dir / f"{uid}_mix.wav", mixture.T, args.target_sr, subtype='PCM_16')
                sf.write(out_dir / f"{uid}_s1.wav", s1, args.target_sr, subtype='PCM_16')
                sf.write(out_dir / f"{uid}_s2.wav", s2, args.target_sr, subtype='PCM_16')

                meta.append({
                    "uid": uid,
                    "mixture": str(out_dir / f"{uid}_mix.wav"),
                    "sources": [str(out_dir / f"{uid}_s1.wav"), str(out_dir / f"{uid}_s2.wav")],
                    "original_sr": int(orig_sr1),
                    "target_sr": args.target_sr,
                    "duration_sec": float(T) / args.target_sr
                })

            except Exception as e:
                print(f"\nError processing {uid}: {e}")
                error_count += 1
                continue

        # Save metadata
        (Path(args.output_root) / f"{split}_metadata.json").write_text(
            json.dumps(meta, indent=2))
        print(f"{split}: {len(meta)} samples (errors: {error_count})")

    print(f"\n✅ Successfully generated 8-channel dataset at: {args.output_root}")

if __name__ == '__main__':
    main()