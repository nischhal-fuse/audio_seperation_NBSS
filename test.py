import soundfile as sf
mix, sr = sf.read("mc_wsj0_2mix_8ch_flat/tr/mix_18770_mix.wav")
print("Shape:", mix.shape)  # → (42695, 8) or (32000, 8)