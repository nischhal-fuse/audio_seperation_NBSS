cd ~/FuseMachines/audio/speech_seperation_NBSS

python nbss_preprocessor.py \
    --wsj0_root /home/ness/FuseMachines/audio/speech_seperation_pytorch/wsj0_2mix \
    --output_root mc_wsj0_2mix_8ch_flat \
    --n_mics 8

    python nbss_preprocessor_gpu.py \
    --wsj0_root /home/ness/FuseMachines/audio/speech_seperation_pytorch/wsj0_2mix \
    --output_root mc_wsj0_2mix_8ch_fixed \
    --n_mics 8 \
    --target_sr 8000

    #npss_preprocessor_gpu

    rm -rf mc_wsj0_2mix_8ch_final
python nbss_preprocessor_gpu.py \
    --wsj0_root /home/ness/FuseMachines/audio/speech_seperation_pytorch/wsj0_2mix \
    --output_root mc_wsj0_2mix_8ch_final \
    --n_mics 8 \
    --target_sr 8000 \
    --batch_size 32 \
    --max_files 16000


    python nbss_preprocessor_gpu.py \
    --wsj0_root /home/ness/FuseMachines/audio/speech_seperation_pytorch/wsj0_2mix \
    --output_root mc_wsj0_2mix_8ch_fast \
    --n_mics 8 \
    --target_sr 8000 \
    --max_length 4.0 \
    --batch_size 1

    python nbss_preprocessor_gpu.py \
    --wsj0_root /home/ness/FuseMachines/audio/speech_seperation_pytorch/wsj0_2mix \
    --output_root mc_wsj0_2mix_8ch_robust \
    --n_mics 8 \
    --target_sr 8000 \
    --max_length 4.0