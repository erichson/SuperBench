export CUDA_VISIBLE_DEVICES=0,1,2,3

python eval.py \
    --model SwinIR \
    --data_name cosmo \
    --crop_size 128 \
    --n_patches 8 \
    --method 'bicubic' \
    --upscale_factor 8 \
    --lr 0.0001 \
    --batch_size 4 \
    --in_channels 2 \
    --out_channels 2 \
    --data_path './datasets/cosmo_2048' &