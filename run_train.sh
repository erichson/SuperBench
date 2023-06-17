export CUDA_VISIBLE_DEVICES=0,1,2,3

# % --- SRCNN ---%
python train.py \
    --model SRCNN \
    --data_name nskt_16k \
    --crop_size 128 \
    --n_patches 8 \
    --method 'bicubic' \
    --upscale_factor 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200  \
    --loss_type l2 \
    --wd 1e-5 \
    --in_channels 3 \
    --out_channels 3 \
    --data_path './datasets/nskt16000_1024' &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

# % --- subpixelCNN ---%
python train.py \
    --model subpixelCNN \
    --data_name nskt_16k \
    --crop_size 128 \
    --method 'bicubic' \
    --upscale_factor 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --wd 1e-4 \
    --in_channels 3 \
    --out_channels 3 \
    --data_path './datasets/cosmo_2048' &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

# % --- EDSR ---%
python train.py \
    --model EDSR \
    --data_name era5 \
    --crop_size 128 \
    --n_patches 8 \
    --method 'bicubic' \
    --upscale_factor 16 \
    --hidden_channels 64 \
    --n_res_blocks 16 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 300 \
    --loss_type l1 \
    --wd 1e-5 \
    --in_channels 3 \
    --out_channels 3 \
    --data_path './datasets/era5' &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

# % --- WDSR ---% 
python train.py \
    --model WDSR \
    --data_name cosmo \
    --crop_size 128 \
    --iterations_per_epoch 1000 \
    --method 'uniform' \
    --upscale_factor 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 100 \
    --in_channels 2 \
    --out_channels 2 \
    --data_path './datasets/cosmo_2048' &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

# % --- SwinIR ---%
python train.py \
    --model SwinIR \
    --data_name cosmo \
    --crop_size 128 \
    --n_patches 8 \
    --method 'bicubic' \
    --upscale_factor 16 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-4 \
    --in_channels 2 \
    --out_channels 2 \
    --data_path './datasets/cosmo_2048' &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6
