export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model SRCNN \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model SRCNN \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model SRCNN \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model SRCNN \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model SRCNN \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model SRCNN \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model SRCNN \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model SRCNN \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model SRCNN \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model SRCNN \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model SRCNN \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model SRCNN \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model SRCNN \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model SRCNN \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model SRCNN \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model SRCNN \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --upscale_factor 8 \
    --model SRCNN \
    --data_name cosmo_lres_sim \
    --data_path './datasets/cosmo_lres_sim_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model subpixelCNN \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model subpixelCNN \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model subpixelCNN \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model subpixelCNN \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model subpixelCNN \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model subpixelCNN \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model subpixelCNN \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model subpixelCNN \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model subpixelCNN \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model subpixelCNN \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model subpixelCNN \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model subpixelCNN \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model subpixelCNN \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model subpixelCNN \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model subpixelCNN \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model subpixelCNN \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --upscale_factor 8 \
    --model subpixelCNN \
    --data_name cosmo_lres_sim \
    --data_path './datasets/cosmo_lres_sim_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 0.0001 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model EDSR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model EDSR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model EDSR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model EDSR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model EDSR \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model EDSR \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model EDSR \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model EDSR \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model EDSR \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model EDSR \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model EDSR \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model EDSR \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model EDSR \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model EDSR \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model EDSR \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model EDSR \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --upscale_factor 8 \
    --model EDSR \
    --data_name cosmo_lres_sim \
    --data_path './datasets/cosmo_lres_sim_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model WDSR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model WDSR \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model WDSR \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model WDSR \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model WDSR \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --upscale_factor 8 \
    --model WDSR \
    --data_name cosmo_lres_sim \
    --data_path './datasets/cosmo_lres_sim_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 300 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 18 \
    --hidden_channels 32  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model SwinIR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model SwinIR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model SwinIR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model SwinIR \
    --data_name nskt_16k \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model SwinIR \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model SwinIR \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model SwinIR \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model SwinIR \
    --data_name nskt_32k \
    --data_path './datasets/nskt32000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model SwinIR \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model SwinIR \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model SwinIR \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model SwinIR \
    --data_name cosmo \
    --data_path './datasets/cosmo_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 8 \
    --model SwinIR \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.05' \
    --upscale_factor 8 \
    --model SwinIR \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'uniform_noise' \
    --noise_ratio '0.1' \
    --upscale_factor 8 \
    --model SwinIR \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --method 'bicubic' \
    --upscale_factor 16 \
    --model SwinIR \
    --data_name era5 \
    --data_path './datasets/nskt16000_1024' \
    --in_channels 3 \
    --out_channles 3 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

python train.py \
    --upscale_factor 8 \
    --model SwinIR \
    --data_name cosmo_lres_sim \
    --data_path './datasets/cosmo_lres_sim_2048' \
    --in_channels 2 \
    --out_channles 2 \
    --crop_size 128 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 200 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --n_res_block 0 \
    --hidden_channels 0  &

pid_file6=$!
echo "PID2 for train.py: $pid_file6" >> pid.log
wait $pid_file6

