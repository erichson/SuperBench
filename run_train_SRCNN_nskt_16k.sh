export CUDA_VISIBLE_DEVICES=0
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

