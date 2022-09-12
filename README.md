# SuperBench


# Training and Evaluation Details

## Train ShallowDecoder

* python train.py --data doublegyre4 --model shallowDecoder --upscale_factor 4  --lr 0.001 --batch_size 4 --epochs 500
* python eval.py --data doublegyre4 --model_path results/model_shallowDecoder_doublegyre4_0.001_5544.npy 

* python train.py --data doublegyre8 --model shallowDecoder --upscale_factor 8 --lr 0.0005 --batch_size 4 --epochs 500
* python eval.py --data doublegyre8 --model_path results/model_shallowDecoder_doublegyre8_0.0005_5544.npy 


* python train.py --data isoflow4 --model shallowDecoder --upscale_factor 4 --lr 0.001 --batch_size 4 --epochs 500
* python eval.py --data isoflow4 --model_path results/model_shallowDecoder_isoflow4_0.001_5544.npy 

* python train.py --data isoflow8 --model shallowDecoder --upscale_factor 8 --lr 0.0001 --batch_size 4 --epochs 500
* python eval.py --data isoflow8 --model_path results/model_shallowDecoder_isoflow8_0.0001_5544.npy 



## Train Sub-pixel CNN 

* python train.py --data doublegyre4 --model subpixelCNN --upscale_factor 4 --lr 0.001 --batch_size 4 --epochs 300
* python eval.py --data doublegyre4 --model_path results/model_subpixelCNN_doublegyre4_0.001_5544.npy 

* python train.py --data doublegyre8 --model subpixelCNN --upscale_factor 8 --lr 0.0005 --batch_size 4 --epochs 300
* python eval.py --data doublegyre8 --model_path results/model_subpixelCNN_doublegyre8_0.0005_5544.npy 


* python train.py --data isoflow4 --model subpixelCNN --upscale_factor 4 --lr 0.0001 --batch_size 4 --epochs 500 --width 2
* python eval.py --data isoflow4 --model_path results/model_subpixelCNN_isoflow4_0.0005_5544.npy 

* python train.py --data isoflow8 --model subpixelCNN --upscale_factor 8 --lr 0.0001 --batch_size 4 --epochs 500 --width 2
* python eval.py --data isoflow8 --model_path results/model_subpixelCNN_isoflow8_0.0001_5544.npy 


* python train.py --data rbc4 --model subpixelCNN --upscale_factor 4 --lr 0.0001 --batch_size 4 --epochs 300
* python eval.py --data rbc4 --model_path results/model_subpixelCNN_rbc4_0.0001_5544.npy 

* python train.py --data rbc8 --model subpixelCNN --upscale_factor 8 --lr 0.0001 --batch_size 4 --epochs 300
* python eval.py --data rbc8 --model_path results/model_subpixelCNN_rbc8_0.0001_5544.npy 


* python train.py --data sst4 --model subpixelCNN --upscale_factor 4 --lr 0.0004 --batch_size 8 --epochs 300
* python eval.py --data sst4 --model_path results/model_subpixelCNN_sst4_0.0005_5544.npy 

* python train.py --data sst8 --model subpixelCNN --upscale_factor 8 --lr 0.0004 --batch_size 8 --epochs 300
* python eval.py --data sst8 --model_path results/model_subpixelCNN_sst8_0.0005_5544.npy 



export CUDA_VISIBLE_DEVICES=4; python train.py --data sst4 --model subpixelCNN --upscale_factor 4 --lr 0.0006 --batch_size 64 --epochs 300

