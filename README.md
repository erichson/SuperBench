# SuperBench


# Training and Evaluation Details

## Train ShallowDecoder

* python train.py --data doublegyre4 --model shallowDecoder --upscale_factor 4  --lr 0.001 --batch_size 4 --epochs 500
* python eval.py --data doublegyre4 --model_path results/model_shallowDecoder_doublegyre4_0.001_5544.npy 

* python train.py --data doublegyre8 --model shallowDecoder --upscale_factor 8 --lr 0.0005 --batch_size 4 --epochs 500
* python eval.py --data doublegyre8 --model_path results/model_shallowDecoder_doublegyre8_0.0005_5544.npy 

* python train.py --data isoflow --model shallowDecoderV2 --lr 0.001 --batch_size 4 --epochs 500
* python eval.py --data isoflow --model_path results/model_shallowDecoderV2_isoflow_0.001_5544.npy 

## Train Sub-pixel CNN 

* python train.py --data doublegyre4 --upscale_factor 4 --model subpixelCNN --lr 0.001 --batch_size 4 --epochs 300
* python eval.py --data doublegyre4 --model_path results/model_subpixelCNN_doublegyre4_0.001_5544.npy 

* python train.py --data doublegyre8 --upscale_factor 8 --model subpixelCNN --lr 0.0005 --batch_size 4 --epochs 300
* python eval.py --data doublegyre8 --model_path results/model_subpixelCNN_doublegyre8_0.0005_5544.npy 

* python train.py --data isoflow --model subpixelCNN --lr 0.001 --batch_size 4 --epochs 300
* python eval.py --data isoflow --model_path results/model_subpixelCNN_isoflow_0.001_5544.npy 
