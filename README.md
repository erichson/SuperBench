# SuperBench


# Training and Evaluation Details

## Train ShallowDecoder

* python train.py --data DoubleGyre --model shallowDecoder --lr 0.001 --batch_size 4 --epochs 500
* python eval.py --data DoubleGyre --model_path results/model_shallowDecoder_DoubleGyre_0.001_5544.npy 

* python train.py --data isoflow --model shallowDecoderV2 --lr 0.001 --batch_size 4 --epochs 500
* python eval.py --data isoflow --model_path results/model_shallowDecoderV2_isoflow_0.001_5544.npy 

## Train Sub-pixel CNN 

* python train.py --data DoubleGyre --model subpixelCNN --lr 0.001 --batch_size 4 --epochs 300
* python eval.py --data DoubleGyre --model_path results/model_subpixelCNN_DoubleGyre_0.001_5544.npy 

* python train.py --data isoflow --model subpixelCNN --lr 0.001 --batch_size 4 --epochs 300
* python eval.py --data isoflow --model_path results/model_subpixelCNN_isoflow_0.001_5544.npy 