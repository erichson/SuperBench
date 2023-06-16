<img src="https://github.com/erichson/SuperBench/blob/main/SuperBench.png" width="550">

# SuperBench: A Super-Resolution Benchmark Dataset for Scientific Machine Learning

SuperBench is a benchmark dataset and evaluation framework for super-resolution (SR) tasks in scientific domains. It provides high-quality datasets and baseline models for evaluating and comparing SR methods in various scientific applications.

## Features

- Diverse datasets: SuperBench includes fluid flow, cosmology, and weather datasets with dimensions up to $2048\times2048$.
- Evaluation metrics: The framework provides comprehensive evaluation metrics for assessing SR performance, including: 
    - Pixel-level difference
    - Human-level perception
    - Domain-motivated error metrics
- Baseline models: Pre-trained baseline models are provided to facilitate comparison with state-of-the-art methods.
- Extensible framework: SuperBench is designed to be easily extendable, allowing the inclusion of new datasets and baseline models.

## Getting Started

### Installation

To use SuperBench, follow these steps:

1. Clone the repository:

```shell
git clone https://github.com/erichson/SuperBench.git
```

2. Install the required dependencies:
```shell
pip install -r requirements.txt
```

### Usage

1. Download the [SuperBench](https://portal.nersc.gov/project/dasrepo/superbench/superbench_v1.tar) datasets:
```shell
wget https://portal.nersc.gov/project/dasrepo/superbench/superbench_v1.tar
```

2. Run the baseline models on the datasets:
```shell
train.py
```

3. Evaluate the model performance (details below):

    - Pixel-level difference: 
        - relative Forbenius norm error (RFNE)
        - infinity norm (IN)
        - peak signal-to-noise ratio (PSNR)
    - Human-level perception: 
        - structural similarity index measure (SSIM)
    - Domain-motivated error metrics:
        - physics errors (e.g., continuity loss)
        - Anomaly Correlation Coefficient (ACC)
        - ...

```shell
# evaluate RFNE, IN, PSNR, SSIM
eval.py 

# evaluate physics loss
eval_phy.py 
```

4. Visualize the SR results
```shell
# for bicubic down-sampling
viz.py  

# for uniform down-sampling and noise
viz_noise.py

# for low-res simulation data
viz_lres_sim.py  
```

For detailed model configurations, please refer to the the folder ```config```.


### Contribution

We welcome contributions from the scientific machine learning community. If you would like to contribute to SuperBench, please follow the guidelines in

### Issues and Support

If you encounter any issues or have any questions, please open an issue on the <u>**GitHub repository**</u>.


### License

SuperBench is released under the <u>**MIT License**</u>.
