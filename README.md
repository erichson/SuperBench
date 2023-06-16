<img src="images/SuperBench.png" width="550">

SuperBench is a benchmark dataset and evaluation framework for super-resolution (SR) tasks in scientific domains. It provides high-quality datasets and baseline models for evaluating and comparing SR methods in various scientific applications.

<figure style="width: 90%;" style="text-align: center;">
  <img src="images/overview.png" alt="Figure Description">
  <figcaption style="text-align: center;">An overview of super-resolution for weather data</figcaption>
</figure>

## Features

- Diverse datasets: SuperBench includes high-resolution fluid flow (left), cosmology (right), and weather datasets (middle) with dimensions up to $2048\times2048$. 

<div style="text-align: center;">
    <img src="images/nskt_snapshot_intro.png" alt="Image 1" width="150" style="display:inline-block;">
    <img src="images/climate_snapshot_intro.png" alt="Image 2" width="300" style="display:inline-block;">
    <img src="images/cosmo_snapshot_intro.png" alt="Image 3" width="150" style="display:inline-block;">
</div>

- Evaluation metrics: The framework provides comprehensive evaluation metrics for assessing SR performance, including: 
    - Pixel-level difference
    - Human-level perception
    - Domain-motivated error metrics
- Baseline models: Pre-trained baseline models are provided to facilitate comparison with state-of-the-art methods.
- Extensible framework: SuperBench is designed to be easily extendable, allowing the inclusion of new datasets and baseline models.

## Results

We have evaluated several state-of-the-art SR models on the SuperBench dataset across different degradation scenarios. Here are an example result on cosmology dataset.

### Baseline Performance

We present the baseline performance of various SR models on SuperBench. Figure 1 shows visual comparisons of the baseline model reconstructions against the ground truth high-resolution images. Table 1 provides quantitative evaluation results in terms of PSNR and SSIM metrics.

<figure style="width: 85%;">
    <figcaption style="text-align: center;">Results for weather data with bicubic down-sampling.</figcaption>
    <img src="images/weather_bicubic.png" alt="Figure Description">
</figure>

<figure style="width: 85%;">
  <img src="images/comp_weather_bicubic_snapshot.png" alt="Figure Description">
  <figcaption style="text-align: left;">An example snasphot of SR results on weather data. (a) and (b) are x8 and x16 up-sampling tasks, respectively.</figcaption>
</figure>

### Additional Results

For more detailed results and analysis, please refer to our paper.

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
# evaluate RFNE, IN, PSNR, SSIM and physics loss
eval.py 
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

We welcome contributions from the scientific machine learning community. If you would like to contribute to SuperBench, please open an issue on the <u>**GitHub repository**</u> and provide a link to your datasets or trained models with model details.

### Issues and Support

If you encounter any issues or have any questions, please open an issue on the <u>**GitHub repository**</u>.


### License

SuperBench is released under the <u>**GNU General Public License v3.0**</u>.