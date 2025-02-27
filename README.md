# Unsupervised Learning for Earth Observation Data

## Overview
This project applies **unsupervised learning techniques** to classify **sea ice and leads** using **Sentinel-2 optical data** and **Sentinel-3 altimetry data**. Two clustering methods, **K-Means** and **Gaussian Mixture Models (GMM)**, were employed to segment and analyze Earth Observation (EO) data, followed by a comparison with European Space Agency (ESA) classifications.

Unsupervised learning is a branch of machine learning where the algorithm identifies patterns and structures in data without pre-existing labels. It is particularly useful for **Earth Observation (EO)** applications, where labeled datasets can be scarce. Instead of being guided by pre-classified examples, unsupervised learning groups similar data points based on their features, making it a powerful tool for environmental and geospatial analysis.

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Unsupervised Learning Methods](#unsupervised-learning-methods)
  - [What is Unsupervised Learning?](#what-is-unsupervised-learning)
  - [K-Means Clustering](#k-means-clustering)
  - [Gaussian Mixture Model (GMM)](#gaussian-mixture-model-gmm)
- [Sentinel-2 Image Classification](#sentinel-2-image-classification)
- [Feature Comparison & Relationships](#feature-comparison-relationships)
- [Aligned & Normalized Waveforms](#aligned-normalized-waveforms)
  - [Waveform Analysis](#waveform-analysis)
  - [Echo Classification](#echo-classification)
  - [Waveform Alignment](#waveform-alignment)
- [Final Comparison with ESA Data](#final-comparison-with-esa-data)
- [References](#references)
- [Contact](#contact)

## Getting Started
### Prerequisites
Before running the code, install the necessary dependencies:
- **NumPy** – Numerical computations  
- **Pandas** – Data processing  
- **Matplotlib & Seaborn** – Visualization  
- **Scikit-Learn** – Clustering (K-Means, GMM)  
- **NetCDF4** – Handling satellite datasets  
- **SciPy** – Statistical analysis  

```bash
pip install numpy pandas matplotlib seaborn scikit-learn netCDF4 scipy rasterio
```

### Google Drive
Run the following codes to mount your Google Drive to the Jupyter Notebook.
```bash
from google.colab import drive
drive.mount('/content/drive')
```

## Unsupervised Learning Methods
### What is Unsupervised Learning?
Unsupervised learning is a type of machine learning where algorithms analyze data without predefined labels. Instead of learning from labeled examples, unsupervised learning detects patterns, structures, and relationships between data points.

For this project, **clustering** was used to categorize Earth Observation data into meaningful groups. Clustering is a technique that groups similar data points together based on shared features. The two clustering methods used in this project are **K-Means** and **Gaussian Mixture Models (GMM)**.

### K-Means Clustering
K-Means is an iterative clustering algorithm that partitions data into **k** clusters, where each cluster has a centroid representing its center.

**Step-by-Step Process:**
1. Select the number of clusters (**k**).
2. Initialize **k** cluster centroids randomly.
3. Assign each data point to the nearest centroid based on Euclidean distance.
4. Recalculate the centroids as the mean of all assigned points.
5. Repeat steps 3 and 4 until centroids no longer change significantly.

**Advantages:**
- Computationally efficient.
- Scales well with large datasets.
- Works well for well-separated, spherical clusters.

**Example K-Means Result:**
![K-Means Clustering](images/kmeans_clustering.png)

### Gaussian Mixture Model (GMM)
GMM is a probabilistic clustering technique that assumes data is generated from multiple Gaussian distributions, allowing for **soft clustering** where each point has a probability of belonging to multiple clusters.

**Step-by-Step Process:**
1. Select the number of Gaussian components (**k**).
2. Initialize parameters for each Gaussian (mean, variance, weight).
3. Compute probabilities of each point belonging to each cluster (Expectation Step).
4. Update Gaussian parameters to maximize likelihood (Maximization Step).
5. Repeat steps 3 and 4 until convergence.

**Advantages:**
- Captures complex distributions better than K-Means.
- Allows more flexible cluster shapes.
- Provides probability-based classification, reducing misclassification.

**Example GMM Result:**
![GMM Clustering](images/gmm_clustering.png)

## Sentinel-2 Image Classification
### K-Means Classification on Sentinel-2
K-Means clustering was applied to **Sentinel-2 optical data** to classify different surface types. The data was reshaped and two clusters were defined for segmentation. The classification result is displayed below:

![K-Means Sentinel-2](images/kmeans_sentinel2.png)

### GMM Classification on Sentinel-2
Gaussian Mixture Models (GMM) were applied to the same **Sentinel-2 dataset**, allowing for probabilistic classification of surface features. The result is shown below:

![GMM Sentinel-2](images/gmm_sentinel2.png)

## Feature Comparison & Relationships
Understanding feature distributions is critical for clustering evaluation. Below are scatter plots of key features used in classification.

### K-Means Feature Relationships
The scatter plots below illustrate the relationships between different features in K-Means clustering:
- **Backscatter coefficient (σ₀) vs Pulse Peakiness (PP)**
- **Backscatter coefficient (σ₀) vs Stack Standard Deviation (SSD)**
- **Pulse Peakiness (PP) vs Stack Standard Deviation (SSD)**

![K-Means Scatter](images/kmeans_scatter_plots.png)

### GMM Feature Relationships
Similar feature comparisons for the **GMM clustering results** are shown below:

![GMM Scatter](images/gmm_scatter_plots.png)



## Aligned & Normalized Waveforms
### Waveform Analysis
Waveform analysis was performed to evaluate how sea ice and leads differ in backscatter characteristics. The goal was to extract distinguishing features to enhance clustering accuracy.

- **K-Means Mean and Standard Deviation:**
  ![K-Means Waveform](images/kmeans_waveform.png)
  *Mean and standard deviation of waveforms classified using K-Means, highlighting intensity variations between clusters.*

- **GMM Mean and Standard Deviation:**
  ![GMM Waveform](images/gmm_waveform.png)
  *Mean and standard deviation of waveforms classified using GMM, showing probabilistic cluster distributions.*

### Echo Classification
The classification of echoes helps assess differences in reflection intensity between leads and sea ice, which is crucial for distinguishing surface types in satellite altimetry data.

#### Raw Echo Comparison
Raw echo distributions were analyzed to observe how the clustering models separated different surface types.

- **K-Means Echo Comparison:**
  ![K-Means Echo](images/kmeans_echo.png)
  *Comparison of raw echo intensities across all data points, classified leads, and classified sea ice using K-Means. Peaks indicate dominant signal responses.*

- **GMM Echo Comparison:**
  ![GMM Echo](images/gmm_echo.png)
  *Comparison of raw echo intensities across all data points, classified leads, and classified sea ice using GMM. Differences in peak structure highlight model variations.*

#### Normalized Echoes
To improve comparability, echoes were normalized to adjust for varying intensity scales.

- **K-Means Normalized Echoes:**
  ![K-Means Normalized Echo](images/kmeans_normalized_echo.png)
  *Normalized waveform of echoes classified using K-Means, ensuring scale consistency for better pattern recognition.*

- **GMM Normalized Echoes:**
  ![GMM Normalized Echo](images/gmm_normalized_echo.png)
  *Normalized waveform of echoes classified using GMM, reducing intensity distortions for clearer comparisons.*

### Waveform Alignment
To account for shifts in waveforms due to sea ice movement, waveforms were aligned using cross-correlation. This ensures better clustering accuracy by aligning similar waveform features.

- **K-Means Aligned Waveform:**
  ![K-Means Aligned](images/kmeans_aligned.png)
  *Waveform alignment using cross-correlation for K-Means clustering, improving classification precision.*

- **GMM Aligned Waveform:**
  ![GMM Aligned](images/gmm_aligned.png)
  *Waveform alignment using cross-correlation for GMM clustering, aligning signals to enhance feature matching.*


## Final Comparison with ESA Data
To evaluate the effectiveness of the clustering methods, the classification results from **K-Means** and **GMM** were compared against reference data provided by the **European Space Agency (ESA)**. This final comparison assesses the clustering accuracy and highlights potential improvements.

### K-Means & GMM vs. ESA Classification
The confusion matrix below illustrates the agreement between both **K-Means** and **GMM** classifications and ESA reference labels.

- **Confusion Matrix (K-Means & GMM vs. ESA):**
  ![Confusion Matrix](images/confusion_matrix.png)
  *Comparison of K-Means and GMM classification results with ESA data, highlighting correctly and incorrectly classified instances for both models.*

### Key Findings
- **Both clustering methods** demonstrated a strong correlation with ESA reference classifications, but GMM provided slightly more refined cluster boundaries due to its probabilistic approach.
- **Misclassifications** were observed in areas where sea ice and leads had similar signal characteristics, emphasizing the need for further feature engineering.
- **Waveform alignment and normalization** improved classification performance by reducing noise-related discrepancies in raw data.
- **Future work** could involve incorporating additional geospatial features to refine cluster separation and improve classification accuracy.

## References
- GEOL0069 Jupyter Book. [https://cpomucl.github.io/GEOL0069-AI4EO/intro.html]

## Contact
For questions or collaboration:
**Your Name** - your.email@example.com


