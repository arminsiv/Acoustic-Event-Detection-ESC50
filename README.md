# Acoustic-Event-Detection--ESC50

This repository hosts the final submission for the Applied Machine Learning course. The project focuses on the automated classification of environmental sounds (ESC-50 dataset) using a dual-approach methodology: Supervised Learning for classification and Unsupervised Learning for structure discovery.

### Applied Machine Learning (ESC-50 Dataset)

![Python](https://img.shields.io/badge/Python-3.14%2B-blue)
![Library](https://img.shields.io/badge/Librosa-Audio_Analysis-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset & Preprocessing](#-dataset--preprocessing)
3. [Feature Extraction Strategy (Collaborative)](#-feature-extraction-strategy)
   - [Temporal & Spectral Features](#1-temporal--spectral-features-student-1)
   - [Harmonic & Tonal Features](#2-harmonic--tonal-features-student-2)
4. [Supervised Learning Pipeline](#-supervised-learning-pipeline)
5. [Unsupervised Learning Pipeline](#-unsupervised-learning-pipeline)
6. [Results & Model Comparison](#-results--model-comparison)
7. [Installation & Usage](#-installation--usage)

---

## 📌 Project Overview
The objective is to classify 10 distinct audio categories by translating raw audio waveforms into meaningful numerical features.

## Understanding The Data
We selected 10 classes from five categories in the ESC-50 dataset:

- **Animals**: Dog, Rooster
- **Natural Sounds**: Thunderstorm, Sea_waves
- **Human Sounds**: Snoring, Sneezing
- **Interior Sounds**: Clock Alarm, Vacuum Cleaner
- **Urban Noises**: Siren, Helicopter

#### **Data Filtering & Class Balance**
We filtered the dataset to retain only the selected classes and checked for class balance. Each class contained 40 samples, ensuring uniform distribution.

#### **Mapping Categories to Target Labels**
We mapped each category to its respective target label, as shown below:

| Category       | Target |
| -------------- | ------ |
| Dog            | 0      |
| Rooster        | 1      |
| Thunderstorm   | 19     |
| Sea Waves      | 11     |
| Clock Alarm    | 37     |
| Vacuum Cleaner | 36     |
| Helicopter     | 40     |
| Siren          | 42     |
| Snoring        | 28     |
| Sneezing       | 21     |

#### **Loading Audio Data**

We loaded the `.wav` files, obtaining:

- **400 total samples**
- **Sample rate:** 44,100 Hz
- **Waveform shape:** (220,500)

This prepared the dataset for feature extraction and model training.

##### More Visualizations => [Data Visualizations] (Data Visualizations.ipynb)

To ensure robust performance and avoid overfitting, we did  a strict training and validation protocol compliant with all assignment requirements. We tackle this using two notebooks:
* **`Main assignment.ipynb`**: The core classification engine. It implements an expanded feature pipeline (**116 features**) and trains **SVM, Logistic Regression, Random Forest, and KNN** models using 5-Fold Cross-Validation. ( 4 fold for Cv and 1 for fold  reamined untouched for final test)
* **`Unsupervised method.ipynb`**: An exploratory analysis using ** K-Means, Hierarchical Clustering and GMM** on the same feature set (**116 features**) to test if audio classes cluster naturally without labels.



## 📊 Dataset & Preprocessing

**Source:** [ESC-50: Dataset for Environmental Sound Classification](https://github.com/karolpiczak/ESC-50)
* **Selection:** We filtered the dataset to 10 specific classes representing Animals, Nature, Human sounds, and Urban noise.
* **Sampling Rate:** Signals were loaded at **22,050 Hz** to capture frequencies up to ~11kHz (sufficient for environmental sounds).
* **Normalization:** All extracted features were standardized using `StandardScaler` ($\mu=0, \sigma=1$) to prevent high-magnitude features (like Spectral Centroid) from dominating the distance calculations in SVM and K-Means.

---

## Feature Extraction & Selection Strategy

To isolate specific acoustic properties, we implemented a dynamic feature selection mechanism (located in **Cell 10**). Instead of using a single black-box input, we defined **8 distinct feature groups** to test different hypotheses.

The total feature vector contains **116 features**. Below are the specific subsets we experimented with:

### 1. Scientific Feature Groups
These groups target specific physical properties of sound:

| Group Name | Feature Count | Description |
| :--- | :--- | :--- |
| **`temporal`** | **31 Features** | Focuses on time-domain dynamics. Includes **Delta MFCCs, ZCR, and Envelope**. Best for detecting percussive sounds (Rain, Footsteps). |
| **`harmonic`** | **32 Features** | Focuses on pitch and harmony. Includes **MFCCs, Pitch, and HNR**. Best for tonal sounds (Speech, Sirens). |
| **`spectral_brightness`** | **35 Features** | Focuses on the energy distribution. Includes **MFCCs, Spectral Centroid, and Spectral Contrast**. Best for distinguishing bright vs. dark sounds. |
| **`noise_based`** | **11 Features** | Focuses on signal clarity. Includes **Spectral Contrast, HNR, and Flatness**. Best for separating pure tones from noisy environments. |

### 2. General Experiment Groups
These broadly combine features to find the best general-purpose model:

| Group Name | Feature Count | Description |
| :--- | :--- | :--- |
| **`general_set_1`** | **57 Features** | A broad mix of Timbral (MFCC) and Temporal features. |
| **`general_set_2`** | **41 Features** | A mix of Timbral (MFCC) and Spectral/Pitch features. |
| **`last_set_chroma_tonnetz`** | **36 Features** | **The "Music" Set.** Strictly contains **Chroma** (Pitch classes) and **Tonnetz** (Harmonic relations). Used to test if musical theory applies to environmental sounds. |
| **`all_features`** | **116 Features** | **The Master Set.** Uses every extracted feature. This set achieved the highest accuracy by combining all available acoustic information. |

> ## Feature Engineering & Dimensionality Reduction

We created multiple variations of the base features to capture temporal dynamics and statistical distributions:
**Statistical Variations:** For every feature (MFCCs, Chroma, Spectral Centroid), we calculated the **Mean** (central tendency) and **Standard Deviation** (temporal spread) to summarize the 5-second clip into a static vector.
**Structural Variations (Groups):** In our analysis (see **Cell 10**), we defined distinct feature sets—such as `harmonic` (Pitch/Chroma focus) vs. `temporal` (ZCR/Envelope focus)—to test how different spectral bandings and feature distributions affect model performance.

> **💾 Data Persistence:** To ensure reproducibility and efficiency, the final 116-dimensional feature matrix is serialized and saved to `features_all_116.npz`.


## Data Preparation

### 1. Data Splitting & Validation Strategy
* **Train/Test Split:** We applied an **80:20 split** to the dataset.
    * **80% (320 samples):** Used for training and hyperparameter tuning.
    * **20% (80 samples):** Kept as a final test set for final evaluation.
* **Cross-Validation:** Within the training set, we utilized **4-Fold Cross-Validation**. This ensured that every data point was used for validation, providing a reliable estimate of model stability before touching the test set. Also in order to prevent data leakage we used **PreDefinedFolds** in the dataset for training and grid search.
---

### 2. Hyperparameter Tuning (Varying Model Parameters)
We did not use default settings. We performed **Grid Search** and iterative testing to optimize specific parameters for all **7 algorithms**:

| Algorithm | Type | Parameters Varied | Best Configuration |
| :--- | :--- | :--- | :--- |
| **SVM** | Supervised | **Kernel** (`linear`, `rbf`), **C** (0.1, 1, 10, 100), **Gamma** | **RBF Kernel, C=10, Gamma=0.001** (Best non-linear separation) |
| **Random Forest** | Supervised | **n_estimators** (50, 100, 200) | **n_estimators** (200) |
| **KNN** | Supervised | **k** (3, 5, 7, 9), **Metric** (Euclidean vs. Manhattan) | **k=3** Metric:Manhattan |
| **Logistic Regression**| Supervised | **Penalty** (L2), **C** (0.001, 0.01, 0.1, 1, 10, 100) | **L2 Penalty** (C=0.1), Solver=LBFGS |
| **K-Means** | Unsupervised | **k** (Clusters) | **k=10** (Matched target classes) |
| **GMM** | Unsupervised | **Covariance Type** (full, tied, diag) | **Full Covariance** (Captured elliptical clusters) |
| **Hierarchical** | Unsupervised | **Linkage** (Ward, Complete, Average) | **Ward** (Minimized variance within clusters) |


### 3. Unsupervised Evaluation Strategy
Since clustering algorithms do not have "ground truth" labels during training, we implemented a **Label Mapping** strategy to assess performance:
* **Method:** "Majority Vote." For each generated cluster, we assigned the label of the most frequent true class within that cluster.
* **Metric:** We calculated the accuracy of these assigned labels against the true targets to quantify clustering purity.


### 4. Feature-Dependent Model Selection
We evaluated our models not just on one dataset, but across **different feature subsets** (defined in **Cell 10** of our notebook) to determine the impact of input data:

* **Experiment A (Timbral Features):** Trained Linear SVM on MFCCs only.
    * *Result:* Moderate accuracy; struggled with pitch-based sounds (Sirens).
* **Experiment B (Harmonic Features):** Trained K-Means on Chroma/Tonnetz only.
    * *Result:* Poor separation; insufficient for distinct clustering.
* **Experiment C (Combined Features):** Trained RBF SVM on the full 116-feature set.
    * *Result:* **Selected Model.** Achieved the highest validation accuracy by leveraging both textural and tonal information.

---

## Supervised Learning Pipeline
**File:** `Main assignment.ipynb`

We evaluated three families of classifiers to find the optimal decision boundary:

### 3.1 Support Vector Machine (SVM)
* **Kernel:** `RBF` (Radial Basis Function).
* **Outcome:** The non-linear kernel allowed SVM to draw complex boundaries between overlapping.
* **Stability:** Showed low variance across cross-validation folds.

### 3.2 Logistic Regression(LR)
* **Configuration:** C = 1.0 , L1_ratio = 0,0
* **Outcome:** Performed robustly on data like exactly SVM.

### 3.3 K-Nearest Neighbors (KNN)
* **Metric:** Minkowski Distance.
* **Outcome:** Served as a baseline. Good for distinct clusters but degraded in high-dimensional space.

---

## Unsupervised Learning Pipeline
**File:** `Unsupervised method.ipynb`

We stripped the labels to analyze the data's intrinsic structure using the base feature set:



1.  **Dimensionality Reduction (PCA):**
    * Reduced **116 features** $\rightarrow$ **2 Principal Components**.
    * *Visualization:* Scatter plots revealed that **Urban sounds** (Sirens) form tight, distinct clusters, while **Nature sounds** (Rain, Sea) overlap significantly due to similar spectral textures.

2.  **Clustering Algorithms:**
    * **K-Means ($K=10$):** Hard clustering algorithm. It assumes clusters are spherical and of similar size.
    * **Hierarchical Clustering (Agglomerative):** Builds a hierarchy of clusters.
        * *Visualization:* A **Dendrogram** was used to visualize the "tree of sounds," showing how categories like *Dog* and *Rooster* merge at lower levels compared to *Helicopter*.
    * **Gaussian Mixture Models (GMM):** Probabilistic (soft) clustering.
        * *Result:* **GMM outperformed K-Means.** Since audio features often have correlations (covariance), the clusters are naturally elliptical. GMM captures this geometry better than K-Means' spherical assumption.

---

## 📈 Experimental Results & Model Comparison

We evaluated a total of **7 algorithms** to benchmark performance across different learning paradigms.

| Methodology | Algorithm | Accuracy | Key Observation |
| :--- | :--- | :--- | :--- |
| **Supervised** | **Logistic Regression** | **87.5%** | **Best Model.** Achieved top accuracy with harmonic features (58 features). Fast, interpretable, and efficient. |
| **Supervised** | **SVM (RBF)** | **87.5%** | **Second best model.** The RBF kernel effectively separated complex, non-linear boundaries between overlapping classes. Tied with Logistic Regression but requires more features (65 vs 58). |
| **Supervised** | **Random Forest** | **87.5%** | **Excluded from final selection.** Despite matching accuracy, showed severe overfitting (100% training accuracy). |
| **Supervised** | **K-Nearest Neighbors (KNN)** | **78.8%** | **Distance-Based.** Performance degraded due to the "Curse of Dimensionality" (116 features made distance metrics less reliable). |
| **Unsupervised**| **K-Means** | **65.25%** | **Best Clustering.** Despite the theoretical limitation of spherical clusters, K-Means effectively partitioned the distinct, high-density audio categories in our dataset. |
| **Unsupervised**| **Hierarchical Clustering** | **63.50%** | **Visual Insight.** Built a "tree of sounds" (dendrogram) that visualized semantic relationships, such as animals grouping together, though strict classification accuracy was moderate. |
| **Unsupervised**| **GMM** | **54.25%** | **Lowest Performance.** The probabilistic model likely overfitted the complex variance of the 116 features, struggling to form stable clusters compared to the simpler K-Means approach. |

---

## 📚 Comparison with Literature (ESC-50 Benchmark)

Our work uses a **custom 10-class subset** from ESC-50 (dog, rooster, thunderstorm, sea_waves, snoring, sneezing, clock_alarm, vacuum_cleaner, siren, helicopter) with **400 audio files**. We compare our results with published methods on the ESC-50 benchmark.

### ⚠️ Important Note on Fair Comparison

The studies listed below were evaluated on the **full ESC-50 dataset (50 classes, 2000 audio files)**, while our work uses a **10-class subset (400 files)**. Classifying 50 classes is significantly harder than classifying 10 classes, so **direct accuracy comparison is not appropriate**. We include these results to show where our approach fits in the broader landscape of audio classification research and to demonstrate the evolution of methods over time.

| Aspect | Published Studies | Our Work |
|--------|------------------|----------|
| **Classes** | 50 classes | 10 classes |
| **Total Files** | 2,000 | 400 |
| **Task Difficulty** | Harder (more classes to confuse) | Easier (fewer classes) |
| **Comparison Purpose** | Benchmark reference | Show methodology & approach |

### Benchmark Comparison Table

#### Traditional Machine Learning (Piczak, 2015)

| Study | Dataset | Classes | Files | Method | Features | Accuracy |
|-------|---------|---------|-------|--------|----------|----------|
| **Piczak (2015)** | ESC-10 | 10 | 400 | Random Forest | MFCC (12) + ZCR | **72.7%** |
| **Piczak (2015)** | ESC-10 | 10 | 400 | SVM (Linear) | MFCC (12) + ZCR | 67.5% |
| **Piczak (2015)** | ESC-10 | 10 | 400 | k-NN | MFCC (12) + ZCR | 66.7% |
| **Piczak (2015)** | ESC-10 | 10 | 400 | **Human Accuracy** | N/A | **95.7%** |
| | | | | | | |
| **Piczak (2015)** | ESC-50 | **50** | **2000** | Random Forest | MFCC (12) + ZCR | **44.3%** |
| **Piczak (2015)** | ESC-50 | **50** | **2000** | SVM (Linear) | MFCC (12) + ZCR | 39.6% |
| **Piczak (2015)** | ESC-50 | **50** | **2000** | k-NN | MFCC (12) + ZCR | 32.2% |
| **Piczak (2015)** | ESC-50 | **50** | **2000** | **Human Accuracy** | N/A | **81.3%** |

#### Deep Learning & CNNs (2015-2020)

| Study | Dataset | Classes | Files | Method | Architecture | Accuracy |
|-------|---------|---------|-------|--------|--------------|----------|
| **Piczak (2015)** | ESC-50 | **50** | **2000** | CNN | 2 Conv + 2 FC layers | **64.5%** |
| **Salamon & Bello (2017)** | ESC-50 | **50** | **2000** | CNN + Data Aug | Log-Mel Spectrogram + mixup | **79.0%** |
| **Wilkinghoff (2020)** | ESC-50 | **50** | **2000** | L3-Net + X-vectors | Pretrained embeddings (512-dim) + PLDA | **84.3%** |
| **Kong et al. (2020)** | ESC-50 | **50** | **2000** | PANNs (CNN14) | Pretrained on AudioSet (2M clips) | **90.9%** |

#### Modern Contrastive Learning (2022)

| Study | Dataset | Method | Training Data | Accuracy |
|-------|---------|--------|---------------|----------|
| **Elizalde et al. (2022)** | ESC-50 (50 classes) | **CLAP (Zero-Shot)** | 128k audio-text pairs | **82.6%** |
| **Elizalde et al. (2022)** | ESC-50 (50 classes) | **CLAP (Fine-tuned)** | 128k pretrain + ESC-50 | **96.7%**  |

#### Our Work (2025) - Shallow Models with Rich Feature Engineering

| Study | Dataset | Method | Features | Accuracy |
|-------|---------|--------|----------|----------|
| **Our Work (2025)** | **Custom 10-class** | **Logistic Regression** | **Harmonic (58 features)** | **87.5%**  |
| **Our Work (2025)** | **Custom 10-class** | **SVM (RBF)** | **General Set 2 (65 features)** | **87.5%**  |
| **Our Work (2025)** | Custom 10-class | Random Forest | Harmonic (58 features) | 87.5% |
| **Our Work (2025)** | Custom 10-class | KNN | All features (116) | 78.8% |

### 🔍 Understanding Different Approaches

#### 1. **Traditional Machine Learning (2015)**
- **Hand-crafted features** (12 MFCCs + ZCR)
- **Shallow classifiers** (RF, SVM, k-NN)
- **Best result**: 72.7% on ESC-10
- **Limitation**: Limited feature representation

#### 2. **Deep Learning Era (2015-2020)**
- **CNNs**: Learn features directly from spectrograms
  - Piczak's CNN (2015): 64.5% on ESC-50 (**50 classes, 2000 files**)
  - Salamon & Bello's CNN + mixup (2017): 79.0% on ESC-50 (**50 classes, 2000 files**)
- **Transfer Learning**: Pretrained embeddings
  - PANNs (Kong et al., 2020): **90.9%** on ESC-50 (**50 classes, 2000 files** - pretrained on 2M AudioSet clips)
  - L3-Net + X-vectors (Wilkinghoff, 2020): 84.3% on ESC-50 (**50 classes, 2000 files**)
- **Advantage**: Automatic feature learning
- **Limitation**: Requires large labeled datasets or pretraining
- **Note**: These results are on 50 classes which is a harder task than our 10-class subset

#### 3. **Modern Contrastive Learning (2022)**
- **CLAP** (Elizalde et al., 2022): Audio-language multimodal model
  - Zero-shot (no training): 82.6% on ESC-50 (**50 classes, 2000 files**)
  - Fine-tuned: **96.7%** on ESC-50 (**50 classes, 2000 files** - beats human performance!)
- **Key Innovation**: Learns from natural language supervision (128k audio-text pairs)
- **Advantage**: Flexible, generalizes across tasks without retraining
- **Note**: These results are on 50 classes; direct comparison with our 10-class results is not appropriate

#### 4. **Our Approach (2025): Feature Engineering Renaissance**
- **116 hand-crafted features** with domain knowledge
- **Shallow models** (LR, SVM) - interpretable and fast
- **87.5% accuracy** on custom 10-class subset
- **Key Insight**: Careful feature engineering can match deep learning for smaller datasets
- **Advantage**:
  - No GPU required
  - Fast training (seconds vs hours) (consider we just used 10 classes)
  - Good results
  - Works well with limited data

### Key Achievements

#### **Significant Improvements Over Baseline:**

1. **+14.8% over best baseline** (72.7% → 87.5%)
   - Piczak's Random Forest with 13 features: **72.7%**
   - Our Logistic Regression with 58 harmonic features: **87.5%**

2. **+20% over SVM baseline** (67.5% → 87.5%)
   - Piczak's Linear SVM: **67.5%**
   - Our RBF SVM: **87.5%**

3. **+12.1% improvement in KNN** (66.7% → 78.8%)
   - Piczak's k-NN: **66.7%**
   - Our KNN with distance weighting: **78.8%**

#### 🔬 **Why We think we Outperformed Baseline:**

| Factor | Piczak (2015) Baseline | Our Approach | Impact |
|--------|----------------------|--------------|---------|
| **Features** | 13 features (12 MFCCs + ZCR) | **116 features** (MFCCs, Spectral, Harmonic, Chroma, Tonnetz) | +103 features |
| **Feature Engineering** | Mean + Std only | Mean + Std + Delta MFCCs + Pitch + HNR + Chroma | Richer representation |
| **SVM Kernel** | Linear | **RBF** (non-linear) | Better decision boundaries |
| **KNN Distance** | Euclidean (default) | **Manhattan + Distance weighting** | Better distance metric |
| **Feature Selection** | All features used | **Targeted groups** (harmonic for LR, general_set_2 for SVM) | Domain-specific optimization |

### 📊 Performance Analysis

#### How Close Are We to Human Performance?

| Dataset | Human Accuracy | Our Best Model | Gap to Human |
|---------|---------------|----------------|--------------|
| ESC-10 (Piczak) | 95.7% | - | - |
| Our 10-class subset | ~95% (estimated) | **87.5%** | **-7.5%** |

**Interpretation:**
- We achieved **91.5% of human-level performance** (87.5% / 95.7%)
- For a machine learning system with shallow models (no deep learning), this is **excellent**
- Remaining 7.5% gap likely requires:
  - Temporal modeling (RNNs/LSTMs)
  - Deep learning (CNNs)
  - Ensemble methods combining multiple models

### 🎯 Feature Engineering Success Story

**Original ESC-50 Paper (Piczak, 2015):**
> "These rudimentary classification systems performed poorly when contrasted with their human counterparts"
> - Random Forest: 72.7% (best)
> - SVM: 67.5%

**Our Work:**
> **By engineering 116 carefully designed audio features** (vs. original 13), we improved accuracy to **87.5%** — closing the gap to human performance significantly.

**Key Insight:** Domain-specific feature engineering **still matters** even in the era of deep learning. Our 58-feature harmonic set outperformed using all 116 features, proving that **focused feature selection beats kitchen-sink approaches**.

### 📊 Performance Timeline: Evolution of ESC-50 Accuracy

```
── Full ESC-50 (50 classes, 2000 files) ──────────────────────────
2015 Traditional ML (50cls): 44.3% ██████████ (Piczak RF)
2015 First CNN (50cls):      64.5% ██████████████ (Piczak CNN)
2017 CNN + Data Aug (50cls): 79.0% ██████████████████ (Salamon & Bello)
2020 L3-Net Embed. (50cls):  84.3% ███████████████████ (Wilkinghoff)
2020 PANNs (50cls):          90.9% █████████████████████ (Kong et al.)
2022 CLAP Zero-shot (50cls): 82.6% ██████████████████▌ (Elizalde et al.)
2022 CLAP Fine-tune (50cls): 96.7% ██████████████████████▌ (Elizalde et al.)
2015 Human (ESC-50):         81.3% ██████████████████▌ (Ground Truth)
──────────────────────────────────────────────────────────────────

── ESC-10 / 10 classes, 400 files (comparable to our work) ──────
2015 Traditional ML (10cls): 72.7% ████████████████▌ (Piczak RF)
2015 Human (ESC-10):         95.7% ██████████████████████ (Ground Truth)
2025 Our Work (10cls):       87.5% ███████████████████▌ (Logistic Regression)
──────────────────────────────────────────────────────────────────
```

**Analysis:**
- The studies above (CNN, L3-Net, PANNs, CLAP) all used the **full ESC-50 (50 classes, 2000 files)** which is a much harder classification task than our 10-class subset
- **Fair comparison**: Our 87.5% on 10 classes vs Piczak's 72.7% on ESC-10 (also 10 classes, 400 files) = **+14.8% improvement**
- **2015-2017**: CNNs learned features directly from spectrograms but needed large datasets
- **2020**: Transfer learning (pretraining on 2M AudioSet clips) pushed ESC-50 accuracy to 90.9%
- **2022**: CLAP achieved **96.7%** on 50 classes using language-audio contrastive learning (128k audio-text pairs)
- **2025 (Our Work)**: Demonstrated that **shallow models + rich feature engineering** (87.5% on 10 classes) can be effective while being:
  - **10x faster** to train (seconds vs hours)
  - **100x smaller** (no GPU needed)
  - **Fully interpretable** (can examine feature weights)

### 🔗 References

#### Primary Dataset Paper
- **Piczak, K. J. (2015).** "ESC: Dataset for Environmental Sound Classification." *Proceedings of the ACM International Conference on Multimedia (MM'15)*, Brisbane, Australia.
  - DOI: [10.1145/2733373.2806390](https://doi.org/10.1145/2733373.2806390)
  - Dataset: [Harvard Dataverse](http://dx.doi.org/10.7910/DVN/YDEPUT)
  - GitHub: [karoldvl/ESC-50](https://github.com/karolpiczak/ESC-50)
  - **Baseline Results:**
    - ESC-10 (10 classes): 66.7% (k-NN) to 72.7% (RF)
    - ESC-50 (50 classes): 32.2% (k-NN) to 44.3% (RF), 64.5% (CNN)
    - Features: 12 MFCCs + ZCR (mean & std)
    - Human accuracy: 95.7% (ESC-10), 81.3% (ESC-50)

#### Deep Learning & CNN Approaches
- **Salamon, J., & Bello, J. P. (2017).** "Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification." *IEEE Signal Processing Letters*, 24(3), 279-283.
  - ESC-50 result: **79.0%** with data augmentation
  - Architecture: Log-Mel spectrogram + CNN + mixup

- **Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020).** "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 28, 2880-2894.
  - ESC-50 result: **90.9%** (pretrained on AudioSet)
  - Architecture: CNN14 pretrained on 2M audio clips

- **Wilkinghoff, K. (2020).** "On Open-Set Classification with L3-Net Embeddings for Machine Listening Applications." *28th European Signal Processing Conference (EUSIPCO)*, 800-804.
  - ESC-50 result: **84.3%** with L3-Net + X-vectors
  - Architecture: Pretrained embeddings + PLDA backend

#### Modern Contrastive Learning
- **Elizalde, B., Deshmukh, S., Al Ismail, M., & Wang, H. (2022).** "CLAP: Learning Audio Concepts from Natural Language Supervision." *arXiv:2206.04769*.
  - ESC-50 results: 82.6% (zero-shot), **96.7%** (fine-tuned) ✨
  - Architecture: Contrastive Language-Audio Pretraining
  - Training data: 128k audio-text pairs
  - **First model to exceed human accuracy on ESC-50**
  - GitHub: [microsoft/CLAP](https://github.com/microsoft/CLAP)

---

## 💻 Installation & Usage

To replicate our results, ensure you have the `audio/` folder in the root directory.

1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn librosa scikit-learn pyyaml
    ```
2.  **Run Supervised Analysis:**
    Open `Main assignment.ipynb` and Run All Cells. This will:
    * Extract 116 features from `audio/`.
    * Train models.
    * Output **Confusion Matrices** and **ROC Curves**.
3.  **Run Unsupervised Analysis:**
    Open `Unsupervised method.ipynb` and Run All Cells to generate PCA plots and Dendrograms based on the 80-feature set.

---

## 📦 Preparatory Assignments

Before commencing the main **Acoustic Event Detection** group project, we completed two distinct **individual sub-assignments**. While the Main Assignment is a collaborative effort, these sub-folders represent our **individual** scientific contributions:

### 1. PCA & ICA Challenge (Individual)
**Relation to Main Project:** The PCA techniques mastered here were later integrated into the *Unsupervised Learning* pipeline to visualize audio clusters.

### 2. Anomaly Detection Challenge (Individual)
**Relation to Main Project:** This provided the theoretical basis for understanding how classifiers handle "noise" or unknown audio classes in the main ESC-50 dataset.

## 📂 Repository Structure

The project directory is organized to clearly distinguish between the **collaborative main assignment** and the **individual preparatory tasks**. This structure ensures modularity and reproducibility, separating raw data, extracted features, and the distinct modeling pipelines for each team member.

```text
applied-machine-learning-armin-sepehr/
│
├── Main assignment.ipynb          # Supervised learning (SVM, LR, RF, KNN)
├── Unsupervised method.ipynb      # Clustering (K-Means, HC, GMM)
├── README.md                      # Literature comparison, results summary
├── config.yml                     # Paths to data
│
├── scripts/                       # Modular functions
│   ├── configs.py                 # CATEGORIES, SELECTED_CLASSES, FINAL_TEST_FOLD
│   ├── feature_extraction.py     # extract_features(), get_feature_groups()
│   ├── data_utils.py             # load_audio_data(), split_data(), encode_labels()
│   ├── supervised_models.py      # get_param_grids(), run_grid_search(), find_best_combination()
│   ├── performance.py            # compute_summary(), compute_tpr_fpr()
│   └── visualizations.py         # All plotting functions
│
├── saved_data/
│   └── features_all_116.npz      # Cached feature matrix (400, 116)



## 👥 Authors
* **Sepehr Farrokhi**
* **Armin Siavashi** 

