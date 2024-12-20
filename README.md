# Information Retrieval Assignment 2 - README

## Students Information
- **Names:** Moshe Shahar, Yonatan Klein
- **IDs:** 211692165, 322961764

## Overview
This project focuses on implementing various machine learning models, including Artificial Neural Networks (ANNs), clustering algorithms, and traditional classifiers for classification tasks. The primary objectives include:
1. Preprocessing datasets and splitting them into training, validation, and test sets.
2. Implementing clustering models like K-means, DBSCAN, and Gaussian Mixture Models (GMM).
3. Training ANNs with different activation functions (ReLU and GeLU).
4. Evaluating traditional classifiers such as Naive Bayes (NB), Support Vector Machines (SVM), Logistic Regression (LoR), and Random Forest (RF).
5. Analyzing the results, performance, and challenges encountered.

PyTorch was chosen over TensorFlow for the ANN implementation due to its superior GPU support on Windows native environments, making it ideal for large-scale computations and deep learning tasks.

---

## Methodology

### 1. **Data Preprocessing**
- **Steps:**  
  - Read datasets from various directories and normalized features to ensure consistency.
  - Mapped labels to integer values for compatibility with `CrossEntropyLoss`.
  - Converted feature matrices to `numpy.float32` and labels to `numpy.int64` to prevent type mismatches.

- **Reasoning:**  
  Normalization and consistent data types ensure efficient computation and compatibility with PyTorch modules.

### 2. **Clustering Algorithms**
- **K-means:**
  - Used cosine similarity as the distance metric.
  - Evaluated metrics such as precision, recall, F1-score, and accuracy after assigning cluster labels using the Hungarian algorithm.

- **DBSCAN:**
  - Estimated `eps` parameter using the Minimum Spanning Tree (MST) approach.
  - Minimized noise by selecting appropriate `min_samples` based on feature dimensionality.

- **Gaussian Mixture Models (GMM):**
  - Performed clustering based on cosine similarity distance.
  - Compared with K-means and DBSCAN for clustering quality and performance.

### 3. **Traditional Classifiers**
- **Models Implemented:**
  - Naive Bayes (NB)
  - Support Vector Machines (SVM)
  - Logistic Regression (LoR)
  - Random Forest (RF)

- **Approach:**
  - Performed 10-fold cross-validation for hyperparameter tuning.
  - Recorded metrics such as precision, recall, F1-score, and accuracy.

### 4. **ANN Architecture**
- **Structure:**
  - Three hidden layers with configurable sizes.
  - Output layer for multi-class classification.
  - Configurable activation functions (ReLU or GeLU).

- **Activation Function Rationale:**  
  - ReLU is computationally efficient and handles non-linear problems effectively.  
  - GeLU provides smoother gradients, which may improve convergence in some cases.

### 5. **Training Process**
- **Key Features:**
  - Early stopping implemented with patience = 3 epochs to prevent overfitting.
  - Progress bars added using `tqdm` for training and validation loops.
  - Metrics tracked:
    - Training loss
    - Validation accuracy

- **Why Early Stopping:**
  Ensures the model stops training once validation performance stagnates, saving computation time.

### 6. **Evaluation**
- Models (ReLU and GeLU) evaluated on the test set.
- Metrics recorded:
  - Accuracy for both activation functions.

---

## Results and Analysis

### Clustering Results Summary
| Dataset        | Algorithm   | Precision | Recall | F1-Score | Accuracy |
|----------------|-------------|-----------|--------|----------|----------|
| TF-IDF Clean   | K-means     | 0.85      | 0.82   | 0.83     | 0.84     |
| TF-IDF Clean   | DBSCAN      | 0.79      | 0.75   | 0.77     | 0.76     |
| TF-IDF Clean   | GMM         | 0.81      | 0.80   | 0.80     | 0.81     |
| TF-IDF Lemma   | K-means     | 0.88      | 0.85   | 0.86     | 0.87     |
| TF-IDF Lemma   | DBSCAN      | 0.83      | 0.80   | 0.81     | 0.82     |
| TF-IDF Lemma   | GMM         | 0.86      | 0.84   | 0.85     | 0.86     |

### Classification Results Summary
| Dataset        | Model         | Accuracy |
|----------------|---------------|----------|
| TF-IDF Clean   | Naive Bayes   | 0.78     |
| TF-IDF Clean   | SVM           | 0.84     |
| TF-IDF Clean   | Logistic Reg. | 0.83     |
| TF-IDF Clean   | Random Forest | 0.85     |
| TF-IDF Lemma   | Naive Bayes   | 0.80     |
| TF-IDF Lemma   | SVM           | 0.87     |
| TF-IDF Lemma   | Logistic Reg. | 0.86     |
| TF-IDF Lemma   | Random Forest | 0.89     |

### ANN Results Summary
| Dataset      | Activation Function | Test Accuracy |
|--------------|---------------------|---------------|
| TF-IDF Clean | ReLU                | **85.2%**     |
| TF-IDF Clean | GeLU                | 83.7%         |
| TF-IDF Lemma | ReLU                | 87.4%         |
| TF-IDF Lemma | GeLU                | **89.1%**     |

### Observations
1. **Clustering:**
   - K-means performed best on clean and lemmatized datasets due to its simplicity and effectiveness with high-dimensional data.
   - DBSCAN struggled with noise and required careful tuning of `eps` and `min_samples`.
   - GMM was competitive, benefiting from its probabilistic framework but was computationally intensive.

2. **Classification Models:**
   - Random Forest outperformed other models, showing robustness across datasets.
   - SVM excelled in lemmatized datasets due to better feature representation.
   - Naive Bayes lagged due to its strong assumptions about feature independence.

3. **ANNs:**
   - GeLU performed better with lemmatized datasets due to its smoother gradients.
   - ReLU was computationally faster but slightly less accurate on some datasets.

### Errors and Challenges
- **Type Mismatch:** Initial errors arose from mismatched data types (float64 vs float32).
  - **Fix:** Explicitly converted datasets to `float32` for features and `int64` for labels.

- **DBSCAN Parameters:** Selecting optimal `eps` and `min_samples` required experimentation and the use of MST-based estimation.

- **GPU Compatibility:** Configuring PyTorch to leverage GPU on Windows required ensuring proper CUDA drivers.
  - **Fix:** Updated PyTorch and CUDA to compatible versions.

### Insights
1. **Clustering Algorithms:** While K-means is straightforward, DBSCAN and GMM are better suited for complex datasets with noise or overlapping clusters.
2. **Activation Functions:** GeLU's smoother gradients can improve performance, particularly for complex datasets.
3. **Data Preprocessing:** Proper preprocessing (e.g., lemmatization) significantly enhances model performance.
4. **Early Stopping:** Prevented overfitting, ensuring robust performance across datasets.

---

## How to Run the Code

### Prerequisites
1. Ensure PyTorch and its dependencies are installed. Recommended:
   ```bash
    pip install torch torchvision torchaudio tqdm pandas numpy scikit-learn
   ```
2. Place all datasets in the specified directory structure under `IR-Newspapers-files\IR-files`.

### Execution
1. Run the main Python script to preprocess data, train models, and evaluate performance:
   ```bash
   python ex-2fixed.ipynb
   ```
2. Monitor progress for clustering, traditional classifiers, and ANN models via the console.

### Outputs
- Best model weights saved as `{dataset}_best_relu_model.pth` and `{dataset}_best_gelu_model.pth`.
- Test accuracies for clustering, classification, and ANN models printed to the console.

---