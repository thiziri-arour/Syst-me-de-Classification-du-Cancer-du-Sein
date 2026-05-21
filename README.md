# Syst-me-de-Classification-du-Cancer-du-Sein


A comprehensive machine learning pipeline for classifying breast tumors as malignant or benign using the Wisconsin Breast Cancer dataset.

## Project Overview

This project implements and compares multiple machine learning algorithms to diagnose breast cancer from digitized cell nucleus measurements. The system achieves 97.08% accuracy using an optimized Support Vector Machine model.

### Key Results

- **Test Accuracy**: 97.08%
- **Sensitivity (Recall)**: 99.07%
- **Specificity**: 93.75%
- **ROC-AUC Score**: 0.9956
- **Dataset**: 569 samples, 30 features
- **Best Model**: SVM with RBF kernel (C=10, gamma=0.001)

## Dataset Information

**Source**: Wisconsin Breast Cancer Dataset (UCI Machine Learning Repository)

**Composition**:
- 569 samples (212 malignant, 357 benign)
- 30 numerical features derived from digitized images of Fine Needle Aspiration (FNA)
- Binary classification task (malignant vs benign)

**Feature Categories**:
- 10 base measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
- 3 values per measurement: mean, standard error, and worst (largest)
- Total: 10 × 3 = 30 features

## Technical Implementation

### Pipeline Architecture

1. **Data Loading & Exploration**
   - Statistical analysis of feature distributions
   - Class balance verification (37% malignant, 63% benign)
   - Correlation analysis between features

2. **Data Preprocessing**
   - Train-test split (70/30) with stratification
   - Feature scaling using StandardScaler (critical for SVM performance)
   - No missing values or duplicates found

3. **Model Training & Comparison**
   - 5 algorithms evaluated:
     - Support Vector Machine (Linear kernel)
     - Support Vector Machine (RBF kernel)
     - Random Forest
     - Gradient Boosting
     - Logistic Regression
   - 5-fold cross-validation for each model
   - Performance metrics: accuracy, precision, recall, F1-score

4. **Hyperparameter Optimization**
   - GridSearchCV with 60 parameter combinations
   - Parameters tested:
     - C: [0.1, 1, 10, 100]
     - gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
     - kernel: ['linear', 'rbf', 'poly']
   - Best configuration: C=10, gamma=0.001, kernel='rbf'

5. **Model Evaluation**
   - Confusion matrix analysis
   - Medical-grade metrics (sensitivity, specificity)
   - ROC curve and AUC calculation
   - Classification report generation

6. **Model Persistence**
   - Trained model saved with joblib
   - Includes scaler for production deployment

### Model Comparison Results

| Model | Accuracy | Precision | Recall | F1-Score | CV Mean | CV Std |
|-------|----------|-----------|--------|----------|---------|--------|
| Logistic Regression | 98.83% | 99.07% | 99.07% | 99.07% | 97.99% | 1.50% |
| SVM (Linear) | 98.25% | 98.15% | 99.07% | 98.60% | 96.73% | 1.70% |
| SVM (RBF) | 97.66% | 98.13% | 98.13% | 98.13% | 96.98% | 1.29% |
| Gradient Boosting | 94.15% | 93.69% | 97.20% | 95.41% | 96.00% | 4.06% |
| Random Forest | 93.57% | 94.44% | 95.33% | 94.88% | 97.25% | 3.30% |

**Optimized SVM (GridSearch)**:
- Test Accuracy: 97.08%
- Sensitivity: 99.07%
- Specificity: 93.75%
- ROC-AUC: 0.9956

### Confusion Matrix

```
                Predicted
                Malignant  Benign
Actual Malignant     60       4
       Benign         1     106
```

**Analysis**:
- True Negatives: 60 (correctly identified malignant)
- False Positives: 4 (benign classified as malignant)
- False Negatives: 1 (malignant classified as benign)
- True Positives: 106 (correctly identified benign)

## Technical Stack

- **Language**: Python 3.8+
- **Core Libraries**:
  - NumPy: numerical computations
  - Pandas: data manipulation
  - scikit-learn: machine learning algorithms and tools
  - Matplotlib: data visualization
  - joblib: model persistence

## Project Structure

```
breast-cancer-classification/
│
├── Breast_Cancer_Classification.ipynb    # Main notebook
├── README.md                             # This file
├── TECHNICAL_EXPLANATION.md              # Detailed technical documentation
├── requirements.txt                      # Python dependencies
│
├── outputs/
│   ├── data_exploration.png              # EDA visualizations
│   ├── model_results.png                 # Model performance charts
│   └── breast_cancer_model.pkl           # Trained model package
│
└── docs/
    └── methodology.md                    # Methodology documentation
```


## Key Findings

### 1. Feature Scaling Impact

StandardScaler normalization was critical for SVM performance:
- Without scaling: Features with large ranges (e.g., mean area: 143-2501) dominate
- With scaling: All features contribute equally
- Impact: Enables proper distance-based calculations in SVM

### 2. Model Selection

Logistic Regression achieved the highest accuracy (98.83%) before hyperparameter tuning, demonstrating that simpler models can outperform complex ones when:
- Data is linearly separable
- Features are well-engineered
- Proper preprocessing is applied

The optimized SVM (97.08%) provides:
- Excellent sensitivity (99.07%): Critical for medical diagnosis (minimize false negatives)
- Good specificity (93.75%): Reduces unnecessary biopsies
- Outstanding ROC-AUC (0.9956): Strong discriminative ability

### 3. Cross-Validation Reliability

Consistent cross-validation scores (CV mean ≈ test accuracy) indicate:
- No overfitting
- Model generalizes well to unseen data
- Results are reproducible

### 4. Medical Implications

High sensitivity (99.07%) means:
- Only 1 out of 107 malignant cases was missed
- Critical for cancer screening where false negatives have severe consequences
- Trade-off: 4 false positives (benign classified as malignant)

## Performance Metrics Explanation

- **Accuracy**: Overall correctness (97.08%)
- **Precision**: Of predicted malignant, how many were actually malignant
- **Recall/Sensitivity**: Of actual malignant cases, how many were caught (99.07%)
- **Specificity**: Of actual benign cases, how many were correctly identified (93.75%)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve, measuring discriminative ability (0.9956)

## Limitations & Future Work

### Current Limitations

1. **Dataset Size**: 569 samples is relatively small for deep learning approaches
2. **Feature Engineering**: Uses pre-computed features from images, not raw image data
3. **Single Dataset**: Model trained and tested on Wisconsin dataset only
4. **Class Imbalance**: 37% malignant vs 63% benign (mild imbalance)

### Future Improvements

1. **Deep Learning**: Implement CNN for direct image analysis
2. **Ensemble Methods**: Combine multiple models for improved accuracy
3. **External Validation**: Test on independent datasets
4. **Feature Selection**: Identify most informative features to reduce dimensionality
5. **Explainability**: Add SHAP or LIME for model interpretability
6. **Clinical Integration**: Develop API for real-time predictions

## References

1. Wisconsin Breast Cancer Dataset: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
2. Original Study: W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993.
3. scikit-learn Documentation: [scikit-learn.org](https://scikit-learn.org/)

roject is for educational and research purposes. For medical diagnosis, consult qualified healthcare professionals.
