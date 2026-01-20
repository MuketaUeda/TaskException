# Task Exception Prediction

A machine learning project for predicting task exceptions in transportation operations using XGBoost classification. This project implements a complete ML pipeline from data cleaning to model evaluation, with a focus on achieving high precision (>50%) as per business requirements.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Model Selection](#model-selection)
- [Project Structure Details](#project-structure-details)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a binary classification model to predict task exceptions (RED) versus normal operations (GOOD) in transportation logistics. The model is designed to prioritize precision over recall, ensuring that when an exception is predicted, it is highly likely to be accurate, minimizing false alarms in operational workflows.

## 🎯 Project Objectives

- **Primary Goal**: Train a model to predict exceptions with **precision > 50%**, prioritizing precision over recall
- **Target Variable**: `Exception_output` (RED = exception occurred, GOOD = no exception)
- **Data Split**: 95% training, 5% test (stratified to maintain class proportions)
- **Model**: XGBoost with precision-focused hyperparameters

## ✨ Key Features

- **Comprehensive Data Pipeline**: Automated data cleaning, preprocessing, and feature engineering
- **Stratified Data Splitting**: Maintains class distribution across train/test sets
- **Precision-Focused Training**: XGBoost model optimized for high precision
- **Threshold Analysis**: Manual threshold selection to balance precision and recall
- **Overfitting Analysis**: Comprehensive evaluation of model generalization
- **Feature Importance Analysis**: Identification of key predictive features
- **Rich Visualizations**: Multiple plots for model interpretation and evaluation
- **Automated Testing**: Comprehensive test suite with high coverage
- **CI/CD Integration**: GitHub Actions for automated testing and quality checks

## 📁 Project Structure

```
TaskException/
├── data/
│   ├── raw/                          # Original dataset
│   └── processed/                    # Cleaned and processed data
├── src/
│   ├── __init__.py
│   ├── cleaning.py                   # Data cleaning transformers
│   ├── modeling.py                   # Model training and evaluation
│   └── visualization.py              # Plotting functions
├── notebook/
│   └── model.ipynb                   # Main analysis notebook
├── test/
│   ├── test_cleaning.py              # Tests for data cleaning
│   └── test_modeling.py              # Tests for modeling functions
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI/CD pipeline
├── requirements.txt                  # Python dependencies
├── setup.cfg                         # Flake8 configuration
├── .coveragerc                       # Coverage configuration
└── README.md                         # This file
```

## 📦 Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

### Key Dependencies

- `pandas>=2.3.3` - Data manipulation
- `numpy>=1.26.0,<2.4` - Numerical computing
- `scikit-learn>=1.8.0` - Machine learning utilities
- `xgboost>=3.1.3` - Gradient boosting model
- `matplotlib>=3.10.8` - Plotting
- `seaborn>=0.13.2` - Statistical visualizations
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd TaskException
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   pytest test/ -v
   ```

## 💻 Usage

### Running the Complete Pipeline

1. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook notebook/model.ipynb
   ```

2. **Execute all cells** to run the complete pipeline:
   - Data loading and cleaning
   - Feature encoding
   - Train/test split (95%/5%)
   - Model training
   - Threshold analysis
   - Model evaluation
   - Visualizations

### Using Individual Modules

#### Data Cleaning
```python
from src.cleaning import NumericCleaner, TextNormalizer

# Clean numeric columns
cleaner = NumericCleaner(target_columns=['Loading_meter [ldm]', ...])
df_cleaned = cleaner.transform(df)

# Normalize text columns
normalizer = TextNormalizer(target_columns=['Exception_output'])
df_normalized = normalizer.transform(df_cleaned)
```

#### Model Training
```python
from src.modeling import train_xgboost, evaluate_model, split_data

# Split data
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.05)

# Train model
model, history = train_xgboost(X_train, y_train, X_test, y_test, 
                               focus_precision=True)

# Evaluate
metrics = evaluate_model(model, X_train, y_train, X_test, y_test, 
                        threshold=0.840)
```

#### Visualizations
```python
from src.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metrics_comparison,
    plot_class_distribution,
    plot_overfitting_analysis,
    plot_roc_curve
)

# Generate visualizations
plot_class_distribution(y_train, y_test)
plot_metrics_comparison(metrics_train, metrics_test)
plot_feature_importance(model, feature_names, top_n=15)
```

## 📊 Model Performance

### Test Set Results

- **Accuracy**: 89.18%
- **Precision**: 65.98% ✅ (exceeds 50% requirement)
- **Recall**: 31.79%
- **F1-Score**: 0.428

### Key Achievements

✅ **Precision Requirement Met**: 65.98% > 50% threshold  
✅ **Good Generalization**: Small gap between train and test metrics  
✅ **Overfitting Control**: Early stopping and regularization prevent overfitting  
✅ **Stratified Split**: Class proportions maintained across datasets

### Model Characteristics

- **Optimal Threshold**: 0.840 (selected through manual analysis)
- **Training Samples**: 89,737 (95%)
- **Test Samples**: 4,723 (5%)
- **Features**: 26
- **Class Distribution**: Imbalanced (GOOD >> RED)

## 🔬 Model Selection

### Why XGBoost?

XGBoost was selected over Random Forest for the following reasons:

1. **Superior Regularization**: Fine-grained control via L1/L2 regularization for precision optimization
2. **Early Stopping**: Built-in mechanism to prevent overfitting
3. **Class Imbalance Handling**: Effective `scale_pos_weight` parameter
4. **Precision Tuning**: Granular hyperparameter control for precision-focused objectives
5. **Gradient Boosting**: Sequential learning minimizes false positives
6. **Computational Efficiency**: Faster training for iterative hyperparameter tuning

### Hyperparameters

The model uses precision-focused hyperparameters:
- `max_depth`: 4 (reduced to prevent overfitting)
- `learning_rate`: 0.03 (conservative learning)
- `min_child_weight`: 7 (increased for precision)
- `gamma`: 0.3 (increased regularization)
- `reg_alpha`: 0.2 (L1 regularization)
- `reg_lambda`: 2.0 (L2 regularization)
- `early_stopping_rounds`: 50

## 📂 Project Structure Details

### Source Code (`src/`)

- **`cleaning.py`**: Custom transformers for data cleaning
  - `NumericCleaner`: Handles European number format conversion
  - `TextNormalizer`: Normalizes text columns

- **`modeling.py`**: Core modeling functions
  - `load_and_prepare_data()`: Data loading and cleaning
  - `encode_categorical_features()`: Label encoding
  - `split_data()`: Stratified train/test split
  - `train_xgboost()`: Model training with early stopping
  - `evaluate_model()`: Performance evaluation
  - `get_feature_importance()`: Feature importance extraction
  - `check_overfitting()`: Overfitting analysis
  - `get_predictions()`: Prediction helper function

- **`visualization.py`**: Plotting utilities
  - `plot_class_distribution()`: Class balance visualization
  - `plot_metrics_comparison()`: Train vs test metrics
  - `plot_feature_importance()`: Top features visualization
  - `plot_confusion_matrix()`: Classification results
  - `plot_roc_curve()`: ROC curve and AUC
  - `plot_overfitting_analysis()`: Overfitting assessment
  - `create_model_report()`: Comprehensive model report

### Notebook (`notebook/`)

The main analysis notebook (`model.ipynb`) contains:
1. Project overview and model selection rationale
2. Imports and configuration
3. Data loading and exploration
4. Data cleaning pipeline
5. Categorical feature encoding
6. Feature and target preparation
7. Train/test split (95%/5%)
8. Model training with XGBoost
9. Threshold analysis and selection
10. Model evaluation
11. Overfitting analysis
12. Feature importance analysis
13. Individual visualizations
14. Summary and conclusions

## 🧪 Testing

The project includes comprehensive test coverage:

```bash
# Run all tests
pytest test/ -v

# Run with coverage
pytest test/ --cov=src --cov-report=html

# Run specific test file
pytest test/test_modeling.py -v
```

### Test Coverage

- **Data Cleaning Tests**: Validate transformer functionality
- **Modeling Tests**: Verify model training and evaluation
- **Edge Cases**: Handle missing data, edge cases
- **Integration Tests**: End-to-end pipeline validation

## 🔄 CI/CD

The project uses GitHub Actions for continuous integration:

- **Automated Testing**: Runs on every push and pull request
- **Code Quality**: Flake8 and Pylint checks
- **Coverage Reporting**: Tracks test coverage
- **Python Version**: Tests on multiple Python versions

See `.github/workflows/ci.yml` for configuration details.

## 📈 Key Insights

1. **Feature Importance**: Top features driving predictions include operational metrics like number of stops, weight, and time windows
2. **Class Imbalance**: Significant imbalance between GOOD and RED classes requires careful handling
3. **Threshold Selection**: Manual threshold analysis (0.840) provides optimal precision-recall balance
4. **Model Generalization**: Small gaps between train/test metrics indicate good generalization

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all functions
- Maintain test coverage above 80%

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**TaskException Team**

## 🙏 Acknowledgments

- XGBoost development team for the excellent gradient boosting library
- scikit-learn for comprehensive ML utilities
- The open-source community for valuable tools and libraries

---

**Note**: This project is designed for educational and research purposes. For production use, additional considerations such as model monitoring, retraining pipelines, and deployment infrastructure should be implemented.
