import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score
import xgboost as xgb
try:
    # Try relative import first (when used as package)
    from .cleaning import NumericCleaner, TextNormalizer
except ImportError:
    # Fallback to absolute import (when run directly or imported differently)
    from src.cleaning import NumericCleaner, TextNormalizer
import warnings
warnings.filterwarnings('ignore')

# Global configuration for verbose output
VERBOSE = True  # Set to False to disable all prints globally


def load_and_prepare_data(csv_path='data/raw/Task_Exception Prediction_Training Test Data.csv', verbose=None):
    """
    Loads CSV and applies cleaning transformers.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Cleaned and prepared DataFrame
    """
    # Use global VERBOSE if verbose is None
    if verbose is None:
        verbose = VERBOSE
    
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    
    df = df.replace(['', 'nan', 'NaN', 'None', 'null', 'NULL', 'N/A', 'n/a'], np.nan)
    
    if verbose:
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
    
    # Define numeric columns that need cleaning (European format)
    numeric_columns = [
        'Loading_meter [ldm]',
        'Gross_weight [kg]',
        'Volume [m3]',
        'Handling_unit_quantity [qty]',
        'Billed freight weight [kg]',
        'Number_of_Stops',
        'Weeks_after_project_GoLive',
        'Pickup_Month',
        'Pickup_Year',
        'Custom clearance needed',
        'Pickup_timewindow_length [hrs]',
        'Delivery_timewindow_length [hrs]',
        'Pickup_weeknumber',
        'Delivery_weeknumber'
    ]

    # Apply the numeric cleaner to the numeric columns
    if verbose:
        print("\nApplying NumericCleaner to numeric columns...")
    numeric_cleaner = NumericCleaner(target_columns=numeric_columns)
    df_cleaned = numeric_cleaner.transform(df)

    # apply the text normalizer to the text/categorical columns
    if verbose:
        print("Applying TextNormalizer to text/categorical columns...")
    text_normalizer = TextNormalizer()
    df_cleaned = text_normalizer.transform(df_cleaned)
    
    if verbose:
        print("\nData cleaned successfully!")
        print(f"Shape after cleaning: {df_cleaned.shape}\n")
    
    return df_cleaned



def encode_categorical_features(df, verbose=None):
    """
    Applies Label Encoding to categorical variables.
    
    Args:
        df: DataFrame with cleaned data
        
    Returns:
        DataFrame with encoded categoricals and dict of encoders for reference
    """
    if verbose is None:
        verbose = VERBOSE
    
    df_encoded = df.copy()

    target_column = 'Exception_output'
    categorical_columns = []

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object' and col != target_column:
            categorical_columns.append(col)
    
    if verbose:
        print(f"Found {len(categorical_columns)} categorical columns: {categorical_columns}")
        for col in categorical_columns:
            unique_count = df_encoded[col].nunique()
            print(f"  - {col}: {unique_count} unique values")
    
    encoders_dict = {}

    for col in categorical_columns:
        le = LabelEncoder()

        has_nan = df_encoded[col].isna().any()

        if has_nan:
            col_series = df_encoded[col].fillna('_MISSING_')
            encoded_values = le.fit_transform(col_series.astype(str))

            if '_MISSING_' in le.classes_:
                missing_label = le.transform(['_MISSING_'])[0]
                encoded_values = np.where(encoded_values == missing_label, np.nan, encoded_values)
            
            df_encoded[col] = encoded_values
        else:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        encoders_dict[col] = le

        if verbose:
            print(f"  ✓ Encoded: {col}")
            print(f"    Classes: {len(le.classes_)} unique categories")
            if len(le.classes_) <= 10:
                mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
                print(f"    Mapping: {mapping}")
            else:
                sample_mapping = {cls: idx for idx, cls in enumerate(le.classes_[:5])}
                print(f"    Mapping (sample): {sample_mapping} ... ({len(le.classes_) - 5} more)")
    
    if verbose:
        print(f"\n✓ Encoding completed!")
        print(f"  Total columns encoded: {len(categorical_columns)}")
        print(f"  Shape: {df_encoded.shape}\n")
    
    return df_encoded, encoders_dict


def prepare_features_and_target(df, verbose=None):
    """
    Separates features (X) and target (y).
    
    Target: Exception_output
    - "RED" = 1 (exception occurred)
    - "GOOD" = 0 (no exception)
    
    Args:
        df: DataFrame with encoded data
        
    Returns:
        X (features), y (target)
    """
    if verbose is None:
        verbose = VERBOSE
    
    # Check if target column exists
    target_column = 'Exception_output'
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    if verbose:
        print(f"Target column: {target_column}")
        print(f"Total columns in dataset: {len(df.columns)} (including target)")
    
    # Separate target (y)
    y = df[target_column].copy()
    
    if verbose:
        print(f"\nTarget distribution (before encoding):")
        value_counts = y.value_counts()
        for value, count in value_counts.items():
            percentage = (count / len(y)) * 100
            print(f"  - {value}: {count} ({percentage:.2f}%)")
    
    # Convert target from string to binary
    # 'red' or 'RED' → 1 (exception occurred)
    # 'good' or 'GOOD' → 0 (no exception)
    y_str = y.astype(str).str.lower().str.strip()
    
    # Use map to avoid FutureWarning from replace
    # map() automatically returns NaN for unmapped values
    y_binary = y_str.map({'red': 1, 'good': 0})
    
    # Check if there are invalid values (not 'red' nor 'good')
    # map() already converts unmapped values to NaN, so we check the NaN
    # that were created by unmapped values
    nan_from_invalid = y_binary.isna() & y_str.notna()  # NaN in result but not in original
    
    if nan_from_invalid.any():
        # Original values that were converted to NaN (invalid)
        invalid_original_values = y_str[nan_from_invalid].unique()
        
        if verbose:
            invalid_count = nan_from_invalid.sum()
            print(f"\n⚠ WARNING: Found {invalid_count} invalid target values:")
            for val in invalid_original_values:
                count = (y_str == val).sum()
                print(f"  - '{val}': {count} occurrences")
            
            nan_count = y_binary.isna().sum()
            print(f"  → Converted {nan_count} invalid values to NaN")
    
    # Convert to int (keeping NaN as NaN)
    y = y_binary.astype('Int64')  # pandas Int64 allows NaN
    
    if verbose:
        print(f"\nTarget after encoding:")
        encoded_counts = y.value_counts(dropna=False)
        for value, count in encoded_counts.items():
            percentage = (count / len(y)) * 100
            label = 'RED (1)' if value == 1 else 'GOOD (0)' if value == 0 else 'NaN'
            print(f"  - {label}: {count} ({percentage:.2f}%)")
    
    # Separar features (X): todas as colunas exceto target
    X = df.drop(columns=[target_column])
    
    if verbose:
        print(f"\nFeatures (X):")
        print(f"  - Shape: {X.shape}")
        print(f"  - Columns: {X.shape[1]}")
        print(f"  - Data types: {X.dtypes.value_counts().to_dict()}")
        
        # Check if there are NaN in features
        nan_counts = X.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        if len(cols_with_nan) > 0:
            print(f"\n  Features with missing values: {len(cols_with_nan)}")
            if len(cols_with_nan) <= 10:
                for col, nan_count in cols_with_nan.items():
                    percentage = (nan_count / len(X)) * 100
                    print(f"    - {col}: {nan_count} ({percentage:.2f}%)")
            else:
                print(f"    (Showing first 10)")
                for col, nan_count in cols_with_nan.head(10).items():
                    percentage = (nan_count / len(X)) * 100
                    print(f"    - {col}: {nan_count} ({percentage:.2f}%)")
        
        print(f"\nTarget (y):")
        print(f"  - Shape: {y.shape}")
        print(f"  - Type: {y.dtype}")
        nan_in_y = y.isna().sum()
        if nan_in_y > 0:
            print(f"  - Missing values: {nan_in_y} ({(nan_in_y/len(y)*100):.2f}%)")
    
    if verbose:
        print("\n✓ Features and target prepared successfully!\n")
    
    return X, y



def split_data(X, y, test_size=0.05, random_state=42, verbose=None):
    """
    Splits data into training and test sets.
    
    Case requirement: 5% for test, maintaining class proportions (stratified).
    
    Args:
        X: Features
        y: Target
        test_size: Proportion for test (default 0.05 = 5%)
        random_state: Seed for reproducibility
        verbose: If True, shows split information (default: None, uses global VERBOSE)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Use global VERBOSE if verbose is None
    if verbose is None:
        verbose = VERBOSE
    
    if verbose:
        print(f"Original dataset shape: {X.shape}")
        print(f"Test size: {test_size*100:.1f}%")
        print(f"Random state: {random_state}")
    
    nan_mask = y.isna()
    nan_count = nan_mask.sum()

    if nan_count > 0:
        valid_mask = ~nan_mask
        X = X[valid_mask].copy()
        y = y[valid_mask].copy()
        if verbose:
            print(f"\n⚠ Removed {nan_count} rows with NaN in target")
            print(f"Dataset shape after removing NaN: {X.shape}")
    
    unique_classes = y.value_counts()
    n_classes = len(unique_classes)

    use_stratify = True
    if n_classes < 2 or (unique_classes < 2).any():
        use_stratify = False
    
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
    
    if verbose:
        print(f"\nSplit completed:")
        print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}%)")
        print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.1f}%)")
        print(f"  Features: {X_train.shape[1]} columns")
        
        # Show class distribution
        print(f"\nClass distribution:")
        train_class_counts = pd.Series(y_train).value_counts().sort_index()
        test_class_counts = pd.Series(y_test).value_counts().sort_index()
        
        print(f"  Training set:")
        for class_val, count in train_class_counts.items():
            percentage = (count / len(y_train)) * 100
            label = 'GOOD (0)' if class_val == 0 else 'RED (1)' if class_val == 1 else f'Class {class_val}'
            print(f"    - {label}: {count} ({percentage:.2f}%)")
        
        print(f"  Test set:")
        for class_val, count in test_class_counts.items():
            percentage = (count / len(y_test)) * 100
            label = 'GOOD (0)' if class_val == 0 else 'RED (1)' if class_val == 1 else f'Class {class_val}'
            print(f"    - {label}: {count} ({percentage:.2f}%)")
        
        if use_stratify:
            print(f"\n✓ Stratified split: Class proportions maintained")
        else:
            print(f"\n⚠ Non-stratified split: Class proportions may vary")
        
        print()
    
    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train, X_test, y_test, focus_precision=True, verbose=None):
    """
    Trains XGBoost model focusing on precision.
    
    Case requirement: Precision >50% (prioritize precision over recall).
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data (for early stopping)
        focus_precision: If True, adjusts to maximize precision
        verbose: If True, shows training information (default: None, uses global VERBOSE)
        
    Returns:
        Trained model and training history
    """
    # Use global VERBOSE if verbose is None
    if verbose is None:
        verbose = VERBOSE
    
    if verbose:
        print("="*80)
        print("STEP 5: Training XGBoost model")
        print("="*80)
        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples (for early stopping)")
        print(f"Focus on precision: {focus_precision}")
    # Check if there are NaN in target
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Target (y) should not contain NaN values. Use split_data to remove them first.")
    
    # Convert to numpy arrays if necessary (XGBoost accepts both)
    if isinstance(X_train, pd.DataFrame):
        X_train_array = X_train.values
        X_test_array = X_test.values
        # Clean feature names (XGBoost doesn't accept [, ], <, etc)
        feature_names = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') 
                        for col in X_train.columns]
    else:
        X_train_array = X_train
        X_test_array = X_test
        feature_names = None
    
    if isinstance(y_train, pd.Series):
        y_train_array = y_train.values
        y_test_array = y_test.values
    else:
        y_train_array = y_train
        y_test_array = y_test
    
    # Calculate class proportion for scale_pos_weight (if imbalanced)
    class_counts = pd.Series(y_train_array).value_counts().sort_index()
    if len(class_counts) == 2:
        negative_class_count = class_counts.iloc[0]  # Class 0 (GOOD)
        positive_class_count = class_counts.iloc[1]  # Class 1 (RED)
        scale_pos_weight = negative_class_count / positive_class_count if positive_class_count > 0 else 1.0
        if verbose:
            print(f"\nClass distribution in training set:")
            print(f"  - GOOD (0): {negative_class_count} ({negative_class_count/len(y_train_array)*100:.2f}%)")
            print(f"  - RED (1): {positive_class_count} ({positive_class_count/len(y_train_array)*100:.2f}%)")
            print(f"  - Scale pos weight: {scale_pos_weight:.3f}")
    else:
        scale_pos_weight = 1.0
        if verbose:
            print(f"\nClass distribution: {dict(class_counts)}")
            print(f"  - Scale pos weight: {scale_pos_weight:.3f} (balanced)")
    
    # Configure base hyperparameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # Adjust parameters if focus_precision=True (more conservative = better precision)
    if focus_precision:
        params.update({
            'min_child_weight': 7,  # Increased from 3 to 7 (more conservative)
            'gamma': 0.3,  # Increased from 0.1 to 0.3 (more conservative)
            'max_depth': 4,  # Reduced from 6 to 4 (less overfitting)
            'learning_rate': 0.03,  # Reduced from 0.1 to 0.03 (more careful)
            'reg_alpha': 0.2,  # Increased L1 regularization
            'reg_lambda': 2.0,  # Increased L2 regularization
        })
    
    if verbose:
        print(f"\nHyperparameters:")
        key_params = ['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight', 
                     'gamma', 'reg_alpha', 'reg_lambda', 'subsample', 'colsample_bytree']
        for key in key_params:
            if key in params:
                print(f"  - {key}: {params[key]}")
        print(f"  - early_stopping_rounds: 50")
        print(f"\nStarting training...")
    
    # Create DMatrix (XGBoost optimized format)
    dtrain = xgb.DMatrix(X_train_array, label=y_train_array, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test_array, label=y_test_array, feature_names=feature_names)
    
    # List of sets for evaluation during training
    evallist = [(dtrain, 'train'), (dtest, 'test')]
    
    # Train model with early stopping
    if verbose:
        print("  (Training in progress, this may take a few minutes...)")
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=evallist,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    if verbose:
        print(f"\n✓ Training completed!")
        print(f"  - Best iteration: {model.best_iteration}")
        print(f"  - Best score (logloss): {model.best_score:.6f}")
    
    # Capture training history
    training_history = {
        'best_iteration': model.best_iteration,
        'best_score': model.best_score,
        'scale_pos_weight': scale_pos_weight,
        'params': params
    }
    
    return model, training_history


def evaluate_model(model, X_train, y_train, X_test, y_test, threshold=None, verbose=None):
    """
    Evaluates model by calculating performance metrics.
    
    Calculated metrics:
    - Accuracy: Proportion of correct predictions
    - Precision: Of those predicted as RED, how many are actually RED?
    - Recall: Of the actual RED cases, how many were predicted?
    
    Case requirement: Focus on precision >50%
    
    Args:
        model: Trained model (XGBoost Booster)
        X_train, y_train: Training data
        X_test, y_test: Test data
        threshold: Threshold for binary classification (default: 0.5)
        verbose: If True, shows calculated metrics (default: None, uses global VERBOSE)
        
    Returns:
        Dict with training and test metrics
    """
    # Use global VERBOSE if verbose is None
    if verbose is None:
        verbose = VERBOSE
    
    if threshold is None:
        threshold = 0.5
    
    if verbose:
        print("="*80)
        print("STEP 6: Evaluating model performance")
        print("="*80)
        print(f"Classification threshold: {threshold}")
    
    # Convert to arrays if necessary
    if isinstance(X_train, pd.DataFrame):
        X_train_array = X_train.values
        X_test_array = X_test.values
        feature_names = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') 
                        for col in X_train.columns]
    else:
        X_train_array = X_train
        X_test_array = X_test
        feature_names = None
    
    if isinstance(y_train, pd.Series):
        y_train_array = y_train.values
        y_test_array = y_test.values
    else:
        y_train_array = y_train
        y_test_array = y_test
    
    # Create DMatrix for predictions
    dtrain = xgb.DMatrix(X_train_array, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test_array, feature_names=feature_names)
    
    # Get probabilities
    y_train_proba = model.predict(dtrain)
    y_test_proba = model.predict(dtest)
    
    # Convert to binary predictions using threshold
    y_train_pred = (y_train_proba >= threshold).astype(int)
    y_test_pred = (y_test_proba >= threshold).astype(int)
    
    # Calculate metrics for training
    train_accuracy = accuracy_score(y_train_array, y_train_pred)
    train_precision = precision_score(y_train_array, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train_array, y_train_pred, zero_division=0)
    
    # Calculate metrics for test
    test_accuracy = accuracy_score(y_test_array, y_test_pred)
    test_precision = precision_score(y_test_array, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test_array, y_test_pred, zero_division=0)
    
    # Create dictionary with metrics
    metrics = {
        'train': {
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall
        },
        'test': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall
        },
        'threshold': threshold
    }
    
    if verbose:
        print(f"\nTraining Set Metrics:")
        print(f"  - Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  - Precision: {train_precision:.4f} ({train_precision*100:.2f}%)")
        print(f"  - Recall:    {train_recall:.4f} ({train_recall*100:.2f}%)")
        
        print(f"\nTest Set Metrics:")
        print(f"  - Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  - Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
        print(f"  - Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
        
        # Check case requirement
        print(f"\nCase Requirement Check:")
        if test_precision > 0.5:
            print(f"  ✓ Precision ({test_precision*100:.2f}%) > 50% - REQUIREMENT MET")
        else:
            print(f"  ✗ Precision ({test_precision*100:.2f}%) ≤ 50% - REQUIREMENT NOT MET")
        
        # Calculate differences (overfitting indicator)
        accuracy_diff = train_accuracy - test_accuracy
        precision_diff = train_precision - test_precision
        recall_diff = train_recall - test_recall
        
        print(f"\nTrain vs Test Differences (overfitting indicator):")
        print(f"  - Accuracy gap:  {accuracy_diff:+.4f} ({accuracy_diff*100:+.2f}%)")
        print(f"  - Precision gap: {precision_diff:+.4f} ({precision_diff*100:+.2f}%)")
        print(f"  - Recall gap:    {recall_diff:+.4f} ({recall_diff*100:+.2f}%)")
        
        if abs(accuracy_diff) > 0.1 or abs(precision_diff) > 0.1 or abs(recall_diff) > 0.1:
            print(f"  ⚠ Warning: Large gap (>10%) may indicate overfitting")
        else:
            print(f"  ✓ Good generalization: Small gap between train and test")
        
        print()
    
    return metrics


def get_feature_importance(model, feature_names=None, top_n=15):
    """
    Extracts and shows the most important features of the model.
    
    Args:
        model: Trained XGBoost model (XGBoost Booster)
        feature_names: List of feature names (optional, if None tries to get from model)
        top_n: Number of top features to show
        
    Returns:
        DataFrame with features sorted by importance
    """
    # Get feature importance from model
    importance_dict = model.get_score(importance_type='weight')
    
    # If feature_names not provided, try to get from model
    if feature_names is None:
        try:
            # Try to get feature names from model
            booster = model.get_booster()
            feature_names = booster.feature_names
            if feature_names is None or len(feature_names) == 0:
                # If can't get, use keys from importance_dict
                feature_names = list(importance_dict.keys())
        except:
            # If can't get, use keys from importance_dict
            feature_names = list(importance_dict.keys())
    
    # Create list of (feature_name, importance)
    importance_list = []
    
    # If we have feature_names, map to importance_dict
    if len(feature_names) > 0:
        for i, feat_name in enumerate(feature_names):
            # XGBoost may use cleaned names (without special characters) or f0, f1, etc.
            feat_key_clean = str(feat_name).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            # Try different key formats
            importance_value = (
                importance_dict.get(feat_key_clean) or
                importance_dict.get(feat_name) or
                importance_dict.get(f'f{i}') or
                importance_dict.get(str(i)) or
                0
            )
            importance_list.append({
                'feature': feat_name,
                'importance': importance_value
            })
    else:
        # If we don't have feature_names, use keys from importance_dict directly
        for feat_key, importance_value in importance_dict.items():
            importance_list.append({
                'feature': feat_key,
                'importance': importance_value
            })
    
    # Create DataFrame
    df_importance = pd.DataFrame(importance_list)
    
    # Sort by importance (descending)
    df_importance = df_importance.sort_values('importance', ascending=False)
    
    # Return only top_n if specified
    if top_n is not None and top_n > 0:
        df_importance = df_importance.head(top_n)
    
    # Reset index
    df_importance = df_importance.reset_index(drop=True)
    
    return df_importance


def check_overfitting(metrics, verbose=False):
    """
    Checks for overfitting by comparing performance on training vs test sets.
    
    Overfitting is detected when:
    - Training metrics are significantly better than test metrics
    - Difference > 10% usually indicates overfitting
    
    Args:
        metrics: Dict with training and test metrics (returned by evaluate_model)
        verbose: If True, prints detailed analysis
        
    Returns:
        Dict with overfitting information
    """
    train_metrics = metrics.get('train', {})
    test_metrics = metrics.get('test', {})
    
    # Calculate differences
    accuracy_diff = train_metrics.get('accuracy', 0) - test_metrics.get('accuracy', 0)
    precision_diff = train_metrics.get('precision', 0) - test_metrics.get('precision', 0)
    recall_diff = train_metrics.get('recall', 0) - test_metrics.get('recall', 0)
    
    # Threshold to consider overfitting (10%)
    overfitting_threshold = 0.10
    
    # Check if there is overfitting
    has_overfitting = (
        accuracy_diff > overfitting_threshold or
        abs(precision_diff) > overfitting_threshold or
        abs(recall_diff) > overfitting_threshold
    )
    
    # Determine severity
    severity = 'Good (no overfitting)'
    if has_overfitting:
        max_diff = max(abs(accuracy_diff), abs(precision_diff), abs(recall_diff))
        if max_diff > 0.20:
            severity = 'Severe'
        elif max_diff > 0.15:
            severity = 'Moderate'
        else:
            severity = 'Mild'
    
    overfitting_info = {
        'has_overfitting': has_overfitting,
        'severity': severity,
        'differences': {
            'accuracy': accuracy_diff,
            'precision': precision_diff,
            'recall': recall_diff
        },
        'threshold': overfitting_threshold
    }
    
    # Print analysis if verbose=True
    if verbose:
        print("="*80)
        print("OVERFITTING ANALYSIS")
        print("="*80)
        print(f"\nOverfitting Status: {severity.upper()}")
        if not has_overfitting:
            print("✅ No overfitting detected - Model generalizes well!")
        else:
            print(f"⚠️  Overfitting detected ({severity} severity)")
        
        print(f"\nDifferences (Train - Test):")
        print(f"  Accuracy:  {accuracy_diff*100:+.2f}%")
        print(f"  Precision: {precision_diff*100:+.2f}%")
        print(f"  Recall:    {recall_diff*100:+.2f}%")
        print("="*80)
    
    return overfitting_info


def get_predictions(model, X, feature_names=None, threshold=0.5):
    """
    Helper function to make predictions with XGBoost model.
    
    This function encapsulates the prediction process, avoiding code duplication
    in the notebook. Returns both probabilities and binary predictions.
    
    Args:
        model: Trained XGBoost model (XGBoost Booster)
        X: Features (DataFrame or numpy array)
        feature_names: List of feature names (optional, if None tries to get from X)
        threshold: Threshold for binary classification (default: 0.5)
        
    Returns:
        y_pred_proba: Array with predicted probabilities (probability of class 1 = RED)
        y_pred: Array with binary predictions (0 = GOOD, 1 = RED)
    """
    # Convert to arrays if necessary
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        if feature_names is None:
            # Clean feature names (XGBoost doesn't accept [, ], <, etc)
            feature_names = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') 
                            for col in X.columns]
    else:
        X_array = X
        # If feature_names not provided and X is not DataFrame, try to get from model
        if feature_names is None:
            try:
                feature_names = model.feature_names
            except:
                feature_names = None
    
    # Clean feature names if provided
    if feature_names is not None:
        feature_names_clean = [str(feat).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') 
                              for feat in feature_names]
    else:
        feature_names_clean = None
    
    # Create DMatrix for predictions
    dtest = xgb.DMatrix(X_array, feature_names=feature_names_clean)
    
    # Get probabilities
    y_pred_proba = model.predict(dtest)
    
    # Convert to binary predictions using threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return y_pred_proba, y_pred


def save_processed_data(df, output_path='data/processed/cleaned_data.csv', verbose=None):
    """
    Saves processed data to CSV file.
    
    Useful to avoid reprocessing large data every time the notebook is run.
    
    Args:
        df: Processed DataFrame
        output_path: Path to save (default: 'data/processed/cleaned_data.csv')
        verbose: If True, shows messages (default: None, uses global VERBOSE)
        
    Returns:
        Path of saved file
    """
    import os
    
    if verbose is None:
        verbose = VERBOSE
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Created directory: {output_dir}")
    
    # Save CSV (use same separator as original file: ';')
    df.to_csv(output_path, index=False, sep=';')
    
    if verbose:
        print(f"✓ Processed data saved to: {output_path}")
        print(f"  Shape: {df.shape}")
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"  File size: {file_size:.2f} MB")
    
    return output_path
