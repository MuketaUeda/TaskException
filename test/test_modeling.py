import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modeling import (
    load_and_prepare_data,
    encode_categorical_features,
    prepare_features_and_target,
    split_data,
    train_xgboost,
    evaluate_model,
    get_feature_importance,
    check_overfitting,
    VERBOSE
)


class TestLoadAndPrepareData:
    """Test suite for load_and_prepare_data function"""
    
    @pytest.fixture
    def csv_path(self):
        """Path to the CSV file"""
        return 'data/raw/Task_Exception Prediction_Training Test Data.csv'
    
    @pytest.fixture
    def raw_df(self, csv_path):
        """Load raw CSV without cleaning"""
        return pd.read_csv(csv_path, sep=';')
    
    @pytest.fixture
    def cleaned_df(self, csv_path):
        """Load and clean data using the function"""
        return load_and_prepare_data(csv_path, verbose=False)
    
    def test_returns_dataframe(self, cleaned_df):
        """Test that function returns a pandas DataFrame"""
        assert isinstance(cleaned_df, pd.DataFrame), "Function should return a pandas DataFrame"
    
    def test_no_rows_lost(self, raw_df, cleaned_df):
        """Test that no rows are lost during cleaning"""
        assert cleaned_df.shape[0] == raw_df.shape[0], \
            f"Lost rows: {raw_df.shape[0]} -> {cleaned_df.shape[0]}"
    
    def test_no_columns_lost(self, raw_df, cleaned_df):
        """Test that no columns are lost during cleaning"""
        assert cleaned_df.shape[1] == raw_df.shape[1], \
            f"Lost columns: {raw_df.shape[1]} -> {cleaned_df.shape[1]}"
    
    def test_numeric_columns_converted(self, raw_df, cleaned_df):
        """Test that numeric columns are converted from string to numeric"""
        numeric_columns = [
            'Gross_weight [kg]',
            'Volume [m3]',
            'Loading_meter [ldm]',
            'Handling_unit_quantity [qty]',
            'Billed freight weight [kg]'
        ]
        
        for col in numeric_columns:
            if col in cleaned_df.columns:
                # Check if column is now numeric
                assert pd.api.types.is_numeric_dtype(cleaned_df[col]), \
                    f"Column {col} should be numeric, got {cleaned_df[col].dtype}"
                
                # Check if values were actually converted (not all NaN)
                non_nan_count = cleaned_df[col].notna().sum()
                assert non_nan_count > 0, \
                    f"Column {col} should have non-NaN values after conversion"
    
    def test_numeric_columns_contain_float_values(self, cleaned_df):
        """Test that numeric columns contain actual numeric values"""
        numeric_col = 'Gross_weight [kg]'
        
        if numeric_col in cleaned_df.columns:
            # Check that first non-null value is numeric
            first_value = cleaned_df[numeric_col].dropna().iloc[0]
            assert isinstance(first_value, (int, float, np.integer, np.floating)), \
                f"First value in {numeric_col} should be numeric, got {type(first_value)}"
    
    def test_text_columns_normalized(self, raw_df, cleaned_df):
        """Test that text columns are normalized (lowercase, strip)"""
        text_columns = [
            'Means_of_transportation',
            'Mode_of_Transportation',
            'Consignor_country',
            'Recipient_country'
        ]
        
        for col in text_columns:
            if col in cleaned_df.columns and col in raw_df.columns:
                # Check that text is normalized (lowercase)
                cleaned_sample = str(cleaned_df[col].dropna().iloc[0])
                assert cleaned_sample == cleaned_sample.lower(), \
                    f"Column {col} should be lowercase, got '{cleaned_sample}'"
                
                # Check that text doesn't have leading/trailing spaces
                assert cleaned_sample == cleaned_sample.strip(), \
                    f"Column {col} should have no leading/trailing spaces, got '{cleaned_sample}'"
    
    def test_exception_output_column_exists(self, cleaned_df):
        """Test that Exception_output column exists (target variable)"""
        assert 'Exception_output' in cleaned_df.columns, \
            "Exception_output column should exist"
    
    def test_exception_output_has_valid_values(self, cleaned_df):
        """Test that Exception_output has only 'red' or 'good' values"""
        if 'Exception_output' in cleaned_df.columns:
            unique_values = cleaned_df['Exception_output'].str.lower().unique()
            valid_values = {'red', 'good', 'nan', np.nan}
            
            # Check that all values are valid (convert to lowercase for comparison)
            for value in unique_values:
                if pd.notna(value):
                    assert str(value).lower() in {'red', 'good'}, \
                        f"Exception_output should only contain 'red' or 'good', got '{value}'"
    
    def test_no_all_nan_columns(self, cleaned_df):
        """Test that no columns are completely NaN after cleaning"""
        for col in cleaned_df.columns:
            non_nan_count = cleaned_df[col].notna().sum()
            assert non_nan_count > 0, \
                f"Column {col} should not be completely NaN"
    
    def test_verbose_parameter(self, csv_path, capsys):
        """Test that verbose parameter controls output"""
        # Test with verbose=False (should not print)
        load_and_prepare_data(csv_path, verbose=False)
        output_with_false = capsys.readouterr().out
        
        # Test with verbose=True (should print)
        load_and_prepare_data(csv_path, verbose=True)
        output_with_true = capsys.readouterr().out
        
        # Output with verbose=True should be longer
        assert len(output_with_true) > len(output_with_false), \
            "Verbose=True should produce more output than verbose=False"
    
    def test_verbose_uses_global_when_none(self, csv_path, capsys, monkeypatch):
        """Test that verbose uses global VERBOSE when None"""
        # Set global VERBOSE to False
        import src.modeling as modeling
        monkeypatch.setattr(modeling, 'VERBOSE', False)
        
        load_and_prepare_data(csv_path, verbose=None)
        output = capsys.readouterr().out
        
        # Should not print when VERBOSE=False
        assert len(output) == 0 or "STEP 1" not in output, \
            "Should not print when VERBOSE=False and verbose=None"
    
    def test_shape_consistency(self, raw_df, cleaned_df):
        """Test that shape is consistent before and after cleaning"""
        assert cleaned_df.shape == raw_df.shape, \
            f"Shape should be consistent: {raw_df.shape} -> {cleaned_df.shape}"


# ============================================================================
# CLASSE 2: Testes para encode_categorical_features()
# ============================================================================
class TestEncodeCategoricalFeatures:
    """Test suite for encode_categorical_features function"""
    
    @pytest.fixture
    def csv_path(self):
        """Path to the CSV file"""
        return 'data/raw/Task_Exception Prediction_Training Test Data.csv'
    
    @pytest.fixture
    def cleaned_df(self, csv_path):
        """Fixture: DataFrame limpo (sem encoding)"""
        return load_and_prepare_data(csv_path, verbose=False)
    
    @pytest.fixture
    def encoded_result(self, cleaned_df):
        """Fixture: Resultado do encoding (df, encoders_dict)"""
        return encode_categorical_features(cleaned_df, verbose=False)
    
    @pytest.fixture
    def encoded_df(self, encoded_result):
        """Fixture: DataFrame encodado"""
        return encoded_result[0]
    
    @pytest.fixture
    def encoders_dict(self, encoded_result):
        """Fixture: Dicionário de encoders"""
        return encoded_result[1]
    
    def test_returns_tuple(self, cleaned_df):
        """Test that function returns tuple (df, encoders_dict)"""
        result = encode_categorical_features(cleaned_df, verbose=False)
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Tuple should have 2 elements"
    
    def test_returns_dataframe_and_dict(self, cleaned_df):
        """Test that function returns DataFrame and dict"""
        df_encoded, encoders = encode_categorical_features(cleaned_df, verbose=False)
        assert isinstance(df_encoded, pd.DataFrame), "First element should be DataFrame"
        assert isinstance(encoders, dict), "Second element should be dict"
    
    def test_no_rows_lost(self, cleaned_df, encoded_df):
        """Test that no rows are lost during encoding"""
        assert encoded_df.shape[0] == cleaned_df.shape[0], \
            f"Lost rows: {cleaned_df.shape[0]} -> {encoded_df.shape[0]}"
    
    def test_no_columns_lost(self, cleaned_df, encoded_df):
        """Test that no columns are lost during encoding"""
        assert encoded_df.shape[1] == cleaned_df.shape[1], \
            f"Lost columns: {cleaned_df.shape[1]} -> {encoded_df.shape[1]}"
    
    def test_shape_consistency(self, cleaned_df, encoded_df):
        """Test that shape is consistent before and after encoding"""
        assert encoded_df.shape == cleaned_df.shape, \
            f"Shape should be consistent: {cleaned_df.shape} -> {encoded_df.shape}"
    
    def test_categorical_columns_encoded(self, cleaned_df, encoded_df, encoders_dict):
        """Test that categorical columns are encoded (not object type)"""
        target_column = 'Exception_output'
        
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object' and col != target_column:
                # Deve estar encodada (não mais object)
                assert pd.api.types.is_numeric_dtype(encoded_df[col]) or encoded_df[col].dtype == 'float64', \
                    f"Column {col} should be numeric after encoding, got {encoded_df[col].dtype}"
                # Deve estar no dict de encoders
                assert col in encoders_dict, \
                    f"Column {col} should be in encoders dict"
    
    def test_exception_output_not_encoded(self, cleaned_df, encoded_df, encoders_dict):
        """Test that Exception_output is NOT encoded"""
        target_column = 'Exception_output'
        
        if target_column in encoded_df.columns:
            assert encoded_df[target_column].dtype == 'object', \
                "Exception_output should still be object (not encoded)"
            assert target_column not in encoders_dict, \
                "Exception_output should not be in encoders dict"
    
    def test_encoders_dict_structure(self, encoders_dict, cleaned_df):
        """Test that encoders dict has correct structure"""
        target_column = 'Exception_output'
        
        # Verificar que encoders_dict tem as colunas categóricas corretas
        categorical_cols = [col for col in cleaned_df.columns 
                           if cleaned_df[col].dtype == 'object' and col != target_column]
        
        assert len(encoders_dict) == len(categorical_cols), \
            f"Encoders dict should have {len(categorical_cols)} encoders, got {len(encoders_dict)}"
        
        # Verificar que cada encoder é um LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        for col, encoder in encoders_dict.items():
            assert isinstance(encoder, LabelEncoder), \
                f"Encoder for {col} should be LabelEncoder instance"
            assert hasattr(encoder, 'classes_'), \
                f"Encoder for {col} should have classes_ attribute"
    
    def test_encoding_preserves_unique_values(self, cleaned_df, encoded_df, encoders_dict):
        """Test that encoding preserves number of unique values"""
        target_column = 'Exception_output'
        
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object' and col != target_column:
                if col in encoders_dict:
                    # Número de valores únicos deve ser igual (ou similar se houver NaN)
                    unique_before = cleaned_df[col].nunique()
                    unique_after = encoded_df[col].nunique()
                    # Pode haver diferença de 1 se houver NaN tratado
                    assert abs(unique_before - unique_after) <= 1, \
                        f"Unique values count should be similar: {unique_before} -> {unique_after}"
    
    def test_encoded_values_are_numeric(self, encoded_df, encoders_dict):
        """Test that encoded values are numeric"""
        for col, encoder in encoders_dict.items():
            if col in encoded_df.columns:
                # Valores encodados devem ser numéricos (inteiros ou NaN)
                sample_values = encoded_df[col].dropna()
                if len(sample_values) > 0:
                    assert all(isinstance(val, (int, float, np.integer, np.floating)) 
                              for val in sample_values), \
                        f"Encoded values in {col} should be numeric"
    
    def test_verbose_parameter(self, cleaned_df, capsys):
        """Test that verbose parameter controls output"""
        # Test with verbose=False (should not print)
        encode_categorical_features(cleaned_df, verbose=False)
        output_with_false = capsys.readouterr().out
        
        # Test with verbose=True (should print)
        encode_categorical_features(cleaned_df, verbose=True)
        output_with_true = capsys.readouterr().out
        
        # Output with verbose=True should be longer
        assert len(output_with_true) > len(output_with_false), \
            "Verbose=True should produce more output than verbose=False"
    
    def test_verbose_uses_global_when_none(self, cleaned_df, capsys, monkeypatch):
        """Test that verbose uses global VERBOSE when None"""
        # Set global VERBOSE to False
        import src.modeling as modeling
        monkeypatch.setattr(modeling, 'VERBOSE', False)
        
        encode_categorical_features(cleaned_df, verbose=None)
        output = capsys.readouterr().out
        
        # Should not print when VERBOSE=False
        assert len(output) == 0 or "STEP 2" not in output, \
            "Should not print when VERBOSE=False and verbose=None"


# ============================================================================
# CLASSE 3: Testes para prepare_features_and_target()
# ============================================================================
class TestPrepareFeaturesAndTarget:
    """Test suite for prepare_features_and_target function"""
    
    @pytest.fixture
    def csv_path(self):
        """Path to the CSV file"""
        return 'data/raw/Task_Exception Prediction_Training Test Data.csv'
    
    @pytest.fixture
    def encoded_df(self, csv_path):
        """Fixture: DataFrame after encoding (ready for feature/target separation)"""
        df_cleaned = load_and_prepare_data(csv_path, verbose=False)
        df_encoded, _ = encode_categorical_features(df_cleaned, verbose=False)
        return df_encoded
    
    @pytest.fixture
    def X_and_y(self, encoded_df):
        """Fixture: Features (X) and target (y) separated"""
        return prepare_features_and_target(encoded_df, verbose=False)
    
    @pytest.fixture
    def X(self, X_and_y):
        """Fixture: Features only"""
        return X_and_y[0]
    
    @pytest.fixture
    def y(self, X_and_y):
        """Fixture: Target only"""
        return X_and_y[1]
    
    def test_returns_tuple(self, encoded_df):
        """Test that function returns tuple (X, y)"""
        result = prepare_features_and_target(encoded_df, verbose=False)
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Tuple should have 2 elements"
    
    def test_returns_dataframe_and_series(self, encoded_df):
        """Test that function returns DataFrame (X) and Series (y)"""
        X, y = prepare_features_and_target(encoded_df, verbose=False)
        assert isinstance(X, pd.DataFrame), "First element should be DataFrame"
        assert isinstance(y, pd.Series), "Second element should be Series"
    
    def test_target_column_not_in_X(self, encoded_df, X):
        """Test that Exception_output column is NOT in features (X)"""
        assert 'Exception_output' not in X.columns, \
            "Exception_output should not be in features (X)"
    
    def test_target_column_removed_from_X(self, encoded_df, X):
        """Test that X has one less column than original DataFrame"""
        assert X.shape[1] == encoded_df.shape[1] - 1, \
            f"X should have {encoded_df.shape[1] - 1} columns, got {X.shape[1]}"
    
    def test_no_rows_lost(self, encoded_df, X, y):
        """Test that no rows are lost during separation"""
        assert X.shape[0] == encoded_df.shape[0], \
            f"Lost rows in X: {encoded_df.shape[0]} -> {X.shape[0]}"
        assert y.shape[0] == encoded_df.shape[0], \
            f"Lost rows in y: {encoded_df.shape[0]} -> {y.shape[0]}"
    
    def test_X_and_y_have_same_number_of_rows(self, X, y):
        """Test that X and y have the same number of rows"""
        assert X.shape[0] == y.shape[0], \
            f"X and y should have same number of rows: X={X.shape[0]}, y={y.shape[0]}"
    
    def test_target_is_binary(self, y):
        """Test that target (y) contains only 0, 1, or NaN"""
        unique_values = y.unique()
        valid_values = {0, 1, np.nan}
        for val in unique_values:
            if pd.notna(val):
                assert val in {0, 1}, \
                    f"Target should only contain 0, 1, or NaN, got {val}"
    
    def test_target_encoding_red_to_one(self, encoded_df):
        """Test that 'red'/'RED' values are encoded to 1"""
        # Create test data with 'red' variations
        test_df = encoded_df.copy()
        if 'Exception_output' in test_df.columns:
            # Find rows with 'red' (case insensitive)
            red_rows = test_df['Exception_output'].astype(str).str.lower().str.strip() == 'red'
            if red_rows.any():
                X, y = prepare_features_and_target(test_df, verbose=False)
                # Check that red values are encoded to 1
                y_red = y[red_rows]
                assert all(y_red.dropna() == 1), \
                    "All 'red' values should be encoded to 1"
    
    def test_target_encoding_good_to_zero(self, encoded_df):
        """Test that 'good'/'GOOD' values are encoded to 0"""
        # Create test data with 'good' variations
        test_df = encoded_df.copy()
        if 'Exception_output' in test_df.columns:
            # Find rows with 'good' (case insensitive)
            good_rows = test_df['Exception_output'].astype(str).str.lower().str.strip() == 'good'
            if good_rows.any():
                X, y = prepare_features_and_target(test_df, verbose=False)
                # Check that good values are encoded to 0
                y_good = y[good_rows]
                assert all(y_good.dropna() == 0), \
                    "All 'good' values should be encoded to 0"
    
    def test_target_case_insensitive(self):
        """Test that target encoding is case insensitive"""
        test_df = pd.DataFrame({
            'Exception_output': ['RED', 'red', 'Good', 'good', 'Red', 'GOOD'],
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [10, 20, 30, 40, 50, 60]
        })
        
        X, y = prepare_features_and_target(test_df, verbose=False)
        
        # All 'red' variations should be 1
        assert y.iloc[0] == 1, "'RED' should be encoded to 1"
        assert y.iloc[1] == 1, "'red' should be encoded to 1"
        assert y.iloc[4] == 1, "'Red' should be encoded to 1"
        
        # All 'good' variations should be 0
        assert y.iloc[2] == 0, "'Good' should be encoded to 0"
        assert y.iloc[3] == 0, "'good' should be encoded to 0"
        assert y.iloc[5] == 0, "'GOOD' should be encoded to 0"
    
    def test_invalid_target_values_become_nan(self):
        """Test that invalid target values are converted to NaN"""
        test_df = pd.DataFrame({
            'Exception_output': ['red', 'good', 'invalid', 'unknown', 'red'],
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        X, y = prepare_features_and_target(test_df, verbose=False)
        
        # Valid values should be encoded
        assert y.iloc[0] == 1, "'red' should be 1"
        assert y.iloc[1] == 0, "'good' should be 0"
        assert y.iloc[4] == 1, "'red' should be 1"
        
        # Invalid values should be NaN
        assert pd.isna(y.iloc[2]), "'invalid' should become NaN"
        assert pd.isna(y.iloc[3]), "'unknown' should become NaN"
    
    def test_target_dtype_is_int64_with_nan(self, y):
        """Test that target dtype is Int64 (pandas nullable integer)"""
        # Int64 (pandas) allows NaN, while int64 (numpy) doesn't
        assert str(y.dtype) == 'Int64' or y.dtype.name == 'Int64', \
            f"Target dtype should be Int64 (pandas nullable integer), got {y.dtype}"
    
    def test_all_features_columns_preserved(self, encoded_df, X):
        """Test that all feature columns (except target) are preserved in X"""
        target_column = 'Exception_output'
        expected_columns = [col for col in encoded_df.columns if col != target_column]
        
        assert set(X.columns) == set(expected_columns), \
            f"X should contain all columns except {target_column}"
    
    def test_missing_target_column_raises_error(self):
        """Test that missing target column raises ValueError"""
        test_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [10, 20, 30]
        })
        
        with pytest.raises(ValueError, match="Target column 'Exception_output' not found"):
            prepare_features_and_target(test_df, verbose=False)
    
    def test_X_has_correct_shape(self, encoded_df, X):
        """Test that X has correct shape (rows, columns-1)"""
        expected_shape = (encoded_df.shape[0], encoded_df.shape[1] - 1)
        assert X.shape == expected_shape, \
            f"X shape should be {expected_shape}, got {X.shape}"
    
    def test_y_has_correct_shape(self, encoded_df, y):
        """Test that y has correct shape (rows,)"""
        expected_shape = (encoded_df.shape[0],)
        assert y.shape == expected_shape, \
            f"y shape should be {expected_shape}, got {y.shape}"
    
    def test_target_distribution_preserved(self, encoded_df):
        """Test that target distribution is preserved (red→1, good→0)"""
        if 'Exception_output' in encoded_df.columns:
            # Count original distribution
            original_counts = encoded_df['Exception_output'].value_counts()
            
            X, y = prepare_features_and_target(encoded_df, verbose=False)
            
            # Count encoded distribution
            encoded_counts = y.value_counts(dropna=False)
            
            # Red should map to 1
            red_count_original = original_counts.get('red', 0)
            red_count_lowercase = sum(
                encoded_df['Exception_output'].astype(str).str.lower().str.strip() == 'red'
            )
            red_count_encoded = encoded_counts.get(1, 0)
            
            # Good should map to 0
            good_count_original = original_counts.get('good', 0)
            good_count_lowercase = sum(
                encoded_df['Exception_output'].astype(str).str.lower().str.strip() == 'good'
            )
            good_count_encoded = encoded_counts.get(0, 0)
            
            # Total should match (accounting for NaN from invalid values)
            total_original = len(encoded_df)
            total_encoded_valid = encoded_counts.get(0, 0) + encoded_counts.get(1, 0)
            total_encoded_nan = encoded_counts.get(np.nan, 0) if pd.isna(encoded_counts.index).any() else 0
            
            assert total_original == total_encoded_valid + total_encoded_nan, \
                "Total counts should match (valid + NaN)"
    
    def test_verbose_parameter(self, encoded_df, capsys):
        """Test that verbose parameter controls output"""
        # Test with verbose=False (should not print)
        prepare_features_and_target(encoded_df, verbose=False)
        output_with_false = capsys.readouterr().out
        
        # Test with verbose=True (should print)
        prepare_features_and_target(encoded_df, verbose=True)
        output_with_true = capsys.readouterr().out
        
        # Output with verbose=True should be longer
        assert len(output_with_true) > len(output_with_false), \
            "Verbose=True should produce more output than verbose=False"
    
    def test_verbose_uses_global_when_none(self, encoded_df, capsys, monkeypatch):
        """Test that verbose uses global VERBOSE when None"""
        # Set global VERBOSE to False
        import src.modeling as modeling
        monkeypatch.setattr(modeling, 'VERBOSE', False)
        
        prepare_features_and_target(encoded_df, verbose=None)
        output = capsys.readouterr().out
        
        # Should not print when VERBOSE=False
        assert len(output) == 0 or "STEP 3" not in output, \
            "Should not print when VERBOSE=False and verbose=None"


# ============================================================================
# CLASSE 4: Testes para split_data()
# ============================================================================
class TestSplitData:
    """Test suite for split_data function"""
    
    @pytest.fixture
    def sample_X(self):
        """Fixture: Sample feature DataFrame"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000)
        })
    
    @pytest.fixture
    def sample_y_balanced(self):
        """Fixture: Balanced target with 2 classes"""
        return pd.Series([0] * 500 + [1] * 500, dtype='Int64')
    
    @pytest.fixture
    def sample_y_imbalanced(self):
        """Fixture: Imbalanced target (80% class 0, 20% class 1)"""
        return pd.Series([0] * 800 + [1] * 200, dtype='Int64')
    
    @pytest.fixture
    def sample_y_with_nan(self):
        """Fixture: Target with NaN values"""
        y = pd.Series([0] * 800 + [1] * 200, dtype='Int64')
        # Add NaN values at random positions
        y.iloc[0:10] = np.nan
        return y
    
    @pytest.fixture
    def sample_y_single_class(self):
        """Fixture: Target with only one class"""
        return pd.Series([0] * 1000, dtype='Int64')
    
    @pytest.fixture
    def sample_y_small(self):
        """Fixture: Small target for edge case testing"""
        return pd.Series([0, 1], dtype='Int64')
    
    def test_returns_tuple(self, sample_X, sample_y_balanced):
        """Test that function returns tuple with 4 elements"""
        result = split_data(sample_X, sample_y_balanced)
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 4, "Tuple should have 4 elements"
    
    def test_returns_correct_types(self, sample_X, sample_y_balanced):
        """Test that function returns correct types (DataFrame and Series)"""
        X_train, X_test, y_train, y_test = split_data(sample_X, sample_y_balanced)
        
        assert isinstance(X_train, pd.DataFrame), "X_train should be DataFrame"
        assert isinstance(X_test, pd.DataFrame), "X_test should be DataFrame"
        assert isinstance(y_train, pd.Series), "y_train should be Series"
        assert isinstance(y_test, pd.Series), "y_test should be Series"
    
    def test_default_test_size_is_5_percent(self, sample_X, sample_y_balanced):
        """Test that default test_size is 5%"""
        X_train, X_test, y_train, y_test = split_data(sample_X, sample_y_balanced)
        
        total_samples = len(X_train) + len(X_test)
        test_percentage = (len(X_test) / total_samples) * 100
        
        # Should be approximately 5% (allow small tolerance for rounding)
        assert abs(test_percentage - 5.0) < 0.5, \
            f"Test size should be ~5%, got {test_percentage:.2f}%"
    
    def test_no_data_lost(self, sample_X, sample_y_balanced):
        """Test that no data is lost during split (train + test = original)"""
        X_train, X_test, y_train, y_test = split_data(sample_X, sample_y_balanced)
        
        assert len(X_train) + len(X_test) == len(sample_X), \
            "Train + test should equal original X"
        assert len(y_train) + len(y_test) == len(sample_y_balanced), \
            "Train + test should equal original y"
    
    def test_shapes_match(self, sample_X, sample_y_balanced):
        """Test that X and y shapes match correctly"""
        X_train, X_test, y_train, y_test = split_data(sample_X, sample_y_balanced)
        
        assert len(X_train) == len(y_train), "X_train and y_train should have same length"
        assert len(X_test) == len(y_test), "X_test and y_test should have same length"
        assert X_train.shape[1] == sample_X.shape[1], "X_train should have same number of columns"
        assert X_test.shape[1] == sample_X.shape[1], "X_test should have same number of columns"
    
    def test_stratified_split_maintains_class_proportion(self, sample_X, sample_y_imbalanced):
        """Test that stratified split maintains class proportions"""
        X_train, X_test, y_train, y_test = split_data(sample_X, sample_y_imbalanced)
        
        # Calculate original proportions
        original_prop_0 = (sample_y_imbalanced == 0).sum() / len(sample_y_imbalanced)
        original_prop_1 = (sample_y_imbalanced == 1).sum() / len(sample_y_imbalanced)
        
        # Calculate train proportions
        train_prop_0 = (y_train == 0).sum() / len(y_train)
        train_prop_1 = (y_train == 1).sum() / len(y_train)
        
        # Calculate test proportions
        test_prop_0 = (y_test == 0).sum() / len(y_test)
        test_prop_1 = (y_test == 1).sum() / len(y_test)
        
        # Proportions should be approximately the same (within 1% tolerance)
        tolerance = 0.01
        assert abs(train_prop_0 - original_prop_0) < tolerance, \
            f"Train class 0 proportion ({train_prop_0:.3f}) should match original ({original_prop_0:.3f})"
        assert abs(train_prop_1 - original_prop_1) < tolerance, \
            f"Train class 1 proportion ({train_prop_1:.3f}) should match original ({original_prop_1:.3f})"
        assert abs(test_prop_0 - original_prop_0) < tolerance, \
            f"Test class 0 proportion ({test_prop_0:.3f}) should match original ({original_prop_0:.3f})"
        assert abs(test_prop_1 - original_prop_1) < tolerance, \
            f"Test class 1 proportion ({test_prop_1:.3f}) should match original ({original_prop_1:.3f})"
    
    def test_removes_nan_from_target(self, sample_X, sample_y_with_nan):
        """Test that NaN values in target are removed before split"""
        original_len = len(sample_y_with_nan)
        nan_count = sample_y_with_nan.isna().sum()
        
        X_train, X_test, y_train, y_test = split_data(sample_X, sample_y_with_nan)
        
        # Total after split should be original minus NaN
        assert len(y_train) + len(y_test) == original_len - nan_count, \
            "Split should exclude NaN values from target"
        
        # No NaN should remain in train or test
        assert y_train.isna().sum() == 0, "y_train should have no NaN"
        assert y_test.isna().sum() == 0, "y_test should have no NaN"
    
    def test_random_state_produces_same_split(self, sample_X, sample_y_balanced):
        """Test that same random_state produces identical splits"""
        # First split
        X_train1, X_test1, y_train1, y_test1 = split_data(
            sample_X, sample_y_balanced, random_state=42
        )
        
        # Second split with same random_state
        X_train2, X_test2, y_train2, y_test2 = split_data(
            sample_X, sample_y_balanced, random_state=42
        )
        
        # Should produce identical splits
        pd.testing.assert_frame_equal(X_train1, X_train2), "X_train should be identical"
        pd.testing.assert_frame_equal(X_test1, X_test2), "X_test should be identical"
        pd.testing.assert_series_equal(y_train1, y_train2), "y_train should be identical"
        pd.testing.assert_series_equal(y_test1, y_test2), "y_test should be identical"
    
    def test_different_random_state_produces_different_split(self, sample_X, sample_y_balanced):
        """Test that different random_state produces different splits"""
        # First split
        X_train1, X_test1, y_train1, y_test1 = split_data(
            sample_X, sample_y_balanced, random_state=42
        )
        
        # Second split with different random_state
        X_train2, X_test2, y_train2, y_test2 = split_data(
            sample_X, sample_y_balanced, random_state=123
        )
        
        # Should produce different splits (at least one should be different)
        # Note: Very small probability they could be the same, but extremely unlikely
        train_different = not X_train1.equals(X_train2) or not y_train1.equals(y_train2)
        test_different = not X_test1.equals(X_test2) or not y_test1.equals(y_test2)
        
        assert train_different or test_different, \
            "Different random_state should produce different splits"
    
    def test_custom_test_size(self, sample_X, sample_y_balanced):
        """Test that custom test_size works correctly"""
        custom_test_size = 0.2  # 20%
        X_train, X_test, y_train, y_test = split_data(
            sample_X, sample_y_balanced, test_size=custom_test_size
        )
        
        total_samples = len(X_train) + len(X_test)
        test_percentage = (len(X_test) / total_samples) * 100
        
        # Should be approximately 20%
        assert abs(test_percentage - 20.0) < 1.0, \
            f"Test size should be ~20%, got {test_percentage:.2f}%"
    
    def test_handles_single_class_without_error(self, sample_X, sample_y_single_class):
        """Test that function handles single class target without error"""
        # Should not raise an error, but won't use stratified split
        X_train, X_test, y_train, y_test = split_data(sample_X, sample_y_single_class)
        
        # Should still return valid splits
        assert len(X_train) > 0, "Should have training samples"
        assert len(X_test) > 0, "Should have test samples"
        assert len(y_train) > 0, "Should have training targets"
        assert len(y_test) > 0, "Should have test targets"
        
        # All should be class 0
        assert (y_train == 0).all(), "All training samples should be class 0"
        assert (y_test == 0).all(), "All test samples should be class 0"
    
    def test_handles_very_small_dataset(self, sample_X, sample_y_small):
        """Test that function handles very small datasets"""
        # Use only first 2 rows of X to match y
        X_small = sample_X.iloc[:2].copy()
        
        # Should handle small dataset without error
        X_train, X_test, y_train, y_test = split_data(X_small, sample_y_small, test_size=0.5)
        
        # Should still return valid splits
        assert len(X_train) + len(X_test) == len(X_small), \
            "No data should be lost"
        assert len(y_train) + len(y_test) == len(sample_y_small), \
            "No data should be lost"
    
    def test_balanced_split_maintains_balance(self, sample_X, sample_y_balanced):
        """Test that balanced dataset remains balanced after split"""
        X_train, X_test, y_train, y_test = split_data(sample_X, sample_y_balanced)
        
        # Original is 50-50
        # After split, both train and test should remain approximately balanced
        train_balance = (y_train == 0).sum() / len(y_train)
        test_balance = (y_test == 0).sum() / len(y_test)
        
        # Should be approximately 50% (within 5% tolerance due to stratified rounding)
        assert abs(train_balance - 0.5) < 0.05, \
            f"Train should remain balanced (~50%), got {train_balance:.3f}"
        assert abs(test_balance - 0.5) < 0.05, \
            f"Test should remain balanced (~50%), got {test_balance:.3f}"
    
    def test_all_samples_in_split(self, sample_X, sample_y_balanced):
        """Test that all original samples appear in either train or test"""
        X_train, X_test, y_train, y_test = split_data(sample_X, sample_y_balanced)
        
        # Combine train and test indices
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        
        # Should not overlap
        assert len(train_indices & test_indices) == 0, \
            "Train and test indices should not overlap"
        
        # Should cover all original indices
        original_indices = set(sample_X.index)
        combined_indices = train_indices | test_indices
        assert combined_indices == original_indices, \
            "All original indices should appear in train or test"


# ============================================================================
# CLASSE 5: Testes para evaluate_model()
# ============================================================================
class TestEvaluateModel:
    """Test suite for evaluate_model function"""
    
    @pytest.fixture
    def sample_X(self):
        """Fixture: Sample feature DataFrame"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000)
        })
    
    @pytest.fixture
    def sample_y(self):
        """Fixture: Target with 2 classes"""
        return pd.Series([0] * 800 + [1] * 200, dtype='Int64')
    
    @pytest.fixture
    def split_data_fixture(self, sample_X, sample_y):
        """Fixture: Split data into train and test"""
        return split_data(sample_X, sample_y, test_size=0.1, random_state=42)
    
    @pytest.fixture
    def X_train(self, split_data_fixture):
        """Fixture: Training features"""
        return split_data_fixture[0]
    
    @pytest.fixture
    def X_test(self, split_data_fixture):
        """Fixture: Test features"""
        return split_data_fixture[1]
    
    @pytest.fixture
    def y_train(self, split_data_fixture):
        """Fixture: Training target"""
        return split_data_fixture[2]
    
    @pytest.fixture
    def y_test(self, split_data_fixture):
        """Fixture: Test target"""
        return split_data_fixture[3]
    
    @pytest.fixture
    def trained_model(self, X_train, y_train, X_test, y_test):
        """Fixture: Trained XGBoost model"""
        model, _ = train_xgboost(X_train, y_train, X_test, y_test, focus_precision=False)
        return model
    
    def test_returns_dict(self, trained_model, X_train, y_train, X_test, y_test):
        """Test that function returns dictionary"""
        metrics = evaluate_model(trained_model, X_train, y_train, X_test, y_test)
        assert isinstance(metrics, dict), "Should return dictionary"
    
    def test_returns_expected_keys(self, trained_model, X_train, y_train, X_test, y_test):
        """Test that function returns expected keys"""
        metrics = evaluate_model(trained_model, X_train, y_train, X_test, y_test)
        
        assert 'train' in metrics, "Should have 'train' key"
        assert 'test' in metrics, "Should have 'test' key"
        assert 'threshold' in metrics, "Should have 'threshold' key"
    
    def test_train_metrics_structure(self, trained_model, X_train, y_train, X_test, y_test):
        """Test that train metrics have correct structure"""
        metrics = evaluate_model(trained_model, X_train, y_train, X_test, y_test)
        
        train_metrics = metrics['train']
        assert 'accuracy' in train_metrics, "Should have 'accuracy' in train"
        assert 'precision' in train_metrics, "Should have 'precision' in train"
        assert 'recall' in train_metrics, "Should have 'recall' in train"
    
    def test_test_metrics_structure(self, trained_model, X_train, y_train, X_test, y_test):
        """Test that test metrics have correct structure"""
        metrics = evaluate_model(trained_model, X_train, y_train, X_test, y_test)
        
        test_metrics = metrics['test']
        assert 'accuracy' in test_metrics, "Should have 'accuracy' in test"
        assert 'precision' in test_metrics, "Should have 'precision' in test"
        assert 'recall' in test_metrics, "Should have 'recall' in test"
    
    def test_metrics_are_float(self, trained_model, X_train, y_train, X_test, y_test):
        """Test that all metrics are float values between 0 and 1"""
        metrics = evaluate_model(trained_model, X_train, y_train, X_test, y_test)
        
        for split in ['train', 'test']:
            for metric_name in ['accuracy', 'precision', 'recall']:
                value = metrics[split][metric_name]
                assert isinstance(value, (int, float, np.integer, np.floating)), \
                    f"{split}.{metric_name} should be numeric, got {type(value)}"
                assert 0 <= value <= 1, \
                    f"{split}.{metric_name} should be between 0 and 1, got {value}"
    
    def test_threshold_default(self, trained_model, X_train, y_train, X_test, y_test):
        """Test that default threshold is 0.5"""
        metrics = evaluate_model(trained_model, X_train, y_train, X_test, y_test)
        
        assert metrics['threshold'] == 0.5, "Default threshold should be 0.5"
    
    def test_custom_threshold(self, trained_model, X_train, y_train, X_test, y_test):
        """Test that custom threshold works"""
        custom_threshold = 0.7
        metrics = evaluate_model(trained_model, X_train, y_train, X_test, y_test, threshold=custom_threshold)
        
        assert metrics['threshold'] == custom_threshold, \
            f"Threshold should be {custom_threshold}, got {metrics['threshold']}"


# ============================================================================
# CLASSE 6: Testes para get_feature_importance()
# ============================================================================
class TestGetFeatureImportance:
    """Test suite for get_feature_importance function"""
    
    @pytest.fixture
    def sample_X(self):
        """Fixture: Sample feature DataFrame"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000)
        })
    
    @pytest.fixture
    def sample_y(self):
        """Fixture: Target with 2 classes"""
        return pd.Series([0] * 800 + [1] * 200, dtype='Int64')
    
    @pytest.fixture
    def split_data_fixture(self, sample_X, sample_y):
        """Fixture: Split data into train and test"""
        return split_data(sample_X, sample_y, test_size=0.1, random_state=42)
    
    @pytest.fixture
    def X_train(self, split_data_fixture):
        """Fixture: Training features"""
        return split_data_fixture[0]
    
    @pytest.fixture
    def X_test(self, split_data_fixture):
        """Fixture: Test features"""
        return split_data_fixture[1]
    
    @pytest.fixture
    def y_train(self, split_data_fixture):
        """Fixture: Training target"""
        return split_data_fixture[2]
    
    @pytest.fixture
    def y_test(self, split_data_fixture):
        """Fixture: Test target"""
        return split_data_fixture[3]
    
    @pytest.fixture
    def trained_model(self, X_train, y_train, X_test, y_test):
        """Fixture: Trained XGBoost model"""
        model, _ = train_xgboost(X_train, y_train, X_test, y_test, focus_precision=False)
        return model
    
    def test_returns_dataframe(self, trained_model, X_train):
        """Test that function returns pandas DataFrame"""
        df_importance = get_feature_importance(trained_model, feature_names=list(X_train.columns))
        assert isinstance(df_importance, pd.DataFrame), "Should return DataFrame"
    
    def test_returns_expected_columns(self, trained_model, X_train):
        """Test that DataFrame has expected columns"""
        df_importance = get_feature_importance(trained_model, feature_names=list(X_train.columns))
        
        assert 'feature' in df_importance.columns, "Should have 'feature' column"
        assert 'importance' in df_importance.columns, "Should have 'importance' column"
    
    def test_all_features_included(self, trained_model, X_train):
        """Test that all features are included when top_n=None"""
        df_importance = get_feature_importance(trained_model, feature_names=list(X_train.columns), top_n=None)
        
        assert len(df_importance) == X_train.shape[1], \
            f"Should have {X_train.shape[1]} features, got {len(df_importance)}"
    
    def test_top_n_limits_results(self, trained_model, X_train):
        """Test that top_n parameter limits results"""
        top_n = 2
        df_importance = get_feature_importance(trained_model, feature_names=list(X_train.columns), top_n=top_n)
        
        assert len(df_importance) <= top_n, \
            f"Should have at most {top_n} features, got {len(df_importance)}"
    
    def test_sorted_by_importance(self, trained_model, X_train):
        """Test that features are sorted by importance (descending)"""
        df_importance = get_feature_importance(trained_model, feature_names=list(X_train.columns))
        
        if len(df_importance) > 1:
            importances = df_importance['importance'].values
            is_sorted = all(importances[i] >= importances[i+1] for i in range(len(importances)-1))
            
            assert is_sorted, "Features should be sorted by importance (descending)"


# ============================================================================
# CLASSE 7: Testes para check_overfitting()
# ============================================================================
class TestCheckOverfitting:
    """Test suite for check_overfitting function"""
    
    def test_returns_dict(self):
        """Test that function returns dictionary"""
        metrics = {
            'train': {'accuracy': 0.85, 'precision': 0.75, 'recall': 0.80},
            'test': {'accuracy': 0.84, 'precision': 0.74, 'recall': 0.79}
        }
        overfitting_info = check_overfitting(metrics)
        assert isinstance(overfitting_info, dict), "Should return dictionary"
    
    def test_returns_expected_keys(self):
        """Test that function returns expected keys"""
        metrics = {
            'train': {'accuracy': 0.85, 'precision': 0.75, 'recall': 0.80},
            'test': {'accuracy': 0.84, 'precision': 0.74, 'recall': 0.79}
        }
        overfitting_info = check_overfitting(metrics)
        
        assert 'has_overfitting' in overfitting_info, "Should have 'has_overfitting' key"
        assert 'severity' in overfitting_info, "Should have 'severity' key"
        assert 'differences' in overfitting_info, "Should have 'differences' key"
        assert 'threshold' in overfitting_info, "Should have 'threshold' key"
    
    def test_has_overfitting_is_boolean(self):
        """Test that has_overfitting is boolean"""
        metrics = {
            'train': {'accuracy': 0.85, 'precision': 0.75, 'recall': 0.80},
            'test': {'accuracy': 0.84, 'precision': 0.74, 'recall': 0.79}
        }
        overfitting_info = check_overfitting(metrics)
        
        assert isinstance(overfitting_info['has_overfitting'], bool), \
            "has_overfitting should be boolean"
    
    def test_severity_is_valid(self):
        """Test that severity is one of valid values"""
        metrics = {
            'train': {'accuracy': 0.85, 'precision': 0.75, 'recall': 0.80},
            'test': {'accuracy': 0.84, 'precision': 0.74, 'recall': 0.79}
        }
        overfitting_info = check_overfitting(metrics)
        
        valid_severities = ['none', 'mild', 'moderate', 'severe']
        assert overfitting_info['severity'] in valid_severities, \
            f"Severity should be one of {valid_severities}, got {overfitting_info['severity']}"
    
    def test_no_overfitting_when_metrics_similar(self):
        """Test that no overfitting detected when metrics are similar"""
        similar_metrics = {
            'train': {'accuracy': 0.85, 'precision': 0.75, 'recall': 0.80},
            'test': {'accuracy': 0.84, 'precision': 0.74, 'recall': 0.79}
        }
        
        overfitting_info = check_overfitting(similar_metrics)
        assert not overfitting_info['has_overfitting'], \
            "Should not detect overfitting when metrics are similar"
        assert overfitting_info['severity'] == 'none', \
            "Severity should be 'none' when no overfitting"
    
    def test_detects_overfitting_when_large_difference(self):
        """Test that overfitting is detected when there's large difference"""
        overfitting_metrics = {
            'train': {'accuracy': 0.99, 'precision': 0.98, 'recall': 0.99},
            'test': {'accuracy': 0.70, 'precision': 0.65, 'recall': 0.68}
        }
        
        overfitting_info = check_overfitting(overfitting_metrics)
        assert overfitting_info['has_overfitting'], \
            "Should detect overfitting when there's large difference"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
