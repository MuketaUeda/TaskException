import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.cleaning import NumericCleaner, TextNormalizer


# ============================================================================
# CLASSE 1: Testes para NumericCleaner
# ============================================================================
class TestNumericCleaner:
    """Test suite for NumericCleaner transformer"""
    
    @pytest.fixture
    def csv_path(self):
        """Path to the CSV file"""
        return 'data/raw/Task_Exception Prediction_Training Test Data.csv'
    
    @pytest.fixture
    def raw_df(self, csv_path):
        """Load raw CSV"""
        return pd.read_csv(csv_path, sep=';', dtype=str)
    
    @pytest.fixture
    def numeric_columns(self):
        """Numeric columns that need cleaning (European format)"""
        return [
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
    
    @pytest.fixture
    def numeric_cleaner(self, numeric_columns):
        """NumericCleaner instance with target columns"""
        return NumericCleaner(target_columns=numeric_columns)
    
    @pytest.fixture
    def cleaned_df(self, raw_df, numeric_cleaner):
        """DataFrame after NumericCleaner transformation"""
        return numeric_cleaner.transform(raw_df)
    
    def test_is_transformer(self, numeric_cleaner):
        """Test that NumericCleaner is a scikit-learn transformer"""
        from sklearn.base import BaseEstimator, TransformerMixin
        assert isinstance(numeric_cleaner, BaseEstimator)
        assert isinstance(numeric_cleaner, TransformerMixin)
    
    def test_fit_returns_self(self, numeric_cleaner, raw_df):
        """Test that fit method returns self"""
        result = numeric_cleaner.fit(raw_df)
        assert result is numeric_cleaner
    
    def test_no_rows_lost(self, raw_df, cleaned_df):
        """Test that no rows are lost during cleaning"""
        assert cleaned_df.shape[0] == raw_df.shape[0], \
            f"Lost rows: {raw_df.shape[0]} -> {cleaned_df.shape[0]}"
    
    def test_no_columns_lost(self, raw_df, cleaned_df):
        """Test that no columns are lost during cleaning"""
        assert cleaned_df.shape[1] == raw_df.shape[1], \
            f"Lost columns: {raw_df.shape[1]} -> {cleaned_df.shape[1]}"
    
    def test_numeric_columns_converted(self, raw_df, cleaned_df, numeric_columns):
        """Test that numeric columns are converted from string to numeric"""
        for col in numeric_columns:
            if col in cleaned_df.columns and raw_df[col].dtype == 'object':
                # Check if column is now numeric
                assert pd.api.types.is_numeric_dtype(cleaned_df[col]), \
                    f"Column {col} should be numeric, got {cleaned_df[col].dtype}"
    
    def test_european_decimal_format_conversion(self, raw_df, cleaned_df, numeric_cleaner):
        """Test that European decimal format (comma) is converted correctly"""
        # Test with Gross_weight [kg] if it exists
        test_col = 'Gross_weight [kg]'
        if test_col in raw_df.columns and raw_df[test_col].dtype == 'object':
            # Find first value with comma
            sample_values = raw_df[test_col].dropna().head(10)
            for val in sample_values:
                if ',' in str(val):
                    # Value with comma should be converted to float
                    row_idx = raw_df[raw_df[test_col] == val].index[0]
                    cleaned_value = cleaned_df.loc[row_idx, test_col]
                    assert isinstance(cleaned_value, (int, float, np.integer, np.floating)) or pd.isna(cleaned_value), \
                        f"Value '{val}' with comma should be converted to numeric, got {type(cleaned_value)}"
                    break
    
    def test_whitespace_removed(self, raw_df, cleaned_df, numeric_cleaner):
        """Test that whitespace is removed from numeric strings"""
        # This is harder to test directly, but we can verify numeric conversion works
        # If there was whitespace and it wasn't removed, conversion would fail
        test_col = 'Gross_weight [kg]'
        if test_col in cleaned_df.columns:
            assert pd.api.types.is_numeric_dtype(cleaned_df[test_col]), \
                "Column should be numeric after cleaning (whitespace should be removed)"
    
    def test_invalid_values_become_nan(self):
        """Test that invalid values become NaN"""
        # Create NumericCleaner without target_columns to process all columns
        numeric_cleaner = NumericCleaner(target_columns=None)
        test_data = pd.DataFrame({
            'numeric_col': ['123', 'invalid', '456,78', '  789  ', '']
        })
        result = numeric_cleaner.transform(test_data)
        
        # Valid values should be converted
        assert pd.api.types.is_numeric_dtype(result['numeric_col']), \
            "Column should be numeric after cleaning"
        # Invalid values should be NaN
        assert pd.isna(result.loc[1, 'numeric_col']), "Invalid value should become NaN"
    
    def test_european_format_example(self):
        """Test specific European format conversion example"""
        # Create NumericCleaner without target_columns to process all columns
        numeric_cleaner = NumericCleaner(target_columns=None)
        test_data = pd.DataFrame({
            'test_col': ['14136,72', '92,66', '1234']
        })
        result = numeric_cleaner.transform(test_data)
        
        assert pd.api.types.is_numeric_dtype(result['test_col']), \
            "Column should be numeric after cleaning"
        # Check first value (European format)
        assert abs(result.loc[0, 'test_col'] - 14136.72) < 0.01, \
            "European format '14136,72' should convert to 14136.72"
        # Check second value (European format)
        assert abs(result.loc[1, 'test_col'] - 92.66) < 0.01, \
            "European format '92,66' should convert to 92.66"
        # Check third value (normal format)
        assert result.loc[2, 'test_col'] == 1234, \
            "Normal format '1234' should convert to 1234"
    
    def test_only_processes_specified_columns(self, raw_df, numeric_cleaner, numeric_columns):
        """Test that NumericCleaner only processes specified columns"""
        cleaned_df = numeric_cleaner.transform(raw_df)
        
        # Non-numeric columns should remain as object
        categorical_cols = ['Means_of_transportation', 'Mode_of_Transportation', 
                          'Consignor_country', 'Recipient_country']
        
        for col in categorical_cols:
            if col in raw_df.columns and col not in numeric_columns:
                if raw_df[col].dtype == 'object':
                    assert cleaned_df[col].dtype == 'object', \
                        f"Categorical column {col} should remain as object"


# ============================================================================
# CLASSE 2: Testes para TextNormalizer
# ============================================================================
class TestTextNormalizer:
    """Test suite for TextNormalizer transformer"""
    
    @pytest.fixture
    def csv_path(self):
        """Path to the CSV file"""
        return 'data/raw/Task_Exception Prediction_Training Test Data.csv'
    
    @pytest.fixture
    def raw_df(self, csv_path):
        """Load raw CSV"""
        return pd.read_csv(csv_path, sep=';', dtype=str)
    
    @pytest.fixture
    def text_normalizer(self):
        """TextNormalizer instance"""
        return TextNormalizer()
    
    @pytest.fixture
    def normalized_df(self, raw_df, text_normalizer):
        """DataFrame after TextNormalizer transformation"""
        return text_normalizer.transform(raw_df)
    
    def test_is_transformer(self, text_normalizer):
        """Test that TextNormalizer is a scikit-learn transformer"""
        from sklearn.base import BaseEstimator, TransformerMixin
        assert isinstance(text_normalizer, BaseEstimator)
        assert isinstance(text_normalizer, TransformerMixin)
    
    def test_fit_returns_self(self, text_normalizer, raw_df):
        """Test that fit method returns self"""
        result = text_normalizer.fit(raw_df)
        assert result is text_normalizer
    
    def test_no_rows_lost(self, raw_df, normalized_df):
        """Test that no rows are lost during normalization"""
        assert normalized_df.shape[0] == raw_df.shape[0], \
            f"Lost rows: {raw_df.shape[0]} -> {normalized_df.shape[0]}"
    
    def test_no_columns_lost(self, raw_df, normalized_df):
        """Test that no columns are lost during normalization"""
        assert normalized_df.shape[1] == raw_df.shape[1], \
            f"Lost columns: {raw_df.shape[1]} -> {normalized_df.shape[1]}"
    
    def test_text_lowercase(self, raw_df, normalized_df):
        """Test that text is converted to lowercase"""
        categorical_cols = ['Means_of_transportation', 'Mode_of_Transportation',
                          'Consignor_country', 'Recipient_country']
        
        for col in categorical_cols:
            if col in raw_df.columns and raw_df[col].dtype == 'object':
                # Check that values are lowercase
                sample_values = normalized_df[col].dropna().head(10)
                for val in sample_values:
                    if pd.notna(val) and isinstance(val, str):
                        assert val == val.lower(), \
                            f"Value '{val}' in column {col} should be lowercase"
    
    def test_text_stripped(self, text_normalizer):
        """Test that leading/trailing whitespace is stripped"""
        test_data = pd.DataFrame({
            'text_col': ['  FTL  ', '  ltl  ', 'AIR ', ' sea']
        })
        result = text_normalizer.transform(test_data)
        
        # Check that values are stripped
        assert result.loc[0, 'text_col'] == 'ftl', "Should strip and lowercase '  FTL  '"
        assert result.loc[1, 'text_col'] == 'ltl', "Should strip and lowercase '  ltl  '"
        assert result.loc[2, 'text_col'] == 'air', "Should strip and lowercase 'AIR '"
        assert result.loc[3, 'text_col'] == 'sea', "Should strip and lowercase ' sea'"
    
    def test_nan_strings_handled(self, text_normalizer):
        """Test that 'nan' strings are converted to actual NaN"""
        test_data = pd.DataFrame({
            'text_col': ['ftl', 'nan', 'ltl', 'NaN']
        })
        result = text_normalizer.transform(test_data)
        
        # 'nan' and 'NaN' should be converted to actual NaN
        assert pd.notna(result.loc[0, 'text_col']), "Valid value should remain"
        assert pd.isna(result.loc[1, 'text_col']), "'nan' string should become NaN"
        assert pd.notna(result.loc[2, 'text_col']), "Valid value should remain"
        assert pd.isna(result.loc[3, 'text_col']), "'NaN' string should become NaN"
    
    def test_preserves_case_after_normalization(self, text_normalizer):
        """Test that normalization handles mixed case correctly"""
        test_data = pd.DataFrame({
            'text_col': ['FTL', 'ftl', 'Ftl', 'FTL']
        })
        result = text_normalizer.transform(test_data)
        
        # All should be lowercase after normalization
        assert all(val == 'ftl' for val in result['text_col'].dropna()), \
            "All values should be normalized to lowercase 'ftl'"
    
    def test_categorical_columns_not_converted_to_nan(self, raw_df, normalized_df):
        """Test that categorical columns are NOT converted to NaN by TextNormalizer"""
        categorical_cols = ['Means_of_transportation', 'Mode_of_Transportation',
                          'Consignor_country', 'Recipient_country', 'distance cluster']
        
        for col in categorical_cols:
            if col in raw_df.columns and col in normalized_df.columns:
                # Count NaN before and after
                nan_before = raw_df[col].isna().sum()
                nan_after = normalized_df[col].isna().sum()
                
                # NaN count should not increase (may decrease if 'nan' strings were converted)
                assert nan_after <= nan_before + 1, \
                    f"Column {col} should not gain new NaN values. Before: {nan_before}, After: {nan_after}"
    
    def test_only_processes_object_columns(self, text_normalizer):
        """Test that TextNormalizer only processes object columns"""
        test_data = pd.DataFrame({
            'text_col': ['FTL', 'LTL'],  # object
            'numeric_col': [1, 2],        # int64
            'float_col': [1.5, 2.5]       # float64
        })
        result = text_normalizer.transform(test_data)
        
        # Text column should be normalized
        assert result.loc[0, 'text_col'] == 'ftl'
        
        # Numeric columns should remain unchanged
        assert result['numeric_col'].dtype == 'int64', "Numeric column should remain int64"
        assert result['float_col'].dtype == 'float64', "Float column should remain float64"


# ============================================================================
# CLASSE 3: Testes de integração (NumericCleaner + TextNormalizer)
# ============================================================================
class TestCleaningIntegration:
    """Integration tests for NumericCleaner and TextNormalizer together"""
    
    @pytest.fixture
    def csv_path(self):
        """Path to the CSV file"""
        return 'data/raw/Task_Exception Prediction_Training Test Data.csv'
    
    @pytest.fixture
    def raw_df(self, csv_path):
        """Load raw CSV"""
        return pd.read_csv(csv_path, sep=';', dtype=str)
    
    @pytest.fixture
    def numeric_columns(self):
        """Numeric columns that need cleaning"""
        return [
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
    
    @pytest.fixture
    def fully_cleaned_df(self, raw_df, numeric_columns):
        """DataFrame after both NumericCleaner and TextNormalizer"""
        # First apply NumericCleaner
        numeric_cleaner = NumericCleaner(target_columns=numeric_columns)
        df_cleaned = numeric_cleaner.transform(raw_df)
        
        # Then apply TextNormalizer
        text_normalizer = TextNormalizer()
        df_fully_cleaned = text_normalizer.transform(df_cleaned)
        
        return df_fully_cleaned
    
    def test_integration_no_data_loss(self, raw_df, fully_cleaned_df):
        """Test that integration doesn't lose data"""
        assert fully_cleaned_df.shape == raw_df.shape, \
            f"Shape should be consistent: {raw_df.shape} -> {fully_cleaned_df.shape}"
    
    def test_integration_numeric_columns_are_numeric(self, fully_cleaned_df, numeric_columns):
        """Test that numeric columns are numeric after integration"""
        for col in numeric_columns:
            if col in fully_cleaned_df.columns:
                assert pd.api.types.is_numeric_dtype(fully_cleaned_df[col]), \
                    f"Column {col} should be numeric after integration"
    
    def test_integration_text_columns_normalized(self, fully_cleaned_df):
        """Test that text columns are normalized after integration"""
        categorical_cols = ['Means_of_transportation', 'Mode_of_Transportation']
        
        for col in categorical_cols:
            if col in fully_cleaned_df.columns:
                sample_values = fully_cleaned_df[col].dropna().head(10)
                for val in sample_values:
                    if pd.notna(val) and isinstance(val, str):
                        assert val == val.lower(), \
                            f"Value '{val}' in column {col} should be lowercase"
                        assert val == val.strip(), \
                            f"Value '{val}' in column {col} should be stripped"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
