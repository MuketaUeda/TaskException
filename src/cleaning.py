from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class NumericCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean numeric columns stored as strings.
    Handles European formatting (e.g., '13,5' -> 13.5) and common formatting issues.
    """
    def __init__(self, target_columns=None):
        self.target_columns = target_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned_dataset = X.copy()
        
        # Process all columns if none specified
        if self.target_columns is not None:
            columns_to_process = self.target_columns
        else:
            columns_to_process = cleaned_dataset.columns

        for column_name in columns_to_process:
            # Only process object-type columns (strings)
            if column_name in cleaned_dataset.columns and cleaned_dataset[column_name].dtype == 'object':
                # Convert to string first to handle mixed types and NaN values
                cleaned_dataset[column_name] = cleaned_dataset[column_name].astype(str)
                
                # Replace comma with dot (European decimal format)
                cleaned_dataset[column_name] = cleaned_dataset[column_name].str.replace(',', '.', regex=False)
                
                # Remove whitespace (e.g., "1 000" -> "1000")
                cleaned_dataset[column_name] = cleaned_dataset[column_name].str.replace(' ', '', regex=False)
                
                # Convert to numeric (invalid values become NaN)
                cleaned_dataset[column_name] = pd.to_numeric(cleaned_dataset[column_name], errors='coerce')
                
        return cleaned_dataset

class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    Transformer to normalize text columns, ensuring 'Mega' and 'mega' 
    are treated as the same category.
    """
    def __init__(self, target_columns=None):
        self.target_columns = target_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        normalized_dataset = X.copy()
        
        # Process all columns if none specified
        if self.target_columns is not None:
            columns_to_process = self.target_columns
        else:
            columns_to_process = normalized_dataset.columns

        for column_name in columns_to_process:
            if column_name in normalized_dataset.columns and normalized_dataset[column_name].dtype == 'object':
                # Convert to lowercase, strip whitespace, and handle 'nan' strings
                s = normalized_dataset[column_name].astype(str).str.lower().str.strip()
                normalized_dataset[column_name] = s.replace('nan', np.nan)
                
        return normalized_dataset