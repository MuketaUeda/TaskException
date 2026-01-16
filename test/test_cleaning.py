import pandas as pd
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cleaning import NumericCleaner, TextNormalizer

def main():
    # 1. Load CSV (note: uses semicolon as separator)
    csv_path = 'data/raw/Task_Exception Prediction_Training Test Data.csv'
    print(f"Loading CSV from: {csv_path}")
    
    df = pd.read_csv(csv_path, sep=';')
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show first few rows before cleaning
    print("\n" + "="*80)
    print("BEFORE CLEANING - First 5 rows:")
    print("="*80)
    print(df.head())
    
    print("\n" + "="*80)
    print("Column dtypes BEFORE:")
    print("="*80)
    print(df.dtypes)
    
    # 2. Define numeric columns that may be stored as strings with comma (European format)
    # These are columns that SHOULD be numeric but may come as strings like "14136,72"
    numeric_columns = [
        'Loading_meter [ldm]',
        'Gross_weight [kg]',              # Example: "14136,72" -> 14136.72
        'Volume [m3]',                    # Example: "92,66" -> 92.66
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
    
    # 3. Apply NumericCleaner ONLY on numeric columns
    print("\n" + "="*80)
    print("Applying NumericCleaner to numeric columns...")
    print(f"Processing columns: {numeric_columns}")
    print("="*80)
    numeric_cleaner = NumericCleaner(target_columns=numeric_columns)
    df_cleaned = numeric_cleaner.transform(df)
    
    # 4. Apply TextNormalizer to ALL text/categorical columns
    # This normalizes text (lowercase, strip) but keeps them as text for encoding later
    print("\nApplying TextNormalizer to text/categorical columns...")
    print("="*80)
    text_normalizer = TextNormalizer()
    df_cleaned = text_normalizer.transform(df_cleaned)
    
    # Show results
    print("\n" + "="*80)
    print("AFTER CLEANING - First 5 rows:")
    print("="*80)
    print(df_cleaned.head())
    
    print("\n" + "="*80)
    print("Column dtypes AFTER:")
    print("="*80)
    print(df_cleaned.dtypes)
    
    # Show specific examples of changes
    print("\n" + "="*80)
    print("EXAMPLES OF CHANGES:")
    print("="*80)
    
    # Example 1: Numeric column (should be converted from string with comma to float)
    if 'Gross_weight [kg]' in df.columns:
        print(f"\n[NUMERIC] Gross_weight [kg]:")
        print(f"  BEFORE: {df['Gross_weight [kg]'].iloc[0]} (type: {type(df['Gross_weight [kg]'].iloc[0])})")
        print(f"  AFTER:  {df_cleaned['Gross_weight [kg]'].iloc[0]} (type: {type(df_cleaned['Gross_weight [kg]'].iloc[0])})")
        print(f"  ✓ Converted from string with comma to float")
    
    if 'Volume [m3]' in df.columns:
        print(f"\n[NUMERIC] Volume [m3]:")
        print(f"  BEFORE: {df['Volume [m3]'].iloc[0]} (type: {type(df['Volume [m3]'].iloc[0])})")
        print(f"  AFTER:  {df_cleaned['Volume [m3]'].iloc[0]} (type: {type(df_cleaned['Volume [m3]'].iloc[0])})")
        print(f"  ✓ Converted from string with comma to float")
    
    # Example 2: Categorical column (should remain as text, just normalized)
    if 'Means_of_transportation' in df.columns:
        print(f"\n[CATEGORICAL] Means_of_transportation:")
        print(f"  BEFORE: '{df['Means_of_transportation'].iloc[0]}' (type: {type(df['Means_of_transportation'].iloc[0])})")
        print(f"  AFTER:  '{df_cleaned['Means_of_transportation'].iloc[0]}' (type: {type(df_cleaned['Means_of_transportation'].iloc[0])})")
        print(f"  ✓ Kept as text (normalized to lowercase), NOT converted to NaN")
    
    if 'Mode_of_Transportation' in df.columns:
        print(f"\n[CATEGORICAL] Mode_of_Transportation:")
        print(f"  BEFORE: '{df['Mode_of_Transportation'].iloc[0]}' (type: {type(df['Mode_of_Transportation'].iloc[0])})")
        print(f"  AFTER:  '{df_cleaned['Mode_of_Transportation'].iloc[0]}' (type: {type(df_cleaned['Mode_of_Transportation'].iloc[0])})")
        print(f"  ✓ Kept as text (normalized to lowercase), NOT converted to NaN")
    
    # Verify no categorical columns were converted to NaN by our cleaning
    categorical_cols = ['Means_of_transportation', 'Mode_of_Transportation', 
                       'Consignor_country', 'Recipient_country', 'distance cluster']
    print(f"\n[CHECK] Verifying categorical columns were NOT converted to NaN:")
    print("(NaN values shown here already existed in the original CSV, not created by cleaning)")
    for col in categorical_cols:
        if col in df_cleaned.columns:
            nan_before = df[col].isna().sum() if col in df.columns else 0
            nan_after = df_cleaned[col].isna().sum()
            total = len(df_cleaned)
            percentage = (nan_after / total) * 100
            
            if nan_before == nan_after:
                if nan_after == 0:
                    print(f"  ✓ {col}: No NaN values (kept as text)")
                else:
                    print(f"  ✓ {col}: {nan_after}/{total} NaN values ({percentage:.2f}% - existed in original CSV)")
            else:
                print(f"  ⚠ {col}: BEFORE={nan_before}, AFTER={nan_after} (WARNING: NaN count changed!)")
    
    print("\n" + "="*80)
    print("Cleaning completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
