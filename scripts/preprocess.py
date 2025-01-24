import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import sparse

def preprocess_data(df, output_file='preprocessed_data.csv'):
    """Preprocesses the dataset and saves it to a new file."""
    
    # Separate numeric and categorical features
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # Numeric transformer: Impute missing values and scale the data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  
        ('scaler', StandardScaler())  
    ])

    # Categorical transformer: Impute missing values and apply OneHotEncoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Impute missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))  # One-hot encoding with sparse matrix output
    ])

    # ColumnTransformer to apply the transformers to respective columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply preprocessing steps to the dataset
    df_processed = preprocessor.fit_transform(df)

    # Check if the processed data is sparse and handle accordingly
    if isinstance(df_processed, sparse.csr_matrix):
        df_processed = pd.DataFrame.sparse.from_spmatrix(df_processed)  # Keep it sparse if it's a sparse matrix

    # Get the column names from the transformer for categorical features
    column_names = (numeric_features.tolist() + 
                    list(preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)))
    
    # Check if column names match the processed data
    if df_processed.shape[1] != len(column_names):
        raise ValueError(f"Shape mismatch: processed data has {df_processed.shape[1]} columns, but column names list has {len(column_names)} columns")

    # Create a DataFrame with the correct column names
    df_processed.columns = column_names

    # Save the processed data to a CSV file
    df_processed.to_csv(output_file, index=False)

    return df_processed

# Example usage
if __name__ == "__main__":
    # Load your raw data (e.g., from a CSV file)
    df = pd.read_csv('E:/data06/data.csv')
    
    # Call the preprocessing function and save the result
    preprocessed_df = preprocess_data(df, 'preprocessed_data.csv')

    print("Preprocessing completed and saved to 'preprocessed_data.csv'")
