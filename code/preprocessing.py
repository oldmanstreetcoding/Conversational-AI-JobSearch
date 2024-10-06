# preprocessing.py

import pandas as pd
import numpy as np
import re
import unicodedata
import warnings
from transformers import GPT2Tokenizer

# Suppress warnings
warnings.filterwarnings('ignore')

def run_preprocessing():
    """
    Run all preprocessing steps, from loading the dataset to handling missing values, normalization, and tokenization.
    This function will be invoked by the main script to perform data preprocessing.
    """
    print(f"\nLoading datasets...")
    train_data, test_data = load_datasets()

    print(f"\nLowercasing relevant columns...")
    train_data_cleaned, test_data_cleaned = lowercase_text_columns(train_data, test_data)

    print(f"\nRemoving duplicated rows...")
    train_data_cleaned = remove_duplicates(train_data_cleaned)
    test_data_cleaned = remove_duplicates(test_data_cleaned)

    print(f"\nHandling missing values in salary columns...")
    train_data_cleaned, test_data_cleaned = process_salary_imputation(train_data_cleaned, test_data_cleaned)

    print(f"\nHandling punctuation, URLs, and icons...")
    train_data_cleaned, test_data_cleaned = clean_datasets(train_data_cleaned, test_data_cleaned)

    # print(f"\nTokenizing text columns...\n")
    # train_data_tokenized = tokenize_text_columns(train_data_cleaned, ['Job Query', 'Job Description'])
    # test_data_tokenized = tokenize_text_columns(test_data_cleaned, ['Job Query', 'Job Description'])

    # # Save preprocessed data
    # train_data_tokenized.to_csv('tokenized_conversational_job_data_train.csv', index=False)
    # test_data_tokenized.to_csv('tokenized_conversational_job_data_test.csv', index=False)

    print(f"\nData Preprocessing Completed. Files Saved.\n")

def load_datasets():
    """
    Load the train and test datasets.
    """
    train_data = pd.read_csv('data/train-1.csv')
    test_data = pd.read_csv('data/test-1.csv')
    return train_data, test_data

def lowercase_text_columns(train_df, test_df):
    """
    Lowercase the text in 'Resume', 'Job-Title', 'Location', and 'Description' columns.
    """
    columns_to_lowercase = ['Resume', 'Job-Title', 'Description', 'Location']
    for col in columns_to_lowercase:
        train_df[col] = train_df[col].str.lower()
        test_df[col] = test_df[col].str.lower()
    return train_df, test_df

def remove_duplicates(df):
    """
    Check for and remove duplicate rows across all columns in the dataset.
    """
    duplicates = df.duplicated()
    print(f">> Number of duplicate rows: {duplicates.sum()}")
    if duplicates.sum() > 0:
        df = df[~duplicates]  # Remove duplicates
        print(f">> Duplicates removed. New size: {df.shape}")
    return df

def process_salary_imputation(train_data_cleaned, test_data_cleaned):
    """
    Process the salary column by applying salary categorization and imputation based on Job Title and Location.
    """
    # 1. Define salary range buckets
    def categorize_salary(salary):
        """
        Convert salary values to general buckets for ease of classification.
        Example: $50K–$75K, $75K–$100K, etc.
        """
        salary = str(salary).replace(',', '').replace('$', '').replace('₹', '').strip()

        if 'year' in salary.lower():
            salary = salary.lower().replace('a year', '').strip()
        try:
            salary = float(salary)
        except ValueError:
            return np.nan

        if salary < 50000:
            return 'Below $50K'
        elif 50000 <= salary < 75000:
            return '$50K–$75K'
        elif 75000 <= salary < 100000:
            return '$75K–$100K'
        elif 100000 <= salary < 150000:
            return '$100K–$150K'
        else:
            return 'Above $150K'

    # Apply salary bucket categorization for known salaries
    def apply_salary_buckets(df):
        df['Salary Bucket'] = df['Salary'].apply(categorize_salary)
        return df

    train_data_cleaned = apply_salary_buckets(train_data_cleaned)
    test_data_cleaned = apply_salary_buckets(test_data_cleaned)

    # 2. Imputation based on Job Title and Location
    def impute_salary_by_title_location(df):
        """
        Impute missing salary values based on median salary for the same Job Title and Location.
        """
        # Group by Job Title and Location to compute median salary bucket
        salary_mapping = df.groupby(['Job-Title', 'Location'])['Salary Bucket'].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan).to_dict()

        # Impute missing salaries
        def impute_salary(row):
            if pd.isnull(row['Salary Bucket']):
                return salary_mapping.get((row['Job-Title'], row['Location']), 'Unknown')
            return row['Salary Bucket']

        df['Imputed Salary Bucket'] = df.apply(impute_salary, axis=1)
        return df

    # 3. Final step: Assign "Unknown" for any remaining missing salary values
    def finalize_salary_imputation(df):
        """
        Finalize the salary imputation process:
        - If 'Imputed Salary Bucket' is null, check the 'Salary' column.
        - If 'Salary' is null, fill 'Imputed Salary Bucket' with 'Unknown'.
        - If 'Salary' is not null, take the value from 'Salary' and assign it to 'Imputed Salary Bucket'.
        """
        def fill_imputed_salary(row):
            if pd.isna(row['Imputed Salary Bucket']):  # If 'Imputed Salary Bucket' is null
                if pd.isna(row['Salary']):  # If 'Salary' is also null, assign 'Unknown'
                    return 'Unknown'
                else:
                    return row['Salary']  # Otherwise, take the 'Salary' value
            return row['Imputed Salary Bucket']  # If 'Imputed Salary Bucket' is not null, keep it

        # Apply the logic to the entire DataFrame
        df['Imputed Salary'] = df.apply(fill_imputed_salary, axis=1)

        return df

    # Run the entire process on both datasets
    def process_salary_imputation(df):
        df = apply_salary_buckets(df)
        df = impute_salary_by_title_location(df)
        df = finalize_salary_imputation(df)
        return df

    # Impute missing salaries in both datasets
    train_data_cleaned = process_salary_imputation(train_data_cleaned)
    test_data_cleaned = process_salary_imputation(test_data_cleaned)

    # Check missing value count in Imputed Salary
    print(">> Missing value count in Imputed Salary (Train):", train_data_cleaned['Imputed Salary'].isnull().sum())
    print(">> Missing value count in Imputed Salary (Test):", test_data_cleaned['Imputed Salary'].isnull().sum())

    # print(f"\nHandling missing values in salary column process complete ...\n")

    return train_data_cleaned, test_data_cleaned
    
def clean_datasets(train_data_cleaned, test_data_cleaned):
    """
    Clean text by removing punctuation, URLs, and icons, as well as normalizing text.
    """
    # Function to remove punctuation
    def remove_punc(text):
        punc_pattern = r'[^\w\s]'
        return re.sub(punc_pattern, '', text)

    # Function to remove URLs and newlines
    def clean_text(text):
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub('\n', ' ', text)  # Replace newline characters with space
        text = re.sub('\n\n', ' ', text)  # Replace double newline characters with space
        return text
    
    # Function to remove all icons (including emojis)
    def remove_icons(text):
        icon_pattern = re.compile("[" 
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002500-\U00002BEF"  # miscellaneous symbols
                            u"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
                            u"\U00002000-\U00002BFF"  # arrows, bullets, stars, math operators
                            u"\U0000FE0F-\U0000FEFF"  # variation selectors
                            u"\U0001F700-\U0001F77F"  # alchemical symbols
                            u"\u200d"                 # zero-width joiner
                            u"\u2640-\u2642"          # gender symbols
                            u"\u2600-\u26FF"          # miscellaneous symbols
                            u"\u23cf"                 # eject symbol
                            u"\u23e9"                 # fast-forward
                            u"\u231a"                 # watch
                            u"\ufe0f"                 # dingbats
                            u"\u3030"                 # wavy dash
                            "]+", flags=re.UNICODE)
        return re.sub(icon_pattern, '', text)

    # Combined function to clean both punctuation and URLs/newlines
    def full_cleaning(text):
        text = remove_punc(text)
        text = clean_text(text)
        text = remove_icons(text)
        return text

    # Apply the cleaning function to relevant columns in the datasets
    def clean_datasets(df):
        df['Resume'] = df['Resume'].apply(full_cleaning)
        df['Description'] = df['Description'].apply(full_cleaning)
        return df

    # Apply to train and test datasets
    train_data_cleaned = clean_datasets(train_data_cleaned)
    test_data_cleaned = clean_datasets(test_data_cleaned)

    return train_data_cleaned, test_data_cleaned


def tokenize_text_columns(df, columns):
    """
    Tokenize the specified columns using the GPT-2 tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    for column in columns:
        df[f'{column}_tokens'] = df[column].apply(lambda x: tokenizer.tokenize(x))
    
    return df
