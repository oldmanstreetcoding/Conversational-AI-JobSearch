# preprocessing.py

import pandas as pd
import numpy as np
import re
import unicodedata
import warnings
import random
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split

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

    print(f"\nHandling punctuation, URLs, Newlines, Icons, Contractions...")
    train_data_cleaned, test_data_cleaned = clean_datasets(train_data_cleaned, test_data_cleaned)

    print(f"\nStandardizing Job Title and Location...")
    train_data_cleaned, test_data_cleaned = handling_jobtitle_location(train_data_cleaned, test_data_cleaned)

    print(f"\nCreating Query Job Pairs...")
    conversational_train_data = create_job_query_pairs(train_data_cleaned, 'conversational_train')
    conversational_test_data = create_job_query_pairs(test_data_cleaned, 'conversational_test')
    
    print(f"\nPreparing Fine Tuning Data...")
    preparing_finetuning_data(conversational_train_data, 'conversational_train')
    preparing_finetuning_data(conversational_test_data, 'conversational_test')

    print(f"\nTokenizing text columns...")
    tokenized_conversational_train_df = tokenize_text_columns(conversational_train_data, ['Job Query', 'Job Description'], 'conversational_train')
    tokenized_conversational_test_df = tokenize_text_columns(conversational_test_data, ['Job Query', 'Job Description'], 'conversational_test')

    # Step 1: Split the training data into training and validation sets
    # Let's use 80% for training and 20% for validation
    train_set, val_set = train_test_split(tokenized_conversational_train_df, test_size=0.2, random_state=42)

    # Display the size of each set
    print(f"Training set size: {train_set.shape}")
    print(f"Validation set size: {val_set.shape}")
    print(f"Test set size: {tokenized_conversational_test_df.shape}")

    # Step 2: Save the split datasets to CSV files for future use
    train_set.to_csv('train_set_for_fine_tuning.csv', index=False)
    val_set.to_csv('validation_set_for_fine_tuning.csv', index=False)
    tokenized_conversational_test_df.to_csv('test_set_for_evaluation.csv', index=False)

    print(f"\nData Preprocessing Completed. Files Saved.")

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
    
    # Function to normalize text by removing extra spaces, expanding contractions, and normalizing Unicode
    def normalize_text(text):
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Expand common contractions (for demonstration, only a few contractions are expanded)
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'ll": " will",
            "'re": " are",
            "'ve": " have",
            "'d": " would",
            "'m": " am"
        }
        for contraction, expanded in contractions.items():
            text = re.sub(contraction, expanded, text)

        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)

        return text

    # Combined function to clean both punctuation and URLs/newlines
    def full_cleaning(text):
        text = remove_punc(text)
        text = clean_text(text)
        text = remove_icons(text)
        text = normalize_text(text)
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

def handling_jobtitle_location(train_data_cleaned, test_data_cleaned):
    # Function to clean and standardize job titles (expanding abbreviations and removing extra spaces)
    def clean_job_title(job_title):
        job_title = str(job_title)  # Convert to string to avoid errors
        job_title = job_title.replace("s/w", "software")
        job_title = job_title.replace("dev", "developer")
        job_title = re.sub(r'\s+', ' ', job_title).strip()  # Remove extra spaces
        return job_title

    # Function to clean and standardize locations (expanding abbreviations and ensuring consistency)
    def clean_location(location):
        location = str(location)  # Convert to string to avoid errors
        location = location.replace("nyc", "new york city")
        location = location.replace("usa", "united states")
        location = re.sub(r'\s+', ' ', location).strip()  # Remove extra spaces
        return location

    # Final function to clean Salary, Imputed Salary, Job Title, and Location
    def clean_additional_columns(df):
        df['Job-Title'] = df['Job-Title'].apply(clean_job_title)
        df['Location'] = df['Location'].apply(clean_location)
        return df

    # Apply to both datasets
    train_data_cleaned = clean_additional_columns(train_data_cleaned)
    test_data_cleaned = clean_additional_columns(test_data_cleaned)

    return train_data_cleaned, test_data_cleaned

def create_job_query_pairs(df, dfname):

    # List of instruction variations
    instruction_templates = [
        "Answer the following job-related query accurately.",
        "Provide job information based on the user's question.",
        "Respond to the job search query with precise details.",
        "Give detailed information in response to the job query.",
        "Answer the user's question about job opportunities.",
        "Provide accurate job-related details based on the user's query.",
        "Respond to the job inquiry with relevant job information.",
        "Give a precise response to the following job-related question.",
        "Answer the job-related question with the required details.",
        "Provide job search information based on the user's request."
    ]

    # List of query variations for different categories
    skills_query_templates = [
        "What jobs are available for {}?",
        "Are there jobs for {}?",
        "What roles are available for {}?",
        "Could you show me jobs for {}?",
        "I'm looking for jobs that require {}.",
        "Find me jobs for {} experts.",
        "What developer jobs are available for {}?",
        "Show me jobs requiring {} skills.",
        "What jobs need {} skills?",
        "Which positions are open for {}?"
    ]

    location_query_templates = [
        "What jobs are open in {}?",
        "What jobs are available in {}?",
        "Give me the information of available jobs in {}.",
        "Are there jobs open in {}?",
        "Any job openings in {}?",
        "Which jobs are available in {}?",
        "Show me jobs in {}.",
        "What positions are open in {}?",
        "What jobs can I find in {}?",
        "List available jobs in {}."
    ]

    salary_query_templates = [
        "What jobs offer a salary of {}?",
        "Which jobs pay over {}?",
        "Are there jobs with a salary of {}?",
        "Show me jobs paying more than {}.",
        "What jobs are available with a salary of {}?",
        "List jobs with a salary of {}.",
        "Any jobs offering a salary above {}?",
        "What jobs provide a salary over {}?",
        "Show me jobs with a salary above {}.",
        "Find me jobs with a salary range of {}."
    ]

    experience_query_templates = [
        "What jobs are available for someone with {} years of experience?",
        "What roles can I find for someone with {} years of experience?",
        "Are there jobs for candidates with {} years of experience?",
        "Show me jobs for someone with {} years of development experience.",
        "What jobs are available for an experienced {}?",
        "Find me jobs for professionals with {} years of experience.",
        "Which jobs require {} years of experience?",
        "What positions are available for someone with {} years in management?",
        "List jobs for people with {} years of experience.",
        "Are there positions for someone with {} years of experience?"
    ]

    # Function to generate varied instructions
    def generate_instruction():
        return random.choice(instruction_templates)

    # Function to generate all variations of queries for each type
    def generate_all_queries(row, query_type):
        queries = []
        if query_type == 'skills':
            skill_info = row['Resume'] + " and " + row['Description']
            for template in skills_query_templates:
                queries.append(template.format(skill_info))
        elif query_type == 'location':
            location = row['Location']
            for template in location_query_templates:
                queries.append(template.format(location))
        elif query_type == 'salary':
            salary = row['Imputed Salary']
            for template in salary_query_templates:
                queries.append(template.format(salary))
        elif query_type == 'experience':
            experience = row['Description']  # Assume experience is described in the 'Description' column
            for template in experience_query_templates:
                queries.append(template.format(experience))
        return queries

    # Function to generate all job descriptions for a given query type
    def generate_all_job_descriptions(row, query_type):
        descriptions = []
        if query_type == 'location':
            for _ in location_query_templates:
                descriptions.append(f"Job available in {row['Location']}\nJob-Title: {row['Job-Title']}\nSalary: {row['Imputed Salary']}\nResume: {row['Resume']}\nDescription: {row['Description']}")
        elif query_type == 'skills':
            for _ in skills_query_templates:
                descriptions.append(f"Job available for {row['Resume']}\nLocation: {row['Location']}\nSalary: {row['Imputed Salary']}\nJob-Title: {row['Job-Title']}\nDescription: {row['Description']}")
        elif query_type == 'salary':
            for _ in salary_query_templates:
                descriptions.append(f"Job available with a salary of {row['Imputed Salary']}\nLocation: {row['Location']}\nJob-Title: {row['Job-Title']}\nResume: {row['Resume']}\nDescription: {row['Description']}")
        elif query_type == 'experience':
            for _ in experience_query_templates:
                descriptions.append(f"Job available for someone with {row['Description']}\nLocation: {row['Location']}\nJob-Title: {row['Job-Title']}\nSalary: {row['Imputed Salary']}\nResume: {row['Resume']}")
        return descriptions

    # Combine everything into a final function that prepares the dataset with all variations
    def prepare_conversational_all_variations(df):
        conversational_data = {
            'Instructions': [],
            'Job Query': [],
            'Job Description': []
        }
        
        query_types = ['skills', 'location', 'salary', 'experience']
        
        for index, row in df.iterrows():
            for query_type in query_types:
                queries = generate_all_queries(row, query_type)
                descriptions = generate_all_job_descriptions(row, query_type)
                
                for i in range(len(queries)):
                    # Generate varied instructions
                    instruction = generate_instruction()
                    conversational_data['Instructions'].append(instruction)
                    
                    # Add all variations of queries and descriptions
                    conversational_data['Job Query'].append(queries[i])
                    conversational_data['Job Description'].append(descriptions[i])
        
        # Convert to a DataFrame
        conversational_df = pd.DataFrame(conversational_data)
        
        return conversational_df
    
    # Apply the function to generate the dataset with all query variations
    conversational_df = prepare_conversational_all_variations(df)
    print(f'{dfname} has been created with shape: {conversational_df.shape}')

    # Save the dataset to a CSV file
    conversational_df.to_csv(f'{dfname}.csv', index=False)

    return conversational_df

def preparing_finetuning_data(df, dfname):
    # Assuming 'conversational_df_all_variations_test_data' is your dataframe with 'Job Query' and 'Job Description'
    # Step 1: Create <Job Query, Job Description> Pairs
    job_queries = df['Job Query'].tolist()
    job_descriptions = df['Job Description'].tolist()

    # Step 2: Prepare Data for Fine-Tuning
    # We create a list of tuples where each tuple is a (query, description) pair
    query_description_pairs = list(zip(job_queries, job_descriptions))

    # Save the query-description pairs to a CSV or any suitable format for fine-tuning
    fine_tuning_data = pd.DataFrame(query_description_pairs, columns=['Job Query', 'Job Description'])

    # # Save to CSV for inspection or later use
    # fine_tuning_data.to_csv('fine_tuning_job_query_description_pairs.csv', index=False)

    # Display a few rows to ensure everything looks correct
    print(f'\n{dfname}: ')
    print(fine_tuning_data.head(2))

def tokenize_text_columns(df, columns, dfname):
    """
    Tokenize the specified columns using the GPT-2 tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    for column in columns:
        df[f'{column}_tokens'] = df[column].apply(lambda x: tokenizer.tokenize(x))

    # Display a few rows to ensure everything looks correct
    print(f'\n{dfname}: ')
    print(df[['Job Query', 'Job Query_tokens', 'Job Description', 'Job Description_tokens']].head(2))

    return df
