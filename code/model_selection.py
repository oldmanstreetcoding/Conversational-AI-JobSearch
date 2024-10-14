# model_selection.py

import time
import pandas as pd
# from unsloth import FastLanguageModel
# from datasets import Dataset
# from unsloth.chat_templates import get_chat_template
# from transformers import TextStreamer

def load_model():
    """
    Load the base model (Phi-3.5-mini-instruct) for fine-tuning.
    
    Returns:
        model: Loaded language model.
        tokenizer: Corresponding tokenizer for the model.
    """
    base_model = "unsloth/Phi-3.5-mini-instruct"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,  # Change based on your hardware capabilities
        dtype=None,  # Use auto detection
        load_in_4bit=True,  # For memory optimization
    )
    
    return model, tokenizer

def load_datasets(train_path, test_path):
    """
    Load training and testing datasets from specified CSV files.
    
    Parameters:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the testing dataset.
    
    Returns:
        tuple: DataFrames for training and testing datasets.
    """
    train_df = pd.read_csv(train_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8')
    
    return train_df, test_df

def format_datasets(train_df, test_df, tokenizer):
    """
    Prepare datasets for model training by formatting inputs.
    
    Parameters:
        train_df (DataFrame): DataFrame containing training data.
        test_df (DataFrame): DataFrame containing testing data.
        tokenizer: Tokenizer for the model.
    
    Returns:
        tuple: Formatted datasets for training and testing.
    """
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Apply chat template for conversation-style input
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-3",  # Custom chat template for Phi-3
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )

    # Format prompts for training using the Job Query and Job Description columns
    def formatting_prompts_func(examples):
        convos = [
            {"from": "human", "value": examples["Job Query"].strip()},
            {"from": "gpt", "value": examples["Job Description"].strip()},
        ]
        formatted_text = tokenizer.apply_chat_template(convos, tokenize=False, add_generation_prompt=False)
        return {"text": formatted_text}

    formatted_train_dataset = train_dataset.map(formatting_prompts_func, batched=False)
    formatted_test_dataset = test_dataset.map(formatting_prompts_func, batched=False)

    return formatted_train_dataset, formatted_test_dataset

def tokenize_datasets(formatted_train_dataset, formatted_test_dataset, tokenizer):
    """
    Tokenize the formatted datasets for model input.
    
    Parameters:
        formatted_train_dataset (Dataset): The formatted training dataset.
        formatted_test_dataset (Dataset): The formatted testing dataset.
        tokenizer: Tokenizer for the model.
    
    Returns:
        tuple: Tokenized datasets for training and testing.
    """
    def tokenize_and_prepare_inputs(examples):
        tokenized = tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        return {
            "input_ids": tokenized["input_ids"].tolist(),
            "attention_mask": tokenized["attention_mask"].tolist()
        }

    tokenized_train_dataset = formatted_train_dataset.map(tokenize_and_prepare_inputs, batched=True)
    tokenized_test_dataset = formatted_test_dataset.map(tokenize_and_prepare_inputs, batched=True)

    return tokenized_train_dataset, tokenized_test_dataset

def define_base_model(train_path, test_path):
    """
    Define the base model and prepare datasets for fine-tuning.
    
    Parameters:
        train_path (str): Path to the training dataset CSV file.
        test_path (str): Path to the testing dataset CSV file.
    
    Returns:
        tuple: Tokenized training and testing datasets, and the model.
    """
    model, tokenizer = load_model()
    train_df, test_df = load_datasets(train_path, test_path)
    formatted_train_dataset, formatted_test_dataset = format_datasets(train_df, test_df, tokenizer)
    tokenized_train_dataset, tokenized_test_dataset = tokenize_datasets(formatted_train_dataset, formatted_test_dataset, tokenizer)

    return tokenized_train_dataset, tokenized_test_dataset, model
