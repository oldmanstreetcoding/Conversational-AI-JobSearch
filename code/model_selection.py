import pandas as pd
import torch

from unsloth import FastLanguageModel
from datasets import Dataset

def define_base_model():
    print('model selection')
    # # Check if CUDA is available and set the device accordingly
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("CUDA is available. Using GPU.")
    # else:
    #     device = torch.device("cpu")
    #     print("CUDA is not available. Using CPU.")

    # # Patch `FastLanguageModel` to avoid CUDA initialization if it's not available
    # if device.type == "cpu":
    #     # Prevent FastLanguageModel from checking CUDA capabilities
    #     torch.cuda.get_device_capability = lambda: (0, 0)  # Dummy function to bypass CUDA checks

    # # Load the base model (Phi-3.5-mini-instruct)
    # model_name = "unsloth/Phi-3.5-mini-instruct"

    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name=model_name,
    #     max_seq_length=2048, # Change based on your hardware capabilities
    #     dtype=None, # Use auto detection
    #     load_in_4bit=True, # For memory optimization
    # )

    # # Unsloth 2024.9.post4 patched 32 layers with 32 QKV layers, 32 O layers and 32 MLP layers.
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r=16,  # Low-rank adaptation dimension
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Layers for LoRA
    #     lora_alpha=16,  # Scaling factor
    #     lora_dropout=0,  # No dropout for faster training
    #     bias="none",  # No additional bias terms
    #     random_state=3407,  # For reproducibility
    #     loftq_config=None,  # No specific LoFTQ configuration
    # )

    # # Load train and test datasets
    # train_df = pd.read_csv('/content/drive/MyDrive/NLX Data/conversational_train.csv', encoding='utf-8')
    # test_df = pd.read_csv('/content/drive/MyDrive/NLX Data/conversational_test.csv', encoding='utf-8')

    # # Prepare datasets as HuggingFace Dataset objects
    # train_dataset = Dataset.from_pandas(train_df)
    # test_dataset = Dataset.from_pandas(test_df)

    # from unsloth.chat_templates import get_chat_template

    # # Apply the chat template for conversation-style input
    # tokenizer = get_chat_template(
    #     tokenizer,
    #     chat_template="phi-3",  # Custom chat template for Phi-3
    #     mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    # )

    # # Function to format prompts for training using the Job Query and Job Description columns
    # def formatting_prompts_func(examples):
    #     # Create a conversation from the Job Query and Job Description
    #     convos = [
    #         {"from": "human", "value": examples["Job Query"].strip()},
    #         {"from": "gpt", "value": examples["Job Description"].strip()},
    #     ]

    #     # Format the conversation as a text prompt using the chat template
    #     formatted_text = tokenizer.apply_chat_template(convos, tokenize=False, add_generation_prompt=False)

    #     return {"text": formatted_text}

    # # Apply the formatting function to each row of the dataset
    # formatted_train_dataset = train_dataset.map(formatting_prompts_func, batched=False)
    # formatted_test_dataset = test_dataset.map(formatting_prompts_func, batched=False)

    # # Tokenize the dataset for model input
    # def tokenize_and_prepare_inputs(examples):
    #     tokenized = tokenizer(
    #         examples["text"],
    #         padding=True,
    #         truncation=True,
    #         return_tensors="pt",
    #         return_attention_mask=True
    #     )
    #     return {
    #         "input_ids": tokenized["input_ids"].tolist(),
    #         "attention_mask": tokenized["attention_mask"].tolist()
    #     }

    # # Apply tokenization to training and test datasets
    # tokenized_train_dataset = formatted_train_dataset.map(tokenize_and_prepare_inputs, batched=True)
    # tokenized_test_dataset = formatted_test_dataset.map(tokenize_and_prepare_inputs, batched=True)

    # return model, tokenizer, tokenized_train_dataset