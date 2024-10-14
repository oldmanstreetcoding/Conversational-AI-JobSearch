# finetuning.py

import time
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import TrainingArguments, SFTTrainer
from evaluate import load

def setup_model(model):
    """
    Setup LoRA adapters for the given model.
    
    Parameters:
        model: The base model to apply LoRA adapters to.
    
    Returns:
        model_lora: The model with LoRA adapters applied.
    """
    model_lora = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Low-rank adaptation dimension
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Layers for LoRA
        lora_alpha=16,  # Scaling factor
        lora_dropout=0,  # No dropout for faster training
        bias="none",  # No additional bias terms
        random_state=3407,  # For reproducibility
        loftq_config=None,  # No specific LoFTQ configuration
    )
    return model_lora

def load_metrics():
    """
    Load various evaluation metrics.
    
    Returns:
        dict: A dictionary of loaded metric functions.
    """
    return {
        "bleu": load("bleu"),
        "meteor": load("meteor"),
        "accuracy": load("accuracy"),
        "rouge": load("rouge"),
        "bertscore": load("bertscore")
    }

def compute_metrics(pred, metrics):
    """
    Compute evaluation metrics based on predictions.
    
    Parameters:
        pred: The prediction output from the model.
        metrics: A dictionary containing loaded metric functions.
    
    Returns:
        dict: A dictionary of computed metric scores.
    """
    labels = pred.label_ids
    preds = pred.predictions

    accuracy = metrics["accuracy"].compute(predictions=preds, references=labels)["accuracy"]
    rouge = metrics["rouge"].compute(predictions=preds, references=labels, use_stemmer=True)
    rougeL_f1 = rouge['rougeL'].mid.fmeasure
    bert_score = metrics["bertscore"].compute(predictions=preds, references=labels, lang="en")["f1"].mean()
    bleu_score = metrics["bleu"].compute(predictions=preds, references=labels)["bleu"]
    meteor_score = metrics["meteor"].compute(predictions=preds, references=labels)["meteor"]

    return {
        "accuracy": accuracy,
        "rougeL_f1": rougeL_f1,
        "bert_score": bert_score,
        "bleu_score": bleu_score,
        "meteor_score": meteor_score
    }

class MetricLogger(TrainerCallback):
    """
    Callback to log and plot metrics during training.
    """
    def __init__(self):
        self.epochs = []
        self.accuracy = []
        self.relevance = []
        self.faithfulness = []
        self.bleu_scores = []
        self.meteor_scores = []
        self.time_taken = []
        self.epoch_start_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        self.time_taken.append(epoch_duration)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'epoch' in logs:
            self.epochs.append(logs['epoch'])
        if 'eval_accuracy' in logs:
            self.accuracy.append(logs['eval_accuracy'])
        if 'eval_rougeL_f1' in logs:
            self.relevance.append(logs['eval_rougeL_f1'])
        if 'eval_bert_score' in logs:
            self.faithfulness.append(logs['eval_bert_score'])
        if 'eval_bleu_score' in logs:
            self.bleu_scores.append(logs['eval_bleu_score'])
        if 'eval_meteor_score' in logs:
            self.meteor_scores.append(logs['eval_meteor_score'])

        # Debugging: print the metrics to check if they are being logged
        print(f"Logs: {logs}")

    def plot_metrics(self):
        # Plot Accuracy, Relevance, Faithfulness, BLEU, and METEOR
        if len(self.epochs) == len(self.accuracy) and len(self.accuracy) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.epochs, self.accuracy, label='Accuracy')
            plt.plot(self.epochs, self.relevance, label='Relevance (ROUGE-L)')
            plt.plot(self.epochs, self.faithfulness, label='Faithfulness (BERT Score)')
            plt.plot(self.epochs, self.bleu_scores, label='BLEU Score')
            plt.plot(self.epochs, self.meteor_scores, label='METEOR Score')
            plt.xlabel('Epochs')
            plt.ylabel('Scores')
            plt.title('Performance Metrics Over Epochs (LoRA Fine-Tuning)')
            plt.legend()
            plt.grid(True)

            plt.show()

        # Plot time taken per epoch
        if len(self.epochs) == len(self.time_taken) and len(self.time_taken) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.epochs, self.time_taken, label='Time Taken (seconds)', color='orange')
            plt.xlabel('Epochs')
            plt.ylabel('Time (seconds)')
            plt.title('Time Taken Per Epoch')
            plt.grid(True)

            plt.show()

def train_model(model, tokenized_train_dataset, tokenizer):
    """
    Fine-tune the model using the SFTTrainer.

    Parameters:
        model: The model to be fine-tuned.
        tokenized_train_dataset: The training dataset in tokenized format.
        tokenizer: The tokenizer associated with the model.

    Returns:
        trainer_stats: Statistics from the training process.
    """
    # Fine-tuning training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=60,
        learning_rate=5e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="/content/drive/MyDrive/NLX Data/lora",
        report_to="none",
    )

    # Initialize the MetricLogger callback
    metric_logger = MetricLogger()

    # Initialize the SFT Trainer with the model and tokenizer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train_dataset,
        args=training_args,
        compute_metrics=lambda pred: compute_metrics(pred, load_metrics()),
        callbacks=[metric_logger],  # Attach the metric logger callback
    )

    # Start fine-tuning
    trainer_stats = trainer.train()

    # After fine-tuning, plot the metrics
    metric_logger.plot_metrics()

    return trainer_stats

def save_model(model, tokenizer):
    """
    Save the fine-tuned model and tokenizer.

    Parameters:
        model: The fine-tuned model.
        tokenizer: The tokenizer associated with the model.
    """
    model.save_pretrained("/content/drive/MyDrive/NLX Data/lora/finetuned_model3")
    tokenizer.save_pretrained("/content/drive/MyDrive/NLX Data/lora/finetuned_model3")
    print("Model and tokenizer saved.")

def load_finetuned_model():
    """
    Load the fine-tuned model and tokenizer with LoRA adapters applied.

    Returns:
        model: The loaded model with LoRA adapters applied.
        tokenizer: The loaded tokenizer.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("unsloth/Phi-3.5-mini-instruct")
    tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/NLX Data/lora/finetuned_model3")
    model = PeftModel.from_pretrained(model, "/content/drive/MyDrive/NLX Data/lora/finetuned_model3")
    model.eval()
    print("Model and tokenizer loaded with LoRA adapters applied and set to evaluation mode.")
    return model, tokenizer
