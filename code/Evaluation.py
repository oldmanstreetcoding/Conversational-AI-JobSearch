# evaluation.py

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import evaluate
from transformers import AutoTokenizer, TextStreamer
from rouge_score import rouge_scorer
from bert_score import score

# Load BLEU and METEOR metrics
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

def generate_and_score(prompt, reference, model, tokenizer):
    """
    Generate a response based on the prompt and calculate evaluation scores.

    Parameters:
        prompt (str): The query to generate a response for.
        reference (str): The ground truth response to compare against.
        model: The model used for generating responses.
        tokenizer: The tokenizer associated with the model.

    Returns:
        tuple: Generated text, BERT score, ROUGE scores, BLEU score, METEOR score, and time taken.
    """
    # Start the timer
    start_time = time.time()

    # Prepare input for inference
    messages = [{"from": "human", "value": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate response
    output_ids = model.generate(input_ids=inputs, max_new_tokens=128, use_cache=True)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Calculate BERT Score
    P, R, F1 = score([generated_text], [reference], lang="en", verbose=False)
    bert_score = F1.item()

    # Calculate ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, generated_text)

    # Calculate BLEU Score
    predictions = [generated_text]
    references = [[reference]]
    bleu_results = bleu.compute(predictions=predictions, references=references)
    bleu_score = bleu_results['bleu']

    # Calculate METEOR Score
    meteor_results = meteor.compute(predictions=predictions, references=[reference])
    meteor_score = meteor_results['meteor']

    # Stop the timer
    end_time = time.time()
    time_taken = end_time - start_time

    return generated_text, bert_score, rouge_scores, bleu_score, meteor_score, time_taken

def evaluate_queries(queries_and_references, model, tokenizer):
    """
    Evaluate multiple queries and return their evaluation results.

    Parameters:
        queries_and_references (dict): Dictionary containing queries and their corresponding references.
        model: The model used for generating responses.
        tokenizer: The tokenizer associated with the model.

    Returns:
        list: Evaluation results for each query.
    """
    results = []
    for query_type, query_info in queries_and_references.items():
        prompt, reference = query_info['prompt'], query_info['reference']
        print(f"\nEvaluating {query_type} query: {prompt}")
        generated_text, bert_score, rouge_scores, bleu_score, meteor_score, time_taken = generate_and_score(prompt, reference, model, tokenizer)
        results.append({
            'query_type': query_type,
            'prompt': prompt,
            'generated_text': generated_text,
            'bert_score': bert_score,
            'rouge_scores': rouge_scores,
            'bleu_score': bleu_score,
            'meteor_score': meteor_score,
            'time_taken': time_taken
        })
    return results

def plot_metrics(query_types, bert_scores, rouge_scores, bleu_scores, meteor_scores):
    """
    Plot evaluation metrics for the queries.

    Parameters:
        query_types (list): List of query types.
        bert_scores (list): List of BERT scores.
        rouge_scores (list): List of ROUGE scores.
        bleu_scores (list): List of BLEU scores.
        meteor_scores (list): List of METEOR scores.
    """
    # Plot BERT, ROUGE, BLEU, and METEOR scores
    ind = np.arange(len(query_types))  # the x locations for the groups
    width = 0.15  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(ind - width * 1.5, bert_scores, width, label='BERT Score', color='blue')
    ax.bar(ind - width / 2, rouge_scores, width, label='ROUGE Score', color='green')
    ax.bar(ind + width / 2, bleu_scores, width, label='BLEU Score', color='orange')
    ax.bar(ind + width * 1.5, meteor_scores, width, label='METEOR Score', color='red')

    ax.set_xlabel('Query Type')
    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Metrics Comparison')
    ax.set_xticks(ind)
    ax.set_xticklabels(query_types)
    ax.legend()

    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/NLX Data/visualizations/evaluation_metrics_comparison.png')
    plt.show()
