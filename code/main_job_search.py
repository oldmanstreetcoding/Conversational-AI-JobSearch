# main_job_search.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppressing TensorFlow floating-point warnings

import preprocessing  # Import the preprocessing module
from model_selection import define_base_model
from finetuning import setup_model, load_metrics, train_model, save_model, load_finetuned_model
from evaluation import evaluate_queries, plot_metrics  # Import evaluation functions
from transformers import AutoTokenizer, FastLanguageModel  # Required for tokenization

# Display menu for user interaction
def display_menu():
    """
    Display the interactive menu for the user.
    Users can choose between different options to proceed with the job search model.
    """
    print("\n== Welcome To Conversational AI for Job Search ==")
    print("Select Which Step You Want to Proceed:")
    print("1. Data Preprocessing")
    print("2. Data Preprocessing with Summarization Process for Resume & Description Columns")
    print("3. Fine-tune the Phi-3 model on <Job Query, Job Description> pairs")
    print("4. Handle user queries for job searches based on different criteria (skills, location, salary).")
    print("5. Evaluate the model using ARES, RoGUE, BLEU, and METEOR metrics.")
    print("6. Exit")
    print("== Welcome To Conversational AI for Job Search ==")

def generate_response(query, model, tokenizer, max_new_tokens=64):
    """
    Generate a response based on the user query using the model and tokenizer.
    
    Parameters:
        query (str): The user query to generate a response for.
        model: The model to use for generating responses.
        tokenizer: The tokenizer to preprocess the input.
        max_new_tokens (int): The maximum number of new tokens to generate in the response.

    Returns:
        tuple: A tuple containing the generated response and the time taken to generate it.
    """
    # Prepare the input using the chat template
    messages = [{"from": "human", "value": query}]
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # Move tokenized input to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = formatted_input["input_ids"].to(device)
    attention_mask = formatted_input.get("attention_mask", None).to(device)

    # Prepare the model for inference
    FastLanguageModel.for_inference(model)

    # Start timing before running inference
    start_time = time.time()

    # Run the model with the input and generate a response
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        use_cache=True
    )

    # End timing after inference is done
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Decode the model's output
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return decoded_outputs, elapsed_time

def main():
    """
    Main function that runs the application.
    
    This function orchestrates the program's flow by displaying the menu and processing user input.
    Depending on the user's choice, it initiates various functionalities including data preprocessing,
    model training, and more.
    """
    display_menu()  # Display the interactive menu for user selection
    option = input("\nEnter your choice (1-6): ")  # Prompt user for their choice

    if option == '1':  # Preprocessing Option
        print("\nStarting the Data Preprocessing Step...")
        preprocessing.run_preprocessing(False)  # Call the preprocessing function from the module

    elif option == '2':  # Preprocessing with Summarization
        print("\nStarting the Data Preprocessing Step with Summarization Process...")
        preprocessing.run_preprocessing(True)  # Call the preprocessing function with summarization

    elif option == '3':  # Fine-tuning
        print("\nFine-tune the Phi-3 Model...")
        print("\nLoad the job description and CV datasets....")

        train_path = "data/conversational_train.csv"  # Update path as necessary
        test_path = "data/conversational_test.csv"    # Update path as necessary

        if os.path.exists(train_path):
            print(f"\nFine-tuning dataset exists.")
            model, tokenizer, tokenized_train_dataset = define_base_model()
            # Train the model here or call a training function
            train_model(model, tokenized_train_dataset, tokenizer)

        else:
            print(f"\nFine-tuning dataset does not exist.")
            preprocessing.run_preprocessing(False)  # Call the preprocessing function to create the datasets

    elif option == '4':  # Handle user queries
        print("\nHandling user queries for job searches...")
        
        # Load the fine-tuned model and tokenizer
        model, tokenizer = load_finetuned_model()
        
        # Example job queries to test the model
        job_queries = [
            "What jobs are available for machine learning engineers?",
            "What jobs are open in Finland?",
            "Which jobs pay over $150K?",
            "What jobs are available for passionate Python developers with 5 years of experience?"
        ]

        for query in job_queries:
            response, time_taken = generate_response(query, model, tokenizer)
            print(f"Query: {query}\nResponse: {response}\nTime Taken: {time_taken:.2f} seconds\n")

    elif option == '5':  # Evaluate model
        print("\nEvaluating the model...")
        # Load the fine-tuned model and tokenizer
        model, tokenizer = load_finetuned_model()
        
        # Define example queries and references for evaluation
        queries_and_references = {
            'skills': {
                'prompt': "What roles are available for machine learning engineer?",
                'reference': 'Job available in Location: Québec city, qc Salary: over $150K Job-Title: senior machine learning engineer...'
            },
            'salary': {
                'prompt': "Which jobs pay over $150K?",
                'reference': 'Job available with a salary of over $150K Location: Calgary, ab Job-Title: intermediate full stack developer...'
            },
            'experience': {
                'prompt': "What jobs are available for passionate python developer with 5 years of experience?",
                'reference': 'Job available in Location: San Francisco, ca Salary: over $75K...'
            },
            'location': {
                'prompt': "What jobs are open in Finland?",
                'reference': 'Job available in Finland Job-Title: senior frontend developer...'
            }
        }

        # Evaluate the queries
        evaluation_results = evaluate_queries(queries_and_references, model, tokenizer)

        # Prepare metrics for plotting
        query_types = [result['query_type'] for result in evaluation_results]
        bert_scores = [result['bert_score'] for result in evaluation_results]
        rouge_scores = [result['rouge_scores']['rougeL'].fmeasure for result in evaluation_results]
        bleu_scores = [result['bleu_score'] for result in evaluation_results]
        meteor_scores = [result['meteor_score'] for result in evaluation_results]

        # Plot metrics
        plot_metrics(query_types, bert_scores, rouge_scores, bleu_scores, meteor_scores)

    elif option == '6':  # Exit option
        print("\nExiting the program. Goodbye!\n")  # Exit message
        return  # Exit the program

    else:  # Handle invalid options
        print("\nInvalid option! Please choose between 1 and 6.\n")  # Prompt for valid input

if __name__ == '__main__':  # Check if this script is being run directly
    main()  # Execute the main function to start the program
