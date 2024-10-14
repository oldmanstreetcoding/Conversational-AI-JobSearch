import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppressing TensorFlow floating-point warnings

import preprocessing  # Import the preprocessing module
from model_selection import define_base_model

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

def main():
    """
    Main function that runs the menu and allows the user to choose preprocessing steps or exit.
    """
    display_menu()
    option = input("\nEnter your choice (1-6): ")

    if option == '1':  # Preprocessing Option
        print("\nStarting the Data Preprocessing Step...")
        preprocessing.run_preprocessing(False)  # Call the preprocessing function from the module

    elif option == '2':
        print("\nStarting the Data Preprocessing Step with Summarization Process...")
        preprocessing.run_preprocessing(True)  # Call the preprocessing function from the module

    elif option == '3':
        print("\nFine-tune the Phi-3 Model...")
        print("\nLoad the job description and CV datasets....")

        # Define the file path
        train_path = "data/conversational_train.csv"  # You can also use an absolute path if needed

        # Check if the file exists
        if os.path.exists(train_path):
            print(f"\nFine-tuning dataset exists.")
            # model, tokenizer, tokenized_train_dataset = define_base_model()
            define_base_model()

        else:
            print(f"\nFine-tuning dataset does not exist.")
            preprocessing.run_preprocessing(False)  # Call the preprocessing function from the module

    elif option == '6':  # Exit option
        print("\nExiting the program. Goodbye!\n")
        return

    else:
        print("\nInvalid option! Please choose between 1 and 2.\n")

if __name__ == '__main__':
    main()
