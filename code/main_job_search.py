# main_job_search.py

import preprocessing  # Import the preprocessing module

# Display menu for user interaction
def display_menu():
    """
    Display the interactive menu for the user.
    Users can choose between different options to proceed with the job search model.
    """
    print("\n== Welcome To Conversational AI for Job Search ==")
    print("Select Which Step You Want to Proceed:")
    print("1. Data Preprocessing")
    print("2. Exit")
    print("== Welcome To Conversational AI for Job Search ==")

def main():
    """
    Main function that runs the menu and allows the user to choose preprocessing steps or exit.
    """
    display_menu()
    option = input("\nEnter your choice (1-2): ")

    if option == '1':  # Preprocessing Option
        print("\nStarting the Data Preprocessing Step...")
        preprocessing.run_preprocessing()  # Call the preprocessing function from the module

    elif option == '2':  # Exit option
        print("\nExiting the program. Goodbye!\n")
        return

    else:
        print("\nInvalid option! Please choose between 1 and 2.\n")

if __name__ == '__main__':
    main()
