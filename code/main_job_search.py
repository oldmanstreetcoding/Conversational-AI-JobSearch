# Display menu for user interaction
def display_menu():
    print("\n== Welcome To Conversational AI for Job Search ==")
    print("Select Which Step You Want to Proceed:")
    print("1. Data Preprocessing") 
    print("2. Exit") 
    print("== Welcome To Conversational AI for Job Search ==")

def main():

    display_menu()
    option = input("\nEnter your choice (1-2): ")

    if option == '2':  # Exit option
        print("\nExiting the program. Goodbye!\n")
        return
    
if __name__ == '__main__':
    main()