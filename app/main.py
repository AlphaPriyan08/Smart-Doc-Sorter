import os
import argparse
from dotenv import load_dotenv

from shared_memory import SharedMemory
from classifier_agent import ClassifierAgent
from json_agent import JSONAgent
from email_agent import EmailAgent

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def run_system(input_data_source, is_file_path_param): 
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found. Please set it in .env file.")
        return

    shared_memory = SharedMemory()
    json_agent = JSONAgent(target_schema={
        'invoice_number': True, 'date': True, 'amount': True, 'vendor': True,
        'customer_name': False, 'item_description': False, 'complaint_details': False
    })
    email_agent = EmailAgent()

    try:
        classifier_orchestrator = ClassifierAgent(
            gemini_api_key=GEMINI_API_KEY,
            json_agent=json_agent,
            email_agent=email_agent,
            shared_memory=shared_memory
        )
    except ValueError as e:
        print(f"Error initializing ClassifierAgent: {e}")
        return
    except Exception as e:
        print(f"Critical Error: Classifier Orchestrator could not be initialized: {e}")
        return

    print(f"\nSYSTEM: Processing input: '{str(input_data_source)[:100]}...' (Path: {is_file_path_param})")
    result = classifier_orchestrator.process_input(input_data_source, input_is_path=is_file_path_param)

    print("\n--- Main System Processing Summary ---")
    print(f"  Status: {result.get('status')}")
    print(f"  Determined Format: {result.get('format')}")
    print(f"  Determined Intent: {result.get('intent')}")
    print(f"  Conversation ID: {result.get('conversation_id')}")
    if result.get('anomalies'):
        print(f"  Anomalies: {result.get('anomalies')}")
    print("\nTo view full logs, check Redis for conversation ID:", result.get('conversation_id'))
    print("--- End of Processing for this input ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Document Processing System")
    parser.add_argument("input_source",
                        help="File path (e.g., 'sample_inputs/invoice.pdf') OR raw string content.")
    args = parser.parse_args()

    raw_cli_argument = args.input_source
    
    processed_input_for_system = raw_cli_argument
    is_determined_to_be_a_file = False

    print(f"DEBUG: CLI argument received: '{raw_cli_argument}'")
    current_cwd = os.getcwd()
    print(f"DEBUG: Current Working Directory (CWD): '{current_cwd}'")

    potential_full_path = os.path.abspath(raw_cli_argument) 

    if os.path.isfile(raw_cli_argument):
        is_determined_to_be_a_file = True
        processed_input_for_system = potential_full_path 
        print(f"SYSTEM: Input '{raw_cli_argument}' resolved to existing file path: '{processed_input_for_system}'.")
    else:
        print(f"DEBUG: os.path.isfile('{raw_cli_argument}' relative to CWD '{current_cwd}') returned False.")
        print(f"DEBUG: Absolute path checked: '{potential_full_path}'")
        if (os.sep in raw_cli_argument or ('.' in os.path.basename(raw_cli_argument) and len(os.path.basename(raw_cli_argument)) > 1)) and len(raw_cli_argument) > 3 :
            print(f"WARNING: Input '{raw_cli_argument}' looks like a file path but was NOT FOUND or is not a regular file.")
            print(f"         Please ensure the file exists at this location relative to CWD, or provide an absolute path.")
            print(f"         Treating as raw string content due to file not being found.")
        else:
            print(f"SYSTEM: Input '{raw_cli_argument[:50]}...' treated as raw string content (does not appear to be a path or file not found).")
    
    print("\n--- Running System with CLI Argument ---")
    run_system(processed_input_for_system, is_file_path_param=is_determined_to_be_a_file)