import os
import argparse
from dotenv import load_dotenv

from shared_memory import SharedMemory
from classifier_agent import ClassifierAgent # This is the new orchestrating classifier
from json_agent import JSONAgent
from email_agent import EmailAgent
# from pdf_agent import PDFAgent # If you create one

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# TEXTRAZOR_API_KEY = os.getenv("TEXTRAZOR_API_KEY") # If using TextRazor

def run_system(input_data_source, is_file_path=False):
    if not GEMINI_API_KEY: # and not TEXTRAZOR_API_KEY (if offering choice)
        print("Error: GEMINI_API_KEY (or relevant API key) not found. Please set it in .env file.")
        return

    # Initialize components
    shared_memory = SharedMemory()
    json_agent = JSONAgent(target_schema={ # Define your schema
        'invoice_number': True, 'date': True, 'amount': True, 'vendor': True,
        'customer_name': False, 'item_description': False
    })
    email_agent = EmailAgent()
    # pdf_agent = PDFAgent() # Initialize if you have one

    try:
        # ClassifierAgent now takes other agents and memory as dependencies
        classifier_orchestrator = ClassifierAgent(
            gemini_api_key=GEMINI_API_KEY, # or textrazor_api_key=TEXTRAZOR_API_KEY
            json_agent=json_agent,
            email_agent=email_agent,
            # pdf_agent=pdf_agent,
            shared_memory=shared_memory
        )
    except ValueError as e:
        print(f"Error initializing ClassifierAgent: {e}")
        return
    except Exception as e:
        print(f"Critical Error: Classifier Orchestrator could not be initialized: {e}")
        # Log this critical failure to shared memory if possible, or to a local log file
        # shared_memory.log(str(uuid.uuid4()), {"system_error": "ClassifierInitFail", "message": str(e)})
        return


    # ClassifierAgent handles format detection, intent, routing, and detailed logging
    print(f"\nProcessing input: '{input_data_source[:100]}...' (Path: {is_file_path})")
    result = classifier_orchestrator.process_input(input_data_source, input_is_path=is_file_path)

    print("\n--- Main System Processing Summary ---")
    print(f"  Status: {result.get('status')}")
    print(f"  Determined Format: {result.get('format')}")
    print(f"  Determined Intent: {result.get('intent')}")
    print(f"  Conversation ID: {result.get('conversation_id')}")
    if result.get('anomalies'):
        print(f"  Anomalies: {result.get('anomalies')}")
    # print(f"  Agent Output Snippet: {str(result.get('output'))[:200]}...") # Optional: print snippet of output
    
    print("\nTo view full logs, check Redis for conversation ID:", result.get('conversation_id'))
    print("--- End of Processing ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Document Processing System (Classifier as Orchestrator)")
    parser.add_argument("input_source", 
                        help="File path (e.g., 'samples/invoice.pdf', 'samples/email.txt') OR raw string content for JSON/Email if not a file.")
    # No --type argument needed anymore

    args = parser.parse_args()

    input_is_file = False
    input_content_or_path = args.input_source

    # Heuristic: if the input_source string looks like a path that exists, treat it as a path.
    # This is basic; more robust path detection might be needed for all edge cases.
    if os.path.exists(args.input_source) and (
        args.input_source.lower().endswith(('.pdf', '.txt', '.eml', '.json')) or 
        os.path.isfile(args.input_source) # More general check
    ):
        input_is_file = True
        print(f"Input '{args.input_source}' detected as a file path.")
    else:
        # If it doesn't exist as a file or doesn't look like a typical file extension we handle,
        # assume it's raw string content (e.g., a JSON string pasted directly).
        # This part might need refinement based on how you expect to pass raw strings.
        # For simplicity, if it's not an existing file, it's treated as raw content.
        input_is_file = False
        print(f"Input '{args.input_source[:50]}...' treated as raw string content.")


    # Sample file creation logic (from previous main.py, ensure paths are correct)
    sample_dir = "sample_inputs"
    if not os.path.exists(sample_dir): os.makedirs(sample_dir)
    # ... (your sample PDF, Email, JSON file creation logic here, same as before) ...
    # Make sure to create sample_inputs/sample_invoice.pdf, sample_inputs/sample_rfq.eml, sample_inputs/sample_complaint.json

    sample_pdf_path = os.path.join(sample_dir, "sample_invoice.pdf")
    if not os.path.exists(sample_pdf_path):
        try:
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(sample_pdf_path)
            c.drawString(72, 720, "Sample Invoice: Pay $100 for INV001 by due date.")
            c.save()
        except: pass # Ignore if reportlab not present

    sample_email_path = os.path.join(sample_dir, "sample_rfq.eml")
    if not os.path.exists(sample_email_path):
        with open(sample_email_path, "w") as f:
            f.write("From: test@example.com\nSubject: RFQ for Widgets\nBody: Please provide a quote for 1000 widgets.")

    sample_json_path = os.path.join(sample_dir, "sample_complaint.json")
    if not os.path.exists(sample_json_path):
        with open(sample_json_path, "w") as f:
            json.dump({"invoice_number": "INV123", "complaint_details": "Product broke."}, f)


    print("\n--- Running System with CLI Argument ---")
    run_system(input_content_or_path, is_file_path=input_is_file)