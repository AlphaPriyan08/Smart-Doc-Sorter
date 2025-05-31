import google.generativeai as genai
import uuid
import json
import os
from pdf_parser import extract_text_from_pdf

class ClassifierAgent:
    def __init__(self, gemini_api_key, json_agent, email_agent, shared_memory):
        self.json_agent = json_agent
        self.email_agent = email_agent
        self.shared_memory = shared_memory
        self.conversation_id = None
       
        genai.configure(api_key=gemini_api_key)
        self.generation_config = {"temperature": 0.1, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        try:
            model_to_use = "models/gemma-3n-e4b-it"

            self.model = genai.GenerativeModel(
                model_name=model_to_use,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            print(f"ClassifierAgent: Attempting to initialize with model: {model_to_use}")
        except Exception as e:
            print(f"ClassifierAgent: Failed to initialize Gemini Pro model: {e}")
            self.model = None

    def _determine_format(self, raw_input_data, input_is_path=False):
        content_for_analysis = ""
        if input_is_path:
            file_path = raw_input_data
            if not os.path.exists(file_path):
                return "Error_FileNotFound", None, f"File not found: {file_path}"
            
            if file_path.lower().endswith(".pdf"):
                try:
                    text_content = extract_text_from_pdf(file_path)
                    if not text_content:
                        return "PDF", "", f"Empty content from PDF: {file_path}" 
                    return "PDF", text_content, None
                except Exception as e:
                    return "Error_PDFParsing", None, f"Error parsing PDF {file_path}: {e}"
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content_for_analysis = f.read()
                except Exception as e:
                    return "Error_FileRead", None, f"Error reading file {file_path}: {e}"
        else:
            content_for_analysis = raw_input_data

        if not content_for_analysis.strip():
            return "Unknown_EmptyInput", "", "Input content is empty or whitespace."

        stripped_content = content_for_analysis.strip()
        if (stripped_content.startswith("{") and stripped_content.endswith("}")) or \
           (stripped_content.startswith("[") and stripped_content.endswith("]")):
            try:
                json.loads(stripped_content)
                return "JSON", stripped_content, None
            except json.JSONDecodeError:
                pass 

        content_lower = content_for_analysis.lower()
        if "from:" in content_lower and ("subject:" in content_lower or "to:" in content_lower or "date:" in content_lower):
            return "Email", content_for_analysis, None
        
        if input_is_path: 
             return "TextFile", content_for_analysis, None

        return "Text", content_for_analysis, None


    def _classify_intent_with_gemini(self, text_content):
        if not self.model:
            print("ClassifierAgent: Model not available for intent classification.")
            return "Error_ClientNotInitialized"
        if not text_content or not text_content.strip(): 
            return "Unknown_EmptyContent"

        possible_intents = ["Invoice", "RFQ", "Complaint", "Regulation", "Other"]
        prompt = f"""
        Analyze the following text and classify its primary business intent.
        Possible intents: {', '.join(possible_intents)}.
        - "Invoice": billing, payment requests, financial statements asking for payment.
        - "RFQ" (Request for Quotation): seeking pricing, proposals for goods or services.
        - "Complaint": expressing dissatisfaction, problems, issues, or grievances.
        - "Regulation": legal documents, compliance requirements, official rules, terms and conditions.
        - "Other": if none of the above clearly match or if the text is too generic.

        Return your answer strictly as a JSON object with a single key "intent" and the classified intent as its value.
        For example: {{"intent": "Invoice"}}

        Text to analyze:
        ---
        {text_content[:4000]} 
        ---
        """
        response = None 
        try:
            response = self.model.generate_content(prompt)
            
            cleaned_response_text = response.text.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:]
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-3]
            cleaned_response_text = cleaned_response_text.strip()
            
            response_json = json.loads(cleaned_response_text)
            intent = response_json.get("intent", "Other")

            return intent if intent in possible_intents else "Other"

        except json.JSONDecodeError as e:
            response_text_snippet = response.text[:200] if response and hasattr(response, 'text') else 'No response text or response object not created.'
            print(f"ClassifierAgent: Error decoding JSON from Gemini response: {e}. Response was: {response_text_snippet}")
            return "Error_ParsingResponse"
        except Exception as e:
            feedback_message = ""
            if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                 feedback_message = f" Prompt blocked due to: {response.prompt_feedback.block_reason}."
            elif response is None and ("NotFound" in str(e) or "is not found for API version" in str(e)): 
                feedback_message = f" Model not found or API issue. Current model name in code: '{self.model.model_name if self.model else 'None'}'. Check available models."


            print(f"ClassifierAgent: Error during Gemini intent classification: {e}.{feedback_message}")
            return "Error_IntentAPI"


    def process_input(self, raw_input_data, input_is_path=False):
        self.conversation_id = str(uuid.uuid4())
        initial_log_data = {
            "input_source_type": "file_path" if input_is_path else "raw_string",
            "input_identifier": raw_input_data if input_is_path else raw_input_data[:100] + "...",
            "conversation_id": self.conversation_id
        }

        doc_format, text_content_for_intent, format_error_message = self._determine_format(raw_input_data, input_is_path)
        
        if "Error_" in doc_format:
            print(f"ClassifierAgent: Error in format determination: {format_error_message}")
            initial_log_data.update({"status": "FAILED_FORMAT_DETERMINATION", "error": format_error_message, "determined_format": doc_format})
            self.shared_memory.log(self.conversation_id, initial_log_data)
            return {"status": "Error", "message": format_error_message, "conversation_id": self.conversation_id}
        
        if doc_format != "Unknown_EmptyInput" and not text_content_for_intent and format_error_message:
             print(f"ClassifierAgent: Format {doc_format} determined, but content is effectively empty or had issues: {format_error_message}")

        initial_log_data["determined_format"] = doc_format

        if doc_format == "Unknown_EmptyInput":
            intent = "Unknown_EmptyInput"
        elif "Error_" in doc_format :
            intent = "Error_Preprocessing"
        else:
            intent = "Error_NoIntentClassifierConfigured"

        initial_log_data["determined_intent"] = intent
        print(f"ClassifierAgent: Format={doc_format}, Intent={intent}, ConvID={self.conversation_id}")
        self.shared_memory.log(self.conversation_id, initial_log_data)


        agent_output = {}
        anomalies = []
        status = "Processed"

        final_log_payload = {
            "step": "agent_processing_result",
            "source_input_type": doc_format, 
            "intent": intent,
        }

        if "Error_" in intent or intent == "Unknown_EmptyInput" or intent == "Unknown_EmptyContent":
            status = "Failed_Intent_Classification_Or_Empty"
            final_log_payload["error_message"] = f"Intent classification issue or empty content: {intent}"
            final_log_payload["extracted_values"] = {"text_snippet": text_content_for_intent[:200] if text_content_for_intent else "N/A"}
        
        elif doc_format == 'JSON':
            try:
                json_data_for_agent = json.loads(text_content_for_intent)
                agent_output, anomalies = self.json_agent.process(json_data_for_agent)
                final_log_payload["extracted_values"] = agent_output
                final_log_payload["anomalies"] = anomalies
            except json.JSONDecodeError as e:
                status = "Failed_JSON_Processing"
                final_log_payload["error_message"] = f"Invalid JSON for JSONAgent: {e}"
                final_log_payload["anomalies"] = [f"Invalid JSON structure for agent: {e}"]
                final_log_payload["extracted_values"] = {"raw_json_string_snippet": text_content_for_intent[:200]}
            except Exception as e:
                status = "Failed_JSON_Processing_Unexpected"
                final_log_payload["error_message"] = f"Unexpected error in JSONAgent processing: {e}"
                final_log_payload["extracted_values"] = {"raw_json_string_snippet": text_content_for_intent[:200]}

        elif doc_format == 'Email':
            try:
                agent_output = self.email_agent.extract(text_content_for_intent)
                final_log_payload["extracted_values"] = agent_output
            except Exception as e:
                status = "Failed_Email_Processing_Unexpected"
                final_log_payload["error_message"] = f"Unexpected error in EmailAgent processing: {e}"
                final_log_payload["extracted_values"] = {"raw_email_string_snippet": text_content_for_intent[:200]}

        elif doc_format == 'PDF':
            agent_output = {"extracted_pdf_text_snippet": text_content_for_intent[:500] if text_content_for_intent else "No text extracted or PDF empty."}
            final_log_payload["extracted_values"] = agent_output
        
        elif doc_format == 'TextFile' or doc_format == 'Text':
            agent_output = {"text_content_snippet": text_content_for_intent[:500] if text_content_for_intent else "No text content."}
            final_log_payload["extracted_values"] = agent_output

        else: 
            status = "Failed_UnknownFormatForRouting"
            final_log_payload["error_message"] = f"Cannot route, unknown or unhandled format: {doc_format}"
            final_log_payload["extracted_values"] = {"raw_content_snippet": text_content_for_intent[:200] if text_content_for_intent else "N/A"}
        
        self.shared_memory.log(self.conversation_id, final_log_payload)
        print(f"ClassifierAgent: Processing complete for ConvID {self.conversation_id}. Status: {status}")

        return {"status": status, "format": doc_format, "intent": intent, "output": agent_output, "anomalies": anomalies, "conversation_id": self.conversation_id}