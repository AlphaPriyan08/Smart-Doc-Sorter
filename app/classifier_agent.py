import google.generativeai as genai
import uuid
import json
import os
import re 
from pdf_parser import extract_text_from_pdf

class ClassifierAgent:
    def __init__(self, gemini_api_key, json_agent, email_agent, shared_memory):
        self.json_agent = json_agent
        self.email_agent = email_agent
        self.shared_memory = shared_memory
        self.conversation_id = None
        self.model = None

        if not gemini_api_key:
            print("ClassifierAgent ERROR: Gemini API key is required but not provided.")
            return

        try:
            genai.configure(api_key=gemini_api_key)
            model_to_use = "gemini-1.5-flash-latest"
            print(f"ClassifierAgent: Attempting to initialize with model: '{model_to_use}'")

            self.safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            self.generation_config = {
                "temperature": 0.2,
                "top_p": 1.0,
                "top_k": 1,
                "max_output_tokens": 256,
            }
            self.model = genai.GenerativeModel(
                model_name=model_to_use,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            print(f"ClassifierAgent: Model '{model_to_use}' initialized successfully.")
        except Exception as e:
            print(f"ClassifierAgent CRITICAL ERROR: Failed to initialize model '{model_to_use}': {e}")
            self.model = None

    def _determine_format(self, raw_input_data, input_is_path=False):
        print(f"\nDEBUG_DETERMINE_FORMAT --- Start ---")
        print(f"DEBUG_DETERMINE_FORMAT: Input `raw_input_data` (type: {type(raw_input_data)}): '{str(raw_input_data)[:150]}...'")
        print(f"DEBUG_DETERMINE_FORMAT: Input `input_is_path`: {input_is_path}")

        content_for_analysis = ""
        file_path_for_logging = None 

        if input_is_path:
            file_path = raw_input_data
            file_path_for_logging = file_path
            print(f"DEBUG_DETERMINE_FORMAT: Processing as FILE PATH: '{file_path}'")

            if not os.path.exists(file_path):
                print(f"DEBUG_DETERMINE_FORMAT: File NOT FOUND: '{file_path}'")
                return "Error_FileNotFound", None, f"File not found: {file_path}"
            print(f"DEBUG_DETERMINE_FORMAT: File EXISTS: '{file_path}'")

            is_pdf_extension = file_path.lower().endswith(".pdf")
            print(f"DEBUG_DETERMINE_FORMAT: Checking for .pdf extension: {is_pdf_extension}")
            if is_pdf_extension:
                print(f"DEBUG_DETERMINE_FORMAT: Path has .pdf extension. Attempting PDF processing.")
                try:
                    text_content = extract_text_from_pdf(file_path) 
                    print(f"DEBUG_DETERMINE_FORMAT: `extract_text_from_pdf` returned (type: {type(text_content)}): '{str(text_content)[:100] if text_content else 'None or Empty'}'")
                    if text_content is None: 
                        print(f"DEBUG_DETERMINE_FORMAT: PDF parser returned None. Classifying as Error_PDFParsing.")
                        return "Error_PDFParsing", None, f"Critical error parsing PDF (parser returned None): {file_path}"

                    print(f"DEBUG_DETERMINE_FORMAT: Successfully processed as PDF. Returning 'PDF'.")
                    return "PDF", text_content, None
                except Exception as e:
                    print(f"DEBUG_DETERMINE_FORMAT: EXCEPTION during PDF processing for '{file_path}': {e}")
                    return "Error_PDFParsing", None, f"Error during PDF processing for {file_path}: {e}"
            else: 
                print(f"DEBUG_DETERMINE_FORMAT: Path does NOT have .pdf extension. Reading content as plain text.")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content_for_analysis = f.read()
                    print(f"DEBUG_DETERMINE_FORMAT: Successfully read content from non-PDF file (length: {len(content_for_analysis)}). Snippet: '{content_for_analysis[:150]}...'")
                except Exception as e:
                    print(f"DEBUG_DETERMINE_FORMAT: EXCEPTION reading non-PDF file '{file_path}': {e}")
                    return "Error_FileRead", None, f"Error reading content from file {file_path}: {e}"
        else: 
            content_for_analysis = raw_input_data
            print(f"DEBUG_DETERMINE_FORMAT: Processing as RAW STRING data (length: {len(content_for_analysis)}). Snippet: '{content_for_analysis[:150]}...'")

        print(f"DEBUG_DETERMINE_FORMAT: Proceeding to common content checks.")
        if not content_for_analysis.strip():
            print(f"DEBUG_DETERMINE_FORMAT: Content for analysis is EMPTY or WHITESPACE.")
            if input_is_path:
                print(f"DEBUG_DETERMINE_FORMAT: Empty content from a non-PDF file. Returning 'TextFile'.")
                return "TextFile", "", f"Empty text content from non-PDF file: {file_path_for_logging or 'unknown path'}"
            print(f"DEBUG_DETERMINE_FORMAT: Empty content from raw input. Returning 'Unknown_EmptyInput'.")
            return "Unknown_EmptyInput", "", "Input content is empty or whitespace."

        stripped_content = content_for_analysis.strip()
        print(f"DEBUG_DETERMINE_FORMAT: Content for JSON/Email check (length {len(stripped_content)}): '{stripped_content[:100]}...'")

        is_json_shape = (stripped_content.startswith("{") and stripped_content.endswith("}")) or \
                       (stripped_content.startswith("[") and stripped_content.endswith("]"))
        print(f"DEBUG_DETERMINE_FORMAT: Checking for JSON shape: {is_json_shape}")
        if is_json_shape:
            try:
                json.loads(stripped_content)
                print(f"DEBUG_DETERMINE_FORMAT: Successfully parsed as JSON. Returning 'JSON'.")
                return "JSON", stripped_content, None
            except json.JSONDecodeError:
                print(f"DEBUG_DETERMINE_FORMAT: Content has JSON-like shape but FAILED to parse as JSON.")
                pass 
            
        print(f"DEBUG_DETERMINE_FORMAT: Checking for Email format.")
        content_lower = content_for_analysis.lower()
        has_from = "from:" in content_lower
        has_subject = "subject:" in content_lower
        has_to = "to:" in content_lower
        has_date = "date:" in content_lower
        has_message_id = "message-id:" in content_lower
        has_mime_version = "mime-version:" in content_lower
        print(f"DEBUG_DETERMINE_FORMAT: Email heuristic checks: from:{has_from}, subject:{has_subject}, to:{has_to}, date:{has_date}, msg_id:{has_message_id}, mime:{has_mime_version}")

        if has_from and (has_subject or has_to or has_date or has_message_id or has_mime_version):
            print(f"DEBUG_DETERMINE_FORMAT: Classified as Email. Returning 'Email'.")
            return "Email", content_for_analysis, None
        else:
            print(f"DEBUG_DETERMINE_FORMAT: Did NOT meet Email classification criteria.")

        if input_is_path:
            print(f"DEBUG_DETERMINE_FORMAT: Input was a file path, not PDF/JSON/Email by content. Returning 'TextFile'.")
            return "TextFile", content_for_analysis, None

        print(f"DEBUG_DETERMINE_FORMAT: Input was raw string, not JSON/Email by content. Returning 'Text'.")
        print(f"DEBUG_DETERMINE_FORMAT --- End ---")
        return "Text", content_for_analysis, None

    def _classify_intent_with_gemini(self, text_content):
        if not self.model:
            print("ClassifierAgent: Model not available for intent classification (was not initialized).")
            return "Error_ClientNotInitialized"
        if not text_content or not text_content.strip():
            print("ClassifierAgent: Content for intent classification is empty. Returning 'Unknown_EmptyContent'.")
            return "Unknown_EmptyContent"

        possible_intents = ["Invoice", "RFQ", "Complaint", "Regulation", "Other"]
        prompt = f"""
        Analyze the following text to determine its primary business intent.
        Choose one intent from this list: {possible_intents}.
        - "Invoice": Documents related to billing, payment requests, or financial statements requiring payment.
        - "RFQ" (Request for Quotation): Communications seeking pricing, proposals for goods/services, or vendor quotes.
        - "Complaint": Expressions of dissatisfaction, problems, issues, or grievances regarding products or services.
        - "Regulation": Official rules, legal documents, compliance requirements, terms and conditions, or policy statements.
        - "Other": If the text does not clearly fit any of the above categories, is too generic, or if the text is very short and lacks clear business context.

        Your response MUST be a valid JSON object with a single key "intent" and the classified intent as its string value.
        Example: {{"intent": "Invoice"}}

        Text to analyze:
        ---
        {text_content[:8000]}
        ---
        Respond now with ONLY the JSON object.
        """
        response = None
        try:
            print(f"ClassifierAgent: Sending text (snippet: '{text_content[:100].replace(os.linesep, ' ')}...') to LLM for intent classification.")
            response = self.model.generate_content(prompt)

            if not response or not hasattr(response, 'text') or not response.text:
                feedback = ""
                if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                    feedback = f" Prompt blocked. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"
                print(f"ClassifierAgent: LLM returned an empty response or no text part.{feedback}")
                return "Error_EmptyResponseFromAPI" if not feedback else "Error_PromptBlocked"

            cleaned_response_text = response.text.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:]
                if cleaned_response_text.endswith("```"):
                    cleaned_response_text = cleaned_response_text[:-3]
            elif cleaned_response_text.startswith("```"):
                 cleaned_response_text = cleaned_response_text[3:]
                 if cleaned_response_text.endswith("```"):
                    cleaned_response_text = cleaned_response_text[:-3]
            cleaned_response_text = cleaned_response_text.strip()
            
            try:
                response_json = json.loads(cleaned_response_text)
            except json.JSONDecodeError:
                match = re.search(r'\{\s*"intent"\s*:\s*"[^"]+"\s*\}', cleaned_response_text)
                if match:
                    try:
                        response_json = json.loads(match.group(0))
                    except json.JSONDecodeError as e_inner:
                        print(f"ClassifierAgent: Error decoding JSON even after regex match from LLM response: {e_inner}. Cleaned response was: '{cleaned_response_text}'")
                        return "Error_ParsingResponse"
                else:
                    print(f"ClassifierAgent: Error decoding JSON from LLM response (no valid JSON found). Cleaned response was: '{cleaned_response_text}'")
                    return "Error_ParsingResponse"

            intent = response_json.get("intent")

            if not intent:
                print(f"ClassifierAgent: LLM JSON response missing 'intent' key. Response: {cleaned_response_text}")
                return "Error_MalformedResponse"
            if intent not in possible_intents:
                print(f"ClassifierAgent: LLM returned an unexpected intent '{intent}'. Defaulting to 'Other'. Response: {cleaned_response_text}")
                return "Other"
            
            print(f"ClassifierAgent: LLM classified intent as '{intent}'.")
            return intent

        except json.JSONDecodeError as e:
            response_text_snippet = cleaned_response_text if 'cleaned_response_text' in locals() else (response.text[:200] if response and hasattr(response, 'text') else 'No response text.')
            print(f"ClassifierAgent: Error decoding JSON from LLM response: {e}. Cleaned response was: '{response_text_snippet}'")
            return "Error_ParsingResponse"
        except Exception as e:
            feedback_message = ""
            if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                 feedback_message = f" Prompt blocked due to: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}."
            elif "API key not valid" in str(e) or "API_KEY_INVALID" in str(e).upper():
                 feedback_message = " Please check your GEMINI_API_KEY."
            
            print(f"ClassifierAgent: Error during LLM intent classification or processing its response: {e}.{feedback_message}")
            if "ക്രമീകരണ республик" in str(e): 
                print("ClassifierAgent: Encountered an unusual error string from API, potential API issue or misconfiguration.")
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
            return {"status": "Error", "message": format_error_message, "conversation_id": self.conversation_id, "format": doc_format, "intent": None}
        
        if doc_format != "Unknown_EmptyInput" and not text_content_for_intent and format_error_message:
             print(f"ClassifierAgent: Format {doc_format} determined, but content is effectively empty or had issues: {format_error_message}")
        elif doc_format != "Unknown_EmptyInput" and not text_content_for_intent and not format_error_message:
            print(f"ClassifierAgent: Format {doc_format} determined, content is empty (no specific error message).")


        initial_log_data["determined_format"] = doc_format

        intent = "N/A" 
        if doc_format == "Unknown_EmptyInput":
            intent = "Unknown_EmptyInput" 
        elif "Error_" in doc_format : 
            intent = "Error_Preprocessing" 
            initial_log_data["error_details_preprocessing"] = format_error_message 
        else: 
            if not self.model: 
                intent = "Error_ModelNotInitialized" 
                print("ClassifierAgent: LLM model not initialized, cannot classify intent.")
            elif not text_content_for_intent.strip(): 
                intent = "Unknown_EmptyContent"
                print("ClassifierAgent: Content for intent classification is empty, skipping LLM call.")
            else:
                intent = self._classify_intent_with_gemini(text_content_for_intent) 

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

        if intent and ("Error_" in intent or intent in ["Unknown_EmptyInput", "Unknown_EmptyContent", "Error_MalformedResponse", "Error_PromptBlocked", "Error_ModelNotInitialized", "Error_ParsingResponse"]):
            status = f"Failed_{intent}"
            final_log_payload["error_message"] = f"Intent classification issue or problematic content/setup: {intent}"
            final_log_payload["extracted_values"] = {"text_snippet": text_content_for_intent[:200] if text_content_for_intent else "N/A"}
        
        elif doc_format == 'JSON':
            try:
                json_data_for_agent = json.loads(text_content_for_intent)
                agent_output, anomalies = self.json_agent.process(json_data_for_agent)
                final_log_payload["extracted_values"] = agent_output
                final_log_payload["anomalies"] = anomalies
            except json.JSONDecodeError as e:
                status = "Failed_JSON_Processing"
                final_log_payload["error_message"] = f"Invalid JSON for JSONAgent (unexpected): {e}"
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
            if intent == "Error_Preprocessing" and "PDFParsing" in (format_error_message or ""): 
                status = "Failed_PDF_Parsing"
                final_log_payload["error_message"] = format_error_message 
                agent_output = {"extracted_pdf_text_snippet": "PDF PARSING FAILED."}
            else: 
                agent_output = {"extracted_pdf_text_snippet": text_content_for_intent[:500] if text_content_for_intent else "No text extracted or PDF empty."}
            final_log_payload["extracted_values"] = agent_output
        
        elif doc_format == 'TextFile' or doc_format == 'Text':
            agent_output = {"text_content_snippet": text_content_for_intent[:500] if text_content_for_intent else "No text content."}
            final_log_payload["extracted_values"] = agent_output
        
        elif "Error_" in doc_format: 
            status = f"Failed_{doc_format}"
            final_log_payload["error_message"] = format_error_message or f"Failed due to {doc_format}"
            final_log_payload["extracted_values"] = {"input_reference": raw_input_data if input_is_path else raw_input_data[:100]}
            agent_output = {"error_detail": format_error_message}

        else: 
            status = "Failed_UnknownFormatForRouting"
            final_log_payload["error_message"] = f"Cannot route, unknown or unhandled format: {doc_format}"
            final_log_payload["extracted_values"] = {"raw_content_snippet": text_content_for_intent[:200] if text_content_for_intent else "N/A"}
        
        self.shared_memory.log(self.conversation_id, final_log_payload)
        print(f"ClassifierAgent: Processing complete for ConvID {self.conversation_id}. Status: {status}")

        return {"status": status, "format": doc_format, "intent": intent, "output": agent_output, "anomalies": anomalies, "conversation_id": self.conversation_id}