import re

class EmailAgent:
    def extract(self, email_text): 
        sender_match = re.search(r'From:\s*(.*)', email_text, re.IGNORECASE)
        subject_match = re.search(r'Subject:\s*(.*)', email_text, re.IGNORECASE)
        
        body_start_index = -1
        lines = email_text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "": 
                if i + 1 < len(lines):
                    body_start_index = i + 1
                    break
        if body_start_index == -1 and lines: 
            for i, line in enumerate(lines):
                if "subject:" in line.lower():
                    if i + 1 < len(lines):
                        body_start_index = i + 1
                    break
            if body_start_index == -1 : body_start_index = 0 

        body = "\n".join(lines[body_start_index:]).strip() if body_start_index !=-1 else email_text

        sender = sender_match.group(1).strip() if sender_match else None
        subject = subject_match.group(1).strip() if subject_match else None

        urgency_keywords = ['urgent', 'asap', 'immediately', 'important', 'critical']
        
        text_to_check_urgency = (subject if subject else "") + " " + (body[:500] if body else "")
        urgency = any(word in text_to_check_urgency.lower() for word in urgency_keywords)

        return {
            "sender": sender,
            "subject": subject,
            "body": body, 
            "urgency": urgency
        }
        
