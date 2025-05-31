class JSONAgent:
    def __init__(self, target_schema):
        self.target_schema = target_schema

    def process(self, json_payload):
        anomalies = []
        output = {}

        for field in self.target_schema:
            if field not in json_payload:
                anomalies.append(f"Missing field: {field}")
                output[field] = None
            else:
                output[field] = json_payload[field]

        return output, anomalies
    
    

