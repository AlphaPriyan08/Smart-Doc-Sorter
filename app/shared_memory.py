import redis
import json
from datetime import datetime, timezone # Use timezone-aware UTC

class SharedMemory:
    def __init__(self, host='localhost', port=6379):
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=0, decode_responses=True)
            self.redis_client.ping() # Check connection
            print("Successfully connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            print(f"Error connecting to Redis: {e}. SharedMemory will not function.")
            self.redis_client = None 
        
    def log(self, conversation_id, data):
        if not self.redis_client:
            print("SharedMemory: Cannot log, Redis connection not available.")
            return

        key = f"conversation:{conversation_id}"
        try:
            existing_raw = self.redis_client.get(key)
            history = []
            if existing_raw:
                try:
                    history = json.loads(existing_raw)
                    if not isinstance(history, list): # Ensure it's a list
                       history = [history] 
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode existing JSON for key {key}. Starting new history.")
                    history = [] # Or handle as an error / corruption
            
            data['timestamp'] = datetime.now(timezone.utc).isoformat() # Use timezone-aware UTC
            history.append(data)
            self.redis_client.set(key, json.dumps(history))
        except redis.exceptions.RedisError as e:
            print(f"Redis error during log operation: {e}")

    def get_history(self, conversation_id):
        if not self.redis_client:
            print("SharedMemory: Cannot get history, Redis connection not available.")
            return []

        key = f"conversation:{conversation_id}"
        try:
            data_raw = self.redis_client.get(key)
            if data_raw:
                try:
                    return json.loads(data_raw)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON history for key {key}.")
                    return [{"error": "Failed to decode history from Redis", "raw_data": data_raw}]
            return []
        except redis.exceptions.RedisError as e:
            print(f"Redis error during get_history operation: {e}")
            return []