from typing import Any, Dict, List
from .base_agent import BaseAgent

class DataCollectionAgent(BaseAgent):
    def __init__(self):
        super().__init__("data_collection_agent", "data_collection")
        self.current_session = None
        self.question_bank = {}

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            action = data.get("action")
            if action == "start_session":
                return await self.start_session(data)
            elif action == "submit_response":
                return await self.handle_response(data)
            elif action == "end_session":
                return await self.end_session(data)
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as e:
            return await self.handle_error(e)

    async def start_session(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = data.get("session_id")
        module = data.get("module")
        self.current_session = {
            "session_id": session_id,
            "module": module,
            "responses": []
        }
        return self.create_response({
            "session_id": session_id,
            "status": "session_started",
            "module": module
        })

    async def handle_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.current_session:
            raise ValueError("No active session")
        
        response = data.get("response")
        self.current_session["responses"].append(response)
        
        return self.create_response({
            "session_id": self.current_session["session_id"],
            "status": "response_recorded"
        })

    async def end_session(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.current_session:
            raise ValueError("No active session")
        
        session_data = self.current_session
        self.current_session = None
        
        return self.create_response({
            "session_id": session_data["session_id"],
            "status": "session_completed",
            "responses_count": len(session_data["responses"])
        })