from abc import ABC, abstractmethod
from typing import Any, Dict
from datetime import datetime

class BaseAgent(ABC):
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.status = "initialized"

    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def create_response(self, data: Dict[str, Any], status: str = "success") -> Dict[str, Any]:
        return {
            "status": status,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id,
            "agent_type": self.agent_type
        }

    async def handle_error(self, error: Exception) -> Dict[str, Any]:
        return self.create_response(
            {"error": str(error), "type": type(error).__name__},
            status="error"
        )