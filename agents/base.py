from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, message: str) -> Any:
        """
        Execute the agent with the given user message.
        """
        pass