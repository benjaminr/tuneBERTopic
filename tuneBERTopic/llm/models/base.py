from abc import ABC, abstractmethod


class LLMModel(ABC):
    """Abstract base class for LLM models."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None
        self.setup_connection()

    @abstractmethod
    def setup_connection(self):
        """Establish connection to the model API."""
        pass

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response based on the prompt."""
        pass