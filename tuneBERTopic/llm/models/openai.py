import os
import json
from tunebertopic.llm.models.base import LLMModel
from openai import OpenAI


class OpenAIModel(LLMModel):
    def setup_connection(self):
        api_key = self.api_key if self.api_key else os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, prompt: str, retry: bool = True) -> str:
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled text summariser. You are particularly adept at distilling key information from long bodies of text and summarising the key points.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        try:
            return json.loads(completion.choices[0].message.content)["summary"]
        except (json.JSONDecodeError, KeyError):
            if retry:
                # Amend the prompt to ask for valid JSON and retry
                amended_prompt = (
                    f"{prompt}\n\n"
                    "The previous response did not include valid JSON. "
                    "Please ensure your response is in the following JSON format: {\"summary\": \"this is the summary of the documents.\"}"
                )
                return self.generate_response(amended_prompt, retry=False)
            else:
                raise ValueError("Failed to get valid JSON response after retrying.")
