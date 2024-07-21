import random
from bertopic import BERTopic

from tunebertopic.llm.models.base import LLMModel
from tunebertopic.llm.prompts.summaries import prompt


class SummaryGenerator:
    def __init__(
        self,
        llm_model: LLMModel,
        sample_size: int = 5,
        template: str = prompt,
    ):
        self.llm_model = llm_model
        self.sample_size = sample_size
        self.template = template

    def generate_summaries(self, topic_model: BERTopic, documents):
        topics = topic_model.get_topics()
        return [
            self.llm_model.generate_response(
                self._create_prompt(
                    [word for word, _ in words], self._get_example_documents(documents)
                )
            )
            for topic_id, words in topics.items()
            if topic_id != -1
        ]

    def _get_example_documents(self, documents):
        return random.sample(documents, min(self.sample_size, len(documents)))

    def _create_prompt(self, keywords, example_docs):
        keywords_str = ", ".join(keywords)
        example_docs_str = "\n\n".join(example_docs)
        return self.template.format(
            keywords_str=keywords_str, example_docs_str=example_docs_str
        )
