import mlflow
import logging
from bertopic import BERTopic
from sklearn.metrics import precision_score
import pandas as pd
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tunebertopic.llm.functions.summaries import SummaryGenerator
from tunebertopic.llm.models.base import LLMModel


logger = logging.getLogger("tunebertopic")

def precision(topic_model: BERTopic, documents):
    pass
    # logger.info("Calculating precision score")
    # ground_truth = [...]  # Add actual ground truth labels
    # predicted_topics = [...]  # Extract predicted topics from topic_model
    # precision = precision_score(ground_truth, predicted_topics, average="binary")
    # mlflow.log_metric(f"precision_score", precision)
    # return precision


def rouge(topic_model: BERTopic, documents, llm_model: LLMModel):
    logger.info("Calculating ROUGE scores")
    if llm_model is None:
        raise ValueError("LLM model is required for ROUGE evaluation")
    summary_generator = SummaryGenerator(llm_model)
    summaries = summary_generator.generate_summaries(topic_model, documents)
    logger.info(f"Generated summaries: {summaries}")
    references = []
    doc_indices = topic_model.get_document_info(documents)
    
    for topic_id in range(len(topic_model.get_topics())):
        topic_docs = doc_indices[doc_indices.Topic == topic_id].Document.tolist()
        references.append(" ".join(topic_docs))
    rouge = Rouge()
    # Placeholder: Add actual references
    overall_scores = []
    for summary, refs, topic_id in zip(summaries, references, range(len(topic_model.get_topics()))):
        scores = rouge.get_scores(summary, refs)
        logger.info(f"Topic: {topic_id}, {topic_model.get_topic(topic_id)}")
        logger.info(f"ROUGE scores: {scores}")
        overall_scores.append(scores[0]["rouge-1"])
    average_scores = pd.DataFrame(overall_scores).mean()
    logger.info(f"Average ROUGE scores: {average_scores}")
    mlflow.log_metric("rouge-1_f", scores[0]["rouge-1"]["f"])
    mlflow.log_metric("rouge-2_f", scores[0]["rouge-2"]["f"])
    mlflow.log_metric("rouge-l_f", scores[0]["rouge-l"]["f"])
    return 1-average_scores["r"]


def bleu(topic_model: BERTopic, documents, llm_model: LLMModel):
    logger.info("Calculating BLEU score")
    if llm_model is None:
        raise ValueError("LLM model is required for BLEU evaluation")
    summary_generator = SummaryGenerator(llm_model)
    summaries = summary_generator.generate_summaries(topic_model, documents)
    # Placeholder: Add actual reference sentences
    references = [...]  # Reference sentences
    bleu_scores = [
        sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method1)
        for ref, cand in zip(references, summaries)
    ]
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    mlflow.log_metric("bleu_score", avg_bleu_score)
    return avg_bleu_score
