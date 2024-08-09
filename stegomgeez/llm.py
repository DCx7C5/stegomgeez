from typing import Tuple

from transformers import pipeline

language_detection = pipeline(
    task="text-classification",
    model="papluca/xlm-roberta-base-language-detection",
)


def detect_language_transformers(text) -> Tuple[str, str] | str:
    try:
        result = language_detection(text)
        language = result[0]['label']
        score = result[0]['score']
        return language, score
    except Exception as e:
        return str(e)
