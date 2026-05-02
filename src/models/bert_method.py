"""
BERT-based sentiment analysis.

Uses a fine-tuned DistilBERT model (distilbert-base-uncased-finetuned-sst-2-english)
to classify text sentiment as positive, neutral, or negative.
"""
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models.base_class import SentimentAnalyserBase
from models.sentiment_result import SentimentResult


class BertSentimentAnalyser(SentimentAnalyserBase):
    """
    Sentiment analysis using a fine-tuned DistilBERT model.
    
    The model is pre-trained on SST-2 and directly outputs positive/negative
    classification logits, which are converted to a polarity score in [-1, 1].
    """

    def __init__(self):
        """Loads the fine-tuned DistilBERT model from HuggingFace."""
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Evaluation mode — disables dropout

    @property
    def name(self) -> str:
        """Return the model identifier."""
        return "BERT"

    def analyse(self, text: str) -> Dict[str, float]:
        """
        Analyse sentiment of the given text using DistilBERT.
        
        Returns a dict with 'text', 'score' (in [-1, 1]), and 'label'.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        try:
            # Tokenise the input text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get logits for negative and positive classes
                logits = outputs.logits[0]
            
            # Convert logits to a polarity score in [-1, 1]
            probabilities = torch.softmax(logits, dim=0)
            pos_prob = probabilities[1].item()
            neg_prob = probabilities[0].item()
            polarity_score = pos_prob - neg_prob
            
            # Convert score to a label
            label = self.score_to_label(
                polarity_score,
                pos_threshold=0.1,
                neg_threshold=-0.1
            )

            result_obj = SentimentResult(
                text=text,
                score=polarity_score,
                label=label
            )
            self.validate_result(result_obj)
            return {
                "text": result_obj.text,
                "score": result_obj.score,
                "label": result_obj.label
            }

        except Exception as e:
            raise RuntimeError(f"Sentiment analysis failed: {str(e)}")

    def analyse_batch(self, texts: list) -> list:
        """Analyse a list of texts and return a list of result dicts."""
        if not texts:
            raise ValueError("Texts list cannot be empty")

        results = []
        for text in texts:
            results.append(self.analyse(text))
        return results