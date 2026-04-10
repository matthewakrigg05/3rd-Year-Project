"""
BERT-based sentiment analysis using contextual bidirectional embeddings.

This module provides a BertSentimentAnalyser class that leverages the pre-trained
BERT (Bidirectional Encoder Representations from Transformers) model to perform
sentiment classification. The approach uses the [CLS] token embeddings as a
representation of the entire input sequence for sentiment prediction.

Key Features:
    - Uses pre-trained bert-base-uncased model from HuggingFace
    - Supports GPU acceleration via PyTorch
    - Produces polarity scores in the range [-1, 1]
    - Outputs ternary sentiment labels: positive, neutral, or negative
    - Batch processing for efficient analysis of multiple texts

@author: MA
"""
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from models.base_class import SentimentAnalyserBase
from models.sentiment_result import SentimentResult


class BertSentimentAnalyser(SentimentAnalyserBase):
    """
    Sentiment analysis using fine-tuned DistilBERT model.
    
    This class uses a DistilBERT model that has been pre-trained and fine-tuned
    specifically for sentiment analysis on the Stanford Sentiment Treebank (SST-2).
    The model directly outputs classification logits for positive and negative
    sentiment, which are converted to a continuous polarity score.
    
    Attributes:
        device (torch.device): Computation device (cuda or cpu)
        tokenizer (AutoTokenizer): Tokeniser for input text processing
        model (AutoModelForSequenceClassification): Fine-tuned sentiment model
    """

    def __init__(self):
        """
        Initialise the fine-tuned DistilBERT sentiment analyser.
        
        Downloads and loads the distilbert-base-uncased-finetuned-sst-2-english
        model from the HuggingFace model hub. This model has been fine-tuned
        specifically for sentiment analysis. Automatically detects GPU
        availability and places the model on the appropriate device.
        
        Raises:
            OSError: If model cannot be downloaded from HuggingFace hub.
        """
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        # Set model to evaluation mode to disable dropout and batch norm updates
        self.model.eval()

    @property
    def name(self) -> str:
        """Return the model identifier."""
        return "BERT"

    def analyse(self, text: str) -> Dict[str, float]:
        """
        Analyse the sentiment of input text using fine-tuned DistilBERT.
        
        Performs the following steps:
        1. Breaks the text into tokens that DistilBERT can understand
        2. Processes tokens through the fine-tuned model
        3. Gets classification logits for positive and negative sentiment
        4. Converts logits to a sentiment score between -1 and 1
        5. Labels it as positive, neutral, or negative
        
        Args:
            text (str): The input text to analyse. Should be non-empty.
            
        Returns:
            Dict[str, float]: A dictionary containing:
                - 'text': The original input text
                - 'score': Sentiment score from -1 (negative) to 1 (positive)
                - 'label': One of 'positive', 'neutral', or 'negative'
                
        Raises:
            ValueError: If text is empty or not a string
            RuntimeError: If sentiment analysis fails
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
            
            # Convert logits to a polarity score
            # Logits: [negative_score, positive_score]
            # Polarity = (positive - negative) / (positive + negative)
            probabilities = torch.softmax(logits, dim=0)
            pos_prob = probabilities[1].item()  # Positive class
            neg_prob = probabilities[0].item()  # Negative class
            
            # Calculate polarity score
            polarity_score = pos_prob - neg_prob  # Range: -1 to 1
            
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
        """
        Analyse sentiment of multiple texts in sequence.
        
        Processes each text individually through the analyse method. For improved
        performance with large batches, consider implementing true batch processing
        with padded sequences.
        
        Args:
            texts (list): List of text strings to analyse
            
        Returns:
            list: List of result dictionaries, one per input text
            
        Raises:
            ValueError: If texts list is empty
            RuntimeError: If analysis of any text fails
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        results = []
        for text in texts:
            results.append(self.analyse(text))
        return results