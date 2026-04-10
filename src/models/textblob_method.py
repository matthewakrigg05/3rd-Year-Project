"""
TextBlob sentiment analysis using pre-trained language models.

This module provides a TextBlobSentimentAnalyser class that uses the TextBlob
library for sentiment analysis. TextBlob combines multiple approaches including
the PatternAnalyzer (pattern-based) for English text, providing a simple and
intuitive interface for sentiment classification.

Key Features:
    - Built on top of pattern library for reliable sentiment analysis
    - Produces polarity scores ranging from -1 (negative) to 1 (positive)
    - Also provides subjectivity scores (not used in basic classification)
    - Simple, lightweight, and fast implementation
    - Outputs ternary sentiment labels: positive, neutral, or negative
    - No neural networks; uses syntactic analysis and pattern matching

References:
    TextBlob: Simplified Text Processing (https://textblob.readthedocs.io/)

@author: MA
"""
from typing import Dict
from textblob import TextBlob

from models.base_class import SentimentAnalyserBase
from models.sentiment_result import SentimentResult


class TextBlobSentimentAnalyser(SentimentAnalyserBase):
    """
    Sentiment analysis using TextBlob library.
    
    TextBlob provides a simple API to common NLP tasks. For sentiment analysis,
    it uses a pre-trained naive Bayes model trained on movie reviews (from the
    pattern library). The polarity score ranges from -1 (most negative) to 1
    (most positive), with 0 being neutral.
    """

    def __init__(self):
        """
        Initialise the TextBlob sentiment analyser.
        
        TextBlob does not require any model downloading or initialisation steps.
        """
        pass

    @property
    def name(self) -> str:
        """Return the model identifier."""
        return "TextBlob"

    def analyse(self, text: str) -> Dict[str, float]:
        """
        Analyse the sentiment of input text using TextBlob.
        
        TextBlob processes input text through its built-in sentiment analyser,
        which applies pattern-based rules and a pre-trained classifier to
        determine the polarity of the sentiment expressed.
        
        The processing steps:
        1. Creates a TextBlob object from the input string
        2. Accesses the sentiment property which contains polarity and subjectivity
        3. Extracts the polarity score (range -1 to 1)
        4. Converts to label based on thresholds
        
        Args:
            text (str): The input text to analyse. Should be non-empty.
            
        Returns:
            Dict[str, float]: A dictionary containing:
                - 'text': The original input text
                - 'score': Polarity score in range [-1, 1] indicating sentiment
                - 'label': Categorical label ('positive', 'neutral', or 'negative')
                
        Raises:
            ValueError: If text is empty or not a string
            RuntimeError: If TextBlob analysis fails unexpectedly
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        try:
            # Create TextBlob object and extract sentiment
            blob = TextBlob(text)
            polarity_score = blob.sentiment.polarity  # Range: -1 to 1

            # Convert polarity score to ternary label
            label = self.score_to_label(
                polarity_score,
                pos_threshold=0.05,
                neg_threshold=-0.05
            )

            result_obj = SentimentResult(
                text=text,
                score=polarity_score,
                label=label
            )
            # Validate output format and label validity
            self.validate_result(result_obj)
            return {
                "text": result_obj.text,
                "score": result_obj.score,
                "label": result_obj.label
            }

        except Exception as e:
            raise RuntimeError(f"TextBlob analysis failed: {str(e)}")

    def analyse_batch(self, texts: list) -> list:
        """
        Analyse sentiment of multiple texts in sequence.
        
        Processes each text individually through the analyse method. TextBlob
        is fast and lightweight, making sequential batch processing efficient.
        
        Args:
            texts (list): List of text strings to analyse. Each should be non-empty.
            
        Returns:
            list: List of result dictionaries, one per input text, in the same
                  order as the input list
            
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
