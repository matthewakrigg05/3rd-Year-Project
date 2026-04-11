"""
VADER sentiment analysis for social media and informal text.

This module provides a VaderSentimentAnalyser class that implements the VADER
(Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis algorithm.
VADER is optimised for informal text, particularly from social media, and handles
emoticons, slang, and linguistic nuances very well.

Key Features:
    - Pre-built lexicon of sentiment-bearing words with weighted values
    - Handles emoticons and acronyms naturally
    - Produces compound scores ranging from -1 (most negative) to 1 (most positive)
    - Fast execution with minimal computational overhead
    - Outputs ternary sentiment labels: positive, neutral, or negative
    - No pre-training required; uses fixed linguistic rules

@author: MA
"""
from typing import Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from models.base_class import SentimentAnalyserBase
from models.sentiment_result import SentimentResult


class VaderSentimentAnalyser(SentimentAnalyserBase):
    """
    Sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    
    VADER is a lexicon and rule-based sentiment analysis tool specifically tuned
    for social media and informal text. It combines a pre-defined lexicon of
    sentiment words with grammatical rules for modifiers (negations, intensifiers)
    to produce interpretable sentiment scores.
    
    Strengths:
        - Excellent for social media text, reviews, and informal language
        - Handles emoticons, emojis, and internet slang
        - Fast and lightweight (no neural network inference needed)
        - Works well with mixed polarity expressions
        
    Attributes:
        analyser (SentimentIntensityAnalyzer): The VADER sentiment intensity analyser
    """

    def __init__(self):
        """
        Initialise the VADER sentiment analyser.
        
        Creates an instance of the SentimentIntensityAnalyzer which loads the
        pre-computed lexicon and initialisation parameters. This is a lightweight
        operation that does not require downloading large models.
        """
        self.analyser = SentimentIntensityAnalyzer()

    @property
    def name(self) -> str:
        """Return the model identifier."""
        return "VADER"

    def analyse(self, text: str) -> Dict[str, float]:
        """
        Analyse the sentiment of input text using VADER.
        
        VADER processes the text through its lexicon-based rules to generate
        sentiment scores. The compound score is most useful as it normalises
        the sentiment across different text lengths and polarities.
        
        The algorithm:
        1. Tokenises the input text
        2. Checks each token against the sentiment lexicon
        3. Applies grammatical rules for negations, intensifiers, and punctuation
        4. Combines scores using proprietary normalisation formulae
        5. Generates pos, neg, neu (positive, negative, neutral) and compound scores
        
        Args:
            text (str): The input text to analyse. Should be non-empty.
            
        Returns:
            Dict[str, float]: A dictionary containing:
                - 'text': The original input text
                - 'score': Compound score in range [-1, 1] indicating sentiment polarity
                - 'label': Categorical label ('positive', 'neutral', or 'negative')
                
        Raises:
            ValueError: If text is empty or not a string
            RuntimeError: If VADER analysis fails unexpectedly
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        try:
            # Calculate sentiment scores using VADER's lexicon and rules
            scores = self.analyser.polarity_scores(text)
            # Extract the compound score (normalised combined sentiment)
            compound_score = scores["compound"]  # Range: -1 to 1

            # Convert compound score to ternary label using standard thresholds
            label = self.score_to_label(
                compound_score,
                pos_threshold=0.05,
                neg_threshold=-0.05
            )

            result_obj = SentimentResult(
                text=text,
                score=compound_score,
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
            raise RuntimeError(f"VADER analysis failed: {str(e)}")

    def analyse_batch(self, texts: list) -> list:
        """
        Analyse sentiment of multiple texts in sequence.
        
        Processes each text through the analyse method. VADER is fast enough
        that even sequential processing is efficient for moderate batch sizes.
        
        Args:
            texts (list): List of text strings to analyse. Each should be non-empty.
            
        Returns:
            list: List of result dictionaries, one per input text, maintaining
                  the same order as the input list
            
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
