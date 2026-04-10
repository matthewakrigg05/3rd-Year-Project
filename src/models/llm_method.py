"""
GPT-2 based sentiment analysis using text generation.

This module provides a GPT2SentimentAnalyser class that performs sentiment
analysis by generating text continuations and analysing sentiment keywords
within the generated output and original text.

Key Features:
    - Uses pre-trained GPT-2 model (124M parameters) from OpenAI
    - Generates contextual text to understand sentiment
    - Counts positive and negative keywords in generated text
    - Produces polarity scores in the range [-1, 1]
    - Outputs ternary sentiment labels: positive, neutral, or negative
    - Fast inference with minimal computational overhead

Algorithm Overview:
    1. Creates a prompt: "The sentiment of '{text}' is"
    2. Uses GPT-2 to generate a short continuation
    3. Counts positive and negative keywords in generated text
    4. Calculates polarity as (positive - negative) / (positive + negative)
    5. Converts score to categorical label

@author: MA
"""
from typing import Dict
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from models.base_class import SentimentAnalyserBase
from models.sentiment_result import SentimentResult


class GPT2SentimentAnalyser(SentimentAnalyserBase):
    """
    Sentiment analysis using GPT-2 text generation.
    
    This class uses the pre-trained GPT-2 model to generate text continuations.
    When given a prompt about text sentiment, the model generates what it thinks
    should come next. By analysing sentiment keywords in the generated text,
    we infer the model's understanding of the input's sentiment.
    
    Attributes:
        device (torch.device): Computation device (cuda or cpu)
        tokenizer (GPT2Tokenizer): Tokeniser for input text processing
        model (GPT2LMHeadModel): Pre-trained GPT-2 language model
        positive_words (set): Vocabulary of positive sentiment keywords
        negative_words (set): Vocabulary of negative sentiment keywords
    """

    def __init__(self):
        """
        Initialise the GPT-2 sentiment analyser.
        
        Downloads and loads the pre-trained GPT-2 model from HuggingFace.
        Automatically detects GPU availability and places the model on the
        appropriate device. Initialises sentiment keyword vocabularies.
        
        Raises:
            OSError: If model cannot be downloaded from HuggingFace hub
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to suppress warnings
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        # Set model to evaluation mode to disable dropout
        self.model.eval()
        
        # Sentiment indicator words for keyword-based analysis
        self.positive_words = {
            "positive", "good", "great", "amazing", "excellent", "love", "best",
            "wonderful", "awesome", "fantastic", "happy", "perfect", "brilliant"
        }
        self.negative_words = {
            "negative", "bad", "terrible", "awful", "hate", "worst", "horrible",
            "disgusting", "poor", "disappointing", "sad", "useless", "waste"
        }

    @property
    def name(self) -> str:
        """Return the model identifier."""
        return "GPT-2"

    def analyse(self, text: str) -> Dict[str, float]:
        """
        Analyse sentiment using GPT-2 text generation.
        
        This method performs sentiment analysis by:
        1. Creating a sentiment prompt with the input text
        2. Generating a continuation using GPT-2
        3. Counting sentiment keywords in generated text
        4. Computing a polarity score from keyword balance
        
        Args:
            text (str): The input text to analyse. Should be non-empty.
            
        Returns:
            Dict[str, float]: A dictionary containing:
                - 'text': The original input text
                - 'score': Polarity score in range [-1, 1] indicating sentiment
                - 'label': Categorical label ('positive', 'neutral', or 'negative')
                
        Raises:
            ValueError: If text is empty or not a string
            RuntimeError: If GPT-2 generation fails
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        try:
            # Create sentiment analysis prompt
            prompt = f"The sentiment of '{text}' is"
            
            # Tokenise the prompt with attention mask
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=True
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Generate text continuation without gradient calculation
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=input_ids.shape[1] + 20,  # Generate 20 additional tokens
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,  # Prevent token repetition
                    top_k=50,  # Restrict to top 50 likely next tokens
                    top_p=0.95,  # Nucleus sampling for diversity
                    temperature=0.9  # Moderate randomness for varied output
                )
            
            # Decode generated tokens to natural text
            generated_text = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            ).lower()
            
            # Count sentiment keywords in generated text
            pos_count = sum(1 for word in self.positive_words if word in generated_text)
            neg_count = sum(1 for word in self.negative_words if word in generated_text)
            
            # Calculate polarity score as normalised difference
            if pos_count + neg_count > 0:
                # Normalise to [-1, 1] based on balance of sentiments
                polarity_score = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                # No clear sentiment keywords found - neutral
                polarity_score = 0.0
            
            # Ensure score is in valid range
            polarity_score = max(-1.0, min(1.0, polarity_score))
            
            # Convert continuous score to categorical label
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
            raise RuntimeError(f"GPT-2 analysis failed: {str(e)}")

    def analyse_batch(self, texts: list) -> list:
        """
        Analyse sentiment of multiple texts in sequence.
        
        Processes each text individually through the analyse method.
        
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