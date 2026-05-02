"""
GPT-2 based sentiment analysis using text generation.

Builds a sentiment prompt from the input text, generates a short continuation
using GPT-2, then counts positive/negative keywords to produce a polarity score.
"""
from typing import Dict
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from models.base_class import SentimentAnalyserBase
from models.sentiment_result import SentimentResult


class GPT2SentimentAnalyser(SentimentAnalyserBase):
    """
    Sentiment analysis using GPT-2 text generation.
    
    Given a sentiment prompt, GPT-2 generates a continuation. Positive and
    negative keywords in the output are counted to produce a polarity score.
    """

    def __init__(self):
        """Loads the pre-trained GPT-2 model and sets up sentiment keyword lists."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to suppress warnings
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        # Set model to evaluation mode to disable dropout
        self.model.eval()  # Evaluation mode — disables dropout
        
        # Sentiment indicator words
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

    def _sanitise_text(self, text: str, max_length: int = 150) -> str:
        """Removes control characters and truncates text to avoid GPU errors."""
        # Remove control characters and excessive whitespace
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        # Normalise whitespace
        text = ' '.join(text.split())
        # Truncate to max_length - more aggressive to prevent GPU issues
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0]  # Cut at word boundary
        return text.strip() if text else "neutral"

    def analyse(self, text: str) -> Dict[str, float]:
        """
        Analyse sentiment of the given text using GPT-2.
        
        Returns a dict with 'text', 'score' (in [-1, 1]), and 'label'.
        Falls back to CPU if GPU generation fails.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        try:
            # Sanitise input before passing to model
            safe_text = self._sanitise_text(text)
            
            # Build sentiment prompt
            prompt = f"The sentiment of '{safe_text}' is"
            
            # Tokenise the prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
                max_length=256  # Reduced from 512 for safety
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Validate input tensors before generation
            if input_ids.shape[1] == 0:
                raise ValueError("Tokenized input is empty after processing")
            
            # Generate text (no gradient needed)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_length=min(input_ids.shape[1] + 15, 256),  # Reduced max_length
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        top_k=40,
                        top_p=0.9,
                        temperature=0.7,  # More conservative
                        repetition_penalty=1.0,
                        length_penalty=1.0
                    )
            except RuntimeError as cuda_error:
                # If GPU generation fails, try CPU as fallback
                if "cuda" in str(cuda_error).lower():
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"GPU generation failed, falling back to CPU: {str(cuda_error)[:100]}")
                    
                    # Move model to CPU temporarily
                    self.model.to("cpu")
                    input_ids_cpu = input_ids.to("cpu")
                    attention_mask_cpu = attention_mask.to("cpu")
                    
                    try:
                        with torch.no_grad():
                            output_ids = self.model.generate(
                                input_ids_cpu,
                                attention_mask=attention_mask_cpu,
                                pad_token_id=self.tokenizer.eos_token_id,
                                max_length=min(input_ids_cpu.shape[1] + 15, 256),
                                num_return_sequences=1,
                                no_repeat_ngram_size=2,
                                top_k=40,
                                top_p=0.9,
                                temperature=0.7,
                                repetition_penalty=1.0,
                                length_penalty=1.0
                            )
                    finally:
                        # Move model back to GPU
                        self.model.to(self.device)
                else:
                    raise
            
            # Decode the generated output
            generated_text = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            ).lower()
            
            # Count sentiment keywords in generated text
            pos_count = sum(1 for word in self.positive_words if word in generated_text)
            neg_count = sum(1 for word in self.negative_words if word in generated_text)
            
            # Calculate polarity score
            if pos_count + neg_count > 0:
                polarity_score = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                polarity_score = 0.0  # No sentiment keywords found
            
            # Ensure score is in valid range
            polarity_score = max(-1.0, min(1.0, polarity_score))
            
            # Convert score to label
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

        except RuntimeError as e:
            # Handle CUDA-specific errors
            if "cuda" in str(e).lower() or "assert" in str(e).lower():
                raise RuntimeError(f"GPT-2 GPU error (persistent): {str(e)}")
            raise RuntimeError(f"GPT-2 analysis failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"GPT-2 analysis failed: {str(e)}")

    def analyse_batch(self, texts: list) -> list:
        """Analyse a list of texts and return a list of result dicts."""
        if not texts:
            raise ValueError("Texts list cannot be empty")

        results = []
        for text in texts:
            results.append(self.analyse(text))
        return results