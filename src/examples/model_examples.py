"""
Example script demonstrating all sentiment analysis models.

This script shows how to use each of the four sentiment analysis models
(BERT, VADER, TextBlob, GPT-2) and displays their outputs in the console.

Usage:
    python -m src.model_examples
"""

from models.bert_method import BertSentimentAnalyser
from models.vader_method import VaderSentimentAnalyser
from models.textblob_method import TextBlobSentimentAnalyser
from models.llm_method import GPT2SentimentAnalyser


def print_result(model_name: str, result: dict) -> None:
    """
    Pretty-print a sentiment analysis result.
    
    Args:
        model_name (str): Name of the model
        result (dict): Result dictionary with text, score, and label
    """
    print(f"  Text: {result['text'][:60]}...")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Label: {result['label'].upper()}")
    print()


def demonstrate_models():
    """Run demonstrations of all four sentiment analysis models."""
    
    # Example texts with varying sentiment
    example_texts = [
        "This product is absolutely amazing! I love it so much!",
        "This is okay, nothing special.",
        "Terrible experience. I hate this and want my money back.",
        "The weather is nice today.",
    ]
    
    print("=" * 70)
    print("SENTIMENT ANALYSIS MODELS DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Initialize all models
    print("Initialising models... (this may take a moment)")
    print()
    
    models = {
        "BERT": BertSentimentAnalyser(),
        "VADER": VaderSentimentAnalyser(),
        "TextBlob": TextBlobSentimentAnalyser(),
        "GPT-2": GPT2SentimentAnalyser(),
    }
    
    print("Models ready!")
    print()
    print("-" * 70)
    print()
    
    # Analyse each text with all models
    for i, text in enumerate(example_texts, 1):
        print(f"TEXT {i}: \"{text}\"")
        print()
        
        for model_name, model in models.items():
            print(f"{model_name} Analysis:")
            result = model.analyse(text)
            print_result(model_name, result)
        
        print("-" * 70)
        print()
    
    # Demonstrate batch analysis
    print("BATCH ANALYSIS EXAMPLE")
    print()
    
    batch_texts = [
        "Great job! Excellent work!",
        "Not bad, could be better.",
        "Awful. Terrible. Disgusting.",
    ]
    
    print(f"Analysing batch of {len(batch_texts)} texts...")
    print()
    
    for model_name, model in models.items():
        print(f"{model_name} Batch Results:")
        results = model.analyse_batch(batch_texts)
        
        for j, result in enumerate(results, 1):
            print(f"  [{j}] {result['text'][:40]}... → {result['label']} ({result['score']:.4f})")
        
        print()
    
    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_models()
