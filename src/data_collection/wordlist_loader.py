import os
from pathlib import Path

def load_wordlist(filename: str) -> list[str]:
    """
    Load a list of words from a text file, one word per line.
    
    Args:
        filename: Path to the wordlist file (relative to project root)
    
    Returns:
        List of words/phrases
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an issue reading the file
    """
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Wordlist file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    return words