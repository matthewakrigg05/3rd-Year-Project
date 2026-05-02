import os
from pathlib import Path

def load_wordlist(filename: str) -> list[str]:
    """Load words from a text file (one per line) relative to the project root."""
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Wordlist file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    return words