"""Basic tests for utility functions. Run with: python -m pytest"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_normalize():
    from app import normalize
    assert normalize("  Happy   DAY ") == "happy day"

def test_tokens():
    from app import tokens
    assert tokens("Well-made words, run!") == ["well-made", "words", "run"]
