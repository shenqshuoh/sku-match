import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import detect


def test_imports():
    assert callable(detect)


if __name__ == "__main__":
    test_imports()
    print("All smoke tests passed!")
