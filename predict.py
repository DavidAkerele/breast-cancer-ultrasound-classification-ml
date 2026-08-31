import os
import sys

# Ensure src module is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import main

if __name__ == "__main__":
    main()
