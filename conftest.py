"""Put the project root on sys.path so `import qkd` works under pytest."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
