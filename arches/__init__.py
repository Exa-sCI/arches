import sys

if sys.version_info < (3, 10):
    sys.exit(f"Must be using at least Python 3.10, you are using {sys.version}")
