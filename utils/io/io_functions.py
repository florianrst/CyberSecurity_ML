import sys 
import os
from pathlib import Path
from contextlib import contextmanager

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

# Ask Claude for a solution because it was triggering me
@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

if __name__ == "__main__":
    print(get_project_root())