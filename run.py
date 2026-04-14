# Imports
import sys

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python run.py <command>")
        sys.exit(1)

    match sys.argv[1]:
        case 'preprocess':
            from data import run_preprocessing
            run_preprocessing()
            
        case 'check-models':
            from app.models import fast_check
            fast_check()

        case _:
            print("Available commands : [preprocess, check-models]")