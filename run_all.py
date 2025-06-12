from src.data_preprocessing import processor
from src.train_model import trainer
from evaluate_2021 import evaluate_pred_2021
from pathlib import Path

ROOT = Path(__file__).parents[0].__str__()

def main():
    print("\nStep 1: Running data preprocessing...")
    processor(ROOT)

    print("\nStep 2: Training model...")
    trainer(ROOT)

    print("\nStep 3: Running test prediction for Jan 2021...")
    evaluate_pred_2021(ROOT)

if __name__ == main():
    main(ROOT)