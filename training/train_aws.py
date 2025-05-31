import os
import argparse

# initialise parameters
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '.')
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'exapandable_segments:True'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer")

    # Data paths
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING, help="Path to the training data directory")
    parser.add_argument("--validation-dir", type=str, default=SM_CHANNEL_VALIDATION, help="Path to the validation data directory")
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST, help="Path to the test data directory")
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR, help="Directory to save the trained model")

    