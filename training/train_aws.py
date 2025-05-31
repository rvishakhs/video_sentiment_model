import json
import os
import argparse
import torchaudio
import torch
import tqdm

from training.meld_training import prepare_dataloaders
from models import MultimodalSentimentalModel, Multimodel_trainer

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



def main():
    # Install ffmpeg

    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Track the GPU memory usage
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated()
        print(f"Initial GPU Usage: {memory_used / (1024 ** 3):.2f} GB")

    # Import the dataloader and model after parsing arguments
    train_loader, test_loader, dev_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv=os.path.join(args.validation_dir, 'dev_sent_emo.csv'),
        dev_video_dir=os.path.join(args.validation_dir, 'dev_splits_complete'),
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir=os.path.join(args.test_dir, 'output_repeated_splits_test'),
        batch_size=args.batch_size
    )

    print(f"Training CSV directory: {os.path.join(args.train_dir, 'train_sent_emo.csv')}")
    print(f"Training Video directory: {os.path.join(args.train_dir, 'train_splits')}")

    model = MultimodalSentimentalModel().to(device)
    trainer = Multimodel_trainer(model=model,
                                 train_loader=train_loader,
                                 test_loader=test_loader,
                                 dev_loader=dev_loader,
                                 epochs=args.epochs,
                                 learning_rate=args.learning_rate,
                                 device=device)
    
    best_trainng_loss = float('inf')
    metrics_data = {
        "train_losses" : [],
        "validation_losses" : [],
        "test_losses" : [],
        "epochs" : [],
    }

    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        training_loss = trainer.train_step()
        eval_loss, metrics = trainer.evaluate(dev_loader)

        # Track the metrics 
        metrics_data["train_losses"].append(training_loss["total"])
        metrics_data["validation_losses"].append(eval_loss["total"])
        metrics_data["epochs"].append(epoch)

        # Log metrics in sagemaker format 
        print(json.dumps({
            "metrics" : [
                {"Name": "train_loss", "Value": training_loss["total"]},
                {"Name": "validation_loss", "Value": eval_loss["total"]},
                {"Name": "validation: Emotion_precision", "Value":metrics["emotion_precision"]},
                {"Name": "validation: sentimental_precision", "Value":metrics["sentimental_precision"]},
                {"Name": "validation: emotion_accuracy", "Value":metrics["emotion_accuracy"]},
                {"Name": "validation: sentimental_accuracy", "Value":metrics["sentimental_accuracy"]},
            ]
        }))

        # Track the GPU memory usage in each epoch
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated()
            print(f"Peak GPU Usage: {memory_used / (1024 ** 3):.2f} GB")

        # Save the model if the validation loss improves
        if eval_loss["total"] < best_trainng_loss:
            best_trainng_loss = eval_loss["total"]
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))    

    # Evaluating on test dataset
    test_loss, test_metrics = trainer.evaluate(test_loader)
    metrics_data["test_losses"].append(test_loss["total"])

    # Log test metrics in sagemaker format 
    print(json.dumps({
        "metrics" : [
            {"Name": "test_loss", "Value": test_loss["total"]},
            {"Name": "test: Emotion_precision", "Value":test_metrics["emotion_precision"]},
            {"Name": "test: sentimental_precision", "Value":test_metrics["sentimental_precision"]},
            {"Name": "test: emotion_accuracy", "Value":test_metrics["emotion_accuracy"]},
            {"Name": "test: sentimental_accuracy", "Value":test_metrics["sentimental_accuracy"]},
        ]
    }))

    




    







if __name__ == "__main__":
    main()