import torch
from training.meld_training import prepare_dataloaders
from training.models import MultimodalSentimentalModel, Multimodel_trainer

def train_and_evaluate(model, train_loader, val_loader, num_epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Trains and evaluates the MultimodalSentimentalModel locally.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        num_epochs (int): Number of training epochs.
        device (str): Device to use for training ('cuda' or 'cpu').
    """
    # Move the model to the specified device
    model.to(device)

    # Initialize the trainer
    trainer = Multimodel_trainer(model, train_loader, val_loader)

    print(f"\nTraining on device: {device}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        # Training phase
        train_losses = trainer.train_step()
        print(f"Train Losses: {train_losses}")

        # Evaluation phase
        val_losses, val_metrics = trainer.evaluate(val_loader, phase='val')
        print(f"Validation Losses: {val_losses}")
        print(f"Validation Metrics: {val_metrics}")

    print("\nTraining completed.")
    return model

# Create the model instance
model = MultimodalSentimentalModel()
train_loader, test_loader, dev_loader = prepare_dataloaders(
    './dataset/train/train_sent_emo.csv', './dataset/train/train_splits',
    './dataset/dev/dev_sent_emo.csv', './dataset/dev/dev_splits_complete',
    './dataset/test/test_sent_emo.csv', './dataset/test/output_repeated_splits_test'  
)
num_epochs = 10  # Set the number of epochs for training


if __name__ == "__main__":
    train_and_evaluate(model=model, 
                       train_loader=train_loader, 
                       val_loader=dev_loader, 
                       num_epochs=num_epochs)
    print("Training job started successfully.")