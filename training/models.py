import torch.nn as nn
import torch
from transformers import BertModel
from torchvision import models as vision_models
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import tensorboard

from meld_training import MELDDataset

class TextEncoder(nn.Module):    
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Freeze the BERT model parameters
        for params in self.bert.parameters():
            params.requires_grad = False
        
        self.projection = nn.Linear(768, 128)  # Project to 128 dimensions

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the text encoder.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input tokens.
        
        Returns:
            torch.Tensor: Projected text features.
        """
        # Extract features using BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use the pooled output (CLS token representation)
        pooled_output = outputs.pooler_output

        # Project the pooled output to 128 dimensions
        projected_output = self.projection(pooled_output)
        
        return projected_output
    


class VideoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define your video encoder architecture here
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        for params in self.backbone.parameters():
            params.requires_grad = False

        num_fc = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fc, 128),  # Project to 128 dimensions
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, video_frames):
        # [batch_size, frames, channels, height, width] -> [batch_size, channels, height, width]
        video_frames = video_frames.transpose(1, 2)
        return self.backbone(video_frames)
    

class AudioEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_layers = nn.Sequential(
            # Lower level convolutional layers
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # higher level convolutional layers
            nn.Conv1d(64,128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Now we need to freeze all the parameters to not getting trained 
        for params in self.conv_layers.parameters():
            params.requires_grad = False

        # Now we need to add a linear projection layer with trainable params 
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)            
        )

    def forward(self, x):
        x = x.squeeze(1)

        audio_features = self.conv_layers(x)  # Apply convolutional layers

        return self.projection(audio_features.squeeze(-1))
    

class MultimodalSentimentalModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Encoders for each modality
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion layer for the model 
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classification heads for emotions 
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 7) # 7 emotions(Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise)
        )


        # Classification heads for sentiments
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3) # 3 Sentiments (Positive, Negative, Neutral)
        ) 

    def forward(self, text_inputs, video_frames, audio_features):
        # Pass the inputs through the different encoders first then concatenate the outputs
        text_features = self.text_encoder(
            input_ids=text_inputs['input_ids'], 
            attention_mask=text_inputs['attention_mask']
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate the features from all modalities
        combined_features = torch.cat([
            text_features,
            video_features,
            audio_features
        ], dim=1) 

        # Pass the combined features through the fusion layer 
        fused_features = self.fusion_layer(combined_features)

        # Pass the fused features through each classifier like emotion and sentiment classifiers
        emotion_logits = self.emotion_classifier(fused_features)
        sentiment_logits = self.sentiment_classifier(fused_features)

        # Return the logits for both emotion and sentiment classification
        return {
            'emotion_logits': emotion_logits,
            'sentiment_logits': sentiment_logits
        }
    
# def compute_class_weights(dataset):
#     emotion_counts = torch.zeros(7)  # Assuming 7 emotions
#     sentiment_counts = torch.zeros(3)  # Assuming 3 sentiments

#     skipped = 0

#     total = len(dataset)
#     for i in range(total):
#         sample = dataset[i]

#         if sample is None:
#             skipped += 1
#             continue

#         emotional_label = sample['emotion_label']
#         sentiment_label = sample['sentiment_label']
#         emotion_counts[emotional_label] += 1
#         sentiment_counts[sentiment_label] += 1

#     valid = total - skipped
#     print(f"Skipped {skipped} samples out of {total} due to missing labels.")



    
class Multimodel_trainer(nn.Module):
    def __init__(self, model, train_loader, val_loader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Log the datset sizes
        train_sizes = len(train_loader.dataset)
        val_sizes = len(val_loader.dataset)
        print("\nDataset Sizes:")
        print(f"Train Dataset Size: {train_sizes}")
        print(f"Validation Dataset Size: {val_sizes}\n")
        print(f"Batches per epoches: {len(train_loader)}")

        timestamp = datetime.now().strftime("%b%b_%H-%M-%S")
        base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir = f"{base_dir}/run_{timestamp}"

        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        # Setting the optimizer and loss functions
        self.optimizer = torch.optim.AdamW([
            {'params' : model.text_encoder.parameters(), 'lr': 8e-6},
            {'params' : model.video_encoder.parameters(), 'lr': 8e-5},
            {'params' : model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params' : model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params' : model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params' : model.sentiment_classifier.parameters(), 'lr': 5e-4},
        ], weight_decay=1e-5)

        # Setting the dynamic training scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=2
        )

        self.current_train_losses = None

        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )

        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )

    def log_metrics(self, losses, metrics, phase="train"):
        if phase == "train":
            self.current_train_losses = losses
        else: # validation phase
            self.writer.add_scalar(
                'loss/total/train', self.current_train_losses['total'], self.global_step
            )
            self.writer.add_scalar(
                'loss/total/val', losses['total'], self.global_step
            )
            # Emotion metrics
            self.writer.add_scalar(
                'loss/emotion/train', self.current_train_losses['emotion'], self.global_step
            )
            self.writer.add_scalar(
                'loss/emotion/val', losses['emotion'], self.global_step
            )
            # Sentimental metrics
            self.writer.add_scalar(
                'loss/sentiment/train', self.current_train_losses['sentiment'], self.global_step
            )
            self.writer.add_scalar(
                'loss/sentiment/val', losses['sentiment'], self.global_step
            )

        if metrics: 
            # Emotion metrics
            self.writer.add_scalar(
                f'{phase}/emotion_precison', metrics['emotion_precision'], self.global_step
            )
            self.writer.add_scalar(
                f'{phase}/emotion_accuracy', metrics['emotion_accuracy'], self.global_step
            )
            self.writer.add_scalar(
                f'{phase}/sentiment_precison', metrics['sentimental_precision'], self.global_step
            )
            self.writer.add_scalar(
                f'{phase}/sentiment_accuracy', metrics['sentimental_accuracy'], self.global_step
            )





    def train_step(self):
        self.model.train()
        running_loss = {
            'total' : 0,
            'emotion' : 0,
            'sentiment' : 0
        }

        for batch in self.train_loader:
            # Set al the tensors into one device 
            device = next(self.model.parameters()).device
            text_inputs = {
                'input_ids' : batch['text_inputs']['input_ids'].to(device),
                'attention_mask' : batch['text_inputs']['attention_mask'].to(device)
            }
            video_frames = batch['video_frames'].to(device),
            audio_features = batch['audio_features'].to(device),
            emotion_label = batch['emotion_label'].to(device),
            sentiment_label = batch['sentiment_label'].to(device)

            # Zero gradient mode
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(text_inputs, video_frames, audio_features)


            # Calculate the losses using cross entropy losses 
            # if isinstance(emotion_label, tuple):
            #     emotion_label = emotion_label[0] 
            #     emotional_loss = self.emotion_criterion(
            #         outputs['emotion_logits'], emotion_label
            #     )

            # if isinstance(sentiment_label, tuple):
            #     sentiment_label = sentiment_label[0]
            #     sentimental_loss = self.sentiment_criterion(
            #         outputs['sentiment_logits'], sentiment_label
            #     )

            # Unwrap if tuple else use directly
            if isinstance(emotion_label, tuple):
                emotion_label = emotion_label[0]

            if isinstance(sentiment_label, tuple):
                sentiment_label = sentiment_label[0]

            # Compute losses unconditionally
            emotional_loss = self.emotion_criterion(outputs['emotion_logits'], emotion_label)
            sentimental_loss = self.sentiment_criterion(outputs['sentiment_logits'], sentiment_label)

            total_loss = emotional_loss + sentimental_loss

            # Backward pass
            total_loss.backward()

            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            self.optimizer.step()

            # Update the running loss
            running_loss['total'] += total_loss.item()
            running_loss['emotion'] += emotional_loss.item()
            running_loss['sentiment'] += sentimental_loss.item()

            # Log the metrics
            self.log_metrics(running_loss, None, phase="train")
            
            self.global_step += 1

        return {k: v/len(self.train_loader) for k, v in running_loss.items()}

    def evaluate(self, dataloader, phase="val"):
        # Setting model in evaluation mode
        self.model.eval()
        losses = {
            'total' : 0,
            'emotion' : 0,
            'sentiment' : 0
        }

        all_emotion_preds = []
        all_emootion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.inference_mode():
            for batch in dataloader:
                # Set all the tensors into one device
                device = next(self.model.parameters()).device
                text_inputs = {
                    'input_ids' : batch['text_inputs']['input_ids'].to(device),
                    'attention_mask' : batch['text_inputs']['attention_mask'].to(device)
                }
                video_frames = batch['video_frames'].to(device),
                audio_features = batch['audio_features'].to(device),
                emotion_label = batch['emotion_label'].to(device),
                sentiment_label = batch['sentiment_label'].to(device)

                # Forward pass
                outputs = self.model(text_inputs, video_frames, audio_features)

                # Calculate the losses using cross entropy losses 
                emotional_loss = self.emotion_criterion(
                    outputs['emotion_logits'], emotion_label
                )

                sentimental_loss = self.sentiment_criterion(
                    outputs['sentiment_logits'], sentiment_label
                )

                total_loss = emotional_loss + sentimental_loss

                all_emotion_preds.extend(
                    outputs['emotion_logits'].argmax(dim=1).cpu().numpy()
                )
                all_emootion_labels.extend(emotion_label.cpu().numpy())
                all_sentiment_preds.extend(
                    outputs['sentiment_logits'].argmax(dim=1).cpu().numpy()
                )
                all_sentiment_labels.extend(sentiment_label.cpu().numpy())

                # Update the running loss
                losses['total'] += total_loss.item()
                losses['emotion'] += emotional_loss.item()
                losses['sentiment'] += sentimental_loss.item()
        
        avg_loss = {k: v/len(dataloader) for k, v in losses.items()}

        # compute the metrics
        emotion_precision = precision_score(
            all_emootion_labels, all_emotion_preds, average='weighted'
        )
        emotion_accuracy = accuracy_score(
            all_emootion_labels, all_emotion_preds
        )

        sentimental_precision = precision_score(
            all_sentiment_labels, all_sentiment_preds, average='weighted'
        )
        sentimental_accuracy = accuracy_score(
            all_sentiment_labels, all_sentiment_preds
        )

        # Log the metrics
        self.log_metrics(avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'sentimental_precision': sentimental_precision,
            'sentimental_accuracy': sentimental_accuracy
        }, phase=phase)


        # step scheduler for maintaining the learning rate if the there is no much difference in learning over two more epochs 
        if phase == "val":
            self.scheduler.step(avg_loss['total'])

        return avg_loss, {
            'emotion_precision' : emotion_precision,
            'emotion_accuracy' : emotion_accuracy,
            'sentimental_precision' : sentimental_precision,
            'sentimental_accuracy' : sentimental_accuracy
        }



if __name__ == "__main__":
    dataset = MELDDataset(
        '../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete'
    )