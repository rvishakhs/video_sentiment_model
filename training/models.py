import torch.nn as nn
import torch
from transformers import BertModel
from torchvision import models as vision_models


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
        x = x.squeeze(1)  # Remove the channel dimension

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

        # Setting the optimizer and loss functions
        self.optimizer = torch.optim.AdamW([
            {'params' : model.text_encoder.parameters(), 'lr': 8e-6},
            {'params' : model.video_encoder.parameters(), 'lr': 8e-5},
            {'params' : model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params' : model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params' : model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params' : model.sentiment_classifier.parameters(), 'lr': 5e-4},
        ], weight_decay=1e-5)


#if __name__ == "__main__":
