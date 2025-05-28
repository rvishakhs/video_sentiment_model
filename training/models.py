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
    

    