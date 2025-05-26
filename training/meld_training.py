from torch.utils.data import Dataset
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import subprocess
import torchaudio

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Emotions and their corresponding indices
        self.emotions = {
            "neutral": 0,
            "sadness": 1,
            "joy": 2,
            "anger": 3,
            "fear": 4,
            "disgust": 5
        }

        self.sentiments_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video file {video_path}")
            
            # try and read first frame to validate the video file
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Could not read frames from video file {video_path}")
            
            # Reset the index to not skip the first frame 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


            while len(frames) < 30 and cap.isOpened(): # Limit to 30 frames
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))  # Resize to 224x224
                frame = frame / 255.0  # Normalize the frame
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Error reading video file {video_path}: {e}")
        
        finally:
            cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames could be extracted from the given video file {video_path}")
        
        # Pad and trucate the frame if it's less than 30 framesn and more than 30 frames 
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))  # Pad with zeros

        else:
            frames = frames[:30]

        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2) # Convert the normal array to np array and rearrange dimensions to (frames, channels, height, width)
    
    def _audio_extract_features(self, video_path):
        print(f"Extracting audio features from {video_path}")
        audio_path = video_path.replace('.mp4', '.wav')  #replace video extension with audio extension
        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

             # Extracting audio features using torchaudio
        
            mel_spectogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512,
        
            )

            mel_spec = mel_spectogram(waveform)

            # Normalizing the mel spectrogram
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))

            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error happend whicle extracting audio as subprocessing {e}")

        except Exception as e:
            raise ValueError(f"Error extracting audio features from {video_path}: {e}")
        
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
    # Altering the length function
    def __len__(self):
        return len(self.data)
    
    # Altering the getitem function
    def __getitem__(self, idx):
        row =self.data.iloc[idx]
        video_file_name = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
        path = os.path.join(self.video_dir, video_file_name)

        video_path = os.path.exists(path)

        if not video_path:
            raise FileNotFoundError(f"Video file {video_file_name} not found in {self.video_dir}")
        
        # Tokenizing the text
        text = self.tokenizer(row['Utterance'],
                              padding='max_length',
                              truncation=True,
                              max_length=128,
                              return_tensors='pt')
        
        # Extracting video frames 
        #video_frames = self._load_video_frames(path)
        auido_features = self._audio_extract_features(path)
        print(auido_features)
        
        
    
if __name__ == "__main__":
    # Example usage
    csv_path = Path(".") / "dataset" / "dev" / "dev_sent_emo.csv"
    video_dir = Path(".") / "dataset" / "dev" / "dev_splits_complete"
    dataset = MELDDataset(csv_path, 
                          video_dir)
    
    print(dataset[0])


