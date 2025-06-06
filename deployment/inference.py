import os
import subprocess
import cv2
import numpy as np
import torch
import torchaudio
from training.models import MultimodalSentimentalModel
import whisper
from transformers import AutoTokenizer

class Video_processing: 
    def video_frames(self, video_path):
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

class Audio_processing:
    def audio_features(self, video_path):
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

class VideoUtteranceProcessor:
    def __init__(self):
        self.video_processor = Video_processing()
        self.audio_processor = Audio_processing()

    def extract_segments(self, video_path, start_time, end_time, temp_dir='/tmp'):
        os.makedirs(temp_dir, exist_ok=True)
        segment_path = os.path.join(temp_dir, f"segment_{start_time}_{end_time}.mp4")

        
        subprocess.run([
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-y'
            segment_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
            raise ValueError(f"Segment extraction failed for {video_path} from {start_time} to {end_time}")
            
        return segment_path

def load_model(model_dir):

    # Check the device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultimodalSentimentalModel()

    model_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model",'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
    print(f"Loading model from {model_path} on device {device}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return {
        'model' : model,
        'tokenizer' : AutoTokenizer.from_pretrained("bert-base-uncased"),
        'transcriber' : whisper.load_model("base.en").to(device),
        'device' : device
    }

if __name__ == "__main__":
    