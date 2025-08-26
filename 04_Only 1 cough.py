import os
import librosa
import soundfile as sf

input_folder = '../Covid_19 Project/Audio Prepro/'
output_folder = '../Covid_19 Project/Peak audio/'

for file in os.listdir(input_folder):
    if file.endswith(".wav"):
        audio_file = os.path.join(input_folder, file)
        # read wav data
        audio, sr = librosa.load(audio_file, sr= 44100, mono=True)
        
        # find the highest point in the audio file
        highest_point = max(audio)
        highest_point_idx = list(audio).index(highest_point)
        
        # add 200ms on each side of the peak
    
        left_padding = int(0.3 * sr)
        right_padding = int(0.3 * sr)
        start_idx = max(0, highest_point_idx - left_padding)
        end_idx = min(len(audio), highest_point_idx + right_padding)
        trimmed_audio = audio[start_idx:end_idx]
        
        # write the trimmed audio to file
        output_file = os.path.join(output_folder, file)
        sf.write(output_file, trimmed_audio, sr)
