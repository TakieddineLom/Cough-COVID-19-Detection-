import os
import subprocess

def convert_to_wav(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through all files in the input folder
    for input_file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, input_file)
        output_path = os.path.join(output_folder, os.path.splitext(input_file)[0] + '.wav')
        
        # Check if the file is an audio file
        if not input_file.endswith('.wav') and not input_file.endswith('.ogg') and not input_file.endswith('.webm'):
            print(f"Skipping file '{input_file}' as it is not an audio file.")
            continue
        
        # Run ffmpeg command to convert file to WAV format
        subprocess.run(['ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', '-ar', '44100', output_path])
        
        print(f"Converted file '{input_file}' to '{output_path}'")

# Example usage
input_folder = 'Dataset\coughvid_20211012'
output_folder = 'Converted files'
convert_to_wav(input_folder, output_folder)
