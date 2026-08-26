from pydub import AudioSegment

def trim_audio(input_path, output_path, start_ms, end_ms):
    # Load audio file
    audio = AudioSegment.from_file(input_path)
    # Trim the audio
    trimmed_audio = audio[start_ms:end_ms]
    # Export trimmed audio
    trimmed_audio.export(output_path, format="wav")

# Example usage:
# Trim from 5 seconds to 10 seconds (5000 ms to 10000 ms)
start = 85
end = 95
input_path = "Audio/audio.wav"
output_path = "Audio/trimmed.wav"
trim_audio(input_path, output_path, start*1000, end*1000)