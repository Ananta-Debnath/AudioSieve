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
start = 50
end = start + 40
input_path = "Audio/Music/poran-jai-jolia-re-title-track-dev-subhashree-jeet-gannguli-svf-kuda-4-o.wav"
output_path = "Audio/trimmed.wav"
trim_audio(input_path, output_path, start*1000, end*1000)