from faster_whisper import WhisperModel
import os

def seconds_to_mmss(seconds):
    if not isinstance(seconds, (int, float)):
        raise ValueError("Input must be a float or integer.")

    if seconds < 0:
        raise ValueError("Input seconds must be non-negative.")

    minutes = int(seconds // 60)
    seconds %= 60

    return f"{minutes:02d}:{int(seconds):02d}"

model_size = "large-v2"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float32")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Create the output folder if it doesn't exist
output_folder = "audios_output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all audio files in the "input_audios" folder
input_folder = "audios_input"
audio_files = [f for f in os.listdir(input_folder) if f.endswith((".mp3"))]  # Modified line

for audio_file in audio_files:
    # Load the audio and pad/trim it to fit 30 seconds
    audio_path = os.path.join(input_folder, audio_file)
    segments, info = model.transcribe(audio_path, language="pt", best_of=5, condition_on_previous_text=False,
                                      beam_size=5, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    final_text_with_timestamps = ""
    final_text = ""

    for segment in segments:
        final_text_with_timestamps += "[%s -> %s] %s" % (seconds_to_mmss(segment.start), seconds_to_mmss(segment.end), segment.text.strip()) + "\n"
        final_text += segment.text.strip() + "\n"
        print("[%s -> %s] %s" % (seconds_to_mmss(segment.start), seconds_to_mmss(segment.end), segment.text))

    # Write the recognized text to the output file
    output_final_file_name = os.path.splitext(audio_file)[0] + "_output_final.txt"
    output_with_timestamps_file_name = os.path.splitext(audio_file)[0] + "_output_with_timestamps.txt"
    output_path = os.path.join(output_folder, output_final_file_name)
    output_with_timestamps_path = os.path.join(output_folder, output_with_timestamps_file_name)
    with open(output_path, "w") as file:
        file.write(final_text)
    with open(output_with_timestamps_path, "w") as file:
        file.write(final_text_with_timestamps)
