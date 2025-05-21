import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import threading
import time
import logging
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from kokoro import KPipeline
import os
import warnings
from unittest.mock import patch

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Global State Variables ---
is_processing = False  # Controls when we listen for new input
current_response = None  # Stores the LLM response
tts_ready = threading.Event()  # Signals when TTS is ready to process

# --- Faster Whisper Configuration ---
model_size = "medium"
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
sample_rate = 16000  # Whisper expects 16kHz audio
silence_threshold = 0.01  # Adjust based on your environment
min_silence_duration = 3.0  # Minimum silence to consider speech ended
buffer_duration = 0.1  # Duration of each audio buffer
audio_queue = queue.Queue()  # Queue for storing audio chunks

# --- Kokoro TTS Configuration ---
output_folder = r"C:\Users\Tilak\OneDrive\Documents\HackathonProject\VoiceStorage"  # Update with your desired path
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Chat Handler Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
conversational_prompt = ChatPromptTemplate.from_template("""
You are Max, a friendly and engaging voice assistant. Respond conversationally with short, correct, and fun answers while maintaining a helpful tone. remeber that you are a project which is being shown to the judges, so when th user asks you to greet them just introduce yourself as the project Keep responses concise unless asked for details. Consider conversation history:

{history}
Context: {context}
User: {question}
Max: 
""")

# --- Combined Classes/Functions ---
class IntegratedChatHandler:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2", streaming=True)
        self.history = []

        # Initialize TTS pipeline with monkey patch
        def patched_open(*args, **kwargs):
            return open(*args, **kwargs, encoding='utf-8')

        with patch('kokoro.pipeline.open', patched_open):
            print("Initializing TTS pipeline...")
            self.tts_pipeline = KPipeline(lang_code='a')

    def generate_response(self, user_input):
        try:
            self.history.append({"speaker": "User", "message": user_input})
            formatted_history = "\n".join(
                f"{turn['speaker']}: {turn['message']}"
                for turn in self.history
            )

            result = ""
            for chunk in (conversational_prompt | self.llm).stream({
                "context": "",
                "question": user_input,
                "history": formatted_history
            }):
                result += chunk

            self.history.append({"speaker": "Max", "message": result})
            return result

        except Exception as e:
            logging.error(f"Error during conversation: {e}")
            return None

def audio_capture():
    def callback(indata, frames, time, status):
        global is_processing
        if status:
            print(f"Error in audio stream: {status}")
        if not is_processing:
            audio_queue.put(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback):
        print("Audio capture started.")
        while True:  # Keep the thread alive
            time.sleep(1)  # Sleep for a while

def process_audio():
    global is_processing, current_response
    audio_buffer = []
    silence_frames = 0
    min_silence_frames = int(min_silence_duration / buffer_duration)
    moving_average_energy = []

    while True:
        try:
            if is_processing:
                # print("process audio: sleeping")
                time.sleep(0.1)
                continue

            chunk = audio_queue.get(timeout=1.0)
            audio_buffer.append(chunk)

            # Energy calculation
            energy = np.sqrt(np.mean(chunk**2))
            moving_average_energy.append(energy)
            if len(moving_average_energy) > 10:
                moving_average_energy.pop(0)
            avg_energy = np.mean(moving_average_energy)

            # Silence detection
            if avg_energy < silence_threshold:
                silence_frames += 1
            else:
                silence_frames = 0

            if silence_frames >= min_silence_frames and len(audio_buffer) > 0:
                audio_data = np.concatenate(audio_buffer)
                audio_buffer = []
                silence_frames = 0

                if len(audio_data) > sample_rate * 1.0:  # Only process audio longer than 1 second
                    is_processing = True
                    segments, info = whisper_model.transcribe(
                        audio_data.flatten().astype(np.float32),
                        beam_size=5,
                    )
                    transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())

                    if transcript:
                        print(f"\nUser: {transcript}")
                        current_response = chat_handler.generate_response(transcript)
                        if current_response:
                            print(f"Max: {current_response}")
                            tts_ready.set()  # Signal TTS to process the response
                        else:
                            is_processing = False
                    else:
                        is_processing = False

        except queue.Empty:
            continue

def text_to_speech():
    global is_processing, current_response
    while True:
        tts_ready.wait()  # Wait for the signal
        try:
            if current_response:
                print(f"\n[TTS] Processing: {current_response}")

                # Generate audio
                generator = chat_handler.tts_pipeline(
                    current_response.strip(),
                    voice='af_sky',
                    speed=0.9,
                    split_pattern=r'\n+'
                )

                audio_played = False
                for _, _, audio in generator:
                    if audio is None or audio.size == 0:
                        print("[TTS] Empty audio chunk detected")
                        continue

                    # Normalize and convert to int16
                    audio_int16 = (audio * 32767).astype(np.int16)

                    # Play audio directly using sounddevice
                    print("[TTS] Playing audio response")
                    sd.play(audio_int16, samplerate=24000)
                    sd.wait() 
                    audio_played = True
                    time.sleep(0.2)

                # Reset state after playback
                if audio_played:
                    current_response = None
                    is_processing = False
                    tts_ready.clear()

        except Exception as e:
            print(f"[TTS ERROR] {str(e)}")
            current_response = None
            is_processing = False
            tts_ready.clear()

# --- Initialization ---
chat_handler = IntegratedChatHandler()  # Create chat_handler instance

# --- Thread Management ---
capture_thread = threading.Thread(target=audio_capture, daemon=True)
process_thread = threading.Thread(target=process_audio, daemon=True)
tts_thread = threading.Thread(target=text_to_speech, daemon=True)

capture_thread.start()
process_thread.start()
tts_thread.start()

def run_voice_chat():
    print("Voice chat ready")

if __name__ == "__main__":
    run_voice_chat()

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("\nStopping...")
