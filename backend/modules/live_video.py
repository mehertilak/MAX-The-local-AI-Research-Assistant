# ---------- Combined Code (Final_Voice_Chat_Visual.py) ----------
# Original voice/tts code remains completely unchanged below
# Only new additions marked with # NEW comments
#MODIFIED BY ADDING FORCE_STOP and signal


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
import atexit
import warnings
from unittest.mock import patch

# NEW VISUAL IMPORTS
import cv2  # NEW
from PIL import Image  # NEW
from transformers import AutoModelForCausalLM, AutoTokenizer  # NEW

# MODIFIED IMPORTS
import signal # MODIFIED

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Global State Variables ---
is_processing = False
current_response = None
tts_ready = threading.Event()

# NEW VISUAL GLOBALS
current_image = None  # NEW
encoded_image = None  # NEW
conversation_active = False  # NEW
cap = cv2.VideoCapture(0)  # NEW Camera init

# --- Faster Whisper Configuration ---
model_size = "medium"
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
sample_rate = 16000
silence_threshold = 0.01
min_silence_duration = 3.0
buffer_duration = 0.1
audio_queue = queue.Queue()

# --- Kokoro TTS Configuration ---
output_folder = r"C:\Users\Tilak\OneDrive\Documents\HackathonProject\VoiceStorage"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# NEW VISION MODEL INIT
vision_model = AutoModelForCausalLM.from_pretrained(  # NEW
    "vikhyatk/moondream2",  # NEW
    revision="2025-01-09",  # NEW
    trust_remote_code=True,  # NEW
    device_map={"": "cuda"}  # NEW
)  # NEW
vision_tokenizer = AutoTokenizer.from_pretrained(  # NEW
    "vikhyatk/moondream2",  # NEW
    revision="2025-01-09",  # NEW
    trust_remote_code=True  # NEW
)  # NEW

# --- Chat Handler Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
conversational_prompt = ChatPromptTemplate.from_template("""
You are Max, a friendly and engaging assistant. Respond conversationally with short, correct, and fun answers while maintaining a helpful tone. Keep responses concise unless asked for details. Consider conversation history:

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
            global encoded_image  # NEW

            self.history.append({"speaker": "User", "message": user_input})
            formatted_history = "\n".join(
                f"{turn['speaker']}: {turn['message']}"
                for turn in self.history
            )

            # NEW VISION HANDLING
            vision_response = None
            if encoded_image and any(kw in user_input.lower() for kw in ["what", "describe", "how", "where", "is there" , "see" ,"can you see"]):
                vision_response = vision_model.answer_question(
                    encoded_image,
                    user_input,
                    vision_tokenizer
                )

            result = ""
            for chunk in (conversational_prompt | self.llm).stream({
                "context": f"Visual Context: {vision_response}" if vision_response else "",  # NEW
                "question": user_input,
                "history": formatted_history
            }):
                result += chunk

            self.history.append({"speaker": "Max", "message": result})
            return result

        except Exception as e:
            logging.error(f"Error during conversation: {e}")
            return None

# ORIGINAL AUDIO FUNCTIONS UNCHANGED
def audio_capture():
    def callback(indata, frames, time, status):
        if status:
            print(f"Error in audio stream: {status}")
        if not is_processing:
            audio_queue.put(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback):
        print("Listening... Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)

def process_audio():
    global is_processing, current_response, conversation_active, encoded_image, current_image  # NEW

    audio_buffer = []
    silence_frames = 0
    min_silence_frames = int(min_silence_duration / buffer_duration)
    moving_average_energy = []

    while True:
        try:
            if is_processing:
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

                if len(audio_data) > sample_rate * 1.0:
                    is_processing = True
                    segments, info = whisper_model.transcribe(
                        audio_data.flatten().astype(np.float32),
                        beam_size=5,
                    )
                    transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())

                    if transcript:
                        print(f"\nUser: {transcript}")

                        # NEW WAKE WORD HANDLING
                        if not conversation_active and "max" in transcript.lower():
                            print("Wake word detected - capturing image...")
                            ret, frame = cap.read()
                            if ret:
                                current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                encoded_image = vision_model.encode_image(current_image)
                                caption = vision_model.answer_question(
                                    encoded_image,
                                    "Describe this person's appearance and emotional state in one sentence.",
                                    vision_tokenizer
                                )
                                conversation_active = True
                                current_response = chat_handler.generate_response(
                                    f"Greet me based on this visual description: {caption}"
                                )
                            else:
                                current_response = "I couldn't capture a clear image. Could you try again?"

                        # NEW CONVERSATION HANDLING
                        elif conversation_active:
                            if "moving on" in transcript.lower():
                                print("Capturing new context image...")
                                ret, frame = cap.read()
                                if ret:
                                    current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                    encoded_image = vision_model.encode_image(current_image)
                                    # Clear conversation history (FIX #1)
                                    chat_handler.history = []
                                    current_response = "Yes , lets talk about something else then..."
                                else:
                                    current_response = "But I still want to talk about this topic...."

                            elif "thank you" in transcript.lower():
                                current_response = "You're welcome! see you soon , have a nice day "
                                tts_ready.set()
                                # Wait for TTS completion (FIX #2)
                                while current_response is not None:
                                    time.sleep(0.1)
                                os.kill(os.getpid(), signal.SIGTERM)

                            else:
                                # NEW - Send subsequent questions to vision model as well!
                                vision_response = None
                                if encoded_image and any(kw in transcript.lower() for kw in ["what", "describe", "how", "where", "is there" , "see" ,"can you see", "am i" ]):
                                    vision_response = vision_model.answer_question(
                                        encoded_image,
                                        transcript,
                                        vision_tokenizer
                                    )

                                result = ""
                                for chunk in (conversational_prompt | chat_handler.llm).stream({
                                    "context": f"Visual Context: {vision_response}" if vision_response else "",  # NEW
                                    "question": transcript,
                                    "history": chat_handler.history  # Make sure history is maintained
                                }):
                                    result += chunk

                                current_response = result
                                chat_handler.history.append({"speaker": "Max", "message": current_response})  # Keep history updated

                        tts_ready.set()

                    else:
                        is_processing = False

        except queue.Empty:
            continue

def text_to_speech():
    global is_processing, current_response
    while True:
        tts_ready.wait()
        try:
            if current_response:
                print(f"\n[TTS] Processing: {current_response}")

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

                    audio_int16 = (audio * 32767).astype(np.int16)

                    print("[TTS] Playing audio response")
                    sd.play(audio_int16, samplerate=24000)
                    sd.wait()
                    audio_played = True
                    time.sleep(0.2)

                if audio_played:
                    current_response = None
                    is_processing = False
                    tts_ready.clear()

        except Exception as e:
            print(f"[TTS ERROR] {str(e)}")
            current_response = None
            is_processing = False
            tts_ready.clear()

# NEW CAMERA PREVIEW THREAD
def camera_preview():
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# MODIFIED CLEANUP
def cleanup():
    print("\nCleaning up...")
    try:
        cap.release()  # NEW
        cv2.destroyAllWindows()  # NEW
    except Exception as e:
        print(f"Error during cleanup: {e}")

# NEW FORCE STOP FUNCTION
def force_stop(signum, frame):
    print("\nForce stopping...")
    cleanup()
    os._exit(0)

# --- Initialization ---
atexit.register(cleanup)
chat_handler = IntegratedChatHandler()

# MODIFIED SIGNAL HANDLER
signal.signal(signal.SIGTERM, force_stop)

# --- Thread Management ---
capture_thread = threading.Thread(target=audio_capture, daemon=True)
process_thread = threading.Thread(target=process_audio, daemon=True)
tts_thread = threading.Thread(target=text_to_speech, daemon=True)
preview_thread = threading.Thread(target=camera_preview, daemon=True)  # NEW

capture_thread.start()
process_thread.start()
tts_thread.start()
preview_thread.start()  # NEW

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping...")
