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
import cv2
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultimodalChat:
    def __init__(self):
        """Initialize the multimodal chat module with models and resources."""
        self.mode = None  # 'voice' or 'live'
        self.is_running = False
        self.is_processing = False
        self.history = []
        self.current_description = None
        self.encoded_image = None
        self.conversation_active = False
        self.stream = None

        # Audio configuration
        self.sample_rate = 16000
        self.silence_threshold = 0.01
        self.min_silence_duration = 3.0
        self.buffer_duration = 0.1
        self.audio_queue = queue.Queue()

        # Initialize models
        self.stt_model = WhisperModel("medium", device="cuda", compute_type="float16")
        self.llm = OllamaLLM(model="llama3.2", streaming=True)
        self.vision_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", revision="2025-01-09", trust_remote_code=True, device_map={"": "cuda"}
        )
        self.vision_tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2", revision="2025-01-09", trust_remote_code=True
        )
        self.tts_pipeline = KPipeline(lang_code='a')

        # Set up camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.error("Failed to open camera.")
            raise RuntimeError("Camera initialization failed.")

        # Prompt templates
        self.voice_chat_prompt = ChatPromptTemplate.from_template("""
        You are Max, a friendly voice assistant. Respond conversationally with short, correct, and fun answers while maintaining a helpful tone. Keep responses concise unless asked for details. Consider conversation history:

        {history}
        User: {question}
        Max:
        """)

        self.live_chat_prompt = ChatPromptTemplate.from_template("""
        You are Max, an assistant that can see and hear the user. Respond conversationally with short, correct, and fun answers while maintaining a helpful tone. Use the visual context when relevant. Keep responses concise unless asked for details. Consider conversation history:

        {history}
        Visual Context: {visual_context}
        User: {question}
        Max:
        """)

    def start(self, mode):
        """Start the multimodal chat in the specified mode ('voice' or 'live')."""
        if mode not in ['voice', 'live']:
            raise ValueError("Invalid mode. Choose 'voice' or 'live'.")
        self.mode = mode
        self.is_running = True

        # Start audio stream with callback
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self.audio_callback
        )
        self.stream.start()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()

        logging.info(f"{mode.capitalize()} chat started.")

    def stop(self):
        """Stop the multimodal chat and release resources."""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Multimodal chat stopped.")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio input stream."""
        if status:
            logging.warning(f"Audio stream status: {status}")
        if self.is_running and not self.is_processing:
            self.audio_queue.put(indata.copy())

    def capture_image(self):
        """Capture an image from the camera and process it with moondream2."""
        ret, frame = self.cap.read()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.encoded_image = self.vision_model.encode_image(image)
            self.current_description = self.vision_model.answer_question(
                self.encoded_image, "Describe this image.", self.vision_tokenizer
            )
            logging.info("Image captured and described.")
        else:
            logging.error("Failed to capture image.")
            self.encoded_image = None
            self.current_description = "Failed to capture image."

    def generate_response(self, transcribed_text):
        """Generate a response based on the mode and transcribed text."""
        formatted_history = "\n".join(
            f"{turn['speaker']}: {turn['message']}" for turn in self.history
        )

        if self.mode == 'voice':
            prompt = self.voice_chat_prompt.format(history=formatted_history, question=transcribed_text)
            result = ""
            for chunk in self.llm.stream(prompt):
                result += chunk
            self.history.append({"speaker": "User", "message": transcribed_text})
            self.history.append({"speaker": "Max", "message": result})
            return result

        elif self.mode == 'live':
            if not self.conversation_active and "max" in transcribed_text.lower():
                self.capture_image()
                greeting = self.vision_model.answer_question(
                    self.encoded_image,
                    "Greet the user based on this visual description.",
                    self.vision_tokenizer
                )
                self.conversation_active = True
                self.history.append({"speaker": "User", "message": transcribed_text})
                self.history.append({"speaker": "Max", "message": greeting})
                return greeting

            elif self.conversation_active:
                if "moving on" in transcribed_text.lower():
                    self.capture_image()
                    self.history = []
                    response = "Let's talk about something else."
                    self.history.append({"speaker": "User", "message": transcribed_text})
                    self.history.append({"speaker": "Max", "message": response})
                    return response

                elif any(kw in transcribed_text.lower() for kw in ["what", "describe", "how", "where", "see", "can you see"]):
                    if self.encoded_image:
                        response = self.vision_model.answer_question(
                            self.encoded_image, transcribed_text, self.vision_tokenizer
                        )
                    else:
                        response = "I don't have a current image to reference."
                    self.history.append({"speaker": "User", "message": transcribed_text})
                    self.history.append({"speaker": "Max", "message": response})
                    return response

                else:
                    visual_context = self.current_description if self.current_description else "No visual context available."
                    prompt = self.live_chat_prompt.format(
                        history=formatted_history,
                        visual_context=visual_context,
                        question=transcribed_text
                    )
                    result = ""
                    for chunk in self.llm.stream(prompt):
                        result += chunk
                    self.history.append({"speaker": "User", "message": transcribed_text})
                    self.history.append({"speaker": "Max", "message": result})
                    return result
            else:
                return None  # No response if conversation not active and no "max"

    def text_to_speech(self, response):
        """Convert text response to speech and play it."""
        if not response:
            return
        generator = self.tts_pipeline(
            response.strip(),
            voice='af_sky',
            speed=0.9,
            split_pattern=r'\n+'
        )
        for _, _, audio in generator:
            if audio is None or audio.size == 0:
                logging.warning("Empty audio chunk detected.")
                continue
            audio_int16 = (audio * 32767).astype(np.int16)
            sd.play(audio_int16, samplerate=24000)
            sd.wait()

    def process_audio(self):
        """Process audio from the queue, transcribe it, and generate responses."""
        audio_buffer = []
        silence_frames = 0
        min_silence_frames = int(self.min_silence_duration / self.buffer_duration)
        moving_average_energy = []

        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                audio_buffer.append(chunk)

                # Calculate energy for silence detection
                energy = np.sqrt(np.mean(chunk ** 2))
                moving_average_energy.append(energy)
                if len(moving_average_energy) > 10:
                    moving_average_energy.pop(0)
                avg_energy = np.mean(moving_average_energy)

                # Detect silence
                if avg_energy < self.silence_threshold:
                    silence_frames += 1
                else:
                    silence_frames = 0

                if silence_frames >= min_silence_frames and len(audio_buffer) > 0:
                    audio_data = np.concatenate(audio_buffer)
                    audio_buffer = []
                    silence_frames = 0
                    moving_average_energy = []

                    # Process only if audio is longer than 1 second
                    if len(audio_data) > self.sample_rate * 1.0:
                        self.is_processing = True
                        segments, _ = self.stt_model.transcribe(
                            audio_data.flatten().astype(np.float32),
                            beam_size=5
                        )
                        transcribed_text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())

                        if transcribed_text:
                            logging.info(f"Transcribed: {transcribed_text}")
                            response = self.generate_response(transcribed_text)
                            if response:
                                self.text_to_speech(response)
                        self.is_processing = False

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in audio processing: {e}")
                self.is_processing = False
                audio_buffer = []
                silence_frames = 0
                moving_average_energy = []

# Example usage for testing
if __name__ == "__main__":
    chat = MultimodalChat()
    try:
        chat.start('live')  # Test with 'live' or 'voice' mode
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        chat.stop()
