import logging
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from modules.chat_module import ChatHandler, conversational_prompt
from modules.rag_model import DocumentRAG, upload_file, process_rag_query
from modules.web_crawler import EphemeralRAG
from modules import image_scanner
from tempfile import TemporaryDirectory
import asyncio
from PIL import Image
import uuid
import subprocess
import signal
import threading
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from kokoro import KPipeline
import warnings
from unittest.mock import patch
import cv2

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global Model Initialization
model_size = "medium"
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
output_folder = r"C:\Users\Tilak\OneDrive\Documents\HackathonProject\VoiceStorage"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def patched_open(*args, **kwargs):
    return open(*args, **kwargs, encoding='utf-8')

with patch('kokoro.pipeline.open', patched_open):
    logging.info("Initializing TTS pipeline globally...")
    tts_pipeline = KPipeline(lang_code='a')

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Initialize the chat handler
chat_handler = ChatHandler()

# Initialize the RAG model
rag_model = DocumentRAG()

# Initialize the Web Crawler
web_crawler = None

# Initialize mode tracker
current_mode = 'chat'

# Global variables for live chat
live_chat_instance = None
live_video_lock = threading.Lock()

# Global variables for voice chat
voice_chat_instance = None
voice_chat_lock = threading.Lock()

# Serve Frontend
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.isfile(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/detected_images/<filename>')
def serve_detected_image(filename):
    return send_from_directory(image_scanner.output_folder, filename)

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handles general chat requests."""
    global current_mode
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            logging.warning("Invalid request: No message received")
            return jsonify({"error": "Invalid request: No message provided"}), 400

        user_message = data['message']
        response = chat_handler.handle_conversation(conversational_prompt, user_message)

        if response is None:
             logging.error("Chat response generation failed")
             return jsonify({"error": "Failed to generate a response"}), 500
        return jsonify({"response": response, "success":True})

    except Exception as e:
        logging.error(f"Unexpected error during chat handling: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/upload', methods=['POST'])
def handle_upload():
    """Handles file uploads for RAG."""
    global current_mode
    try:
        if 'file' not in request.files:
             logging.warning("Invalid request: No file received")
             return jsonify({"error": "No file part", "success":False}), 400

        file = request.files['file']

        if file.filename == '':
           logging.warning("Invalid request: No file selected")
           return jsonify({'error': 'No selected file', "success":False}), 400

        temp_dir = TemporaryDirectory()
        upload_folder = temp_dir.name if not rag_model.persist_data else "data/uploads"
        file_path, filename = upload_file(file, upload_folder)

        if not file_path:
            temp_dir.cleanup()
            return jsonify({"error": "File saving failed", "success":False}), 500

        content = rag_model.load_document(file_path)

        if not content:
            temp_dir.cleanup()
            return jsonify({"error": "Document loading failed", "success":False}), 500

        documents = rag_model.process_content(content, filename)

        if not documents:
             temp_dir.cleanup()
             return jsonify({"error": "Document processing failed", "success":False}), 500

        rag_model.create_vector_store(documents)
        temp_dir.cleanup()
        return jsonify({'filename': filename, "success":True}), 200

    except Exception as e:
         logging.error(f"Error during file upload: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/rag_query', methods=['POST'])
def handle_rag_query():
    """Handles RAG queries."""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            logging.warning("Invalid request: No question received")
            return jsonify({"error": "Invalid request: No question provided", "success":False}), 400

        question = data['question']
        response = process_rag_query(question, rag_model)

        if response is None:
            logging.error("Error getting the answer from RAG model")
            return jsonify({"error": "Failed to generate a response", "success":False}), 500

        return jsonify({'response':response['result'], 'document_name': response['source_documents'][0].metadata['source'], "success":True}), 200

    except Exception as e:
         logging.error(f"Error during rag query: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/cleanup_rag', methods=['POST'])
def handle_cleanup_rag():
    """Handles the cleanup of the rag model."""
    try:
        rag_model.cleanup()
        return jsonify({'message': 'RAG mode cleanup successfully', "success":True}), 200
    except Exception as e:
         logging.error(f"Error during rag cleanup: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/crawl_website', methods=['POST'])
async def handle_crawl_website():
    """Handles website crawling requests."""
    global current_mode
    global web_crawler
    try:
         data = request.get_json()
         if not data or 'url' not in data:
            logging.warning("Invalid request: No URL received")
            return jsonify({"error": "Invalid request: No URL provided", "success":False}), 400

         url = data['url']
         if web_crawler:
             web_crawler.cleanup()
         web_crawler = EphemeralRAG()

         content = await web_crawler.crawl_website(url)

         if not content:
            return jsonify({"error": "Website crawling failed", "success":False}), 500

         documents = web_crawler.process_content(content, url)
         web_crawler.create_vector_store(documents)

         return jsonify({'message': 'Website crawled successfully', "success":True}), 200

    except Exception as e:
         logging.error(f"Error during web crawling: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/web_query', methods=['POST'])
def handle_web_query():
    """Handles web crawl queries."""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            logging.warning("Invalid request: No question received")
            return jsonify({"error": "Invalid request: No question provided", "success":False}), 400

        question = data['question']
        if web_crawler is None:
            logging.error("Web Crawler is None")
            return jsonify({"error": "Web Crawler not initialized", "success":False}), 500

        response = web_crawler.setup_qa_chain().invoke({"query":question})

        if response is None:
            logging.error("Error getting the answer from Web Crawler model")
            return jsonify({"error": "Failed to generate a response", "success":False}), 500

        return jsonify({'response':response['result'], 'document_name': response['source_documents'][0].metadata['source'], "success":True}), 200

    except Exception as e:
         logging.error(f"Error during web query: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/cleanup_web', methods=['POST'])
def handle_cleanup_web():
    """Handles the cleanup of the web crawler."""
    try:
         if web_crawler:
             web_crawler.cleanup()
         return jsonify({'message': 'Web crawler mode cleanup successfully', "success":True}), 200
    except Exception as e:
         logging.error(f"Error during web cleanup: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/upload_image', methods=['POST'])
def handle_upload_image():
     """Handles image uploads for image processing."""
     global current_mode
     try:
          if 'image' not in request.files:
               logging.warning("Invalid request: No image received")
               return jsonify({"error": "No image part", "success":False}), 400
          image_file = request.files['image']
          if image_file.filename == '':
               logging.warning("Invalid request: No image selected")
               return jsonify({'error': 'No selected image', "success":False}), 400

          temp_dir = TemporaryDirectory()
          filename = os.path.join(temp_dir.name, f"{uuid.uuid4()}{os.path.splitext(image_file.filename)[1]}")
          image_file.save(filename)

          with Image.open(filename) as image:
              caption_result = image_scanner.get_image_caption(image, filename)

          image_scanner.set_temp_dir(temp_dir)

          if not caption_result['success']:
               logging.error(f"Error during image caption: {caption_result['error']}")
               return jsonify({'error': 'Error generating image caption', "success":False}), 500

          current_mode = 'image_chat'
          return jsonify({'response': "I am looking at the image you sent me. I have to say it is looking interesting", 'caption': caption_result['response'], "success":True}), 200

     except Exception as e:
          logging.error(f"Error during image upload: {e}")
          return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/image_query', methods=['POST'])
def handle_image_query():
     """Handles image processing queries."""
     try:
          question = request.form.get('question')
          if not question:
               logging.warning("Invalid request: No question received in FormData")
               return jsonify({"error": "Invalid request: No question provided", "success":False}), 400

          if "detect" in question.lower():
            object_to_detect = question.split("detect")[1].strip()
            detection_result = image_scanner.detect_objects(object_to_detect)
            if not detection_result["success"]:
                logging.error(f"Error during object detection: {detection_result['error']}")
                return jsonify({"error": detection_result['response'], "success": False})
            return jsonify(detection_result)

          elif "point" in question.lower():
            query = question.split("point")[1].strip()
            point_result = image_scanner.point_at_object(query)
            if not point_result["success"]:
                logging.error(f"Error during object pointing: {point_result['error']}")
                return jsonify({"error": point_result['response'], "success": False})
            return jsonify(point_result)

          else:
             answer_result = image_scanner.answer_image_question(question)
             if not answer_result["success"]:
                logging.error(f"Error during image question answering: {answer_result['error']}")
                return jsonify({"error": answer_result["error"], "success":False}), 500
             return jsonify(answer_result)

     except Exception as e:
          logging.error(f"Error during image processing: {e}")
          return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

@app.route('/cleanup_image', methods=['POST'])
def handle_cleanup_image():
    """Handles the cleanup of the image resources."""
    global current_mode
    try:
        cleanup_result = image_scanner.cleanup_image_resources()
        if not cleanup_result["success"]:
            logging.error(f"Error during image cleanup: {cleanup_result['error']}")
            return jsonify({"error": "Error during image cleanup", "success":False}), 500
        current_mode = 'chat'
        return jsonify(cleanup_result), 200
    except Exception as e:
         logging.error(f"Error during image cleanup: {e}")
         return jsonify({"error": f"An unexpected error occurred: {e}", "success":False}), 500

class VoiceChatHandler:
    def __init__(self):
        self.sample_rate = 16000
        self.silence_threshold = 0.01
        self.min_silence_duration = 3.0
        self.buffer_duration = 0.1
        self.audio_queue = queue.Queue()
        self.is_processing = False
        self.current_response = None
        self.tts_ready = threading.Event()
        self.running = False
        self.llm = OllamaLLM(model="llama3.2", streaming=True)
        self.history = []
        self.conversational_prompt = ChatPromptTemplate.from_template("""
You are Max, a friendly and engaging voice assistant. Respond conversationally with short, correct, and fun answers while maintaining a helpful tone. Remember that you are a project being shown to the judges, so when the user asks you to greet them, just introduce yourself as the project. Keep responses concise unless asked for details. Consider conversation history:

{history}
Context: {context}
User: {question}
Max: 
""")
        self.capture_thread = threading.Thread(target=self.audio_capture, daemon=True)
        self.process_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.tts_thread = threading.Thread(target=self.text_to_speech, daemon=True)

    def generate_response(self, user_input):
        try:
            self.history.append({"speaker": "User", "message": user_input})
            formatted_history = "\n".join(f"{turn['speaker']}: {turn['message']}" for turn in self.history)
            result = ""
            for chunk in (self.conversational_prompt | self.llm).stream({
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

    def audio_capture(self):
        def callback(indata, frames, time, status):
            if status:
                logging.error(f"Error in audio stream: {status}")
            if not self.is_processing and self.running:
                self.audio_queue.put(indata.copy())

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', callback=callback):
            logging.info("Audio capture started.")
            while self.running:
                time.sleep(1)

    def process_audio(self):
        audio_buffer = []
        silence_frames = 0
        min_silence_frames = int(self.min_silence_duration / self.buffer_duration)
        moving_average_energy = []

        while self.running:
            try:
                if self.is_processing:
                    time.sleep(0.1)
                    continue
                chunk = self.audio_queue.get(timeout=1.0)
                audio_buffer.append(chunk)
                energy = np.sqrt(np.mean(chunk**2))
                moving_average_energy.append(energy)
                if len(moving_average_energy) > 10:
                    moving_average_energy.pop(0)
                avg_energy = np.mean(moving_average_energy)
                if avg_energy < self.silence_threshold:
                    silence_frames += 1
                else:
                    silence_frames = 0

                if silence_frames >= min_silence_frames and len(audio_buffer) > 0:
                    audio_data = np.concatenate(audio_buffer)
                    audio_buffer = []
                    silence_frames = 0
                    if len(audio_data) > self.sample_rate * 1.0:
                        self.is_processing = True
                        segments, info = whisper_model.transcribe(
                            audio_data.flatten().astype(np.float32),
                            beam_size=5,
                        )
                        transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
                        if transcript:
                            logging.info(f"User: {transcript}")
                            self.current_response = self.generate_response(transcript)
                            if self.current_response:
                                logging.info(f"Max: {self.current_response}")
                                self.tts_ready.set()
                            else:
                                self.is_processing = False
                        else:
                            self.is_processing = False
            except queue.Empty:
                continue

    def text_to_speech(self):
        while self.running:
            self.tts_ready.wait()
            try:
                if self.current_response:
                    logging.info(f"[TTS] Processing: {self.current_response}")
                    generator = tts_pipeline(
                        self.current_response.strip(),
                        voice='af_sky',
                        speed=0.9,
                        split_pattern=r'\n+'
                    )
                    audio_played = False
                    for _, _, audio in generator:
                        if audio is None or audio.size == 0:
                            logging.warning("[TTS] Empty audio chunk detected")
                            continue
                        audio_int16 = (audio * 32767).astype(np.int16)
                        logging.info("[TTS] Playing audio response")
                        sd.play(audio_int16, samplerate=24000)
                        sd.wait()
                        audio_played = True
                        time.sleep(0.2)
                    if audio_played:
                        self.current_response = None
                        self.is_processing = False
                        self.tts_ready.clear()
            except Exception as e:
                logging.error(f"[TTS ERROR] {str(e)}")
                self.current_response = None
                self.is_processing = False
                self.tts_ready.clear()

    def start(self):
        if not self.running:
            self.running = True
            self.capture_thread.start()
            self.process_thread.start()
            self.tts_thread.start()
            logging.info("Voice chat started.")

    def stop(self):
        if self.running:
            self.running = False
            self.capture_thread.join(timeout=1)
            self.process_thread.join(timeout=1)
            self.tts_thread.join(timeout=1)
            logging.info("Voice chat stopped.")

class LiveChatHandler:
    def __init__(self):
        # Audio settings
        self.sample_rate = 16000
        self.silence_threshold = 0.01
        self.min_silence_duration = 3.0
        self.buffer_duration = 0.1
        self.audio_queue = queue.Queue()
        
        # State variables
        self.is_processing = False
        self.current_response = None
        self.tts_ready = threading.Event()
        self.running = False
        self.conversation_active = False
        
        # Vision settings
        self.cap = cv2.VideoCapture(0)
        self.current_image = None
        self.encoded_image = None
        
        # Chat configuration
        self.llm = OllamaLLM(model="llama3.2", streaming=True)
        self.history = []
        self.conversational_prompt = ChatPromptTemplate.from_template("""
You are Max, a friendly and engaging assistant. Respond conversationally with short, correct, and fun answers while maintaining a helpful tone. The visual context describes the user, not you. Use it to inform your responses about the user accurately. Keep responses concise unless asked for details. Consider conversation history:

{history}
Context: {context}
User: {question}
Max:
""")
        
        # Threads
        self.capture_thread = threading.Thread(target=self.listen_audio, daemon=True)
        self.process_thread = threading.Thread(target=self.understand_input, daemon=True)
        self.tts_thread = threading.Thread(target=self.output_voice, daemon=True)
        self.preview_thread = threading.Thread(target=self.camera_preview, daemon=True)

    def generate_response(self, user_input):
        try:
            self.history.append({"speaker": "User", "message": user_input})
            formatted_history = "\n".join(f"{turn['speaker']}: {turn['message']}" for turn in self.history)

            vision_response = None
            if self.encoded_image and any(kw in user_input.lower() for kw in ["what", "describe", "how", "where", "is there", "see", "can you see", "do i", "am i"]):
                logging.info("Generating vision response...")
                temp_filename = "temp_image.jpg"
                self.current_image.save(temp_filename)
                caption_result = image_scanner.get_image_caption(self.current_image, temp_filename)
                if caption_result['success']:
                    logging.info(f"Raw caption: {caption_result['response']}")
                    vision_answer = image_scanner.answer_image_question(user_input)
                    vision_response = vision_answer.get('response', "I see something interesting!") if vision_answer['success'] else "I couldn’t process that clearly."
                    logging.info(f"Vision response: {vision_response}")
                else:
                    vision_response = "I couldn’t process the image clearly."
                    logging.error(f"Image caption failed: {caption_result['error']}")
                os.remove(temp_filename)

            logging.info("Streaming LLM response...")
            result = ""
            for chunk in (self.conversational_prompt | self.llm).stream({
                "context": f"Visual Context (describes the user): {vision_response}" if vision_response else "",
                "question": user_input,
                "history": formatted_history
            }):
                result += chunk
                logging.debug(f"LLM chunk: {chunk}")

            self.history.append({"speaker": "Max", "message": result})
            logging.info(f"Generated response: {result}")
            return result
        except Exception as e:
            logging.error(f"Error during live chat conversation: {e}")
            return None

    def listen_audio(self):
        def callback(indata, frames, time, status):
            if status:
                logging.error(f"Error in audio stream: {status}")
            if not self.is_processing and self.running:
                self.audio_queue.put(indata.copy())

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', callback=callback):
            logging.info("Live chat audio capture started.")
            while self.running:
                time.sleep(0.1)

    def understand_input(self):
        audio_buffer = []
        silence_frames = 0
        min_silence_frames = int(self.min_silence_duration / self.buffer_duration)
        moving_average_energy = []

        while self.running:
            try:
                if self.is_processing:
                    time.sleep(0.1)
                    continue
                chunk = self.audio_queue.get(timeout=1.0)
                audio_buffer.append(chunk)
                energy = np.sqrt(np.mean(chunk**2))
                moving_average_energy.append(energy)
                if len(moving_average_energy) > 10:
                    moving_average_energy.pop(0)
                avg_energy = np.mean(moving_average_energy)
                if avg_energy < self.silence_threshold:
                    silence_frames += 1
                else:
                    silence_frames = 0

                if silence_frames >= min_silence_frames and len(audio_buffer) > 0:
                    audio_data = np.concatenate(audio_buffer)
                    audio_buffer = []
                    silence_frames = 0
                    duration = len(audio_data) / self.sample_rate
                    logging.info(f"Processing audio with duration {time.strftime('%H:%M:%S', time.gmtime(duration))}")
                    if duration > 1.0:
                        self.is_processing = True
                        logging.info("Transcribing audio...")
                        segments, info = whisper_model.transcribe(
                            audio_data.flatten().astype(np.float32),
                            beam_size=5,
                        )
                        transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
                        logging.info(f"Transcription result: {transcript}")
                        if transcript:
                            logging.info(f"User: {transcript}")

                            if not self.conversation_active and "max" in transcript.lower():
                                logging.info("Wake word detected - capturing image...")
                                ret, frame = self.cap.read()
                                if ret:
                                    self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                    self.encoded_image = True
                                    temp_filename = "temp_image.jpg"
                                    self.current_image.save(temp_filename)
                                    caption_result = image_scanner.get_image_caption(self.current_image, temp_filename)
                                    os.remove(temp_filename)
                                    if caption_result['success']:
                                        caption = caption_result['response']
                                        logging.info(f"Initial caption: {caption}")
                                        self.conversation_active = True
                                        self.current_response = self.generate_response(
                                            f"Hi! I see you: {caption}. How can I assist you today?"
                                        )
                                    else:
                                        self.current_response = "I couldn’t capture a clear image. Could you try again?"
                                        logging.error(f"Image caption failed: {caption_result['error']}")

                            elif self.conversation_active:
                                if "moving on" in transcript.lower():
                                    logging.info("Capturing new context image...")
                                    ret, frame = self.cap.read()
                                    if ret:
                                        self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                        self.encoded_image = True
                                        self.history = []
                                        self.current_response = "Yes, let’s talk about something else then..."
                                    else:
                                        self.current_response = "But I still want to talk about this topic..."

                                elif "thank you" in transcript.lower():
                                    self.current_response = "You’re welcome! See you soon, have a nice day."
                                    self.tts_ready.set()
                                    while self.current_response is not None:
                                        time.sleep(0.1)
                                    self.stop()
                                    return

                                else:
                                    self.current_response = self.generate_response(transcript)

                            if self.current_response:
                                logging.info("Setting TTS ready...")
                                self.tts_ready.set()
                            else:
                                self.is_processing = False
                        else:
                            self.is_processing = False
                            logging.info("No transcript generated.")
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in understand_input: {e}")
                self.is_processing = False

    def output_voice(self):
        while self.running:
            self.tts_ready.wait()
            try:
                if self.current_response:
                    logging.info(f"[TTS] Processing: {self.current_response}")
                    generator = tts_pipeline(
                        self.current_response.strip(),
                        voice='af_sky',
                        speed=0.9,
                        split_pattern=r'\n+'
                    )
                    audio_played = False
                    for _, _, audio in generator:
                        if audio is None or audio.size == 0:
                            logging.warning("[TTS] Empty audio chunk detected")
                            continue
                        audio_int16 = (audio * 32767).astype(np.int16)
                        logging.info("[TTS] Playing audio response")
                        sd.play(audio_int16, samplerate=24000)
                        sd.wait()
                        audio_played = True
                        time.sleep(0.2)
                    if audio_played:
                        logging.info("TTS completed, clearing state.")
                        self.current_response = None
                        self.is_processing = False
                        self.tts_ready.clear()
                    else:
                        logging.warning("No audio played.")
                        self.is_processing = False
                        self.tts_ready.clear()
            except Exception as e:
                logging.error(f"[TTS ERROR] {str(e)}")
                self.current_response = None
                self.is_processing = False
                self.tts_ready.clear()

    def camera_preview(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def start(self):
        if not self.running:
            self.running = True
            self.capture_thread.start()
            self.process_thread.start()
            self.tts_thread.start()
            self.preview_thread.start()
            logging.info("Live chat started.")

    def stop(self):
        if self.running:
            self.running = False
            self.capture_thread.join(timeout=1)
            self.process_thread.join(timeout=1)
            self.tts_thread.join(timeout=1)
            self.preview_thread.join(timeout=1)
            self.cap.release()
            cv2.destroyAllWindows()
            logging.info("Live chat stopped.")

@app.route('/start_live_chat', methods=['POST'])
def start_live_chat():
    """Starts the live chat process."""
    global live_chat_instance, voice_chat_instance

    with live_video_lock:
        if live_chat_instance is not None and live_chat_instance.running:
            logging.warning("Live chat already running")
            return jsonify({'error': 'Live chat already running', "success":False}), 400

        if voice_chat_instance is not None and voice_chat_instance.running:
            with voice_chat_lock:
                try:
                    voice_chat_instance.stop()
                    voice_chat_instance = None
                    logging.info("Voice chat stopped successfully")
                except Exception as e:
                    logging.error(f"Failed to stop voice chat: {e}")

        try:
            live_chat_instance = LiveChatHandler()
            live_chat_instance.start()
            logging.info("Live chat started successfully")
            return jsonify({'message': 'Live chat started', "success":True}), 200
        except Exception as e:
            logging.error(f"Failed to start live chat: {e}")
            return jsonify({'error': f'Failed to start live chat: {e}', "success":False}), 500

@app.route('/stop_live_chat', methods=['POST'])
def stop_live_chat():
    """Stops the live chat process."""
    global live_chat_instance

    with live_video_lock:
        if live_chat_instance is None or not live_chat_instance.running:
            logging.warning("Live chat is not running")
            return jsonify({'error': 'Live chat not running', "success":False}), 400

        try:
            live_chat_instance.stop()
            live_chat_instance = None
            logging.info("Live chat stopped successfully")
            return jsonify({'message': 'Live chat stopped', "success":True}), 200
        except Exception as e:
            logging.error(f"Failed to stop live chat: {e}")
            return jsonify({'error': f'Failed to stop live chat: {e}', "success":False}), 500

@app.route('/start_voice_chat', methods=['POST'])
def start_voice_chat():
    """Starts the voice chat process."""
    global voice_chat_instance, live_chat_instance

    with voice_chat_lock:
        if voice_chat_instance is not None and voice_chat_instance.running:
            logging.warning("Voice chat already running")
            return jsonify({'error': 'Voice chat already running', "success": False}), 400

        if live_chat_instance is not None and live_chat_instance.running:
            with live_video_lock:
                try:
                    live_chat_instance.stop()
                    live_chat_instance = None
                    logging.info("Live chat stopped successfully")
                except Exception as e:
                    logging.error(f"Failed to stop live chat: {e}")

        try:
            voice_chat_instance = VoiceChatHandler()
            voice_chat_instance.start()
            logging.info("Voice chat started successfully")
            return jsonify({'message': 'Voice chat started', "success": True}), 200
        except Exception as e:
            logging.error(f"Failed to start voice chat: {e}")
            return jsonify({'error': f'Failed to start voice chat: {e}', "success": False}), 500

@app.route('/stop_voice_chat', methods=['POST'])
def stop_voice_chat():
    """Stops the voice chat process."""
    global voice_chat_instance

    with voice_chat_lock:
        if voice_chat_instance is None or not voice_chat_instance.running:
            logging.warning("Voice chat is not running")
            return jsonify({'error': 'Voice chat not running', "success": False}), 400

        try:
            voice_chat_instance.stop()
            voice_chat_instance = None
            logging.info("Voice chat stopped successfully")
            return jsonify({'message': 'Voice chat stopped', "success": True}), 200
        except Exception as e:
            logging.error(f"Failed to stop voice chat: {e}")
            return jsonify({'error': f'Failed to stop voice chat: {e}', "success": False}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
