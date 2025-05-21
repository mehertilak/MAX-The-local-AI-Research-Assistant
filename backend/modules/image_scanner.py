import logging
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageDraw
import os
import shutil
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the moondream2 model and tokenizer (Load only once at module level)
try:
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        device_map={"": "cuda"}
    )
    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2025-01-09", trust_remote_code=True)
    logging.info("Moondream2 model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Moondream2 model: {e}")
    model = None
    tokenizer = None

# Folder to save detected/pointed images
output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'detected_images')
os.makedirs(output_folder, exist_ok=True)

# Module-level variable to store the encoded image and image path
_stored_encoded_image = None
_last_uploaded_image_path = None
_temp_dir = None

# Helper Functions

def _encode_and_store_image(image):
    """Encodes the image and stores the encoded representation."""
    global _stored_encoded_image
    try:
        if model is None:
            raise Exception("Model is not loaded")
        _stored_encoded_image = model.encode_image(image)
        logging.info("Image encoded and stored successfully.")
        return True
    except Exception as e:
        logging.error(f"Error encoding and storing image: {e}")
        _stored_encoded_image = None
        return False

def set_temp_dir(temp_dir):
    """Stores the TemporaryDirectory instance."""
    global _temp_dir
    _temp_dir = temp_dir

def get_image_caption(image, image_path):
    """Generate a short caption for the image."""
    global _stored_encoded_image, _last_uploaded_image_path
    try:
        if not _encode_and_store_image(image):
            return {"error": "Image encoding failed", "success": False}

        if _stored_encoded_image is None or tokenizer is None:
            raise Exception("Encoded image or tokenizer not available")

        _last_uploaded_image_path = image_path
        caption = model.answer_question(_stored_encoded_image, "Describe this image in one sentence.", tokenizer)
        return {"response": caption, "success":True}
    except Exception as e:
        logging.error(f"Error generating image caption: {e}")
        return {"error": f"Error generating image caption: {e}", "success": False}

def answer_image_question(question):
    """Answer a question about the image using the stored encoded image."""
    global _stored_encoded_image
    try:
        if _stored_encoded_image is None or tokenizer is None:
            raise Exception("Encoded image or tokenizer not available. Please upload an image first.")

        answer = model.answer_question(_stored_encoded_image, question, tokenizer)
        return {"response": answer, "success":True}
    except Exception as e:
         logging.error(f"Error answering image question: {e}")
         return {"error": f"Error answering image question: {e}", "success": False}

def detect_objects(object_to_detect):
    """Detect objects in the image."""
    global _stored_encoded_image, _last_uploaded_image_path
    try:
        if _stored_encoded_image is None or tokenizer is None:
            raise Exception("Encoded image or tokenizer not available. Please upload an image first.")

        if not _last_uploaded_image_path or not os.path.exists(_last_uploaded_image_path):
            raise Exception("Image file not found on server path: " + str(_last_uploaded_image_path))

        original_image = Image.open(_last_uploaded_image_path)
        detect_result = model.detect(original_image, object_to_detect)

        if not detect_result["objects"]:
            return {"response": f"No '{object_to_detect}' found in the image.", "success": False}

        img_width, img_height = original_image.size
        annotated_image = original_image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for detection in detect_result["objects"]:
            x_min = int(detection["x_min"] * img_width)
            y_min = int(detection["y_min"] * img_height)
            x_max = int(detection["x_max"] * img_width)
            y_max = int(detection["y_max"] * img_height)
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)

        image_name = os.path.basename(_last_uploaded_image_path)
        output_path = os.path.join(output_folder, f"detected_{object_to_detect}_{image_name}")
        annotated_image.save(output_path)
        return {"image_path": f"/detected_images/{os.path.basename(output_path)}", "success":True}

    except Exception as e:
        logging.error(f"Error during object detection: {e}")
        return {"error": f"Error during object detection: {e}", "success": False}

def point_at_object(query):
    """Point at objects in the image."""
    global _stored_encoded_image, _last_uploaded_image_path
    try:
        if _stored_encoded_image is None or tokenizer is None:
            raise Exception("Encoded image or tokenizer not available. Please upload an image first.")

        if not _last_uploaded_image_path or not os.path.exists(_last_uploaded_image_path):
            raise Exception("Image file not found on server path: " + str(_last_uploaded_image_path))

        original_image = Image.open(_last_uploaded_image_path)
        points = model.point(original_image, query)

        if not points["points"]:
              return {"response": f"No '{query}' found in the image.", "success": False}

        img_width, img_height = original_image.size
        annotated_image = original_image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for point in points["points"]:
            x = int(point["x"] * img_width)
            y = int(point["y"] * img_height)
            marker_radius = 10
            draw.ellipse([(x - marker_radius, y - marker_radius), (x + marker_radius, y + marker_radius)], fill="blue", outline="blue")

        image_name = os.path.basename(_last_uploaded_image_path)
        output_path = os.path.join(output_folder, f"pointed_{query}_{image_name}")
        annotated_image.save(output_path)
        return {"image_path": f"/detected_images/{os.path.basename(output_path)}", "success":True}
    except Exception as e:
        logging.error(f"Error during object pointing: {e}")
        return {"error": f"Error during object pointing: {e}", "success": False}

def cleanup_image_resources():
    """Cleans up the image processing resources."""
    global _stored_encoded_image, _last_uploaded_image_path, _temp_dir
    try:
        _stored_encoded_image = None
        _last_uploaded_image_path = None
        if _temp_dir:
            _temp_dir.cleanup()
            _temp_dir = None
        logging.info("Image chat cleanup completed")
        return {"message": "Image resources cleanup completed successfully", "success": True}
    except Exception as e:
        logging.error(f"Error during image cleanup: {e}")
        return {"error": f"Error during image cleanup: {e}", "success": False}
