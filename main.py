import os
import base64
import mimetypes
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Load API key from .env
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "qwen/qwen3.8-27b"


# Convert local image into Base64
def encode_image(image_path):

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return encoded_image

