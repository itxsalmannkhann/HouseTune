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
