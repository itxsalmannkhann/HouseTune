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


# Send image to Groq and get description
def generate_description(image_path):

    image_path = Path(image_path)

    # Detect image format
    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type not in ["image/png", "image/jpeg", "image/webp"]:
        raise ValueError("Unsupported image format")

    # Convert image to Base64
    base64_image = encode_image(image_path)

    # Create image URL
    image_url = f"data:{mime_type};base64,{base64_image}"

    # Send request to Groq
    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": """
                You are an expert architectural floor-plan analyst.
                Carefully examine the supplied floor-plan image.
                Describe the visible rooms, their approximate
                positions, adjacency, doors, windows and layout.

                Do not invent information that cannot be identified.
                If something is unclear, explicitly say so.
                Do not assume dimensions or measurements that
                are not visible.
                """
            },

            {
                "role": "user",

                "content": [
                    {
                        "type": "text",
                        "text": """
                        Analyze this residential floor plan.

                        Describe:
                        1. The rooms you can identify.
                        2. Their approximate positions.
                        3. Which rooms are adjacent.
                        4. The overall arrangement of the house.

                        Return a clear natural-language description.
                        """
                    },

                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],

        temperature=0.2,
        max_completion_tokens=1500
    )

    description = response.choices[0].message.content

    return description


