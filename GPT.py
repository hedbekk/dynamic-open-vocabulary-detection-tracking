# Standard library
import base64
import os

# Third-party
import cv2
from openai import OpenAI

class GPT:
    def __init__(self, prompt, model="gpt-4o"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Environment variable OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.prompt = prompt
        self.model = model

    def send_prompt(self, frame):
        """Send self.prompt and one BGR image frame to the model and return the text output."""
        if frame is None:
            raise ValueError("Frame is None")

        # Encode frame as a JPEG image in memory
        success, image = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Could not encode the image")

        # Convert the JPEG bytes to a Base64 string and format as a data URL for sending to the API
        image_base64 = base64.b64encode(image.tobytes()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{image_base64}"

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": self.prompt },
                            {
                                "type": "input_image",
                                "image_url": data_url,
                            },
                        ],
                    }
                ],
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI request failed: {e}")

        return response.output_text
