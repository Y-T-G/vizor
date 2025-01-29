from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for models."""

    @abstractmethod
    def predict(self, image):
        """Perform prediction on the given image."""
        pass


class VLMOutput:
    """Structued output schema for the VLM output."""

    name: str
    class_id: int


class TransformersModel(BaseModel):
    """Hugging-Face VLM model."""

    def __init__(self, model):
        super().__init__()
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2"
            if self.device == "cuda"
            else "eager",
        ).to(self.device)

    def predict(self, crop, message, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Can you describe this image?"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=[crop], return_tensors="pt")
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_texts[0]


class VLMModel(BaseModel):
    """Vision-Language Model supporting multiple APIs."""

    def __init__(self, model, api_type, credentials, message=None):
        self.model = model
        self.api_type = api_type
        self.credentials = credentials
        self.client = self.initialize_client(api_type, credentials)
        self.message = message

    def initialize_client(self, api_type, api_key, **kwargs):
        if api_type == "openai":
            from openai import OpenAI

            return OpenAI(api_key=api_key, **kwargs)
        elif api_type == "groq":
            from groq import Groq

            return Groq(api_key=api_key, **kwargs)
        elif api_type == "api2":
            # Initialize API2 client
            pass
        else:
            raise ValueError("Unsupported API type")

    def encode(self, crop):
        import base64
        import cv2

        _, buffer = cv2.imencode(".jpg", crop)
        encoded_crop = base64.b64encode(buffer).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{encoded_crop}"
        return image_url

    def predict(self, crop, message=None, **kwargs):
        if self.api_type in ("openai", "groq"):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": self.encode(crop)},
                            },
                            {"type": "text", "text": message or self.message},
                        ],
                    }
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            return response.choices[0].message
        else:
            raise NotImplementedError
