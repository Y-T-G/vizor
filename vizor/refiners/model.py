from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for models."""

    def predict(self, image):
        """Perform prediction on the given image."""
        raise NotImplementedError

class PrimaryModel(BaseModel):
    """Abstract base class for primary models."""

    def to_tracks(self, inp):
        """Transforms the output of primary model to Tracks."""
        return inp

class SecondaryModel(BaseModel):
    """Abstract base class for secondary models."""

    def to_preds(preds):
        """Transforms the output of secondary model to Outs."""
        raise NotImplementedError

    def out_transform(self, out, *args, **kwarg):
        """Transforms the output of refiner to Outs."""
        return out


class VLMOutput:
    """Structued output schema for the VLM output."""

    name: str
    class_id: int


class TransformersModel(BaseModel):
    """Hugging-Face VLM model."""

    def __init__(
        self,
        model,
        task="causallm",
        parser=None,
        task_prompt="<CAPTION_TO_PHRASE_GROUNDING>",
    ):
        super().__init__()
        from transformers import AutoProcessor
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.task = task
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        if task == "vision2seq":
            from transformers import AutoModelForVision2Seq as AutoModel
            self.generate_kwargs = dict(max_new_tokens=500)
        elif task == "causallm":
            from transformers import AutoModelForCausalLM as AutoModel
            self.generate_kwargs = dict(max_new_tokens=1024, num_beams=3)
        try:
            self.model = AutoModel.from_pretrained(
                model,
                torch_dtype=self.dtype,
                _attn_implementation="flash_attention_2"
                if self.device == "cuda"
                else "eager",
                trust_remote_code=True,
            ).to(self.device)
        except ValueError:
            # Fallback to eager mode if flash_attention_2 is not available
            self.model = AutoModel.from_pretrained(
                model,
                torch_dtype=self.dtype,
                _attn_implementation="eager",
                trust_remote_code=True,
            ).to(self.device)

        if self.parser is not None:
            if task == "causallm":
                self.parser = self.processor.post_process_generation
                self.task_prompt = task_prompt
            else:
                self.parser = parser

    def parser(self, text):
        return text

    def predict(self, crop, message, **kwargs):
        if self.task == "vqa":
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
        elif self.task == "causallm":
            prompt = self.task_prompt + message
        inputs = self.processor(text=prompt, images=[crop], return_tensors="pt")
        inputs = inputs.to(self.device, self.dtype)

        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False if self.task == "causallm" else True
        )[0]

        parser_kwargs = (
            dict(image_size=crop.shape[:2][::-1], task=self.task_prompt) if self.task == "causallm" else {}
        )
        return self.parser(generated_text, **parser_kwargs)


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
