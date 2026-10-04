import os

os.environ["HF_HOME"] = "$SCRATCH"

from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

model_id = "Qwen/Qwen3-VL-8B-Instruct"

# default: Load the model on the available device(s)
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    model_id, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

print("Model and processor downloaded successfully.")