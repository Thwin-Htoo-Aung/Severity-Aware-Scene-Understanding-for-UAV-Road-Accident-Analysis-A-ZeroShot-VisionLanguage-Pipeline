from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import SamplingParams
import os

os.environ["HF_HOME"] = "$SCRATCH"

max_seq_length = 16384  # Must be this long for VLMs


def prepare_inputs_for_vllm(messages, processor, tokenizer):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    inputs = tokenizer(
        image_inputs,
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    return inputs


if __name__ == "__main__":
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
    #             },
    #             {"type": "text", "text": "这段视频有多长"},
    #         ],
    #     }
    # ]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/receipt.png",
                },
                {"type": "text", "text": "Read all the text in the image."},
            ],
        }
    ]

    # TODO: change to your own checkpoint path
    checkpoint_path = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=False,  # Enable vLLM fast inference
        gpu_memory_utilization=1.0,  # Reduce if out of memory
    )
    FastVisionModel.for_inference(model)  # Enable native 2x faster inference
    model.eval()

    inputs = [
        prepare_inputs_for_vllm(message, processor, tokenizer) for message in [messages]
    ][0]

    outputs = model.generate(
        **inputs, max_new_tokens=1024, use_cache=True, temperature=1.0, min_p=0.1
    )
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
