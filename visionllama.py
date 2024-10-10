import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# model_id = "meta-llama/Llama-3.2-1B-Instruct"
model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
	device_map="auto",
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(model_id)

model.tie_weights()
url = "https://www.ign.com.cn/sm/t/ign_cn/cover/s/silent-hil/silent-hill-2-remake_21e1.300.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Can you please describe this image in just one sentence?"}
    ]}
]

input_text = processor.apply_chat_template(
    messages, add_generation_prompt=True,
)
inputs = processor(
    image, input_text, return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=70)

print(processor.decode(output[0][inputs["input_ids"].shape[-1]:]))