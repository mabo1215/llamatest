import torch
import torchaudio
import torchaudio.transforms as transforms
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import sounddevice as sd
import numpy as np
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import pyttsx3



# 使用麦克风进行语音输入
def recognize_speech():
    model_path = "facebook/wav2vec2-base-960h"  # 使用的模型路径
    SAMPLE_RATE = 16000
    DURATION = 5  # 录音持续时间（秒）

    # 加载特征提取器和模型
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()  # 设置为评估模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.half()  # 使用半精度计算（如果你的GPU支持）

    print("speak plz...")

    # 实时录音
    audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # 等待录音完成

    # 播放录制的音频
    sd.play(audio_data, samplerate=SAMPLE_RATE)
    sd.wait()  # 等待播放完成

    waveform = torch.tensor(audio_data).T

    # 确保输入数据为正确的形状
    input_values = processor(audio_data.flatten(), sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
    input_values = input_values.half().to(device)  # 半精度和移动到设备

    # 检查输入长度
    if input_values.shape[1] < 10:  # 这里的10是卷积核的大小
        print("Input audio is too short, please try again.")
        return None

    # 进行推理
    with torch.no_grad():
        logits = model(input_values).logits

    # 获取预测的 ID
    predicted_ids = torch.argmax(logits, dim=-1)

    # 解码为文本
    transcription = processor.batch_decode(predicted_ids)
    print(f"your said: {transcription}")
    return transcription[0]

# 使用LLaMA处理输入并生成回复
def process_text(input_text,model_path):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {"role": "user", "content": input_text},
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0]["generated_text"][-1]
    print(f"助手回复: {response}")
    return response

# 使用语音合成引擎播报助手的回复
def speak_text(text,engine):
    engine.say(text)
    engine.runAndWait()

# 主程序流程
if __name__ == "__main__":
    # 初始化语音合成引擎
    engine = pyttsx3.init()
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

    while True:
        # 语音识别
        user_input = recognize_speech()
        if user_input:
            # 语言处理
            response = process_text(user_input, model_path)
            # 语音合成
            speak_text(response,engine)
