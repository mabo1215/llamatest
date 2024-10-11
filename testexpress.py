import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import keras_ocr

# Create a pipeline object
pipeline = keras_ocr.pipeline.Pipeline()


# 设置 Tesseract 的安装路径
pytesseract.pytesseract.tesseract_cmd = r'D:/Program Files/Tesseract-OCR/tesseract.exe'
tessdata_dir_config = '--tessdata-dir ' + f'F:/source/llamatest/tessdata_best'

image_path = "F:/source/llamatest/threshold_image.png"
image = keras_ocr.tools.read(image_path)

# Use the pipeline to perform OCR on the image
prediction_groups = pipeline.recognize([image])

# Output the results
for text, box in prediction_groups[0]:
    print(f'Recognized text: {text}')


# 加载要识别的图片
image = Image.open(image_path)

# 转换为灰度图像
image = image.convert('L')

# 提高对比度
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(2)

# 二值化处理
threshold = 128
image = image.point(lambda p: p > threshold and 255)

# 保存处理后的图像用于测试
image.save('processed_image.png')




# 定义 Tesseract OCR 的配置
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789+-*/=()÷;'
full_config = f'{custom_config} {tessdata_dir_config}'





print(f"使用配置: {full_config}")
# 使用 Tesseract OCR 进行识别，并指定 tessdata 路径和配置
text = pytesseract.image_to_string(image, config=full_config)

# 输出识别结果
print(f"识别出的数学公式: {text}")
