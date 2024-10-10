import pytesseract
from PIL import Image

# Tesseract 的安装路径
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

# 定义 tessdata 目录路径
tessdata_dir_config = r'--tessdata-dir "D:/Program Files/Tesseract-OCR/tessdata_best"'
# tessdata_dir_config = r'--tessdata-dir "D:/Program Files/Tesseract-OCR/tessdata"'

image_path = "F:/source/llamatest/threshold_image.png"
# 加载要识别的图片
image = Image.open(image_path)

# 使用 Tesseract OCR 进行识别，并指定 tessdata 路径和配置
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/=()÷;'
text = pytesseract.image_to_string(image, config=custom_config + ' ' + tessdata_dir_config)

# 输出识别结果
print(f"识别出的数学公式: {text}")
