from PIL import Image
import pytesseract
import cv2
import re
import numpy as np
import pyautogui
from sympy import sympify
import time
from drawnumber import draw_character


pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'


def ch_filter(expression):
	clean_expression = re.sub(r'[=@#$&!\r\n]', '', expression)
	return clean_expression


def recog_expression():
	# 读取图像并进行预处理
	# image_path = 'F:/work/datasets/test/54.png'
	# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	x, y, width, height = 534, 225, 105, 80  # 修改为你需要的区域
	screenshot = pyautogui.screenshot(region=(x, y, width, height))
	# screenshot.save("screenshot.png")
	screenshot_np = np.array(screenshot)
	screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
	gray_image = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
	_, thresh_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
	pil_image = Image.fromarray(thresh_image)
	expression = pytesseract.image_to_string(pil_image, config='--psm 6')
	expression = ch_filter(expression)

	if not expression or expression.strip() == '':
		print("No expression detected. Exiting the program.")
		return 0

	result = sympify(expression)
	print(f"The result of the expression '{expression}' is: {result}")

	start_x, start_y = 630, 630  # 固定区域的起始坐标


	result_string = str(result)
	pyautogui.moveTo(start_x, start_y)
	# 开始绘制字符
	for char in result_string:
		draw_character(char, start_x, start_y)
		time.sleep(0.01)

	# 结束绘制
	return 1


if __name__ == '__main__':
	while True:
		res = recog_expression()
		if res == 0:
			break
		time.sleep(0.3)

