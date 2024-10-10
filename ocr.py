from PIL import Image
import pytesseract
import cv2
import re
import numpy as np
import pyautogui
from sympy import symbols, Eq, solve, sympify
from fractions import Fraction
import time
from drawnumber import draw_character
# from pix2text import Pix2Text, merge_line_texts

pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

# tessdata_dir_config = r'--tessdata-dir "D:\\Program Files\\Tesseract-OCR\\tessdata_best"'

def ch_filter(expression):
    clean_expression = re.sub(r'[@#$&!\r\n]', '', expression)
    return clean_expression


def convert_fraction_to_decimal(expression):
    """
    将表达式中的所有分数转换为小数
    """
    parts = expression.split()
    for i, part in enumerate(parts):
        if '/' in part:
            try:
                parts[i] = str(float(Fraction(part)))
            except ValueError:
                continue
    return ' '.join(parts)

def clean_expression(expression):
    """
    去除表达式中的多余空格
    """
    return expression.replace(' ', '')

def solve_expression(expression):
    """
    解包含未知数 ? 的表达式，支持小数和分数的加减乘除运算
    """
    # 检查表达式是否包含等号
    if '=' not in expression:
        expression += '=unknown'

    # 将表达式拆分为左右两部分
    left_side, right_side = expression.split('=')

    # 如果右侧为空，则自动替换为 '?'，用 SymPy 可识别的符号替代空白部分
    if not right_side.strip():
        right_side = 'unknown'  # 代替未知数的符号

    # 清理表达式中的空格并转换分数
    left_side = convert_fraction_to_decimal(clean_expression(left_side))
    right_side = convert_fraction_to_decimal(clean_expression(right_side))

    # 定义未知数符号
    unknown = symbols('unknown')

    # 替换未知数符号 '?' 为 SymPy 的符号 'unknown'
    left_side = left_side.replace('?', 'unknown')
    right_side = right_side.replace('?', 'unknown')

    left_side = left_side.replace('✖', '*')
    left_side = left_side.replace('×', '*')
    left_side = left_side.replace('x', '*')
    left_side = left_side.replace('➗', '/')  # 替换除法符号
    right_side = right_side.replace('✖', '*')
    right_side = right_side.replace('×', '*')
    left_side = left_side.replace('x', '*')
    right_side = right_side.replace('➗', '/')  # 替换除法符号

    print(f"Left expression'{left_side}'", f"Right expression: '{right_side}'")
    try:
        # 使用 sympify 将字符串转换为 SymPy 表达式
        left_expr = sympify(left_side)
        print(f"Left expression after sympify: {left_expr}")  # 打印左侧表达式
        right_expr = sympify(right_side)
        print(f"Right expression after sympify: {right_expr}")  # 打印右侧表达式
    except Exception as e:
        raise ValueError(f"无法解析表达式，请检查输入格式: {e}")

    equation = Eq(left_expr, right_expr)

    # 求解未知数
    solution = solve(equation, unknown)
    return solution

def recog_expression():
    # 读取图像并进行预处理
    # image_path = 'F:/work/datasets/test/54.png'
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # x, y, width, height = 540, 245, 105, 80  # 修改为你需要的区域
    x, y, width, height = 570, 260, 230, 80  # 修改为你需要的区域
    start_x, start_y = 350, 650  # 固定区域的起始坐标

    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    screenshot.save("screenshot.png")
    screenshot_np = np.array(screenshot)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    pil_image = Image.fromarray(thresh_image)

    pil_image.save("threshold_image.png")

    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/=()÷;'
    # expression = pytesseract.image_to_string(pil_image, config=custom_config + ' ' + tessdata_dir_config)
    expression = pytesseract.image_to_string(pil_image, config='--psm 6')

    # p2t = Pix2Text()
    # expression = p2t.recognize(pil_image, return_text=True, save_analysis_res='en1-out.jpg')

    expression = ch_filter(expression)

    if not expression or expression.strip() == '':
        print("No expression detected. Exiting the program.")
        return 0

    expression = str(expression)
    print(f"The expression '{expression}'")
    result = solve_expression(expression)
    print(f"未知数的值为: {result}")

    # result = sympify(expression)


    result_string = str(result)
    print(f"The result of the expression '{expression}' is: {result_string}")
    pyautogui.moveTo(start_x, start_y)
    for char in result_string:
        draw_character(char, start_x, start_y)
        start_x += 70
        time.sleep(0.01)

    # 结束绘制
    return 1


if __name__ == '__main__':
    while True:
        res = recog_expression()
        if res == 0:
            break

        time.sleep(0.28)

