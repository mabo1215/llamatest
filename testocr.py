from sympy import symbols, Eq, solve, sympify
from fractions import Fraction

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

# 示例表达式测试
expression = "98 ➗ 10 = "  # 测试除法表达式
try:
    result = solve_expression(expression)
    print(f"未知数的值为: {result}")
except ValueError as e:
    print(f"错误: {e}")
