import pyautogui
import time

# 定义绘制数字的路径
def draw_character(char, start_x, start_y):
    if char == '0':
        print(f'绘制字符: 0')
        draw_zero(start_x, start_y)
    elif char == '1':
        print(f'绘制字符: 1')
        draw_one(start_x, start_y)
    elif char == '2':
        print(f'绘制字符: 2')
        draw_two(start_x, start_y)
    elif char == '3':
        print(f'绘制字符: 3')
        draw_three(start_x, start_y)
    elif char == '4':
        print(f'绘制字符: 4')
        draw_four(start_x, start_y)
    elif char == '5':
        print(f'绘制字符: 5')
        draw_five(start_x, start_y)
    elif char == '6':
        print(f'绘制字符: 6')
        draw_six(start_x, start_y)
    elif char == '7':
        print(f'绘制字符: 7')
        draw_seven(start_x, start_y)
    elif char == '8':
        print(f'绘制字符: 8')
        draw_eight(start_x, start_y)
    elif char == '9':
        print(f'绘制字符: 9')
        draw_nine(start_x, start_y)
    elif char == '.':
        print(f'绘制字符: .')
        draw_dot(start_x, start_y)
    elif char == '-':
        print(f'绘制字符: -')
        draw_neg(start_x, start_y)
    elif char == '/':
        print(f'绘制字符: /')
        draw_devide(start_x, start_y)
    else:
        print(f"Character '{char}' not recognized.")

def draw_devide(start_x, start_y):
    pyautogui.moveTo(start_x + 20, start_y-20)  # Top horizontal line
    pyautogui.mouseDown()
    pyautogui.moveTo(start_x - 20, start_y + 20)  # Top horizontal line
    pyautogui.mouseUp()

def draw_neg(start_x, start_y):
    pyautogui.moveTo(start_x-10, start_y)  # Top horizontal line
    pyautogui.mouseDown()
    pyautogui.moveTo(start_x+10, start_y)  # Top horizontal line
    pyautogui.mouseUp()
def draw_dot(start_x, start_y):
    pyautogui.moveTo(start_x, start_y+20)  # Top horizontal line
    pyautogui.mouseDown()
    pyautogui.moveTo(start_x, start_y + 21)  # Top horizontal line
    pyautogui.mouseUp()

def draw_zero(start_x, start_y):
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    pyautogui.moveTo(start_x + 20, start_y)  # Top horizontal line
    pyautogui.moveTo(start_x + 20, start_y + 30)  # Right vertical line
    pyautogui.moveTo(start_x, start_y + 30)  # Bottom horizontal line
    pyautogui.moveTo(start_x, start_y)  # Left vertical line
    pyautogui.mouseUp()

def draw_one(start_x, start_y):
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()  # Press down the mouse button
    pyautogui.moveTo(start_x-5 , start_y + 50)  # Increase the length to 50 pixels (or adjust as needed)
    pyautogui.moveTo(start_x, start_y + 50)  # Increase the length to 50 pixels (or adjust as needed)
    pyautogui.mouseUp()  # Release the mouse button

def draw_two(start_x, start_y):
    pyautogui.moveTo(start_x, start_y)  # Start position
    pyautogui.mouseDown()

    # Draw the top horizontal line
    pyautogui.moveTo(start_x + 20, start_y+10)  # Move right
    # Draw the diagonal line to the bottom right
    pyautogui.moveTo(start_x + 20, start_y + 15)  # Move down to the middle
    # Draw the bottom curve
    pyautogui.moveTo(start_x-5, start_y + 30)  # Move down to the bottom
    pyautogui.moveTo(start_x + 20, start_y + 30)  # Move right to finish the bottom
    pyautogui.mouseUp()

def draw_three(start_x, start_y):
    pyautogui.moveTo(start_x, start_y)  # Move to the starting position
    pyautogui.mouseDown()  # Press down the mouse button

    # Draw the top horizontal line
    pyautogui.moveTo(start_x + 20, start_y)  # Move right

    # Draw the right upper arc
    pyautogui.moveTo(start_x + 20, start_y + 15)  # Move down to the middle

    # Draw the middle horizontal line
    pyautogui.moveTo(start_x, start_y + 15)  # Move left to middle

    # Draw the right lower arc
    pyautogui.moveTo(start_x + 20, start_y + 15)  # Move right to the end of middle line
    pyautogui.moveTo(start_x + 20, start_y + 30)  # Move down to bottom
    pyautogui.moveTo(start_x, start_y + 30)  # Move left to bottom left

    # Release the mouse button
    pyautogui.mouseUp()  # Release the mouse button

def draw_four(start_x, start_y):
    pyautogui.moveTo(start_x, start_y)  # Move to the starting position
    pyautogui.mouseDown()  # Press down the mouse button

    # Draw the left vertical line
    pyautogui.moveTo(start_x, start_y - 40)  # Move down to the bottom of 4

    # Draw the top horizontal line
    pyautogui.moveTo(start_x - 30, start_y - 20)  # Move right to the top

    # Draw the right vertical line
    pyautogui.moveTo(start_x + 30, start_y - 20)  # Move down to the middle

    pyautogui.mouseUp()  # Release the mouse button

    # Draw the bottom horizontal line
    pyautogui.moveTo(start_x, start_y - 40)  # Move back to the left

    pyautogui.mouseDown()  # Press down the mouse button
    pyautogui.moveTo(start_x, start_y + 10)  # Move back to the left
    pyautogui.mouseUp()  # Release the mouse button


def draw_five(start_x, start_y):
    # Move to the starting position
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()

    pyautogui.moveTo(start_x+20, start_y)
    pyautogui.mouseUp()

    pyautogui.moveTo(start_x, start_y-5)
    pyautogui.mouseDown()
    pyautogui.moveTo(start_x+15, start_y + 12)
    pyautogui.moveTo(start_x + 15, start_y + 17)

    # Draw the middle horizontal line
    pyautogui.moveTo(start_x - 10, start_y + 25)  # Move right for the middle

    # Return to the starting position to complete the shape
    pyautogui.mouseUp()

def draw_six(start_x, start_y):
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    pyautogui.moveTo(start_x + 20, start_y)  # Top horizontal line
    pyautogui.moveTo(start_x + 20, start_y + 15)  # Right vertical line
    pyautogui.moveTo(start_x, start_y + 15)  # Middle horizontal line
    pyautogui.moveTo(start_x+10, start_y - 20)  # Bottom horizontal line
    pyautogui.mouseUp()

def draw_seven(start_x, start_y):
    pyautogui.moveTo(start_x-5, start_y)
    pyautogui.mouseDown()
    pyautogui.moveTo(start_x + 20, start_y)  # Top horizontal line
    pyautogui.moveTo(start_x - 20, start_y + 50)  # Vertical line
    pyautogui.mouseUp()


def draw_eight(start_x, start_y):
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()

    pyautogui.moveTo(start_x - 10, start_y + 10)  # Top center


    pyautogui.moveTo(start_x, start_y + 15)  # Top center
    pyautogui.moveTo(start_x + 20, start_y+20)  # Right top

    pyautogui.moveTo(start_x, start_y + 25)  # Bottom center

    pyautogui.moveTo(start_x, start_y)  # Left top
    pyautogui.mouseUp()

def draw_nine(start_x, start_y):
    pyautogui.moveTo(start_x, start_y)          # Move to starting position
    pyautogui.mouseDown()                        # Start drawing

    # Draw the top horizontal line
    pyautogui.moveTo(start_x -10, start_y-10)     # Top horizontal line

    # Draw the right vertical curve
    pyautogui.moveTo(start_x - 20, start_y)  # Move down
    pyautogui.moveTo(start_x - 10 , start_y + 10)     # Move left for middle horizontal line
    pyautogui.moveTo(start_x, start_y) # Move back to the right

    # Draw the bottom vertical line
    pyautogui.moveTo(start_x, start_y + 30) # Bottom vertical line
    pyautogui.mouseUp()                            # Stop drawing
