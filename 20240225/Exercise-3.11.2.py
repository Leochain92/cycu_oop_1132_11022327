def print_right(text):
    total_width = 40
    leading_spaces = total_width - len(text)
    print(" " * leading_spaces + text)

# 測試 print_right 函數
print_right("Monty")
print_right("Python's")
print_right("Flying Circus")