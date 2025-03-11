# 1. 正確的變數賦值
n = 17
print(n)  # 輸出: 17

# 2. 錯誤的變數賦值（這行會導致語法錯誤）
17 = n  # SyntaxError: cannot assign to literal

# 3. 連續賦值
x = y = 1
print(x, y)  # 輸出: 1 1

# 4. 語句結尾加上分號（合法，但不推薦）
a = 10;
print(a);  # 輸出: 10

# 5. 語句結尾加上句點（會導致語法錯誤）
b = 5.
print(b.)  # SyntaxError: invalid syntax

# 6. 錯誤的模組名稱（導致 ImportError）
import maath  # ModuleNotFoundError: No module named 'maath'