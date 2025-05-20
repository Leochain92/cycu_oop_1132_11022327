import pandas as pd

# Load CSV data
df = pd.read_csv('20250520/midterm_scores.csv')

subjects = ['Chinese', 'English', 'Math', 'History', 'Geography', 'Physics', 'Chemistry']

total_subjects = len(subjects)
# 計算超過一半不及格的科目數量門檻
failing_threshold = total_subjects / 2

# 儲存結果的列表
failing_students = []

for idx, row in df.iterrows():
    failed_subjects_count = 0
    failed_subjects_list = []

    for subj in subjects:
        if row[subj] < 60:
            failed_subjects_count += 1
            failed_subjects_list.append(subj)

    if failed_subjects_count > failing_threshold:
        failing_students.append({
            'Name': row['Name'],
            'StudentID': row['StudentID'],
            'Failed_Count': failed_subjects_count,
            'Failed_Subjects': ', '.join(failed_subjects_list)
        })

# 將結果轉換為 DataFrame
failing_students_df = pd.DataFrame(failing_students)

# 列印結果
print("超過一半科目不及格的學生：")
print(failing_students_df)

# 將結果輸出為 CSV 檔案
output_path = '20250520/failing_students.csv'
failing_students_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"結果已輸出至 {output_path}")