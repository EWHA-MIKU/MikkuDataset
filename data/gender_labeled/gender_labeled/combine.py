import pandas as pd

# 세 개의 CSV 파일 경로를 설정합니다.
csv_file_1 = '/mnt/aix21006/data/gender_labeled/gender_labeled/gender_labeled_except_predict_1.csv'
csv_file_2 = '/mnt/aix21006/data/gender_labeled/gender_labeled/gender_labeled_except_predict_2.csv'
csv_file_3 = '/mnt/aix21006/data/gender_labeled/gender_labeled/gender_labeled_except_predict_3.csv'

# 각 CSV 파일을 데이터프레임으로 읽어옵니다.
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)
df3 = pd.read_csv(csv_file_3)

# 데이터프레임을 순서대로 연결합니다.
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# 결과를 새로운 CSV 파일로 저장합니다.
combined_df.to_csv('gender_labeled_except_predict_0.csv', encoding='utf-8',index=False)

print("CSV 파일이 성공적으로 결합되었습니다.")
