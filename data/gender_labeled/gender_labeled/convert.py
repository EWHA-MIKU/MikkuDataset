import pandas as pd

# 엑셀 파일 경로를 설정합니다.
excel_file = '/mnt/aix21006/data/gender_labeled/gender_labeled/gender_labeled_except_predict_3_e.xlsx'

# 엑셀 파일을 데이터프레임으로 읽어옵니다.
df = pd.read_excel(excel_file)

# 데이터프레임을 CSV 파일로 저장합니다.
csv_file = 'gender_labeled_except_predict_3.csv'
df.to_csv(csv_file, encoding='utf-8', index=False)

print(f"엑셀 파일이 성공적으로 {csv_file}로 변환되었습니다.")
