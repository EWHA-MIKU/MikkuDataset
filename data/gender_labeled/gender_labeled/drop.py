import pandas as pd

# 원본 CSV 파일 경로를 설정합니다.
input_csv_file = '/mnt/aix21006/data/gender_labeled/gender_labeled/gender_labeled_except_predict_0.csv'

# CSV 파일을 데이터프레임으로 읽어옵니다.
df = pd.read_csv(input_csv_file)

# gender_real 컬럼 값이 0인 데이터를 제거합니다.
filtered_df = df[df['gender_real'] != 0]

# 결과를 새로운 CSV 파일로 저장합니다.
output_csv_file = 'gender_labeled_except_predict.csv'
filtered_df.to_csv(output_csv_file, encoding='utf-8', index=False, columns=['ko', 'figure', 'gender_real'])

print(f"gender_real 컬럼 값이 0인 데이터를 제거한 결과가 {output_csv_file}로 저장되었습니다.")
