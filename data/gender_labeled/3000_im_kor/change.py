import pandas as pd

# 세 개의 CSV 파일 경로를 설정합니다.
csv_file_1 = 'gender_labeled_except_predict_3000_im_train.csv'
csv_file_2 = 'gender_labeled_except_predict_3000_im_test.csv'

# 각 CSV 파일을 데이터프레임으로 읽어옵니다.
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

# 매핑
df1['gender_real'] = df1['gender_real'].map({1: '남성', 2: '여성', 3: '남성 또는 여성'})
df2['gender_real'] = df2['gender_real'].map({1: '남성', 2: '여성', 3: '남성 또는 여성'})

# 변경된 데이터프레임을 다시 CSV 파일로 저장
output_file_path_1 = 'gender_labeled_except_predict_3000_im_kor_train.csv'  # 저장할 CSV 파일 경로를 입력하세요
output_file_path_2 = 'gender_labeled_except_predict_3000_im_kor_test.csv'  # 저장할 CSV 파일 경로를 입력하세요

df1.to_csv(output_file_path_1, index=False)
df2.to_csv(output_file_path_2, index=False)
