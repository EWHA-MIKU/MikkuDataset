# -*- coding: utf-8 -*-

import pandas as pd
import time

def get_random_state():
    # 현재 시간을 기반으로 시드를 생성
    return int(time.time())

# CSV 파일 읽기
#df = pd.read_csv('gender_labeled_except_predict.csv')
df = pd.read_csv('gender_labeled_except_predict_aug.csv')

# 4500
random_state = get_random_state()
data_1_4500 = df[df['gender_real'] == 1].sample(n=1500, random_state=random_state)
data_2_4500 = df[df['gender_real'] == 2].sample(n=1500, random_state=random_state)
data_3_4500= df[df['gender_real'] == 3].sample(n=1500, random_state=random_state)

combined_data = pd.concat([data_1_4500, data_2_4500, data_3_4500], ignore_index=True)
combined_data.to_csv('gender_labeled_except_predict_4500_raw.csv', index=False)

print("Combined data saved to 'gender_labeled_except_predict_4500_raw.csv'.")

# 결합된 데이터를 8:2 비율로 train:test 데이터로 나누기
train_data_1_4500 = data_1_4500.sample(n=1200, random_state=get_random_state())
test_data_1_4500 = data_1_4500.drop(train_data_1_4500.index)

train_data_2_4500 = data_2_4500.sample(n=1200, random_state=get_random_state())
test_data_2_4500 = data_2_4500.drop(train_data_2_4500.index)

train_data_3_4500 = data_3_4500.sample(n=1200, random_state=get_random_state())
test_data_3_4500 = data_3_4500.drop(train_data_3_4500.index)

train_data_4500 = pd.concat([train_data_1_4500, train_data_2_4500, train_data_3_4500], ignore_index=True)
test_data_4500 = pd.concat([test_data_1_4500, test_data_2_4500, test_data_3_4500], ignore_index=True)

train_data_4500.to_csv('gender_labeled_except_predict_4500_train.csv', index=False)
test_data_4500.to_csv('gender_labeled_except_predict_4500_test.csv', index=False)

# 9000_im
random_state = get_random_state()
data_1_4500_im = df[df['gender_real'] == 1].sample(n=1200, random_state=random_state)
data_2_4500_im = df[df['gender_real'] == 2].sample(n=1200, random_state=random_state)
data_3_4500_im= df[df['gender_real'] == 3].sample(n=2100, random_state=random_state)

combined_data = pd.concat([data_1_4500_im, data_2_4500_im, data_3_4500_im], ignore_index=True)
combined_data.to_csv('gender_labeled_except_predict_4500_im_raw.csv', index=False)

print("Combined data saved to 'gender_labeled_except_predict_4500_im_raw.csv'.")

# 결합된 데이터를 8:2 비율로 train:test 데이터로 나누기
train_data_1_4500_im = data_1_4500_im.sample(n=960, random_state=get_random_state())
test_data_1_4500_im = data_1_4500_im.drop(train_data_1_4500_im.index)

train_data_2_4500_im = data_2_4500_im.sample(n=960, random_state=get_random_state())
test_data_2_4500_im = data_2_4500_im.drop(train_data_2_4500_im.index)

train_data_3_4500_im = data_3_4500_im.sample(n=1680, random_state=get_random_state())
test_data_3_4500_im = data_3_4500_im.drop(train_data_3_4500_im.index)

train_data_4500_im = pd.concat([train_data_1_4500_im, train_data_2_4500_im, train_data_3_4500_im], ignore_index=True)
test_data_4500_im = pd.concat([test_data_1_4500_im, test_data_2_4500_im, test_data_3_4500_im], ignore_index=True)

train_data_4500_im.to_csv('gender_labeled_except_predict_4500_im_train.csv', index=False)
test_data_4500_im.to_csv('gender_labeled_except_predict_4500_im_test.csv', index=False)