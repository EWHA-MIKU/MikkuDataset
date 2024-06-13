import pandas as pd
import time

def get_random_state():
    # 현재 시간을 기반으로 시드를 생성
    return int(time.time())

# CSV 파일 읽기
#df = pd.read_csv('gender_labeled_except_predict.csv')
df = pd.read_csv('gender_labeled_except_predict_aug.csv')

# 9000
random_state = get_random_state()
data_1_9000 = df[df['gender_real'] == 1].sample(n=3000, random_state=random_state)
data_2_9000 = df[df['gender_real'] == 2].sample(n=3000, random_state=random_state)
data_3_9000= df[df['gender_real'] == 3].sample(n=3000, random_state=random_state)

combined_data = pd.concat([data_1_9000, data_2_9000, data_3_9000], ignore_index=True)
combined_data.to_csv('gender_labeled_except_predict_9000_raw.csv', index=False)

print("Combined data saved to 'gender_labeled_except_predict_9000_raw.csv'.")

# 결합된 데이터를 8:2 비율로 train:test 데이터로 나누기
train_data_1_9000 = data_1_9000.sample(n=2400, random_state=get_random_state())
test_data_1_9000 = data_1_9000.drop(train_data_1_9000.index)

train_data_2_9000 = data_2_9000.sample(n=2400, random_state=get_random_state())
test_data_2_9000 = data_2_9000.drop(train_data_2_9000.index)

train_data_3_9000 = data_3_9000.sample(n=2400, random_state=get_random_state())
test_data_3_9000 = data_3_9000.drop(train_data_3_9000.index)

train_data_9000 = pd.concat([train_data_1_9000, train_data_2_9000, train_data_3_9000], ignore_index=True)
test_data_9000 = pd.concat([test_data_1_9000, test_data_2_9000, test_data_3_9000], ignore_index=True)

train_data_9000.to_csv('gender_labeled_except_predict_9000_train.csv', index=False)
test_data_9000.to_csv('gender_labeled_except_predict_9000_test.csv', index=False)

# 9000_im
random_state = get_random_state()
data_1_9000_im = df[df['gender_real'] == 1].sample(n=2000, random_state=random_state)
data_2_9000_im = df[df['gender_real'] == 2].sample(n=2000, random_state=random_state)
data_3_9000_im= df[df['gender_real'] == 3].sample(n=5000, random_state=random_state)

combined_data = pd.concat([data_1_9000_im, data_2_9000_im, data_3_9000_im], ignore_index=True)
combined_data.to_csv('gender_labeled_except_predict_9000_im_raw.csv', index=False)

print("Combined data saved to 'gender_labeled_except_predict_9000_im_raw.csv'.")

# 결합된 데이터를 8:2 비율로 train:test 데이터로 나누기
train_data_1_9000_im = data_1_9000_im.sample(n=1600, random_state=get_random_state())
test_data_1_9000_im = data_1_9000_im.drop(train_data_1_9000_im.index)

train_data_2_9000_im = data_2_9000_im.sample(n=1600, random_state=get_random_state())
test_data_2_9000_im = data_2_9000_im.drop(train_data_2_9000_im.index)

train_data_3_9000_im = data_3_9000_im.sample(n=4000, random_state=get_random_state())
test_data_3_9000_im = data_3_9000_im.drop(train_data_3_9000_im.index)

train_data_9000_im = pd.concat([train_data_1_9000_im, train_data_2_9000_im, train_data_3_9000_im], ignore_index=True)
test_data_9000_im = pd.concat([test_data_1_9000_im, test_data_2_9000_im, test_data_3_9000_im], ignore_index=True)

train_data_9000_im.to_csv('gender_labeled_except_predict_9000_im_train.csv', index=False)
test_data_9000_im.to_csv('gender_labeled_except_predict_9000_im_test.csv', index=False)


# # 1500
# random_state = get_random_state()
# data_1_1500 = df[df['gender_real'] == 1].sample(n=500, random_state=random_state)
# data_2_1500 = df[df['gender_real'] == 2].sample(n=500, random_state=random_state)
# data_3_1500= df[df['gender_real'] == 3].sample(n=500, random_state=random_state)

# combined_data = pd.concat([data_1_1500, data_2_1500, data_3_1500], ignore_index=True)
# combined_data.to_csv('gender_labeled_except_predict_1500_raw.csv', index=False)

# print("Combined data saved to 'gender_labeled_except_predict_1500_raw.csv'.")

# # 결합된 데이터를 8:2 비율로 train:test 데이터로 나누기
# train_data_1_1500 = data_1_1500.sample(n=400, random_state=get_random_state())
# test_data_1_1500 = data_1_1500.drop(train_data_1_1500.index)

# train_data_2_1500 = data_2_1500.sample(n=400, random_state=get_random_state())
# test_data_2_1500 = data_2_1500.drop(train_data_2_1500.index)

# train_data_3_1500 = data_3_1500.sample(n=400, random_state=get_random_state())
# test_data_3_1500 = data_3_1500.drop(train_data_3_1500.index)

# train_data_1500 = pd.concat([train_data_1_1500, train_data_2_1500, train_data_3_1500], ignore_index=True)
# test_data_1500 = pd.concat([test_data_1_1500, test_data_2_1500, test_data_3_1500], ignore_index=True)

# train_data_1500.to_csv('gender_labeled_except_predict_1500_train.csv', index=False)
# test_data_1500.to_csv('gender_labeled_except_predict_1500_test.csv', index=False)

# # 1500_im
# random_state = get_random_state()
# data_1_1500_im = df[df['gender_real'] == 1].sample(n=250, random_state=random_state)
# data_2_1500_im = df[df['gender_real'] == 2].sample(n=250, random_state=random_state)
# data_3_1500_im= df[df['gender_real'] == 3].sample(n=1000, random_state=random_state)

# combined_data = pd.concat([data_1_1500_im, data_2_1500_im, data_3_1500_im], ignore_index=True)
# combined_data.to_csv('gender_labeled_except_predict_1500_im_raw.csv', index=False)

# print("Combined data saved to 'gender_labeled_except_predict_1500_im_raw.csv'.")

# # 결합된 데이터를 8:2 비율로 train:test 데이터로 나누기
# train_data_1_1500_im = data_1_1500_im.sample(n=200, random_state=get_random_state())
# test_data_1_1500_im = data_1_1500_im.drop(train_data_1_1500_im.index)

# train_data_2_1500_im = data_2_1500_im.sample(n=200, random_state=get_random_state())
# test_data_2_1500_im = data_2_1500_im.drop(train_data_2_1500_im.index)

# train_data_3_1500_im = data_3_1500_im.sample(n=800, random_state=get_random_state())
# test_data_3_1500_im = data_3_1500_im.drop(train_data_3_1500_im.index)

# train_data_1500_im = pd.concat([train_data_1_1500_im, train_data_2_1500_im, train_data_3_1500_im], ignore_index=True)
# test_data_1500_im = pd.concat([test_data_1_1500_im, test_data_2_1500_im, test_data_3_1500_im], ignore_index=True)

# train_data_1500_im.to_csv('gender_labeled_except_predict_1500_im_train.csv', index=False)
# test_data_1500_im.to_csv('gender_labeled_except_predict_1500_im_test.csv', index=False)


# # 3000
# random_state = get_random_state()
# data_1_3000 = df[df['gender_real'] == 1].sample(n=1000, random_state=random_state)
# data_2_3000 = df[df['gender_real'] == 2].sample(n=1000, random_state=random_state)
# data_3_3000= df[df['gender_real'] == 3].sample(n=1000, random_state=random_state)

# combined_data = pd.concat([data_1_3000, data_2_3000, data_3_3000], ignore_index=True)
# combined_data.to_csv('gender_labeled_except_predict_3000_raw.csv', index=False)

# print("Combined data saved to 'gender_labeled_except_predict_3000_raw.csv'.")

# # 결합된 데이터를 8:2 비율로 train:test 데이터로 나누기
# train_data_1_3000 = data_1_3000.sample(n=800, random_state=get_random_state())
# test_data_1_3000 = data_1_3000.drop(train_data_1_3000.index)

# train_data_2_3000 = data_2_3000.sample(n=800, random_state=get_random_state())
# test_data_2_3000 = data_2_3000.drop(train_data_2_3000.index)

# train_data_3_3000 = data_3_3000.sample(n=800, random_state=get_random_state())
# test_data_3_3000 = data_3_3000.drop(train_data_3_3000.index)

# train_data_3000 = pd.concat([train_data_1_3000, train_data_2_3000, train_data_3_3000], ignore_index=True)
# test_data_3000 = pd.concat([test_data_1_3000, test_data_2_3000, test_data_3_3000], ignore_index=True)

# train_data_3000.to_csv('gender_labeled_except_predict_3000_train.csv', index=False)
# test_data_3000.to_csv('gender_labeled_except_predict_3000_test.csv', index=False)

# # 3000_im
# random_state = get_random_state()
# data_1_3000_im = df[df['gender_real'] == 1].sample(n=500, random_state=random_state)
# data_2_3000_im = df[df['gender_real'] == 2].sample(n=500, random_state=random_state)
# data_3_3000_im= df[df['gender_real'] == 3].sample(n=2000, random_state=random_state)

# combined_data = pd.concat([data_1_3000_im, data_2_3000_im, data_3_3000_im], ignore_index=True)
# combined_data.to_csv('gender_labeled_except_predict_3000_im_raw.csv', index=False)

# print("Combined data saved to 'gender_labeled_except_predict_3000_im_raw.csv'.")

# # 결합된 데이터를 8:2 비율로 train:test 데이터로 나누기
# train_data_1_3000_im = data_1_3000_im.sample(n=400, random_state=get_random_state())
# test_data_1_3000_im = data_1_3000_im.drop(train_data_1_3000_im.index)

# train_data_2_3000_im = data_2_3000_im.sample(n=400, random_state=get_random_state())
# test_data_2_3000_im = data_2_3000_im.drop(train_data_2_3000_im.index)

# train_data_3_3000_im = data_3_3000_im.sample(n=1600, random_state=get_random_state())
# test_data_3_3000_im = data_3_3000_im.drop(train_data_3_3000_im.index)

# train_data_3000_im = pd.concat([train_data_1_3000_im, train_data_2_3000_im, train_data_3_3000_im], ignore_index=True)
# test_data_3000_im = pd.concat([test_data_1_3000_im, test_data_2_3000_im, test_data_3_3000_im], ignore_index=True)

# train_data_3000_im.to_csv('gender_labeled_except_predict_3000_im_train.csv', index=False)
# test_data_3000_im.to_csv('gender_labeled_except_predict_3000_im_test.csv', index=False)