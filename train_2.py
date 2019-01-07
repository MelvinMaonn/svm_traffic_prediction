import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.svm import SVR
import pandas as pd

import metrics

import datetime
import config

ROAD_NUM = 829

STEP = 12

PRED_TIME = 2


data_path = 'G:\SRCN\seq2seq\seq2seq_data\seq2seq\\'

road_id = range(829)
train_data = np.load(data_path + 'train_7_9_weekday_1_12.npz')
val_data = np.load(data_path + 'val_7_9_weekday_1_12.npz')

train_data_x = train_data['x']
train_data_y = train_data['y']

val_data_x = val_data['x']
val_data_y = val_data['y']

TIME_LENGTH = len(train_data_x)

train_data_x = np.reshape(train_data_x, (-1, 829))

train_data_x = train_data_x.T

X_train = np.zeros(shape=(TIME_LENGTH * ROAD_NUM, STEP + 1))

y_train = np.zeros(shape=(TIME_LENGTH * ROAD_NUM))

for index, road_velocity in enumerate(train_data_x):

    X_temp = np.zeros(shape=(TIME_LENGTH, STEP))

    for i in range(0, road_velocity.size, STEP):
        X_temp[i//STEP] = np.array(road_velocity[i:i + STEP])
        y_train[index * TIME_LENGTH + i//STEP] = train_data_y[i // STEP][PRED_TIME - 1][index]

    id_seq = np.full((TIME_LENGTH, 1), road_id[index])
    result = np.hstack((id_seq, X_temp))
    X_train[index * TIME_LENGTH: (index + 1) * TIME_LENGTH] = result

# print(X_train.size)
# print(y_train.size)

# df = pd.DataFrame(X_train)
# df.to_csv("prediction/X_train_2min.csv", index=False, header=False)
# df = pd.DataFrame(y_train)
# df.to_csv("prediction/y_train_2min.csv", index=False, header=False)


TIME_LENGTH = len(val_data_x)

val_data_x = np.reshape(val_data_x, (-1, 829))

val_data_x = val_data_x.T

X_test = np.zeros(shape=(TIME_LENGTH * ROAD_NUM, STEP + 1))

y_test = np.zeros(shape=(TIME_LENGTH * ROAD_NUM))


for index, road_velocity in enumerate(val_data_x):

    X_temp = np.zeros(shape=(TIME_LENGTH, STEP))

    for i in range(0, road_velocity.size, STEP):
        X_temp[i//STEP] = np.array(road_velocity[i:i + STEP])
        y_test[index * TIME_LENGTH + i//STEP] = val_data_y[i // STEP][PRED_TIME - 1][index]

    id_seq = np.full((TIME_LENGTH, 1), road_id[index])
    result = np.hstack((id_seq, X_temp))
    X_test[index * TIME_LENGTH: (index + 1) * TIME_LENGTH] = result

# print(X_test.size)
# print(y_test.size)

# df = pd.DataFrame(X_test)
# df.to_csv("prediction/X_val_2min.csv", index=False, header=False)
# df = pd.DataFrame(y_test)
# df.to_csv("prediction/y_val_2min.csv", index=False, header=False)



# 正规化
X_train = preprocessing.scale(X_train)

X_train_slim = []
y_train_slim = []

for i in range(0, X_train.size//13, 64):
    X_train_slim.append(X_train[i])
    y_train_slim.append(y_train[i])

X_test = preprocessing.scale(X_test)

# 径向基核函数初始化的SVR
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train_slim, y_train_slim)

start = datetime.datetime.now()
rbf_svr_y_predict = rbf_svr.predict(X_test)
end = datetime.datetime.now()
print("训练预测耗时：" + str(end - start))

df = pd.DataFrame(rbf_svr_y_predict)
df.to_csv("prediction/" + config.global_start_time + "_prediction.csv", index=False, header=False)

print(' ')
start = datetime.datetime.now()
print('R-squared value of RBF SVR is', rbf_svr.score(X_test, y_test))
print('The root mean squared error of RBF SVR is', metrics.masked_rmse_np(rbf_svr_y_predict, y_test, 0))
print('The mean absolute error of RBF SVR is', metrics.masked_mae_np(rbf_svr_y_predict, y_test, 0))
print('The mean absolute percentage error of RBF SVR is', metrics.masked_mape_np(rbf_svr_y_predict, y_test, 0))
end = datetime.datetime.now()
print("计算metrics耗时：" + str(end - start))

# joblib.dump(rbf_svr, 'save/rbf_svr_100_1min_3day.pkl')
