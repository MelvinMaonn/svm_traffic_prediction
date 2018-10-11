import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

import metrics
import datetime

ROAD_NUM = 100

STEP = 12

PRED_TIME = 1

TIME_LENGTH = 1021*30 - STEP - PRED_TIME + 1


road_id = np.genfromtxt('G:/SRCN/beijing_union_second_ring_800r.txt')
data = np.genfromtxt('G:/SRCN/November_800r_velocity_cnn.txt')

data = data.T

X = np.zeros(shape=(TIME_LENGTH * ROAD_NUM, STEP+2))

y = np.zeros(shape=(TIME_LENGTH * ROAD_NUM))

time_seq = np.arange(1021)

for i in range(28):
    time_seq = np.append(time_seq,np.arange(1021))

time_seq = np.append(time_seq,np.arange(1021 - STEP - PRED_TIME + 1))

for index, road_velocity in enumerate(data):

    if index % 6 != 0 or index == 0:
        continue

    if index / 6 > 100:
        break

    X_temp = np.zeros(shape=(TIME_LENGTH, STEP))

    for i in range(0, road_velocity.size - STEP - PRED_TIME + 1):
        X_temp[i] = np.array(road_velocity[i:i + STEP])
        y[(int)(index / 6 - 1) * TIME_LENGTH + i] = road_velocity[i + STEP + PRED_TIME - 1]

    id_seq = np.full((TIME_LENGTH,), road_id[index])
    id_time = np.vstack((id_seq, time_seq))
    id_time = id_time.T
    result = np.hstack((id_time, X_temp))
    X[(int)(index / 6 - 1) * TIME_LENGTH:(int)(index / 6) * TIME_LENGTH] = result

print(X.size)
print(y.size)

# 正规化
X = preprocessing.scale(X)

# 分割训练集和测试集
X_test = X[21*1021:]
y_test = y[21*1021:]


# 径向基核函数初始化的SVR
rbf_svr = joblib.load('save/rbf_svr_100_1min_test_2.pkl')

start = datetime.datetime.now()
rbf_svr_y_predict = rbf_svr.predict(X_test)
end = datetime.datetime.now()
print("预测耗时：" + str(end - start))

np.savetxt('prediction_100r_1min_test_test_2.txt', rbf_svr_y_predict)

print(' ')
start = datetime.datetime.now()
print('R-squared value of RBF SVR is', rbf_svr.score(X_test, y_test))
print('The root mean squared error of RBF SVR is', metrics.masked_rmse_np(rbf_svr_y_predict, y_test, 0))
print('The mean absolute error of RBF SVR is', metrics.masked_mae_np(rbf_svr_y_predict, y_test, 0))
print('The mean absolute percentage error of RBF SVR is', metrics.masked_mape_np(rbf_svr_y_predict, y_test, 0))
end = datetime.datetime.now()
print("计算metrics耗时：" + str(end - start))
