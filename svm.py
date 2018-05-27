import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

ROAD_NUM = 655

STEP = 12

TIME_LENGTH = 1021 - STEP

PRED_TIME = 3

road_id = np.genfromtxt('E:/data/Data0/output4/SRCN/beijing_union_second_ring_800r.txt')
data = np.genfromtxt('E:/y.txt')

data = data.T

# print(data.shape)

X = np.zeros(shape=(TIME_LENGTH * ROAD_NUM, STEP+2))

y = np.zeros(shape=(TIME_LENGTH * ROAD_NUM))

time_seq = np.arange(TIME_LENGTH)

for index, road_velocity in enumerate(data):

    X_temp = np.zeros(shape=(TIME_LENGTH, STEP))

    for i in range(0, road_velocity.size - STEP - PRED_TIME + 1):
        X_temp[i] = np.array(road_velocity[i:i + STEP])
        y[index*TIME_LENGTH + i] = road_velocity[i + STEP + PRED_TIME - 1]

    id_seq = np.full((TIME_LENGTH,), road_id[index])
    id_time = np.vstack((id_seq, time_seq))
    id_time = id_time.T
    result = np.hstack((id_time, X_temp))
    X[index*TIME_LENGTH :(index+1)*TIME_LENGTH] = result

# 正规化
X = preprocessing.scale(X)

# 分割训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)


# 线性核函数初始化的SVR
# linear_svr = SVR(kernel='linear')
# linear_svr.fit(X_train, y_train)
# linear_svr_y_predict = linear_svr.predict(X_test)

# 多项式核函数初始化的SVR
# poly_svr = SVR(kernel='poly')
# poly_svr.fit(X_train, y_train)
# poly_svr_y_predict = poly_svr.predict(X_test)

# 径向基核函数初始化的SVR
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

ss_X=StandardScaler()
ss_y=StandardScaler()


# print('R-squared value of linear SVR is',linear_svr.score(X_test,y_test))
# print('The mean squared error of linear SVR is',mean_squared_error(y_test,linear_svr_y_predict))
# print('The mean absolute error of linear SVR is',mean_absolute_error(y_test,linear_svr_y_predict))

# print(' ')
# print('R-squared value of Poly SVR is',poly_svr.score(X_test,y_test))
'''  
print('The mean squared error of Poly SVR is',mean_squared_error(ss_y.inverse_transform(y_test),
                                                                 ss_y.inverse_transform(poly_svr_y_predict)))
print('The mean absolute error of Poly SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                   ss_y.inverse_transform(poly_svr_y_predict)))
'''

print(' ')
print('R-squared value of RBF SVR is',rbf_svr.score(X_test,y_test))
print('The mean squared error of RBF SVR is',mean_squared_error(y_test,rbf_svr_y_predict))
print('The mean absolute error of RBF SVR is',mean_absolute_error(y_test,rbf_svr_y_predict))