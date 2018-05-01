import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.datasets.samples_generator import make_regression

# 读入数据
# data = sp.genfromtxt('workday_13150_improve.txt')
data = sp.genfromtxt('100_roads_1106_data.txt')

print(data.size)

# 为 X , y 赋值
X = data[:,:17]
y = data[:,17]

# print(X)

# X.shape = (1021,1)
# y.shape = (1021,1)



# X = np.vstack((time,X))
# y = np.vstack((time,y))

# plt.scatter(time, X)
# plt.scatter(time, y)

# print(y)

# plt.show()

# 正规化
X = preprocessing.scale(X)

# 分割训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)


# 线性核函数初始化的SVR
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

# 多项式核函数初始化的SVR
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

# 径向基核函数初始化的SVR
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

'''
time = np.arange(rbf_svr_y_predict.size)

plt.scatter(time, rbf_svr_y_predict)
plt.show()
'''

ss_X=StandardScaler()
ss_y=StandardScaler()


print('R-squared value of linear SVR is',linear_svr.score(X_test,y_test))
print('The mean squared error of linear SVR is',mean_squared_error(y_test,linear_svr_y_predict))
print('The mean absolute error of linear SVR is',mean_absolute_error(y_test,linear_svr_y_predict))

print(' ')
print('R-squared value of Poly SVR is',poly_svr.score(X_test,y_test))
'''  
print('The mean squared error of Poly SVR is',mean_squared_error(ss_y.inverse_transform(y_test),
                                                                 ss_y.inverse_transform(poly_svr_y_predict)))
print('The mean absolute error of Poly SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),
                                                                   ss_y.inverse_transform(poly_svr_y_predict)))
'''

print(' ')
print('R-squared value of RBF SVR is',rbf_svr.score(X_test,y_test))
'''
print(y_test)
print(rbf_svr_y_predict)
'''

print('The mean squared error of RBF SVR is',mean_squared_error(y_test,rbf_svr_y_predict))
print('The mean absolute error of RBF SVR is',mean_absolute_error(y_test,rbf_svr_y_predict))
