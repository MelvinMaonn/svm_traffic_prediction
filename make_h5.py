import numpy as np
import scipy as sp
import pandas as pd

row_name = sp.genfromtxt('1106_timestamp.txt',dtype=str,delimiter='\n')
col_name = sp.genfromtxt('second_ring.txt',dtype=int)
data = sp.genfromtxt('1106_velocity.txt')
# print(data)
# print(row_name.size)
# print(col_name.size)

dataFrame = pd.DataFrame(data=data,index=row_name,columns=col_name)
# print(dataFrame.all())

h5 = pd.HDFStore('1106_velocity.h5','w')

h5['data'] = dataFrame

h5.close()
