import pandas as pd

data=pd.read_hdf('1106_velocity.h5',key='data')

print(data)