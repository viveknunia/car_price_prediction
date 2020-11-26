import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import seaborn as sns
from matplotlib.pyplot import xticks

#loading the data
df = pd.read_csv('https://raw.githubusercontent.com/viveknunia/car_price/main/cars_price.csv')




#data cleaning : removing the nan values



from statistics import mode
l = []
for i in df['segment']:
    if i != math.nan:
        l.append(i)
seg_mode = mode(l)

l = []
for i in df['volume(cm3)']:
    if i != math.nan:
        l.append(i)
vol_mode = (mode(l))

l = []

for i in df['drive_unit']:
    if i != math.nan:
        l.append(i)
drive_unit_mode =  mode(l)

for j in df.columns:
    #print(j)
    ct = 0
    for i in range(len(df)) :
        if type(df[j][i]) != type('sdf'):
            if math.isnan(df[j][i]):
                if(j == 'volume(cm3)'):
                    df[j][i] = vol_mode
                if(j == 'drive_unit'):
                    df[j][i] = drive_unit_mode
                if(j == 'segment'):
                    df[j][i] = seg_mode
                ##print('NaN')
                ct+=1
    #print(ct)


auto = df[['condition','mileage(kilometers)','fuel_type','volume(cm3)','color','transmission','drive_unit','segment','brand_avg_price','brand_category','priceUSD']]
price = df['priceUSD']

#getting dummie variables for the categorical features
auto = pd.get_dummies(auto)

from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train, df_test = train_test_split(auto,train_size = 0.8, test_size = 0.2, random_state = 100)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_var = ['mileage(kilometers)','volume(cm3)','brand_avg_price','priceUSD']
df_train[num_var] = scaler.fit_transform(df_train[num_var])
df_test[num_var] = scaler.fit_transform(df_test[num_var])

y_train = df_train.pop('priceUSD')
x_train = df_train

y_test = df_test.pop('priceUSD')
x_test = df_test


close_path = 'C:/Users/91707/OneDrive - LNMIIT/car_price/model.ckpt'
classifier.load_weights(close_path)


y_pred = classifier.predict(x_test)

num_var = ['mileage(kilometers)','volume(cm3)','brand_avg_price']
df_pred_test = df_test[num_var]
df_pred_test['priceUSD'] = y_pred
df_test_ori = df_test[num_var]
df_test_ori['priceUSD'] = y_test

num_var = ['mileage(kilometers)','volume(cm3)','brand_avg_price','priceUSD']

df_pred_test = scaler.inverse_transform(df_pred_test)
df_test_ori = scaler.inverse_transform(df_test_ori)

col_val = ['1st','2nd','3rd','price']

df_pred_test = pd.DataFrame(data=df_pred_test,columns=col_val)

df_test_ori = pd.DataFrame(data=df_test_ori,columns=col_val)


from sklearn.metrics import mean_squared_error
print(math.sqrt(mean_squared_error(df_pred_test['price'],df_test_ori['price'])), ' = root_mean_squared_error of the test data')
