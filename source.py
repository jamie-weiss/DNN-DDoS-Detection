import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from sklearn import linear_model
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

def custom_one_hot_encoding(df, column_name, threshold):
    value_counts = df[column_name].value_counts()
    #print(value_counts)
    new_column_names = []
    new_column_val = []
    for count_key in value_counts.keys():
        count = value_counts[count_key]
        if count >= threshold:
            new_column_names.append(column_name + '_' + str(count_key))
            new_column_val.append(count_key)
    #print(new_column_names)
    new_dict = {}
    for v in new_column_val:
        new_dict[v] = []
    new_dict['Other'] = []
    for value in df[column_name]:
        if value in new_column_val:
            new_dict['Other'].append(0)
            for v in new_column_val:
                if v == value:
                    new_dict[v].append(1)
                else:
                    new_dict[v].append(0)
        else:
            new_dict['Other'].append(1)
            for k in new_dict.keys():
                if k == 'Other':
                    continue
                else:
                    new_dict[k].append(0)
    new_column_names.append(column_name + '_Other')
    new_named_dict = {}
    for key, name in zip(new_dict.keys(), new_column_names):
        #print(key + ': ' + str(len(new_dict[key])))
        new_named_dict[name] = new_dict[key]
    new_df = pd.DataFrame(new_named_dict)
    df.pop(column_name)
    return df.join(new_df)

data_df = pd.read_csv('ddos-datasets/ddos_imbalanced/unbalaced_20_80_dataset.csv')
data_df = data_df.drop(columns=['Flow Byts/s', 'Flow Pkts/s'])
data_df = data_df.drop(columns=['Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'])
data_df = data_df.drop(columns=['Fwd URG Flags', 'Bwd URG Flags'])

data_df = custom_one_hot_encoding(data_df, 'Src IP', 100000)
data_df = custom_one_hot_encoding(data_df, 'Src Port', 100000)
data_df = custom_one_hot_encoding(data_df, 'Dst IP', 1000000)
data_df = custom_one_hot_encoding(data_df, 'Dst Port', 100000)

sample = data_df.sample(frac=1)
result = np.array_split(sample, 20)
df = result[0]
df['Label'].value_counts()

'''
	Benign    316159
	ddos       64667
	Name: Label, dtype: int64

'''

df = df.drop(columns=['Src IP_Other', 'Fwd Seg Size Min'])

df_no_ip_test = df.copy()

feature_names = [c for c in df.columns if 'Label' not in c]
X_df = df[feature_names]
y_df = df['Label']

feature_names = df.columns
X_np = X_df.to_numpy()
y_np = y_df.to_numpy().ravel()

print(X_np.shape)
print(y_np.shape)

'''
(380826, 83)
(380826,)

'''

scaler = StandardScaler()
scaler = scaler.fit(X_np)
X_norm = scaler.transform(X_np)

print(X_norm)

'''

[[-0.60183224 -0.22803504 -0.0081779  ... -0.21725688 -0.12442605
   1.95162346]
 [-0.60183224 -0.38068871 -0.01036338 ... -0.21725688 -0.12442605
  -0.51239392]
 [-0.60183224 -0.23404075 -0.0081779  ... -0.21725688 -0.12442605
   1.95162346]
 ...
 [-0.60183224 -0.38069096 -0.01145612 ... -0.21725688 -0.12442605
   1.95162346]
 [-0.60183224 -0.38069057 -0.01145612 ... -0.21725688 -0.12442605
   1.95162346]
 [-0.60183224 -0.37740874 -0.0027142  ... -0.21725688 -0.12442605
  -0.51239392]]

'''

le = preprocessing.LabelEncoder()
le = le.fit(y_np)
labels = le.transform(y_np)

X_train, X_valid, y_train, y_valid = train_test_split(X_norm, labels, test_size=0.15, random_state=11)

print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)

'''
(323702, 83)
(323702,)
(57124, 83)
(57124,)

'''

classifier = Sequential()
classifier.add(Dense(83, input_dim=83, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(40,  activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.01, momentum=0.8)

classifier.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
classifier.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size = 100, epochs = 100)

df_no_ip_test = df_no_ip_test.drop(columns = ['Src IP_18.219.193.20'])
df_no_ip_test = df_no_ip_test.drop(columns = ['Src IP_172.31.69.28'])
df_no_ip_test = df_no_ip_test.drop(columns = ['Src Port_80', 'Src Port_443', 'Src Port_445',
       'Src Port_0', 'Src Port_Other', 'Dst IP_172.31.0.2', 'Dst IP_Other',
       'Dst Port_53', 'Dst Port_80', 'Dst Port_443', 'Dst Port_3389',
       'Dst Port_445', 'Dst Port_0', 'Dst Port_Other'])

df_no_ip_test_copy = df_no_ip_test.copy()
df_no_ip_test_copy = pd.get_dummies(df_no_ip_test_copy, columns=['Label'], drop_first=True)

feature_names_2 = [c for c in df_no_ip_test.columns if 'Label' not in c]
X_df_2 = df_no_ip_test[feature_names_2]
y_df_2 = df_no_ip_test['Label']

X_np_2 = X_df_2.to_numpy()
y_np_2 = y_df_2.to_numpy().ravel()

print(X_np_2.shape)
print(y_np_2.shape)
'''
(380826, 66)
(380826,)

'''

scaler_2 = StandardScaler()
scaler_2 = scaler_2.fit(X_np_2)
X_norm_2 = scaler_2.transform(X_np_2)

le_2 = preprocessing.LabelEncoder()
le_2 = le_2.fit(y_np)
labels_2 = le_2.transform(y_np)

X_train_2, X_valid_2, y_train_2, y_valid_2 = train_test_split(X_norm_2, labels_2, test_size=0.15, random_state=11)

print(X_train_2.shape)
print(y_train_2.shape)
print(X_valid_2.shape)
print(y_valid_2.shape)

'''
(323702, 66)
(323702,)
(57124, 66)
(57124,)

'''

classifier = Sequential()
classifier.add(Dense(66, input_dim=66, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(30,  activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.01, momentum=0.8)

classifier.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

classifier.fit(X_train_2, y_train_2, validation_data=(X_valid_2, y_valid_2), batch_size = 100, epochs = 100)

cm = confusion_matrix(y_valid_2, y_pred)
print(cm)

'''
[[47150   285]
 [   78  9611]]
 
'''