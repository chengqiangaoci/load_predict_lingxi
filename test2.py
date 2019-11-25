import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.optimizers import Adam
from keras.models import Sequential
from numpy import concatenate

#读取文件
data_w = pd.read_excel('C:/Users/23956/Desktop/负荷预测/日负荷预测绵阳.xlsx')
q_1 = data_w["负荷"].quantile(.25)
q_2 = data_w["负荷"].quantile(.5)
q_3 = data_w["负荷"].quantile(.75)
iqr = q_3 - q_1
up = q_3 + 1.5*iqr
down = q_1 - 1.5*iqr
v_mean = data_w["负荷"].mean()
for i in range(data_w["负荷"].shape[0]):
	if data_w.loc[i,"负荷"] <= 0 or data_w.loc[i,"负荷"] > up or data_w.loc[i,"负荷"] < down:
		data_w.loc[i,"负荷"] = v_mean
max_y = max(data_w["负荷"])
min_y = min(data_w["负荷"])

# #获取表头
header = data_w.columns
header_dic = {'序号':0,'季节':1, '星期':2, '节假日':3, '最高温':4, '最低温':5, '平均温度':6, '风速':7, '湿度':8, '开工率':9, 
       '负荷':10, '用电量':11}

#检测哪些列没有值
nan_nan = data_w.columns[data_w.isna().all()].tolist()

nan_nan_number = []
for i in nan_nan:
	nan_nan_number.append(header_dic[i])
print(str(nan_nan)+"这几列缺失")
if "负荷" in nan_nan:
	print("缺少负荷值，请用户上传")
if "用电量" in nan_nan:
	print("缺少用电量值，请用户上传")

#转换为数组
data = np.array(data_w)

#获取矩阵大小
row = data.shape[0]
line = data.shape[1]

#定义季节映射值
season = {"春天":1,"夏天":2,"秋天":3,"冬天":4}
#季节映射
if "季节" in nan_nan:
	pass
else:
	for index, value in enumerate(data[:,1]):
		data[index,1] = season[value]

#定义星期映射值
week = {"星期一":1,"星期二":1,"星期三":1,"星期四":1,"星期五":1,"星期六":0.5,"星期日":0.6}
#星期映射
if "星期" in nan_nan:
	pass
else:
	for index, value in enumerate(data[:,2]):
		data[index,2] = week[value]

#剔除缺失列
data_reject = np.delete(data, nan_nan_number, axis=1)
data_y = data_reject[int(row)-30:int(row),-3]
line_new = data_reject.shape[1]

#标准化处理
min_max_scaler = MinMaxScaler(feature_range=(0,1))
print(data_reject)
data = min_max_scaler.fit_transform(data_reject[:,0:int(line_new)-2])#这一步已经将日期、用电量剔除了
#转换为DataFrame
data_new = pd.DataFrame(data)
line_new_1 = data_new.shape[1]

data_train = data_new.iloc[0:int(row)-30,:] #读入训练数据
data_test = data_new.iloc[int(row)-30:int(row),:] #读入测试数据
train_y = data_train.iloc[:,line_new_1-1].values #训练样本标签列
train_x = data_train.iloc[:,0:line_new_1-1].values #训练样本特征
test_y = data_test.iloc[:,line_new_1-1].values#测试样本标签列
test_x = data_test.iloc[:,0:line_new_1-1].values #测试样本特征

#构建神经网络
#把数据转换为3维
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
def create_model():
    model = Sequential()
    #输入数据的shape为(n_samples, timestamps, features)
    #隐藏层设置为50, input_shape元组第二个参数1意指features为1
    model.add(LSTM(units=8,input_shape=(train_x.shape[1], train_x.shape[2])))
    #后接全连接层，直接输出单个值，故units为1
    model.add(Dense(units=1))
    model.add(Activation('linear'))#选用线性激活函数
    model.compile(loss='mse',optimizer=Adam(lr=0.001))#损失函数为平均均方误差，优化器为Adam，学习率为0.01
    return model

model = create_model()
history =model.fit(train_x, train_y, epochs=100, batch_size=72, validation_data=(test_x,test_y))

# make a prediction
yhat = model.predict(test_x)
# test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
 
# inv_yhat = concatenate((yhat, test_x[:, -(line_new_1-1):]), axis=1)
# inv_yhat = min_max_scaler.inverse_transform(inv_yhat)
# print(inv_yhat)
# inv_yhat = inv_yhat[:,-1]#反归一化之后的预测值
inv_yhat = pd.DataFrame(yhat)
for i in range(30):
	inv_yhat[0][i] = inv_yhat[0][i]*(max_y-min_y)+min_y

writer = pd.ExcelWriter('C:/Users/23956/Desktop/负荷预测/输出.xlsx')
inv_yhat.to_excel(writer, index=False,encoding='utf-8',sheet_name='Sheet1')
writer.save()

corr = []
for i in range(30):
	test = 100-100*(abs(inv_yhat[0][i] - data_y[i])/data_y[i])
	corr.append(test)
corr_sum = sum(corr)/len(corr)
print("LSTM月度预测准确率为："+ str(corr_sum) + "%")


