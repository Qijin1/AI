import dataset
import numpy as np
import plot_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
m=100
# 一个神经元两个输入特征值的模型
X,Y=dataset.get_beans(m)
plot_utils.show_scatter(X,Y)

model=Sequential()
model.add(Dense(units=1,activation='sigmoid',input_dim=2))
# model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='mean_squared_error',optimizer=SGD(learning_rate=0.05),metrics=['accuracy'])

model.fit(X,Y,epochs=5000,batch_size=10)
pres=model.predict(X)

plot_utils.show_scatter_surface(X,Y,model)
