import numpy as np
from  keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from keras.utils import to_categorical

m=100
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
print("X_train.shape:"+str(X_train.shape))
print("Y_train.shape:"+str(Y_train.shape))
print("X_test.shape:"+str(X_test.shape))
print("Y_test.shape:"+str(Y_test.shape))

print("Y_train[0]:"+str(Y_train[0]))
plt.imshow(X_train[0],cmap='gray')
plt.show()

X_train=X_train.reshape(60000,784)
X_test=X_test.reshape(10000,784)

Y_train=to_categorical(Y_train,num_classes=10)
Y_test=to_categorical(Y_test,num_classes=10)

model = Sequential()
model.add(Dense(units=256,activation='relu',input_dim=784))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

model.compile(loss='mean_squared_error',optimizer=SGD(learning_rate=0.05),metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=5000,batch_size=4096)
# pres=model.predict(X)




