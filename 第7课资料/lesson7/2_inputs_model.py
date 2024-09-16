import numpy as np
import dataset
import plot_utils
m=100
X,Y=dataset.get_beans(m)

plot_utils.show_scatter(X,Y)
#一个神经元的两个输入权值w1、w及偏置量b共3个参数初始化
w1=0.1
w2=0.1
b=0.1


