import numpy as np
import dataset
import plot_utils
m=100
X,Y=dataset.get_beans(m)

plot_utils.show_scatter(X,Y)
#一个神经元的两个输入权值w1、w及偏置量b共3个参数初始化
# w1=0.1
# w2=0.1
W=np.array([0.1,0.1])
# b=0.1
B=np.array([0.1])

def forward_propgation(X):
    Z=np.dot(X,W)+B
    A=1/(1+np.exp(-Z))
    return A

plot_utils.show_scatter_surface(X,Y,forward_propgation) 

for _ in range(500 ):
    for i in range(m):
        Xi=X[i]
        Yi=Y[i]
        A=forward_propgation(Xi)
        Ei=(Yi-A)**2
        dEdA=-2*(Yi-A)
        dAdZ=A*(1-A)
        dZdW=Xi
        dZdB=1

        dEdW=dEdA*dAdZ*dZdW
        dEdB=dEdA*dAdZ*dZdB

        alpha=0.01
        W=W-alpha*dEdW
        B=B-alpha*dEdB

plot_utils.show_scatter_surface(X,Y,forward_propgation)
