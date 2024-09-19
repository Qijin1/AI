import numpy as np
import dataset
import plot_utils
m=100
xs,ys=dataset.get_beans(m)

#一个神经元的两个输入权值w1、w及偏置量b共3个参数初始化
w1=0.1
w2=0.1
b=0.1

x1s=xs[:,0]
x2s=xs[:,1]

z=w1*x1s+w2*x2s+b
a=1/(1+np.exp(-z))

def forward_propgation(x1,x2):
    z=w1*x1+w2*x2+b
    a=1/(1+np.exp(-z))
    return a

plot_utils.show_scatter_surface(xs,ys,forward_propgation)

for  _ in range(500):
    for i in range(m):
        x1=xs[i][0]
        x2=xs[i][1]
        y=ys[i]
        a=forward_propgation(x1,x2)
        e=(y-a)**2
        deda=-2*(y-a)
        dadz=a*(1-a)
        dzdw1=x1
        dzdw2=x2

        dedw1=deda*dadz*dzdw1
        dedw2=deda*dadz*dzdw2
        dedb=deda*dadz

        alpha=0.01
        w1=w1-alpha*dedw1
        w2=w2-alpha*dedw2
        b=b-alpha*dedb

plot_utils.show_scatter_surface(xs,ys,forward_propgation)

