import dataset
import matplotlib.pyplot as plt
import numpy as np
#导入数据
xs,ys=dataset.get_beans(100)

#配置图像
plt.title('Size-Toxicity Function',fontsize=15)
plt.xlabel('Bean Size')
plt.ylabel('Toxicity')
plt.scatter(xs, ys)

w=0.1
b=0.1
z=w*xs+b
a=1/(1+np.exp(-z))

plt.plot(xs,a)
plt.show()

#随机梯度下降
for m in range(50000):
    for i in range(100):
        x=xs[i] 
        y=ys[i]
        #对w和b求偏导
        z=w*x+b
        a=1/(1+np.exp(-z)) 
        e=(y-a)**2 
        deda=2*(a-y)
        dadz=a*(1-a)
        dzdw=x

        dedw=deda*dadz*dzdw
        dzdb=1
        dedb=deda*dadz*dzdb
        alpha=0.05
        #更新w和b
        w=w-alpha*dedw
        b=b-alpha*dedb
    
    if(m%100==0):
        plt.clf()
        plt.scatter(xs, ys)
        z=w*xs+b
        a=1/(1+np.exp(-z))
        plt.xlim(0,1)
        plt.ylim(0,1.2)
        plt.plot(xs,a)
        plt.pause(0.01)