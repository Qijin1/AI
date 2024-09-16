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
y_pre=w*xs+b
plt.plot(xs,y_pre)
plt.show()

#随机梯度下降
for m in range(500):
    for i in range(100):
        x=xs[i]
        y=ys[i]
        #a=x^2
        #b=-2*x*y
        #c=y^2
        #k=2aw+b
        dw=2*(x**2)*w+2*x*b-2*x*y
        db=2*b+2*x*w-2*y
        alpha=0.1
        w=w-alpha*dw
        b=b-alpha*db
    plt.clf()
    plt.scatter(xs, ys)
    y_pre=w*xs+b
    plt.xlim(0,1)
    plt.ylim(0,1.2)
    plt.plot(xs,y_pre)
    plt.pause(0.01)