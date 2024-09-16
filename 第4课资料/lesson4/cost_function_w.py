import matplotlib.pyplot as plt
import dataset
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


m=100
xs,ys=dataset.get_beans(m)
alpha=0.1


#配置图像
plt.title('Size-Toxicity Function',fontsize=15)
plt.xlabel('Bean Size')
plt.ylabel('Toxicity')
plt.xlim(0,1)
plt.ylim(0,1.5)

plt.scatter(xs, ys)

w=0.1
b=0.1
y_pre=w*xs+b

plt.plot(xs,y_pre)
plt.show()
fig=plt.figure()
ax=Axes3D(fig)
ax.set_zlim(0,2)


ws=np.arange(-1,2,0.1)
bs=np.arange(-2,2,0.1)

for b in bs:
    es=[]
    for w in ws:
        y_pre=w*xs+b
        e=np.sum((y_pre-ys)**2)/m
        es.append(e)
    # plt.plot(ws,es)
    ax.plot(ws,es,b,zdir='y')


plt.show()


