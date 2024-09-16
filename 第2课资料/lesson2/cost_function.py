import dataset
import matplotlib.pyplot as plt
import numpy as np

xs,ys=dataset.get_beans(100)

#配置图像  
plt.title('Size-Toxicity Function',fontsize=15)
plt.xlabel('Size')
plt.ylabel('Toxicity')
plt.scatter(xs, ys)
#y=w*x+b
w=0.1
b=0
y_pre=w*xs+b
plt.plot(xs,y_pre)
plt.show()

es=(ys-y_pre)**2
sum_e=np.sum(es)
ave_e=np.average(es)
print("sum_e="+str(sum_e)+"ave_e="+str(ave_e))
ws=np.arange(0,3,0.1)

es=[]
for w in ws:
    y_pre = w * xs + b
    e = (1/100)*np.sum((ys - y_pre) ** 2)
    es.append(e)
plt.title('cost function',fontsize=15)
plt.xlabel('w')
plt.ylabel('e')
plt.plot(ws,es)
plt.show()

#使cost function最小的w
w_min=np.sum(xs*ys)/np.sum(xs**2)
print("e最小点w:"+str(w_min))

y_pre=w_min*xs+b
plt.title('Size-Toxicity Function',fontsize=15)
plt.xlabel('Size')
plt.ylabel('Toxicity')
plt.scatter(xs, ys)
plt.plot(xs,y_pre)
plt.show()



