import dataset
import matplotlib.pyplot as plt
xs,ys=dataset.get_beans(100)
print(xs)
print(ys)   

#配置图像  
plt.title('Size-Toxicity Function',fontsize=15)
plt.xlabel('Size')
plt.ylabel('Toxicity')
plt.scatter(xs, ys)
#y=w*x+b
w=0.5
b=0
for m in range(100):
    for i in range(100):
        x=xs[i]
        y=ys[i]
        y_pre=w*x+b
        e=y-y_pre
        alpha=0.05
        w=w+alpha*e*x
y_pre=w*xs 
plt.plot(xs,y_pre)
plt.show()


