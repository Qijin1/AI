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
y_pre=w*xs+b
plt.plot(xs,y_pre)
plt.show()


