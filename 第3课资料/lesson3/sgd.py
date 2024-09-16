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
y_pre=w*xs
plt.plot(xs,y_pre)
plt.show()

# #随机梯度下降
# for m in range(100):
#     for i in range(100):
#         x=xs[i]
#         y=ys[i]
#         #a=x^2
#         #b=-2*x*y
#         #c=y^2
#         #k=2aw+b
#         k=2*(x**2)*w+(-2*x*y)
#         alpha=0.1
#         w=w-alpha*k
#         plt.clf()
#         plt.scatter(xs, ys)
#         y_pre=w*xs
#         plt.xlim(0,1)
#         plt.ylim(0,1.2)
#         plt.plot(xs,y_pre)
#         plt.pause(0.01)

#批量梯度下降
alpha=0.1
for m in range(100):
    #代价函数：e=(y-w*x)^2
    #a=x^2
    #b=-2*x*y
    #求解斜率：k=2aw+b
    k=2*np.sum(xs**2)*w+np.sum(-2*xs*ys)
    k=k/100
    w=w-alpha*k
    #绘制预测曲线
    plt.clf()
    plt.title('Size-Toxicity Function'+str(m),fontsize=15)
    plt.scatter(xs, ys)
    y_pre=w*xs
    plt.xlim(0,1)
    plt.ylim(0,1.2)
    plt.plot(xs,y_pre)
    plt.pause(0.01)

# #固定步长下降
# alpha=0.1
# step=0.01
# for m in range(500):
#     #代价函数：e=(y-w*x)^2
#     #a=x^2
#     #b=-2*x*y
#     #求解斜率：k=2aw+b
#     k=2*np.sum(xs**2)*w+np.sum(-2*xs*ys)
#     k=k/100
#     if k<0:
#         w=w+step
#     else:
#         w=w-step
#     #绘制预测曲线
#     plt.clf()
#     plt.title('Size-Toxicity Function'+str(m),fontsize=15)
#     plt.scatter(xs, ys)
#     y_pre=w*xs
#     plt.xlim(0,1)
#     plt.ylim(0,1.2)
#     plt.plot(xs,y_pre)
#     plt.pause(0.01)

# #重新绘制散点图和预测曲线
# plt.title('Size-Toxicity Function',fontsize=15)
# plt.xlabel('Bean Size')
# plt.ylabel('Toxicity')
# plt.scatter(xs,ys)
# y_pre=w*xs
# plt.plot(xs,y_pre)
# plt.show()