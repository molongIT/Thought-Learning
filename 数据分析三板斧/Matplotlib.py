
import numpy as np
from matplotlib import pyplot as plt

x = np.arange(1,11)
y = 2 * x
plt.title("demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x,y)
plt.show()

### 用圆来表示点
plt.plot(x,y,"ob")
plt.show()

### 绘制正弦波
x = np.arange(0,3*np.pi,0.1)
y = np.sin(x)
plt.plot(x,y)
plt.show()

## subplot() 函数允许你在同一图中绘制不同的东西。
x = np.arange(0,3*np.pi,0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.subplot(2,1,1)
plt.plot(x,y_sin)
plt.title('Sine')
plt.subplot(2,1,2)
plt.plot(x,y_cos)
plt.title('Cosine')
plt.show()

## pyplot 提供bar()函数来生成条形图
x = [5,8,10]
y = [12,16,6]
x2 = [6,9,11]
y2 = [6,15,7]
plt.bar(x,y,color='g')
plt.bar(x2,y2)
plt.title('Bar graph')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

## numpy.histogram() 函数是数据的频率分布的图形表示
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
hist,bins = np.histogram(a,bins=[0,20,40,60,80,100])
print(hist) #hist 存储了每个分组中元素的数量。
print(bins) #bins 存储了实际的分组边界值
plt.hist(a,bins=[0,20,40,60,80,100])
plt.show()
