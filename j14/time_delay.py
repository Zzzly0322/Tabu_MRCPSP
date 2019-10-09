#Author:Zhao fei




x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
y=[0, 4, 5, 9, 10, 8, 8, 6, 14, 12, 14, 19, 9, 17, 19, 19]
y1=[0, 7, 7, 11, 5, 10, 9, 5, 7, 13, 5, 14, 8, 16, 18, 18]
y2=[0, 8, 7, 11, 6, 10, 10, 6, 8, 13, 6, 19, 9, 16, 18, 19]
y3=[0, 9, 7, 11, 7, 10, 11, 7, 9, 13, 7, 15, 10, 16, 18, 18]
y4=[0 ,4, 7, 11, 8, 10, 9, 8, 10, 13, 8, 14, 11, 16, 18, 18]
y5=[0, 4, 5, 11, 9, 10, 9, 8, 11, 13, 9, 19, 11, 16, 18, 19]
y6=[0, 4, 5, 11, 10, 10, 10, 6, 12, 13, 13, 19, 11, 16, 18, 19]
y7=[0, 4, 5, 11, 11, 10, 11, 6, 13, 13, 11, 15, 11, 16, 18, 18]
y8=[0, 4, 5, 11, 12, 8, 8, 6, 12, 14, 12, 19, 11, 17, 19, 19]


import matplotlib.pylab as plt  # 导入绘图包
import matplotlib.pyplot as mp
from pylab import * #图像中的title,xlabel,ylabel均使用中文
import numpy as np
import  matplotlib.font_manager
matplotlib.rcParams["font.family"]="Kaiti"
matplotlib.rcParams["font.size"]=20
b=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
c=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# a=[]
# z=[[[y1[i]-y[i]] for i in range(len(y))], [[y2[i]-y[i]] for i in range(len(y))],[[y3[i]-y[i]] for i in range(len(y))],[[y4[i]-y[i]] for i in range(len(y))], [[y5[i]-y[i]] for i in range(len(y))],[[y6[i]-y[i]] for i in range(len(y))], [[y7[i]-y[i]] for i in range(len(y))], [[y8[i]-y[i]] for i in range(len(y))]]
# for i in z:
#     a.append(sum(i)/19)

mp.gcf().set_facecolor(np.ones(3) * 240/255)#设置背景色
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
ax1.plot(x,y,'-', c= '#030303',label='原始计划', linewidth = 2) #绘制折线图像1,圆形点，标签，线宽
ax1.plot(x,y1, '*-.', c='#FF8C00',label='中断点-"1"', linewidth = 1)
ax1.plot(x,y2,'d--', c='r',label='中断点-"2"', linewidth = 1)
ax1.plot(x,y3,'v:', c= 'b',label='中断点-"3"', linewidth = 1) #绘制折线图像1,圆形点，标签，线宽
ax1.plot(x,y4, '*-.', c='#8968CD',label='中断点-"4"', linewidth = 1)
ax1.plot(x,y5,'d--', c='#8B4726',label='中断点-"5"', linewidth = 1)
ax1.plot(x,y6,'v:', c= 'b',label='中断点-"6"', linewidth = 1) #绘制折线图像1,圆形点，标签，线宽
ax1.plot(x,y7,'*-.', c='g',label='中断点-"7"', linewidth = 1)
ax1.plot(x,y8,'d--', c='firebrick',label='中断点-"8"', linewidth = 1)
mp.legend(loc=4,fontsize=8)

# ax2 = ax1.twinx() # 创建第二个坐标轴
# ax2.plot(x, y2, 'o-', c='blue',label='y2', linewidth = 1) #同上
# mp.legend(loc=1)

ax1.set_xlabel('活动编号',size=16)
ax1.set_ylabel('完工时间',size=16)
ax1.set_xticks(b)

# # ax2.set_ylabel('y2', fontproperties=myfont,size=18)
#自动适应刻度线密度，包括x轴，y轴

plt.savefig("res.png", dpi=600, bbox_inches = 'tight')
plt.show()
