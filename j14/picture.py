#Author:Zhao fei




x=[1,2,3,4,5,6,7,8,9,10,11]
y1=[67,67,58,50,32,23,23,17,80,80,80]
y2=[18,23,26,22,15,11,8,12,80,80,80]
y3=[49,44,32,28,17,12,15,5,80,80,80]

import matplotlib.pylab as plt  # 导入绘图包
import matplotlib.pyplot as mp
from pylab import * #图像中的title,xlabel,ylabel均使用中文
import numpy as np
import  matplotlib.font_manager

matplotlib.rcParams["font.family"]="Kaiti"
matplotlib.rcParams["font.size"]=20
fig, ax1 = plt.subplots()


fig, ax1 = plt.subplots() # 使用subplots()创建窗口
ax1.plot(x,y1,'v:', c= 'b',label='总损失', linewidth = 2) #绘制折线图像1,圆形点，标签，线宽
ax1.plot(x, y2,'*-.', c='#FF8C00',label='延误成本', linewidth = 2)
ax1.plot(x, y3,'d--', c='firebrick',label='资源追加成本', linewidth = 2)
mp.legend(loc=4,fontsize=10)

# ax2 = ax1.twinx() # 创建第二个坐标轴
# ax2.plot(x, y2, 'o-', c='blue',label='y2', linewidth = 1) #同上
# mp.legend(loc=1)

ax1.set_xlabel('中断时间',size=16)
ax1.set_ylabel('中断损失成本',size=18)
ax1.set_xticks([1,2,3,4,5,6,7,8,9,10,11])
ax1.set_yticks([10,20,30,40,50,60,70])
# # ax2.set_ylabel('y2', fontproperties=myfont,size=18)
#自动适应刻度线密度，包括x轴，y轴

plt.savefig("中断res.png", dpi=600, bbox_inches = 'tight')
plt.show()
