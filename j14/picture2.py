#Author:Zhao fei





x=[1,2,3,4,5,6,7,8,9,10,11]
y1=[28,35,47,32,28,29,36,24,100,100,100]
y2=[52,49,46,31,26,22,15,1,100,100,100]
y3=[80,84,93,63,54,51,51,25,100,100,100]

import matplotlib.pylab as plt  # 导入绘图包
import matplotlib.pyplot as mp
from pylab import * #图像中的title,xlabel,ylabel均使用中文
import numpy as np
import  matplotlib.font_manager
matplotlib.rcParams["font.family"]="Kaiti"
matplotlib.rcParams["font.size"]=20


mp.gcf().set_facecolor(np.ones(3) * 240/255)#设置背景色
fig, ax1 = plt.subplots() # 使用subplots()创建窗口
ax1.plot(x,y1,'v:', c= 'b',label='tatall_cost', linewidth = 2) #绘制折线图像1,圆形点，标签，线宽
ax1.plot(x, y2,'*-.', c='#FF8C00',label='time_cost', linewidth = 2)
ax1.plot(x, y3,'d--', c='firebrick',label='resource_cost', linewidth = 2)
mp.legend(loc=4,fontsize=10)

# ax2 = ax1.twinx() # 创建第二个坐标轴
# ax2.plot(x, y2, 'o-', c='blue',label='y2', linewidth = 1) #同上
# mp.legend(loc=1)

ax1.set_xlabel('disruption_time',size=16)
ax1.set_ylabel('cost',size=18)
ax1.set_xticks([1,2,3,4,5,6,7,8,9,10,11])
ax1.set_yticks([10,20,30,40,50,60,70,80,90,100])
# # ax2.set_ylabel('y2', fontproperties=myfont,size=18)
#自动适应刻度线密度，包括x轴，y轴

plt.savefig("res2.png")
plt.show()
