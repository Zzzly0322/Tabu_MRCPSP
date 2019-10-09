#Author:Zhao fei
import pandas  as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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

matplotlib.rcParams["font.family"]="Kaiti"
matplotlib.rcParams["font.size"]=20
fig, ax1 = plt.subplots()

df = pd.DataFrame({0:y,1: y1, 2:y2,3:y3,4:y4,5:y5,6:y6,7:y7,8:y8})
media=np.median(y)
Me=pd.DataFrame(media for i in range(12))
ax1.plot(Me,"-.",c="black")
# df.plot(kind='box', notch=True, grid=False,ax=ax1)
ax1.boxplot([y,y1,y2,y3,y4,y5,y6,y7,y8],labels=[0,1,2,3,4,5,6,7,8],notch=True)
ax1.set_xlabel('中断时间',size=16)
ax1.set_ylabel('活动完工时间分布',size=16)
plt.savefig("箱型res.png", dpi=600, bbox_inches = 'tight')
plt.show()
