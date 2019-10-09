#Author:Zhao fei


import random

class test():
    def __init__(self):
        self.asdf=[2,3,4]

    def cahnge(self):
        self.asdf[0]=4

q=[1, 2, 3,  4,  5,  6,  7, 8, 9,  10, 11, 12, 13, 14, 15, 16]
a=[0, 0, 0,  9,  9,  9,  9, 4, 9,  11,  9, 11,  9, 13, 16, 18]
b=[0, 4, 5, 11, 11, 10, 11, 6, 13, 13, 11, 15, 11, 16, 18, 18]






start_time=[0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 7, 9]
finish_time=[0, 0, 0, 1, 2, 0, 0, 0, 2, 4, 2, 9, 1, 7, 9, 9]
# num1=0
# for i in start_time:
#     start_time[num1]+=10
#     num1+=1
# num2=0
# for x in finish_time:
#     finish_time[num2]+=10
#     num2 += 1
#
# print("start_time=",start_time,"\n","finish_time=",finish_time)



y=7

x=[1,2,3,4,5]

for i in range(5):
    y+=i
    x.append(y)

print(x)

