#Author:Zhao fei


origin_start=[0, 0, 0, 0, 6, 1, 8, 6, 13, 13, 13, 18, 20, 28, 8, 18, 28, 12, 33, 18, 20, 28, 36, 29, 22, 33, 23,37, 36, 42, 44, 49]
origin_finish=[0, 1, 6, 1, 10, 4, 13, 8, 18, 20, 23, 22, 28, 33, 12, 25, 31, 20, 36, 23, 29, 29, 42, 31, 26, 37,31, 44, 39, 49, 49, 49]

dis_start=[0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 9, 17, 0, 4, 17, 2, 20, 5, 9, 17, 23, 18, 9, 20, 10, 24, 13, 29, 25, 35]
dis_finish=[0, 0, 0, 0, 0, 0, 4, 0, 5, 9, 10, 9, 17, 20, 2, 9, 20, 6, 23, 10, 17, 18, 29, 20, 10, 24, 13, 25, 16, 35, 26, 35]

dis_time= 10

really_start=[]
really_finisih=[]

count=0
for i in dis_finish:
    if i==0:
        really_start.append(origin_start[count])
        really_finisih.append(origin_finish[count])
    else:
        really_start.append(dis_start[count]+dis_time)
        really_finisih.append(dis_finish[count]+dis_time)
    count+=1
print(really_start,"\n",really_finisih)

