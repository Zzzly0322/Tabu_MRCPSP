#Author:Zhao fei
import random



def jobClassfication(start_time,finish_time):
    # model=[]
    # solution=[]
    A = []
    B = []
    C = []
    D = []
    del_list=[0,start_time[-1]]
    choice_list=[i for i in start_time  if i not in del_list]
    disrution_start=random.choice(choice_list)
    disruption_end=disrution_start+2
    job_action_time=list(zip(start_time,finish_time))
    for action_time in job_action_time:
        if action_time[1]<=disrution_start:
            A.append(job_action_time.index(action_time)+1)
        elif action_time[0]<disrution_start<=action_time[1]:
            B.append(job_action_time.index(action_time)+1)
        elif disrution_start<=action_time[0]<disruption_end:
            C.append(job_action_time.index(action_time)+1)
        elif  disruption_end<=action_time[0]:
            D.append(job_action_time.index(action_time)+1)
    # print(start_time,"\n",finish_time)
    # print(disrution_start,disruption_end)





start_time=[0, 7, 0, 0, 10, 10, 10, 18, 17, 18, 24, 19, 28, 33]
finish_time=[0, 10, 7, 4, 12, 18, 17, 19, 24, 20, 28, 24, 33, 33]
res=jobClassfication(start_time,finish_time)
print(res)


