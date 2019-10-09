import  os
import data_read30
import random
import matplotlib.pyplot as plt
import  copy
import time

class Instance():
    def __init__(self):
        self.successors=[]
        self.job_num_successors=[]
        self.job_predecessors = []
        self.job_successors=[]
        self.job_model_resource={1:{1:[0,0,0,0]},32:{1:[0,0,0,0]}}
        self.job_model_duration={1:{1:0},32:{1:0}}
        self.resource_capacity=[]
        self.check_interupt = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1]
        self.job_weight = [1 for i in range(32)]
        # self.job_weight = [1 for i in range(16)]

        self.number_job =None
        self.number_renewable_resources = None
        self.number_unrenewable_resources = None
        self.resource_capacity = None
        self.disruption_time=None
        self.upper_bound=228

    def loadData(self,file):
        data_read30.dataStore(self, file)
    def initialSolution(self):
        """
        initial the path
        :return: a adapative path
        """
        complete_set=[1]
        for w in range(self.number_job-1):
            ready_set = []
            for job in range(2,self.number_job+1):
                if job not in complete_set:
                    if set(self.job_predecessors[job-1]).issubset(complete_set):
                        ready_set.append(job)
            if len(ready_set)!=0:
                act_job=random.choice(ready_set)
                complete_set.append(act_job)
        initial_set=complete_set
        model_set=[]
        [model_set.append(random.choice([1,2,3])) for i in range(self.number_job)]
        model_set[0]=1
        model_set[self.number_job-1]=1
        return initial_set, model_set

    def solutionCost(self,solution_set,solution_model):
        scheduled_list = [1]
        set_origin = [i for i in range(1, self.number_job-1)]
        # print("set_origin",set_origin)
        for i in set_origin:
            if i not in solution_set:
                scheduled_list.append(i)
        start_time = [0 for _ in range(self.number_job)]
        finish_time = [0 for _ in range(self.number_job)]
        use_renewable_resource_1=[0 for i in range(self.upper_bound)]
        use_renewable_resource_2= [0 for i in range(self.upper_bound)]
        if 1 in solution_set:
            solution_set.remove(1)
        # print("solution_set_cost2", solution_set)
        for job in solution_set:
            if set(self.job_predecessors[job - 1]).issubset(scheduled_list):
                start_time[job-1]=max(finish_time[pre_job-1] for pre_job in self.job_predecessors[job-1])
                # print(start_time[job-1],self.job_model_duration[job][solution_model[job-1]])
                finish_time[job-1]=start_time[job-1]+self.job_model_duration[job][solution_model[job-1]]
                while True:
                    count=0
                    for i in range(self.job_model_duration[job][solution_model[job-1]]):
                        # print( self.job_model_resource[job][solution_model[job-1]][0]+use_renewable_resource_1[start_time[job-1]+i])
                        # print( self.job_model_resource[job][solution_model[job-1]][1]+use_renewable_resource_2[start_time[job-1]+i])
                        if self.job_model_resource[job][solution_model[job-1]][0]+use_renewable_resource_1[start_time[job-1]+i]<=self.resource_capacity[0] and \
                                self.job_model_resource[job][solution_model[job-1]][1]+use_renewable_resource_2[start_time[job-1]+i]<= self.resource_capacity[1]:
                                count+=1
                    if count==self.job_model_duration[job][solution_model[job-1]]:
                        for duration in range(start_time[job-1],start_time[job-1]+self.job_model_duration[job][solution_model[job-1]]):
                            use_renewable_resource_1[duration]+=self.job_model_resource[job][solution_model[job-1]][0]
                            use_renewable_resource_2[duration]+=self.job_model_resource[job][solution_model[job-1]][1]
                        scheduled_list.append(job)
                        break
                    else:
                        start_time[job-1]+=1
                        finish_time[job-1]+=1
            else:
                print("Solution Order Error !")
                print(self.job_predecessors[job - 1],scheduled_list)
        use_renewable_resource_1=use_renewable_resource_1[:finish_time[self.number_job-1]]
        use_renewable_resource_2=use_renewable_resource_2[:finish_time[self.number_job-1]]

        unrenewable_resource1=0
        unrenewable_resource2=0
        for job in solution_set:
             unrenewable_resource1+=self.job_model_resource[job][solution_model[job-1]][2]
             unrenewable_resource2+=self.job_model_resource[job][solution_model[job-1]][3]
        # penalty1=max(0,unrenewable_resource1-self.resource_capacity[2])
        # penalty2=max(0,unrenewable_resource2 -self.resource_capacity[3])
        # finish_time[-1]+=12*(penalty1+penalty2)
        solution_set.insert(0,1)
        return  start_time,finish_time,use_renewable_resource_1,use_renewable_resource_2,unrenewable_resource1,unrenewable_resource2
    def fitness(self,solution_set,solution_model):
        start_time, finish_time, use_renewable_resource_1, use_renewable_resource_2,unrenewable_resource_1, unrenewable_resource_2=self.solutionCost(solution_set,solution_model)
        fit=1/finish_time[-1]
        return fit

    def resolutionCost(self,solution_set,solution_model):
        scheduled_list = [1]
        set_origin = [i for i in range(1, self.number_job-1)]
        # print("set_origin",set_origin)
        for i in set_origin:
            if i not in solution_set:
                scheduled_list.append(i)
        start_time = [0 for _ in range(self.number_job)]
        finish_time = [0 for _ in range(self.number_job)]
        use_renewable_resource_1=[0 for i in range(self.upper_bound)]
        use_renewable_resource_2= [0 for i in range(self.upper_bound)]

        if 1 in solution_set:
            solution_set.remove(1)
        solution_copy = copy.copy(solution_set)
        solution_copy.sort()
        for job in solution_set:
            if set(self.job_predecessors[job - 1]).issubset(scheduled_list):
                start_time[job-1]=max(finish_time[pre_job-1] for pre_job in self.job_predecessors[job-1])
                # print(start_time[job-1],self.job_model_duration[job][solution_model[job-1]])
                finish_time[job-1]=start_time[job-1]+self.job_model_duration[job][solution_model[solution_copy.index(job)]]
                while True:
                    count=0
                    for i in range(self.job_model_duration[job][solution_model[solution_copy.index(job)]]):
                        # print( self.job_model_resource[job][solution_model[job-1]][0]+use_renewable_resource_1[start_time[job-1]+i])
                        # print( self.job_model_resource[job][solution_model[job-1]][1]+use_renewable_resource_2[start_time[job-1]+i])
                        if self.job_model_resource[job][solution_model[solution_copy.index(job)]][0]+use_renewable_resource_1[start_time[job-1]+i]<=self.resource_capacity[0] and \
                                self.job_model_resource[job][solution_model[solution_copy.index(job)]][1]+use_renewable_resource_2[start_time[job-1]+i]<= self.resource_capacity[1]:
                                count+=1
                    if count==self.job_model_duration[job][solution_model[solution_copy.index(job)]]:
                        for duration in range(start_time[job-1],start_time[job-1]+self.job_model_duration[job][solution_model[solution_copy.index(job)]]):
                            use_renewable_resource_1[duration]+=self.job_model_resource[job][solution_model[solution_copy.index(job)]][0]
                            use_renewable_resource_2[duration]+=self.job_model_resource[job][solution_model[solution_copy.index(job)]][1]
                        scheduled_list.append(job)
                        break
                    else:
                        start_time[job-1]+=1
                        finish_time[job-1]+=1
            else:
                print("Solution Order Error !")
                print(self.job_predecessors[job - 1],scheduled_list)
        use_renewable_resource_1=use_renewable_resource_1[:finish_time[self.number_job-1]]
        use_renewable_resource_2=use_renewable_resource_2[:finish_time[self.number_job-1]]

        unrenewable_resource1=0
        unrenewable_resource2=0
        for job in solution_set:
             unrenewable_resource1+=self.job_model_resource[job][solution_model[solution_copy.index(job)]][2]
             unrenewable_resource2+=self.job_model_resource[job][solution_model[solution_copy.index(job)]][3]
        # penalty1=max(0,unrenewable_resource1-self.resource_capacity[2])
        # penalty2=max(0,unrenewable_resource2 -self.resource_capacity[3])
        # finish_time[-1]+=12*(penalty1+penalty2)
        return  start_time,finish_time,use_renewable_resource_1,use_renewable_resource_2,unrenewable_resource1,unrenewable_resource2

    def refitness(self,solution_set,solution_model):
        start_time, finish_time, use_renewable_resource_1, use_renewable_resource_2,unrenewable_resource1,unrenewable_resource2=self.resolutionCost(solution_set,solution_model)
        fit=1/finish_time[-1]
        return fit

    def tabu(self,iter_times,move_times, tabu_length,candidate_length):

        def candidateGenerate(solution_gene,model):

            limit_solution_set=[]
            limit_model_set = []
            solution_candidate=[]
            model_candidate=[]
            fit_candidate=[]
            move_candidate=[]
            # print("-----------------初始解",solution_gene)
            for i in range(candidate_length):
                # print("solution_gene", solution_gene)
                solution_new,model_new,move_new=solutionMove(solution_gene,model,limit_solution_set,limit_model_set)

                # print('solution_new',solution_new)
                fitness=self.fitness(solution_new,model_new)
                solution_candidate.append(solution_new)
                model_candidate.append(model_new)
                fit_candidate.append(fitness)
                move_candidate.append(move_new)
            return  solution_candidate,model_candidate,fit_candidate,move_candidate


        def solutionMove(solution_gene,model,limit_solution_set,limit_model_set):
            while True:
                # print("solution",solution)
                move_job1=random.choice(solution_gene[1:self.number_job])
                ava_move_job=[]
                move_job=()
                solution_set_copy=copy.copy(solution_gene)
                # if move_job not in limit_solution_set:
                if True:
                    #before
                    # print("move_job1:",move_job1)
                    # print("solution_set_copy",solution_set_copy)
                    # print([i for i in self.job_predecessors[move_job1-1]])
                    premove_maxindex=max([solution_set_copy.index(i) for i in self.job_predecessors[move_job1-1]])
                    # print("premove_maxindex",premove_maxindex)
                    if len( solution_set_copy[premove_maxindex+1:solution_set_copy.index(move_job1)])!=0:
                        for job in solution_set_copy[premove_maxindex+1:solution_set_copy.index(move_job1)]:
                                insidepre=[]
                                for job_inside1 in solution_set_copy[solution_set_copy.index(job) + 1:solution_set_copy.index(move_job1)]:
                                    # print("before",job)
                                    # print("succer_before",self.job_successors[job-1])
                                    insidepre+=self.job_predecessors[job_inside1-1]
                                if job not in insidepre:
                                    ava_move_job.append(job)

                    #back
                    scheduled_list=solution_set_copy[:solution_set_copy.index(move_job1)]
                    for job in solution_set_copy[solution_set_copy.index(move_job1)+1:]:
                        if set(self.job_predecessors[job-1]).issubset(scheduled_list) :
                            if len(solution_set_copy[solution_set_copy.index(move_job1)+1:solution_set_copy.index(job)]) !=0:
                                insideaft = []
                                for job_inside2 in solution_set_copy[solution_set_copy.index(move_job1) + 1:solution_set_copy.index(job)]:
                                    insideaft+=self.job_predecessors[job_inside2-1]
                                if move_job1 not in insideaft:
                                    ava_move_job.append(job)


                    if len(ava_move_job)!=0:
                        move_job2=random.choice(ava_move_job)
                        move_job=(move_job1,move_job2)
                        solution_set_copy[solution_set_copy.index(move_job[0])] = move_job[1]
                        solution_set_copy[solution_set_copy.index(move_job[1])] = move_job[0]
                        limit_solution_set.append(move_job)
                        break


            while True:
                move_model=random.sample(model,2)
                move_model.sort()
                model_copy=model[:]
                if True:
                    model_copy[model_copy.index(move_model[0])] = move_model[1]
                    model_copy[model_copy.index(move_model[1])] = move_model[0]
                    limit_model_set.append(move_model)
                    break
            model_candidate=model_copy
            solution_candidate=solution_set_copy
            move=[move_job,move_model]
            return solution_candidate, model_candidate,move

        def chooseBest(solution_list,model_list,fit_list,move_list):
            max_index=fit_list.index(max(fit_list))
            solution_best=solution_list[max_index]
            model_best=model_list[max_index]
            move_best=move_list[max_index]
            fit_best=max(fit_list)
            return solution_best,model_best,move_best,fit_best



        finall_solution=[]
        finall_cost=[]
        finall_model=[]
        best_generation = []
        for i in range(iter_times):    # increasing the initial solution muti
            solution, model = self.initialSolution()
            # print("initial",solution)
            tabu_fitness=[]
            tabu_move=[]
            for i in range(move_times):     # based one solution change
                solution_candidate, model_candidate, fit_candidate, move_candidate = candidateGenerate(solution, model)

                while True:
                    solution_best, model_bset, move_best, fit_best = chooseBest(solution_candidate, model_candidate,fit_candidate, move_candidate)
                    if move_best in tabu_move:
                        index=tabu_move.index(move_best)
                        if fit_best>max(tabu_fitness):
                            finall_solution.append(solution_best)
                            finall_cost.append(1/fit_best)
                            finall_model.append(model_bset)
                            tabu_fitness.append(fit_best)
                            tabu_move.append(move_best)
                            if len(tabu_fitness) > tabu_length:
                                del tabu_fitness[0]
                            if len(tabu_move) > tabu_length:
                                del tabu_move[0]
                            solution=solution_best
                            break

                        else:

                            solution_candidate.remove(solution_best)
                            model_candidate.remove(model_bset)
                            fit_candidate.remove(fit_best)
                            move_candidate.remove(move_best)
                            if len(solution_candidate)==0:
                                break
                    else:
                        finall_solution.append(solution_best)
                        finall_cost.append(1 / fit_best)
                        finall_model.append(model_bset)
                        tabu_fitness.append(fit_best)
                        tabu_move.append(move_best)
                        if len(tabu_fitness) > tabu_length:
                            del tabu_fitness[0]
                        if len(tabu_move) > tabu_length:
                            del tabu_move[0]

                        solution = solution_best
                        break
                print("完成了一次初始值")
                best_generation.append(min(finall_cost))
        index=finall_cost.index(min(finall_cost))
        finall_best_solution=finall_solution[index]
        finall_best_model=finall_model[index]
        print("best model", "\n", finall_best_model)
        print("best solution", "\n", finall_best_solution)
        print("best time","\n",min(finall_cost))
        start_time, finish_time, use_renewable_resource_1, use_renewable_resource_2,unrenewable_resource1,unrenewable_resource2=self.solutionCost(finall_best_solution,finall_best_model)
        print("start_time",start_time,"\n","finish_time",finish_time)
        print("renewable_resource:",use_renewable_resource_1,"\n",use_renewable_resource_2)
        print("unrenewable_resource:", unrenewable_resource1, "\n", unrenewable_resource2)
        return finall_best_solution,finall_best_model,start_time,finish_time
        # plt.plot(best_generation)
        # plt.ylabel("Min_cost")
        # plt.xlabel("Generation")
        #
        # plt.savefig("2.png", dpi=600)
        # plt.show()

    def jobClassfication(self,start_time, finish_time,dis_time):
        # model=[]
        # solution=[]
        A = []
        B = []
        C = []
        D = []
        del_list = [0, start_time[-1]]
        choice_list = [i for i in start_time if i not in del_list]
        # disrution_start = random.choice(choice_list)
        # disrution_start = 1
        disruption_end = dis_time + 2
        self.disruption_time=disruption_end
        job_action_time = list(zip(start_time, finish_time))
        for action_time in job_action_time:
            if action_time[1] <= dis_time:
                A.append(job_action_time.index(action_time) + 1)
                job_action_time[job_action_time.index(action_time)]=0
            elif action_time[0] < dis_time <= action_time[1]:
                B.append(job_action_time.index(action_time) + 1)
                job_action_time[job_action_time.index(action_time)] = 0
            elif dis_time <= action_time[0] < disruption_end:
                C.append(job_action_time.index(action_time) + 1)
                job_action_time[job_action_time.index(action_time)] = 0
            elif disruption_end <= action_time[0]:
                D.append(job_action_time.index(action_time) + 1)
                job_action_time[job_action_time.index(action_time)] = 0
        doc = open('out.txt30', 'a')
        print("disrution_start:",dis_time,"disruption_end",disruption_end,file=doc)
        print(A,B,C,D,file=doc)
        return A,B,C,D,dis_time

    def reschedulGenerate(self,start_time,finish_time,model,check_interupt,dis_time):
        A,B,C,D,disruption_start=self.jobClassfication(start_time,finish_time,dis_time)
        reschedul_list=C+D
        used_unrenewable1 =0
        used_unrenewable2 =0
        unrenewable_resource_left=self.resource_capacity[2:]
        if len(A)!=0:
            for job in A:
                used_unrenewable1+=self.job_model_resource[job][model[job-1]][2]
                used_unrenewable2+=self.job_model_resource[job][model[job-1]][3]
        if len(B)!=0:
            for job in B:
                used_unrenewable1 += ((disruption_start - start_time[job - 1]) / self.job_model_duration[job][model[job - 2]]) * self.job_model_resource[job][model[job - 1]][2]
                used_unrenewable2 += ((disruption_start - start_time[job - 1]) / self.job_model_duration[job][model[job - 2]]) * self.job_model_resource[job][model[job - 1]][3]
                if check_interupt[job-1]:   #可中断
                    choice=[1,2,3]
                    choice.remove(model[job-1])
                    for m in choice:
                        self.job_model_resource[job][m][2]=self.job_model_duration[job][model[job-1]]-((disruption_start - start_time[job - 1]) / self.job_model_duration[job][model[job - 2]]) * self.job_model_resource[job][model[job - 1]][2]
                        self.job_model_resource[job][m][3]=self.job_model_duration[job][model[job-1]]-((disruption_start - start_time[job - 1]) / self.job_model_duration[job][model[job - 2]]) * self.job_model_resource[job][model[job - 1]][3]
                        self.job_model_duration[job][m]=self.job_model_duration[job][model[job-1]]-(disruption_start-start_time[job-1])
                    self.job_model_duration[job][model[job-1]]=self.job_model_duration[job][model[job - 1]] - (disruption_start - start_time[job - 1])
                    reschedul_list.append(job)
                else:
                    reschedul_list.append(job)


                unrenewable_resource_left[0]-=used_unrenewable1
                unrenewable_resource_left[1]-=used_unrenewable2
                self.resource_capacity[2:]=unrenewable_resource_left
        # self.number_job=len(reschedul_list)
        print("reschedul_list",reschedul_list)
        return reschedul_list


    def reschedul_initial(self,reschedul_list):
        # print(reschedul_list)
        complete_set2=[]
        set_origin =[i for i in range(1,self.number_job-1)]
        # print("set_origin",set_origin)
        for i in set_origin:
            if i not in reschedul_list:
                complete_set2.append(i)
        re_list=[]
        for i in range(len(reschedul_list)):
            ready_set = []
            for job in reschedul_list:
                # print(complete_set2)
                if job not in complete_set2:
                    if set(self.job_predecessors[job-1]).issubset(complete_set2):
                        ready_set.append(job)
            if len(ready_set)!= 0:
                act_job = random.choice(ready_set)
                complete_set2.append(act_job)
                re_list.append(act_job)
        initial_set = re_list
        model_set = []
        [model_set.append(random.choice([1, 2, 3])) for _ in range(len(reschedul_list))]
        # model_set[0] = 1
        model_set[len(reschedul_list) - 1] = 1
        return initial_set, model_set

    def ignoreResoucesolution(self,solution_set,solution_model):
            # [solution_model.append(random.choice([1, 2, 3])) for _ in range(len(solution_set))]
            solution_model[-1]=1
            scheduled_list = [1]
            set_origin = [i for i in range(1, self.number_job-1)]
            # print("set_origin",set_origin)
            for i in set_origin:
                if i not in solution_set:
                    scheduled_list.append(i)
            start_time = [0 for _ in range(self.number_job)]
            finish_time = [0 for _ in range(self.number_job)]
            use_renewable_resource_1 = [0 for i in range(self.upper_bound)]
            use_renewable_resource_2 = [0 for i in range(self.upper_bound)]
            if 1 in solution_set:
                solution_set.remove(1)
            solution_copy = copy.copy(solution_set)
            solution_copy.sort()
            for job in solution_set:
                if set(self.job_predecessors[job - 1]).issubset(scheduled_list):
                    start_time[job - 1] = max(finish_time[pre_job - 1] for pre_job in self.job_predecessors[job - 1])
                    finish_time[job - 1] = start_time[job - 1] + self.job_model_duration[job][ solution_model[solution_copy.index(job)]]
                    for duration in range(start_time[job - 1], start_time[job - 1] + self.job_model_duration[job][solution_model[solution_copy.index(job)]]):
                        use_renewable_resource_1[duration] += \
                        self.job_model_resource[job][solution_model[solution_copy.index(job)]][0]
                        use_renewable_resource_2[duration] += \
                        self.job_model_resource[job][solution_model[solution_copy.index(job)]][1]
                    scheduled_list.append(job)
                else:
                    print("Solution Order Error !")
                    print(self.job_predecessors[job - 1], scheduled_list)
            use_renewable_resource_1 = use_renewable_resource_1[:finish_time[self.number_job - 1]]
            use_renewable_resource_2 = use_renewable_resource_2[:finish_time[self.number_job - 1]]
            unrenewable_resource1 = 0
            unrenewable_resource2 = 0
            for job in solution_set:
                unrenewable_resource1 += self.job_model_resource[job][solution_model[solution_copy.index(job)]][2]
                unrenewable_resource2 += self.job_model_resource[job][solution_model[solution_copy.index(job)]][3]
            return start_time,finish_time,use_renewable_resource_1,use_renewable_resource_2

    def reschedul_tabu(self,iter_times,move_times, tabu_length,candidate_length,relist):

        def recandidateGenerate(solution_gene,model):
            limit_solution_set=[]
            limit_model_set = []
            solution_candidate=[]
            model_candidate=[]
            fit_candidate=[]
            move_candidate=[]
            # print("-----------------初始解", solution_gene)
            for i in range(candidate_length):
                # print("solution_gene", solution_gene)
                solution_new,model_new,move_new=resolutionMove(solution_gene,model,limit_solution_set,limit_model_set)

                # print('solution_new',solution_new)
                fitness=self.refitness(solution_new,model_new)
                solution_candidate.append(solution_new)
                model_candidate.append(model_new)
                fit_candidate.append(fitness)
                move_candidate.append(move_new)
            return  solution_candidate,model_candidate,fit_candidate,move_candidate

        def resolutionMove(solution_gene,model,limit_solution_set,limit_model_set):
            while True:
                # print("solution",solution)
                move_job1=random.choice(solution_gene[1:])
                ava_move_job=[]
                solution_set_copy=copy.copy(solution_gene)
                # if move_job not in limit_solution_set:
                if True:
                    index_pre=[]
                    for i in self.job_predecessors[move_job1-1]:
                        if i in solution_set_copy:
                            index_pre.append(solution_set_copy.index(i))
                        else:
                            index_pre.append(0)
                    premove_maxindex=max(index_pre)
                    if len( solution_set_copy[premove_maxindex+1:solution_set_copy.index(move_job1)])!=0:
                        for job in solution_set_copy[premove_maxindex+1:solution_set_copy.index(move_job1)]:
                                insidepre=[]
                                for job_inside1 in solution_set_copy[solution_set_copy.index(job) + 1:solution_set_copy.index(move_job1)]:
                                    # print("before",job)
                                    # print("succer_before",self.job_successors[job-1])
                                    insidepre+=self.job_predecessors[job_inside1-1]
                                if job not in insidepre:
                                    ava_move_job.append(job)
                    #back
                    scheduled_list=solution_set_copy[:solution_set_copy.index(move_job1)]
                    for job in solution_set_copy[solution_set_copy.index(move_job1)+1:]:
                        if set(self.job_predecessors[job-1]).issubset(scheduled_list) :
                            if len(solution_set_copy[solution_set_copy.index(move_job1)+1:solution_set_copy.index(job)]) !=0:
                                insideaft = []
                                for job_inside2 in solution_set_copy[solution_set_copy.index(move_job1) + 1:solution_set_copy.index(job)]:
                                    insideaft+=self.job_predecessors[job_inside2-1]
                                if move_job1 not in insideaft:
                                    ava_move_job.append(job)
                    if len(ava_move_job)!=0:
                        move_job2=random.choice(ava_move_job)
                        move_job=(move_job1,move_job2)
                        solution_set_copy[solution_set_copy.index(move_job[0])] = move_job[1]
                        solution_set_copy[solution_set_copy.index(move_job[1])] = move_job[0]
                        limit_solution_set.append(move_job)
                        break
            while True:
                move_model=random.sample(model,2)
                move_model.sort()
                model_copy=model[:]
                if True:
                    model_copy[model_copy.index(move_model[0])] = move_model[1]
                    model_copy[model_copy.index(move_model[1])] = move_model[0]
                    limit_model_set.append(move_model)
                    break
            model_copy[-1]=1
            model_candidate=model_copy
            solution_candidate=solution_set_copy
            # print("solution_candidate_after",solution_candidate)
            move=[move_job,move_model]
            return solution_candidate, model_candidate,move

        def rechooseBest(solution_list,model_list,fit_list,move_list):
            max_index=fit_list.index(max(fit_list))
            solution_best=solution_list[max_index]
            model_best=model_list[max_index]
            move_best=move_list[max_index]
            fit_best=max(fit_list)
            return solution_best,model_best,move_best,fit_best

        reschedul_finall_solution=[]
        reschedul_finall_cost=[]
        reschedul_finall_model=[]
        best_generation=[]
        for i in range(iter_times):    # increasing the initial solution muti
            solution, model = self.reschedul_initial(relist)
            tabu_fitness=[]
            tabu_move=[]
            for i in range(move_times):     # based one solution change
                # print("before solution",solution)
                solution_candidate, model_candidate, fit_candidate, move_candidate = recandidateGenerate(solution, model)
                while True:
                    solution_best, model_bset, move_best, fit_best = rechooseBest(solution_candidate, model_candidate,fit_candidate, move_candidate)
                    if move_best in tabu_move:
                        index=tabu_move.index(move_best)
                        if fit_best>max(tabu_fitness):
                            reschedul_finall_solution.append(solution_best)
                            reschedul_finall_cost.append(1/fit_best)
                            reschedul_finall_model.append(model_bset)
                            tabu_fitness.append(fit_best)
                            tabu_move.append(move_best)
                            if len(tabu_fitness) > tabu_length:
                                del tabu_fitness[0]
                            if len(tabu_move) > tabu_length:
                                del tabu_move[0]
                            solution=solution_best
                            break
                        else:
                            solution_candidate.remove(solution_best)
                            model_candidate.remove(model_bset)
                            fit_candidate.remove(fit_best)
                            move_candidate.remove(move_best)
                            if len(solution_candidate)==0:
                                break
                    else:
                        reschedul_finall_solution.append(solution_best)
                        reschedul_finall_cost.append(1 / fit_best)
                        reschedul_finall_model.append(model_bset)
                        tabu_fitness.append(fit_best)
                        tabu_move.append(move_best)
                        if len(tabu_fitness) > tabu_length:
                            del tabu_fitness[0]
                        if len(tabu_move) > tabu_length:
                            del tabu_move[0]

                        solution = solution_best
                        break
                # print("完成了一次初始值")
                best_generation.append(min(reschedul_finall_cost))
        index=reschedul_finall_cost.index(min(reschedul_finall_cost))
        finall_best_solution=reschedul_finall_solution[index]
        finall_best_model=reschedul_finall_model[index]
        print("best model", "\n", finall_best_model)
        print("best solution", "\n", finall_best_solution)
        print("best time","\n",min(reschedul_finall_cost))
        start_time, finish_time, use_renewable_resource_1, use_renewable_resource_2, unrenewable_resource1, unrenewable_resource2=self.resolutionCost(finall_best_solution,finall_best_model)
        print("start_time", start_time, "\n", "finish_time", finish_time)
        # print("renewable_resource:", use_renewable_resource_1, "\n", use_renewable_resource_2)
        # print("unrenewable_resource:", unrenewable_resource1, "\n", unrenewable_resource2)
        return finall_best_solution,finall_best_model


    def resource_Tuba(self,disruption_time,resolution,plan_finish_time,itertimes,finishtime):


        def resolutionMove(solution_gene):
            while True:
                move_job1=random.choice(solution_gene[:])
                ava_move_job=[]
                solution_set_copy=copy.copy(solution_gene)
                if True:
                    index_pre=[]
                    for i in self.job_predecessors[move_job1-1]:
                        if i in solution_set_copy:
                            index_pre.append(solution_set_copy.index(i))
                        else:
                            index_pre.append(0)
                    #before
                    premove_maxindex=max(index_pre)
                    if len( solution_set_copy[premove_maxindex+1:solution_set_copy.index(move_job1)])!=0:
                        for job in solution_set_copy[premove_maxindex+1:solution_set_copy.index(move_job1)]:
                                insidepre=[]
                                for job_inside1 in solution_set_copy[solution_set_copy.index(job) + 1:solution_set_copy.index(move_job1)]:
                                    # print("before",job)
                                    # print("succer_before",self.job_successors[job-1])
                                    insidepre+=self.job_predecessors[job_inside1-1]
                                if job not in insidepre:
                                    ava_move_job.append(job)
                    #back
                    scheduled_list=solution_set_copy[:solution_set_copy.index(move_job1)]
                    for job in solution_set_copy[solution_set_copy.index(move_job1)+1:]:
                        if set(self.job_predecessors[job-1]).issubset(scheduled_list) :
                            if len(solution_set_copy[solution_set_copy.index(move_job1)+1:solution_set_copy.index(job)]) !=0:
                                insideaft = []
                                for job_inside2 in solution_set_copy[solution_set_copy.index(move_job1) + 1:solution_set_copy.index(job)]:
                                    insideaft+=self.job_predecessors[job_inside2-1]
                                if move_job1 not in insideaft:
                                    ava_move_job.append(job)
                    if len(ava_move_job)!=0:
                        move_job2=random.choice(ava_move_job)
                        move_job=(move_job1,move_job2)
                        solution_set_copy[solution_set_copy.index(move_job[0])] = move_job[1]
                        solution_set_copy[solution_set_copy.index(move_job[1])] = move_job[0]

                        break
            solution_candidate=solution_set_copy
            # print("solution_candidate_after",solution_candidate)
            return solution_candidate,move_job

        finish_time = finishtime
        #
        finall_model2 = []
        finall_solution2 = []
        finall_start2 = []
        finall_finish2 = []
        finall_cost_group2 = []
        draw=[]
        exceed_cost2=[]
        count1=0

        while count1<itertimes:
            finall_model1 = []
            finall_solution1 = []
            finall_start1 = []
            finall_finish1 = []
            finall_cost_group1 = []
            exceed_cost1 = []
            resource_solution,resource_model=self.reschedul_initial(resolution)
            tabu_list=[]
            resource_model = [1 for _ in range(len(resource_solution))]

            for i in range(20):
                time_pannish = []
                resource_add = []
                start_set = []
                finish_set = []
                solution_set=[]
                move_set=[]
                model_set = []
                num = 0
                while num < 20:
                    pannish=0
                    choice = [1, 2, 3]
                    # change_index = random.randint(0, len(resource_model) - 2)

                    b=[i for i in range(len(resource_model)-1)]
                    change_index=random.sample(b,1)
                    # choice.remove(resource_model[change_index])
                    orign_model=copy.copy(resource_model[change_index[0]])

                    resource_model[change_index[0]] =random.choice(choice)
                    # resource_model = [1 for _ in range(len(resource_solution))]
                    solution,path_move=resolutionMove(resource_solution)
                    move=[change_index,path_move]
                    ignore_start_time, ignore_finish_time, ignore_use_renewable_resource_1, ignore_use_renewable_resource_2=self.ignoreResoucesolution(solution,resource_model)
                    print(ignore_finish_time[-1]+disruption_time)

                    if ignore_finish_time[-1]+disruption_time<=plan_finish_time:
                        exceed_resource1=0
                        exceed_resource2=0
                        for resource1 in ignore_use_renewable_resource_1:
                            if resource1>self.resource_capacity[0]:
                                exceed_resource1+=resource1-self.resource_capacity[0]
                        for resource2 in ignore_use_renewable_resource_2:
                            if resource2>self.resource_capacity[1]:
                                exceed_resource2+=resource2-self.resource_capacity[1]
                        resource_add.append([exceed_resource1,exceed_resource2])

                        move_set.append(move)
                        count2 = 0
                        for ig_finishtime in ignore_finish_time:
                            if ig_finishtime!=0:
                                a = finish_time[ignore_finish_time.index(ig_finishtime)]
                                pannish+=max(0,disruption_time+ig_finishtime-finish_time[count2])*self.job_weight[count2]
                            count2+=1

                        time_pannish.append(pannish)

                        copymodel=copy.copy(resource_model)
                        model_set.append(copymodel)

                        solution_set.append(solution)

                        start_set.append(ignore_start_time)
                        finish_set.append(ignore_finish_time)
                        num+=1
                    else:
                        resource_model[change_index[0]]=orign_model

                tp=[i for i in time_pannish]
                q=[ a[0]+a[1] for a in resource_add]
                cost=[tp[i]+q[i] for i in range(len(tp))]
                #一次移动中所有候选中最好的
                exceed_cost1.append(min(cost))

                finall_model1.append(model_set[cost.index(min(cost))])

                finall_start1.append(start_set[cost.index(min(cost))])
                finall_finish1.append(finish_set[cost.index(min(cost))])
                finall_cost_group1.append([tp[cost.index(min(cost))],q[cost.index(min(cost))]])
                finall_solution1.append(solution_set[cost.index(min(cost))])

                resource_model=copy.copy(finall_model1[exceed_cost1.index(min(exceed_cost1))])

                tabu_list.append(finall_model1[exceed_cost1.index(min(exceed_cost1))])

                if len(tabu_list)>=20:
                    tabu_list.remove(tabu_list[-1])
            #每次移动中最好的
            print("exceed_cost1",exceed_cost1)

            exceed_cost2.append(min(exceed_cost1))


            finall_model2.append(finall_model1[exceed_cost1.index(min(exceed_cost1))])
            # print("finall_model2",finall_model2)
            finall_start2.append(finall_start1[exceed_cost1.index(min(exceed_cost1))])
            finall_finish2.append(finall_finish1[exceed_cost1.index(min(exceed_cost1))])

            finall_cost_group2.append(finall_cost_group1[exceed_cost1.index(min(exceed_cost1))])
            finall_solution2.append(finall_solution1[exceed_cost1.index(min(exceed_cost1))])

            draw.append(min(exceed_cost2))
            count1 += 1
            print("完成一次初始化搜索")

        doc = open('out.txt30', 'a')
        print("exced_cost2",exceed_cost2,file=doc)
        print("min(exceed_resource)",min(exceed_cost2),file=doc)
        print("group:",finall_cost_group2[exceed_cost2.index(min(exceed_cost2))],file=doc)

        print(finall_model2[exceed_cost2.index(min(exceed_cost2))],file=doc)
        print(finall_solution2[exceed_cost2.index(min(exceed_cost2))],file=doc)

        print(finall_start2[exceed_cost2.index(min(exceed_cost2))])
        print(finall_finish2[exceed_cost2.index(min(exceed_cost2))])
        doc.close()
        plt.plot(draw)
        plt.ylabel("Min_cost")
        plt.xlabel("Generation")
        plt.savefig("Tabu_ressource.png", dpi=600)
        # plt.show()


        solu=finall_solution2[exceed_cost2.index(min(exceed_cost2))]
        mod=finall_model2[exceed_cost2.index(min(exceed_cost2))]

        return solu,mod




filename= "\data\j30.mm\j306_1.mm"
file = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + filename
ins = Instance()
ins.loadData(file)



finall_best_solution,finall_best_model,start_time1,finish_time1=ins.tabu(iter_times=100,move_times=100,tabu_length=100,candidate_length=30)


s_t=start_time1
f_t=finish_time1
mod=finall_best_model
doc = open('out.txt30', 'w')
print("s_t:",s_t,file=doc)
print("f_t:",f_t,file=doc)
doc.close()

for dis_time in range(1,16):
    file = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + filename
    ins = Instance()
    ins.loadData(file)


    time_count_start = time.time()
    listr = ins.reschedulGenerate(start_time=s_t, finish_time=f_t, model=mod,check_interupt=ins.check_interupt, dis_time=dis_time,)
    finall_solution,finall_model=ins.resource_Tuba(ins.disruption_time,listr,f_t[-1],100,finishtime=f_t)

    doc = open('out.txt30', 'a')
    print("------------------------我是分割线--------------------------------",file=doc)
    time_count_end = time.time()

    ignore_start_time, ignore_finish_time, ignore_use_renewable_resource_1, ignore_use_renewable_resource_2 = ins.ignoreResoucesolution(finall_solution, finall_model)
    really_start = []
    really_finisih = []
    dis_end = dis_time + 2
    count = 0
    for i in ignore_finish_time:
        if i == 0:
            really_start.append(s_t[count])
            really_finisih.append(f_t[count])
        else:
            really_start.append(ignore_start_time[count] + dis_end)
            really_finisih.append(ignore_finish_time[count] + dis_end)
        count += 1
    print(really_start, "\n", really_finisih,file=doc)

    print(ignore_use_renewable_resource_1, "\n", ignore_use_renewable_resource_2,file=doc)

    print("算法时间：", time_count_end - time_count_start, "s",file=doc)
    print("\n","-----------------------一次中断----------------------------",file=doc)

    doc.close()



