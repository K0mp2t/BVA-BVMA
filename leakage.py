import numpy as np
from collections import Counter
import time
from numpy.lib.function_base import average
from numpy.lib.shape_base import tile
import utils
import math
import sys
import random


class Leakage: 
    inject_size_threshold_counter = 0 
    def __init__(self, cli_doc, adv_doc, chosen_kws, queries, trend_matrix_norm, adv_trend_matrix_norm, inject_size_threshold_count):
        self.cli_doc = cli_doc
        self.adv_doc = adv_doc
        self.chosen_kws = chosen_kws
        self.queries = queries
        self.trend_matrix_norm = trend_matrix_norm
        self.adv_trend_matrix_norm = adv_trend_matrix_norm
        Leakage.inject_size_threshold_counter = inject_size_threshold_count
        self.running_time = 0
        self.accuracy = 0
        
        return 

    class IKK:
        
        def __init__(self, adv_doc, cli_doc, chosen_kws, queries):
            #leakage = Leakage(cli_doc, adv_doc, chosen_kws, queries, [], [])
            #leakage.inject_size_threshold_count
            self.adv_doc = adv_doc
            self.cli_doc = cli_doc
            self.chosen_kws = chosen_kws
            self.queries = queries

        def build_kws_co_occurrence_matrix(self, doc_kws_matrix):
        # @param: 文档是否拥有关键字的矩阵
        # @return：返回关键字归一化共生矩阵
            
            return (np.matmul(doc_kws_matrix.T, doc_kws_matrix) / np.double(len(doc_kws_matrix)))

        def compute_cost(self, has_matching_query_list, state, adv_matrix, query_matrix):
            total_cost = 0
            for i in range(len(state)):
                for j in range(len(state)):
                    if i in has_matching_query_list and j in has_matching_query_list:
                        total_cost += (adv_matrix[state[i]][state[j]] - query_matrix[i][j]) ** 2
            return total_cost

        def IKK_attack(self):
            # 建立对手已知的关键字共生矩阵
            now_kws_count = 0
            adv_known_kw_to_kw_id = {} # map kw to kw_id
            for adv_doc_id, adv_doc_kws in enumerate(self.adv_doc):
                for kw in adv_doc_kws:
                     if kw not in adv_known_kw_to_kw_id.keys():
                         adv_known_kw_to_kw_id[kw] = now_kws_count
                         now_kws_count += 1

            adv_doc_kws_matrix = np.zeros((len(self.adv_doc), now_kws_count))
            for kw in adv_known_kw_to_kw_id.keys():
                for adv_doc_id, adv_doc_kws in enumerate(self.adv_doc):
                    if kw in adv_doc_kws:
                        adv_doc_kws_matrix[adv_doc_id, adv_known_kw_to_kw_id[kw]] = 1
            adv_known_kws_co_occurrence_matrix = self.build_kws_co_occurrence_matrix(adv_doc_kws_matrix)

            # print(now_kws_count)
            # print(len(self.chosen_kws))
            # adv_kws = np.random.permutation(self.chosen_kws)
            adv_kw_id_to_cli_kw_id = {} # map adv_kw_id to cli_kw_id
            #print(adv_known_kw_to_kw_id.keys())
            # 对每一周的query，建立query共生矩阵
            # 随机生成的query可能不存在匹配文档，因此一定无法进行恢复，在建立共生矩阵时可删除，计算accuracy时添加回来
            cli_doc_query_matrix = np.zeros((len(self.cli_doc), len(self.chosen_kws))) #query从chosen_kws中采样
            has_matching_query_list = []
            query_list = []
            no_matching_doc_queries = 0
            for i_week in self.queries:
                for query in i_week:
                    if query in query_list:
                        continue
                    query_list.append(query)
                    has_matching_flag = False
                    for cli_doc_id, cli_doc_kws in enumerate(self.cli_doc):
                        if self.chosen_kws[query] in cli_doc_kws:
                            has_matching_flag = True
                            cli_doc_query_matrix[cli_doc_id, query] = 1
                    if not has_matching_flag:
                        no_matching_doc_queries += 1
                    else:
                        has_matching_query_list.append(query)
            query_co_occurrence_matrix = self.build_kws_co_occurrence_matrix(cli_doc_query_matrix)

            print(adv_known_kws_co_occurrence_matrix)
            print()
            print(query_co_occurrence_matrix)
            real_tag = {}
            recover_tag = {}
            print("has_matching_query_list is {}".format(has_matching_query_list))

            for query in has_matching_query_list:
                if self.chosen_kws[query] in adv_known_kw_to_kw_id.keys():
                    real_tag[query] = adv_known_kw_to_kw_id[self.chosen_kws[query]]
                   
                else:
                    no_matching_doc_queries += 1
            print(real_tag)
            print()
            """
             for i in real_tag.keys():
                for j in real_tag.keys():
                    if i!=j:
                        print(query_co_occurrence_matrix[i][j])
                        print(adv_known_kws_co_occurrence_matrix[adv_known_kw_to_kw_id[self.chosen_kws[i]]][adv_known_kw_to_kw_id[self.chosen_kws[j]]])
            """
           
            initial_state = list(range(len(self.chosen_kws)))
            for kw in range(len(self.chosen_kws)):
                if kw in has_matching_query_list:
                    initial_state[kw] = kw
                else:
                    initial_state[kw] = 0

            print(len(initial_state))
            
            real_state = list(range(len(self.chosen_kws)))
            for kw in range(len(self.chosen_kws)):
                if kw in has_matching_query_list:
                    real_state[kw] = real_tag[kw]
                else:
                    real_state[kw] = 0

            initial_temp = 100
            final_state = self.annealing_simulation(initial_state, has_matching_query_list, [adv_known_kw_to_kw_id[kw] for kw in adv_known_kw_to_kw_id.keys()], adv_known_kws_co_occurrence_matrix, query_co_occurrence_matrix)
            cost = self.compute_cost(has_matching_query_list, final_state, adv_known_kws_co_occurrence_matrix, query_co_occurrence_matrix)
            r_cost = self.compute_cost(has_matching_query_list, real_state, adv_known_kws_co_occurrence_matrix, query_co_occurrence_matrix)
            print("cost is {}".format(cost))
            print("r_cost is {}".format(r_cost))

            for query in query_list:
                recover_tag[query] = final_state[query]
            print(recover_tag)
            accuracy = np.mean(np.array([1 if real == predict else 0 
                for (real, predict) in zip(real_tag.values(), recover_tag.values())]))       
            print("IKK_attack'accuracy is {}".format(accuracy))


        def annealing_simulation(self, initial_state, has_matching_query_list, adv_known_kw_id, adv_known_kws_co_occurrence_matrix, query_co_occurrence_matrix, initial_temp=200, cooling_rate=0.999, reject_threshold=1500):
            current_state = initial_state
            current_cost = self.compute_cost(has_matching_query_list, initial_state, adv_known_kws_co_occurrence_matrix, query_co_occurrence_matrix)
            current_temp = initial_temp
            succ_reject = 0
            iter = 0
            while current_temp > 1e-10 and succ_reject < reject_threshold: # 运行足够多步或多次cost比原来大，则退出循环
                next_state = current_state[:]  # copy
                query_to_replace = np.random.choice(has_matching_query_list)
                new_kw = np.random.choice(adv_known_kw_id) # keyword
                next_state[query_to_replace] = new_kw
                next_cost = self.compute_cost(has_matching_query_list, next_state, adv_known_kws_co_occurrence_matrix, query_co_occurrence_matrix)
                if next_cost < current_cost or np.random.rand() < np.exp(-(next_cost - current_cost) / current_temp): # 是否接受作为当前解
                    current_state = next_state
                    current_cost = next_cost
                    succ_reject = 0
                else:
                    succ_reject += 1
                current_temp *= cooling_rate

                iter += 1

            return current_state

    class Count:
        """
        假定对手知晓一部分文档以及关键字域
        """
        def __init__(self, adv_doc, cli_doc, chosen_kws, queries):
            self.accuracy = 0
            self.time = 0
            self.real_tag = {}
            self.recover_tag = {}
            self.adv_doc = adv_doc
            self.cli_doc = cli_doc
            self.chosen_kws = chosen_kws
            self.queries = queries

        def count_main(self):
            kws_volume, adv_kws_co_occurrence_matrix = self.build_adv_co_occurrence()
            self.time = self.count(kws_volume, adv_kws_co_occurrence_matrix)
            self.accuracy = self.accuracy

        def build_kws_co_occurrence_matrix(self, doc_kws_matrix):
        # @param: 文档是否拥有关键字的矩阵
        # @return：返回关键字归一化共生矩阵
            return (np.matmul(doc_kws_matrix.T, doc_kws_matrix))

        def build_adv_co_occurrence(self):
            adv_known_rate = len(self.adv_doc)/len(self.cli_doc) # 对手已知文档率，该攻击中通常设为1

            adv_doc_kws_matrix = np.zeros((len(self.adv_doc), len(self.chosen_kws)))
            #now_kws_count = 0
            #adv_kw_to_kw_id = {}
            kws_volume = {}

            for kw_id in range(len(self.chosen_kws)):
                kws_volume[kw_id] = 0
                for adv_doc_id, adv_doc_kws in enumerate(self.adv_doc):
                    if self.chosen_kws[kw_id] in adv_doc_kws:
                        adv_doc_kws_matrix[adv_doc_id][kw_id] = 1
                        kws_volume[kw_id] += 1
                kws_volume[kw_id] /= len(self.adv_doc)
            
            adv_kws_co_occurrence_matrix = self.build_kws_co_occurrence_matrix(adv_doc_kws_matrix)/len(self.adv_doc)
            #print(kws_volume)
            #print(adv_kws_co_occurrence_matrix)
            return kws_volume, adv_kws_co_occurrence_matrix

        def build_query_co_occurrence(self):

            cli_doc_query_matrix = np.zeros((len(self.cli_doc), len(self.chosen_kws)))    
            query_volume = {}     
            for i_week in self.queries:
                for query in i_week:
                    self.real_tag[query] = query
                    query_volume[query] = 0
                    for cli_doc_id, cli_doc_kws in enumerate(self.cli_doc):
                        if self.chosen_kws[query] in cli_doc_kws:
                            cli_doc_query_matrix[cli_doc_id][query] = 1
                            query_volume[query] += 1
                    query_volume[query] /= len(self.cli_doc)

            cli_query_co_occurrence_matrix = self.build_kws_co_occurrence_matrix(cli_doc_query_matrix)/len(self.cli_doc)
            #print(query_volume)
            #print(cli_query_co_occurrence_matrix)
            return query_volume, cli_query_co_occurrence_matrix

        @ utils._timeit
        def count(self, kws_volume, adv_kws_co_occurrence_matrix):
            query_volume, cli_query_co_occurrence_matrix = self.build_query_co_occurrence()
            candidate_recover_set = {}

            #print(self.real_tag)
            for query in self.real_tag.keys():
                temp_query_set = []
                #temp_query_volume_by_rate = cli_queries_volume[query]*adv_known_rate
                #if cli_queries_volume[query] == 0:
                #    recover_tag[query] = -1
                #    continue
                for kw_id in kws_volume.keys():
                    if query_volume[query] == kws_volume[kw_id]:
                        temp_query_set.append(kw_id)
                if len(temp_query_set) == 1:
                    self.recover_tag[query] = temp_query_set[0]
                elif len(temp_query_set) > 1:
                    candidate_recover_set[query] = temp_query_set
                else:
                    temp_volume_minus = query_volume[query]
                    temp_kw_id = -1
                    for kw_id in kws_volume.keys():
                        if temp_volume_minus > abs(query_volume[query] - kws_volume[kw_id]):
                            temp_volume_minus = abs(query_volume[query] - kws_volume[kw_id])
                            temp_kw_id = kw_id
                    self.recover_tag[query] = temp_kw_id

            #print(self.recover_tag)
            #print(len(self.recover_tag)/len(self.real_tag))
            sorted(candidate_recover_set, key=lambda k: len(candidate_recover_set[k]))

            for query in candidate_recover_set.keys():
                lun = 0
                while len(candidate_recover_set[query]) > 1 and lun < 1000:
                    lun += 1
                    for candidate_kw_id in candidate_recover_set[query]:
                        for recover_q in self.recover_tag.keys():
                            if  self.recover_tag[recover_q] != -1:
                                if cli_query_co_occurrence_matrix[recover_q][query] != adv_kws_co_occurrence_matrix[self.recover_tag[recover_q]][candidate_kw_id]:
                                    candidate_recover_set[query].remove(candidate_kw_id)
                                    break
                            else:
                                continue
                if len(candidate_recover_set[query]) == 1:
                    self.recover_tag[query] = candidate_recover_set[query][0]
                else:
                    self.recover_tag[query] = -1
            #print(self.recover_tag)
            for k in self.real_tag.keys():
                if self.real_tag[k] == self.recover_tag[k]:
                    self.accuracy += 1
            self.accuracy /= len(self.real_tag)    
            #print("Count_attack'accuracy is {}".format(self.accuracy))



    class Zhang:

        injected_doc_contain_kws = {} # not necessary
        injected_kws_contain_doc = {} # for query simulation
        injected_doc_level = [] # 每层最大文档id+1
        num_kws_each_group = 0
        num_doc_each_group = 0

        def __init__(self, T, chosen_kws, queries, accuracy, num_inject_doc):
            
            self.time = 0
            self.totals_inject_doc = 0

            self.real_tag = {}
            self.recover_tag = {}

            self.T = T
            self.total_inject_doc = {}
            self.chosen_kws = chosen_kws
            self.queries = queries
            self.accuracy = accuracy
            self.num_inject_doc = num_inject_doc
            self.inject_time = 0
            self.total_inject_size = 0
            self.injected_doc_contain_kws = {}
            self.injected_kws_contain_doc = {}
            self.injected_doc_level = []
            self.num_kws_each_group = 0
            self.num_doc_each_group = 0


            self.first_inject_doc = {}
            self.second_inject_doc = {}
            self.num_FF_each_group = {}
            return
            
        """
        减少文档注入
        改进方案

        """
        def improved_main(self):
            s = time.time()
            self.improved_inject_main()
            #print("self.injected_doc_contain_kws:{}".format(len(self.injected_doc_contain_kws)))
            e = time.time()
            self.inject_time = e - s
            self.time = self.improved_recover_main()
            #print("self.injected_doc_contain_kws:{}".format(len(self.injected_doc_contain_kws)))
            self.totals_inject_doc = len(self.injected_doc_contain_kws)-1
            self.total_inject_size = min(Leakage.inject_size_threshold_counter, self.totals_inject_doc*self.T)
            #
            #self.accuracy = utils.compute_accuracy(self.real_tag, self.recover_tag)
            assert self.accuracy <= 1

        def improved_injected_kws_id_and_doc_id(self, doc_id, kws_id):
            """
            @ param: 注入的文档id集合doc_id及相应的注入关键字集合kws_id
            @ return: 编码为0，即未进行注入的第一个关键字
            """
            if len(kws_id) == 0:
                raise ValueError("kws_id less")
            
            doc_offset = doc_id[0]
            kws_offset = kws_id[0]
            kws_divide = 1
            if len(kws_id) > 1 and kws_id[0] + 1 != kws_id[1]:
                kws_divide = kws_id[1] - kws_id[0]

            for kw_id in kws_id:
                for d_id in doc_id:
                    if (int ((kw_id - kws_offset)/kws_divide) >> (d_id - doc_offset)) & 1 == 1:
                        
                        if d_id in self.injected_doc_contain_kws.keys():
                            self.injected_doc_contain_kws[d_id].append(kw_id)
                        else:
                            temp_kws_set = []
                            temp_kws_set.append(kw_id)
                            self.injected_doc_contain_kws[d_id] = temp_kws_set
                        

                        if kw_id in self.injected_kws_contain_doc.keys():
                            self.injected_kws_contain_doc[kw_id].append(d_id)
                        else:
                            temp_doc_set = []
                            temp_doc_set.append(d_id)
                            self.injected_kws_contain_doc[kw_id] = temp_doc_set

            return kws_id[0]
       
        def improved_inject_main(self):
            """
            进行所有关键字的注入
            总的注入有多层，每层有多组
            """
            # 一层一层进行注入，remain_kws保存当前层需要注入的关键字
            remain_kws = []
            for i in range(len(self.chosen_kws)):
                remain_kws.append(i)
            
            doc_now_id = 0 # 当前注入的文档id

            self.num_kws_each_group = self.T*2
            self.num_doc_each_group = math.ceil(math.log2(self.num_kws_each_group))

            now_inject_size = 0

            remain_size_threshold_counter = Leakage.inject_size_threshold_counter
            while(len(remain_kws) > 1):
                if now_inject_size>Leakage.inject_size_threshold_counter:
                    break
                kws_now_id = 0 # remain_kws索引
                num_group = math.ceil(len(remain_kws)/self.num_kws_each_group) # 当前层的组数
                temp_kws_set = [] # 剩余kws，保存下一层的remain_kws
                now_inject_size += num_group*self.T

                for group_id in range(num_group): # 每一组进行注入
                    remain_size_threshold_counter = max(0, remain_size_threshold_counter - self.num_doc_each_group*self.T)
                    if remain_size_threshold_counter<=0:
                        break
                    doc_id_set = []
                    kws_id_set = []

                    for i in range(self.num_doc_each_group):
                        doc_id_set.append(doc_now_id)
                        doc_now_id += 1
                    for j in range(self.num_kws_each_group):
                        if kws_now_id >= len(remain_kws):
                            break
                        kws_id_set.append(remain_kws[kws_now_id])
                        kws_now_id += 1
                    #print("kws_id_set: {}".format(kws_id_set))
                    #print("doc_id_set: {}".format(doc_id_set))
                    
                    temp_kws_set.append(self.improved_injected_kws_id_and_doc_id(doc_id_set, kws_id_set))

                #print("self.injected_doc_contain_kws:{}".format(len(self.injected_doc_contain_kws)))

                remain_kws = temp_kws_set
                temp_doc_and_level_add = []
                temp_doc_and_level_add.append(doc_now_id)
                if len(self.injected_doc_level) == 0:
                    temp_doc_and_level_add.append(1)
                else:
                    final_id = len(self.injected_doc_level) - 1
                    temp_doc_and_level_add.append(self.injected_doc_level[final_id][1]*self.num_kws_each_group)
                self.injected_doc_level.append(temp_doc_and_level_add)


            #print(self.injected_doc_contain_kws)
            #print(self.injected_kws_contain_doc)
            #print(self.inverted)
            #print(self.injected_doc_level)

        @ utils._timeit
        def improved_recover_main(self):
            # recover
            self.recover_tag = {}
            self.real_tag = {}
            total_query = 0
            recov_query = 0
            for i_week in self.queries:
                for query in i_week:
                    total_query+=1
                    self.real_tag[query] = query
                    if query not in self.injected_kws_contain_doc.keys():
                        self.recover_tag[query] = 0
                        if self.recover_tag[query] == self.real_tag[query]:
                            recov_query += 1
                        break

                    # 计算层数 = now_level
                    gain_doc = self.injected_kws_contain_doc[query].copy() # 查询返回结果
                    now_level = 0
                    for i in range(len(self.injected_doc_level)):
                        if gain_doc[0] < self.injected_doc_level[i][0]:
                            now_level = i
                            break
                    #print(now_level)
                    # 映射到第一层
                    temp_minus = gain_doc
                    while(temp_minus[0] >= self.injected_doc_level[now_level-1][0]):
                        for i in range(len(temp_minus)):
                            temp_minus[i] -= self.injected_doc_level[now_level-1][0]
                    #print(temp_minus)
                    # 计算组数 = temp_one_group_level
                    temp_one_group_level = 0
                    temp_doc = temp_minus[0]
                    #while(temp_doc >= self.num_doc_each_group):
                    #    temp_doc -= self.num_doc_each_group
                    #    temp_one_group_level += 1
                    temp_one_group_level = int (temp_doc/self.num_doc_each_group)
                    #print(temp_one_group_level)
                    

                    kw_id = 0
                    for doc in temp_minus: # 
                        kw_id += 1 << (doc - temp_one_group_level*self.num_doc_each_group) # 计算第一层第一组
                    kw_id += temp_one_group_level*self.num_kws_each_group # 还原组
                    kw_id *= self.injected_doc_level[now_level][1]# pow(self.num_kws_each_group, now_level) # 还原层
                    self.recover_tag[query] = kw_id

                    if self.recover_tag[query] == self.real_tag[query]:
                        recov_query += 1
            
            if total_query==0:
                self.accuracy = 0.5
            else:
                self.accuracy = recov_query/total_query

            
            #print(recover_tag)
            #self.accuracy = np.mean(np.array([1 if real == predict else 0 
            #   for (real, predict) in zip(self.real_tag.values(), self.recover_tag.values())]))    
            #print(self.accuracy)
            

        """"
        zhang的文章中加入每个文档中关键字阈值后的攻击方案
        
        """
        def original_main(self):
            s = time.time()
            offset = self.original_inject_main()
            
            contain_query_doc = self.injected_doc_contain_query(offset)
            e = time.time()
            self.inject_time = e - s
            self.time = self.original_recover_main(offset, contain_query_doc)
            self.totals_inject_doc = self.num_inject_doc#len(self.first_inject_doc) + len(self.second_inject_doc)
            self.total_inject_size = min(Leakage.inject_size_threshold_counter, (self.totals_inject_doc)*self.T)
            #self.accuracy = utils.compute_accuracy(self.real_tag, self.recover_tag)
            assert self.accuracy <= 1

        def injected_doc_contain_query(self, offset):
            """"
            为模拟真实查询
            生成对于每一个查询所返回的注入文档
            相当于查询后的返回结果
            """
            injected_doc = {}
            for i_week in self.queries:
                for query in i_week:
                    if query in injected_doc.keys():
                        continue
                    temp_doc = []
                    flag = False
                    for group_id in self.first_inject_doc.keys(): # 定位到第一部分注入的位置
                        if query in self.first_inject_doc[group_id]:
                            flag = True
                            temp_doc.append(group_id)

                            for doc_id in range((math.floor((group_id - offset)/2))*self.num_FF_each_group, # 起始位置
                                min((math.floor((group_id - offset)/2) + 1)*self.num_FF_each_group, len(self.second_inject_doc))): # 终止位置
                                
                                if query in self.second_inject_doc[doc_id]:
                                    temp_doc.append(doc_id)
                        if flag:
                            break
                    if not flag:
                        temp_doc.append(-1)
                        for doc_id in range((math.floor((len(self.first_inject_doc))/2))*self.num_FF_each_group, 
                            min((math.floor((len(self.first_inject_doc))/2) + 1)*self.num_FF_each_group, len(self.second_inject_doc))):
                            if query in self.second_inject_doc[doc_id]:
                                temp_doc.append(doc_id)
                    injected_doc[query] = temp_doc
            
            return injected_doc
            
        def original_inject_main(self):
            inject_kws = []
            for i in range(len(self.chosen_kws)):
                inject_kws.append(i)
            num_inject_doc_set = math.ceil(len(inject_kws)/(self.T + self.T)) # 需要分成的文档组数

            num_each_group = math.ceil(np.log2(self.T + self.T)) # 每组需要注入的文档数


            offset = num_inject_doc_set*num_each_group # 需要分成两组注入，第一组Fi和第二组FFi之间的偏移

            self.first_inject_doc = {}

            # 第一组注入
            num_F = math.ceil(len(inject_kws)/self.T)
            for i in range(min((num_F - 1), int (Leakage.inject_size_threshold_counter/self.T))): # 最后一组关键字可不用注入
                """
                
                """
                temp_kws = []
                for j in range(i*self.T, (i+1)*self.T):
                    temp_kws.append(j)
                self.first_inject_doc[i+offset] = temp_kws
            print("self.first_inject_doc:{}".format(len(self.first_inject_doc)))
            #print("self.first_inject_doc: {}".format(self.first_inject_doc))
            #self.num_inject_doc += num_F - 1

            # 第二组注入
            num_group_FF = num_inject_doc_set # 需要分成的文档组数
            num_duplicate_doc = num_group_FF
            #print("num_group_FF{}".format(num_group_FF))
            self.num_FF_each_group = num_each_group # 每组需要注入的文档数
            #print("组数: {}".format(num_group_FF))
            #print("每组文档数: {}".format(self.num_FF_each_group))

            self.second_inject_doc = {} # 组数
            
            start_ind = 0 # 注入文档的起始id

            for i in range(min(num_group_FF, int (Leakage.inject_size_threshold_counter/self.T - num_F + 1))):
                #print(len(self.second_inject_doc))
                inject_kws_start_index = i*(self.T + self.T)
                inject_kws_end_index = (i+1)*(self.T + self.T)

                temp_num_FF_each_group = self.num_FF_each_group
                temp_end_ind = (i+1)*(self.T + self.T)
                if temp_end_ind > len(inject_kws):
                    inject_kws_end_index = len(inject_kws)
                    temp_num_FF_each_group = math.ceil(np.log2(inject_kws_end_index - inject_kws_start_index)) # 每组需要注入的文档数

                now_group_kws = inject_kws[inject_kws_start_index:inject_kws_end_index] 

                inject_doc = []
                
                for j in range(temp_num_FF_each_group):
                    inject_doc.append(start_ind)
                    start_ind += 1

                for kws_ind in range(len(now_group_kws)):          
                   
                    for num_ind in range(temp_num_FF_each_group): # 每一组注入
                        if num_ind == self.num_FF_each_group:
                            """
                            @
                            """
                            self.second_inject_doc[inject_doc[num_ind]] = self.first_inject_doc[inject_doc[num_ind] - offset]
                            continue
                        if ((kws_ind >> num_ind) & 1) == 1 :
                            if inject_doc[num_ind] in self.second_inject_doc.keys():
                                self.second_inject_doc[inject_doc[num_ind]].append(now_group_kws[kws_ind])
                            else:
                                temp_kws_l = []
                                temp_kws_l.append(now_group_kws[kws_ind])
                                self.second_inject_doc[inject_doc[num_ind]] = temp_kws_l 

            print("self.second_inject_doc: {}".format(len(self.second_inject_doc)))
            #self.num_inject_doc = num_group_FF*(self.num_FF_each_group-1)
            self.num_inject_doc = len(self.first_inject_doc) + len(self.second_inject_doc) - num_duplicate_doc
            print("self.num_inject_doc: {}".format((self.num_inject_doc)))
            return offset

        @ utils._timeit
        def original_recover_main(self, offset, contain_query_doc):
            # 恢复
            self.recover_tag = {}
            self.real_tag = {}
            total_query = 0
            recov_query = 0
            #print("contain_query_doc: {}".format(contain_query_doc))
            for i_week in self.queries:
                for query in i_week:
                    total_query += 1
                    self.real_tag[query] = query
                    self.recover_tag[query] = -1
                    kw_id = 0
                    if query not in contain_query_doc.keys():
                        self.recover_tag[query] = kw_id
                        if self.recover_tag[query] == self.real_tag[query]:
                            recov_query += 1
                        continue
                    if contain_query_doc[query][0] == -1:
                        kw_id += math.floor((len(self.first_inject_doc))/2)*self.T*2
                        for doc_id in contain_query_doc[query][1:]:
                            kw_id += 1 << (doc_id - math.floor((len(self.first_inject_doc)/2))*self.num_FF_each_group)
                    else:
                        kw_id += math.floor((contain_query_doc[query][0] - offset)/2)*self.T*2
                        for doc_id in contain_query_doc[query][1:]:
                            kw_id += 1 << (doc_id - (math.floor((contain_query_doc[query][0] - offset)/2))*self.num_FF_each_group)
                    self.recover_tag[query] = kw_id

                    if self.recover_tag[query] == self.real_tag[query]:
                        recov_query += 1
            if total_query==0:
                self.accuracy = 0.5
            else:
                self.accuracy = recov_query/total_query

            
            #self.accuracy = np.mean(np.array([1 if real == predict else 0 
            #   for (real, predict) in zip(self.real_tag.values(), self.recover_tag.values())]))

            assert self.accuracy <= 1

            #print(self.accuracy) 
        def improved_2_main(self):
            s = time.time()
            inverted_index, num_each_group, inject_doc = self.improved_inject_2_main() # 保存注入的文档id
            e = time.time()
            self.inject_time = e - s
            self.totals_inject_doc = inject_doc
            self.time = self.improved_recover_2_main(inverted_index, num_each_group)
            self.total_inject_size = self.totals_inject_doc*self.T
            #self.accuracy = utils.compute_accuracy(self.real_tag, self.recover_tag)
            assert self.accuracy <= 1
            #print("self.zhgtie{}".format(self.time))
        
        def improved_inject_2_main(self):
            inject_kws = []
            for i in range(len(self.chosen_kws)):
                inject_kws.append(i)
            num_inject_doc_set = math.ceil(len(inject_kws)/(self.T + self.T)) # 需要分成的文档组数
            inverted_index = {}
            num_each_group = math.ceil(np.log2(self.T + self.T + 1)) # 每组需要注入的文档数
            inject_doc = num_inject_doc_set*num_each_group
            for kw in inject_kws:
                temp_l = []
                group = (int) (kw / (2*self.T))
                used_kw = kw % (2*self.T) 
                for num_ind in range(num_each_group):
                    if (((used_kw + 1) >> num_ind) & 1) == 1: # 每个关键字向右偏移1，确保总有注入文档包含任一关键字
                        real_doc_ind = num_ind + group*num_each_group
                        temp_l.append(real_doc_ind)                
                inverted_index[kw] = temp_l 
            return inverted_index, num_each_group, inject_doc

        @ utils._timeit
        def improved_recover_2_main(self, inverted_index, num_each_group):   
            # recover
            self.recover_tag = {}
            self.real_tag = {}
            total_query = 0
            recov_query = 0
            #ttt = 0
            for i_week in self.queries:
                for query in i_week:
                    total_query += 1
                    self.real_tag[query] = query
                    self.recover_tag[query] = -1
                    kw_id = 0
                    group = 0
                    if query in inverted_index and len(inverted_index[query]) > 0:
                        group = (int) (inverted_index[query][0] / num_each_group)
                    if query in inverted_index:
                        for doc_ind in inverted_index[query]: # 获取返回结果中注入的文档id
                            used_doc_ind = doc_ind % num_each_group
                            kw_id += (1 << used_doc_ind) # pow(2, kws_ind)
                        kw_id -= 1
                        kw_id += group*self.T*2
                        self.recover_tag[query] = kw_id
                    #ttt += kw_id
                    if self.recover_tag[query] == self.real_tag[query]:
                        recov_query += 1
            
            self.accuracy = recov_query/total_query
        """
        zhang's binary attack, no Theshold
        zhang的文章中最开始的攻击设想，并不限制每个文档中的关键字数量，因此没个文档可以包含一半的关键字

        """
        def binary_main(self):
            inverted_index = self.binary_inject() # 保存注入的文档id
            self.totals_inject_doc = math.ceil(np.log2(len(self.chosen_kws)))
            self.total_inject_size = min(Leakage.inject_size_threshold_counter, self.totals_inject_doc*len(self.chosen_kws)/2)
            self.time = self.binary_recover(inverted_index)
            #self.accuracy = utils.compute_accuracy(self.real_tag, self.recover_tag)
            assert self.accuracy <= 1
            #print("self.zhgtie{}".format(self.time))

        def binary_inject(self):
            # suppose all the query could find the matching result
            # injection
            kws_each_doc = math.ceil(len(self.chosen_kws)/2)
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
            inverted_index = {}
            # inject_doc_kws = {}
            for kws_ind in range(len(self.chosen_kws)):
                temp_l = []
                for num_ind in range(min(num_injection_doc, (int) (Leakage.inject_size_threshold_counter/kws_each_doc))):
                    if ((kws_ind >> num_ind) & 1) == 1 :
                        temp_l.append(num_ind)                                          
                inverted_index[kws_ind] = temp_l 
            return inverted_index  

        @ utils._timeit
        def binary_recover(self, inverted_index):
                 
            # recover
            self.recover_tag = {}
            self.real_tag = {}
            #ttt = 0
            total_queries = 0
            recover_queries = 0
            for i_week in self.queries:
                for query in i_week:
                    total_queries += 1
                    self.real_tag[query] = query
                    self.recover_tag[query] = 0
                    kw_id = 0
                    if query in inverted_index.keys():
                        for kws_ind in inverted_index[query]: # 获取返回结果中注入的文档id
                            kw_id += (1 << kws_ind) # pow(2, kws_ind)
                        self.recover_tag[query] = kw_id

                    if self.real_tag[query]==self.recover_tag[query]:
                        recover_queries += 1
            if total_queries==0:
                self.accuracy = 0.5
            else:
                self.accuracy = recover_queries/total_queries
                    #ttt += kw_id
            
            #print(ttt)     
            # print("zhang_binary_search_attack'accuracy is {}".format(self.accuracy))
            # print("word_length_attack' running total time {}".format(self.running_time))
            # return self.accuracy  
        
        

    class Blackstone:

        def __init__(self, cli_doc, adv_doc, chosen_kws, queries, observed_queries, accuracy, T, kws_leak_percent):

            self.real_tag = {}
            self.recover_tag = {}

            self.recover_queries_num = 0
            self.total_queries_num = 0
            self.kws_leak_percent = kws_leak_percent
            self.total_inject_size = 0
            self.time = 0
            self.inject_time = 0
            self.cli_doc = cli_doc
            self.adv_doc = adv_doc
            self.chosen_kws = chosen_kws
            self.queries = queries
            self.observed_queries = observed_queries
            self.accuracy = accuracy
            self.T = T

        @ utils._timeit
        def word_length_attack(self):
            # 建立翻转索引，数据存储形式
            observed_word_length = {}
            adv_known_count_length = {}

            for kw_id in range(len(self.chosen_kws)):
                observed_word_length[kw_id] = 0
                for _, cli_doc_kws in enumerate(self.cli_doc):   
                    if self.chosen_kws[kw_id] in cli_doc_kws:
                        observed_word_length[kw_id] += len(cli_doc_kws)

            for kw_id in range(len(self.chosen_kws)):
                adv_known_count_length[kw_id] = 0
                for _, adv_doc_kws in enumerate(self.adv_doc):       
                    if self.chosen_kws[kw_id] in adv_doc_kws:
                        adv_known_count_length[kw_id] += len(adv_doc_kws)

            adv_known_rate = len(self.adv_doc)/len(self.cli_doc)

            cost_word_length = np.array([[(abs(observed_word_length[observed_count_ind] * adv_known_rate - adv_known_count_length[adv_count_ind])) 
                for adv_count_ind in range(len(adv_known_count_length))] 
                for observed_count_ind in range(len(observed_word_length))])

            recover_tag = {}
            real_tag = {}

            for i_week in self.queries:
                for query in i_week:
                    kw_id = np.argmin(cost_word_length[query, :])
                    recover_tag[query] = kw_id
                    real_tag[query] = query
            self.accuracy = np.mean(np.array([1 if real == predict else 0 
                for (real, predict) in zip(real_tag.values(), recover_tag.values())]))       
            print("sel_word_length_attack'accuracy is {}".format(self.accuracy))
            # print("word_length_attack' running total time {}".format(self.running_time))
            return self.accuracy  

        @ utils._timeit
        def sel_word_length_attack(self):
            # 建立翻转索引，数据存储形式
            observed_word_length = {}
            observed_volume = {}
            adv_known_word_length = {}
            adv_known_volume = {}

            adv_total_volume = 0
            adv_total_word_length = 0
            seta = 0 #advasary known average document word length

            for kw_id in range(len(self.chosen_kws)):
                observed_word_length[kw_id] = 0
                observed_volume[kw_id] = 0
                for _, cli_doc_kws in enumerate(self.cli_doc):   
                    if self.chosen_kws[kw_id] in cli_doc_kws:
                        observed_word_length[kw_id] += len(cli_doc_kws)
                        observed_volume[kw_id] += 1

                # print()
                # print(observed_word_length[kw_id])


            for kw_id in range(len(self.chosen_kws)):
                adv_known_word_length[kw_id] = 0
                adv_known_volume[kw_id] = 0
                for _, adv_doc_kws in enumerate(self.adv_doc): 
                    adv_total_volume += 1      
                    adv_total_word_length += len(adv_doc_kws)
                    if self.chosen_kws[kw_id] in adv_doc_kws:
                        adv_known_word_length[kw_id] += len(adv_doc_kws)
                        adv_known_volume[kw_id] += 1

                # print()
                # print(adv_known_count_length[kw_id])
            # window matching
            candidate_queries = {}
            adv_known_rate = len(self.adv_doc)/len(self.cli_doc)
            seta = int (adv_total_word_length/adv_total_volume)

            for i_week in self.queries:
                for query in i_week:
                    query_candidate_set = []
                    for adv_known_word_ind in range(len(adv_known_word_length)):
                        if adv_known_word_length[adv_known_word_ind] >= observed_word_length[query]*adv_known_rate - seta *100 and adv_known_word_length[adv_known_word_ind] <= observed_word_length[query]*adv_known_rate:
                        # + seta *100:
                        # if adv_known_word_length[adv_known_word_ind] >= observed_word_length[query]*adv_known_rate and adv_known_word_length[adv_known_word_ind] <= observed_word_length[query]:
                            query_candidate_set.append(adv_known_word_ind)
                    if len(query_candidate_set) == 0:
                        query_candidate_set.append(np.argmax([adv_known_word_length[ind] if adv_known_word_length[ind] < observed_word_length[query]*adv_known_rate else -1 for ind in range(len(adv_known_word_length))]))
                    candidate_queries[query] = query_candidate_set
            #print(candidate_queries)
            # filtering
            recover_tag = {}
            real_tag = {}
            # error_epsilon = 1
            #print(observed_volume)
            #print(adv_known_volume)
            for query in candidate_queries.keys():
                real_tag[query] = query
                if len(candidate_queries[query]) == 1:
                    recover_tag[query] = candidate_queries[query]
                    continue
                # first filter
                # should satisfy known_volume <= observed_volume 
                for candi_recover in candidate_queries[query]:
                    if adv_known_volume[candi_recover] > observed_volume[query]:
                        candidate_queries[query].remove(candi_recover)
                    if len(candidate_queries[query]) == 1:
                        recover_tag[query] = candidate_queries[query]
                    continue 
                #print(candidate_queries[query])

                if len(candidate_queries[query]) == 1:
                    recover_tag[query] = candidate_queries[query]
                    continue
                
                # second filter
                temp_l = {}
                for candi_recover in candidate_queries[query]:
                    baseline = observed_volume[query] - (observed_word_length[query]  - adv_known_word_length[candi_recover])/seta
                    # if baseline <= adv_known_volume[candi_recover]:
                    temp_l[candi_recover] = baseline - adv_known_volume[candi_recover]
                recover_tag[query] = min(temp_l, key=temp_l.get)

                    
            self.accuracy = np.mean(np.array([1 if real == predict else 0 
                for (real, predict) in zip(real_tag.values(), recover_tag.values())]))       
            print("word_length_attack'accuracy is {}".format(self.accuracy))
            # print("word_length_attack' running total time {}".format(self.running_time))
            return self.accuracy
        #  for kw_id, i_week in enumerate(self.queries):
        #      for kw_id in range(i_week):
        #          for doc_id in range(inverted_index[kw_id]):
        #             count[kw_id] += len(self.adv_doc[doc_id])
        #  print(self.adv_doc)
        #  print(count)        

        def subgraph_main(self, pattern):
            adv_known_each_doc_word_length, adv_known_volume, adv_known_doc_id = self.adv_precompute()
            self.time = self.subgraph_attack(adv_known_each_doc_word_length, adv_known_volume, adv_known_doc_id, pattern)
            self.accuracy = self.accuracy

        def adv_precompute(self):
            adv_known_each_doc_word_length = {}
            adv_known_volume = {}
            adv_known_doc_id = {}

            adv_total_volume = 0
            adv_total_word_length = 0
            for kw_id in range(len(self.chosen_kws)):
                adv_known_each_doc_word_length[kw_id] = []
                adv_known_volume[kw_id] = 0
                adv_known_doc_id[kw_id] = []
                for _, adv_doc_kws in enumerate(self.adv_doc): #same doc_id
                    adv_total_volume += 1      
                    adv_total_word_length += len(adv_doc_kws)
                    if self.chosen_kws[kw_id] in adv_doc_kws:
                        adv_known_each_doc_word_length[kw_id].append(len(adv_doc_kws))
                        adv_known_volume[kw_id] += 1
                        for cli_doc_id, cli_doc_kws in enumerate(self.cli_doc):
                            if cli_doc_kws == adv_doc_kws:
                                adv_known_doc_id[kw_id].append(cli_doc_id)
                                break
            return adv_known_each_doc_word_length, adv_known_volume, adv_known_doc_id
                            
                            
        @ utils._timeit
        def subgraph_attack(self, adv_known_each_doc_word_length, adv_known_volume, adv_known_doc_id, pattern):
            # 建立翻转索引，数据存储形式
            # 全部针对kws建立索引
            # doc_id统一
                observed_each_doc_word_length = {}
                observed_volume = {}
                observed_doc_id = {}
                
                seta = 0 #advasary known average document word length

                for kw_id in range(len(self.chosen_kws)):
                    observed_each_doc_word_length[kw_id] = []
                    observed_volume[kw_id] = 0
                    observed_doc_id[kw_id] = []
                    for cli_doc_id, cli_doc_kws in enumerate(self.cli_doc):   
                        if self.chosen_kws[kw_id] in cli_doc_kws:
                            observed_each_doc_word_length[kw_id].append(len(cli_doc_kws))
                            observed_volume[kw_id] += 1
                            observed_doc_id[kw_id].append(cli_doc_id)

                

                # filtering 
                # S_1
                recover_kws_to_queries = {}
                recover_tag = {}
                real_tag = {}
                candidate_set1 = {}

                for i_week in self.queries:
                    for query in i_week:

                        real_tag[query] = query

                        query_set1 = []
                        if pattern == 'access_doc_id':
                            for kw_id in adv_known_doc_id.keys():
                                flag = True
                                for doc_id in adv_known_doc_id[kw_id]:
                                    if doc_id not in observed_doc_id[query]:
                                        flag = False
                                        break
                                if flag:
                                    query_set1.append(kw_id)
                        elif pattern == 'word_length':
                            for kw_id in adv_known_each_doc_word_length.keys():
                                flag = True
                                for doc_word_length in adv_known_each_doc_word_length[kw_id]:
                                    if doc_word_length not in observed_each_doc_word_length[query]:
                                        flag = False
                                        break
                                if flag:
                                    query_set1.append(kw_id)
                        else:
                            raise ValueError("symbols not be recognized")

                        if(len(query_set1) == 1):
                            recover_tag[query] = query_set1[0]
                            recover_kws_to_queries[query_set1[0]] = query
                        else:
                            candidate_set1[query] = query_set1

                delta = len(self.adv_doc)/len(self.cli_doc) # adv. known doc. rate
                eplison_error = 300

                # S_2
                candidate_set2 = {}
                for recover_query in candidate_set1.keys():
                    query_set2 = []
                    for candidate_kw_id in candidate_set1[recover_query]:
                        if adv_known_volume[candidate_kw_id] >= observed_volume[recover_query]*delta - eplison_error:
                            query_set2.append(candidate_kw_id)
                    # sorted(query_set2, key = lambda k: len(observed_doc_id[k])
                    if(len(query_set2) == 1):
                        recover_tag[recover_query] = query_set2[0]
                        recover_kws_to_queries[query_set2[0]] = recover_query
                    else:
                        candidate_set2[recover_query] = query_set2

                # S_3
                # cross_filtering
                # this step in original paper has some questions, we improve it
                candidate_set3 = {}
                if pattern == 'access_doc_id':
                    for recover_query in candidate_set2.keys():
                        query_set3 = []
                        temp_set = candidate_set2[recover_query]
                        sorted(temp_set, key = lambda k: len(observed_doc_id[k]), reverse = True)
                        for kw_id2 in temp_set:
                            flag = True
                            for kw_id3 in query_set3:
                                if(set(observed_doc_id[kw_id2]).issubset(set(observed_doc_id[kw_id3]))):
                                    flag = False
                                    break
                            if flag:
                                query_set3.append(kw_id2)

                        if(len(query_set3) == 1):
                            recover_tag[recover_query] = query_set3[0]
                            recover_kws_to_queries[query_set3[0]] = recover_query
                        else:
                            candidate_set3[recover_query] = query_set3
                elif pattern == 'word_length':
                    for recover_query in candidate_set2.keys():
                        query_set3 = []
                        temp_set = candidate_set2[recover_query]
                        sorted(temp_set, key = lambda k: len(observed_each_doc_word_length[k]), reverse = True)
                        for kw_id2 in temp_set:
                            flag = True
                            for kw_id3 in query_set3:
                                if(set(observed_each_doc_word_length[kw_id2]).issubset(set(observed_each_doc_word_length[kw_id3]))):
                                    flag = False
                                    break
                            if flag:
                                query_set3.append(kw_id2)

                        if(len(query_set3) == 1):
                            recover_tag[recover_query] = query_set3[0]
                            recover_kws_to_queries[query_set3[0]] = recover_query
                        else:
                            candidate_set3[recover_query] = query_set3
                else:
                    raise ValueError("symbols not be recognized")
                
                # final filtering
                sorted(candidate_set3, key = lambda k: len(candidate_set3[k]))
                for query in candidate_set3.keys():
                    for kw_id in candidate_set3[query]:
                        if kw_id not in recover_kws_to_queries.keys():
                            recover_tag[query] = kw_id
                            recover_kws_to_queries[kw_id] = query
                            break
                
                # accuracy 
                success_count = 0
                for query in real_tag.keys():
                    if query in recover_tag.keys() and real_tag[query] == recover_tag[query]:
                        success_count += 1
                self.accuracy = success_count/len(real_tag)
                # print("subgraph_attack'{} accuracy is {}".format(pattern, self.accuracy))

        def improved_T_decoding_main(self):
            observed_word_length, max_observed_length = self.improved_decoding_observed_for_sim()
            offset = self.compute_decoding_offset(observed_word_length, max_observed_length)
            injected_word_length, inverted_index = self.improved_decoding_inject(offset, observed_word_length)
            self.time = self.improved_decoding_recover(offset, observed_word_length, injected_word_length, inverted_index)

            kws_each_doc = math.ceil(len(self.chosen_kws)/2)
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))

             
            self.totals_inject_doc = num_injection_doc*math.ceil(kws_each_doc/self.T)
            self.accuracy = np.mean(np.array([1 if real == predict else 0 
                for (real, predict) in zip(self.real_tag.values(), self.recover_tag.values())]))

        def improved_decoding_main(self):
            self.total_queries_num = 0
            self.recover_queries_num = 0
            s = time.time()
            observed_word_length, max_observed_length = self.decoding_observed_for_sim()
            offset = self.compute_decoding_offset(observed_word_length, max_observed_length)
            injected_word_length, inverted_index = self.improved_decoding_inject(offset, observed_word_length)
            e = time.time()
            self.inject_time = e - s
            self.time = self.improved_decoding_recover(offset, observed_word_length, injected_word_length, inverted_index)

            kws_each_doc = math.ceil(len(self.chosen_kws)/2)
            self.totals_inject_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
            self.accuracy = self.recover_queries_num/self.total_queries_num
            #self.accuracy = np.mean(np.array([1 if real == predict else 0 
            #    for (real, predict) in zip(self.real_tag.values(), self.recover_tag.values())]))

        
        def improved_decoding_inject(self, offset, observed_word_length, real_wordlength):
            kws_each_doc = math.ceil(((int) (len(self.chosen_kws)*self.kws_leak_percent))/2)
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))

            word_length_each_doc = [] # 要注入的文档word length
            if num_injection_doc >= 1:
                word_length_each_doc.append(offset)
                self.total_inject_size += word_length_each_doc[0]
            if num_injection_doc >= 2:
                for i in range(1, num_injection_doc):
                    if self.total_inject_size >= Leakage.inject_size_threshold_counter:
                        word_length_each_doc.append(0)
                        continue
                    word_length_each_doc.append(word_length_each_doc[i-1] + word_length_each_doc[i-1])
                    self.total_inject_size += word_length_each_doc[i]
           
            inverted_index = {} # 添加的word length映射到对应keyword
            for kws_ind in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
                for num_ind in range(num_injection_doc):
                    
                    if ((kws_ind >> num_ind) & 1) == 1 :
                        real_wordlength[kws_ind] += word_length_each_doc[num_ind]
         
            return inverted_index

        @ utils._timeit
        def improved_decoding_recover(self, offset, observed_word_length, injected_word_length, inverted_index):
            # 不依赖于kw_id进行恢复，因此阈值施加只影响注入文档数，不影响算法
            # recover
            self.recover_tag = {}
            self.real_tag = {}
            for i_week in self.queries:
                for query in i_week:
                    self.real_tag[query] = query
                    self.recover_tag[query] = -1
                    for kw_id in observed_word_length.keys():
                        if query in injected_word_length.keys() and (injected_word_length[query] - observed_word_length[kw_id])%offset == 0:
                        #if injected_word_length[query] - observed_word_length[kw_id] in inverted_index.keys():
                            #print(injected_word_length[query] - observed_word_length[kw_id])
                            #print(inverted_index[injected_word_length[query] - observed_word_length[kw_id]])
                            self.recover_tag[query] = (injected_word_length[query] - observed_word_length[kw_id])/offset
                            break
                    
                    if self.recover_tag[query] == -1 and query in injected_word_length.keys() and injected_word_length[query] % offset == 0:
                        self.recover_tag[query] = injected_word_length[query] / offset
                    if self.recover_tag[query] == self.real_tag[query]:
                        self.recover_queries_num += 1
                    self.total_queries_num += 1
            #print("improved decoding accuracy is {}".format(self.accuracy))
    
                
            

        def binary_main(self):
            self.time = self.binary_recover()
            self.totals_inject_doc = self.totals_inject_doc
            self.accuracy = np.mean(np.array([1 if real == predict else 0 
                for (real, predict) in zip(self.real_tag.values(), self.recover_tag.values())]))

        @ utils._timeit
        def binary_recover(self):
            """
            对手知识：关键字域
            该攻击分三个阶段：
            baseline：尽可能的观测不同word length
            target：确定一个攻击的关键字，根据其word length设定offset
            recovery：将关键字域划分两部分，一部分注入包含其全部关键字的文档，一部分无，定位一半，不断二分

            故实际上该攻击需要多个阶段观测，初始观测，然后再观测确定一个攻击目标，随后不断观测二分定位

            该攻击失败两种情况：
            一为recovery阶段的query并未在baseline阶段观测到(保留）
            二为存在相同word length，多数是不存在对应文档的query，因此返回word length均为0
            """
            # 无法达到100%恢复率
            # 相比较于decoding_attack，该注入方式即便对于高的word_length重复率，也可通过不断二分的方式可将其分在不同的集合中，
            # 因此一半集合注入，一半未注入，仍然可以将两者区分开
            # 该方式的不足之处在于，需要不断观测新的查询返回的word_length，例如，注入前需要尽可能观测到所有的查询，每次只针对一个query恢复，每次二分便需要新的注入，观测新的查询以确定其位置
            # baseline. observe the word length for each observed_query
            
            #print(len(self.chosen_kws))
            # targeting. select a keyword to inject.

            observed_word_length = {}
            max_observed_length = 0

            for i_week in self.observed_queries:
                for query in i_week:
                    #if query in observed_word_length.keys():
                    #    continue
                    observed_word_length[query] = 0
                    for _, cli_doc_kws in enumerate(self.cli_doc):   
                        if self.chosen_kws[query] in cli_doc_kws:
                            observed_word_length[query] += len(cli_doc_kws)
                    if max_observed_length < observed_word_length[query]:
                        max_observed_length = observed_word_length[query]
            #for no_doc_kw_id in range(len(self.chosen_kws)):
            #    if no_doc_kw_id not in observed_word_length.keys():
            #        observed_word_length[no_doc_kw_id] = 0
            #print(observed_word_length)
            
            self.real_tag = {}
            self.recover_tag = {}
            
            cc = 0

            self.totals_inject_doc = 0

            for i_week in self.queries:
                for query in i_week:
                    #if query in recover_tag.keys():
                    #    continue
                    #offset = 0 # generate injection offset
                    #print(observed_word_length)
                    self.real_tag[query] = query
                    # 先前并未观测到该query的word length，则添加到其中
                    if query not in observed_word_length.keys():
                        observed_word_length[query] = 0
                        for _, cli_doc_kws in enumerate(self.cli_doc):   
                            if self.chosen_kws[query] in cli_doc_kws:
                                observed_word_length[query] += len(cli_doc_kws)
                        if max_observed_length < observed_word_length[query]:
                            max_observed_length = observed_word_length[query]
                    #print("{}: {}".format(query, observed_word_length[query]))
                    #print(observed_word_length)

                    if observed_word_length[query] == 0:
                        cc += 1
                    w0_0 = list(range(len(self.chosen_kws)))[:int (len(self.chosen_kws)/2)]
                    w1_0 = list(range(len(self.chosen_kws)))[int (len(self.chosen_kws)/2):]
                    recover_flag = [False]
                    temp_observed_word_length = observed_word_length.copy()
                    

                    self.b_search(self.recover_tag, query, temp_observed_word_length[query], w0_0, w1_0, temp_observed_word_length, max_observed_length, recover_flag)
                    #print("{}: {}".format(real_tag[query], recover_tag[query]))
                # for kw_id in range(len(self.chosen_kws)):
            #print(len(real_tag))


            print("zero number is {}".format(cc))
            

        def b_search(self, recover_tag, target_query, target_word_length, w_0, w_1, temp_observed_word_length, max_observed_length, recover_flag):
            self.totals_inject_doc += math.ceil(len(w_1)/self.T)
            if(recover_flag[0]):
                return
            # generate the offset
            offset = len(w_1) # 因为对手在观测之后进行关键字划分，但并不知道具体的每个关键字对应的word length， 因此从最大word length开始
            #temp_observed_word_length = observed_word_length
            for injection_offset in range(len(w_1) + 1, sys.maxsize >> 1):
                find_flag = True
                for kw_id in w_0:
                    if kw_id in temp_observed_word_length and injection_offset == abs(target_word_length - temp_observed_word_length[kw_id]):
                        find_flag = False
                        break
                if find_flag:
                    offset = injection_offset
                    break
            #print(offset)
            # inject the doc. that contains all the kws in w_1 with word_length = offset
            for kw_id in w_1:
                if kw_id in temp_observed_word_length:
                    temp_observed_word_length[kw_id] += offset
                else:
                    temp_observed_word_length[kw_id] = offset




            for i_week in self.queries:
                for query in i_week:
                    if temp_observed_word_length[query] == target_word_length + offset:
                        #print("w_1: {}   {}".format(w_1, offset))
                        if len(w_1) == 1:
                            recover_tag[target_query] = w_1[0]
                            recover_flag[0] = True
                            return
                        else:

                            """
                            temp_max_observed_length = 0
                            for qu in temp_observed_word_length.keys():
                                if qu in w_1[int (len(w_1)/2):] and max_observed_length < temp_observed_word_length[qu]:
                                    temp_max_observed_length = temp_observed_word_length[qu]
                            """
    

                            self.b_search(recover_tag, target_query, target_word_length + offset, w_1[:int (len(w_1)/2)],  w_1[int (len(w_1)/2):], temp_observed_word_length, max_observed_length + offset, recover_flag)
                            if(recover_flag[0]):
                                return
                    elif temp_observed_word_length[query] == target_word_length:
                        #print("w_0: {}   {}".format(w_0, offset))
                        if len(w_0) == 1:
                            recover_tag[target_query] = w_0[0]
                            recover_flag[0] = True
                            return
                        else:
                            """
                            temp_max_observed_length = 0
                            for qu in temp_observed_word_length.keys():
                                if qu in w_1[int (len(w_1)/2):] and max_observed_length < temp_observed_word_length[qu]:
                                    temp_max_observed_length = temp_observed_word_length[qu]

                            """
                            
                            self.b_search(recover_tag, target_query, target_word_length, w_0[:int (len(w_0)/2)],  w_0[int (len(w_0)/2):], temp_observed_word_length, max_observed_length, recover_flag)
                            if(recover_flag[0]):
                                return
                    else:
                        continue
                if(recover_flag[0]):
                    return
            if not recover_flag[0]:
                recover_tag[target_query] = -1
                recover_flag[0] = True
                return        
     
            return



    class WL:
        def __init__(self, cli_doc, adv_doc, chosen_kws, queries, observed_queries, accuracy, T, kws_leak_percent, trend_matrix_norm, real_wordlength, real_volume, off_gama):
            self.real_tag = {}
            self.recover_tag = {}

            self.recover_queries_num = 0
            self.total_queries_num = 0
            self.kws_leak_percent = kws_leak_percent
            self.total_inject_size = 0
            self.time = 0
            self.inject_time = 0
            self.cli_doc = cli_doc
            self.adv_doc = adv_doc
            self.chosen_kws = chosen_kws
            self.queries = queries
            self.observed_queries = observed_queries
            self.accuracy = accuracy
            self.T = T  
            self.trend_matrix_norm = trend_matrix_norm
            """
            获取文档中所有关键字的真实word length，用于模拟真实世界查询获得的word length
            """
            self.real_wordlength, self.real_volume = real_wordlength, real_volume #self.get_real_wordlength_volume()
            #print(len(self.real_wordlength))
            #print(len(self.real_volume))
            """
            baseline: 尽可能多的观测word length，并统计最大word length，作为offset上限
            """
            self.observed_word_length, self.max_observed_length, self.observed_volume = self.get_baseline_observed_wordlength_volume(self.real_wordlength, self.real_volume)
            
            #print(self.observed_word_length)
            print(self.max_observed_length)
            """
            计算offset
            """

            self.offset = off_gama
            #self.offset = self.compute_decoding_offset(self.observed_word_length, self.max_observed_length)    
            print(self.offset)

        def get_real_wordlength_volume(self):
            """
            获取文档中所有关键字的真实word length，用于模拟真实世界查询获得的word length
            """
            real_wordlength = {}
            real_volume = {}
            for kw_id in range(len(self.chosen_kws)):
                real_wordlength[kw_id] = 0
                real_volume[kw_id] = 0
                for _, cli_doc_kws in enumerate(self.cli_doc):   
                    if self.chosen_kws[kw_id] in cli_doc_kws:
                        real_wordlength[kw_id] += len(cli_doc_kws)
                        real_volume[kw_id] += 1

            return real_wordlength, real_volume

        def get_baseline_observed_wordlength_volume(self, real_wordlength, real_volume):
            """
            baseline阶段word length的观测结果
            """
            observed_word_length = {}
            observed_volume = {}
            max_observed_length = 0
            for i_week in self.observed_queries:
                for query in i_week:
                    observed_word_length[query] = real_wordlength[query]
                    observed_volume[query] = real_volume[query]
                    if max_observed_length < observed_word_length[query]:
                        max_observed_length = observed_word_length[query]
            print("diff query:{}".format(len(observed_volume)))
            return observed_word_length, max_observed_length, observed_volume


        def compute_decoding_offset(self, observed_word_length, max_observed_length):          
            """
            计算offset
            """
            divided_list = {}
            for i in observed_word_length.keys():
                for j in observed_word_length.keys():
                    temp_minus = abs(observed_word_length[i] - observed_word_length[j])
                    if temp_minus!=0:
                        divided_list[temp_minus] = 0
            print(len(divided_list))
            offset = 0
            for injection_offset in range(100, max_observed_length+1):
                flag = True
                for divisor in divided_list.keys():
                    if divisor % injection_offset == 0:
                        flag = False
                        break
                
                if flag:
                    offset = injection_offset
                    break
                
            if offset == 0:
                offset = max_observed_length
            print("offset: {}".format(offset))
            return offset

        def BM2A_Freq_main(self, query_type):
            self.total_queries_num = 0
            self.recover_queries_num = 0
            self.totals_inject_doc = 0
            self.total_inject_size = 0
            self.accuracy = 0
            """
            计算baseline阶段频率
            """
            baseline_trend_matrix = np.zeros((self.trend_matrix_norm.shape[0], len(self.observed_queries))) #行kws， 列week
            if query_type == 'trend':
                for i_week, weekly_tags in enumerate(self.observed_queries):
                    if len(weekly_tags) > 0:
                        counter = Counter(weekly_tags)
                        for key in counter:
                            baseline_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)
            """
            注入
            """
            s = time.time()
            real_wordlength_after_injection = self.real_wordlength.copy()
            real_volume_after_injection = self.real_volume.copy()
            self.BM2A_Freq_inject(real_wordlength_after_injection, real_volume_after_injection)
            e = time.time()
            self.inject_time = e - s
            """
            恢复
            """
            observed_wordlength_in_baseline = self.observed_word_length.copy()
            #print(observed_wordlength_in_baseline)
            observed_volume_in_baseline = self.observed_volume.copy()
            self.time = self.BM2A_Freq_recover(query_type, baseline_trend_matrix, observed_wordlength_in_baseline, observed_volume_in_baseline, real_wordlength_after_injection, real_volume_after_injection)
            
            kws_each_doc = math.ceil(len(self.chosen_kws)/2)
            self.totals_inject_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
            self.accuracy = self.recover_queries_num/self.total_queries_num
            #assert self.accuracy <= 1 or (self.real_tag.keys() != observed_word_length.keys())

                    
        @ utils._timeit
        def BM2A_Freq_recover(self, query_type, baseline_trend_matrix, observed_wordlength_in_baseline, observed_volume_in_baseline, real_wordlength_after_injection, real_volume_after_injection):
            self.real_tag = {}
            self.recover_tag = {}
            """
            计算频率
            """
            if query_type == 'trend':
                recover_trend_matrix = np.zeros((self.trend_matrix_norm.shape[0], self.trend_matrix_norm.shape[1]))
                for i_week, weekly_tags in enumerate(self.queries):
                    if len(weekly_tags) > 0:
                        counter = Counter(weekly_tags)
                        for key in counter:
                            recover_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)
                
                                        
            t_w_k = 0
            tototot = 0
            for i_week in self.queries:
                t_w_k += 1
                for query in i_week:
                    min_cost = 10000
                    real_key = 1
                    #print(real_wordlength_after_injection[query])
                    self.real_tag[query] = query
                    self.recover_tag[query] = -1
                    #candi_kws = []
                    #print(len(observed_wordlength_in_baseline))
                    #print(observed_wordlength_in_baseline)
                    for kw_id in observed_wordlength_in_baseline.keys():
                        if query in real_wordlength_after_injection.keys(): 
                            if (real_wordlength_after_injection[query] - observed_wordlength_in_baseline[kw_id]) >= 0 and (real_wordlength_after_injection[query] - observed_wordlength_in_baseline[kw_id]) < len(self.chosen_kws)*1:
                                #recover_tag[query] = kw_id
                                if (real_wordlength_after_injection[query] - observed_wordlength_in_baseline[kw_id])%1==0:
                                    re_kw_id = (int) ((real_wordlength_after_injection[query] - observed_wordlength_in_baseline[kw_id])/1)
                                    if query_type == 'random':
                                        real_key = re_kw_id
                                        #candi_kws.append(re_kw_id)
                                        break
                                        
                                    else:
                                        tmp = np.linalg.norm([[recover_trend_matrix[query][t] - baseline_trend_matrix[re_kw_id][t2] for t2 in range(baseline_trend_matrix.shape[1])] for t in range(recover_trend_matrix.shape[1])])#np.linalg.norm(
                                        if tmp < min_cost:
                                            min_cost = tmp
                                            real_key = re_kw_id
                    #if query_type == 'random':
                        #print(query)
                        #print(candi_kws)
                        #print(real_key)
                        #real_key = np.random.choice(candi_kws)
                        #if len(candi_kws) == 1:
                        #    tototot += 1
                    
                    self.recover_tag[query] = real_key
                    if self.recover_tag[query] == self.real_tag[query]:
                        self.recover_queries_num += 1
                    self.total_queries_num += 1
            print(tototot)

        def BM2A_Freq_inject(self, real_wordlength_after_injection, real_volume_after_injection):
            """
            对手知识：所有关键字域
            对手观测：对手尽可能的观测每次查询的word length
            该攻击分为两个阶段：
            baseline：观测所有word length，计算offset，对每个关键字根据offset注入对应word length的文档
            recovery：根据注入前word length和注入后要恢复的query的word length进行恢复

            该攻击失败的唯一情况：recovery阶段的query并未在baseline阶段观测到，并且其真实word length不为0
            """
           
            """
            每个文档包含关键字数
            注入文档数
            """
            
            return self.BM2A_Vol_inject(real_wordlength_after_injection, real_volume_after_injection)

        def padding_binary_main(self):
            inverted_index = self.padding_binary_inject() # 保存注入的文档id
            self.totals_inject_doc = math.ceil(np.log2(len(self.chosen_kws)))
            self.total_inject_size = min(Leakage.inject_size_threshold_counter, self.totals_inject_doc*len(self.chosen_kws)/2)
            self.time = self.padding_binary_recover(inverted_index)
            #self.accuracy = utils.compute_accuracy(self.real_tag, self.recover_tag)
            assert self.accuracy <= 1
            #print("self.zhgtie{}".format(self.time))

        def padding_binary_inject(self):
            # suppose all the query could find the matching result
            # injection
            kws_each_doc = math.ceil(len(self.chosen_kws)/2)
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
            inverted_index = {}
            import pickle
            with open(r'C:\Users\admin\yan\Attack-evaluation\datasets_pro\real_data\Lucene\lucene_obfus_after_padding_2.pkl', "rb") as f:
                obfus = pickle.load(f) #840601
                f.close()
            inject_doc_kws = {}
            for kws_ind in range(len(self.chosen_kws)):
                temp_l = []
                #temp_l = obfus[kws_ind]
                for ij in obfus[kws_ind]:
                    if random.randint(0,1)==0:
                        temp_l.append(ij)
                #random.choice()
                #for t in obfus[kws_ind]:
                #    temp_l.append(t+1)
                for num_ind in range(min(num_injection_doc, (int) (Leakage.inject_size_threshold_counter/kws_each_doc))):
                    if ((kws_ind >> num_ind) & 1) == 1 :
                        if num_ind not in temp_l:
                            if random.randint(0,1)==0:
                                temp_l.append(num_ind)                                          
                inverted_index[kws_ind] = temp_l 
            return inverted_index  

        @ utils._timeit
        def padding_binary_recover(self, inverted_index):
                 
            # recover
            self.recover_tag = {}
            self.real_tag = {}
            #ttt = 0
            total_queries = 0
            recover_queries = 0
            for i_week in self.queries:
                for query in i_week:
                    total_queries += 1
                    self.real_tag[query] = query
                    self.recover_tag[query] = 0
                    kw_id = 0
                    if query in inverted_index.keys():
                        for kws_ind in inverted_index[query]: # 获取返回结果中注入的文档id
                            kw_id += (1 << kws_ind) # pow(2, kws_ind)
                        self.recover_tag[query] = kw_id

                    if self.real_tag[query]==self.recover_tag[query]:
                        recover_queries += 1
            if total_queries==0:
                self.accuracy = 0.5
            else:
                self.accuracy = recover_queries/total_queries
                    #ttt += kw_id
            
            #print(ttt)     
            # print("zhang_binary_search_attack'accuracy is {}".format(self.accuracy))
            # print("word_length_attack' running total time {}".format(self.running_time))
            # return self.accuracy  

        def BM2A_main(self, query_type):
            self.total_queries_num = 0
            self.recover_queries_num = 0
            self.totals_inject_doc = 0
            self.total_inject_size = 0
            self.accuracy = 0
            """
            计算baseline阶段频率
            """
            baseline_trend_matrix = np.zeros((self.trend_matrix_norm.shape[0], len(self.observed_queries))) #行kws， 列week
            if query_type == 'trend':
                for i_week, weekly_tags in enumerate(self.observed_queries):
                    if len(weekly_tags) > 0:
                        counter = Counter(weekly_tags)
                        for key in counter:
                            baseline_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)

            """
            注入
            """
            s = time.time()
            real_wordlength_after_injection = self.real_wordlength.copy()
            real_volume_after_injection = self.real_volume.copy()
            self.BM2A_inject(real_wordlength_after_injection, real_volume_after_injection)
            e = time.time()
            self.inject_time = e - s
            """
            恢复
            """
            observed_wordlength_in_baseline = self.observed_word_length.copy()
            observed_volume_in_baseline = self.observed_volume.copy()
            self.time = self.BM2A_recover(query_type, baseline_trend_matrix, observed_wordlength_in_baseline, observed_volume_in_baseline, real_wordlength_after_injection, real_volume_after_injection)
            
            kws_each_doc = math.ceil(len(self.chosen_kws)/2)
            self.totals_inject_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
            self.accuracy = self.recover_queries_num/self.total_queries_num
            #assert self.accuracy <= 1 or (self.real_tag.keys() != observed_word_length.keys())

                    
        @ utils._timeit
        def BM2A_recover(self, query_type, baseline_trend_matrix, observed_wordlength_in_baseline, observed_volume_in_baseline, real_wordlength_after_injection, real_volume_after_injection):
            self.real_tag = {}
            self.recover_tag = {}
            """
            计算频率
            """
            if query_type == 'trend':
                recover_trend_matrix = np.zeros((self.trend_matrix_norm.shape[0], self.trend_matrix_norm.shape[1]))
                for i_week, weekly_tags in enumerate(self.queries):
                    if len(weekly_tags) > 0:
                        counter = Counter(weekly_tags)
                        for key in counter:
                            recover_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)
                
            for i_week in self.queries:
                for query in i_week:
                    #print(real_wordlength_after_injection[query])
                    self.real_tag[query] = query
                    self.recover_tag[query] = -1
                    CA = []
                    for kw_id in observed_wordlength_in_baseline.keys():
                        if query in real_wordlength_after_injection.keys(): 
                            if (real_wordlength_after_injection[query] - observed_wordlength_in_baseline[kw_id]) >= 0 and (real_wordlength_after_injection[query] - observed_wordlength_in_baseline[kw_id]) < len(self.chosen_kws)*1:
                                #recover_tag[query] = kw_id
                                if (real_wordlength_after_injection[query] - observed_wordlength_in_baseline[kw_id])%1==0:
                                    re_kw_id = (int) ((real_wordlength_after_injection[query] - observed_wordlength_in_baseline[kw_id])/1)
                                    tmp_kw_id = re_kw_id
                                    ttt = 0
                                    while tmp_kw_id!=0:
                                        if tmp_kw_id&1==1:
                                            ttt += 1
                                        tmp_kw_id >>= 1
                                    if real_volume_after_injection[query] - observed_volume_in_baseline[kw_id] == ttt:
                                        #print(query)
                                        #print(re_kw_id)
                                        #print(kw_id)
                                        if query_type == 'random':
                                            self.recover_tag[query] = re_kw_id
                                            break
                                        CA.append(re_kw_id)
                    
                    if query_type == 'trend':
                        min_cost =1000
                        real_key = -1
                        for kw in CA:
                            tmp = np.linalg.norm([[recover_trend_matrix[query][t] - baseline_trend_matrix[kw][t2] for t2 in range(baseline_trend_matrix.shape[1])] for t in range(recover_trend_matrix.shape[1])])#np.linalg.norm(
                            if tmp < min_cost:
                                min_cost = tmp
                                real_key = kw
                        self.recover_tag[query] = real_key
                    if self.recover_tag[query] == self.real_tag[query]:
                        self.recover_queries_num += 1
                    self.total_queries_num += 1


        def BM2A_inject(self, real_wordlength_after_injection, real_volume_after_injection):
            """
            对手知识：所有关键字域
            对手观测：对手尽可能的观测每次查询的word length
            该攻击分为两个阶段：
            baseline：观测所有word length，计算offset，对每个关键字根据offset注入对应word length的文档
            recovery：根据注入前word length和注入后要恢复的query的word length进行恢复

            该攻击失败的唯一情况：recovery阶段的query并未在baseline阶段观测到，并且其真实word length不为0
            """
           
            """
            每个文档包含关键字数
            注入文档数
            """
            kws_each_doc = math.ceil(((int) (len(self.chosen_kws)*self.kws_leak_percent))/2)
            if kws_each_doc==0:
                num_injection_doc=0
            else:
                num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
            """
            生成注入文档
            """
            word_length_each_doc = [] # 要注入的文档word length
            if num_injection_doc >= 1:
                word_length_each_doc.append(1)
                self.total_inject_size += word_length_each_doc[0]
                self.total_inject_size += math.ceil(len(self.chosen_kws)/2)
            if num_injection_doc >= 2:
                for i in range(1, num_injection_doc):
                    if self.total_inject_size >= Leakage.inject_size_threshold_counter:
                        self.total_inject_size = Leakage.inject_size_threshold_counter
                        word_length_each_doc.append(0)
                        continue
                    word_length_each_doc.append(word_length_each_doc[i-1] + word_length_each_doc[i-1])
                    self.total_inject_size += word_length_each_doc[i]
                    self.total_inject_size += math.ceil(len(self.chosen_kws)/2)
            """
            统计
            """
            for kws_ind in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
                for num_ind in range(num_injection_doc):
                    if ((kws_ind >> num_ind) & 1) == 1 :
                        real_wordlength_after_injection[kws_ind] += word_length_each_doc[num_ind]
                        real_volume_after_injection[kws_ind] += 1
   
            return real_wordlength_after_injection, real_volume_after_injection

        def BM2A_Vol_main(self):
            self.total_queries_num = 0
            self.recover_queries_num = 0
            self.totals_inject_doc = 0
            self.total_inject_size = 0
            self.accuracy = 0

            """
            注入
            """
            s = time.time()
            real_wordlength_after_injection = self.real_wordlength.copy()
            real_volume_after_injection = self.real_volume.copy()
            self.BM2A_Vol_inject(real_wordlength_after_injection, real_volume_after_injection)
            e = time.time()
            self.inject_time = e - s
            """
            恢复
            """
            observed_wordlength_in_baseline = self.observed_word_length.copy()
            observed_volume_in_baseline = self.observed_volume.copy()
            self.time = self.BM2A_Vol_recover(observed_wordlength_in_baseline, observed_volume_in_baseline, real_wordlength_after_injection, real_volume_after_injection)
            
            kws_each_doc = math.ceil(len(self.chosen_kws)/2)
            self.totals_inject_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
            self.accuracy = self.recover_queries_num/self.total_queries_num
            #assert self.accuracy <= 1 or (self.real_tag.keys() != observed_word_length.keys())

                    
        @ utils._timeit
        def BM2A_Vol_recover(self, observed_wordlength_in_baseline, observed_volume_in_baseline, real_wordlength_after_injection, real_volume_after_injection):
            self.real_tag = {}
            self.recover_tag = {}

            for i_week in self.queries:
                for query in i_week:
                    #print(real_wordlength_after_injection[query])
                    self.real_tag[query] = query
                    self.recover_tag[query] = -1
                    for kw_id in observed_wordlength_in_baseline.keys():
                        if query in real_wordlength_after_injection.keys(): 
                            if (real_wordlength_after_injection[query] - observed_wordlength_in_baseline[kw_id]) >= 0 and (real_wordlength_after_injection[query] - observed_wordlength_in_baseline[kw_id]) < len(self.chosen_kws)*1:
                                #recover_tag[query] = kw_id
                                if (real_wordlength_after_injection[query] - observed_wordlength_in_baseline[kw_id])%1==0:
                                    re_kw_id = (int) ((real_wordlength_after_injection[query] - observed_wordlength_in_baseline[kw_id])/1)
                                    tmp_kw_id = re_kw_id
                                    ttt = 0
                                    while tmp_kw_id!=0:
                                        if tmp_kw_id&1==1:
                                            ttt += 1
                                        tmp_kw_id >>= 1
                                    if real_volume_after_injection[query] - observed_volume_in_baseline[kw_id] == ttt:
                                        #print(query)
                                        #print(re_kw_id)
                                        #print(kw_id)
                                        self.recover_tag[query] = re_kw_id
                                        break
                    if self.recover_tag[query] == self.real_tag[query]:
                        self.recover_queries_num += 1
                    self.total_queries_num += 1


        def BM2A_Vol_inject(self, real_wordlength_after_injection, real_volume_after_injection):
            """
            对手知识：所有关键字域
            对手观测：对手尽可能的观测每次查询的word length
            该攻击分为两个阶段：
            baseline：观测所有word length，计算offset，对每个关键字根据offset注入对应word length的文档
            recovery：根据注入前word length和注入后要恢复的query的word length进行恢复

            该攻击失败的唯一情况：recovery阶段的query并未在baseline阶段观测到，并且其真实word length不为0
            """
           
            """
            每个文档包含关键字数
            注入文档数
            """
            kws_each_doc = math.ceil(((int) (len(self.chosen_kws)*self.kws_leak_percent))/2)
            if kws_each_doc==0:
                num_injection_doc=0
            else:
                num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
            """
            生成注入文档
            """
            word_length_each_doc = [] # 要注入的文档word length
            if num_injection_doc >= 1:
                word_length_each_doc.append(1)
                self.total_inject_size += word_length_each_doc[0]
                self.total_inject_size += math.ceil(len(self.chosen_kws)/2)
            if num_injection_doc >= 2:
                for i in range(1, num_injection_doc):
                    if self.total_inject_size >= Leakage.inject_size_threshold_counter:
                        self.total_inject_size = Leakage.inject_size_threshold_counter
                        word_length_each_doc.append(0)
                        continue
                    word_length_each_doc.append(word_length_each_doc[i-1] + word_length_each_doc[i-1])
                    self.total_inject_size += word_length_each_doc[i]
                    self.total_inject_size += math.ceil(len(self.chosen_kws)/2)
            """
            统计
            """
            for kws_ind in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
                for num_ind in range(num_injection_doc):
                    if ((kws_ind >> num_ind) & 1) == 1 :
                        real_wordlength_after_injection[kws_ind] += word_length_each_doc[num_ind]
                        real_volume_after_injection[kws_ind] += 1
   
            return real_wordlength_after_injection, real_volume_after_injection

        def BDA_main(self):
            self.total_queries_num = 0
            self.recover_queries_num = 0
            self.totals_inject_doc = 0
            self.total_inject_size = 0
            self.accuracy = 0

            """
            注入
            """
            s = time.time()
            real_wordlength_after_injection = self.real_wordlength.copy()
            self.BDA_inject(real_wordlength_after_injection)
            e = time.time()
            self.inject_time = e - s
            """
            恢复
            """
            observed_wordlength_in_baseline = self.observed_word_length.copy()
            self.time = self.BDA_recover(observed_wordlength_in_baseline, real_wordlength_after_injection)
            
            kws_each_doc = math.ceil(len(self.chosen_kws)/2)
            self.totals_inject_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
            self.accuracy = self.recover_queries_num/self.total_queries_num
            #assert self.accuracy <= 1 or (self.real_tag.keys() != observed_word_length.keys())

        def BDA_recover(self, observed_word_length_in_baseline, real_wordlength_after_injection):
            self.DA_recover(observed_word_length_in_baseline, real_wordlength_after_injection)
        def BDA_inject(self, real_wordlength_after_injection):
            """
            每个文档包含关键字数
            注入文档数
            """
            kws_each_doc = math.ceil(((int) (len(self.chosen_kws)*self.kws_leak_percent))/2)
            if kws_each_doc==0:
                num_injection_doc=0
            else:
                num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
            """
            生成注入文档
            """
            word_length_each_doc = [] # 要注入的文档word length
            if num_injection_doc >= 1:
                word_length_each_doc.append(self.offset)
                self.total_inject_size += word_length_each_doc[0]
            if num_injection_doc >= 2:
                for i in range(1, num_injection_doc):
                    if self.total_inject_size >= Leakage.inject_size_threshold_counter:
                        self.total_inject_size = Leakage.inject_size_threshold_counter
                        word_length_each_doc.append(0)
                        continue
                    word_length_each_doc.append(word_length_each_doc[i-1] + word_length_each_doc[i-1])
                    self.total_inject_size += word_length_each_doc[i]
            """
            统计注入后word length结果
            """
            for kws_ind in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
                for num_ind in range(num_injection_doc):
                    if ((kws_ind >> num_ind) & 1) == 1 :
                        real_wordlength_after_injection[kws_ind] += word_length_each_doc[num_ind]       

        def DA_main(self):
            self.total_queries_num = 0
            self.recover_queries_num = 0
            self.totals_inject_doc = 0
            self.total_inject_size = 0
            self.accuracy = 0
            """
            注入
            """
            s = time.time()
            real_wordlength_after_injection = self.real_wordlength.copy()
            self.DA_inject(real_wordlength_after_injection)
            e = time.time()
            self.inject_time = e - s
            """
            恢复
            """
            observed_wordlength_in_baseline = self.observed_word_length.copy()
            self.time = self.DA_recover(observed_wordlength_in_baseline, real_wordlength_after_injection)
            
            #self.ti
            self.totals_inject_doc = (int) (len(self.chosen_kws)*self.kws_leak_percent) - 1
            self.accuracy = self.recover_queries_num/self.total_queries_num
            #assert self.accuracy <= 1 or (self.real_tag.keys() != observed_word_length.keys())

        @ utils._timeit
        def DA_recover(self, observed_word_length_in_baseline, real_wordlength_after_injection):
            self.real_tag = {}
            self.recover_tag = {}

            for i_week in self.queries:
                for query in i_week:
                    self.real_tag[query] = query
                    self.recover_tag[query] = -1
                    for kw_id in observed_word_length_in_baseline.keys():
                        if query in real_wordlength_after_injection.keys(): 
                            if (real_wordlength_after_injection[query] - observed_word_length_in_baseline[kw_id]) % self.offset == 0:
                                self.recover_tag[query] = (real_wordlength_after_injection[query] - observed_word_length_in_baseline[kw_id]) / self.offset
                                break
                    if self.recover_tag[query] == self.real_tag[query]:
                        self.recover_queries_num += 1
                    self.total_queries_num += 1


        def DA_inject(self, real_wordlength_after_injection):
            """
            对手知识：所有关键字域
            对手观测：对手尽可能的观测每次查询的word length
            该攻击分为两个阶段：
            baseline：观测所有word length，计算offset，对每个关键字根据offset注入对应word length的文档
            recovery：根据注入前word length和注入后要恢复的query的word length进行恢复

            该攻击失败的唯一情况：recovery阶段的query并未在baseline阶段观测到
            """
        
            """
            注入部分，统计注入之后的word length
            """
            for kw_id in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
                if self.total_inject_size >= Leakage.inject_size_threshold_counter:
                    self.total_inject_size = Leakage.inject_size_threshold_counter
                    break
                self.total_inject_size += kw_id*self.offset 
                real_wordlength_after_injection[kw_id] += kw_id*self.offset

        

    class WLP:
        def __init__(self, queries, observed_queries, chosen_kws, cli_doc, adv_doc, trend_matrix_norm, adv_trend_matrix_norm, kws_leak_percent):
            self.queries = queries
            self.observed_queries = observed_queries
            self.chosen_kws = chosen_kws
            self.cli_doc = cli_doc
            self.adv_doc = adv_doc
            self.trend_matrix_norm = trend_matrix_norm
            self.adv_trend_matrix_norm = adv_trend_matrix_norm
            self.accuracy = 0
            self.recover_queries_num = 0
            self.total_queries_num = 0
            self.time = 0
            self.inject_time = 0
            self.total_inject_size = 0
            self.kws_leak_percent = kws_leak_percent

        def wlp_vol_main(self):
            self.total_queries_num = 0
            self.recover_queries_num = 0
            self.totals_inject_doc = 0
            self.total_inject_size = 0
            s = time.time()
            observed_word_length, injected_word_length, observed_volume, injected_volume = self.wlp_vol_inject()
            e = time.time()
            self.inject_time = e-s
            self.time = self.wlp_vol_recover(observed_word_length, injected_word_length, observed_volume, injected_volume)
            
            self.totals_inject_doc = math.ceil(math.log2(len(self.chosen_kws)))
            #self.accuracy = np.mean(np.array([1 if real == predict else 0 
            #    for (real, predict) in zip(self.real_tag.values(), self.recover_tag.values())]))
            self.accuracy = self.recover_queries_num/self.total_queries_num
            #self.total_inject_size = (len(self.chosen_kws)-1)*len(self.chosen_kws)*1
            #assert self.accuracy <= 1 or (self.real_tag.keys() != observed_word_length.keys())
        def wlp_freq_main(self):
            self.total_queries_num = 0
            self.recover_queries_num = 0
            self.totals_inject_doc = 0
            self.total_inject_size = 0
            s = time.time()
            observed_word_length, injected_word_length, baseline_trend_matrix = self.wlp_freq_inject()
            e = time.time()
            self.inject_time = e-s
            self.time = self.wlp_freq_recover(observed_word_length, injected_word_length, baseline_trend_matrix)
            
            self.totals_inject_doc = math.ceil(math.log2(len(self.chosen_kws)))
            self.accuracy = self.recover_queries_num/self.total_queries_num
            
           
            
        @ utils._timeit
        def wlp_freq_recover(self, observed_word_length, injected_word_length, baseline_trend_matrix):
            self.real_tag = {}
            self.recover_tag = {}
            
            #print(observed_word_length)

            # 频率分析
            
            recover_trend_matrix = np.zeros((self.trend_matrix_norm.shape[0], self.trend_matrix_norm.shape[1]))
            for i_week, weekly_tags in enumerate(self.queries):
                if len(weekly_tags) > 0:
                    counter = Counter(weekly_tags)
                    for key in counter:
                        recover_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)
            t_w_k = 0
            for i_week in self.queries:
                t_w_k += 1
                for query in i_week:
                    #if query in self.real_tag.keys():
                    #    continue
                    #candidate_kws = []
                    self.real_tag[query] = query
                    self.recover_tag[query] = -1
                    
                    min_cost = 1
                    real_key = -1

                    for kw_id in observed_word_length.keys():
                        if query in injected_word_length.keys(): 
                            if (injected_word_length[query] - observed_word_length[kw_id]) >= 0 and (injected_word_length[query] - observed_word_length[kw_id]) < len(self.chosen_kws)*1:
                                #recover_tag[query] = kw_id
                                if (injected_word_length[query] - observed_word_length[kw_id])%1==0:
                                    re_kw_id = (int) ((injected_word_length[query] - observed_word_length[kw_id])/1)
                                    #candidate_kws.append(re_kw_id)
                                    #print(recover_trend_matrix[query])
                                    #real_key = re_kw_id
                                    
                                    tmp = np.linalg.norm([recover_trend_matrix[query][t] - baseline_trend_matrix[re_kw_id][0] for t in range(t_w_k-1, min(t_w_k+9, len(self.queries)))])#np.linalg.norm(
                                    if tmp < min_cost:
                                        min_cost = tmp
                                        real_key = re_kw_id
                                    
                                    
                    #print(len(candidate_kws))
                    # 频率分析
                    
                    self.recover_tag[query] = real_key
                    
                    if self.recover_tag[query] == -1 and query in injected_word_length.keys() and injected_word_length[query] >= 0 and injected_word_length[query] < len(self.chosen_kws)*1:
                        self.recover_tag[query] = (int) (injected_word_length[query]/1)
                                           
                    if self.recover_tag[query] == self.real_tag[query]:
                        self.recover_queries_num += 1
                    self.total_queries_num += 1

        def wlp_freq_inject(self):
            """
            对手知识：所有关键字域
            对手观测：对手尽可能的观测每次查询的word length
            该攻击分为两个阶段：
            baseline：观测所有word length，计算offset，对每个关键字根据offset注入对应word length的文档
            recovery：根据注入前word length和注入后要恢复的query的word length进行恢复

            该攻击失败的唯一情况：recovery阶段的query并未在baseline阶段观测到，并且其真实word length不为0
            """
            # 计算baseline观测到的freq
            baseline_trend_matrix = np.zeros((self.trend_matrix_norm.shape[0], len(self.observed_queries)))
            for i_week, weekly_tags in enumerate(self.observed_queries):
                if len(weekly_tags) > 0:
                    counter = Counter(weekly_tags)
                    for key in counter:
                        baseline_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)
            # 注入
            observed_word_length = {}
            #print(len(baseline_trend_matrix))
            #print(len(baseline_trend_matrix[0]))
            #print((baseline_trend_matrix[0]))
            for i_week in self.observed_queries:
                for query in i_week:
                    if query in observed_word_length.keys():
                        continue
                    observed_word_length[query] = 0
                    
                    #total_doc = 0
                    for _, cli_doc_kws in enumerate(self.cli_doc):  
                        #total_doc += len(cli_doc_kws)
                        if self.chosen_kws[query] in cli_doc_kws:
                            observed_word_length[query] += len(cli_doc_kws)



            kws_each_doc = math.ceil(((int) (len(self.chosen_kws)*self.kws_leak_percent))/2)
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))

            word_length_each_doc = [] # 要注入的文档word length
            if num_injection_doc >= 1:
                word_length_each_doc.append(1)
                self.total_inject_size += word_length_each_doc[0]
                self.total_inject_size += math.ceil(len(self.chosen_kws)/2)
            if num_injection_doc >= 2:
                for i in range(1, num_injection_doc):
                    if self.total_inject_size >= Leakage.inject_size_threshold_counter:
                        word_length_each_doc.append(0)
                        continue
                    word_length_each_doc.append(word_length_each_doc[i-1] + word_length_each_doc[i-1])
                    self.total_inject_size += word_length_each_doc[i]
                    self.total_inject_size += math.ceil(len(self.chosen_kws)/2)

            injected_word_length = {}
            for kw_id in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
                if kw_id in observed_word_length.keys():
                    injected_word_length[kw_id] = observed_word_length[kw_id]
                else: # 不影响本实验结果，为模拟需要加上
                    injected_word_length[kw_id] = 0
           
            inverted_index = {} # 添加的word length映射到对应keyword
            for kws_ind in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
                for num_ind in range(num_injection_doc):
                    if ((kws_ind >> num_ind) & 1) == 1 :
                        injected_word_length[kws_ind] += word_length_each_doc[num_ind]
         


            
            """
            print("len(diff wl): {}".format(len(observed_word_length)))
            injected_word_length = {}
            for kw_id in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
                if kw_id in observed_word_length.keys():
                    injected_word_length[kw_id] = observed_word_length[kw_id] + kw_id*1
                else: # 不影响本实验结果，为模拟需要加上
                    injected_word_length[kw_id] = kw_id*1
              
            """                
             
            return observed_word_length, injected_word_length, baseline_trend_matrix
            
        @ utils._timeit
        def wlp_vol_recover(self, observed_word_length, injected_word_length, observed_volume, injected_volume):
            self.real_tag = {}
            self.recover_tag = {}

            for i_week in self.queries:
                for query in i_week:
                    #print(injected_word_length[query])
                    #if query in self.real_tag.keys():
                    #    continue
                    #candidate_kws = []
                    self.real_tag[query] = query
                    self.recover_tag[query] = -1
                    for kw_id in observed_word_length.keys():
                        if query in injected_word_length.keys(): 
                            if (injected_word_length[query] - observed_word_length[kw_id]) >= 0 and (injected_word_length[query] - observed_word_length[kw_id]) < len(self.chosen_kws)*1:
                                #recover_tag[query] = kw_id
                                if (injected_word_length[query] - observed_word_length[kw_id])%1==0:
                                    re_kw_id = (int) ((injected_word_length[query] - observed_word_length[kw_id])/1)
                                    tmp_kw_id = re_kw_id
                                    ttt = 0
                                    while tmp_kw_id!=0:
                                        if tmp_kw_id&1==1:
                                            ttt += 1
                                        tmp_kw_id >>= 1
                                    if injected_volume[query] - observed_volume[kw_id] == ttt:
                                        self.recover_tag[query] = re_kw_id
                                        break
        
                    #if self.recover_tag[query] == -1 and query in injected_word_length.keys() and injected_word_length[query] >= 0 and injected_word_length[query] < len(self.chosen_kws)*1:
                    #    self.recover_tag[query] = (int) (injected_word_length[query]/1)
                                           
                    if self.recover_tag[query] == self.real_tag[query]:
                        self.recover_queries_num += 1
                    self.total_queries_num += 1


        def wlp_vol_inject(self):
            """
            对手知识：所有关键字域
            对手观测：对手尽可能的观测每次查询的word length
            该攻击分为两个阶段：
            baseline：观测所有word length，计算offset，对每个关键字根据offset注入对应word length的文档
            recovery：根据注入前word length和注入后要恢复的query的word length进行恢复

            该攻击失败的唯一情况：recovery阶段的query并未在baseline阶段观测到，并且其真实word length不为0
            """
            # 注入
            observed_word_length = {}
            observed_volume = {}

            for i_week in self.observed_queries:
                for query in i_week:
                    if query in observed_word_length.keys():
                        continue
                    observed_word_length[query] = 0
                    observed_volume[query] = 0
                    #total_doc = 0
                    for _, cli_doc_kws in enumerate(self.cli_doc):  
                        #total_doc += len(cli_doc_kws)
                        if self.chosen_kws[query] in cli_doc_kws:
                            observed_word_length[query] += len(cli_doc_kws)
                            observed_volume[query] += 1
            print("len(diff wl): {}".format(len(observed_word_length)))
            #print(total_doc/len(self.cli_doc))
            #print(observed_volume)
            injected_word_length = {}
            injected_volume = {}


            #pr
            
            kws_each_doc = math.ceil(((int) (len(self.chosen_kws)*self.kws_leak_percent))/2)
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))

            word_length_each_doc = [] # 要注入的文档word length
            if num_injection_doc >= 1:
                word_length_each_doc.append(1)
                self.total_inject_size += word_length_each_doc[0]
                self.total_inject_size += math.ceil(len(self.chosen_kws)/2)
            if num_injection_doc >= 2:
                for i in range(1, num_injection_doc):
                    if self.total_inject_size >= Leakage.inject_size_threshold_counter:
                        word_length_each_doc.append(0)
                        continue
                    word_length_each_doc.append(word_length_each_doc[i-1] + word_length_each_doc[i-1])
                    self.total_inject_size += word_length_each_doc[i]
                    self.total_inject_size += math.ceil(len(self.chosen_kws)/2)

            injected_word_length = {}
            for kw_id in range((int) (len(self.chosen_kws))):
                if kw_id in observed_word_length.keys():
                    injected_word_length[kw_id] = observed_word_length[kw_id]
                    injected_volume[kw_id] = observed_volume[kw_id]
                else: # 不影响本实验结果，为模拟需要加上
                    injected_word_length[kw_id] = 0
                    injected_volume[kw_id] = 0
           
            for kws_ind in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
                for num_ind in range(num_injection_doc):
                    if ((kws_ind >> num_ind) & 1) == 1 :
                        injected_word_length[kws_ind] += word_length_each_doc[num_ind]
                        injected_volume[kws_ind] += 1
   
            return observed_word_length, injected_word_length, observed_volume, injected_volume

        def wlp_vol_inject2(self):
            """
            对手知识：所有关键字域
            对手观测：对手尽可能的观测每次查询的word length
            该攻击分为两个阶段：
            baseline：观测所有word length，计算offset，对每个关键字根据offset注入对应word length的文档
            recovery：根据注入前word length和注入后要恢复的query的word length进行恢复

            该攻击失败的唯一情况：recovery阶段的query并未在baseline阶段观测到，并且其真实word length不为0
            """
            # 注入
            observed_word_length = {}
            observed_volume = {}

            for i_week in self.observed_queries:
                for query in i_week:
                    if query in observed_word_length.keys():
                        continue
                    observed_word_length[query] = 0
                    observed_volume[query] = 0
                    #total_doc = 0
                    for _, cli_doc_kws in enumerate(self.cli_doc):  
                        #total_doc += len(cli_doc_kws)
                        if self.chosen_kws[query] in cli_doc_kws:
                            observed_word_length[query] += len(cli_doc_kws)
                            observed_volume[query] += 1
            print("len(diff wl): {}".format(len(observed_word_length)))
            #print(total_doc/len(self.cli_doc))
            #print(observed_volume)
            injected_word_length = {}
            injected_volume = {}


            #pr
            """
            kws_each_doc = math.ceil(((int) (len(self.chosen_kws)*self.kws_leak_percent))/2)
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))

            word_length_each_doc = [] # 要注入的文档word length
            if num_injection_doc >= 1:
                word_length_each_doc.append(1)
                self.total_inject_size += word_length_each_doc[0]
            if num_injection_doc >= 2:
                for i in range(1, num_injection_doc):
                    word_length_each_doc.append(word_length_each_doc[i-1] + word_length_each_doc[i-1])
                    self.total_inject_size += word_length_each_doc[i]

            injected_word_length = {}
            for kw_id in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
                if kw_id in observed_word_length.keys():
                    injected_word_length[kw_id] = observed_word_length[kw_id]
                    injected_volume[kw_id] = observed_volume[kw_id]
                else: # 不影响本实验结果，为模拟需要加上
                    injected_word_length[kw_id] = 0
                    injected_volume[kw_id] = 0
           
            inverted_index = {} # 添加的word length映射到对应keyword
            for kws_ind in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
                for num_ind in range(num_injection_doc):
                    if ((kws_ind >> num_ind) & 1) == 1 :
                        injected_word_length[kws_ind] += word_length_each_doc[num_ind]
                        injected_volume[kws_ind] += 1
         


            #permutation = np.random.permutation(len(self.chosen_kws))
            #chosen_kws = [keywords[idx] for idx in range(n_kw)]
            #chosen_kws = [self.chosen_kws[idx] for idx in permutation[: (int) (len(self.chosen_kws)/10)]]
            """
            for kw_id in range((int) (len(self.chosen_kws)*self.kws_leak_percent)):
                tmp_kw_id = kw_id
                injected_volume[kw_id] = 0
                while tmp_kw_id!=0:
                    if tmp_kw_id&1==1:
                        injected_volume[kw_id] += 1
                    tmp_kw_id >>= 1
                if kw_id in observed_word_length.keys():
                    injected_volume[kw_id] += observed_volume[kw_id]
                    injected_word_length[kw_id] = observed_word_length[kw_id] + kw_id*1
                else: # 不影响本实验结果，为模拟需要加上
                    #observed_word_length[kw_id] = 0
                    injected_word_length[kw_id] = kw_id*1
            
            
            return observed_word_length, injected_word_length, observed_volume, injected_volume


    class Frequency:
        def __init__(self, queries, chosen_kws, cli_doc, adv_doc, trend_matrix_norm, adv_trend_matrix_norm, combine_kw_id):
            self.queries = queries
            self.chosen_kws = chosen_kws
            self.cli_doc = cli_doc
            self.adv_doc = adv_doc
            self.trend_matrix_norm = trend_matrix_norm
            self.adv_trend_matrix_norm = adv_trend_matrix_norm
            self.combine_kw_id = combine_kw_id
            self.accuracy = 0
            self.time = 0

        def frequency_main(self):
            self.time = self.frequency_attack()
            self.accuracy = self.accuracy

        def combine_frequency_main(self):
            self.time = self.combine_frequency_attack()
            self.accuracy = self.accuracy
        @ utils._timeit
        def frequency_attack(self):

            #print(adver_known_freq)

            # 计算每个query的frequency
            
            tag_trend_matrix = np.zeros((self.trend_matrix_norm.shape[0], self.trend_matrix_norm.shape[1]))
            for i_week, weekly_tags in enumerate(self.queries):
                if len(weekly_tags) > 0:
                    counter = Counter(weekly_tags)
                    for key in counter:
                        tag_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)

            cost_freq = np.array([[np.linalg.norm(tag_trend_matrix[observed_trend_ind] - self.adv_trend_matrix_norm[adv_known_trend_ind]) 
                for adv_known_trend_ind in range(len(self.adv_trend_matrix_norm))] 
                for observed_trend_ind in range(len(tag_trend_matrix))])
    
            

            recover_queries = {}
            real_queries = {}
            for need_recover_kw_id in range(len(cost_freq)):
                recover_kw_id = np.argmin(cost_freq[need_recover_kw_id, :])
                recover_queries[need_recover_kw_id] = recover_kw_id
                real_queries[need_recover_kw_id] = need_recover_kw_id

            self.accuracy = np.mean(np.array([1 if real == predict else 0 
                for (real, predict) in zip(real_queries.values(), recover_queries.values())]))
            # print("freq_attack'accuracy is {}".format(self.accuracy))
            # print("freq_attack' running total time {}".format(self.running_time))
        @ utils._timeit
        def frequency_attack2(self):

            observed_query_freq = [0]*len(self.chosen_kws)
            total_query = 0
            for i_week in self.queries:
                for query in i_week:
                    total_query += 1
                    observed_query_freq[query] += 1
            for i in range(len(observed_query_freq)):

                observed_query_freq[i] = observed_query_freq[i]/total_query

            
            adver_known_freq = [0]*len(self.chosen_kws)
            for key_id in range(self.adv_trend_matrix_norm.shape[0]):
                tmp_k_f = 0
                for wee in range(self.adv_trend_matrix_norm.shape[1]):
                    tmp_k_f += self.adv_trend_matrix_norm[key_id][wee]
                adver_known_freq[key_id] = tmp_k_f/self.adv_trend_matrix_norm.shape[1]
            
            cost_freq = np.array([[abs(adver_known_freq[adv_id] - observed_query_freq[ob_id]) 
                for adv_id in range(len(adver_known_freq))]
                for ob_id in range(len(observed_query_freq))])
        
            #print(observed_query_freq)
            #print(adver_known_freq)
            #print(cost_freq)

            recover_queries = {}
            real_queries = {}
            for need_recover_kw_id in range(len(cost_freq)):
                recover_kw_id = np.argmin(cost_freq[need_recover_kw_id, :])
                recover_queries[need_recover_kw_id] = recover_kw_id
                real_queries[need_recover_kw_id] = need_recover_kw_id

            self.accuracy = np.mean(np.array([1 if real == predict else 0 
                for (real, predict) in zip(real_queries.values(), recover_queries.values())]))
            # print("freq_attack'accuracy is {}".format(self.accuracy))
            # print("freq_attack' running total time {}".format(self.running_time))

        @ utils._timeit
        def combine_frequency_attack(self):
            """
            假定对手知晓合并方式
            """
            observed_query_freq = {}
            total_query = 0
            for i_week in self.queries:
                for query in i_week:
                    (q1, q2) = random.choice(self.combine_kw_id[query])
                    if (q1, q2) in observed_query_freq.keys():
                        observed_query_freq[(q1, q2)] += 1
                    else:
                        observed_query_freq[(q1, q2)] = 1
                    total_query += 1
            """
            根据返回结果来确定合并后的不同query的freq，
            理论上一个关键字可能和多个不同关键字合并，因此观测到的freq是要小于已知的freq大小
            """        
            for i in observed_query_freq.keys():

                observed_query_freq[i] = observed_query_freq[i]/total_query

            
            adver_known_freq = [0]*len(self.chosen_kws)
            for key_id in range(self.adv_trend_matrix_norm.shape[0]):
                tmp_k_f = 0
                for wee in range(self.adv_trend_matrix_norm.shape[1]):
                    tmp_k_f += self.adv_trend_matrix_norm[key_id][wee]
                adver_known_freq[key_id] = tmp_k_f/self.adv_trend_matrix_norm.shape[1]
            
            recover_queries = {}
            real_queries = {}
            for (q1, q2) in observed_query_freq.keys():
                #if (q1, q2) in real_queries.keys():
                #    continue
                real_queries[(q1, q2)] = (q1, q2)
                min_value = 1
                min_recover = (-1,-1)
                for key_id in range(len(adver_known_freq)):
                    if abs(adver_known_freq[key_id] - observed_query_freq[(q1, q2)]) < min_value:
                        min_value = abs(adver_known_freq[key_id] - observed_query_freq[(q1, q2)])
                        min_recover = (key_id ,-1)
                for key_id1 in range(len(adver_known_freq)):
                    for key_id2 in range(key_id1, len(adver_known_freq)):
                        if abs(adver_known_freq[key_id1] + adver_known_freq[key_id2] - observed_query_freq[(q1, q2)]) < min_value:
                            min_value = abs(adver_known_freq[key_id1] + adver_known_freq[key_id2] - observed_query_freq[(q1, q2)])
                            min_recover = (key_id1 ,key_id2)
                recover_queries[(q1, q2)] = min_recover


            

            self.accuracy = np.mean(np.array([1 if real == predict else 0 
                for (real, predict) in zip(real_queries.values(), recover_queries.values())]))
            # print("freq_attack'accuracy is {}".format(self.accuracy))
            # print("freq_attack' running total time {}".format(self.running_time))

    
    class FHDSSE:
        def __init__(self,  queries, observed_queries, chosen_kws, cli_doc, adv_doc, trend_matrix_norm, adv_trend_matrix_norm,  accuracy, T, combine_kw_id):
            self.queries = queries
            self.chosen_kws = chosen_kws
            self.cli_doc = cli_doc
            self.adv_doc = adv_doc
            self.trend_matrix_norm = trend_matrix_norm
            self.adv_trend_matrix_norm = adv_trend_matrix_norm
            self.combine_kw_id = combine_kw_id
            #self.return_kw_id = return_kw_id
            self.accuracy = 0
            self.time = 0
            self.observed_queries = observed_queries
            self.accuracy = accuracy
            self.T = T

        def improved_T_decoding_main(self):

            observed_word_length, max_observed_length = self.improved_decoding_observed_for_sim()
            offset = self.compute_decoding_offset(observed_word_length, max_observed_length)
            injected_word_length, inverted_index = self.improved_decoding_inject(offset, observed_word_length)
            self.time = self.improved_decoding_recover(offset, observed_word_length, injected_word_length, inverted_index)

            kws_each_doc = math.ceil(len(self.chosen_kws)/2)
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))

            self.totals_inject_doc = num_injection_doc*math.ceil(kws_each_doc/self.T)
            self.accuracy = np.mean(np.array([1 if real == predict else 0 
                for (real, predict) in zip(self.real_tag.values(), self.recover_tag.values())]))

        def improved_decoding_observed_for_sim(self):
            #{0: 50, 2: 389, 7: 480, 8: 5184}
            observed_word_length = {}
            max_observed_length = 0

            for i_week in self.observed_queries:
                for query in i_week:
                    #if query in observed_word_length.keys():
                    #    continue
                    (q1, q2) = random.choice(self.combine_kw_id[query])
                    observed_word_length[(q1, q2)] = 0
                    for _, cli_doc_kws in enumerate(self.cli_doc):   
                        if self.chosen_kws[q1] in cli_doc_kws or (q2>=0 and self.chosen_kws[q2] in cli_doc_kws):
                            observed_word_length[(q1, q2)] += len(cli_doc_kws)
                    if max_observed_length < observed_word_length[(q1, q2)]:
                        max_observed_length = observed_word_length[(q1, q2)]

            return  observed_word_length, max_observed_length


        def compute_decoding_offset(self, observed_word_length, max_observed_length):          
            # find the injection offset
            divided_list = []
            for i in observed_word_length.keys():
                for j in observed_word_length.keys():
                    temp_minus = abs(observed_word_length[i] - observed_word_length[j])
                    if temp_minus != 0 and temp_minus not in divided_list:
                        divided_list.append(temp_minus)

            offset = 0
            for injection_offset in range(2, max_observed_length):
                flag = True
                for divisor in divided_list:
                    if divisor % injection_offset == 0:
                        flag = False
                        break
                
                if flag:
                    offset = injection_offset
                    break
                
            if offset == 0:
                raise ValueError("can not find the result")
            
            return offset

        def improved_decoding_inject(self, offset, observed_word_length):
            kws_each_doc = math.ceil(len(self.chosen_kws)/2)
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))

            word_length_each_doc = [] # 要注入的文档word length
            if num_injection_doc >= 1:
                word_length_each_doc.append(offset)
            if num_injection_doc >= 2:
                for i in range(1, num_injection_doc):
                    word_length_each_doc.append(word_length_each_doc[i-1] + word_length_each_doc[i-1])

            

            injected_word_length = {}
            for kw_id in range(len(self.chosen_kws)):

                # (kw1, kw2) = random.choice(self.combine_kw_id[kw_id])
                for (kw1, kw2) in self.combine_kw_id[kw_id]:
                    if (kw1, kw2) in observed_word_length.keys():
                        injected_word_length[(kw1, kw2)] = observed_word_length[(kw1, kw2)]
                    else: # 不影响本实验结果，为模拟需要加上
                        injected_word_length[(kw1, kw2)] = 0
           
            inverted_index = {} # 添加的word length映射到对应keyword
            for kws_ind in range(len(self.chosen_kws)):
                for num_ind in range(num_injection_doc):
                    if ((kws_ind >> num_ind) & 1) == 1 :
                        for (kw1, kw2) in self.combine_kw_id[kw_id]:
                            injected_word_length[(kw1, kw2)] += word_length_each_doc[num_ind]
                        #injected_word_length[kws_ind] += word_length_each_doc[num_ind]
         
            return injected_word_length, inverted_index

        @ utils._timeit
        def improved_decoding_recover(self, offset, observed_word_length, injected_word_length, inverted_index):
            # 不依赖于kw_id进行恢复，因此阈值施加只影响注入文档数，不影响算法
            # recover
            self.recover_tag = {}
            self.real_tag = {}
            for i_week in self.queries:
                for query in i_week:
                    self.real_tag[query] = query
                    self.recover_tag[query] = -1

                    (q1, q2) = random.choice(self.combine_kw_id[query])

                    for (k1, k2) in observed_word_length.keys():
                        if (injected_word_length[(q1, q2)] - observed_word_length[(k1, k2)])%offset == 0:
                        #if injected_word_length[query] - observed_word_length[kw_id] in inverted_index.keys():
                            #print(injected_word_length[query] - observed_word_length[kw_id])
                            #print(inverted_index[injected_word_length[query] - observed_word_length[kw_id]])
                            self.recover_tag[query] = random.randint(0, (int) ((injected_word_length[(q1, q2)] - observed_word_length[(k1, k2)])/offset + 1))
                            break
                    
                    if self.recover_tag[query] == -1 and injected_word_length[(q1, q2)] % offset == 0:
                        self.recover_tag[query] = random.randint(0, (int) ((injected_word_length[(q1, q2)])/offset + 1))
            #print("improved decoding accuracy is {}".format(self.accuracy))


        def improved_2_main(self):
            inverted_index, num_each_group, inject_doc = self.improved_inject_2_main() # 保存注入的文档id
            self.totals_inject_doc = inject_doc
            self.time = self.improved_recover_2_main(inverted_index, num_each_group)
            self.accuracy = utils.compute_accuracy(self.real_tag, self.recover_tag)
            #assert self.accuracy <= 1
            #print("self.zhgtie{}".format(self.time))
        
        def improved_inject_2_main(self):
            inject_kws = []
            for i in range(len(self.chosen_kws)):
                inject_kws.append(i)
            num_inject_doc_set = math.ceil(len(inject_kws)/(self.T + self.T)) # 需要分成的文档组数
            inverted_index = {}
            num_each_group = math.ceil(np.log2(self.T + self.T + 1)) # 每组需要注入的文档数
            inject_doc = num_inject_doc_set*num_each_group
            for kw in inject_kws:
                temp_l = []
                group = (int) (kw / (2*self.T))
                used_kw = kw % (2*self.T) 
                for num_ind in range(num_each_group):
                    if (((used_kw + 1) >> num_ind) & 1) == 1: # 每个关键字向右偏移1，确保总有注入文档包含任一关键字
                        real_doc_ind = num_ind + group*num_each_group
                        temp_l.append(real_doc_ind)                
                inverted_index[kw] = temp_l 
            return inverted_index, num_each_group, inject_doc

        @ utils._timeit
        def improved_recover_2_main(self, inverted_index, num_each_group):   
            # recover
            self.recover_tag = {}
            self.real_tag = {}
            #ttt = 0
            for i_week in self.queries:
                for query in i_week:
                    self.real_tag[query] = query
                    (q1, q2) = random.choice(self.combine_kw_id[query])
                    inverted_doc = inverted_index[query]
                    if q2>=0:
                        for ind in inverted_index[q2]:
                            if ind not in inverted_doc:
                                inverted_doc.append(ind)
                    kw_id = 0
                    group = 0
                    if len(inverted_doc) > 0:
                        group = (int) (inverted_doc[0] / num_each_group)
                    for doc_ind in inverted_doc: # 获取返回结果中注入的文档id
                        used_doc_ind = doc_ind % num_each_group
                        kw_id += (1 << used_doc_ind) # pow(2, kws_ind)
                    kw_id -= 1
                    kw_id += group*self.T*2
                    self.recover_tag[query] = random.randint(0, kw_id+1)
                    #ttt += kw_id
        

        def binary_main(self):
            inverted_index = self.binary_inject() # 保存注入的文档id
            self.totals_inject_doc = math.ceil(np.log2(len(self.chosen_kws)))
            self.time = self.binary_recover(inverted_index)
            self.accuracy = utils.compute_accuracy(self.real_tag, self.recover_tag)
            assert self.accuracy <= 1
            #print("self.zhgtie{}".format(self.time))

        def binary_inject(self):
            # suppose all the query could find the matching result
            # injection
            kws_each_doc = math.ceil(len(self.chosen_kws)/2)
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
            inverted_index = {}
            # inject_doc_kws = {}
            for kws_ind in range(len(self.chosen_kws)):
                temp_l = []
                for num_ind in range(num_injection_doc):
                    if ((kws_ind >> num_ind) & 1) == 1 :
                        temp_l.append(num_ind)
                        """
                        if num_ind in inject_doc_kws.keys():
                            inject_doc_kws[num_ind].append(kws_ind)
                        else:
                            temp_kws_l = []
                            temp_kws_l.append(kws_ind)
                            inject_doc_kws[num_ind] = temp_kws_l 
                        """
                                           
                inverted_index[kws_ind] = temp_l 
            return inverted_index  

        @ utils._timeit
        def binary_recover(self, inverted_index):
                 
            # recover
            self.recover_tag = {}
            self.real_tag = {}
            #ttt = 0
            for i_week in self.queries:
                for query in i_week:
                    self.real_tag[query] = query
                    kw_id = 0
                    for kws_ind in inverted_index[query]: # 获取返回结果中注入的文档id
                        kw_id += (1 << kws_ind) # pow(2, kws_ind)
                    #self.recover_tag[query] = kw_id
                    candidate_list = [0]
                    weib = 1
                    while kw_id!=0:
                        tmp = []
                        if kw_id & 1 == 1:
                            for c_kw in candidate_list:
                                tmp.append(weib+c_kw)
                            candidate_list.append(tmp)
                        weib <<= 1
                        kw_id >>= 1
                    self.recover_tag[query] = random.choice(candidate_list)
                    #ttt += kw_id
            
            #print(ttt)     
            # print("zhang_binary_search_attack'accuracy is {}".format(self.accuracy))
            # print("word_length_attack' running total time {}".format(self.running_time))
            # return self.accuracy  

