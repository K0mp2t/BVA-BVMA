from collections import Counter
import numpy as np

import time

from numpy import random

method_name = []
total_time = []

def _timeit(method):
    def timed(self, *args, **kw):
        ts = time.time()
        result = method(self, *args, **kw)
        te = time.time()
        method.running_time =np.round((te - ts), 3)
        total_time.append(method.running_time)
        method_name.append(method.__name__)
        #print("{} running total time {}".format(method.__name__, method.running_time))
        return method.running_time
    return timed

def print_time():
    for i in range(len(total_time)):
        
        print("{} running total time {}".format(method_name[i], total_time[i]))

def build_trend_matrix(traces, n_tags):
    n_weeks = len(traces)
    tag_trend_matrix = np.zeros((n_tags, n_weeks))
    for i_week, weekly_tags in enumerate(traces):
        if len(weekly_tags) > 0:
            counter = Counter(weekly_tags)
            for key in counter:
                tag_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)
    return tag_trend_matrix

def compute_accuracy(real_tag, recover_tag):
    return np.mean(np.array([1 if real == predict else 0 
                for (real, predict) in zip(real_tag.values(), recover_tag.values())]))  


def random_build_new_same_freq_kws(chosen_kws, freq_vector):
    """
    @ param: 关键字及其对应的frequency向量
    @ return: 合并后的关键字及freq，形式为(kw1, kw2, count) = freq，count表示相同关键字合并次数；第二个为(kw_id1, kw_id2, count)
    """
    kw_number_freq = [] # 存储一个三元组----(关键字，拆分数，拆分后频率)
    should_freq = 1/len(chosen_kws) # 理论上每个关键字的频率

    total_number_after_divide = 0 # 拆分后所有关键字数量
    for kw_id in range(len(chosen_kws)):
        num = (int) (np.ceil(freq_vector[kw_id]*len(chosen_kws)))
        if num == 0:
            num = 1
        total_number_after_divide += num
        freq = freq_vector[kw_id]/num
        kw_number_freq.append((kw_id, num, freq))
    
    # 按照每个关键字的拆分数进行排序，优先合并拆分数量多的关键字；
    # 如果拆分数量相同，则优先合并拆分后频率小的关键字
    # 可不需要，完全随机合并
    #kw_number_freq = sorted(kw_number_freq, key =lambda k: (-k[1]))

    return_kw_dinum_freq = kw_number_freq.copy()
    
    return_kw_id = {} # 返回值，字典，key = (kw_id_1, kw_id_2, count); value = freq_after_divide
    return_kw = {}
    tmp_count = {} # 对合并两个相同关键字计数

    rlist = list(range(len(kw_number_freq)))
    rlist.append(-1)
    for k in range(len(kw_number_freq)):
        #print("k: {}".format(k))
        while(kw_number_freq[k][1] > 0):
            min_index = -1
            #min_value = abs(should_freq - kw_number_freq[k][2]*kw_number_freq[k][1])
            min_value = 1
            #for m in range(k, len(kw_number_freq)):
            choose_times = 0
            max_times = 100
            sel_index = -1
            #this_list = rlist
            while(True):
                m = random.choice(rlist)
                
                """
                if m==-1:
                    min_index = m
                    break
                if m == k and kw_number_freq[k][1] <= 1:
                    rlist.remove(m)
                    #this_list.remove(m)
                    choose_times+=1
                    continue
                if 0 == kw_number_freq[m][1]:
                    rlist.remove(m)
                    #this_list.remove(m)
                    choose_times+=1
                    continue
                min_index = m
                break
                """
                if m==-1:
                    if (kw_number_freq[k][0],-1) in tmp_count.keys():
                        if choose_times>max_times:
                            #min_index = m
                            break
                        else:
                            choose_times+=1
                            continue
                    else:
                        min_index = m
                        break
                else:
                    if m == k and kw_number_freq[k][1] <= 1:
                        rlist.remove(m)
                        #this_list.remove(m)
                        choose_times+=1
                        continue
                    if 0 == kw_number_freq[m][1]:
                        rlist.remove(m)
                        #this_list.remove(m)
                        choose_times+=1
                        continue
                    if (kw_number_freq[k][0],kw_number_freq[m][0]) in tmp_count.keys():
                #if abs(should_freq - kw_number_freq[k][2] - kw_number_freq[m][2]) < min_value:
                #   min_value = abs(should_freq - kw_number_freq[k][2] - kw_number_freq[m][2])
                        if choose_times>max_times:
                            #min_index = m
                            break
                        else:
                            if kw_number_freq[m][2]<min_value:
                                min_value = kw_number_freq[m][2]
                                min_index = kw_number_freq[m][0]
                            choose_times+=1
                            continue
                    else:
                        min_index = m
                        break
            #if k==1:
            #    print(min_value)
            #    print(min_index)
            #print("k: {}".format(k))
            if min_index != -1:
                kw_number_freq[min_index] = (kw_number_freq[min_index][0], kw_number_freq[min_index][1] - 1,kw_number_freq[min_index][2])#kw_number_freq[min_index][1] - 1
                if (kw_number_freq[k][0],kw_number_freq[min_index][0]) in tmp_count.keys():
                    now_count = tmp_count[(kw_number_freq[k][0],kw_number_freq[min_index][0])]
                    return_kw_id[(min(kw_number_freq[k][0],kw_number_freq[min_index][0]), max(kw_number_freq[k][0],kw_number_freq[min_index][0]), now_count)] = kw_number_freq[min_index][2] + kw_number_freq[k][2]
                    tmp_count[(kw_number_freq[k][0],kw_number_freq[min_index][0])] = now_count + 1
                else:
                    
                    return_kw_id[(min(kw_number_freq[k][0],kw_number_freq[min_index][0]),max(kw_number_freq[k][0],kw_number_freq[min_index][0]),0)] = kw_number_freq[min_index][2] + kw_number_freq[k][2]
                    tmp_count[(kw_number_freq[k][0],kw_number_freq[min_index][0])] = 1
                #print(kw_number_freq[min_index][0])
                #print(return_kw_id[(kw_number_freq[k][0],kw_number_freq[min_index][0])])
                kw_number_freq[k] = (kw_number_freq[k][0], kw_number_freq[k][1] - 1,kw_number_freq[k][2])
            else:
                #if (kw_number_freq[k][0],-1) in tmp_count.keys():
                #    count = tmp_count[(kw_number_freq[k][0],-1)]+kw_number_freq[k][1] 
                if (kw_number_freq[k][0],-1) in tmp_count.keys():
                    now_count = tmp_count[(kw_number_freq[k][0],-1)]
                    return_kw_id[(kw_number_freq[k][0], -1, now_count)] = kw_number_freq[k][2]
                    tmp_count[(kw_number_freq[k][0],-1)] = now_count + 1
                else:
                    
                    return_kw_id[(kw_number_freq[k][0], -1,0)] = kw_number_freq[k][2]
                    tmp_count[(kw_number_freq[k][0],-1)] = 1
                

                #tmp_count[(kw_number_freq[k][0],-1)] = kw_number_freq[k][1] 
                #for ppp_c in range(0, kw_number_freq[k][1]):
                #    return_kw_id[(kw_number_freq[k][0], -1, ppp_c)] = kw_number_freq[k][2]
                #print(-1)
                #print(return_kw_id[(kw_number_freq[k][0],-1)])
                kw_number_freq[k] = (kw_number_freq[k][0], kw_number_freq[k][1] - 1, kw_number_freq[k][2])
            # print("k: {};;;re: {}".format(k, kw_number_freq[k][1]))

    for k in return_kw_id.keys():
        if k[1] == -1:
            return_kw[(chosen_kws[k[0]], -1, k[2])] = return_kw_id[k]
        else:
            return_kw[(chosen_kws[k[0]], chosen_kws[k[1]], k[2])] = return_kw_id[k]
    tottt = 0
    for kkk in return_kw_id.keys():
        if kkk[0] != -1:
            tottt += 1
        if kkk[1] != -1:
            tottt += 1
    #print(return_kw_id)
    #print(tottt)
    assert total_number_after_divide == tottt

    return return_kw_id, return_kw, total_number_after_divide, return_kw_dinum_freq

def build_new_same_freq_kws(chosen_kws, freq_vector):
    """
    @ param: 关键字及其对应的frequency向量
    @ return: 合并后的关键字及freq，形式为(kw1, kw2, count) = freq，count表示相同关键字合并次数；第二个为(kw_id1, kw_id2, count)
    """
    kw_number_freq = [] # 存储一个三元组----(关键字，拆分数，拆分后频率)
    should_freq = 1/len(chosen_kws) # 理论上每个关键字的频率

    total_number_after_divide = 0 # 拆分后所有关键字数量
    for kw_id in range(len(chosen_kws)):
        num = (int) (np.ceil(freq_vector[kw_id]*len(chosen_kws)))
        if num == 0:
            num = 1
        total_number_after_divide += num
        freq = freq_vector[kw_id]/num
        kw_number_freq.append((kw_id, num, freq))
    
    # 按照每个关键字的拆分数进行排序，优先合并拆分数量多的关键字；
    # 如果拆分数量相同，则优先合并拆分后频率小的关键字
    kw_number_freq = sorted(kw_number_freq, key =lambda k: (-k[1]))

    return_kw_dinum_freq = kw_number_freq.copy()
    
    return_kw_id = {} # 返回值，字典，key = (kw_id_1, kw_id_2, count); value = freq_after_divide
    return_kw = {}
    tmp_count = {} # 对合并两个相同关键字计数

    for k in range(len(kw_number_freq)):
        #print("k: {}".format(k))
        while(kw_number_freq[k][1] > 0):
            min_index = -1
            min_value = abs(should_freq - kw_number_freq[k][2]*kw_number_freq[k][1])
            for m in range(k, len(kw_number_freq)):
           
                if m == k and kw_number_freq[k][1] <= 1:
                    continue
                if 0 == kw_number_freq[m][1]:
                    continue
                if abs(should_freq - kw_number_freq[k][2] - kw_number_freq[m][2]) < min_value:
                    min_value = abs(should_freq - kw_number_freq[k][2] - kw_number_freq[m][2])
                    min_index = m
            #if k==1:
            #    print(min_value)
            #    print(min_index)
            #print("k: {}".format(k))
            if min_index != -1:
                kw_number_freq[min_index] = (kw_number_freq[min_index][0], kw_number_freq[min_index][1] - 1,kw_number_freq[min_index][2])#kw_number_freq[min_index][1] - 1
                if (kw_number_freq[k][0],kw_number_freq[min_index][0]) in tmp_count.keys():
                    now_count = tmp_count[(kw_number_freq[k][0],kw_number_freq[min_index][0])]
                    return_kw_id[(min(kw_number_freq[k][0],kw_number_freq[min_index][0]), max(kw_number_freq[k][0],kw_number_freq[min_index][0]), now_count)] = kw_number_freq[min_index][2] + kw_number_freq[k][2]
                    tmp_count[(kw_number_freq[k][0],kw_number_freq[min_index][0])] = now_count + 1
                else:
                    
                    return_kw_id[(min(kw_number_freq[k][0],kw_number_freq[min_index][0]),max(kw_number_freq[k][0],kw_number_freq[min_index][0]),0)] = kw_number_freq[min_index][2] + kw_number_freq[k][2]
                    tmp_count[(kw_number_freq[k][0],kw_number_freq[min_index][0])] = 1
                #print(kw_number_freq[min_index][0])
                #print(return_kw_id[(kw_number_freq[k][0],kw_number_freq[min_index][0])])
                kw_number_freq[k] = (kw_number_freq[k][0], kw_number_freq[k][1] - 1,kw_number_freq[k][2])
            else:
                #if (kw_number_freq[k][0],-1) in tmp_count.keys():
                #    count = tmp_count[(kw_number_freq[k][0],-1)]+kw_number_freq[k][1] 
                #    tmp_count[(kw_number_freq[k][0],-1)] = count
                    
                #else:
                #count = kw_number_freq[k][1] 
                tmp_count[(kw_number_freq[k][0],-1)] = kw_number_freq[k][1] 
                for ppp_c in range(0, kw_number_freq[k][1]):
                    return_kw_id[(kw_number_freq[k][0], -1, ppp_c)] = kw_number_freq[k][2]
                #print(-1)
                #print(return_kw_id[(kw_number_freq[k][0],-1)])
                kw_number_freq[k] = (kw_number_freq[k][0], 0, kw_number_freq[k][2])
            # print("k: {};;;re: {}".format(k, kw_number_freq[k][1]))

    for k in return_kw_id.keys():
        if k[1] == -1:
            return_kw[(chosen_kws[k[0]], -1, k[2])] = return_kw_id[k]
        else:
            return_kw[(chosen_kws[k[0]], chosen_kws[k[1]], k[2])] = return_kw_id[k]
    tottt = 0
    for kkk in return_kw_id.keys():
        if kkk[0] != -1:
            tottt += 1
        if kkk[1] != -1:
            tottt += 1
    #print(return_kw_id)
    #print(tottt)
    assert total_number_after_divide == tottt

    return return_kw_id, return_kw, total_number_after_divide, return_kw_dinum_freq
    #print(trend_matrix_norm.shape[0])  
    #print(trend_matrix_norm.shape[1]) 
    #print(len(trend_matrix_norm))
def kw_id_to_combine(return_kw_id):
    """
    返回合并后的关键字映射
    """
    res = {}
    for key in return_kw_id.keys():
        if key[0] != -1:
            if key[0] in res.keys():
                res[key[0]].append((key[0], key[1]))
            else:
                tmp_combine = [(key[0], key[1])]
                res[key[0]] = tmp_combine
        if key[1] != -1:        
            if key[1] in res.keys():
                res[key[1]].append((key[0], key[1]))
            else:
                tmp_combine = [(key[0], key[1])]
                res[key[1]] = tmp_combine
    return res

def combine_kw_freq(return_kw_id):
    """
    返回合并后关键字的freq
    """
    res = {}
    for i in return_kw_id.keys():
        if (i[0], i[1]) in res.keys():
            res[(i[0], i[1])] += return_kw_id[i]
        else:
            res[(i[0], i[1])] = return_kw_id[i]
    return res

def compute_combine_volume(res, chosen_kws, cli_doc):
    observed_each_doc_word_length = {}
    observed_volume = {}
    for (kw_id_1, kw_id_2) in res.keys():
        observed_each_doc_word_length[(kw_id_1, kw_id_2)] = 0
        observed_volume[(kw_id_1, kw_id_2)] = 0
        for _, cli_doc_kws in enumerate(cli_doc):   
            if chosen_kws[kw_id_1] in cli_doc_kws or (kw_id_2 != -1 and chosen_kws[kw_id_2] in cli_doc_kws):
                observed_each_doc_word_length[(kw_id_1, kw_id_2)] += len(cli_doc_kws)
                observed_volume[(kw_id_1, kw_id_2)] += 1
    return observed_each_doc_word_length, observed_volume

def compute_original_volume(chosen_kws, cli_doc):
    observed_each_doc_word_length = {}
    observed_volume = {}

    for kw_id in range(len(chosen_kws)):
        observed_each_doc_word_length[kw_id] = 0
        observed_volume[kw_id] = 0
        for _, cli_doc_kws in enumerate(cli_doc):   
            if chosen_kws[kw_id] in cli_doc_kws:
                observed_each_doc_word_length[kw_id] += len(cli_doc_kws)
                observed_volume[kw_id] += 1
    return observed_each_doc_word_length, observed_volume
