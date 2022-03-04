import os 
import pickle

from numpy.random.mtrand import rand
import manage_dataset
import leakage
#from pylab import *
from pandas import DataFrame
import numpy as np
import random
import sys
from multiprocessing import Pool
def get_real_wordlength_volume(chosen_kws):
    """
    获取文档中所有关键字的真实word length，用于模拟真实世界查询获得的word length
    """
    real_wordlength = {}
    real_volume = {}
    for kw_id in range(len(chosen_kws)):
        real_wordlength[kw_id] = 0
        real_volume[kw_id] = 0
        for _, cli_doc_kws in enumerate(cli_doc):   
            if chosen_kws[kw_id] in cli_doc_kws:
                real_wordlength[kw_id] += len(cli_doc_kws)
                real_volume[kw_id] += 1

    return real_wordlength, real_volume

def compute_decoding_offset(observed_word_length):          
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
    for injection_offset in range(11, sys.maxsize):
        flag = True
        for divisor in divided_list.keys():
            if divisor % injection_offset == 0:
                flag = False
                break
        
        if flag:
            offset = injection_offset
            break
        
    if offset == 0:
        offset = sys.maxsize>>5
    print("offset: {}".format(offset))
    return offset

Enron_path = r'C:\Users\admin\yan\Attack-evaluation\datasets_pro\real_data\Enron'
with open(os.path.join(Enron_path,"enron_doc_after_padding_2.pkl"), "rb") as f:
    doc = pickle.load(f)
    f.close()
with open(os.path.join(Enron_path,"enron_kws_dict.pkl"), "rb") as f:
    kw_dict = pickle.load(f)
    f.close()
kws_num = (int) (len(kw_dict))  
print(len(doc))  
print(kws_num)
cli_doc, adv_doc = manage_dataset.generate_cli_adv_doc((doc, kw_dict), 0.1)

exp_times = 10
t_kws = list(kw_dict.keys())
for e in [30, 300, 1000, 3000]:
    real_wordlength, real_volume = get_real_wordlength_volume(t_kws[:e])
    off = compute_decoding_offset(real_wordlength)
    print(off)