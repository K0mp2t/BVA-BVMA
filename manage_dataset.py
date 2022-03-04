import os
import pickle
from numpy.core.fromnumeric import sort

from numpy.random.mtrand import f
import utils
import numpy as np
import leakage
from multiprocessing.pool import ThreadPool


def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        raise ValueError("The file {} does not exist".format(dataset_path))
    
    with open(dataset_path, "rb") as f:
        doc, kw_dict = pickle.load(f)

    return doc, kw_dict

def generate_cli_adv_doc(dataset, adv_rate):
    doc, _ = dataset
    client_doc = doc
    permutation = np.random.permutation((len(client_doc)))
    adv_doc = [client_doc[i] for i in permutation[:int(len(client_doc)*adv_rate)]]
    return client_doc, adv_doc

def generate_keyword_trend_matrix(dataset, n_kw, n_weeks, offset):
    """
    generate keywords and their queries' trend matrix(row: kws; colume: weeks)
    @ param: dataset, n_kw, n_weeks, adv. known weeks offset
    @ return: chosen_kws, trend_matrix, offset_trend_matrix
    """
    _, kw_dict = dataset
    
    # 随机生成n_kw个keywords
    keywords = list(kw_dict.keys())
    permutation = np.random.permutation(len(keywords))
    #chosen_kws = [keywords[idx] for idx in range(n_kw)]
    chosen_kws = [keywords[idx] for idx in permutation[: n_kw]]
    #print(chosen_kws)
    # 选择后n_weeks周的trend
    trend_matrix_norm = np.array([[float(kw_dict[kw]['trend'][i]) for i in range(len(kw_dict[kw]['trend']))] for kw in chosen_kws])
    #print((trend_matrix_norm[:, 1]))
    for i_col in range(trend_matrix_norm.shape[1]):
        if sum(trend_matrix_norm[:, i_col]) == 0:
            print("The {d}th column of the trend matrix adds up to zero, making it uniform!")
            trend_matrix_norm[:, i_col] = 1 / n_kw
        else:
            trend_matrix_norm[:, i_col] =  trend_matrix_norm[:, i_col] / (float) (sum(trend_matrix_norm[:, i_col]))
    #print((trend_matrix_norm[:, 1]))
    #print(max(trend_matrix_norm[:, 1]))
    #print(trend_matrix_norm[:, 1].index(max(trend_matrix_norm[:, 1])))
    return chosen_kws, trend_matrix_norm[:, -n_weeks:], trend_matrix_norm[:, -n_weeks:] if offset ==0 else trend_matrix_norm[:, -offset-n_weeks: -offset]


def generate_queries(trend_matrix_norm, q_mode, n_qr):
    """
    generate queries from different query modes
    @ param: trend_matrix, q_mode = ['possion', 'multi', 'uniform']
    @ return: queries matrix(each week)(each kw_id in the chosen_kws)
    """
    queries = []
    n_kw, n_weeks = trend_matrix_norm.shape
    if q_mode == 'trend':
        for i_week in range(n_weeks):
            #n_qr_i_week = np.random.poisson(n_qr)
            n_qr_i_week = n_qr
            # 采样值范围和对应概率，采样数
            queries_i_week = list(np.random.choice(list(range(n_kw)), n_qr_i_week, p = trend_matrix_norm[:, i_week]))
            queries.append(queries_i_week)

    elif q_mode == 'random':
        for i_week in range(n_weeks):
            # 采样值范围和对应概率，采样数
            queries_i_week = list(np.random.choice(list(range(n_kw)), n_qr))
            queries.append(queries_i_week)
    else:
        raise ValueError("Query params not recognized")        
    return queries

test_n_kws = [10, 50, 200]#, 500, 1000, 3000]
#test_n_kws = [10]
test_n_weeks = [50, 50, 50 ,50]
T = [2,5,8,23,50,101,203]
zhang_binary_accuracy = {}
zhang_hierar_accuracy = {}
zhang_improved_accuracy = {}
zhang_improved_2_accuracy = {}
blackstone_decoding_attack_accuracy = {}
blackstone_binary_attack_accuracy = {}
blackstone_improved_decoding_attack_accuracy = {}
blackstone_improved_T_decoding_attack_accuracy = {}

zhang_binary_time = {}
zhang_hierar_time = {}
zhang_improved_time = {}
zhang_improved_2_time = {}
blackstone_decoding_attack_time = {}
blackstone_binary_attack_time = {}
blackstone_improved_decoding_attack_time = {}
blackstone_improved_T_decoding_attack_time = {}

zhang_binary_doc = {}
zhang_hierar_doc = {}
zhang_improved_doc = {}
zhang_improved_2_doc = {}
blackstone_decoding_attack_doc = {}
blackstone_binary_attack_doc = {}
blackstone_improved_decoding_attack_doc = {}
blackstone_improved_T_decoding_attack_doc = {}


leak = []

def run_attack(test_times_each_group, test_n_kws, T, attack_name, accuracy, time, inject_doc, kws_or_T):
    # Datasets = (d, kw_dic); kw_dict = {key1:value1, ..., keyN: valueN}; value = [count, trend]; trend = [T1, T2, ..., T260]
    doc, kw_dict = load_dataset(os.path.join("datasets_pro","enron_db.pkl"))
    # print(len(doc))
    # print(len(kw_dict))
    # tt = list(kw_dict.keys())
    # print(len(kw_dict[tt[0]]['trend']))
    # print(len(doc))
    # print(len(kw_dict))
    cli_doc, adv_doc = generate_cli_adv_doc((doc, kw_dict), 1)
    #print(len(test_n_kws))

    for i in range(len(test_n_kws)):
        
        
        for t_index in range(len(T)):

            print(test_n_kws[i])
            temp_de_accuracy = []
            temp_bi_accuracy = []
            temp_improved_de_accuracy = []
            temp_improved_T_de_accuracy = []
            temp_zhang_binary_accuracy = []
            temp_zhang_hie_accuracy = []
            temp_zhang_improved_accuracy = []
            temp_zhang_improved_2_accuracy = []

            temp_de_time = []
            temp_bi_time = []
            temp_improved_de_time = []
            temp_improved_T_de_time = []
            temp_zhang_binary_time = []
            temp_zhang_hie_time = []
            temp_zhang_improved_time = []
            temp_zhang_improved_2_time = []

            temp_de_doc = []
            temp_bi_doc = []
            temp_improved_de_doc = []
            temp_improved_T_de_doc = []
            temp_zhang_binary_doc = []
            temp_zhang_hie_doc = []
            temp_zhang_improved_doc = []
            temp_zhang_improved_2_doc = []

            for j in range(test_times_each_group):
                print("{}: {}".format(test_n_kws[i], j))
                chosen_kws, trend_matrix_norm, adv_trend_matrix_norm = generate_keyword_trend_matrix((doc, kw_dict), test_n_kws[i], 50, 5)
                recovery_queries = generate_queries(trend_matrix_norm, 'possion', 50)
                observed_queries = generate_queries(trend_matrix_norm, 'possion', 50)

                attack = leakage.Leakage(cli_doc, adv_doc, chosen_kws, recovery_queries, trend_matrix_norm, adv_trend_matrix_norm)
                
                #ikk = attack.IKK(attack.adv_doc, attack.cli_doc, attack.chosen_kws, attack.queries)
                #count = attack.Count(adv_doc, cli_doc, chosen_kws, recovery_queries)
                #observed_queries = queries#generate_queries(trend_matrix_norm, 'possion', 50)
                gege_accuracy = 0
                num_inject_doc = 0
                zhang = attack.Zhang(T[t_index], attack.chosen_kws, attack.queries, gege_accuracy, num_inject_doc)
                
                #zhang.binary_attack()
                if "zhang_binary" in attack_name:
                    zhang.binary_main()
                    temp_zhang_binary_time.append(zhang.time)
                    temp_zhang_binary_accuracy.append(zhang.accuracy)
                    temp_zhang_binary_doc.append(zhang.totals_inject_doc)
                if "zhang_hie" in attack_name:
                    zhang.original_main()
                    temp_zhang_hie_time.append(zhang.time)
                    temp_zhang_hie_accuracy.append(zhang.accuracy)
                    temp_zhang_hie_doc.append(zhang.totals_inject_doc)
                if "zhang_improved" in attack_name:
                    zhang.improved_main()
                    temp_zhang_improved_time.append(zhang.time)
                    temp_zhang_improved_accuracy.append(zhang.accuracy)
                    temp_zhang_improved_doc.append(zhang.totals_inject_doc)
                if "zhang_improved_2" in attack_name:
                    zhang.improved_2_main()
                    temp_zhang_improved_2_time.append(zhang.time)
                    temp_zhang_improved_2_accuracy.append(zhang.accuracy)
                    temp_zhang_improved_2_doc.append(zhang.totals_inject_doc)
                #blackstone.word_length_attack()
                #blackstone.sel_word_length_attack()
                #blackstone.subgraph_attack('access_doc_id')
                #blackstone.subgraph_attack('word_length')
                #IKK.IKK_attack()
                #count.count()
                blackstone = attack.Blackstone(cli_doc, adv_doc, chosen_kws, recovery_queries, observed_queries, gege_accuracy, T[t_index])
                if "blackstone_binary" in attack_name:
                    blackstone.binary_main()
                    temp_bi_time.append(blackstone.time)
                    temp_bi_accuracy.append(blackstone.accuracy)
                    temp_bi_doc.append(blackstone.totals_inject_doc)
                if "blackstone_decoding" in attack_name:
                    blackstone.decoding_main()
                    temp_de_time.append(blackstone.time)
                    temp_de_accuracy.append(blackstone.accuracy)
                    temp_de_doc.append(blackstone.totals_inject_doc)   
                if "blackstone_improved_decoding" in attack_name:
                    blackstone.improved_decoding_main()
                    temp_improved_de_time.append(blackstone.time)
                    temp_improved_de_accuracy.append(blackstone.accuracy)
                    temp_improved_de_doc.append(blackstone.totals_inject_doc)  
                if "blackstone_improved_T_decoding" in attack_name:
                    blackstone.improved_T_decoding_main()
                    temp_improved_T_de_time.append(blackstone.time)   
                    temp_improved_T_de_accuracy.append(blackstone.accuracy)
                    temp_improved_T_de_doc.append(blackstone.totals_inject_doc)
                # print(zhang.superclass.queries)
                #leakage.frequency_attack()
            if kws_or_T == "kws":
                x_name = '{}'.format(test_n_kws[i])
            elif kws_or_T == "T":
                x_name = '{}'.format(T[t_index])




            if "zhang_binary" in attack_name:
                zhang_binary_accuracy[x_name] = temp_zhang_binary_accuracy
                zhang_binary_time[x_name] = np.mean(temp_zhang_binary_time)
                zhang_binary_doc[x_name] = np.mean(temp_zhang_binary_doc)
                
            if "zhang_hie" in attack_name:
                zhang_hierar_accuracy[x_name] = temp_zhang_hie_accuracy
                zhang_hierar_time[x_name] = np.mean(temp_zhang_hie_time)
                zhang_hierar_doc[x_name] = np.mean(temp_zhang_hie_doc)
                
            if "zhang_improved" in attack_name:
                zhang_improved_accuracy[x_name] = temp_zhang_improved_accuracy
                zhang_improved_time[x_name] = np.mean(temp_zhang_improved_time)
                zhang_improved_doc[x_name] = np.mean(temp_zhang_improved_doc)
                
            if "zhang_improved_2" in attack_name:
                zhang_improved_2_accuracy[x_name] = temp_zhang_improved_2_accuracy
                zhang_improved_2_time[x_name] = np.mean(temp_zhang_improved_2_time)
                zhang_improved_2_doc[x_name] = np.mean(temp_zhang_improved_2_doc)
                

            #blackstone.word_length_attack()
            #blackstone.sel_word_length_attack()
            #blackstone.subgraph_attack('access_doc_id')
            #blackstone.subgraph_attack('word_length')
            #IKK.IKK_attack()
            #count.count()
            if "blackstone_decoding" in attack_name:
                blackstone_decoding_attack_accuracy[x_name] = temp_de_accuracy
                blackstone_decoding_attack_time[x_name] = np.mean(temp_de_time)
                blackstone_decoding_attack_doc[x_name] = np.mean(temp_de_doc)
                
            if "blackstone_binary" in attack_name:
                blackstone_binary_attack_accuracy[x_name] = temp_bi_accuracy
                blackstone_binary_attack_time[x_name] = np.mean(temp_bi_time)
                blackstone_binary_attack_doc[x_name] = np.mean(temp_bi_doc)
                
            if "blackstone_improved_decoding" in attack_name:
                blackstone_improved_decoding_attack_accuracy[x_name] = temp_improved_de_accuracy
                blackstone_improved_decoding_attack_time[x_name] = np.mean(temp_improved_de_time)
                blackstone_improved_decoding_attack_doc[x_name] = np.mean(temp_improved_de_doc)
                
            if "blackstone_improved_T_decoding" in attack_name:
                blackstone_improved_T_decoding_attack_accuracy[x_name] = temp_improved_T_de_accuracy
                blackstone_improved_T_decoding_attack_time[x_name] = np.mean(temp_improved_T_de_time)
                blackstone_improved_T_decoding_attack_doc[x_name] = np.mean(temp_improved_T_de_doc)
                

    if "zhang_binary" in attack_name:
        accuracy.append(zhang_binary_accuracy)
        time.append(zhang_binary_time)
        inject_doc.append(zhang_binary_doc)
        
    if "zhang_hie" in attack_name:
        accuracy.append(zhang_hierar_accuracy)
        time.append(zhang_hierar_time)
        inject_doc.append(zhang_hierar_doc)
        
    if "zhang_improved" in attack_name:
        accuracy.append(zhang_improved_accuracy)
        time.append(zhang_improved_time)
        inject_doc.append(zhang_improved_doc)

    if "zhang_improved_2" in attack_name:
        accuracy.append(zhang_improved_2_accuracy)
        time.append(zhang_improved_2_time)
        inject_doc.append(zhang_improved_2_doc)
        
    if "blackstone_binary" in attack_name:
        accuracy.append(blackstone_binary_attack_accuracy)
        time.append(blackstone_binary_attack_time)
        inject_doc.append(blackstone_binary_attack_doc)
        
    if "blackstone_decoding" in attack_name:
        accuracy.append(blackstone_decoding_attack_accuracy)
        time.append(blackstone_decoding_attack_time)
        inject_doc.append(blackstone_decoding_attack_doc)
        
        
    if "blackstone_improved_decoding" in attack_name:
        accuracy.append(blackstone_improved_decoding_attack_accuracy)
        time.append(blackstone_improved_decoding_attack_time)
        inject_doc.append(blackstone_improved_decoding_attack_doc)
        
    if "blackstone_improved_T_decoding" in attack_name:
        accuracy.append(blackstone_improved_T_decoding_attack_accuracy)
        time.append(blackstone_improved_T_decoding_attack_time)
        inject_doc.append(blackstone_improved_T_decoding_attack_doc)

    #print("dsfsf{}".format(zhang_binary_time))   

    #utils.print_time()
    
def multip_attack(test_times_each_group, test_n_kws, attack_name, accuracy, time):
    doc, kw_dict = load_dataset(os.path.join("datasets_pro","enron_db.pkl"))

    cli_doc, adv_doc = generate_cli_adv_doc((doc, kw_dict), 0.5)
    for i in range(len(test_n_kws)):
        chosen_kws, trend_matrix_norm, adv_trend_matrix_norm = generate_keyword_trend_matrix((doc, kw_dict), test_n_kws[i], 50, 5)
        recovery_queries = generate_queries(trend_matrix_norm, 'possion', 100)
        observed_queries = generate_queries(trend_matrix_norm, 'possion', 100)


frequency_accuracy = {}
combine_frequency_accuracy = {}
count_accuracy = {}
word_access_accuracy = {}
word_volume_accuracy = {}
improved_T_decoding_to_FHDSSE_accuracy = {}
blackstone_improved_T_decoding_attack_accuracy = {}
zhang_improved_2_accuracy = {}
zhang_improved_2_to_FHDSSE_accuracy = {}

#blackstone_improved_T_decoding_attack_doc = {}
frequency_time = {}
combine_frequency_time = {}
count_time = {}
word_access_time = {}
word_volume_time = {}
improved_T_decoding_to_FHDSSE_time = {}
blackstone_improved_T_decoding_attack_time = {}
zhang_improved_2_time = {}
zhang_improved_2_to_FHDSSE_time = {}

def run_attack2(test_times_each_group, test_n_kws, attack_name, accuracy, time):
    doc, kw_dict = load_dataset(os.path.join("datasets_pro","enron_db.pkl"))

    cli_doc, adv_doc = generate_cli_adv_doc((doc, kw_dict), 0.5)

    for i in range(len(test_n_kws)):

        print(test_n_kws[i])
        temp_frequency_accuracy = []
        temp_combine_frequency_accuracy = []
        temp_count_accuracy = []
        temp_word_access_accuracy = []
        temp_word_volume_accuracy = []
        temp_improved_T_decoding_to_FHDSSE_accuracy = []
        temp_improved_T_de_accuracy = []   
        temp_zhang_improved_2_accuracy = []  
        temp_zhang_improved_2_to_FHDSSE_accuracy = []

        temp_frequency_time = []
        temp_combine_frequency_time = []
        temp_count_time = []
        temp_word_access_time = []
        temp_word_volume_time = []
        temp_improved_T_decoding_to_FHDSSE_time = []
        temp_improved_T_de_time = []
        temp_zhang_improved_2_time = []
        temp_zhang_improved_2_to_FHDSSE_time = []

        for j in range(test_times_each_group):
            print("{}: {}".format(test_n_kws[i], j))
            chosen_kws, trend_matrix_norm, adv_trend_matrix_norm = generate_keyword_trend_matrix((doc, kw_dict), test_n_kws[i], 50, 5)
            recovery_queries = generate_queries(trend_matrix_norm, 'possion', 100)
            observed_queries = generate_queries(trend_matrix_norm, 'possion', 100)

            attack = leakage.Leakage(cli_doc, adv_doc, chosen_kws, recovery_queries, trend_matrix_norm, adv_trend_matrix_norm)
            combine_kw_id = {}
            if "combine_frequency" in attack_name:
                return_kw_id, _, _, _ = utils.build_new_same_freq_kws(chosen_kws, trend_matrix_norm[:, 0])
                combine_kw_id = utils.kw_id_to_combine(return_kw_id)

            frequency = attack.Frequency(recovery_queries, chosen_kws, cli_doc, adv_doc, trend_matrix_norm, adv_trend_matrix_norm, combine_kw_id)
            if "frequency" in attack_name:
                frequency.frequency_main()
                temp_frequency_time.append(frequency.time)
                temp_frequency_accuracy.append(frequency.accuracy)
            if "combine_frequency" in attack_name:
                frequency.combine_frequency_main()
                temp_combine_frequency_time.append(frequency.time)
                temp_combine_frequency_accuracy.append(frequency.accuracy)

            blackstone = attack.Blackstone(cli_doc, adv_doc, chosen_kws, recovery_queries, observed_queries, accuracy, 10)    
            if "blackstone_improved_T_decoding" in attack_name:
                blackstone.improved_T_decoding_main()
                temp_improved_T_de_time.append(blackstone.time)   
                temp_improved_T_de_accuracy.append(blackstone.accuracy)
                #temp_improved_T_de_doc.append(blackstone.totals_inject_doc)
            FHDSSE = attack.FHDSSE(recovery_queries, observed_queries, chosen_kws, cli_doc, adv_doc, trend_matrix_norm, adv_trend_matrix_norm,  accuracy, 10, combine_kw_id)
            if "improved_T_decoding_to_FHDSSE" in attack_name:
                FHDSSE.improved_T_decoding_main()
                temp_improved_T_decoding_to_FHDSSE_time.append(FHDSSE.time)
                temp_improved_T_decoding_to_FHDSSE_accuracy.append(FHDSSE.accuracy)

            zhang = attack.Zhang(50, attack.chosen_kws, attack.queries, accuracy, num_inject_doc=0)
            if "zhang_improved_2" in attack_name:
                zhang.improved_2_main()
                temp_zhang_improved_2_time.append(zhang.time)
                temp_zhang_improved_2_accuracy.append(zhang.accuracy)
                #temp_zhang_improved_2_doc.append(zhang.totals_inject_doc)
            if "zhang_improved_2_to_FHDSSE" in attack_name:
                FHDSSE.improved_2_main()
                temp_zhang_improved_2_to_FHDSSE_time.append(FHDSSE.time)
                temp_zhang_improved_2_to_FHDSSE_accuracy.append(FHDSSE.accuracy)


            count = attack.Count(adv_doc, cli_doc, chosen_kws, recovery_queries)
            if "count" in attack_name:
                count.count_main()
                temp_count_time.append(count.time)
                temp_count_accuracy.append(count.accuracy)
            

            blackstone = attack.Blackstone(cli_doc, adv_doc, chosen_kws, recovery_queries, observed_queries, 0, 10)
            if "word_access" in attack_name:
                blackstone.subgraph_main("access_doc_id")
                temp_word_access_time.append(blackstone.time)
                temp_word_access_accuracy.append(blackstone.accuracy)
            if "word_volume" in  attack_name:
                blackstone.subgraph_main("word_length")
                temp_word_volume_time.append(blackstone.time)
                temp_word_volume_accuracy.append(blackstone.accuracy)
            
        x_name = '{}'.format(test_n_kws[i])
        if "frequency" in attack_name:
            frequency_accuracy[x_name] = temp_frequency_accuracy
            frequency_time[x_name] = np.mean(temp_frequency_time)
        if "combine_frequency" in attack_name:
            combine_frequency_accuracy[x_name] = temp_combine_frequency_accuracy
            combine_frequency_time[x_name] = np.mean(temp_combine_frequency_time)
        if "blackstone_improved_T_decoding" in attack_name:
                blackstone_improved_T_decoding_attack_accuracy[x_name] = temp_improved_T_de_accuracy
                blackstone_improved_T_decoding_attack_time[x_name] = np.mean(temp_improved_T_de_time)
        if "improved_T_decoding_to_FHDSSE" in attack_name:
            improved_T_decoding_to_FHDSSE_accuracy[x_name] = temp_improved_T_decoding_to_FHDSSE_accuracy
            improved_T_decoding_to_FHDSSE_time[x_name] = np.mean(temp_improved_T_decoding_to_FHDSSE_time)
        if "zhang_improved_2" in attack_name:
            zhang_improved_2_accuracy[x_name] = temp_zhang_improved_2_accuracy
            zhang_improved_2_time[x_name] = np.mean(temp_zhang_improved_2_time)
        if "zhang_improved_2_to_FHDSSE" in attack_name:
            zhang_improved_2_to_FHDSSE_accuracy[x_name] = temp_zhang_improved_2_to_FHDSSE_accuracy
            zhang_improved_2_to_FHDSSE_time[x_name] = np.mean(temp_zhang_improved_2_to_FHDSSE_time)

        if "count" in attack_name:
            count_accuracy[x_name] = temp_count_accuracy
            count_time[x_name] = np.mean(temp_count_time)
        if "word_access" in attack_name:
            word_access_accuracy[x_name] = temp_word_access_accuracy
            word_access_time[x_name] = np.mean(temp_word_access_time)
        if "word_volume" in  attack_name:
            word_volume_accuracy[x_name] = temp_word_volume_accuracy
            word_volume_time[x_name] = np.mean(temp_word_volume_time)
            
    if "frequency" in attack_name:
        accuracy.append(frequency_accuracy)
        time.append(frequency_time)
    if "combine_frequency" in attack_name:
        accuracy.append(combine_frequency_accuracy)
        time.append(combine_frequency_time)
    if "blackstone_improved_T_decoding" in attack_name:
        accuracy.append(blackstone_improved_T_decoding_attack_accuracy)
        time.append(blackstone_improved_T_decoding_attack_time)
    if "improved_T_decoding_to_FHDSSE" in attack_name:
        accuracy.append(improved_T_decoding_to_FHDSSE_accuracy)
        time.append(improved_T_decoding_to_FHDSSE_time)
    if "zhang_improved_2" in attack_name:
        accuracy.append(zhang_improved_2_accuracy)
        time.append(zhang_improved_2_time)
    if "zhang_improved_2_to_FHDSSE" in attack_name:
        accuracy.append(zhang_improved_2_to_FHDSSE_accuracy)
        time.append(zhang_improved_2_to_FHDSSE_time)
    if "count" in attack_name:
        accuracy.append(count_accuracy)
        time.append(count_time)
    if "word_access" in attack_name:
        accuracy.append(word_access_accuracy)
        time.append(word_access_time)
    if "word_volume" in  attack_name:
        accuracy.append(word_volume_accuracy)
        time.append(word_volume_time)
    #utils.print_time()




if __name__ == '__main__':
    
    lucene_path = r'C:\Users\admin\yan\Attack-evaluation\datasets_pro\Lucene'
    with open(os.path.join(lucene_path,"lucene_doc.pkl"), "rb") as f:
        doc = pickle.load(f)
        f.close()
    with open(os.path.join(lucene_path,"lucene_kws_dict.pkl"), "rb") as f:
        kw_dict = pickle.load(f)
        f.close()
           
    #doc = load_dataset())
    #kw_dict = load_dataset(os.path.join(lucene_path,"lucene_kws_dict.pkl"))
    cli_doc, adv_doc = generate_cli_adv_doc((doc, kw_dict), 1)
    for i in (10, 50000):
        offset = 5
        chosen_kws, trend_matrix_norm, adv_trend_matrix_norm = generate_keyword_trend_matrix((doc, kw_dict), i, 11, 5)
        recovery_queries = generate_queries(trend_matrix_norm[:, 1:], 'possion', 10000)
        observed_queries = generate_queries(trend_matrix_norm[:, :1], 'possion', 10000)


        #chosen_kws = sorted(chosen_kws, key=lambda k: trend_matrix_norm[k][0])
       

        for kws_leak_percent in (1, 1):#(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1):
            attack = leakage.Leakage(cli_doc, adv_doc, chosen_kws, recovery_queries, trend_matrix_norm, adv_trend_matrix_norm)
            WLP_F = attack.WLP(recovery_queries, observed_queries, chosen_kws, cli_doc, adv_doc, trend_matrix_norm, adv_trend_matrix_norm, kws_leak_percent)
            WLP_F.freq_wlp_main()
            print(WLP_F.inject_time)
            print(WLP_F.time)
            print(WLP_F.inject_time+WLP_F.time)
            print(WLP_F.accuracy)
            print(WLP_F.total_inject_size)
            print()
        """
        blackstone = attack.Blackstone(cli_doc, adv_doc, chosen_kws, recovery_queries, observed_queries, 0, 10)
        blackstone.improved_decoding_main()
        print(blackstone.inject_time)
        print(blackstone.time)
        print(blackstone.inject_time+blackstone.time)
        print(blackstone.accuracy)
        print(blackstone.total_inject_size)
        print()
        print()
        """
        
        
        #F = attack.Frequency(recovery_queries, chosen_kws, cli_doc, adv_doc, trend_matrix_norm, adv_trend_matrix_norm, {})
        #F.frequency_main()
        #print(F.accuracy)
        """
        zhang = attack.Zhang(23, attack.chosen_kws, attack.queries, 0, 0)
        zhang.improved_main()
        print(zhang.accuracy)
        print(zhang.totals_inject_doc)
        print(zhang.time)
        zhang.improved_2_main()
        print(zhang.accuracy)
        print(zhang.totals_inject_doc)
        print(zhang.time)
        """
        
        #return_kw_id, return_kw, total_divide, kw_num_freq = utils.build_new_same_freq_kws(chosen_kws, trend_matrix_norm[:, 0])
        #print(kw_num_freq)
        #print()
        #print(return_kw_id)
        #print()
        #combine_kw_id = utils.kw_id_to_combine(return_kw_id)
        #print(combine_kw_id)
        #freqfreq = utils.combine_kw_freq(return_kw_id)
    
        #observed_each_doc_word_length, observed_volume = utils.compute_combine_volume(freqfreq, chosen_kws, cli_doc)
        #observed_each_doc_word_length, observed_volume = utils.compute_original_volume(chosen_kws, cli_doc)
        #print(observed_each_doc_word_length)
        #print(observed_volume)
        #print()
        #freq = utils.combine_kw_freq(return_kw_id)
        #print(freq)
        
        
    
        # 0.1
        #print(len(attack.queries))
        #print(len(attack.trend_matrix_norm)) # 横keyword， 纵week
        """
        frequency = attack.Frequency(recovery_queries, chosen_kws, cli_doc, adv_doc, trend_matrix_norm, adv_trend_matrix_norm)
        frequency.frequency_main()
        print(frequency.accuracy)
        print(frequency.time)
        frequency.frequency_main2()
        print(frequency.accuracy)
        print(frequency.time)
        frequency.frequency_attack_with_improved_scheme(combine_kw_id)
        print(frequency.accuracy)
        print(frequency.time)
        """
        
        #count = attack.Count(adv_doc, cli_doc, chosen_kws, recovery_queries)
        #count.count_main()
        #print(count.accuracy)
        #print(count.time)
        #blackstone = attack.Blackstone(cli_doc, adv_doc, chosen_kws, recovery_queries, observed_queries, 0, 10)
        #blackstone.sel_word_length_attack()
        #blackstone.subgraph_main("access_doc_id")
        #print(blackstone.accuracy)
        #print(blackstone.time)
        #blackstone.subgraph_main("word_length")

        #print(blackstone.accuracy)
        #print(blackstone.time)
        #count.build_adv_co_occurrence()
        #count.build_query_co_occurrence()
        """
        count.count2()
        count.accuracy
        blackstone = attack.Blackstone(cli_doc, adv_doc, chosen_kws, recovery_queries, observed_queries, 0, 10)
        blackstone.sel_word_length_attack()
        blackstone.subgraph_attack("access_doc_id")
        blackstone.subgraph_attack("word_length")
        """
        
        """
        known_queries = {}
        matching_queries = []
        for i in range(int (len(attack.queries)/2)):
            known_queries[i] = i
            matching_queries.append(i)

        ikk = attack.IKK2(attack.adv_doc, attack.cli_doc, attack.chosen_kws, attack.queries, known_queries,  matching_queries)
        ikk.IKK_attack2()
        """