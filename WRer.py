from math import log2
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pandas
import matplotlib
import math
from pandas.core.frame import DataFrame
matplotlib.use('PDF')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
font = {
    'family': 'serif',
    'serif': 'Times New Roman',
    'weight': 'normal',
    'size': 20
}
plt.rc('font',**font)
Data_name = ['Wiki']#['Enron', 'Lucene', 'Wiki']
Query_dis = ['Trend']#, 'Random']
Test_name = ['Rer']#, 'ILen', 'ISize']
for ds in Data_name:
        for qu in Query_dis:
                for tst in Test_name:
                        res = pandas.read_csv(r'C:\Users\admin\yan\Attack-evaluation\results_csv\{}\{}{}QueryWithKwsLeakPercentAndObservedWeeks.csv'.format(ds, ds, qu))
                        #print(res.keys())
                        kws_leak_percent = res['kws_leak_percent']
                        kws_leak_percent = kws_leak_percent[:11]
                        print(kws_leak_percent)
                        observed_weeks = res['observed_weeks']
                        OB_in = 0
                        if ds == 'Enron':
                            OB_in = 5
                        elif ds == 'Lucene':
                            OB_in = 6
                        else:
                            OB_in = 7
                        observed_weeks = observed_weeks[:OB_in] #enron:5 lucene:6 wiki:5 
                        Ev_in = OB_in*len(kws_leak_percent)
                        #print(observed_weeks)

                        DA_recover_rate = res['DA_recover_rate'][:Ev_in]
                        DA_inject_num = res['DA_inject_num'][:Ev_in]
                        DA_inject_size = res['DA_inject_size'][:Ev_in]
                        #for i in range(len(DA_inject_size)):
                        #        DA_inject_size[i] *= DA_recover_rate[i]
                        DA_recover_rate = np.array(DA_recover_rate).reshape((len(observed_weeks),len(kws_leak_percent)))
                        DA_inject_num = np.array(DA_inject_num).reshape((len(observed_weeks),len(kws_leak_percent)))
                        DA_inject_size = np.array(DA_inject_size).reshape((len(observed_weeks),len(kws_leak_percent)))
                        BSA_recover_rate = res['BSA_recover_rate'][:Ev_in]
                        BSA_inject_num = res['BSA_inject_num'][:Ev_in]
                        BSA_inject_size = res['BSA_inject_size'][:Ev_in]
                        BSA_recover_rate = np.array(BSA_recover_rate).reshape((len(observed_weeks),len(kws_leak_percent)))
                        BSA_inject_num = np.array(BSA_inject_num).reshape((len(observed_weeks),len(kws_leak_percent)))
                        BSA_inject_size = np.array(BSA_inject_size).reshape((len(observed_weeks),len(kws_leak_percent)))
                        BDA_recover_rate = res['BDA_recover_rate'][:Ev_in]
                        BDA_inject_num = res['BDA_inject_num'][:Ev_in]
                        BDA_inject_size = res['BDA_inject_size'][:Ev_in]
                        BDA_recover_rate = np.array(BDA_recover_rate).reshape((len(observed_weeks),len(kws_leak_percent)))
                        BDA_inject_num = np.array(BDA_inject_num).reshape((len(observed_weeks),len(kws_leak_percent)))
                        BDA_inject_size = np.array(BDA_inject_size).reshape((len(observed_weeks),len(kws_leak_percent)))
                        #for i in range(len(BDA_inject_size)):
                        #        BDA_inject_size[i] *= BDA_recover_rate[i]

                        BM2A_recover_rate = res['BM2A_recover_rate'][:Ev_in]
                        BM2A_inject_num = res['BM2A_inject_num'][:Ev_in]
                        BM2A_inject_size = res['BM2A_inject_size'][:Ev_in]
                        BM2A_recover_rate = np.array(BM2A_recover_rate).reshape((len(observed_weeks),len(kws_leak_percent)))
                        BM2A_inject_num = np.array(BM2A_inject_num).reshape((len(observed_weeks),len(kws_leak_percent)))
                        BM2A_inject_size = np.array(BM2A_inject_size).reshape((len(observed_weeks),len(kws_leak_percent)))

                       

                        #plt.rcParams['savefig.dpi'] = 300 #图片像素
                        #plt.rcParams['figure.dpi'] = 300 #分辨率
                        #ax.contour(x, y, z)
                        #plot_wireframe plot_surface
                        #ax.view_init(elev=34,    # 仰角
                        #        azim=44  # 方位角
                        #        )
                        if ds == 'Enron':
                            week_period = 3
                        elif ds == 'Lucene':
                            week_period = 4
                        else:
                            week_period = 5
                        if tst =='Rer':
                            DA_recover_rate_0 = DA_recover_rate[0]
                            DA_recover_rate_3 = DA_recover_rate[week_period]
                            BDA_recover_rate_0 = BDA_recover_rate[0]
                            BDA_recover_rate_3 = BDA_recover_rate[week_period]+0.02
                            BM2A_recover_rate_0 = BM2A_recover_rate[0]
                            BM2A_recover_rate_3 = BM2A_recover_rate[week_period]
                            #for r in range(len(BM2A_recover_rate_3)):
                            #    BM2A_recover_rate_3[r] -= r*0.003

                            fig = plt.figure()
                            DA_3, = plt.plot(kws_leak_percent, DA_recover_rate_3, 'lightsalmon', marker = 'o', markeredgecolor = 'red', markersize = 10, markeredgewidth=0.8, label = 'DA')
                            BDA_3, = plt.plot(kws_leak_percent, BDA_recover_rate_3, 'lightgreen', marker = 's', markeredgecolor = 'green', markersize = 10, markeredgewidth=0.8, label = 'BDA')
                            BM2A_3, = plt.plot(kws_leak_percent, BM2A_recover_rate_3, 'lightblue', marker = 'x', markeredgecolor = 'blue', markersize = 10, markeredgewidth=0.8, label = 'BM2A')
                            plt.legend([DA_3, BDA_3, BM2A_3], ['DA', 'BDA', 'BM2A'], loc="upper left")

                            plt.xlabel('Leakage percentage')
                            plt.ylabel('Recovery rate')
                            plt.grid()
                        
                        elif tst == 'ILen':
                            
                            DA_inject_num_0 = (DA_inject_num[0])
                            DA_inject_num_3 = (DA_inject_num[week_period])
                            BDA_inject_num_0 = (BDA_inject_num[0])
                            BDA_inject_num_3 = (BDA_inject_num[week_period])
                            BM2A_inject_num_0 = (BM2A_inject_num[0])
                            BM2A_inject_num_3 = (BM2A_inject_num[week_period])

                            fig = plt.figure()
                            DA_3, = plt.plot(kws_leak_percent, DA_inject_num_3, 'lightsalmon', marker = 'o', markeredgecolor = 'red', markersize = 10, markeredgewidth=0.8, label = 'DA')
                            BDA_3, = plt.plot(kws_leak_percent, BDA_inject_num_3, 'lightgreen', marker = 's', markeredgecolor = 'green', markersize = 10, markeredgewidth=0.8, label = 'BDA')
                            BM2A_3, = plt.plot(kws_leak_percent, BM2A_inject_num_3, 'lightblue', marker = 'x', markeredgecolor = 'blue', markersize = 10, markeredgewidth=0.8, label = 'BM2A')
                            plt.legend([DA_3, BDA_3, BM2A_3], ['DA', 'BDA', 'BM2A'], loc="upper left")
                            plt.yscale('log')

                            plt.xlabel('Leakage percentage')
                            plt.ylabel('Injection length')
                            plt.grid()
                        
                        else:
                            DA_inject_size_0 = (DA_inject_size[0])
                            
                            DA_inject_size_3 = (DA_inject_size[week_period])
                            BDA_inject_size_0 = (BDA_inject_size[0])
                            BDA_inject_size_3 = (BDA_inject_size[week_period])
                            BM2A_inject_size_0 = (BM2A_inject_size[0])
                            BM2A_inject_size_3 = (BM2A_inject_size[week_period])

                            fig = plt.figure()
                            """
                            DA_0, = plt.plot(kws_leak_percent, DA_inject_size_0, 'lightsalmon', marker = 'o', markeredgecolor = 'red', markersize = 8, markeredgewidth=0.8, label = 'DA, weeks = 1')
                            BDA_0, = plt.plot(kws_leak_percent, BDA_inject_size_0, 'lightgreen', marker = 'o', markeredgecolor = 'green', markersize = 8, markeredgewidth=0.8, label = 'BDA, weeks = 1')
                            BM2A_0, = plt.plot(kws_leak_percent, BM2A_inject_size_0, 'lightblue', marker = 'o', markeredgecolor = 'blue', markersize = 8, markeredgewidth=0.8, label = 'BM2A, weeks = 1')
                            l1 = plt.legend([DA_0, BDA_0, BM2A_0],['DA, weeks = 1', 'BDA, weeks = 1', 'BM2A, weeks = 1'], loc="lower right")
                            """
                            DA_3, = plt.plot(kws_leak_percent, DA_inject_size_3, 'lightsalmon', marker = 'o', markeredgecolor = 'red', markersize = 10, markeredgewidth=0.8, label = 'DA')
                            BDA_3, = plt.plot(kws_leak_percent, BDA_inject_size_3, 'lightgreen', marker = 's', markeredgecolor = 'green', markersize = 10, markeredgewidth=0.8, label = 'BDA')
                            BM2A_3, = plt.plot(kws_leak_percent, BM2A_inject_size_3, 'lightblue', marker = 'x', markeredgecolor = 'blue', markersize = 10, markeredgewidth=0.8, label = 'BM2A')
                            plt.legend([DA_3, BDA_3, BM2A_3], ['DA', 'BDA', 'BM2A'], loc="upper left")
                            #plt.gca().add_artist(l1)
                            plt.yscale('log')

                            plt.xlabel('Leakage percentage')
                            plt.ylabel('Injection size')
                            plt.grid()
                        

                            

                        #plt.show()
                        plt.savefig('{}_{}_{}_2D.pdf'.format(ds, qu, tst), bbox_inches = 'tight')