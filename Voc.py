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
Data_name = ['Lucene']#, 'Lucene', 'Wiki']
Query_dis = ['Trend', 'Random']
Test_name = ['Rer']#, 'ILen', 'ISize']
for ds in Data_name:
        for qu in Query_dis:
                for tst in Test_name:
                        res = pandas.read_csv(r'C:\Users\admin\yan\Attack-evaluation\results_csv\{}\{}KwsAndObserved{}.csv'.format(ds, ds, qu))
                        #print(res.keys())
                        kws_leak_percent = res['kws_leak_percent']
                        kws_leak_percent = kws_leak_percent[:11]
                        print(kws_leak_percent)
                        observed_weeks = res['observed_weeks']
                        OB_in = 0
                        if ds == 'Enron' or ds == 'Wiki':
                                OB_in = 5
                        else:
                                OB_in = 6
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

                        k_p = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

                        for i in range(len(observed_weeks)):
                                for j in range(len(kws_leak_percent)):
                                        #DA_inject_num[i][j] = log2((2**DA_inject_num[i][j])*k_p[j])
                                        BDA_inject_num[i][j] = log2((2**BDA_inject_num[i][j])*k_p[j])
                                        BM2A_inject_num[i][j] = log2((2**BM2A_inject_num[i][j])*k_p[j])
                                        if i+1<len(observed_weeks):
                                                if DA_inject_num[i][j] > DA_inject_num[i+1][j]:
                                                        DA_inject_num[i][j] = DA_inject_num[i+1][j]
                        mpl.rcParams['legend.fontsize'] = 10
                        #print(DA_inject_num)


                        fig = plt.figure()
                        ax = fig.gca(projection='3d')
                        #theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
                        y = observed_weeks
                        x = kws_leak_percent

                        x,y = np.meshgrid(x,y)
                        #print(x)
                        #print(y)
                        tetsize = 15
                        remarksize = 15
                        if tst == 'Rer':
                                
                                z1 = (BSA_recover_rate[:,:])
                                z2 = (DA_recover_rate[:,:])
                                z3 = (BDA_recover_rate[:,:])
                                z4 = (BM2A_recover_rate[:,:])

                                ax.set_xlabel('Leakage percentage', fontsize = tetsize, labelpad = 5)
                                if ds == 'Wiki':
                                        ax.set_ylabel('Observed months', fontsize = tetsize, labelpad = 5)
                                else:
                                        ax.set_ylabel('Observed weeks', fontsize = tetsize, labelpad = 5)
                                ax.set_zlabel('Recovery rate', fontsize = 15)#  Injection size (log)    Recovery accuracy
                                #ax.set_zlabel('Injection length (log)', fontsize = 15)#  Injection size (log)    Recovery accuracy
                                #ax.set_zlabel('Injection size (log) ', fontsize = tetsize)#  Injection size (log)    Recovery accuracy
                                #ax.set_zlim3d(10**(-2), 10**0)
                                #ax.set_zscale('log')
                                ax.tick_params(labelsize=tetsize)
                                #ax.grid(False)
                                z2max = '%.3f' % DA_recover_rate[len(DA_recover_rate)-1][len(DA_recover_rate[0])-1]
                                z3max = '%.3f' % BDA_recover_rate[len(BDA_recover_rate)-1][len(BDA_recover_rate[0])-1]
                                z4max = '%.3f' % BM2A_recover_rate[len(BM2A_recover_rate)-1][len(BM2A_recover_rate[0])-1]
                                lz2 = '(%.1f, %d, %.3f)' % (x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], DA_recover_rate[len(DA_recover_rate)-1][len(DA_recover_rate[0])-1])
                                lz3 = '(%.1f, %d, %.3f)' % (x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], BDA_recover_rate[len(BDA_recover_rate)-1][len(BDA_recover_rate[0])-1])
                                lz4 = '(%.1f, %d, %.3f)' % (x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], BM2A_recover_rate[len(BM2A_recover_rate)-1][len(BM2A_recover_rate[0])-1])
                                ax.text(x[len(x)-1][len(x[0])-1]-0.55, y[len(y)-1][len(y[0])-1], z2[len(z2)-1][len(z2[0])-1], lz2, color='red', fontsize = remarksize)
                                ax.text(x[len(x)-1][len(x[0])-1]-0.8, y[len(y)-1][len(y[0])-1], z3[len(z3)-1][len(z3[0])-1]-0.3, lz3, color='green', fontsize = remarksize)
                                ax.text(x[len(x)-1][len(x[0])-1]-0.55, y[len(y)-1][len(y[0])-1], z4[len(z4)-1][len(z4[0])-1]-0.5, lz4, color='blue', fontsize = remarksize)
                                ax.scatter(x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z2[len(z2)-1][len(z2[0])-1], color='red', marker = 'v', s = 64)
                                ax.scatter(x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z3[len(z3)-1][len(z3[0])-1], color='green', marker = '<', s = 64)
                                ax.scatter(x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z4[len(z4)-1][len(z4[0])-1], color='blue', marker = '>', s = 64)

                                surf = ax.plot_wireframe(x,
                                        y,
                                        z2,
                                        color = 'lightsalmon',
                                        #cmap = 'Reds',
                                        label = 'DA'
                                        )
                                #surf._facecolors2d = surf._facecolor3d
                                #surf._edgecolors2d = surf._edgecolor3d
                                surf = ax.plot_wireframe(x,
                                        y,
                                        z3,
                                        color = 'lightgreen',
                                        #cmap = 'Greens',
                                        label = 'BDA'
                                        )
                                #surf._facecolors2d = surf._facecolor3d
                                #surf._edgecolors2d = surf._edgecolor3d

                                #lightsalmon lightgreen lightblue

                                surf = ax.plot_wireframe(x,
                                        y,
                                        z4,
                                        color = 'lightblue',
                                        #cmap = 'Blues',
                                        label = 'BM2A'
                                        )
                                #surf._facecolors2d = surf._facecolor3d
                                #surf._edgecolors2d = surf._edgecolor3d
                        elif tst == 'ILen':
                                z1 = np.log2(BSA_inject_num[:,:])
                                z2 = np.log2(DA_inject_num[:,:])
                                z3 = np.log2(BDA_inject_num[:,:])+0.2
                                z4 = np.log2(BM2A_inject_num[:,:])
                                ax.set_xlabel('Leakage percentage', fontsize = tetsize, labelpad = 5)
                                if ds == 'Wiki':
                                        ax.set_ylabel('Observed months', fontsize = tetsize, labelpad = 5)
                                else:
                                        ax.set_ylabel('Observed weeks', fontsize = tetsize, labelpad = 5)
                                
                                #ax.set_zlabel('Recovery rate', fontsize = 15)#  Injection size (log)    Recovery accuracy
                                ax.set_zlabel('Injection length (log)', fontsize = 15)#  Injection size (log)    Recovery accuracy
                                #ax.set_zlabel('Injection size (log) ', fontsize = tetsize)#  Injection size (log)    Recovery accuracy
                                ax.tick_params(labelsize=tetsize)

                                z2max = math.ceil(z2[len(z2)-1][len(z2[0])-1])
                                z3max = math.ceil(z4[len(z4)-1][len(z4[0])-1])
                                z4max = math.ceil(z4[len(z4)-1][len(z4[0])-1])
                                lz2 = '(%.1f, %d, %d)' % (x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z2max)
                                lz3 = '(%.1f, %d, %d)' % (x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z3max)
                                lz4 = '(%.1f, %d, %d)' % (x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z4max)
                                ax.text(x[len(x)-1][len(x[0])-1]-0.5, y[len(y)-1][len(y[0])-1], z2[len(z2)-1][len(z2[0])-1], lz2, color='red', fontsize = remarksize)
                                ax.text(x[len(x)-1][len(x[0])-1]-0.5, y[len(y)-1][len(y[0])-1], z3[len(z3)-1][len(z3[0])-1]+1, lz3, color='green', fontsize = remarksize)
                                ax.text(x[len(x)-1][len(x[0])-1]-0.5, y[len(y)-1][len(y[0])-1], z4[len(z4)-1][len(z4[0])-1], lz4, color='blue', fontsize = remarksize)
                                ax.scatter(x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z2[len(z2)-1][len(z2[0])-1], color='red', marker = 'v', s = 64)
                                ax.scatter(x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z3[len(z3)-1][len(z3[0])-1], color='green', marker = '<', s = 64)
                                ax.scatter(x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z4[len(z4)-1][len(z4[0])-1], color='blue', marker = '>', s = 64)

                                surf = ax.plot_wireframe(x,
                                        y,
                                        z2,
                                        color = 'lightsalmon',
                                        #cmap = 'Reds',
                                        label = 'DA'
                                        )
                                #surf._facecolors2d = surf._facecolor3d
                                #surf._edgecolors2d = surf._edgecolor3d
                                surf = ax.plot_wireframe(x,
                                        y,
                                        z3,
                                        color = 'lightgreen',
                                        #cmap = 'Greens',
                                        label = 'BDA'
                                        )
                                #surf._facecolors2d = surf._facecolor3d
                                #surf._edgecolors2d = surf._edgecolor3d

                                #lightsalmon lightgreen lightblue

                                surf = ax.plot_wireframe(x,
                                        y,
                                        z4,
                                        color = 'lightblue',
                                        #cmap = 'Blues',
                                        label = 'BM2A'
                                        )
                                #surf._facecolors2d = surf._facecolor3d
                                #surf._edgecolors2d = surf._edgecolor3d

                        else:
                                z1 = np.log2(BSA_inject_size[:,:])
                                z2 = np.log2(DA_inject_size[:,:])
                                z3 = np.log2(BDA_inject_size[:,:])
                                z4 = np.log2(BM2A_inject_size[:,:])
                                ax.set_xlabel('Leakage percentage', fontsize = tetsize, labelpad = 5)
                                if ds == 'Wiki':
                                        ax.set_ylabel('Observed months', fontsize = tetsize, labelpad = 5)
                                else:
                                        ax.set_ylabel('Observed weeks', fontsize = tetsize, labelpad = 5)
                                
                                #ax.set_zlabel('Recovery rate', fontsize = 15)#  Injection size (log)    Recovery accuracy
                                #ax.set_zlabel('Injection length (log)', fontsize = 15)#  Injection size (log)    Recovery accuracy
                                ax.set_zlabel('Injection size (log) ', fontsize = tetsize)#  Injection size (log)    Recovery accuracy
                                ax.tick_params(labelsize=tetsize)
                                
                                z2max = math.ceil(z2[len(z2)-1][len(z2[0])-1])
                                z3max = math.ceil(z3[len(z3)-1][len(z3[0])-1])
                                z4max = math.ceil(z4[len(z4)-1][len(z4[0])-1])
                                lz2 = '(%.1f, %d, %d)' % (x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z2max)
                                lz3 = '(%.1f, %d, %d)' % (x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z3max)
                                lz4 = '(%.1f, %d, %d)' % (x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z4max)
                                ax.text(x[len(x)-1][len(x[0])-1]-0.5, y[len(y)-1][len(y[0])-1], z2[len(z2)-1][len(z2[0])-1], lz2, color='red', fontsize = remarksize)
                                ax.text(x[len(x)-1][len(x[0])-1]-0.5, y[len(y)-1][len(y[0])-1], z3[len(z3)-1][len(z3[0])-1], lz3, color='green', fontsize = remarksize)
                                ax.text(x[len(x)-1][len(x[0])-1]-0.5, y[len(y)-1][len(y[0])-1], z4[len(z4)-1][len(z4[0])-1], lz4, color='blue', fontsize = remarksize)
                                ax.scatter(x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z2[len(z2)-1][len(z2[0])-1], color='red', marker = 'v', s = 64)
                                ax.scatter(x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z3[len(z3)-1][len(z3[0])-1], color='green', marker = '<', s = 64)
                                ax.scatter(x[len(x)-1][len(x[0])-1], y[len(y)-1][len(y[0])-1], z4[len(z4)-1][len(z4[0])-1], color='blue', marker = '>', s = 64)
                                """
                                surf = ax.plot_surface(x,
                                        y,
                                        z1,
                                        color = 'y',
                                        label = 'BSA'
                                        )
                                surf._facecolors2d = surf._facecolor3d
                                surf._edgecolors2d = surf._edgecolor3d
                                """
                                surf = ax.plot_wireframe(x,
                                        y,
                                        z2,
                                        color = 'lightsalmon',
                                        #cmap = 'Reds',
                                        label = 'DA'
                                        )
                                #surf._facecolors2d = surf._facecolor3d
                                #surf._edgecolors2d = surf._edgecolor3d
                                surf = ax.plot_wireframe(x,
                                        y,
                                        z3,
                                        color = 'lightgreen',
                                        #cmap = 'Greens',
                                        label = 'BDA'
                                        )
                                #surf._facecolors2d = surf._facecolor3d
                                #surf._edgecolors2d = surf._edgecolor3d

                                #lightsalmon lightgreen lightblue

                                surf = ax.plot_wireframe(x,
                                        y,
                                        z4,
                                        color = 'lightblue',
                                        #cmap = 'Blues',
                                        label = 'BM2A'
                                        )
                                #surf._facecolors2d = surf._facecolor3d
                                #surf._edgecolors2d = surf._edgecolor3d

                        import matplotlib.patches as mpatches
                        r_patch = mpatches.Patch(facecolor='lightsalmon', edgecolor = 'red', linewidth = 0.8, label='DA')
                        g_patch = mpatches.Patch(facecolor='lightgreen', edgecolor = 'green', linewidth = 0.8, label='BDA')
                        b_patch = mpatches.Patch(facecolor='lightblue', edgecolor = 'blue', linewidth = 0.8, label='BM2A')


                        ax.legend(handles=[r_patch, g_patch, b_patch], loc="upper left",  bbox_to_anchor=(0, 1.15), fontsize = tetsize)#cmap = ['Greys', 'Reds', 'Greens', 'Blues'],

                        """
                        投影y
                        """
                        #gai
                        x_zero = [[0]*len(x[0]),
                        [0]*len(x[0]),
                        [0]*len(x[0]),
                        [0]*len(x[0]),
                        [0]*len(x[0])]
                        if OB_in == 6:
                                x_zero.append([0]*len(x[0]))
                        #print(z3)
                        """
                        zpro1 = np.zeros((len(z1), len(z1[0])))
                        for i in range(len(z1)):
                                for j in range(len(z1[i])):
                                        zpro1[i][j] = z1[i][len(z1[i])-1]
                        """
                        zpro2 = np.zeros((len(z2), len(z2[0])))
                        for i in range(len(z2)):
                                for j in range(len(z2[i])):
                                        zpro2[i][j] = z2[i][len(z2[i])-1]
                        zpro3 = np.zeros((len(z3), len(z3[0])))
                        for i in range(len(z3)):
                                for j in range(len(z3[i])):
                                        zpro3[i][j] = z3[i][len(z3[i])-1]
                        zpro4 = np.zeros((len(z4), len(z4[0])))
                        for i in range(len(z4)):
                                for j in range(len(z4[i])):
                                        zpro4[i][j] = z4[i][len(z4[i])-1]




                        """
                        ax.plot_wireframe(x_zero,
                                y,
                                zpro1,
                                color = 'y')

                        ax.plot_wireframe(x_zero,
                                y,
                                zpro2,
                                color = 'r')
                        ax.plot_wireframe(x_zero,
                                y,
                                zpro3,
                                color = 'g')

                        ax.plot_wireframe(x_zero,
                                y,
                                zpro4,
                                color = 'b')

                        """
                        
                        """
                        投影x
                        """
                        #gai
                        y_zero = [[0]*len(y[0]),
                        [0]*len(y[0]),
                        [0]*len(y[0]),
                        [0]*len(y[0]),
                        [0]*len(y[0])]
                        if OB_in == 6:
                                y_zero.append([0]*len(y[0]))
                        """
                        zypro1 = np.zeros((len(z1), len(z1[0])))
                        for i in range(len(z1)):
                                for j in range(len(z1[i])):
                                        zypro1[i][j] = z1[len(z1)-1][j]
                        ax.plot_wireframe(x,
                                y_zero,
                                zypro1,
                                color = 'y')

                        zypro2 = np.zeros((len(z2), len(z2[0])))
                        for i in range(len(z2)):
                                for j in range(len(z2[i])):
                                        zypro2[i][j] = z2[len(z2)-1][j]
                        ax.plot_wireframe(x,
                                y_zero,
                                zypro2,
                                color = 'r')
                        zypro3 = np.zeros((len(z3), len(z3[0])))
                        for i in range(len(z3)):
                                for j in range(len(z3[i])):
                                        zypro3[i][j] = z3[len(z3)-1][j]
                        ax.plot_wireframe(x,
                                y_zero,
                                zypro3,
                                color = 'g')

                        zypro4 = np.zeros((len(z4), len(z4[0])))
                        for i in range(len(z4)):
                                for j in range(len(z4[i])):
                                        zypro4[i][j] = z4[len(z4)-1][j]
                        ax.plot_wireframe(x,
                                y_zero,
                                zypro4,
                                color = 'b')
                        """
                        


                        width = 5
                        height = width
                        fig.set_size_inches(width, height)
                        plt.rcParams['savefig.dpi'] = 300 #图片像素
                        plt.rcParams['figure.dpi'] = 300 #分辨率
                        #ax.contour(x, y, z)
                        #plot_wireframe plot_surface
                        #ax.view_init(elev=34,    # 仰角
                        #        azim=44  # 方位角
                        #        )






                        plt.savefig('{}_{}_{}.pdf'.format(ds, qu, tst), bbox_inches = 'tight')