from math import log2
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pandas
from pylab import *
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import matplotlib
from pandas import DataFrame
from scipy.interpolate import make_interp_spline
from scipy import interpolate
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
def Smooth(x, y):
        x_new = np.linspace(np.array(x).min(), np.array(x).max(), len(x)*50) #300 represents number of points to make between T.min and T.max
        #print(y_new)
        #func = interpolate.interp1d(x, y, kind='zero')
        y_power_smooth = make_interp_spline(x, y)(x_new)
        for i in range(-len(y_power_smooth)+1, -1):
                if y_power_smooth[-i] < y_power_smooth[-i-1]:
                        y_power_smooth[-i-1] = y_power_smooth[-i]#-0.0005
        return x_new, y_power_smooth

plt.rc('font',**font)
Data_name = ['Wiki']# 'Lucene', 'Wiki']
Query_dis = ['Trend', 'Random']
Alpha = [0.33, 0.5, 0.25, 0.25]
Beta = [0.33, 0.25, 0.5, 0.25]
Gama = [0.33, 0.25, 0.25, 0.5]

for ds in Data_name:
        for qu in Query_dis:
                for itt in range(len(Alpha)):
                        res = pandas.read_csv(r'D:\yan\Attack-evaluation\results_csv\{}\{}{}QueryWithKwsLeakPercentAndSizeThreshold.csv'.format(ds,ds,qu))
                        print(res.keys())
                        kws_leak_percent = res['kws_leak_percent']
                        kws_leak_percent = kws_leak_percent[:11]
                        #print(kws_leak_percent)
                        size_threshold = res['size_threshold']
                        if ds=='Enron':
                                size_threshold = size_threshold[:51] #week:4 size:10 10 11
                        if ds=='Lucene':
                                size_threshold = size_threshold[:56] #week:4 size:10 10 11
                        else:
                                size_threshold = size_threshold[:61] #week:4 size:10 10 11

                        #print(size_threshold)
                        BSA_recover_rate = res['BSA_recover_rate']#[:100]
                        BSA_inject_num = res['BSA_inject_num']#[:100]
                        BSA_inject_size = res['BSA_inject_size']#[:100]
                        DA_recover_rate = res['DA_recover_rate']#[:100]
                        DA_inject_num = res['DA_inject_num']#[:100]
                        DA_inject_size = res['DA_inject_size']#[:100]
                        BDA_recover_rate = res['BDA_recover_rate']#[:100]
                        BDA_inject_num = res['BDA_inject_num']#[:100]
                        BDA_inject_size = res['BDA_inject_size']#[:100]
                        BM2A_vol_recover_rate = res['BM2A_vol_recover_rate']#[:100]
                        BM2A_vol_inject_num = res['BM2A_vol_inject_num']#[:100]
                        BM2A_vol_inject_size = res['BM2A_vol_inject_size']#[:100]
                        BM2A_freq_recover_rate = res['BM2A_freq_recover_rate']#[:100]
                        BM2A_freq_inject_num = res['BM2A_freq_inject_num']#[:100]
                        BM2A_freq_inject_size = res['BM2A_freq_inject_size']#[:100]
                        BM2A_recover_rate = res['BM2A_recover_rate']#[:100]
                        BM2A_inject_num = res['BM2A_inject_num']#[:100]
                        BM2A_inject_size = res['BM2A_inject_size']#[:100]

                        DA_power = []
                        BDA_power = []
                        BM2A_vol_power = []
                        BM2A_freq_power = []
                        BM2A_power = []
                        BSA_power = []
                        HSA_power = []
                        IHSA_power = []
                        ap = 1
                        aip = 1
                        sp = 2
                        vp = 2
                        wlp = 2
                        dlp = 3
                        dsp = 3
                        totalp = 6

                        alpha = Alpha[itt]
                        beta = Beta[itt]
                        gama = Gama[itt]
                        """
                        print(log2(size_threshold[len(size_threshold)-1]))
                        print(log2(DA_inject_size[len(DA_inject_size)-1]))
                        print(log2(BDA_inject_size[len(BDA_inject_size)-1]))
                        print(log2(BM2A_vol_inject_size[len(BM2A_vol_inject_size)-1]))
                        print(log2(BM2A_freq_inject_size[len(BM2A_freq_inject_size)-1]))
                        """

                        for i in range(len(size_threshold)):
                                for j in range(len(kws_leak_percent)):
                                        t = j + i*len(kws_leak_percent)
                                        #size_threshold[i] += 2
                                        
                                        if DA_inject_size[t]>=size_threshold[i]:
                                                DA_inject_size[t]=size_threshold[i]
                                        if BDA_inject_size[t]>=size_threshold[i]:
                                                BDA_inject_size[t]=size_threshold[i]
                                        if BSA_inject_size[t]>=size_threshold[i]:
                                                BSA_inject_size[t]=size_threshold[i]
                                        if BM2A_inject_size[t]>=size_threshold[i]:
                                                BM2A_inject_size[t]=size_threshold[i]
                                        if kws_leak_percent[j] < 0.1:
                                                
                                                DA_power.append(alpha*wlp/totalp + beta*(1-log2(DA_inject_size[t]+2)/log2(size_threshold[i]+2)) + (1-alpha-beta)*DA_recover_rate[t])
                                                BDA_power.append(alpha*wlp/totalp + beta*(1-log2(BDA_inject_size[t]+2)/log2(size_threshold[i]+2)) + (1-alpha-beta)*BDA_recover_rate[t])
                                                BM2A_power.append(alpha*(wlp/totalp + vp*vp/(totalp*totalp) + sp*sp*sp/(totalp*totalp*totalp)) + beta*(1-log2(BM2A_inject_size[t]+2)/log2(size_threshold[i]+2)) + (1-alpha-beta)*BM2A_recover_rate[t])
                                                BSA_power.append(alpha*aip/totalp + beta*(1-log2(BSA_inject_size[t]+2)/log2(size_threshold[i]+2)) + (1-alpha-beta)*BSA_recover_rate[t])
                                        else:
                                                DA_power.append(alpha*wlp/totalp + beta*(1-log2(DA_inject_size[t]+2)/log2(size_threshold[i]+2)) + (1-alpha-beta)*DA_recover_rate[t])
                                                BDA_power.append(alpha*wlp/totalp + beta*(1-log2(BDA_inject_size[t]+2)/log2(size_threshold[i]+2)) + (1-alpha-beta)*BDA_recover_rate[t])
                                                BM2A_power.append(alpha*(wlp/totalp + vp*vp/(totalp*totalp) + sp*sp*sp/(totalp*totalp*totalp)) + beta*(1-log2(BM2A_inject_size[t]+2)/log2(size_threshold[i]+2)) + (1-alpha-beta)*BM2A_recover_rate[t])
                                                BSA_power.append(alpha*aip/totalp + beta*(1-log2(BSA_inject_size[t]+2)/log2(size_threshold[i]+2)) + (1-alpha-beta)*BSA_recover_rate[t])
                                                #HSA_
                                        """
                                        if DA_inject_size[t] <= 10:
                                                DA_power.append(alpha*wlp/totalp + (1-alpha-beta)*DA_recover_rate[t])
                                                BDA_power.append(alpha*wlp/totalp +  (1-alpha-beta)*BDA_recover_rate[t])
                                                BM2A_power.append(alpha*(wlp/totalp + vp*vp/(totalp*totalp) + sp*sp*sp/(totalp*totalp*totalp)) + (1-alpha-beta)*BM2A_recover_rate[t])
                                                BSA_power.append(alpha*aip/totalp + (1-alpha-beta)*BSA_recover_rate[t])
                                                #HSA_power.append(alpha*wlp/totalp + beta*(1-log(HSA_inject_size[i])/log(kws_leak_percent[i])) + (1-alpha-beta)*HSA_recover_rate[i])
                                                #IHSA_power.append(alpha*wlp/totalp + beta*(1-log(IHSA_inject_size[i])/log(kws_leak_percent[i])) + (1-alpha-beta)*IHSA_recover_rate[i])
                                        elif kws_leak_percent[j] < 0.1:
                                                DA_power.append(-0.05 + alpha*wlp/totalp + beta*(1-log2(DA_inject_size[t])/log2(size_threshold[i])) + (1-alpha-beta)*DA_recover_rate[t])
                                                BDA_power.append(-0.05 + alpha*wlp/totalp + beta*(1-log2(BDA_inject_size[t])/log2(size_threshold[i])) + (1-alpha-beta)*BDA_recover_rate[t])
                                                BM2A_power.append(-0.05 + alpha*(wlp/totalp + vp*vp/(totalp*totalp) + sp*sp*sp/(totalp*totalp*totalp)) + beta*(1-log2(BM2A_inject_size[t])/log2(size_threshold[i])) + (1-alpha-beta)*BM2A_recover_rate[t])
                                                BSA_power.append(-0.05 + alpha*aip/totalp + beta*(1-log2(BSA_inject_size[t])/log2(size_threshold[i])) + (1-alpha-beta)*BSA_recover_rate[t])
                                        else:
                                        """
                                #res['DA_inject_size'] = DA_inject_size
                                #res['BDA_inject_size'] = BDA_inject_size
                                #res['BSDA_inject_size'] = BSA_inject_size
                                #res['BM2A_inject_size'] = BM2A_inject_size
                                #export_csv = res.to_csv(r'C:\Users\admin\yan\Attack-evaluation\results_csv\Wiki\WikiTrendQueryWithKwsLeakPercentAndSizeThreshold2.csv', index = None, header=True)

                        BSA_recover_rate = np.array(BSA_recover_rate).reshape((len(size_threshold),len(kws_leak_percent)))
                        BSA_inject_num = np.array(BSA_inject_num).reshape((len(size_threshold),len(kws_leak_percent)))
                        BSA_inject_size = np.array(BSA_inject_size).reshape((len(size_threshold),len(kws_leak_percent)))
                        BSA_power = np.array(BSA_power).reshape((len(size_threshold),len(kws_leak_percent)))
                        DA_recover_rate = np.array(DA_recover_rate).reshape((len(size_threshold),len(kws_leak_percent)))
                        DA_inject_num = np.array(DA_inject_num).reshape((len(size_threshold),len(kws_leak_percent)))
                        DA_inject_size = np.array(DA_inject_size).reshape((len(size_threshold),len(kws_leak_percent)))
                        DA_power = np.array(DA_power).reshape((len(size_threshold),len(kws_leak_percent)))
                        BDA_recover_rate = np.array(BDA_recover_rate).reshape((len(size_threshold),len(kws_leak_percent)))
                        BDA_inject_num = np.array(BDA_inject_num).reshape((len(size_threshold),len(kws_leak_percent)))
                        BDA_inject_size = np.array(BDA_inject_size).reshape((len(size_threshold),len(kws_leak_percent)))
                        BDA_power = np.array(BDA_power).reshape((len(size_threshold),len(kws_leak_percent)))
                        BM2A_recover_rate = np.array(BM2A_recover_rate).reshape((len(size_threshold),len(kws_leak_percent)))
                        BM2A_inject_num = np.array(BM2A_inject_num).reshape((len(size_threshold),len(kws_leak_percent)))
                        BM2A_inject_size = np.array(BM2A_inject_size).reshape((len(size_threshold),len(kws_leak_percent)))
                        BM2A_power = np.array(BM2A_power).reshape((len(size_threshold),len(kws_leak_percent)))

                        mpl.rcParams['legend.fontsize'] = 10
                        
                        fig = plt.figure()
                        ax = fig.gca(projection='3d')

                        #theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
                        y = np.log2(size_threshold)
                        x = kws_leak_percent[1:]

                        x,y = np.meshgrid(x,y)
                        print(x)
                        print(y)
                        #z = np.array(x+y)
                        """

                        z1 = np.log2(DA_inject_size[:4,:])
                        z2 = np.log2(BDA_inject_size[:4,:])
                        z3 = np.log2(BM2A_vol_inject_size[:4,:])+0.5
                        z4 = np.log2(BM2A_freq_inject_size[:4,:])

                        z1 = np.log2(DA_inject_num[:4,:])
                        z2 = np.log2(BDA_inject_num[:4,:])+0.3
                        z3 = np.log2(BM2A_vol_inject_num[:4,:])+0.15
                        z4 = np.log2(BM2A_freq_inject_num[:4,:])

                        z1 = (BSA_recover_rate[:,:])
                        z2 = (DA_recover_rate[:,:])
                        z3 = (BDA_recover_rate[:,:])
                        z4 = (BM2A_recover_rate[:,:])

                        """

                        z1 = (BSA_power[:,1:])
                        z2 = (DA_power[:,1:])
                        z3 = (BDA_power[:,1:])
                        z4 = (BM2A_power[:,1:])

                        tetsize = 15
                        ax.set_xlabel('Leakage percentage', fontsize = tetsize, labelpad = 5)
                        ax.set_ylabel('Size threshold (log)', fontsize = tetsize, labelpad = 5)
                        ax.set_zlabel('Power', fontsize = tetsize)#  Injection volume (log) Recovery accuracy , labelpad = 10
                        ax.tick_params(labelsize = tetsize)

                        #ax.set_ylim(2**0, 2**4)
                        #ax.set_yscale('linear')
                        #ax.set_ylim(10**0, 10**1)
                        #ax.set_zlim(10**8, 10**10)
                        #ax.set_zscale('log')
                        #ax.set_yscale('log')

                        #r = z ** 2 + 1
                        """
                        投影y
                        """
                        x_zero = [-0.1]*len(size_threshold)
                        zpro1 = np.zeros(len(z1))
                        zpro2 = np.zeros(len(z2))
                        zpro3 = np.zeros(len(z3))
                        zpro4 = np.zeros(len(z4))
                        for i in range(len(z1)):
                                zpro1[i] = z1[i][len(z1[i])-1]
                        for i in range(len(z2)):
                                zpro2[i] = z2[i][len(z2[i])-1]
                        for i in range(len(z1)):
                                zpro3[i] = z3[i][len(z3[i])-1]
                        for i in range(len(z1)):
                                zpro4[i] = z4[i][len(z4[i])-1]

                        from scipy.interpolate import make_interp_spline
                        from scipy import interpolate
                        print(y[:, 0])
                        y_new = np.linspace(np.array(y[:, 0]).min(), np.array(y[:, 0]).max(), len(y[:, 0])*20) #300 represents number of points to make between T.min and T.max
                        #print(y_new)
                        #func = interpolate.interp1d(y[:, 0], zpro1, kind='zero')

                        y_new, zxpro1_power_smooth = Smooth(y[:, 0], zpro1)
                        _, zxpro2_power_smooth = Smooth(y[:, 0], zpro2)
                        _, zxpro3_power_smooth = Smooth(y[:, 0], zpro3)
                        _, zxpro4_power_smooth = Smooth(y[:, 0], zpro4)


                        ax.plot([-0.1]*len(y[10:, 0]),
                                y[10:, 0],
                                zpro1[10:],
                                #c = zxpro1_power_smooth,
                                #cmap = 'Greys',
                                #s = 2
                                color = 'lightgray'
                                )
                        ax.plot([-0.1]*len(y[10:, 0]),
                                y[10:, 0],
                                zpro2[10:],
                                #c = zxpro2_power_smooth,
                                #cmap = 'Reds',
                                #s = 2
                                color = 'lightsalmon'
                                )
                        ax.plot([-0.1]*len(y[10:, 0]),
                                y[10:, 0],
                                zpro3[10:],
                                #c = zxpro3_power_smooth,
                                #cmap = 'Greens',
                                #s = 2
                                color = 'lightgreen'
                                )
                        ax.plot([-0.1]*len(y[10:, 0]),
                                y[10:, 0],
                                zpro4[10:],
                                #c = zxpro4_power_smooth,
                                #cmap = 'Blues',
                                #s = 2
                                color = 'lightblue'
                                )


                        """
                        ax.plot(x_zero,
                                y[:, 0],
                                zpro1,
                                color = 'gray')
                        ax.plot(x_zero,
                                y[:, 0],
                                zpro2,
                                color = 'r')
                        ax.plot(x_zero,
                                y[:, 0],
                                zpro3,
                                color = 'g')
                        ax.plot(x_zero,
                                y[:, 0],
                                zpro4,
                                color = 'b') 
                        投影x
                        """
                        y_zero = [-0.1]*len(kws_leak_percent)
                        zypro1 = np.zeros(len(z1[0]))
                        zypro2 = np.zeros(len(z2[0]))
                        zypro3 = np.zeros(len(z3[0]))
                        zypro4 = np.zeros(len(z4[0]))
                        for j in range(len(z1[0])):
                                zypro1[j] = z1[len(z1)-1][j]
                        for j in range(len(z2[0])):
                                zypro2[j] = z2[len(z2)-1][j]
                        for j in range(len(z3[0])):
                                zypro3[j] = z3[len(z3)-1][j]
                        for j in range(len(z4[0])):
                                zypro4[j] = z4[len(z4)-1][j]

                        x_new = np.linspace(np.array(x[0, :]).min(), np.array(x[0, :]).max(), len(x[0, :])*20) #300 represents number of points to make between T.min and T.max
                        #print(y_new)
                        func = interpolate.interp1d(x[0, :], zypro1, kind='nearest')
                        zypro1_power_smooth = make_interp_spline(x[0, :], zypro1)(x_new)
                        zypro2_power_smooth = make_interp_spline(x[0, :], zypro2)(x_new)
                        zypro3_power_smooth = make_interp_spline(x[0, :], zypro3)(x_new)
                        zypro4_power_smooth = make_interp_spline(x[0, :], zypro4)(x_new)
                        ax.plot(x_new,
                                [-0.1]*len(x_new),
                                zypro1_power_smooth,
                                #c = zypro1_power_smooth,
                                #cmap = 'Greys',
                                color = 'lightgray',
                                #s = 2
                                )
                        ax.plot(x_new,
                                [-0.1]*len(x_new),
                                zypro2_power_smooth,
                                #c = zypro2_power_smooth,
                                ##cmap = 'Reds',
                                color = 'lightsalmon',
                                #s = 2
                                )
                        ax.plot(x_new,
                                [-0.1]*len(x_new),
                                zypro3_power_smooth,
                                #c = zypro3_power_smooth,
                                #cmap = 'Greens',
                                color = 'lightgreen',
                                #s = 2
                                )
                        ax.plot(x_new,
                                [-0.1]*len(x_new),
                                zypro4_power_smooth,
                                #c = zypro4_power_smooth,
                                #cmap = 'Blues',
                                color = 'lightblue',
                                #s = 2
                                )
                        """
                        ax.plot(x[0, :],
                                y_zero,
                                zypro1,
                                color = 'gray')
                        ax.plot(x[0, :],
                                y_zero,
                                zypro2,
                                color = 'r')
                        ax.plot(x[0, :],
                                y_zero,
                                zypro3,
                                color = 'g')
                        ax.plot(x[0, :],
                                y_zero,
                                zypro4,
                                color = 'b')
                        """

                        """
                        3D图
                        """
                        surf1 = ax.plot_surface(x, y, z1, rstride = 1,#rows stride：指定行的跨度为1（只能是int）
                                                cstride = 1,
                                                cmap = 'Greys',
                                                label = 'BSA')
                        surf1._facecolors2d = surf1._facecolor3d
                        surf1._edgecolors2d = surf1._edgecolor3d
                        surf2 = ax.plot_surface(x, y, z2, rstride = 1,#rows stride：指定行的跨度为1（只能是int）
                                                cstride = 1,
                                                cmap = 'Reds',
                                                label = 'DA')
                        surf2._facecolors2d = surf2._facecolor3d
                        surf2._edgecolors2d = surf2._edgecolor3d
                        surf3 = ax.plot_surface(x, y, z3, rstride = 1,#rows stride：指定行的跨度为1（只能是int）
                                                cstride = 1,
                                                cmap = 'Greens',
                                                label = 'BDA')
                        surf3._facecolors2d = surf3._facecolor3d
                        surf3._edgecolors2d = surf3._edgecolor3d
                        surf4 = ax.plot_surface(x, y, z4, rstride = 1,#rows stride：指定行的跨度为1（只能是int）
                                                cstride = 1,
                                                cmap = 'Blues',
                                                label = 'BM2A')
                        surf4._facecolors2d = surf4._facecolor3d
                        surf4._edgecolors2d = surf4._edgecolor3d


                        gray_patch = mpatches.Patch(edgecolor='gray', facecolor = 'lightgray', linewidth = 0.8, label='BSA')
                        r_patch = mpatches.Patch(edgecolor='red', facecolor = 'lightsalmon', linewidth = 0.8, label='DA')
                        g_patch = mpatches.Patch(edgecolor='green', facecolor = 'lightgreen', linewidth = 0.8, label='BDA')
                        b_patch = mpatches.Patch(edgecolor='blue', facecolor = 'lightblue', linewidth = 0.8, label='BM2A')


                        ax.legend(handles=[gray_patch, r_patch, g_patch, b_patch], loc="upper left",  bbox_to_anchor=(0, 1.15), fontsize = tetsize)#cmap = ['Greys', 'Reds', 'Greens', 'Blues'],


                        width = 5
                        height = width
                        fig.set_size_inches(width, height)
                        #plt.rcParams['savefig.dpi'] = 300 #图片像素
                        #plt.rcParams['figure.dpi'] = 300 #分辨率
                        #ax.contour(x, y, z)
                        #plot_wireframe plot_surface
                        ax.view_init(elev=33,    # 仰角
                                azim=32  # 方位角
                                )

                        plt.savefig('{}_{}_{}_{}_{}_3D.pdf'.format(ds, qu, alpha, beta, gama), bbox_inches = 'tight')