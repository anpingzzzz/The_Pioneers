import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
np.random.seed(0)
'''
This file generates Fig. S14 used in our paper. We set different change frequencies 
in simulation and see the results of NCEs and gains.
'''

hybrid = pd.read_csv('output3/org_hybrid_entropy.csv' )['H(pos|user)/H(pos)']
self_org = pd.read_csv('output3/org_self_entropy.csv' )['H(pos|user)/H(pos)']
centralized = pd.read_csv('output3/centralized_entropy_100sup.csv' )['H(pos|user)/H(pos)']

hybrid0 = pd.read_csv('output3/hybrid_entropy_1change.csv' )['H(pos|user)/H(pos)']
self_org0 = pd.read_csv('output3/self_entropy_1change.csv' )['H(pos|user)/H(pos)']
centralized0 = pd.read_csv('output3/centralized_entropy_1change.csv' )['H(pos|user)/H(pos)']

hybrid1 = pd.read_csv('output3/hybrid_entropy_5change.csv' )['H(pos|user)/H(pos)']
self_org1 = pd.read_csv('output3/self_entropy_5change.csv' )['H(pos|user)/H(pos)']
centralized1 = pd.read_csv('output3/centralized_entropy_5change.csv' )['H(pos|user)/H(pos)']




ave_gain_self0 = np.load('output3/gain_self_1change.npy')
ave_gain_hybrid0 = np.load('output3/gain_hybrid_1change.npy')
ave_gain_centralized0 =  np.load('output3/gain_centralized_1change.npy')

ave_gain_self = np.load('output3/gain_self_org.npy')
ave_gain_hybrid = np.load('output3/gain_hybrid_org.npy')
ave_gain_centralized =  np.load('output3/gain_centralized_org.npy')

ave_gain_self1 = np.load('output3/gain_self_5change.npy')
ave_gain_hybrid1 = np.load('output3/gain_hybird_5change.npy')
ave_gain_centralized1 =  np.load('output3/gain_centralized_5change.npy')

plt.figure(figsize=(20,8))
plt.tick_params(labelsize=30)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)


plt.subplot(2,3,1)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot( centralized0, 'green',label= 'Centralized Scheme',linewidth=2.5)
plt.plot( self_org0, 'red',label= 'Self-organized Scheme',linewidth=2.5)
plt.plot( hybrid0, 'blue',label= 'Hybrid Scheme',linewidth=2.5)

plt.text(-40, 0.2, "NCEs", size = 20)

plt.legend(bbox_to_anchor=(1,1),
                 loc=3,#图例的位置
                 ncol=3,#列数
                 mode="None",#当值设置为“expend”时，图例会水平扩展至整个坐标轴区域
                 borderaxespad=0,#坐标轴和图例边界之间的间距
                 #title="System Type",#图例标题
                 shadow=False,#是否为线框添加阴影
                 fontsize=15,
                 fancybox=True)#线框圆角处理参数
plt.subplot(2,3,2)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot( centralized, 'green',label= 'Centralized Scheme',linewidth=2.5)
plt.plot( self_org, 'red',label= 'Self-organized Scheme',linewidth=2.5)
plt.plot( hybrid, 'blue',label= 'Hybrid Scheme',linewidth=2.5)




plt.subplot(2,3,3)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot( centralized1, 'green',label= 'Centralized Scheme',linewidth=2.5)
plt.plot( self_org1, 'red',label= 'Self-organized Scheme',linewidth=2.5)
plt.plot( hybrid1, 'blue',label= 'Hybrid Scheme',linewidth=2.5)



plt.subplot(2,3,4)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot( ave_gain_centralized0, 'green',label= 'Centralized Scheme',linewidth=2.5)
plt.plot( ave_gain_self0, 'red',label= 'Self-organized Scheme',linewidth=2.5)
plt.plot( ave_gain_hybrid0, 'blue',label= 'Hybrid Scheme',linewidth=2.5)
plt.text(-40, 750, "Gains", size = 18)
plt.text(25, 120, "Frequency = 2", size = 18)

plt.subplot(2,3,5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot( ave_gain_centralized, 'green',label= 'Centralized Scheme',linewidth=2.5)
plt.plot( ave_gain_self, 'red',label= 'Self-organized Scheme',linewidth=2.5)
plt.plot( ave_gain_hybrid, 'blue',label= 'Hybrid Scheme',linewidth=2.5)
plt.text(25, 455, "Frequency = 3", size = 18)

plt.subplot(2,3,6)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot( ave_gain_centralized1, 'green',label= 'Centralized Scheme',linewidth=2.5)
plt.plot( ave_gain_self1, 'red',label= 'Self-organized Scheme',linewidth=2.5)
plt.plot( ave_gain_hybrid1, 'blue',label= 'Hybrid Scheme',linewidth=2.5)
plt.text(25, 427, "Frequency = 5", size = 18)




plt.savefig('output3/change_all_100sup.pdf')
plt.show()


