# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 15:39:37 2019

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir("F://ds")

Period=pd.read_csv("C:\\Users\\HP\\Downloads\\internship\\batch\\4.HEALTHCARE\\Period.csv")

Symptom=pd.read_csv("C:\\Users\\HP\\Downloads\\internship\\batch\\4.HEALTHCARE\\Symptom.csv")

User=pd.read_csv("C:\\Users\\HP\\Downloads\\internship\\batch\\4.HEALTHCARE\\User.csv")

pr=Period.copy()
sy=Symptom.copy()
us=User.copy()

Period.shape
Period.columns
Period.pop('id')
Period.pop('User_id')

Symptom.columns
Symptom.pop('id')
Symptom.pop('user_id')
Symptom.pop('date')
Symptom.shape

User.columns
User.pop('id')
User.pop('dob')
User.shape


Symptom.replace(0,np.nan,inplace=True)
"{0} - {1}".format(1, 1 + 4)
labels=["{0} - {1}".format(i, i + 4)for i in range(1, 101, 5)]
Symptom.columns
col=Symptom.columns.tolist()

for cl in col:
    Symptom[cl]=pd.cut(Symptom[cl],range(1, 106, 5),right=False,labels=labels)
Symptom=Symptom.astype(str)
Symptom.replace('nan','no_issue',inplace=True)
Symptom.info()

Symptom[cl]+'_'+str(cl)

for cl in col:
    Symptom[cl]=Symptom[cl]+'_'+str(cl)
    
%matplotlib inline
for cl in Symptom:
    Symptom[cl].value_counts().sort_values(ascending=True).plot.barh()
    plt.show()

for cl in Symptom:
    Symptom[Symptom[cl]!='no_issue_'+str(cl)][cl].value_counts().sort_values(ascending=True).plot.barh()
    plt.title(str(cl))
    plt.show()
    Symptom[Symptom[cl]=='no_issue_'+str(cl)][cl].value_counts().sort_values(ascending=True).plot.bar()
    plt.title(str(cl))
    plt.show()

for cl in User:
    User[cl]=pd.cut(User[cl],range(1, 106, 5),right=False,labels=labels)

User=User.astype(str)
for col in User:
    User[col].value_counts().sort_values(ascending=True).plot.bar()
    plt.title(str(col))
    plt.show()


User.columns
User.cycle_length_initial.unique()
User['id']=us.id
Symptom['user_id']=sy.user_id
data_final=Symptom.merge(User,left_on='user_id',right_on='id')
data_final.columns
data_final.cycle_length_initial=data_final.cycle_length_initial+'_'+'cycle_length'
data_final.period_length_initial=data_final.period_length_initial+'_'+'period_length'
data_final.columns
data_final_1=data_final.copy()
data_final_1.columns
data_final_1.pop('user_id')
data_final_1.pop('id')

records=[]
for i in range(0,13512):
    records.append([str(data_final_1.values[i,j]) for j in range(0, 12)])

#to remove nan values

l_new=[]
for lis_outter in records:
    l_temp=[]
    for val in lis_outter:
        if val !='nan':
            l_temp.append(val)
    l_new.append(l_temp)
l=l_new
records=l_new


#to remove duplicate values
    
records_new=[]
for l in records:
    final_list=[]
    for num in l:
        if num not in final_list:
            final_list.append(num)
    records_new.append(final_list)
    


from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
    
te=TransactionEncoder()
te_data=te.fit(records_new).transform(records_new)
data_x=pd.DataFrame(te_data,columns=te.columns_)
print(data_x.head()) 

 
frequent_pains=apriori(data_x,use_colnames=True,min_support=0.0045)
rules=association_rules(frequent_pains,metric='lift',min_threshold=10)
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot  as plt
rules.columns
x=rules.support
y=rules.confidence
z=rules.lift

%matplotlib
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('x Label support')
ax.set_ylabel('y Label confidence')
ax.set_zlabel('z Label lift')
plt.show()
