import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

# import factor analyzer library
from factor_analyzer import FactorAnalyzer

# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

#set working directory
os.chdir('C:/Users/Zedric Cheung/Desktop/Towards Data Science/5_Classification of economies in world')

#open raw data donwloaded from World Bank
data = pd.read_csv('raw data selected.csv')
data = data.sort_values(by=['Series Name', 'Country Name'])
data = data.set_index('Series Name')

indicators = sorted(set(data.index))
economies = sorted(set(data['Country Name']))
num_econ = len(economies)
columns = list(data.columns)
columns = columns[1:]
years = columns[2:]
yearlist = []

data = data[columns]

#set up a dataframe
df = pd.DataFrame(index = indicators, columns = economies)
datalog = pd.DataFrame(index = indicators)

#construct an usable dataframe
for indicator in indicators:
    
    year = 0
    
    #filtering out the indicators that too few countries provide
    for i in range(len(years)):
        if list(data.loc[indicator][years[i]] != '..').count(False) <= 35:
            if (year != 0) and (year != 1):
                if list(data.loc[indicator][years[i]] != '..').count(False) <= list(data.loc[indicator][years[year]] != '..').count(False):
                    year = i
            else:
                year = i
    
    #print the indicators and their latest years
    print(indicator, '-', years[year])
    yearlist.append(years[year])
    
    for economy in economies:
        try:
            #print(data.loc[data['Country Name'] == economy].loc[indicator].loc[years[year]])
            df.at[indicator, economy] = data.loc[data['Country Name'] == economy].loc[indicator].loc[years[year]]
        except:
            df.at[indicator, economy] = np.nan

#print the indicators and their years
datalog['Year'] = yearlist
#print(yearlist.count(years[0]))

datalog_selected = datalog[datalog['Year'] != years[0]] 
indicators_selected = list(datalog_selected.index)

df_selected = df.loc[indicators_selected]

indicators_count = []

for economy in economies:
    indicators_count.append(list(df_selected[economy] == '..').count(False))
    print(economy, '-', list(df_selected[economy] == '..').count(False))

print(indicators_count.count(datalog_selected.size))

count = dict(zip(economies, indicators_count))

#show selected economies
economies_selected = {key: count[key] for key in count if (count[key] == datalog_selected.size)}
#show dropped economies
economies_dropped = {key: count[key] for key in count if (count[key] < datalog_selected.size) and (count[key] >= datalog_selected.size-2)}

df_final= df_selected[economies_selected.keys()]
df_final = df_final.astype(float)

df_dropped = df_selected[economies_dropped.keys()]


#=============================Factor Analysis==================================

#Plot correlation matrix of indicators
plt.figure(figsize=(10,8))
corrMatrix = df_final.T.corr()
sns.heatmap(corrMatrix)

fa = FactorAnalyzer()
fa.fit(df_final.T, 25)

ev, v = fa.get_eigenvalues()

# Create scree plot using matplotlib
plt.figure(figsize=(6,4))
plt.scatter(range(1,df_final.T.shape[1]+1),ev)
plt.plot(range(1,df_final.T.shape[1]+1),ev)
plt.hlines(1, 0, df_final.T.shape[1], colors='r')
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

#Perform Factor Analysis
fa = FactorAnalyzer(list(ev >= 0.95).count(True), rotation='varimax')
fa.fit(df_final.T)
loads = fa.loadings_
print(loads)
loads = pd.DataFrame(loads, index=df_final.index)

#Heatmap of loadings
plt.figure(figsize=(15,15))
sns.heatmap(loads, annot=True, cmap="YlGnBu")

# Get variance of each factors
fa_var = fa.get_factor_variance()
fa_var = pd.DataFrame(fa_var, index=['SS loadings', 'Proportion Var', 'Cumulative Var'])
print(fa_var)


#=============================Cluster Analysis=================================

#standardization along columns
df_final_std=(df_final.T-df_final.T.mean())/df_final.T.std()

#Create dendrogram
dendrogram = sch.dendrogram(sch.linkage(df_final_std, method='ward'))
plt.axhline(y=11, color='r', linestyle='--')

# create clusters
hc = AgglomerativeClustering(n_clusters=12, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
y_hc = hc.fit_predict(df_final_std)

df_final_T = df_final.T
df_final_T['cluster'] = y_hc
df_final_T.sort_values("cluster", inplace = True, ascending=True)

df_final_std['cluster'] = y_hc
df_final_std.sort_values("cluster", inplace = True, ascending=True)

df_cluster = df_final_T.groupby('cluster').mean()
df_cluster_std = df_final_std.groupby('cluster').mean()

#Heatmap of cluster characteristics
plt.figure(figsize=(10,20))
sns.heatmap(df_cluster_std.T, cmap="Blues", linewidths=.5)

num_of_countries = []
for n in range(len(set(y_hc))):
    num_of_countries.append(sum(df_final_T['cluster'] == n))
    
df_cluster['num of countries'] = num_of_countries
df_cluster_std['num of countries'] = num_of_countries

columns = list(df_cluster.columns)
columns = columns[-1:] + columns[:-1]

df_cluster = df_cluster.reindex(columns=columns)
df_cluster_std = df_cluster_std.reindex(columns=columns)


#=============================Output as excel==================================
output = 'output_' + time.asctime(time.localtime(time.time())).replace(' ','_').replace(':','') + '.xlsx'

with pd.ExcelWriter(output) as writer:
    df_cluster.to_excel(writer, sheet_name='Cluster')
    df_cluster_std.to_excel(writer, sheet_name='Cluster_std')
    df_final_T.to_excel(writer, sheet_name='Result')
    df_final_std.to_excel(writer, sheet_name='Result_std')
    df_final.to_excel(writer, sheet_name='Final data')
    df_dropped.to_excel(writer, sheet_name='Dropped')
    df.to_excel(writer, sheet_name='Raw data')
    loads.to_excel(writer, sheet_name='Loading')
    corrMatrix.to_excel(writer, sheet_name='Correlation Matrix')
    datalog.to_excel(writer, sheet_name='Datalog')
    datalog_selected.to_excel(writer, sheet_name='Datalog_selected')
