#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[83]:


df = pd.read_csv('city_temperature.csv')


# In[84]:


df[df['Country']=='China']['City'].unique()


# In[85]:


dfshanghai = df[df['City']=='Shanghai']
dfshanghaihttp://localhost:8888/notebooks/Py_DS_ML_Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/My%20personal%20project%202-%20Longetivity%20of%20the%20Siberian%20High%20vs.%20Arctic%20High.ipynb#


# In[86]:


dfbeijing = df[df['City']=='Beijing']
dfbeijing


# In[87]:


dfcolumbus = df[(df['City']=='Columbus') & (df['State']=='Ohio')]
dfcolumbus


# In[88]:


dfsavannah = df[(df['State']=='Georgia') & (df['City']=='Savannah')]
dfsavannah


# In[89]:


dfcolumbus['Month and year'] = dfcolumbus['Month'].astype(str) + '-' + dfcolumbus['Year'].astype(str)
dfcolumbus


# In[90]:


#dfcolumbus.groupby('Month and year').mean().sort_values(by=['Month and year'], ascending=True).head(45)
dfcolumbusinjan = dfcolumbus[dfcolumbus['Month']==1]
dfcolumbusinjan


# In[91]:


dfcolumbusinjan.groupby('Month and year', as_index = False).mean()


# In[92]:


dfsavannah


# In[93]:


dfsavannah['Month and year'] = dfsavannah['Month'].astype(str) + '-' + dfsavannah['Year'].astype(str)
dfsavannah


# In[94]:


dfsavannahinjan = dfsavannah[dfsavannah['Month']== 1]
dfsavannahinjan


# In[95]:


dfsavannahinjan.groupby('Month and year', as_index = False).mean()


# In[97]:


plt.figure(figsize=(10,7),dpi=100)
plt.grid()
plt.xticks(rotation = 'vertical')
plt.plot(dfcolumbusinjan.groupby('Month and year', as_index = False).mean()['Month and year'], dfcolumbusinjan.groupby('Month and year', as_index = False).mean()['AvgTemperature'], label = 'Columbus', marker = '.')
plt.plot(dfsavannahinjan.groupby('Month and year', as_index = False).mean()['Month and year'], dfsavannahinjan.groupby('Month and year', as_index = False).mean()['AvgTemperature'], label = 'savannah', marker = '.')
plt.legend()


# In[98]:


dfbeijing = df[df['City']=='Beijing']
dfbeijing


# In[99]:


dfbeijing['Month and year'] = dfbeijing['Month'].astype(str) + '-' + dfbeijing['Year'].astype(str)
dfbeijing


# In[100]:


dfbeijing[dfbeijing['Month']== 1].groupby('Month and year', as_index=False).mean()


# In[101]:


plt.figure(figsize=(10,7),dpi=100)
plt.grid()
plt.xticks(rotation = 'vertical')
plt.plot(dfcolumbusinjan.groupby('Month and year', as_index = False).mean()['Month and year'], dfcolumbusinjan.groupby('Month and year', as_index = False).mean()['AvgTemperature'], label = 'Columbus', marker = '.')
plt.plot(dfbeijing[dfbeijing['Month']== 1].groupby('Month and year', as_index=False).mean()['Month and year'], dfbeijing[dfbeijing['Month']== 1].groupby('Month and year', as_index=False).mean()['AvgTemperature'], label = 'Beijing', marker = '.')
plt.legend()


# In[102]:


dfshanghai = df[df['City']=='Shanghai']
dfshanghai


# In[103]:


dfshanghai['Month and year'] = dfshanghai['Month'].astype(str) + '-' + dfshanghai['Year'].astype(str)
dfshanghai


# In[104]:


dfshanghai[dfshanghai['Month']==1].groupby('Month and year', as_index = False).mean()


# In[105]:


plt.figure(figsize=(10,7),dpi=(100))
plt.grid()
plt.xticks(rotation = 'vertical')
plt.plot(dfshanghai[dfshanghai['Month']==1].groupby('Month and year', as_index = False).mean()['Month and year'], dfshanghai[dfshanghai['Month']==1].groupby('Month and year', as_index = False).mean()['AvgTemperature'], label = 'Shanghai', marker='.')
plt.plot(dfsavannah[dfsavannah['Month']==1].groupby('Month and year', as_index = False).mean()['Month and year'], dfsavannah[dfsavannah['Month']==1].groupby('Month and year', as_index = False).mean()['AvgTemperature'], label = 'Savannah', marker='.')
plt.legend()


# In[106]:


#Can you conclude whether the arctic air patterns vs the siberian air patterns are completely different or not?


# In[107]:


chinadiff = dfbeijing[dfbeijing['Month']==1].groupby('Month and year').mean()['AvgTemperature'] - dfshanghai[dfshanghai['Month']==1].groupby('Month and year').mean()['AvgTemperature']
chinadiff


# In[108]:


usdiff = dfcolumbus[dfcolumbus['Month']==1].groupby('Month and year').mean()['AvgTemperature'] - dfsavannah[dfsavannah['Month']==1].groupby('Month and year').mean()['AvgTemperature']
-usdiff
usdiff.index


# In[109]:


dfcolumbusjanavg = dfcolumbus[dfcolumbus['Month']==1].groupby('Month and year', as_index = False).mean()
#dfcolumbus['diffwithsavannah'] = dfsavannah[dfsavannah['Month']==1].groupby('Month and year').mean()['AvgTemperature'] - dfcolumbus[dfcolumbus['Month']==1].groupby('Month and year').mean()['AvgTemperature']
#dfcolumbus
dfcolumbusjanavg
dfcolumbusjanavg['diffwithsavannah'] = -usdiff
dfcolumbusjanavg


# In[110]:


differencedata = {'Month and year': chinadiff.index,
                  'usdiff': -usdiff,
               'chinadiff': -chinadiff}
differencedf = pd.DataFrame(differencedata)
differencedf


# In[111]:


plt.figure(figsize=(10,7), dpi=100)
plt.grid()
plt.xticks(rotation = 'vertical')
plt.plot(differencedf['Month and year'],differencedf['usdiff'], label = 'US', marker = '.')
plt.plot(differencedf['Month and year'],differencedf['chinadiff'], label = 'China', marker = '.')
plt.legend()
plt.title('Difference in annual Temps of 2 cities same latitude')
plt.xlabel('year')
plt.ylabel('temp in F')


# In[112]:


dfcolumbusjanavg.iloc[dfcolumbusjanavg[['AvgTemperature']].idxmin()['AvgTemperature']]


# In[113]:


differencedf['std of columbus'] = dfcolumbus[dfcolumbus['Month']==1].groupby('Month and year').std()['AvgTemperature']
differencedf['std of beijing'] = dfbeijing[dfbeijing['Month']==1].groupby('Month and year').std()['AvgTemperature']
differencedf


# In[114]:


plt.figure(figsize=(10,7), dpi=100)
plt.grid()
plt.xticks(rotation = 'vertical')
plt.plot(differencedf['Month and year'],differencedf['std of columbus'], label = 'US', marker = '.')
plt.plot(differencedf['Month and year'],differencedf['std of beijing'], label = 'China', marker = '.')
plt.legend()
plt.title('Strength of the arctic high vs siberian high')
plt.xlabel('year')
plt.ylabel('temp in F')


# In[115]:


dfcolumbus[dfcolumbus['Year']==2009].head(31)


# In[116]:


plt.figure(figsize=(6,6), dpi=100)
plt.grid()
plt.scatter(dfcolumbus[dfcolumbus['Month']==1].groupby('Month and year').mean()['AvgTemperature'], dfbeijing[dfbeijing['Month']==1].groupby('Month and year').mean()['AvgTemperature'])
plt.xlabel('Temp of columbus')
plt.ylabel('Temp of Beijing')
plt.show()


# In[117]:


from scipy import stats
stats.pearsonr(dfcolumbus[dfcolumbus['Month']==1].groupby('Month and year').mean()['AvgTemperature'], dfbeijing[dfbeijing['Month']==1].groupby('Month and year').mean()['AvgTemperature'])


# In[118]:


#An R-Value of 0.24 means that there is hardly any correlation between the activity of the Arctic High and the Siberian High


# In[119]:


#What about the correlation between Columbus and Savannah vs. Beijing and Shanghai?


# In[120]:


plt.figure(figsize=(6,6), dpi=100)
plt.grid()
plt.scatter(dfsavannah[dfsavannah['Month']==1].groupby('Month and year').mean()['AvgTemperature'], dfcolumbus[dfcolumbus['Month']==1].groupby('Month and year').mean()['AvgTemperature'])
plt.xlabel('Temp of savannah')
plt.ylabel('Temp of columbus')
plt.show()


# In[121]:


stats.pearsonr(dfsavannah[dfsavannah['Month']==1].groupby('Month and year').mean()['AvgTemperature'], dfcolumbus[dfcolumbus['Month']==1].groupby('Month and year').mean()['AvgTemperature'])


# In[122]:


plt.figure(figsize=(6,6), dpi=100)
plt.grid()
plt.scatter(dfshanghai[dfshanghai['Month']==1].groupby('Month and year').mean()['AvgTemperature'], dfbeijing[dfbeijing['Month']==1].groupby('Month and year').mean()['AvgTemperature'])
plt.xlabel('Temp of shanghai')
plt.ylabel('Temp of beijing')
plt.show()


# In[123]:


stats.pearsonr(dfshanghai[dfshanghai['Month']==1].groupby('Month and year').mean()['AvgTemperature'], dfbeijing[dfbeijing['Month']==1].groupby('Month and year').mean()['AvgTemperature'])


# In[124]:


#Lets study the correlation between the influence of the arctic high on the West coast vs. on the East coast!


# In[126]:


dfsacramento = df[df['City']=='Sacramento']
dfsacramento


# In[127]:


dfsacramento['Month and year'] = dfsacramento['Month'].astype(str) + '-' + dfsacramento['Year'].astype(str)
dfsacramento


# In[128]:


plt.figure(figsize=(6,6), dpi=100)
plt.grid()
plt.scatter(dfcolumbus[dfcolumbus['Month']==1].groupby('Month and year').mean()['AvgTemperature'], dfsacramento[dfsacramento['Month']==1].groupby('Month and year').mean()['AvgTemperature'])
plt.xlabel('Temp of columbus')
plt.ylabel('Temp of Sacramento')
plt.show()


# In[129]:


stats.pearsonr(dfcolumbus[dfcolumbus['Month']==1].groupby('Month and year').mean()['AvgTemperature'], dfsacramento[dfsacramento['Month']==1].groupby('Month and year').mean()['AvgTemperature'])[0]


# In[130]:


dftest = df[df['City']=='Austin'] 
dftest['Month and year'] = dftest['Month'].astype(str) + '-' + dftest['Year'].astype(str)
stats.pearsonr(dfcolumbus[dfcolumbus['Month']==12].groupby('Month and year').mean()['AvgTemperature'], dftest[dftest['Month']==12].groupby('Month and year').mean()['AvgTemperature'])[0]**2


# In[131]:


#Lets make a heat map out of this!!! What other countries do you think you can make a heat map for?


# In[132]:


citydata = []
citystate = []
cityname = []
for i in df[df['Country']=='US']['City'].unique():
    if df[df['City']==i].count()[0] == df[(df['City']=='Columbus') & (df['State']=='Ohio')].count()[0]:
        dftest = df[df['City']==i]
        dftest.loc[:, ('Month and year')] = dftest.loc[:, ('Month')].astype(str) + '-' + dftest.loc[:, ('Year')].astype(str)
        cityname.append(i)
        citystate.append([i, df[(df['City']==i) & (df['Country']=='US')]['State'].iloc[0]])
        citydata.append((stats.pearsonr(dfcolumbus[dfcolumbus['Month']==1].groupby('Month and year').mean()['AvgTemperature'], dftest[dftest['Month']==1].groupby('Month and year').mean()['AvgTemperature']))[0])
        


# In[133]:


dflist = pd.DataFrame(list(zip(cityname, citystate, citydata)), columns=['City', 'State', 'R-value'])
dflist.sort_values(by=['R-value'],ascending=True).head(20)


# In[134]:


import geopandas as gpd


# In[135]:


map_df = gpd.read_file('USA_adm/USA_adm1.shp')


# In[136]:


import cartopy.crs as crs
import cartopy.feature as cfeature


# df.head()

# In[137]:


from geopy.geocoders import Nominatim
import math


# In[138]:


geolocator = Nominatim(user_agent = 'jason_duong96@hotmail.com')
geolocator.geocode('Birmingham, AL')[1]


# In[139]:


latitude = []
longitude = []
for i in dflist['State']:
    latitude.append(geolocator.geocode(i)[1][0])
    longitude.append(geolocator.geocode(i)[1][1])


# In[140]:


dflist = dflist.assign(Latitude = latitude, Longitude = longitude)


# In[141]:


for index, i in enumerate(dflist['Longitude']):
    if i > -50:
        dflist = dflist.drop(index, axis = 0)
        


# In[142]:


dflist


# In[143]:


figure = plt.figure(figsize=(20,12))
ax = figure.add_subplot(1,1,1, projection=crs.Mercator())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES)
# Zoom in on the US by setting longitude/latitude parameters
ax.set_extent(
    [-130, # minimum latitude
        -60, # min longitude
        24, # max latitude
        52 # max longitude
    ],
    crs=crs.PlateCarree()
)
plt.scatter(x=dflist['Longitude'], y=dflist['Latitude'], c = dflist['R-value'], s=250, alpha=1, cmap = 'hot', transform = crs.PlateCarree())
plt.colorbar(plt.scatter(x=dflist['Longitude'], y=dflist['Latitude'], c = dflist['R-value'], s=250, alpha=1, cmap = 'hot', transform = crs.PlateCarree()))
plt.show()


# In[144]:


#Nice! You've created a map that shows the influence of the Arctic High on most major U.S cities relative to Columbus! Now can you create a film that shows this map when the month looked at is different?


# In[145]:


from matplotlib.animation import FuncAnimation


# In[146]:


f_d = ax.plot([], [], linewidth=2.5)
temp = ax.text(1, 1, '', ha='right', va='top', fontsize=24)


# In[147]:


def animate(m):
    k = np.linspace(1,12,12)
    plt.title('Month of {k}')
    


# In[148]:



dftest


# 

# In[149]:


example = []
example.append([2,4])


# In[150]:


example


# In[151]:


dflist


# In[152]:


dflist= dflist[['City', 'State', 'Latitude', 'Longitude', 'R-value']]
dflist


# In[153]:


dfexample = dflist


# In[154]:


dfcolumbus


# In[155]:


for i in range(1,13,1):
    citydata = []
    for k in df[df['Country']=='US']['City'].unique():
        if df[df['City']==k].count()[0] == df[(df['City']=='Columbus') & (df['State']=='Ohio')].count()[0]:
            dftest = df[df['City']==k]
            dftest.loc[:, ('Month and year')] = dftest.loc[:, ('Month')].astype(str) + '-' + dftest.loc[:, ('Year')].astype(str)
            citydata.append((stats.pearsonr(dfcolumbus[dfcolumbus['Month']==i].groupby('Month and year').mean()['AvgTemperature'], dftest[dftest['Month']==i].groupby('Month and year').mean()['AvgTemperature']))[0])
    dflist.insert(3+i, 'x', citydata, True)
    


# In[156]:


dflist


# In[210]:


from IPython import display
from matplotlib import animation
months = ['January','Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
for i in range(1,13,1):
    figure = plt.figure(figsize=(20,12))
    ax = figure.add_subplot(1,1,1, projection=crs.Mercator())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)
    # Zoom in on the US by setting longitude/latitude parameters
    ax.set_extent(
    [-130, # minimum latitude
        -60, # min longitude
        24, # max latitude
        52 # max longitude
    ],
    crs=crs.PlateCarree()
    )
    display.clear_output(wait=True)
    plt.scatter(x=dflist['Longitude'], y=dflist['Latitude'], c = dflist.iloc[:, 3+i], s=250, alpha=1, cmap = 'hot', transform = crs.PlateCarree())
    plt.title('Spread of arctic air to U.S cities relative to Columbus')
    plt.colorbar(plt.scatter(x=dflist['Longitude'], y=dflist['Latitude'], c = dflist.iloc[:, 3+i], s=250, alpha=1, cmap = 'hot', transform = crs.PlateCarree()), label='R-Value', shrink=0.75)
    plt.clim(vmin=-0.5, vmax=1)
    plt.text(0.95,0.95,months[i-1],horizontalalignment='center', verticalalignment='center', transform = ax.transAxes, size= 'xx-large')
    plt.pause(0.5)
    plt.show()
    


# In[ ]:




