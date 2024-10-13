#preset do notebook
!pip install -q sktime==0.18.0
!pip install prophet
!pip install openpyxl
!pip statsmodels.api

%matplotlib inline
%load_ext autoreload
%autoreload 2

#Bibliotecas
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import gzip
import matplotlib.pyplot as plt
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.datasets import load_airline
import prophet

plt.rcParams['figure.figsize'] = [15, 5]
plt.rcParams['font.size'] = 15
sns.set(color_codes=True)
sns.set(font_scale=1.5)
sns.set_palette("bright")
sns.set_style("whitegrid")

import automl
from automl import init
from automl.interface.utils import plot_forecast

df0 = pd.read_excel('volume_tratado_corte_mes.xlsx')
df0
df1 = df0[['data','volume']]
df1 = df0[['data', 'volume']].rename(columns={'data': 'ds', 'volume': 'y'})
df1
df1_2 = df1
df1_2['ds'] = pd.to_datetime(df1_2['ds'], format="%d/%m/%Y %H:%M:%S")
df1_2
#df1_2 = df1_2.loc[df1_2['y'] <= 400]
#df1_2 = df1_2.loc[df1_2['y'] >= 200]
#df1_2
df2 = df1_2
df2.set_index('ds', inplace=True)
df_dia = df2.resample('d').sum()
df_dia.plot()
df_dia
#df3 = df_dia.loc[df_dia['y'] > 82000]
#df3

mean_y = df_dia['y'].mean()
std_y = df_dia['y'].std()

lower_limit = mean_y - 3 * std_y
upper_limit = mean_y + 3 * std_y

# Filtra o DataFrame para manter apenas os valores dentro dos limites
df3_1 = df_dia[(df_dia['y'] >= lower_limit) & (df_dia['y'] <= upper_limit)]
df3_2 = df3_1.loc[df3_1['y'] > 83000]
df3_3 = df3_2.loc[df3_2['y'] < 87000]

df4 = df3_3
# drop NaNs for the time period where data wasn't recorded
df4.dropna(inplace=True)

df4.index = pd.to_datetime(df4.index, format='%Y-%m-%d %H:%M:%S')

y = pd.DataFrame(df4.to_numpy(),
                 index=df4.index,
                columns=['y'])
y = y.asfreq('D')
y.fillna(y['y'].mean(), inplace=True)
print(y.index)
print("Time Index is", "" if y.index.is_monotonic else "NOT", "monotonic.")
print("Train datatype", type(y))
y
y.plot()
y_train, y_test = temporal_train_test_split(y, test_size=33)
print("Training length: ", len(y_train)," Testing length: ", len(y_test))
init(engine='dask')
init(engine='local', check_deprecation_warnings=False)

est1 = automl.Pipeline(task='forecasting', 
                       n_algos_tuned=4)

est1.fit(X=None, y=y_train)

print('Selected model: {}'.format(est1.selected_model_))
print('Selected model params: {}'.format(est1.selected_model_params_))

summary_frame = est1.forecast(periods=len(y_test), alpha=0.05)
print(summary_frame)

future_index = y_train.index[-5:].union(y_test.index[:5])
print(future_index)

est1.predict(X=pd.DataFrame(index=future_index))

automl.interface.utils.plot_forecast(fitted_pipeline=est1, summary_frame=summary_frame, 
                                           additional_frames=dict(y_test=y_test)) 