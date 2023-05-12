# -*- coding: utf-8 -*-
"""
MSDM5058 Information Science 
Computational Project II:
    Portfolio Management Using Prediction Rules and Communication in Social Networks 
@author: Helin Yang
"""
import warnings
warnings.filterwarnings("ignore")

#%% Q1.1 Plot 𝑆(𝑡) and its daily return rate 

import sys
from datetime import datetime
import numpy as np
import pandas as pd
import time
from pylab import plt
# import matplotlib as mpl
from matplotlib.dates import YearLocator, AutoDateFormatter
# mpl.rcParams['font.family'] = 'serif'
def make_url(ticker_symbol, unix_start, unix_end, daily):
    if daily:
        link = 'https://query1.finance.yahoo.com/v7/finance/download/' + ticker_symbol + '?period1=' + str(unix_start) + '&period2=' + str(unix_end) + '&interval=1d&events=history'
    else:
        link = 'https://query1.finance.yahoo.com/v7/finance/download/' + ticker_symbol + '?period1=' + str(unix_start) + '&period2=' + str(unix_end) + '&interval=1mo&events=history'
    return link
def pull_historical_data(ticker_symbol, output_path, unix_start, unix_end, daily, is_save):
    import requests, re, json
    r = requests.get(make_url(ticker_symbol, unix_start, unix_end, daily), headers={'User-Agent': 'Custom'})
    totaldata=r.text.split('\n')
    totaldate=[]
    totaladj_close=[]
    for i in range(1,len(totaldata)):
        date=totaldata[i].split(',')[0]
        adj_close=(totaldata[i].split(','))[-2]
        totaldate.append(date)
        totaladj_close.append(adj_close)
    data={'date':totaldate,'adj_close':totaladj_close}
    df = pd.DataFrame(data)
    df=df.sort_values(by="date" , ascending=False) 
    if is_save:     
        df.to_csv(ticker_symbol+'.csv') 
    return df
def pull_stock_df(stocklist, start_date, end_date, daily):    
    output_path=''
    start=datetime(int(start_date.split('-')[0]), int(start_date.split('-')[1]), int(start_date.split('-')[2]))
    end=datetime(int(end_date.split('-')[0]), int(end_date.split('-')[1]), int(end_date.split('-')[2]))
    unix_start=int(time.mktime(start.timetuple()))
    day_end=end.replace(hour=23, minute=59, second=59)
    unix_end=int(time.mktime(day_end.timetuple()))
    stock_df = pd.DataFrame()
    ### START CODE HERE ###
    for stock in stocklist:
        date=pull_historical_data(stock, output_path, unix_start, unix_end, daily, True).iloc[:,0]
        data=pull_historical_data(stock, output_path, unix_start, unix_end, daily, True).iloc[:,-1]
        date=pd.DataFrame({'date':date})
        data=pd.DataFrame({stock:data})
        if stock_df.empty:
            stock_df=date
            stock_df=pd.concat([stock_df, data], axis = 1)
        else:
            stock_df = pd.concat([stock_df, data], axis = 1)
    ### END CODE HERE ###
    return stock_df
stocklist = ['GOOGL']
# stocklist = ['RMD']
# stocklist = ['T']
start_date='2005-01-01'
end_date = '2023-01-01'
daily = True
data = pull_stock_df(stocklist, start_date, end_date, daily)
data.set_index(["date"], inplace=True)
data=data.astype('float')
data=data[::-1]

# St plot
plt.figure(figsize=(20,8),dpi=500)
plt.plot(data)
plt.legend(data.columns)
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('S_t for price')
plt.savefig('1-1.jpg')

# return rate
returndata = np.log(data/data.shift(1)).dropna()
# return rate plot
plt.figure(figsize=(20,8),dpi=500)
plt.plot(returndata)
plt.legend(returndata.columns)
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('Daily return rate')
plt.savefig('1-2.jpg')

# divide the future and past data
div = int(3 * (returndata.shape[0]) / 4)
past = returndata.iloc[:div]
future = returndata.iloc[div:]
st_future = data.iloc[div+1:]

#%% Q1.2 autocorrelation function and the partial autocorrelatio

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# acf
fig, ax = plt.subplots(figsize=(12,6),dpi=300)
sm.graphics.tsa.plot_acf(returndata, lags=30, ax=ax)
fig.savefig('1-3.jpg')


# pacf
fig, ax = plt.subplots(figsize=(12,6),dpi=300)
sm.graphics.tsa.plot_pacf(returndata, lags=30, ax=ax)
fig.savefig('1-4.jpg')


#%% Q1.3 augmented Dickey-Fuller test

import pandas as pd
from statsmodels.tsa.stattools import adfuller
# ADF
result = adfuller(returndata)
print('ADF：', result[0])
print('p-value：', result[1])
print('lag：', result[2])
print('critical value：')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
    
if result[0] < result[4]['5%']:
    print('Time series data is stationary')
else:
    print('Time series data is not stationary')


#%% Q1.4 Digitize the returndata with setting the constant value as 0.002

con=0.002
x_1=returndata
x_1['digit']=np.zeros((x_1.shape[0], 1))
for i in range(x_1.shape[0]):
    if x_1.iloc[i,0]>con:
        x_1.iloc[i,1]='U'
    elif x_1.iloc[i,0]<-con:
        x_1.iloc[i,1]='D'
    else:
        x_1.iloc[i,1]='H'
digitdata=x_1.iloc[:,1]
print('constant value:',con)
print(digitdata[:5])
print(digitdata.value_counts())

#%% Q2.1 Plot the conditional CDF: F_U(𝑥) ≡ CDF[ 𝑋(𝑡) = 𝑥 ∣ 𝑌 (𝑡 + 1) = U ]

import random
random.seed(20938289)
list1=returndata.iloc[0:digitdata.shape[0]-1,0]
list2=digitdata.iloc[1:digitdata.shape[0]]
# print(list1)
# print(list2)

list1=list(list1)
list2=list(list2)
# latter=u
cond_prob1 = []
for i in range(len(list2)):
    if list2[i] == 'U':
        cond_prob1.append(list1[i])
cond_prob1 = np.array(cond_prob1)
cond_prob1=np.sort(cond_prob1)
y1 = np.arange(1, len(cond_prob1)+1) / len(cond_prob1)

plt.figure(figsize=(6,4),dpi=400)
plt.plot(cond_prob1, y1)
plt.xlabel('X_t')
plt.ylabel('CDF')
plt.title('condition CDF of P(X_t=x|Y_t+1=U)')
plt.savefig('2-1.jpg')


#%% Q2.2 Plot the conditional CDF: 𝐹_D(𝑥) ≡ CDF[ 𝑋(𝑡) = 𝑥 ∣ 𝑌 (𝑡 + 1) = D ]

cond_prob2 = []
for i in range(len(list2)):
    if list2[i] == 'D':
        cond_prob2.append(list1[i])
cond_prob2 = np.array(cond_prob2)
cond_prob2=np.sort(cond_prob2)
y2 = np.arange(1, len(cond_prob2)+1) / len(cond_prob2)

plt.figure(figsize=(6,4),dpi=400)
plt.plot(cond_prob2, y2)
plt.xlabel('X_t')
plt.ylabel('CDF')
plt.title('condition CDF of P(X_t=x|Y_t+1=D)')
plt.savefig('2-2.jpg')

#%% Q3.1 fit 𝐹_U(𝑥) and 𝐹_D(𝑥) and plot f_U(x) and f_D(x)

from scipy.optimize import curve_fit
x1=cond_prob1
x2=cond_prob2
def fermi_dirac(x, a, b):
    return 1 / (1 + np.exp((-1)*b*(x - a)))
popt1, pcov1 = curve_fit(fermi_dirac, x1, y1)
print('a =', popt1[0])
print('b =', popt1[1])
popt2, pcov2 = curve_fit(fermi_dirac, x2, y2)
print('a =', popt2[0])
print('b =', popt2[1])


# fitted F_U(x) and F_D(x)
ydata1=fermi_dirac(x1, *popt1)
dy_dx1= np.diff(ydata1) / np.diff(x1)

ydata2=fermi_dirac(x2, *popt2)
dy_dx2 = np.diff(ydata2) / np.diff(x2)

plt.figure(figsize=(6,4),dpi=300)
plt.plot(x1[:-1], dy_dx1, label='f_U(x)')
plt.plot(x2[:-1], dy_dx2,label='f_D(x)')  
plt.legend()
plt.title('PDF')
plt.savefig('3-1.jpg')


#%% Q3.2 Gaussian distribution

from scipy.stats import norm

sequence1 = cond_prob1
sequence2 = cond_prob2
mu1, std1 = np.mean(sequence1), np.std(sequence1)
mu2, std2 = np.mean(sequence2), np.std(sequence2)

# build Guassion model
pdf_gaussian1 = norm(loc=mu1, scale=std1)
pdf_gaussian2 = norm(loc=mu2, scale=std2)

# get some x squence and calculate the pdf
x_1 = np.linspace(min(sequence1), max(sequence1), 1000)
y_1 = pdf_gaussian1.pdf(x_1)
x_2 = np.linspace(min(sequence2), max(sequence2), 1000)
y_2 = pdf_gaussian2.pdf(x_2)

plt.figure(figsize=(6,4),dpi=300)
plt.plot(x_1,y_1,label='f_u(x)')
plt.plot(x_2,y_2,label='f_d(x)')  
plt.legend()
plt.title('Gaussian Distribution Fitted to Sequence PDF')
plt.savefig('3-2.jpg')

#%% Q4.1 prior probabilities

D=0
H=0
U=0
for i in range(len(list2)):
    if list2[i]=='D':
        D+=1
    elif list2[i]=='H':
        H+=1
    else:
        U+=1
p_d=D/len(list2)
p_u=U/len(list2)
print(D,H,U)
print('P_t+1=U:',p_u)
print('P_t+1=D:',p_d)

total = np.append(x_1,x_2)
min_,max_ = min(total),max(total)
lis_x = np.linspace(min_,max_,200)

#%% Q4.2 Plot 𝑥 = 𝑥1 and 𝑥 = 𝑥2 on the graph of 𝐿'u(𝑥) and 𝐿'd(𝑥). 

pdf_u = np.diff(fermi_dirac(lis_x, *popt1)) / np.diff(lis_x)
pdf_d = np.diff(fermi_dirac(lis_x, *popt2)) / np.diff(lis_x)
p_u_x = p_u*pdf_u/(p_u*pdf_u+p_d*pdf_d)
p_d_x = p_d*pdf_d/(p_u*pdf_u+p_d*pdf_d)
diff = p_u_x-p_d_x

def find_sign_changes(arr):
    signs = np.sign(arr)  # 计算数组中每个元素的符号
    sign_changes = np.diff(signs)  # 计算符号数组的差分
    sign_change_indices = np.where(sign_changes != 0)[0]  # 找到非零差分的索引
    return sign_change_indices

x1=lis_x[find_sign_changes(diff)[0]]
x2=lis_x[find_sign_changes(diff)[1]]


print('x1 is equal to',x1)
print('x2 is equal to',x2)
logistic_detector = (x1,x2)
print('the logistic detector is :',logistic_detector)

plt.figure(figsize = (8,5),dpi=300)
plt.plot(lis_x[:-1], p_u_x, label='f(U|x)')
plt.plot(lis_x[:-1], p_d_x, label='f(D|x)')
plt.axvline(x1,label = 'x1',ls='dashed', color='red')
plt.axvline(x2,label = 'x2',ls='dashed', color='grey')
plt.xlabel('X(t)')
plt.ylabel('Probability Density')
plt.title('Bayes Detector for logistic PDFs')
plt.legend(loc = 'upper right')
plt.savefig('4-1.jpg')

total = np.append(x_1,x_2)
min_,max_ = min(total),max(total)
lis_x = np.linspace(min_,max_,200)


#%% Q4.3 Plot 𝑥 = 𝑥1 and 𝑥 = 𝑥2 on the graph of Gaussion pdfs

pdf_u1 = pdf_gaussian1.pdf(lis_x) 
pdf_d1 = pdf_gaussian2.pdf(lis_x) 
p_u_x1= p_u*pdf_u1/(p_u*pdf_u1+p_d*pdf_d1)
p_d_x1 = p_d*pdf_d1/(p_u*pdf_u1+p_d*pdf_d1)
diff = p_u_x1-p_d_x1

def find_sign_changes(arr):
    signs = np.sign(arr)  # 计算数组中每个元素的符号
    sign_changes = np.diff(signs)  # 计算符号数组的差分
    sign_change_indices = np.where(sign_changes != 0)[0]  # 找到非零差分的索引
    return sign_change_indices

x1=lis_x[find_sign_changes(diff)[0]]
x2=lis_x[find_sign_changes(diff)[1]]

print('x1 is equal to',x1)
print('x2 is equal to',x2)
gaussion_detector = (x1,x2)
print('the gaussion detector is :',gaussion_detector)

plt.figure(figsize = (8,5),dpi=300)
plt.plot(lis_x, p_u_x1, label='f(U|x)')
plt.plot(lis_x, p_d_x1, label='f(D|x)')
plt.axvline(x1,label = 'x1',ls='dashed', color='red')
plt.axvline(x2,label = 'x2',ls='dashed', color='grey')
plt.xlabel('X(t)')
plt.ylabel('Probability Density')
plt.title('Bayes Detector for gaussian PDFs')
plt.legend(loc = 'upper right')
plt.savefig('4-2.jpg')

#%% Q5.1 Association Rules: 1-day upward & downward rules

list3=digitdata.iloc[0:digitdata.shape[0]-1]
list3=list(list3)
data={'former':list3,'latter':list2}
data=pd.DataFrame(data)
print(data[:5])
counts = data.groupby(['former', 'latter']).size().reset_index(name='counts')
print(counts)


print('===== Best 1-day upward rule =====')
counts_U=counts.loc[counts['latter']=='U']
sumvalue=counts_U['counts'].sum()
counts_U['probility_U']=counts_U['counts']/sumvalue
counts_U=counts_U.sort_values(by=['probility_U'], ascending=False) 
print(counts_U)
strategy_U=counts_U.iloc[0]
print('best 1-day strategy for U is:',strategy_U.iloc[0])
print('The stratgy summary is:')
print(strategy_U)
strategy_k1_U = strategy_U.iloc[0]


print('===== Best 1-day downward rule =====')
counts_D=counts.loc[counts['latter']=='D']
sumvalue=counts_D['counts'].sum()
counts_D['probility_D']=counts_D['counts']/sumvalue
counts_D=counts_D.sort_values(by=['probility_D'], ascending=False) 
print(counts_D)
strategy_D=counts_D.iloc[0]
print('best 1-day strategy for D is:',strategy_D.iloc[0])
print('The stratgy summary is:')
print(strategy_D)
strategy_k1_D = strategy_D.iloc[0]

#%% Q5.2 Association Rules: 5-day upward & downward rules

totallist=[]
latter2=[]
for i in range(5,digitdata.shape[0]):
    m=tuple(list(digitdata.iloc[i-5:i]))
    latterone=digitdata.iloc[i]
    totallist.append(m)
    latter2.append(latterone)
data2={'former':totallist,'latter':latter2}
data2=pd.DataFrame(data2)
print(data2[:5],'\n')
counts1 = data2.groupby(['former', 'latter']).size().reset_index(name='counts')
print(counts,'\n')


print('===== Best 5-day upward rule =====')
counts_U=counts1.loc[counts1['latter']=='U']
sumvalue=counts_U['counts'].sum()
counts_U['probility_U']=counts_U['counts']/sumvalue
counts_U=counts_U.sort_values(by=['probility_U'], ascending=False) 
print(counts_U)
strategy_U=counts_U.iloc[0]
print('best 5-day strategy for U is:',strategy_U.iloc[0])
print('The stratgy summary is:')
print(strategy_U,'\n')
strategy_k5_U = ''.join(strategy_U.iloc[0])


print('===== Best 5-day downward rule =====')
counts_D=counts1.loc[counts1['latter']=='D']
sumvalue=counts_D['counts'].sum()
counts_D['probility_D']=counts_D['counts']/sumvalue
counts_D=counts_D.sort_values(by=['probility_D'], ascending=False) 
print(counts_D)
strategy_D=counts_D.iloc[0]
print('best 5-day strategy for D is:',strategy_D.iloc[0])
print('The stratgy summary is:')
print(strategy_D)
strategy_k5_D = ''.join(strategy_D.iloc[0])

#%% Q6.1 Trade with logistic detector and k=1

def v_k1(future,dec,gamma):
    m = 100000
    n = 0
    m_list = np.zeros((len(future),1))
    n_list = np.zeros((len(future),1))
    k1=0
    k2=0
    k3=0
    k4=0
    
    for i in range(len(future)):
        if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
            if digitdata[div-1+i] == strategy_k1_U:
                delta_m = gamma*m
                m = m - delta_m
                n = n + delta_m/st_future.iloc[i][0]
                m_list[i] = m
                n_list[i] = n
                k1 = k1+1
            else:
                m_list[i] = m
                n_list[i] = n
                k3 = k3+1
        else:
            if digitdata[div-1+i] == strategy_k1_D:
                delta_n = gamma*n
                m = m + delta_n * st_future.iloc[i][0]
                n = n - delta_n
                m_list[i] = m
                n_list[i] = n
                k2 = k2+1
            else:
                m_list[i] = m
                n_list[i] = n
                k4 = k4+1
                
    return m_list,n_list,(k1,k2,k3,k4)

m_list,n_list,num_tran = v_k1(future,logistic_detector,gamma = 0.1)
vl1 = m_list + n_list * st_future

plt.figure(figsize=(12,5),dpi=300)
plt.plot(vl1,label='portfolio’s value')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('transactions based on logistic detector and best 1-day rule')
plt.savefig('6-1.jpg')
print('Buy:',num_tran[0])
print('Sell:',num_tran[1])
print(num_tran)


#%% Q6.2 Trade with gaussion detector and k=1

m_list,n_list,num_tran = v_k1(future,gaussion_detector,gamma = 0.1)
vg1 = m_list + n_list * st_future

plt.figure(figsize=(12,5),dpi=300)
plt.plot(vg1,label='portfolio’s value')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('transactions based on Gaussion detector and best 1-day rule')
plt.savefig('6-2.jpg')
print('Buy:',num_tran[0])
print('Sell:',num_tran[1])
print(num_tran)

#%% Q6.3 Trade with logistic detector and k=5

def v_k5(future,dec,gamma):
    m_list = np.zeros((len(future),1))
    n_list = np.zeros((len(future),1))
    k1=0
    k2=0
    k3=0
    k4=0
    m = 100000
    n = 0
    for i in range(len(future)):
        # kday = ''.join(digitdata[div-1-4+i:div-1+i].values)
        kday = ''.join(digitdata[div-1-5+i:div-1+i].values)
        if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
            if kday == strategy_k5_U:
                delta_m = gamma*m
                m = m - delta_m
                n = n + delta_m/st_future.iloc[i][0]
                m_list[i] = m
                n_list[i] = n
                k1 = k1+1
            else:
                m_list[i] = m
                n_list[i] = n
                k3 = k3+1
        else:
            if kday == strategy_k5_D:
                delta_n = gamma*n
                m = m + delta_n * st_future.iloc[i][0]
                n = n - delta_n
                m_list[i] = m
                n_list[i] = n
                k2 = k2+1
            else:
                m_list[i] = m
                n_list[i] = n
                k4 = k4+1
    return m_list,n_list,(k1,k2,k3,k4)


m_list,n_list,num_tran = v_k5(future,logistic_detector,gamma = 0.1)
vl5 = m_list + n_list * st_future

plt.figure(figsize=(12,5),dpi=300)
plt.plot(vl5,label='portfolio’s value')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('transactions based on logistic detector and best 5-day rule')
plt.savefig('6-3.jpg')
print('Buy:',num_tran[0])
print('Sell:',num_tran[1])
print(num_tran)

plt.figure(figsize=(12,5),dpi=300)
plt.plot(st_future)
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('price of the stock')
plt.savefig('6.jpg')

#%% Q6.4 Trade with gaussion detector and k=5

m_list,n_list,num_tran = v_k5(future,gaussion_detector,gamma = 0.1)     
vg5 = m_list + n_list * st_future

plt.figure(figsize=(12,5),dpi=300)
plt.plot(vg5,label='portfolio’s value')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('transactions based on gaussion detector and best 5-day rule')
plt.savefig('6-4.jpg')
print('Buy:',num_tran[0])
print('Sell:',num_tran[1])
print(num_tran)

#%% Q6.5 Campariasion
plt.figure(figsize=(12,5),dpi=300)
plt.plot(vl1,label='log k=1',color='orange')
plt.plot(vg1,label='gaus k=1',color='steelblue')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.savefig('6-5.jpg')
# k1买入次数，k2卖出次数。k=1交易次数频繁，能有效将行情上涨转化为自己的价值

plt.figure(figsize=(12,5),dpi=300)
plt.plot(vl5,label='log k=5',color='orange')
plt.plot(vg5,label='gaus k=5',color='steelblue')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.savefig('6-6.jpg')
# k=5买入卖出策略保守，交易次数较少，由于卖出动作分别为0和1，总价值伴随股价波动
# 观察st_future的股票价格，21-09-28股价下跌
# 总价值突变的原因是股价跳水前夕（09-27），gaussion（k=5)模型在股价最高点卖出了股票

print(st_future.iloc[812:817])
print(n_list[813:818])

#%% Q7.1 Trade with gaussian detector, k=1, tax = 0.2%

tax = 0.002
greed = 0.01  

def v_k1_tax(future,dec,gamma,xigma):
    m = 100000
    n = 0
    m_list = np.zeros((len(future),1))
    n_list = np.zeros((len(future),1))
    k1=0
    k2=0
    k3=0
    k4=0
    for i in range(len(future)):
        if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
            if digitdata[div-1+i] == strategy_k1_U:
                delta_m = gamma*m
                m = m - delta_m
                n = n + (1-xigma)*delta_m/st_future.iloc[i][0]
                m_list[i] = m
                n_list[i] = n
                k1 = k1+1
            else:
                m_list[i] = m
                n_list[i] = n
                k3 = k3+1
        else:
            if digitdata[div-1+i] == strategy_k1_D:
                delta_n = gamma*n
                m = m +(1-xigma)* delta_n * st_future.iloc[i][0]
                n = n - delta_n
                m_list[i] = m
                n_list[i] = n
                k2 = k2+1
            else:
                m_list[i] = m
                n_list[i] = n
                k4 = k4+1
    return m_list,n_list,(k1,k2,k3,k4)
m_list,n_list,num_tran = v_k1_tax(future,gaussion_detector,greed,tax)
vg1_tax1 = m_list + n_list * st_future

plt.figure(figsize=(12,5),dpi=300)
plt.plot(vg1_tax1,label='portfolio’s value after tax',color='orange')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title(r'Portfolio value after tax with gaussian detector and best 1-day rule ($\zeta$={} % and $\gamma$={} %)'.format(tax*100,greed*100))
plt.savefig('7-1.jpg')
print('Buy:',num_tran[0])
print('Sell:',num_tran[1])
print('zeta = {} %'.format(tax*100))
print('the final portfolio value is ',vg1_tax1[len(data)-div:].values[0][0])


#%% Q7.2 Trade with gaussion detector, k=5, xigma = 0.2%
def v_k5_tax(future,dec,gamma,xigma):
    m = 100000
    n = 0
    num_future = len(future)
    m_list = np.zeros((num_future,1))
    n_list = np.zeros((num_future,1))
    k1=0
    k2=0
    k3=0
    k4=0
    
    for i in range(num_future):
        kday = ''.join(digitdata[div-1-5+i:div-1+i].values)
        if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
            if kday == strategy_k5_U:
                delta_m = gamma*m
                m = m - delta_m
                n = n + (1-xigma)*delta_m/st_future.iloc[i][0]
                m_list[i] = m
                n_list[i] = n
                k1 = k1+1
            else:
                m_list[i] = m
                n_list[i] = n
                k3 = k3+1
        else:
            if kday == strategy_k5_D:
                delta_n = gamma*n
                m = m +(1-xigma)* delta_n * st_future.iloc[i][0]
                n = n - delta_n
                m_list[i] = m
                n_list[i] = n
                k2 = k2+1
            else:
                m_list[i] = m
                n_list[i] = n
                k4 = k4+1
    return m_list,n_list,(k1,k2,k3,k4)
m_list,n_list,num_tran = v_k5_tax(future,gaussion_detector,greed,tax)     
vg5_tax1 = m_list + n_list * st_future

plt.figure(figsize=(12,5),dpi=300)
plt.plot(vg5_tax1,label='portfolio’s value')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title(r'Portfolio value after tax with gaussian detector and best 5-day rule ($\zeta$={} % and $\gamma$={} %)'.format(tax*100,greed*100))
plt.savefig('7-2.jpg')
print('Buy:',num_tran[0])
print('Sell:',num_tran[1])
print('zeta = {} %'.format(tax*100))
print('the final portfolio value is ',vg5_tax1[len(data)-div:].values[0][0])

#%% Q7.3 xigma = 0.1%

tax = 0.001
greed = 0.01

m_list,n_list,ntran_vg1_tax2 = v_k1_tax(future,gaussion_detector,greed,tax)
vg1_tax2 = m_list + n_list * st_future

m_list,n_list,ntran_vg5_tax2 = v_k5_tax(future,gaussion_detector,greed,tax)     
vg5_tax2 = m_list + n_list * st_future

plt.figure(figsize=(12,5),dpi=300)
plt.plot(vg1_tax2,label='k = 1',color='orange')
plt.plot(vg5_tax2,label='k = 5',color='steelblue')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title(r'Portfolio value after tax with gaussian detector ($\zeta$={} % and $\gamma$={} %)'.format(tax*100,greed*100))
plt.savefig('7-3.jpg')
print("Buy and Sell")
print(ntran_vg1_tax2[:2])
print(ntran_vg5_tax2[:2])
print('zeta = {} %'.format(tax*100))
print('the final portfolio value of k=1 is ',vg1_tax2[len(data)-div:].values[0][0])
print('the final portfolio value of k=5 is ',vg5_tax2[len(data)-div:].values[0][0])



#%% Q7.4 xigma = 0.5%

tax = 0.005
greed = 0.01

m_list,n_list,ntran_vg1_tax2 = v_k1_tax(future,gaussion_detector,greed,tax)
vg1_tax3 = m_list + n_list * st_future

m_list,n_list,ntran_vg5_tax2 = v_k5_tax(future,gaussion_detector,greed,tax)     
vg5_tax3 = m_list + n_list * st_future

plt.figure(figsize=(12,5),dpi=300)
plt.plot(vg1_tax3,label='k = 1',color='orange')
plt.plot(vg5_tax3,label='k = 5',color='steelblue')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title(r'Portfolio value after tax with gaussian detector ($\zeta$={} % and $\gamma$={} %)'.format(tax*100,greed*100))
plt.savefig('7-4.jpg')
# print("Buy and Sell")
# print(ntran_vg1_tax2[:2])
# print(ntran_vg5_tax2[:2])
print('zeta = {} %'.format(tax*100))
print('the final portfolio value of k=1 is ',vg1_tax3[len(data)-div:].values[0][0])
print('the final portfolio value of k=5 is ',vg5_tax3[len(data)-div:].values[0][0])


#%% 8.1 Trade with logistic and gaussion detector, 𝑟 = 0.001% with 𝜉 = 0.2% and 𝛾 = 𝛾0

tax = 0.001
greed = 0.01  
rate = 0.00001
def v_k_rate(future,dec,gamma,xigma,r):
    m = 100000
    n = 0
    m_list = np.zeros((len(future),1))
    n_list = np.zeros((len(future),1))
    k1=0
    k2=0
    k3=0
    k4=0
    for i in range(len(future)):
        if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
            if digitdata[div-1+i] == 'U':
                delta_m = gamma*m
                m = m - delta_m
                n = n + (1-xigma)*delta_m/st_future.iloc[i][0]
                m_list[i] = m
                n_list[i] = n
                k1 = k1+1
            else:
                m_list[i] = m
                n_list[i] = n
                k3 = k3+1
        else:
            if digitdata[div-1+i] == 'U':
                delta_n = gamma*n
                m = m +(1-xigma)* delta_n * st_future.iloc[i][0]
                n = n - delta_n
                m_list[i] = m
                n_list[i] = n
                k2 = k2+1
            else:
                m_list[i] = m
                n_list[i] = n
                k4 = k4+1
        m = m* (1+r)
    return m_list,n_list,(k1,k2,k3,k4)

# m_list,n_list,num_tran = v_k_rate(future,logistic_detector,greed,tax,rate)
# vl1_rate1 = m_list + n_list * st_future

m_list,n_list,ntran = v_k_rate(future,gaussion_detector,greed,tax,rate)
vg1_rate1 = m_list + n_list * st_future

plt.figure(figsize=(12,5),dpi=300)
# plt.plot(vl1_rate1,label='logistic detector')
plt.plot(vg1_rate1,label='gaussion detector')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('gaussian detector with best 1-day rule ($r$ = {}%)'.format(rate*100))
plt.savefig('8-1.jpg')
print('tax: {} %'.format(tax*100),'greed: {} %'.format(greed*100),'interest rate: {} %'.format(rate*100))
print('the resultant portfolio value: ',round(vg1_rate1[len(data)-div:].values[0][0],2))
print("Buy and Sell")
print(ntran[:2])


#%% 8.2 Ratio of portfolio compare to savings in bank

tax = 0.001
greed = 0.01  
rate = 0.001/100

def v_b_rate(future,r):
    m = 100000
    money = np.zeros((len(future),1))
    for i in range(len(future)):
        earn = m * (1+r)**i
        money[i] = earn
    return money

vb = v_b_rate(future,rate)

m_list,n_list,ntran_vg1_tax2 = v_k_rate(future,gaussion_detector,greed,tax,rate)
vg1_rate1 = m_list + n_list * st_future

ratio_gaus1 = vg1_rate1/vb

plt.figure(figsize=(12,5),dpi=300)
plt.plot(ratio_gaus1,label='gaussion detector')
plt.axhline(y=1.0,label = 'Natural growth',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('Ratio of gaussion detector return with pure interest saving. ($r$ = {}%)'.format(rate*100))
plt.savefig('8-2.jpg')
print('tax: {} %'.format(tax*100),'greed: {} %'.format(greed*100),'interest rate: {} %'.format(rate*100))
print('the resultant ratio of return with pure interest saving: ',round(ratio_gaus1[len(data)-div:].values[0][0],2))

#%% 8.3 Change the Interest rate and compare

rate = 0.005/100
vb2 = v_b_rate(future,rate)
m_list,n_list,ntran_vg1_tax2 = v_k_rate(future,gaussion_detector,greed,tax,rate)
vg1_rate2 = m_list + n_list * st_future
ratio_gaus2 = vg1_rate2/vb2

rate = 0.01/100
vb3 = v_b_rate(future,rate)
m_list,n_list,ntran_vg1_tax2 = v_k_rate(future,gaussion_detector,greed,tax,rate)
vg1_rate3 = m_list + n_list * st_future
ratio_gaus3 = vg1_rate3/vb3

rate = 0
vb = v_b_rate(future,rate)
m_list,n_list,ntran_vg1_tax2 = v_k_rate(future,gaussion_detector,greed,tax,rate)
vg1_rate = m_list + n_list * st_future
ratio_gaus = vg1_rate/vb

plt.figure(figsize=(12,5),dpi=300)
plt.plot(ratio_gaus,label='Interest rate=0%')
plt.plot(ratio_gaus1,label='Interest rate=0.001%')
plt.plot(ratio_gaus2,label='nterest rate=0.005%')
plt.plot(ratio_gaus3,label='Interest rate=0.01%')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('Ratio of gaussian detectors return with pure interest saving.')
plt.savefig('8-3.jpg')
print('the resultant ratio (rate = 0.000%): ',round(ratio_gaus[len(data)-div:].values[0][0],2))
print('the resultant ratio (rate = 0.001%): ',round(ratio_gaus1[len(data)-div:].values[0][0],2))
print('the resultant ratio (rate = 0.005%): ',round(ratio_gaus2[len(data)-div:].values[0][0],2))
print('the resultant ratio (rate = 0.01%): ',round(ratio_gaus3[len(data)-div:].values[0][0],2))


#%% 8.3 Ratio of portfolio compare to savings in bank, 𝑟 = 0.005%

tax = 0.002
greed = 0.01  
rate = 0.00005

vb1 = v_b_rate(future,rate)
m_list,n_list,num_tran = v_k_rate(future,logistic_detector,greed,tax,rate)
vl1_rate2 = m_list + n_list * st_future

m_list,n_list,ntran_vg1_tax2 = v_k_rate(future,gaussion_detector,greed,tax,rate)
vg1_rate2 = m_list + n_list * st_future

ratio_log1 = vl1_rate2/vb1
ratio_gaus1 = vg1_rate2/vb1

plt.figure(figsize=(12,5),dpi=300)
plt.plot(ratio_log1,label='logistic detector')
plt.plot(ratio_gaus1,label='gaussion detector')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('Ratio of 2 detectors return with pure interest saving. (Interest rate=0.005%)')
plt.savefig('8-3.jpg')
print('the resultant ratio of return with pure interest saving: ',round(ratio_gaus1[len(data)-div:].values[0][0],2))
print('the resultant ratio of return with pure interest saving: ',round(ratio_gaus2[len(data)-div:].values[0][0],2))
print('the resultant ratio of return with pure interest saving: ',round(ratio_gaus3[len(data)-div:].values[0][0],2))



#%% 9.1 choose 20 different value for gamma

def v_k_rate(future,dec,gamma,xigma,r):
    m = 100000
    n = 0
    m_list = np.zeros((len(future),1))
    n_list = np.zeros((len(future),1))
    k1=0
    k2=0
    k3=0
    k4=0
    for i in range(len(future)):
        if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
            if digitdata[div-1+i] == 'U':
                delta_m = gamma*m
                m = m - delta_m
                n = n + (1-xigma)*delta_m/st_future.iloc[i][0]
                m_list[i] = m
                n_list[i] = n
                k1 = k1+1
            else:
                m_list[i] = m
                n_list[i] = n
                k3 = k3+1
        else:
            if digitdata[div-1+i] == 'U':
                delta_n = gamma*n
                m = m +(1-xigma)* delta_n * st_future.iloc[i][0]
                n = n - delta_n
                m_list[i] = m
                n_list[i] = n
                k2 = k2+1
            else:
                m_list[i] = m
                n_list[i] = n
                k4 = k4+1
        m = m * (1+r)
    return m_list,n_list,(k1,k2,k3,k4)


gamma_choice = np.linspace(0,1,20, endpoint = False)
gamma_choice = np.round(gamma_choice, 2)
tax = 0.002
rate = 0.00001
portfolio_values_gaussion = []
for gamma in gamma_choice:
    m_list,n_list,ntran_vg1_tax2 = v_k_rate(future,gaussion_detector,gamma,tax,rate)
    vg1_rate5 = m_list + n_list * st_future
    portfolio_values_gaussion.append(vg1_rate5)
    

plt.figure(figsize=(12,5),dpi=300)
for gamma, vg1_rate5 in zip(gamma_choice, portfolio_values_gaussion):
    plt.plot(vg1_rate5, label=f"Gamma = {gamma}")

plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('Transactions based on gaussion detector with different greed value')
plt.legend(loc="upper left", fontsize=8)
plt.savefig('9-1.jpg')
plt.show()


#%% 9.2 Plot the portfolios' final values, highest value and time average against 𝛾.

final_value_gaussion = []
highest_value_gaussion=[]
time_average_gaussion=[]

for vg1_rate5 in portfolio_values_gaussion:
    final_value_gaussion.append(vg1_rate5.iloc[-1,0])
    highest_value_gaussion.append(vg1_rate5.iloc[:, 0].max())
    time_average_gaussion.append(vg1_rate5.iloc[:, 0].mean())

plt.figure(figsize=(12,5),dpi=300)
plt.plot(gamma_choice,final_value_gaussion,label='final value')
plt.plot(gamma_choice,highest_value_gaussion,label='highest value')
plt.plot(gamma_choice,time_average_gaussion,label='time average')
plt.legend()
plt.savefig('9-2.jpg')
plt.show()

#%% 10 Trade with m and n as usual, 𝑟 = 0.001% with 𝜉 = 50% and 𝛾 = 𝛾0

tax = 0.5
greed = 0.01  
rate = 0.00001

def v_k_rate(future,dec,gamma,xigma,r):
    m = 100000
    n = 0
    m_list = np.zeros((len(future),1))
    n_list = np.zeros((len(future),1))
    k1=0
    k2=0
    k3=0
    k4=0
    for i in range(len(future)):
        if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
            if digitdata[div-1+i] == 'U':
                delta_m = gamma*m
                m = m - delta_m
                n = n + (1-xigma)*delta_m/st_future.iloc[i][0]
                m_list[i] = m
                n_list[i] = n
                k1 = k1+1
            else:
                m_list[i] = m
                n_list[i] = n
                k3 = k3+1
        else:
            if digitdata[div-1+i] == 'U':
                delta_n = gamma*n
                m = m +(1-xigma)* delta_n * st_future.iloc[i][0]
                n = n - delta_n
                m_list[i] = m
                n_list[i] = n
                k2 = k2+1
            else:
                m_list[i] = m
                n_list[i] = n
                k4 = k4+1
        m = m * (1+r)
    return m_list,n_list,(k1,k2,k3,k4)

m_list,n_list,num_tran = v_k_rate(future,logistic_detector,greed,tax,rate)
vl1_rate1 = m_list + n_list * st_future
m_list,n_list,ntran_vg1_tax2 = v_k_rate(future,gaussion_detector,greed,tax,rate)
vg1_rate1 = m_list + n_list * st_future

plt.figure(figsize=(12,5),dpi=300)
plt.plot(vg1_rate1,label='gaussion detector')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('2 detectors with best 1-day rule (m and n as usual)')
plt.savefig('10-1.jpg')

#%% Trade with  𝑚̃   and  𝑛̃  , 𝑟 = 0.001% with 𝜉 = 50% and 𝛾 = 𝛾0
tax = 0.5
greed = 0.01  
rate = 0.00001

def v_k_rate(future,dec,gamma,xigma,r):
    m = 100000
    n = 0
    m_list = np.zeros((len(future),1))
    n_list = np.zeros((len(future),1))
    k1=0
    k2=0
    k3=0
    k4=0
    for i in range(len(future)):
        V = m + n * st_future.iloc[i][0]
        if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
            if digitdata[div-1+i] == 'U':
                gamma_u = (gamma*V) / (V-(1-gamma)*tax*m)
                delta_m = gamma_u*m
                m = m - delta_m
                n = n + (1-xigma)*delta_m/st_future.iloc[i][0]
                m_list[i] = m
                n_list[i] = n
                k1 = k1+1
            else:
                m_list[i] = m
                n_list[i] = n
                k3 = k3+1
        else:
            if digitdata[div-1+i] == 'U':
                gamma_d = (gamma*V) / (V-(1-gamma)*tax*n*st_future.iloc[i][0])
                delta_n = gamma_d*n
                m = m +(1-xigma)* delta_n * st_future.iloc[i][0]
                n = n - delta_n
                m_list[i] = m
                n_list[i] = n
                k2 = k2+1
            else:
                m_list[i] = m
                n_list[i] = n
                k4 = k4+1
        m = m * (1+r)

    return m_list,n_list,(k1,k2,k3,k4)

m_list,n_list,num_tran = v_k_rate(future,logistic_detector,greed,tax,rate)
vl1_rate2 = m_list + n_list * st_future

m_list,n_list,ntran_vg1_tax2 = v_k_rate(future,gaussion_detector,greed,tax,rate)
vg1_rate2 = m_list + n_list * st_future

plt.figure(figsize=(12,5),dpi=300)
plt.plot(vg1_rate2,label='gaussion detector')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('2 detectors with best 1-day rule ('+r'$\tilde{m}$'+' and'+r' $\tilde{n}$)')
plt.savefig('10-2.jpg')


#%% 10.3 Compare

plt.figure(figsize=(12,5),dpi=300)
plt.plot(vg1_rate1,label='Constant gamma - Gaussion')
plt.plot(vg1_rate2,label='Variation gamma - Gaussion')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('2 detectors with best 1-day rule ('+r'$\tilde{m}$'+' and'+r' $\tilde{n}$)')
plt.savefig('10-3.jpg')



#%% 11.1 Posterior analysis

def v_k_rate(future,dec,gamma,xigma,r):
    m = 100000
    n = 0
    m_list = np.zeros((len(future),1))
    n_list = np.zeros((len(future),1))
    k1=0
    k2=0
    k3=0
    k4=0
    for i in range(len(future)):
        if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
            if digitdata[div-1+i] == 'U':
                delta_m = gamma*m
                m = m - delta_m
                n = n + (1-xigma)*delta_m/st_future.iloc[i][0]
                m_list[i] = m
                n_list[i] = n
                k1 = k1+1
            else:
                m_list[i] = m
                n_list[i] = n
                k3 = k3+1
        else:
            if digitdata[div-1+i] == 'U':
                delta_n = gamma*n
                m = m +(1-xigma)* delta_n * st_future.iloc[i][0]
                n = n - delta_n
                m_list[i] = m
                n_list[i] = n
                k2 = k2+1
            else:
                m_list[i] = m
                n_list[i] = n
                k4 = k4+1
        m = m * (1+r)
    return m_list,n_list,(k1,k2,k3,k4)

gamma_choice = np.linspace(0,1,20, endpoint = False)
gamma_choice = np.round(gamma_choice, 2)
tax = 0.002
rate = 0.00001

portfolio_gaussion = pd.DataFrame(index=st_future.index)
for gamma in gamma_choice:
    m_list,n_list,ntran_vg1_tax2 = v_k_rate(future,gaussion_detector,gamma,tax,rate)
    vg1_rate6 = m_list + n_list * st_future
    portfolio_gaussion = pd.concat([portfolio_gaussion, vg1_rate6.rename(columns={st_future.columns[0]: 'gamma=%.2f'%gamma})], axis=1)

gammag_i_star = (portfolio_gaussion / portfolio_gaussion.shift(1)).idxmax(axis=1)
gammag_i_star.dropna(inplace=True)
gammag_i_star = pd.to_numeric(gammag_i_star.str[6:])


plt.figure(figsize=(30,5),dpi=300)
plt.plot(range(1, len(gammag_i_star)+1), gammag_i_star.values, '.-')
plt.xlabel('i')
plt.ylabel('$\gamma^{*}_{i}$')
plt.title('The optimal greed with gaussion detector')
plt.savefig('11-1.jpg')


#%% 11.1 Plot the resultant portfolio’s value

def v_k_rate_gaussion(future,dec,xigma,r):
    m = 100000
    n = 0
    m_list = np.zeros((len(future),1))
    n_list = np.zeros((len(future),1))
    k1=0
    k2=0
    k3=0
    k4=0
    for i in range(len(future)):
        gamma = gammag_i_star[i]
        if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
            if digitdata[div-1+i] == 'U':
                delta_m = gamma*m
                m = m - delta_m
                n = n + (1-xigma)*delta_m/st_future.iloc[i][0]
                m_list[i] = m
                n_list[i] = n
                k1 = k1+1
            else:
                m_list[i] = m
                n_list[i] = n
                k3 = k3+1
        else:
            if digitdata[div-1+i] == 'U':
                delta_n = gamma*n
                m = m +(1-xigma)* delta_n * st_future.iloc[i][0]
                n = n - delta_n
                m_list[i] = m
                n_list[i] = n
                k2 = k2+1
            else:
                m_list[i] = m
                n_list[i] = n
                k4 = k4+1
        m = m * (1+r)
    return m_list,n_list,(k1,k2,k3,k4)

tax = 0.002
rate = 0.00001


m_list,n_list,ntran_vg1_tax2 = v_k_rate_gaussion(future[1:],gaussion_detector,tax,rate)
vg1_rate7 = m_list + n_list * st_future[1:]

plt.figure(figsize=(12,5),dpi=300)
plt.plot(vg1_rate7,label='gaussion detector')
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('Portfolio\'s value $V^{*}(t)$ based on the optimal greed $\gamma^{*}_{i}$')
plt.savefig('11-2.jpg')


#%% 11.2 Prior analysis


tax = 0.002
rate = 0.00001
gamma_A = 0.7
gamma_C = 0.3
max_Bob = []
final_Bob = []
inter = np.linspace(1,100,99, endpoint = False)
for inter_ in inter:
    def v_k_rate_Bob(future,dec,xigma,r):
        m = 100000
        n = 0
        m_list = np.zeros((len(future),1))
        n_list = np.zeros((len(future),1))
        k1=0
        k2=0
        k3=0
        k4=0
        flag = 1
        gamma_choice = [0.7, 0.3]
        for i in range(len(future)):
            if i % inter_ == 0:
                flag = (flag+1) % 2
            gamma = gamma_choice[flag]
            if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
                if digitdata[div-1+i] == 'U':
                    delta_m = gamma*m
                    m = m - delta_m
                    n = n + (1-xigma)*delta_m/st_future.iloc[i][0]
                    m_list[i] = m
                    n_list[i] = n
                    k1 = k1+1
                else:
                    m_list[i] = m
                    n_list[i] = n
                    k3 = k3+1
            else:
                if digitdata[div-1+i] == 'U':
                    delta_n = gamma*n
                    m = m +(1-xigma)* delta_n * st_future.iloc[i][0]
                    n = n - delta_n
                    m_list[i] = m
                    n_list[i] = n
                    k2 = k2+1
                else:
                    m_list[i] = m
                    n_list[i] = n
                    k4 = k4+1
            m = m * (1+r)
        return m_list,n_list,(k1,k2,k3,k4)
    
    def v_k_rate(future,dec,gamma,xigma,r):
        m = 100000
        n = 0
        m_list = np.zeros((len(future),1))
        n_list = np.zeros((len(future),1))
        k1=0
        k2=0
        k3=0
        k4=0
    
        for i in range(len(future)):
            if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
                if digitdata[div-1+i] == 'U':
                    delta_m = gamma*m
                    m = m - delta_m
                    n = n + (1-xigma)*delta_m/st_future.iloc[i][0]
                    m_list[i] = m
                    n_list[i] = n
                    k1 = k1+1
                else:
                    m_list[i] = m
                    n_list[i] = n
                    k3 = k3+1
            else:
                if digitdata[div-1+i] == 'U':
                    delta_n = gamma*n
                    m = m +(1-xigma)* delta_n * st_future.iloc[i][0]
                    n = n - delta_n
                    m_list[i] = m
                    n_list[i] = n
                    k2 = k2+1
                else:
                    m_list[i] = m
                    n_list[i] = n
                    k4 = k4+1
            m = m * (1+r)
        return m_list,n_list,(k1,k2,k3,k4)



    m_list,n_list,num_tran = v_k_rate_Bob(future,gaussion_detector,tax,rate)
    vl1_rate_Bob = m_list + n_list * st_future
    max_Bob.append( vl1_rate_Bob.max()[0])  
    final_Bob.append(vl1_rate_Bob[len(data)-div:].values[0][0])

plt.figure(figsize=(14,5),dpi=300)
plt.plot(inter,max_Bob, label='max of Bob', linewidth=2)
plt.plot(inter,final_Bob, label='final of Bob', linewidth=2)
plt.legend()
plt.savefig('11-3.jpg')
print('When the interval = 10 days, Bob stragety perfoms best')

def v_k_rate_Bob(future,dec,xigma,r):
    m = 100000
    n = 0
    m_list = np.zeros((len(future),1))
    n_list = np.zeros((len(future),1))
    k1=0
    k2=0
    k3=0
    k4=0
    flag = 1
    gamma_choice = [0.7, 0.3]
    for i in range(len(future)):
        if i % 10 == 0:
            flag = (flag+1) % 2
        gamma = gamma_choice[flag]
        if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
            if digitdata[div-1+i] == 'U':
                delta_m = gamma*m
                m = m - delta_m
                n = n + (1-xigma)*delta_m/st_future.iloc[i][0]
                m_list[i] = m
                n_list[i] = n
                k1 = k1+1
            else:
                m_list[i] = m
                n_list[i] = n
                k3 = k3+1
        else:
            if digitdata[div-1+i] == 'U':
                delta_n = gamma*n
                m = m +(1-xigma)* delta_n * st_future.iloc[i][0]
                n = n - delta_n
                m_list[i] = m
                n_list[i] = n
                k2 = k2+1
            else:
                m_list[i] = m
                n_list[i] = n
                k4 = k4+1
        m = m * (1+r)
    return m_list,n_list,(k1,k2,k3,k4)

def v_k_rate(future,dec,gamma,xigma,r):
    m = 100000
    n = 0
    m_list = np.zeros((len(future),1))
    n_list = np.zeros((len(future),1))
    k1=0
    k2=0
    k3=0
    k4=0

    for i in range(len(future)):
        if (future.iloc[i][0] > dec[0]) and (future.iloc[i][0]<dec[1]):
            if digitdata[div-1+i] == 'U':
                delta_m = gamma*m
                m = m - delta_m
                n = n + (1-xigma)*delta_m/st_future.iloc[i][0]
                m_list[i] = m
                n_list[i] = n
                k1 = k1+1
            else:
                m_list[i] = m
                n_list[i] = n
                k3 = k3+1
        else:
            if digitdata[div-1+i] == 'U':
                delta_n = gamma*n
                m = m +(1-xigma)* delta_n * st_future.iloc[i][0]
                n = n - delta_n
                m_list[i] = m
                n_list[i] = n
                k2 = k2+1
            else:
                m_list[i] = m
                n_list[i] = n
                k4 = k4+1
        m = m * (1+r)
    return m_list,n_list,(k1,k2,k3,k4)

tax = 0.002
rate = 0.00001
gamma_A = 0.7
gamma_C = 0.3

m_list,n_list,num_tran = v_k_rate_Bob(future,gaussion_detector,tax,rate)
vl1_rate_Bob = m_list + n_list * st_future

m_list,n_list,num_tran = v_k_rate(future,gaussion_detector,gamma_A, tax,rate)
vl1_rate_Alice = m_list + n_list * st_future

m_list,n_list,num_tran = v_k_rate(future,gaussion_detector,gamma_C, tax,rate)
vl1_rate_Charlie = m_list + n_list * st_future

plt.figure(figsize=(14,5),dpi=300)
plt.plot(vl1_rate_Bob, label='Bob', linewidth=1)
plt.plot(vl1_rate_Alice, label='Alice', linewidth=1)
plt.plot(vl1_rate_Charlie, label='Charlie', linewidth=1)
plt.axhline(y=100000,label = 'original value',ls='dashed', color='red')
plt.legend()
plt.gca().xaxis.set_major_locator(YearLocator())
plt.xticks(rotation=0, ha='right')
plt.title('$V_A(t)$, $V_B(t)$ and $V_C(t)$')
plt.savefig('11-4.jpg')



