# -*- coding: utf-8 -*-
"""
金融資料視覺化看板

@author: 
"""
import random
from itertools import product  # 添加這行
# 載入必要模組
import os
#import haohaninfo
#from order_Lo8 import Record
import numpy as np
#from talib.abstract import SMA,EMA, WMA, RSI, BBANDS, MACD
#import sys
import indicator_f_Lo2_short,datetime, indicator_forKBar_short
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 
from order_streamlit import Record
import matplotlib.pyplot as plt
import matplotlib

#%%
####### (1) 開始設定 #######
###### 設定網頁標題介面 
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">   
		<h1 style="color:white;text-align:center;">金融看板與程式交易平台 </h1>
		<h2 style="color:white;text-align:center;">Financial Dashboard and Program Trading </h2>
		</div>
		"""
stc.html(html_temp)


###### 讀取資料
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def load_data(path):
    df = pd.read_pickle(path)
    return df
# ##### 讀取 excel 檔
# df_original = pd.read_excel("kbars_2330_2022-01-01-2022-11-18.xlsx")


###### 選擇金融商品
st.subheader("選擇金融商品: ")
choices = [
    '台積電 2330: 2020.01.02 至 2025.04.16',
    '大台指期貨2024.12到期: 2023.12 至 2024.4.11',
    '小台指期貨2024.12到期: 2023.12 至 2024.4.11',
    '英業達 2356: 2020.01.02 至 2024.04.12',
    '堤維西 1522: 2020.01.02 至 2024.04.12',
    '0050 台灣50ETF: 2020.01.02 至 2025.03.10',
    '00631L 台灣50正2: 2023.04.17 至 2025.04.17',
    '華碩 2357: 2023.04.17 至 2025.04.16',
    '金融期貨 CBF: 2023.04.17 至 2025.04.17',
    '電子期貨 CCF: 2023.04.17 至 2025.04.16',
    '小型電子期貨 CDF: 2020.03.02 至 2025.04.14',
    '非金電期貨 CEF: 2023.04.17 至 2025.04.16',
    '摩台期貨 CMF: 2023.04.17 至 2025.04.17',
    '小型金融期貨 CQF: 2023.04.17 至 2025.04.17',
    '美元指數期貨 FXF: 2020.03.02 至 2025.04.14'
]
choice = st.selectbox('選擇金融商品', choices, index=0)

# 對應每個選項的 pkl 檔與日期範圍
product_info = {
    choices[0]: ('exported/kbars_1min_2330_2020-01-02_To_2025-04-16.pkl', '台積電 2330', '2020-01-02', '2025-04-16'),
    choices[1]: ('exported/kbars_TXF202412_2023-12-21-2024-04-11.pkl', '大台指期貨', '2023-12-21', '2024-04-11'),
    choices[2]: ('exported/kbars_MXF202412_2023-12-21-2024-04-11.pkl', '小台指期貨', '2023-12-21', '2024-04-11'),
    choices[3]: ('exported/kbars_2356_2020-01-01-2024-04-12.pkl', '英業達 2356', '2020-01-02', '2024-04-12'),
    choices[4]: ('exported/kbars_1522_2020-01-01-2024-04-12.pkl', '堤維西 1522', '2020-01-02', '2024-04-12'),
    choices[5]: ('exported/kbars_1min_0050_2020-01-02_To_2025-03-10.pkl', '台灣50ETF 0050', '2020-01-02', '2025-03-10'),
    choices[6]: ('exported/kbars_1min_00631L_2023-04-17_To_2025-04-17.pkl', '台灣50正2 00631L', '2023-04-17', '2025-04-17'),
    choices[7]: ('exported/kbars_1min_2357_2023-04-17_To_2025-04-16.pkl', '華碩 2357', '2023-04-17', '2025-04-16'),
    choices[8]: ('exported/kbars_1min_CBF_2023-04-17_To_2025-04-17.pkl', '金融期貨 CBF', '2023-04-17', '2025-04-17'),
    choices[9]: ('exported/kbars_1min_CCF_2023-04-17_To_2025-04-16.pkl', '電子期貨 CCF', '2023-04-17', '2025-04-16'),
    choices[10]: ('exported/kbars_1min_CDF_2020-03-02_To_2025-04-14.pkl', '小型電子期貨 CDF', '2020-03-02', '2025-04-14'),
    choices[11]: ('exported/kbars_1min_CEF_2023-04-17_To_2025-04-16.pkl', '非金電期貨 CEF', '2023-04-17', '2025-04-16'),
    choices[12]: ('exported/kbars_1min_CMF_2023-04-17_To_2025-04-17.pkl', '摩台期貨 CMF', '2023-04-17', '2025-04-17'),
    choices[13]: ('exported/kbars_1min_CQF_2023-04-17_To_2025-04-17.pkl', '小型金融期貨 CQF', '2023-04-17', '2025-04-17'),
    choices[14]: ('exported/kbars_1min_FXF_2020-03-02_To_2025-04-14.pkl', '美元指數期貨 FXF', '2020-03-02', '2025-04-14'),
}

# 載入資料
pkl_path, product_name, default_start, default_end = product_info[choice]
df_original = load_data(pkl_path)

###### 選擇資料區間
st.subheader("選擇資料時間區間")
start_date_str = st.text_input(f'輸入開始日期 (格式: YYYY-MM-DD)，區間: {default_start} 至 {default_end}', default_start)
end_date_str = st.text_input(f'輸入結束日期 (格式: YYYY-MM-DD)，區間: {default_start} 至 {default_end}', default_end)

# 轉為 datetime 並篩選資料
start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

#%%
####### (2) 轉化為字典 #######
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def To_Dictionary_1(df, product_name):
    KBar_dic = df.to_dict()
    KBar_open_list = list(KBar_dic['open'].values())
    KBar_dic['open']=np.array(KBar_open_list)
    
    KBar_dic['product'] = np.repeat(product_name, KBar_dic['open'].size)
    #KBar_dic['product'].size   ## 1596
    #KBar_dic['product'][0]      ## 'tsmc'
    
    KBar_time_list = list(KBar_dic['time'].values())
    KBar_time_list = [i.to_pydatetime() for i in KBar_time_list] ## Timestamp to datetime
    KBar_dic['time']=np.array(KBar_time_list)
    
    KBar_low_list = list(KBar_dic['low'].values())
    KBar_dic['low']=np.array(KBar_low_list)
    
    KBar_high_list = list(KBar_dic['high'].values())
    KBar_dic['high']=np.array(KBar_high_list)
    
    KBar_close_list = list(KBar_dic['close'].values())
    KBar_dic['close']=np.array(KBar_close_list)
    
    KBar_volume_list = list(KBar_dic['volume'].values())
    KBar_dic['volume']=np.array(KBar_volume_list)
    
    KBar_amount_list = list(KBar_dic['amount'].values())
    KBar_dic['amount']=np.array(KBar_amount_list)
    
    return KBar_dic

KBar_dic = To_Dictionary_1(df, product_name)


#%%
#######  (3) 改變 KBar 時間長度 & 形成 KBar 字典 (新週期的) & Dataframe #######
###### 定義函數: 進行 K 棒更新  &  形成 KBar 字典 (新週期的): 設定cycle_duration可以改成你想要的 KBar 週期
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Change_Cycle(Date,cycle_duration,KBar_dic,product_name):
    ###### 進行 K 棒更新
    KBar = indicator_forKBar_short.KBar(Date,cycle_duration)    ## 設定cycle_duration可以改成你想要的 KBar 週期
    for i in range(KBar_dic['time'].size):
        #time = datetime.datetime.strptime(KBar_dic['time'][i],'%Y%m%d%H%M%S%f')
        time = KBar_dic['time'][i]
        #prod = KBar_dic['product'][i]
        open_price= KBar_dic['open'][i]
        close_price= KBar_dic['close'][i]
        low_price= KBar_dic['low'][i]
        high_price= KBar_dic['high'][i]
        qty =  KBar_dic['volume'][i]
        amount = KBar_dic['amount'][i]
        #tag=KBar.TimeAdd(time,price,qty,prod)
        tag=KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)
    
    ###### 形成 KBar 字典 (新週期的):
    KBar_dic = {}
    KBar_dic['time'] =  KBar.TAKBar['time']   
    #KBar_dic['product'] =  KBar.TAKBar['product']
    KBar_dic['product'] = np.repeat(product_name, KBar_dic['time'].size)
    KBar_dic['open'] = KBar.TAKBar['open']
    KBar_dic['high'] =  KBar.TAKBar['high']
    KBar_dic['low'] =  KBar.TAKBar['low']
    KBar_dic['close'] =  KBar.TAKBar['close']
    KBar_dic['volume'] =  KBar.TAKBar['volume']
    
    return KBar_dic
    

###### 改變日期資料型態
Date = start_date.strftime("%Y-%m-%d")  ## 變成字串


st.subheader("設定技術指標視覺化圖形之相關參數:")

###### 設定 K 棒的時間長度(分鐘): 
with st.expander("設定K棒相關參數:"):
    choices_unit = ['以分鐘為單位','以日為單位','以週為單位','以月為單位']
    choice_unit = st.selectbox('選擇計算K棒時間長度之單位', choices_unit, index=1)
    if choice_unit == '以分鐘為單位':
        cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:分鐘, 一日=1440分鐘)', value=1, key="KBar_duration_分")
        cycle_duration = float(cycle_duration)
    if choice_unit == '以日為單位':
        cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:日)', value=1, key="KBar_duration_日")
        cycle_duration = float(cycle_duration)
        cycle_duration = cycle_duration*1440
    if choice_unit == '以週為單位':
        cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:週)', value=1, key="KBar_duration_週")
        cycle_duration = float(cycle_duration)
        cycle_duration = cycle_duration*7*1440
    if choice_unit == '以月為單位':
        cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:月, 一月=30天)', value=1, key="KBar_duration_月")
        cycle_duration = float(cycle_duration)
        cycle_duration = cycle_duration*30*1440


###### 進行 K 棒更新  & 形成 KBar 字典 (新週期的)
KBar_dic = Change_Cycle(Date,cycle_duration,KBar_dic,product_name)   ## 設定cycle_duration可以改成你想要的 KBar 週期

###### 將K線 Dictionary 轉換成 Dataframe
KBar_df = pd.DataFrame(KBar_dic)


#%%
####### (4) 計算各種技術指標 #######

#%%
######  (i) 移動平均線策略 
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Calculate_MA(df, period=10):
    ##### 計算長短移動平均線
    ma = df['close'].rolling(window=period).mean()
    return ma
  
#####  設定長短移動平均線的 K棒 長度:
with st.expander("設定長短移動平均線的 K棒 長度:"):
    # st.subheader("設定計算長移動平均線(MA)的 K棒週期數目(整數, 例如 10)")
    LongMAPeriod=st.slider('設定計算長移動平均線(MA)的 K棒週期數目(整數, 例如 10)', 0, 100, 10, key='visualization_MA_long')
    # st.subheader("設定計算短移動平均線(MA)的 K棒週期數目(整數, 例如 2)")
    ShortMAPeriod=st.slider('設定計算短移動平均線(MA)的 K棒週期數目(整數, 例如 2)', 0, 100, 2, key='visualization_MA_short')

##### 計算長短移動平均線
KBar_df['MA_long'] = Calculate_MA(KBar_df, period=LongMAPeriod)
KBar_df['MA_short'] = Calculate_MA(KBar_df, period=ShortMAPeriod)

##### 尋找最後 NAN值的位置
last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]


#%%
######  (ii) RSI 策略 
##### 假设 df 是一个包含价格数据的Pandas DataFrame，其中 'close' 是KBar週期收盤價
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Calculate_RSI(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
  
##### 順勢策略
#### 設定長短 RSI 的 K棒 長度:
with st.expander("設定長短 RSI 的 K棒 長度:"):
    # st.subheader("設定計算長RSI的 K棒週期數目(整數, 例如 10)")
    LongRSIPeriod=st.slider('設定計算長RSI的 K棒週期數目(整數, 例如 10)', 0, 1000, 10, key='visualization_RSI_long')
    # st.subheader("設定計算短RSI的 K棒週期數目(整數, 例如 2)")
    ShortRSIPeriod=st.slider('設定計算短RSI的 K棒週期數目(整數, 例如 2)', 0, 1000, 2, key='visualization_RSI_short')

#### 計算 RSI指標長短線, 以及定義中線
KBar_df['RSI_long'] = Calculate_RSI(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = Calculate_RSI(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle']=np.array([50]*len(KBar_dic['time']))

#### 尋找最後 NAN值的位置
last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]


# ##### 逆勢策略
# #### 建立部位管理物件
# OrderRecord=Record() 
# #### 計算 RSI指標, 天花板與地板
# RSIPeriod=5
# Ceil=80
# Floor=20
# MoveStopLoss=30
# KBar_dic['RSI']=RSI(KBar_dic,timeperiod=RSIPeriod)
# KBar_dic['Ceil']=np.array([Ceil]*len(KBar_dic['time']))
# KBar_dic['Floor']=np.array([Floor]*len(KBar_dic['time']))

# #### 將K線 Dictionary 轉換成 Dataframe
# KBar_RSI_df=pd.DataFrame(KBar_dic)


#%%
######  (iii) Bollinger Band (布林通道) 策略 
##### 假设df是包含价格数据的Pandas DataFrame，'close'列是每日收盘价格
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Calculate_Bollinger_Bands(df, period=20, num_std_dev=2):
    df['SMA'] = df['close'].rolling(window=period).mean()
    df['Standard_Deviation'] = df['close'].rolling(window=period).std()
    df['Upper_Band'] = df['SMA'] + (df['Standard_Deviation'] * num_std_dev)
    df['Lower_Band'] = df['SMA'] - (df['Standard_Deviation'] * num_std_dev)
    return df


#####  設定布林通道(Bollinger Band)相關參數:
with st.expander("設定布林通道(Bollinger Band)相關參數:"):
    # st.subheader("設定計算布林通道(Bollinger Band)上中下三通道之K棒週期數目(整數, 例如 20)")
    period = st.slider('設定計算布林通道(Bollinger Band)上中下三通道之K棒週期數目(整數, 例如 20)', 0, 100, 20, key='BB_period')
    # st.subheader("設定計算布林通道(Bollinger Band)上中(或下中)通道之帶寬(例如 2 代表上中通道寬度為2倍的標準差)")
    num_std_dev = st.slider('設定計算布林通道(Bollinger Band)上中(或下中)通道之帶寬(例如 2 代表上中通道寬度為2倍的標準差)', 0, 100, 2, key='BB_heigh')

##### 計算布林通道上中下通道:
KBar_df = Calculate_Bollinger_Bands(KBar_df, period, num_std_dev)

##### 尋找最後 NAN值的位置
last_nan_index_BB = KBar_df['SMA'][::-1].index[KBar_df['SMA'][::-1].apply(pd.isna)][0]


#%%
######  (iv) MACD(異同移動平均線) 策略 
# 假设df是包含价格数据的Pandas DataFrame，'price'列是每日收盘价格
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Calculate_MACD(df, fast_period=12, slow_period=26, signal_period=9):
    df['EMA_Fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']  ## DIF
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()   ## DEA或信號線
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']  ## MACD = DIF-DEA
    return df

#####  設定MACD三種週期的K棒長度:
with st.expander("設定MACD三種週期的K棒長度:"):
    # st.subheader("設定計算 MACD的快速線週期(例如 12根日K)")
    fast_period = st.slider('設定計算 MACD快速線的K棒週期數目(例如 12根日K)', 0, 100, 12, key='visualization_MACD_quick')
    # st.subheader("設定計算 MACD的慢速線週期(例如 26根日K)")
    slow_period = st.slider('設定計算 MACD慢速線的K棒週期數目(例如 26根日K)', 0, 100, 26, key='visualization_MACD_slow')
    # st.subheader("設定計算 MACD的訊號線週期(例如 9根日K)")
    signal_period = st.slider('設定計算 MACD訊號線的K棒週期數目(例如 9根日K)', 0, 100, 9, key='visualization_MACD_signal')

##### 計算MACD:
KBar_df = Calculate_MACD(KBar_df, fast_period, slow_period, signal_period)

##### 尋找最後 NAN值的位置
# last_nan_index_MACD = KBar_df['MACD'][::-1].index[KBar_df['MACD'][::-1].apply(pd.isna)][0]
#### 試著找出最後一個 NaN 值的索引，但在這之前要檢查是否有 NaN 值
nan_indexes_MACD = KBar_df['MACD'][::-1].index[KBar_df['MACD'][::-1].apply(pd.isna)]
if len(nan_indexes_MACD) > 0:
    last_nan_index_MACD = nan_indexes_MACD[0]
else:
    last_nan_index_MACD = 0

#%%
######  (v) ATR（Average True Range）波動率指標 
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Calculate_ATR(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

with st.expander("設定 ATR 波動率參數:"):
    atr_period = st.slider("設定 ATR 計算週期", 1, 100, 14, key='visualization_ATR')

KBar_df['ATR'] = Calculate_ATR(KBar_df, atr_period)
last_nan_index_ATR = KBar_df['ATR'][::-1].index[KBar_df['ATR'][::-1].apply(pd.isna)][0]

#%%
######  (vi) OBV（On-Balance Volume）指標
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Calculate_OBV(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['close'][i] > df['close'][i-1]:
            obv.append(obv[-1] + df['volume'][i])
        elif df['close'][i] < df['close'][i-1]:
            obv.append(obv[-1] - df['volume'][i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

KBar_df['OBV'] = Calculate_OBV(KBar_df)
last_nan_index_OBV = 0  # OBV 一開始就不會有 NaN

#%%
#####  (vii) CCI - 商品通道指標
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Calculate_CCI(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad)
    return cci

with st.expander("設定 CCI 參數"):
    cci_period = st.slider("CCI 計算週期", 1, 100, 20)
KBar_df['CCI'] = Calculate_CCI(KBar_df, cci_period)


#%%
##### (viii) Stochastic Oscillator - KD 隨機震盪指標

@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Calculate_KD(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    rsv = 100 * (df['close'] - low_min) / (high_max - low_min)
    k = rsv.ewm(alpha=1/d_period, adjust=False).mean()
    d = k.ewm(alpha=1/d_period, adjust=False).mean()
    return k, d

with st.expander("設定 KD 隨機震盪指標參數"):
    k_period = st.slider("K 週期", 1, 100, 14)
    d_period = st.slider("D 平滑週期", 1, 10, 3)
KBar_df['K'], KBar_df['D'] = Calculate_KD(KBar_df, k_period, d_period)

#%%
#####  (ix) WILLR - 威廉指標（逆勢超買超賣）

@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Calculate_WILLR(df, period=14):
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    willr = -100 * (high_max - df['close']) / (high_max - low_min)
    return willr

with st.expander("設定 威廉指標 (WILLR) 參數"):
    willr_period = st.slider("WILLR 計算週期", 1, 100, 14)
KBar_df['WILLR'] = Calculate_WILLR(KBar_df, willr_period)

#%%
##### (x) MFI - 資金流量指標（RSI + 成交量）

@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Calculate_MFI(df, period=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos_mf = mf.where(tp > tp.shift(), 0)
    neg_mf = mf.where(tp < tp.shift(), 0)
    mfr = pos_mf.rolling(window=period).sum() / neg_mf.rolling(window=period).sum()
    mfi = 100 - (100 / (1 + mfr))
    return mfi

with st.expander("設定 MFI（資金流量）參數"):
    mfi_period = st.slider("MFI 計算週期", 1, 100, 14)
KBar_df['MFI'] = Calculate_MFI(KBar_df, mfi_period)

#%%
#####  (xi) ROC - 價格變動率指標
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Calculate_ROC(df, period=12):
    roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
    return roc

with st.expander("設定 ROC（變動率）參數"):
    roc_period = st.slider("ROC 計算週期", 1, 100, 12)
KBar_df['ROC'] = Calculate_ROC(KBar_df, roc_period)

#%%
#####  (xii) Momentum - 動能指標
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Calculate_MOM(df, period=10):
    return df['close'] - df['close'].shift(period)

with st.expander("設定 MOM（動能）參數"):
    mom_period = st.slider("MOM 計算週期", 1, 100, 10)
KBar_df['MOM'] = Calculate_MOM(KBar_df, mom_period)

#%%
#####  (xiii) TRIX - 三重平滑移動平均動能
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Calculate_TRIX(df, period=15):
    ema1 = df['close'].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    trix = 100 * (ema3 - ema3.shift()) / ema3.shift()
    return trix

with st.expander("設定 TRIX（三重 EMA 動能）參數"):
    trix_period = st.slider("TRIX 計算週期", 1, 100, 15)
KBar_df['TRIX'] = Calculate_TRIX(KBar_df, trix_period)

#%%
#####  (xiv) PSAR - 停損轉向指標（Parabolic SAR）
@st.cache_data(ttl=3600)
def Calculate_PSAR(df, af_start=0.02, af_step=0.02, af_max=0.2):
    high = df['high'].values
    low  = df['low'].values
    close= df['close'].values
    psar = np.zeros_like(close)
    psar[:] = np.nan
    trend = 1  # 1=向上, -1=向下
    af = af_start
    ep = high[0]  # extreme point
    psar[0] = low[0]

    for i in range(1, len(close)):
        prev_psar = psar[i-1]
        psar[i] = prev_psar + af * (ep - prev_psar)
        if trend == 1:
            psar[i] = min(psar[i], low[i-1], low[i])
            if high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)
            if close[i] < psar[i]:
                trend = -1
                ep = low[i]
                af = af_start
                psar[i] = ep
        else:
            psar[i] = max(psar[i], high[i-1], high[i])
            if low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)
            if close[i] > psar[i]:
                trend = 1
                ep = high[i]
                af = af_start
                psar[i] = ep
    return pd.Series(psar, index=df.index)



# ####### (5) 將 Dataframe 欄位名稱轉換(第一個字母大寫)  ####### 
# KBar_df_original = KBar_df
# KBar_df.columns = [ i[0].upper()+i[1:] for i in KBar_df.columns ]



#%%
####### (5) 畫圖 #######
st.subheader("技術指標視覺化圖形")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
#from plotly.offline import plot
# import plotly.offline as pyoff
if 'PSAR' not in KBar_df.columns:
    KBar_df['PSAR'] = Calculate_PSAR(KBar_df)

#%% 回測函式

# RSI逆勢策略回测函数
def back_test_rsi_strategy(record_obj, KBar_df, MoveStopLoss, RSIPeriod, OverSold, OverBought, Order_Quantity):
    # 计算RSI
    KBar_df['RSI'] = Calculate_RSI(KBar_df, period=RSIPeriod)
    
    # 寻找最后NAN值的位置
    last_nan_index = KBar_df['RSI'].last_valid_index() or 0
    
    # 回测逻辑
    StopLossPoint = 0
    for i in range(last_nan_index + 1, len(KBar_df) - 1):
        # 进场: 如果无未平仓部位 
        if record_obj.GetOpenInterest() == 0:
            # 多单进場: RSI低于超卖线
            if KBar_df['RSI'][i] < OverSold:
                record_obj.Order('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue
            # 空单进場: RSI高于超买线
            if KBar_df['RSI'][i] > OverBought:
                record_obj.Order('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        
        # 多单出场: 如果有多单部位   
        elif record_obj.GetOpenInterest() > 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_df['close'][i] - MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] < StopLossPoint:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
            # RSI超买出场
            if KBar_df['RSI'][i] > OverBought:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
        
        # 空单出场: 如果有空单部位
        elif record_obj.GetOpenInterest() < 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], -record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_df['close'][i] + MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] > StopLossPoint:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
            # RSI超卖出场
            if KBar_df['RSI'][i] < OverSold:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
    
    return record_obj

# 布林通道策略回测函数
def back_test_bb_strategy(record_obj, KBar_df, MoveStopLoss, BBPeriod, NumStdDev, Order_Quantity):
    # 计算布林通道
    KBar_df = Calculate_Bollinger_Bands(KBar_df, period=BBPeriod, num_std_dev=NumStdDev)
    
    # 寻找最后NAN值的位置
    last_nan_index = KBar_df['SMA'].last_valid_index() or 0
    
    # 回测逻辑
    StopLossPoint = 0
    for i in range(last_nan_index + 1, len(KBar_df) - 1):
        # 进场: 如果无未平仓部位 
        if record_obj.GetOpenInterest() == 0:
            # 多单进場: 价格触及下轨
            if KBar_df['close'][i] < KBar_df['Lower_Band'][i]:
                record_obj.Order('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue
            # 空单进場: 价格触及上轨
            if KBar_df['close'][i] > KBar_df['Upper_Band'][i]:
                record_obj.Order('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        
        # 多单出场: 如果有多单部位   
        elif record_obj.GetOpenInterest() > 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], record_obj.GetOpenInterest())
                continue
            # 触及中轨出场
            if KBar_df['close'][i] > KBar_df['SMA'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_df['close'][i] - MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] < StopLossPoint:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
        
        # 空单出场: 如果有空单部位
        elif record_obj.GetOpenInterest() < 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], -record_obj.GetOpenInterest())
                continue
            # 触及中轨出场
            if KBar_df['close'][i] < KBar_df['SMA'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_df['close'][i] + MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] > StopLossPoint:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
    
    return record_obj

# MACD策略回测函数
def back_test_macd_strategy(record_obj, KBar_df, MoveStopLoss, FastPeriod, SlowPeriod, SignalPeriod, Order_Quantity):
    # 计算MACD
    KBar_df = Calculate_MACD(KBar_df, fast_period=FastPeriod, slow_period=SlowPeriod, signal_period=SignalPeriod)
    
    # 寻找最后NAN值的位置
    last_nan_index = KBar_df['MACD'].last_valid_index() or 0
    
    # 回测逻辑
    StopLossPoint = 0
    for i in range(last_nan_index + 1, len(KBar_df) - 1):
        # 进场: 如果无未平仓部位 
        if record_obj.GetOpenInterest() == 0:
            # 多单进場: MACD上穿信号线
            if KBar_df['MACD'][i-1] < KBar_df['Signal_Line'][i-1] and KBar_df['MACD'][i] > KBar_df['Signal_Line'][i]:
                record_obj.Order('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue
            # 空单进場: MACD下穿信号线
            if KBar_df['MACD'][i-1] > KBar_df['Signal_Line'][i-1] and KBar_df['MACD'][i] < KBar_df['Signal_Line'][i]:
                record_obj.Order('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        
        # 多单出场: 如果有多单部位   
        elif record_obj.GetOpenInterest() > 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_df['close'][i] - MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] < StopLossPoint:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
            # MACD下穿信号线出场
            if KBar_df['MACD'][i-1] > KBar_df['Signal_Line'][i-1] and KBar_df['MACD'][i] < KBar_df['Signal_Line'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
        
        # 空单出场: 如果有空单部位
        elif record_obj.GetOpenInterest() < 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], -record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_df['close'][i] + MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] > StopLossPoint:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
            # MACD上穿信号线出场
            if KBar_df['MACD'][i-1] < KBar_df['Signal_Line'][i-1] and KBar_df['MACD'][i] > KBar_df['Signal_Line'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
    
    return record_obj

# ATR策略回测函数
def back_test_atr_strategy(record_obj, KBar_df, MoveStopLoss, ATRPeriod, ATRMultiplier, Order_Quantity):
    # 计算ATR
    KBar_df['ATR'] = Calculate_ATR(KBar_df, period=ATRPeriod)
    
    # 寻找最后NAN值的位置
    last_nan_index = KBar_df['ATR'].last_valid_index() or 0
    
    # 回测逻辑
    StopLossPoint = 0
    for i in range(last_nan_index + 1, len(KBar_df) - 1):
        # 进场: 如果无未平仓部位 
        if record_obj.GetOpenInterest() == 0:
            # 使用简单趋势跟随策略
            if KBar_df['close'][i] > KBar_df['close'][i-5]:  # 5期上涨
                record_obj.Order('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice - (ATRMultiplier * KBar_df['ATR'][i])
                continue
            elif KBar_df['close'][i] < KBar_df['close'][i-5]:  # 5期下跌
                record_obj.Order('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice + (ATRMultiplier * KBar_df['ATR'][i])
                continue
        
        # 多单出场: 如果有多单部位   
        elif record_obj.GetOpenInterest() > 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], record_obj.GetOpenInterest())
                continue
            # 更新ATR停损点
            current_stop = KBar_df['close'][i] - (ATRMultiplier * KBar_df['ATR'][i])
            if current_stop > StopLossPoint:
                StopLossPoint = current_stop
            # 如果触及停损价
            if KBar_df['close'][i] < StopLossPoint:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
        
        # 空单出场: 如果有空单部位
        elif record_obj.GetOpenInterest() < 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], -record_obj.GetOpenInterest())
                continue
            # 更新ATR停损点
            current_stop = KBar_df['close'][i] + (ATRMultiplier * KBar_df['ATR'][i])
            if current_stop < StopLossPoint:
                StopLossPoint = current_stop
            # 如果触及停损价
            if KBar_df['close'][i] > StopLossPoint:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
    
    return record_obj

# KD策略回测函数
def back_test_kd_strategy(record_obj, KBar_df, MoveStopLoss, KPeriod, DPeriod, OverSold, OverBought, Order_Quantity):
    # 计算KD
    KBar_df['K'], KBar_df['D'] = Calculate_KD(KBar_df, k_period=KPeriod, d_period=DPeriod)
    
    # 寻找最后NAN值的位置
    last_nan_index = KBar_df['K'].last_valid_index() or 0
    
    # 回测逻辑
    StopLossPoint = 0
    for i in range(last_nan_index + 1, len(KBar_df) - 1):
        # 进场: 如果无未平仓部位 
        if record_obj.GetOpenInterest() == 0:
            # 多单进場: K线上穿D线且在超卖区
            if KBar_df['K'][i-1] < KBar_df['D'][i-1] and KBar_df['K'][i] > KBar_df['D'][i] and KBar_df['K'][i] < OverSold:
                record_obj.Order('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue
            # 空单进場: K线下穿D线且在超买区
            if KBar_df['K'][i-1] > KBar_df['D'][i-1] and KBar_df['K'][i] < KBar_df['D'][i] and KBar_df['K'][i] > OverBought:
                record_obj.Order('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        
        # 多单出场: 如果有多单部位   
        elif record_obj.GetOpenInterest() > 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_df['close'][i] - MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] < StopLossPoint:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
            # K线下穿D线出场
            if KBar_df['K'][i-1] > KBar_df['D'][i-1] and KBar_df['K'][i] < KBar_df['D'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
        
        # 空单出场: 如果有空单部位
        elif record_obj.GetOpenInterest() < 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], -record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_df['close'][i] + MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] > StopLossPoint:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
            # K线上穿D线出场
            if KBar_df['K'][i-1] < KBar_df['D'][i-1] and KBar_df['K'][i] > KBar_df['D'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
    
    return record_obj

# 多策略组合回测函数
def back_test_multi_strategy(record_obj, KBar_df, MoveStopLoss, Order_Quantity, params):
    # 初始化信号计数器
    buy_signals = 0
    sell_signals = 0
    
    # 计算所有策略的信号
    # 移动平均线策略信号
    KBar_df['MA_long'] = Calculate_MA(KBar_df, period=params['LongMAPeriod'])
    KBar_df['MA_short'] = Calculate_MA(KBar_df, period=params['ShortMAPeriod'])
    
    # RSI策略信号
    KBar_df['RSI'] = Calculate_RSI(KBar_df, period=params['RSIPeriod'])
    
    # 布林通道策略信号
    KBar_df = Calculate_Bollinger_Bands(KBar_df, period=params['BBPeriod'], num_std_dev=params['NumStdDev'])
    
    # 寻找最后NAN值的位置
    last_nan_index = max(
        KBar_df['MA_long'].last_valid_index() or 0,
        KBar_df['RSI'].last_valid_index() or 0,
        KBar_df['SMA'].last_valid_index() or 0
    )
    
    # 回测逻辑
    StopLossPoint = 0
    for i in range(last_nan_index + 1, len(KBar_df) - 1):
        # 重置信号计数器
        buy_signals = 0
        sell_signals = 0
        
        # MA策略信号
        if KBar_df['MA_short'][i] > KBar_df['MA_long'][i]:
            buy_signals += params['Weight_MA']
        elif KBar_df['MA_short'][i] < KBar_df['MA_long'][i]:
            sell_signals += params['Weight_MA']
        
        # RSI策略信号
        if KBar_df['RSI'][i] < params['OverSold']:
            buy_signals += params['Weight_RSI']
        elif KBar_df['RSI'][i] > params['OverBought']:
            sell_signals += params['Weight_RSI']
        
        # 布林通道策略信号
        if KBar_df['close'][i] < KBar_df['Lower_Band'][i]:
            buy_signals += params['Weight_BB']
        elif KBar_df['close'][i] > KBar_df['Upper_Band'][i]:
            sell_signals += params['Weight_BB']
        
        # 进场: 如果无未平仓部位 
        if record_obj.GetOpenInterest() == 0:
            # 多单进場: 综合信号达到阈值
            if buy_signals > 0.7:  # 70%权重支持买入
                record_obj.Order('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue
            # 空单进場: 综合信号达到阈值
            if sell_signals > 0.7:  # 70%权重支持卖出
                record_obj.Order('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        
        # 多单出场: 如果有多单部位   
        elif record_obj.GetOpenInterest() > 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_df['close'][i] - MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] < StopLossPoint:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
            # 综合信号转为卖出
            if sell_signals > buy_signals and sell_signals > 0.5:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
        
        # 空单出场: 如果有空单部位
        elif record_obj.GetOpenInterest() < 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], -record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_df['close'][i] + MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] > StopLossPoint:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
            # 综合信号转为买入
            if buy_signals > sell_signals and buy_signals > 0.5:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
    
    return record_obj
#%%
###### K線圖, 移動平均線MA
with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.update_layout(yaxis=dict(fixedrange=False,  # 允許y軸縮放
                                  autorange=True    # 自動調整範圍
                                  ),
                       xaxis=dict(rangeslider=dict(visible=True)  # 保留下方的範圍滑桿
                                  )
                       )
    
    #### include candlestick with rangeselector
    fig1.add_trace(go.Candlestick(x=KBar_df['time'],
                    open=KBar_df['open'], high=KBar_df['high'],
                    low=KBar_df['low'], close=KBar_df['close'], name='K線'),
                   secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    #### include a go.Bar trace for volumes
    fig1.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['volume'], name='成交量', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
    fig1.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines',line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    fig1.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines',line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    
    fig1.layout.yaxis2.showgrid=True
    st.plotly_chart(fig1, use_container_width=True)


###### K線圖, RSI
with st.expander("長短 RSI"):
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.update_layout(yaxis=dict(fixedrange=False,  # 允許y軸縮放
                                  autorange=True    # 自動調整範圍
                                  ),
                       xaxis=dict(rangeslider=dict(visible=True)  # 保留下方的範圍滑桿
                                  )
                       )
    #### include candlestick with rangeselector
    # fig2.add_trace(go.Candlestick(x=KBar_df['Time'],
    #                 open=KBar_df['Open'], high=KBar_df['High'],
    #                 low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
    #                secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    fig2.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines',line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'), 
                  secondary_y=False)
    fig2.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines',line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'), 
                  secondary_y=False)
    
    fig2.layout.yaxis2.showgrid=True
    st.plotly_chart(fig2, use_container_width=True)
    

###### K線圖, Bollinger Band    
with st.expander("K線圖,布林通道"):
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.update_layout(yaxis=dict(fixedrange=False,  # 允許y軸縮放
                                  autorange=True    # 自動調整範圍
                                  ),
                       xaxis=dict(rangeslider=dict(visible=True)  # 保留下方的範圍滑桿
                                  )
                       )
    fig3.add_trace(go.Candlestick(x=KBar_df['time'],
                    open=KBar_df['open'], high=KBar_df['high'],
                    low=KBar_df['low'], close=KBar_df['close'], name='K線'),
                    secondary_y=True)    
    fig3.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_BB+1:], y=KBar_df['SMA'][last_nan_index_BB+1:], mode='lines',line=dict(color='black', width=2), name='布林通道中軌道'), 
                  secondary_y=False)
    fig3.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_BB+1:], y=KBar_df['Upper_Band'][last_nan_index_BB+1:], mode='lines',line=dict(color='red', width=2), name='布林通道上軌道'), 
                  secondary_y=False)
    fig3.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_BB+1:], y=KBar_df['Lower_Band'][last_nan_index_BB+1:], mode='lines',line=dict(color='blue', width=2), name='布林通道下軌道'), 
                  secondary_y=False)
    
    fig3.layout.yaxis2.showgrid=True

    st.plotly_chart(fig3, use_container_width=True)



###### MACD
with st.expander("MACD(異同移動平均線)"):
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.update_layout(yaxis=dict(fixedrange=False,  # 允許y軸縮放
                                  autorange=True    # 自動調整範圍
                                  ),
                       xaxis=dict(rangeslider=dict(visible=True)  # 保留下方的範圍滑桿
                                  )
                       )
    
    # #### include candlestick with rangeselector
    # fig4.add_trace(go.Candlestick(x=KBar_df['Time'],
    #                 open=KBar_df['Open'], high=KBar_df['High'],
    #                 low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
    #                secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    #### include a go.Bar trace for volumes
    fig4.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['MACD_Histogram'], name='MACD Histogram', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
    fig4.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MACD+1:], y=KBar_df['Signal_Line'][last_nan_index_MACD+1:], mode='lines',line=dict(color='orange', width=2), name='訊號線(DEA)'), 
                  secondary_y=True)
    fig4.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MACD+1:], y=KBar_df['MACD'][last_nan_index_MACD+1:], mode='lines',line=dict(color='pink', width=2), name='DIF'), 
                  secondary_y=True)
    
    fig4.layout.yaxis2.showgrid=True
    st.plotly_chart(fig4, use_container_width=True)

with st.expander("ATR（波動率指標）"):
    fig5 = make_subplots(specs=[[{"secondary_y": True}]])
    fig5.update_layout(
        yaxis=dict(fixedrange=False, autorange=True),
        xaxis=dict(rangeslider=dict(visible=True))
    )
    fig5.add_trace(go.Scatter(
        x=KBar_df['time'][last_nan_index_ATR+1:], 
        y=KBar_df['ATR'][last_nan_index_ATR+1:], 
        mode='lines', line=dict(color='purple', width=2), name='ATR'
    ), secondary_y=False)
    st.plotly_chart(fig5, use_container_width=True)

with st.expander("OBV（量價關係指標）"):
    fig6 = make_subplots(specs=[[{"secondary_y": True}]])
    fig6.update_layout(
        yaxis=dict(fixedrange=False, autorange=True),
        xaxis=dict(rangeslider=dict(visible=True))
    )
    fig6.add_trace(go.Scatter(
        x=KBar_df['time'], 
        y=KBar_df['OBV'], 
        mode='lines', line=dict(color='green', width=2), name='OBV'
    ), secondary_y=False)
    st.plotly_chart(fig6, use_container_width=True)

# 計算各指標的最後 NaN 索引位置
last_nan_index_CCI   = KBar_df['CCI'][::-1].index[KBar_df['CCI'][::-1].apply(pd.isna)][0]
last_nan_index_KD    = KBar_df['K'][::-1].index[KBar_df['K'][::-1].apply(pd.isna)][0]
last_nan_index_WILLR = KBar_df['WILLR'][::-1].index[KBar_df['WILLR'][::-1].apply(pd.isna)][0]
last_nan_index_MFI   = KBar_df['MFI'][::-1].index[KBar_df['MFI'][::-1].apply(pd.isna)][0]
last_nan_index_ROC   = KBar_df['ROC'][::-1].index[KBar_df['ROC'][::-1].apply(pd.isna)][0]
last_nan_index_MOM   = KBar_df['MOM'][::-1].index[KBar_df['MOM'][::-1].apply(pd.isna)][0]
last_nan_index_TRIX  = KBar_df['TRIX'][::-1].index[KBar_df['TRIX'][::-1].apply(pd.isna)][0]
# PSAR 通常不產生 NaN，可直接從頭繪製

# CCI
with st.expander("CCI - 商品通道指標"):
    fig6 = make_subplots(specs=[[{"secondary_y": False}]])
    fig6.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig6.add_trace(
        go.Scatter(
            x=KBar_df['time'][last_nan_index_CCI+1:],
            y=KBar_df['CCI'][last_nan_index_CCI+1:],
            mode='lines',
            name=f'CCI({cci_period})'
        )
    )
    st.plotly_chart(fig6, use_container_width=True)

# KD 隨機震盪指標
with st.expander("KD 隨機震盪指標"):
    fig7 = make_subplots(specs=[[{"secondary_y": False}]])
    fig7.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig7.add_trace(
        go.Scatter(
            x=KBar_df['time'][last_nan_index_KD+1:],
            y=KBar_df['K'][last_nan_index_KD+1:],
            mode='lines',
            name=f'K({k_period})'
        )
    )
    fig7.add_trace(
        go.Scatter(
            x=KBar_df['time'][last_nan_index_KD+1:],
            y=KBar_df['D'][last_nan_index_KD+1:],
            mode='lines',
            name=f'D({d_period})'
        )
    )
    st.plotly_chart(fig7, use_container_width=True)

# WILLR 威廉指標
with st.expander("WILLR - 威廉指標"):
    fig8 = make_subplots(specs=[[{"secondary_y": False}]])
    fig8.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig8.add_trace(
        go.Scatter(
            x=KBar_df['time'][last_nan_index_WILLR+1:],
            y=KBar_df['WILLR'][last_nan_index_WILLR+1:],
            mode='lines',
            name=f'WILLR({willr_period})'
        )
    )
    st.plotly_chart(fig8, use_container_width=True)

# MFI 資金流量指標
with st.expander("MFI - 資金流量指標"):
    fig9 = make_subplots(specs=[[{"secondary_y": False}]])
    fig9.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig9.add_trace(
        go.Scatter(
            x=KBar_df['time'][last_nan_index_MFI+1:],
            y=KBar_df['MFI'][last_nan_index_MFI+1:],
            mode='lines',
            name=f'MFI({mfi_period})'
        )
    )
    st.plotly_chart(fig9, use_container_width=True)

# ROC 價格變動率指標
with st.expander("ROC - 價格變動率指標"):
    fig10 = make_subplots(specs=[[{"secondary_y": False}]])
    fig10.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig10.add_trace(
        go.Scatter(
            x=KBar_df['time'][last_nan_index_ROC+1:],
            y=KBar_df['ROC'][last_nan_index_ROC+1:],
            mode='lines',
            name=f'ROC({roc_period})'
        )
    )
    st.plotly_chart(fig10, use_container_width=True)

# MOM 動能指標
with st.expander("MOM - 動能指標"):
    fig11 = make_subplots(specs=[[{"secondary_y": False}]])
    fig11.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig11.add_trace(
        go.Scatter(
            x=KBar_df['time'][last_nan_index_MOM+1:],
            y=KBar_df['MOM'][last_nan_index_MOM+1:],
            mode='lines',
            name=f'MOM({mom_period})'
        )
    )
    st.plotly_chart(fig11, use_container_width=True)

# TRIX 三重平滑移動平均動能
with st.expander("TRIX - 三重平滑移動平均動能"):
    fig12 = make_subplots(specs=[[{"secondary_y": False}]])
    fig12.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig12.add_trace(
        go.Scatter(
            x=KBar_df['time'][last_nan_index_TRIX+1:],
            y=KBar_df['TRIX'][last_nan_index_TRIX+1:],
            mode='lines',
            name=f'TRIX({trix_period})'
        )
    )
    st.plotly_chart(fig12, use_container_width=True)

# PSAR 停損轉向指標
with st.expander("PSAR - 停損轉向指標"):
    fig13 = make_subplots(specs=[[{"secondary_y": True}]])
    fig13.update_layout(
        yaxis=dict(fixedrange=False, autorange=True),
        xaxis=dict(rangeslider=dict(visible=True))
    )
    # 繪製 K 線
    fig13.add_trace(
        go.Candlestick(
            x=KBar_df['time'], open=KBar_df['open'], high=KBar_df['high'],
            low=KBar_df['low'], close=KBar_df['close'], name='K線'
        ), secondary_y=True
    )
    # 繪製 PSAR 點位
    fig13.add_trace(
        go.Scatter(
            x=KBar_df['time'], y=KBar_df['PSAR'], mode='markers',
            marker=dict(symbol='circle-open', size=6), name='PSAR'
        ), secondary_y=True
    )
    st.plotly_chart(fig13, use_container_width=True)

#%% 新增回测函数和绩效分析
# 移动平均线策略回测
def back_test_ma_strategy(record_obj, KBar_df, MoveStopLoss, LongMAPeriod, ShortMAPeriod, Order_Quantity):
    # 计算移动平均线
    KBar_df['MA_long'] = Calculate_MA(KBar_df, period=LongMAPeriod)
    KBar_df['MA_short'] = Calculate_MA(KBar_df, period=ShortMAPeriod)
    
    # 寻找最后NAN值的位置
    last_nan_index = KBar_df['MA_long'].last_valid_index() or 0
    
    # 回测逻辑
    StopLossPoint = 0  # 初始化停损点
    for i in range(last_nan_index + 1, len(KBar_df) - 1):
        # 先判断long MA的上一笔值是否为空值
        if not np.isnan(KBar_df['MA_long'][i-1]):
            # 进场: 如果无未平仓部位 
            if record_obj.GetOpenInterest() == 0:
                # 多单进場: 黄金交叉: short MA 向上突破 long MA
                if KBar_df['MA_short'][i-1] <= KBar_df['MA_long'][i-1] and KBar_df['MA_short'][i] > KBar_df['MA_long'][i]:
                    record_obj.Order('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                    OrderPrice = KBar_df['open'][i+1]
                    StopLossPoint = OrderPrice - MoveStopLoss
                    continue
                # 空单进場: 死亡交叉: short MA 向下突破 long MA
                if KBar_df['MA_short'][i-1] >= KBar_df['MA_long'][i-1] and KBar_df['MA_short'][i] < KBar_df['MA_long'][i]:
                    record_obj.Order('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                    OrderPrice = KBar_df['open'][i+1]
                    StopLossPoint = OrderPrice + MoveStopLoss
                    continue
            
            # 多单出场: 如果有多单部位   
            elif record_obj.GetOpenInterest() > 0:
                # 结算平仓(期货才使用, 股票除非是下市柜)
                if KBar_df['product'][i+1] != KBar_df['product'][i]:
                    record_obj.Cover('Sell', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], record_obj.GetOpenInterest())
                    continue
                # 逐笔更新移动停损价位
                if KBar_df['close'][i] - MoveStopLoss > StopLossPoint:
                    StopLossPoint = KBar_df['close'][i] - MoveStopLoss
                # 如果上一根K的收盘价触及停损价位，则在最新时间出场
                elif KBar_df['close'][i] < StopLossPoint:
                    record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                    continue
            
            # 空单出场: 如果有空单部位
            elif record_obj.GetOpenInterest() < 0:
                # 结算平仓(期货才使用, 股票除非是下市柜)
                if KBar_df['product'][i+1] != KBar_df['product'][i]:
                    record_obj.Cover('Buy', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], -record_obj.GetOpenInterest())
                    continue
                # 逐笔更新移动停损价位
                if KBar_df['close'][i] + MoveStopLoss < StopLossPoint:
                    StopLossPoint = KBar_df['close'][i] + MoveStopLoss
                # 如果上一根K的收盘价触及停损价位，则在最新时间出场
                elif KBar_df['close'][i] > StopLossPoint:
                    record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                    continue
    
    return record_obj

# RSI策略回测（逆勢策略）
def back_test_rsi_strategy(record_obj, KBar_df, MoveStopLoss, RSIPeriod, OverSold, OverBought, Order_Quantity):
    # 计算RSI
    KBar_df['RSI'] = Calculate_RSI(KBar_df, period=RSIPeriod)
    
    # 寻找最后NAN值的位置
    last_nan_index = KBar_df['RSI'].last_valid_index() or 0
    
    # 回测逻辑
    StopLossPoint = 0
    for i in range(last_nan_index + 1, len(KBar_df) - 1):
        # 进场: 如果无未平仓部位 
        if record_obj.GetOpenInterest() == 0:
            # 多单进場: RSI低于超卖线
            if KBar_df['RSI'][i] < OverSold:
                record_obj.Order('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue
            # 空单进場: RSI高于超买线
            if KBar_df['RSI'][i] > OverBought:
                record_obj.Order('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        
        # 多单出场: 如果有多单部位   
        elif record_obj.GetOpenInterest() > 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_df['close'][i] - MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] < StopLossPoint:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
            # RSI超买出场
            if KBar_df['RSI'][i] > OverBought:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
        
        # 空单出场: 如果有空单部位
        elif record_obj.GetOpenInterest() < 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], -record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_df['close'][i] + MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] > StopLossPoint:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
            # RSI超卖出场
            if KBar_df['RSI'][i] < OverSold:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
    
    return record_obj

# 布林通道策略回测
def back_test_bb_strategy(record_obj, KBar_df, MoveStopLoss, BBPeriod, NumStdDev, Order_Quantity):
    # 计算布林通道
    KBar_df = Calculate_Bollinger_Bands(KBar_df, period=BBPeriod, num_std_dev=NumStdDev)
    
    # 寻找最后NAN值的位置
    last_nan_index = KBar_df['SMA'].last_valid_index() or 0
    
    # 回测逻辑
    StopLossPoint = 0
    for i in range(last_nan_index + 1, len(KBar_df) - 1):
        # 进场: 如果无未平仓部位 
        if record_obj.GetOpenInterest() == 0:
            # 多单进場: 价格触及下轨
            if KBar_df['close'][i] < KBar_df['Lower_Band'][i]:
                record_obj.Order('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue
            # 空单进場: 价格触及上轨
            if KBar_df['close'][i] > KBar_df['Upper_Band'][i]:
                record_obj.Order('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        
        # 多单出场: 如果有多单部位   
        elif record_obj.GetOpenInterest() > 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], record_obj.GetOpenInterest())
                continue
            # 触及中轨出场
            if KBar_df['close'][i] > KBar_df['SMA'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_df['close'][i] - MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] < StopLossPoint:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
        
        # 空单出场: 如果有空单部位
        elif record_obj.GetOpenInterest() < 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], -record_obj.GetOpenInterest())
                continue
            # 触及中轨出场
            if KBar_df['close'][i] < KBar_df['SMA'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_df['close'][i] + MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] > StopLossPoint:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
    
    return record_obj

# MACD策略回测
def back_test_macd_strategy(record_obj, KBar_df, MoveStopLoss, FastPeriod, SlowPeriod, SignalPeriod, Order_Quantity):
    # 计算MACD
    KBar_df = Calculate_MACD(KBar_df, fast_period=FastPeriod, slow_period=SlowPeriod, signal_period=SignalPeriod)
    
    # 寻找最后NAN值的位置
    last_nan_index = KBar_df['MACD'].last_valid_index() or 0
    
    # 回测逻辑
    StopLossPoint = 0
    for i in range(last_nan_index + 1, len(KBar_df) - 1):
        # 进场: 如果无未平仓部位 
        if record_obj.GetOpenInterest() == 0:
            # 多单进場: MACD上穿信号线
            if KBar_df['MACD'][i-1] < KBar_df['Signal_Line'][i-1] and KBar_df['MACD'][i] > KBar_df['Signal_Line'][i]:
                record_obj.Order('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue
            # 空单进場: MACD下穿信号线
            if KBar_df['MACD'][i-1] > KBar_df['Signal_Line'][i-1] and KBar_df['MACD'][i] < KBar_df['Signal_Line'][i]:
                record_obj.Order('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        
        # 多单出场: 如果有多单部位   
        elif record_obj.GetOpenInterest() > 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_df['close'][i] - MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] < StopLossPoint:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
            # MACD下穿信号线出场
            if KBar_df['MACD'][i-1] > KBar_df['Signal_Line'][i-1] and KBar_df['MACD'][i] < KBar_df['Signal_Line'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
        
        # 空单出场: 如果有空单部位
        elif record_obj.GetOpenInterest() < 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], -record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_df['close'][i] + MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] > StopLossPoint:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
            # MACD上穿信号线出场
            if KBar_df['MACD'][i-1] < KBar_df['Signal_Line'][i-1] and KBar_df['MACD'][i] > KBar_df['Signal_Line'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
    
    return record_obj

# ATR策略回测（波动性止损）
def back_test_atr_strategy(record_obj, KBar_df, MoveStopLoss, ATRPeriod, ATRMultiplier, Order_Quantity):
    # 计算ATR
    KBar_df['ATR'] = Calculate_ATR(KBar_df, period=ATRPeriod)
    
    # 寻找最后NAN值的位置
    last_nan_index = KBar_df['ATR'].last_valid_index() or 0
    
    # 回测逻辑
    StopLossPoint = 0
    for i in range(last_nan_index + 1, len(KBar_df) - 1):
        # 进场: 如果无未平仓部位 
        if record_obj.GetOpenInterest() == 0:
            # 使用简单趋势跟随策略
            if KBar_df['close'][i] > KBar_df['close'][i-5]:  # 5期上涨
                record_obj.Order('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice - (ATRMultiplier * KBar_df['ATR'][i])
                continue
            elif KBar_df['close'][i] < KBar_df['close'][i-5]:  # 5期下跌
                record_obj.Order('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice + (ATRMultiplier * KBar_df['ATR'][i])
                continue
        
        # 多单出场: 如果有多单部位   
        elif record_obj.GetOpenInterest() > 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], record_obj.GetOpenInterest())
                continue
            # 更新ATR停损点
            current_stop = KBar_df['close'][i] - (ATRMultiplier * KBar_df['ATR'][i])
            if current_stop > StopLossPoint:
                StopLossPoint = current_stop
            # 如果触及停损价
            if KBar_df['close'][i] < StopLossPoint:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
        
        # 空单出场: 如果有空单部位
        elif record_obj.GetOpenInterest() < 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], -record_obj.GetOpenInterest())
                continue
            # 更新ATR停损点
            current_stop = KBar_df['close'][i] + (ATRMultiplier * KBar_df['ATR'][i])
            if current_stop < StopLossPoint:
                StopLossPoint = current_stop
            # 如果触及停损价
            if KBar_df['close'][i] > StopLossPoint:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
    
    return record_obj

# KD策略回测
def back_test_kd_strategy(record_obj, KBar_df, MoveStopLoss, KPeriod, DPeriod, OverSold, OverBought, Order_Quantity):
    # 计算KD
    KBar_df['K'], KBar_df['D'] = Calculate_KD(KBar_df, k_period=KPeriod, d_period=DPeriod)
    
    # 寻找最后NAN值的位置
    last_nan_index = KBar_df['K'].last_valid_index() or 0
    
    # 回测逻辑
    StopLossPoint = 0
    for i in range(last_nan_index + 1, len(KBar_df) - 1):
        # 进场: 如果无未平仓部位 
        if record_obj.GetOpenInterest() == 0:
            # 多单进場: K线上穿D线且在超卖区
            if KBar_df['K'][i-1] < KBar_df['D'][i-1] and KBar_df['K'][i] > KBar_df['D'][i] and KBar_df['K'][i] < OverSold:
                record_obj.Order('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue
            # 空单进場: K线下穿D线且在超买区
            if KBar_df['K'][i-1] > KBar_df['D'][i-1] and KBar_df['K'][i] < KBar_df['D'][i] and KBar_df['K'][i] > OverBought:
                record_obj.Order('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        
        # 多单出场: 如果有多单部位   
        elif record_obj.GetOpenInterest() > 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_df['close'][i] - MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] < StopLossPoint:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
            # K线下穿D线出场
            if KBar_df['K'][i-1] > KBar_df['D'][i-1] and KBar_df['K'][i] < KBar_df['D'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
        
        # 空单出场: 如果有空单部位
        elif record_obj.GetOpenInterest() < 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], -record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_df['close'][i] + MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] > StopLossPoint:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
            # K线上穿D线出场
            if KBar_df['K'][i-1] < KBar_df['D'][i-1] and KBar_df['K'][i] > KBar_df['D'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
    
    return record_obj

# 多策略组合回测
def back_test_multi_strategy(record_obj, KBar_df, MoveStopLoss, Order_Quantity, params):
    # 初始化信号计数器
    buy_signals = 0
    sell_signals = 0
    
    # 计算所有策略的信号
    # 移动平均线策略信号
    KBar_df['MA_long'] = Calculate_MA(KBar_df, period=params['LongMAPeriod'])
    KBar_df['MA_short'] = Calculate_MA(KBar_df, period=params['ShortMAPeriod'])
    
    # RSI策略信号
    KBar_df['RSI'] = Calculate_RSI(KBar_df, period=params['RSIPeriod'])
    
    # 布林通道策略信号
    KBar_df = Calculate_Bollinger_Bands(KBar_df, period=params['BBPeriod'], num_std_dev=params['NumStdDev'])
    
    # 寻找最后NAN值的位置
    last_nan_index = max(
        KBar_df['MA_long'].last_valid_index() or 0,
        KBar_df['RSI'].last_valid_index() or 0,
        KBar_df['SMA'].last_valid_index() or 0
    )
    
    # 回测逻辑
    StopLossPoint = 0
    for i in range(last_nan_index + 1, len(KBar_df) - 1):
        # 重置信号计数器
        buy_signals = 0
        sell_signals = 0
        
        # MA策略信号
        if KBar_df['MA_short'][i] > KBar_df['MA_long'][i]:
            buy_signals += params['Weight_MA']
        elif KBar_df['MA_short'][i] < KBar_df['MA_long'][i]:
            sell_signals += params['Weight_MA']
        
        # RSI策略信号
        if KBar_df['RSI'][i] < params['OverSold']:
            buy_signals += params['Weight_RSI']
        elif KBar_df['RSI'][i] > params['OverBought']:
            sell_signals += params['Weight_RSI']
        
        # 布林通道策略信号
        if KBar_df['close'][i] < KBar_df['Lower_Band'][i]:
            buy_signals += params['Weight_BB']
        elif KBar_df['close'][i] > KBar_df['Upper_Band'][i]:
            sell_signals += params['Weight_BB']
        
        # 进场: 如果无未平仓部位 
        if record_obj.GetOpenInterest() == 0:
            # 多单进場: 综合信号达到阈值
            if buy_signals > 0.7:  # 70%权重支持买入
                record_obj.Order('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue
            # 空单进場: 综合信号达到阈值
            if sell_signals > 0.7:  # 70%权重支持卖出
                record_obj.Order('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], Order_Quantity)
                OrderPrice = KBar_df['open'][i+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        
        # 多单出场: 如果有多单部位   
        elif record_obj.GetOpenInterest() > 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Sell', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_df['close'][i] - MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] < StopLossPoint:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
            # 综合信号转为卖出
            if sell_signals > buy_signals and sell_signals > 0.5:
                record_obj.Cover('Sell', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], record_obj.GetOpenInterest())
                continue
        
        # 空单出场: 如果有空单部位
        elif record_obj.GetOpenInterest() < 0:
            # 结算平仓(期货才使用)
            if KBar_df['product'][i+1] != KBar_df['product'][i]:
                record_obj.Cover('Buy', KBar_df['product'][i], KBar_df['time'][i], KBar_df['close'][i], -record_obj.GetOpenInterest())
                continue
            # 逐笔更新移动停损价
            if KBar_df['close'][i] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_df['close'][i] + MoveStopLoss
            # 如果触及停损价
            elif KBar_df['close'][i] > StopLossPoint:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
            # 综合信号转为买入
            if buy_signals > sell_signals and buy_signals > 0.5:
                record_obj.Cover('Buy', KBar_df['product'][i+1], KBar_df['time'][i+1], KBar_df['open'][i+1], -record_obj.GetOpenInterest())
                continue
    
    return record_obj

# 最佳化参数搜索函数
def optimize_ma_strategy(KBar_df, long_range, short_range, max_iterations=50):
    best_performance = -float('inf')
    best_params = (0, 0)
    
    # 生成参数组合
    long_options = np.linspace(long_range[0], long_range[1], 10, dtype=int)
    short_options = np.linspace(short_range[0], short_range[1], 10, dtype=int)
    param_combinations = list(product(long_options, short_options))
    
    # 随机抽样减少计算量
    if len(param_combinations) > max_iterations:
        param_combinations = random.sample(param_combinations, max_iterations)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (long_period, short_period) in enumerate(param_combinations):
        if short_period >= long_period:
            continue
            
        # 建立临时记录物件
        temp_record = Record(isFuture=False)  # 简化的Record初始化
        
        # 执行回测
        temp_record = back_test_ma_strategy(
            temp_record, KBar_df.copy(),
            MoveStopLoss=30,  # 使用固定停损
            LongMAPeriod=long_period,
            ShortMAPeriod=short_period,
            Order_Quantity=1
        )
        
        # 计算绩效指标
        if len(temp_record.Profit) > 0:
            total_profit = sum(temp_record.Profit)
            win_rate = len([p for p in temp_record.Profit if p > 0]) / len(temp_record.Profit)
            performance = total_profit * win_rate  # 自定义绩效指标
            
            if performance > best_performance:
                best_performance = performance
                best_params = (long_period, short_period)
        
        # 更新进度条
        progress = (i + 1) / len(param_combinations)
        progress_bar.progress(progress)
        status_text.text(f"测试参数组合: {i+1}/{len(param_combinations)} | 当前最佳: {best_params} 绩效: {best_performance:.2f}")
    
    progress_bar.empty()
    status_text.empty()
    
    return best_params

# RSI策略最佳化參數搜索
def optimize_rsi_strategy(KBar_df, rsi_range, over_sold_range, over_bought_range, max_iterations=50):
    best_performance = -float('inf')
    best_params = (0, 0, 0)
    
    # 生成參數組合
    rsi_options = np.linspace(rsi_range[0], rsi_range[1], 10, dtype=int)
    os_options = np.linspace(over_sold_range[0], over_sold_range[1], 10, dtype=int)
    ob_options = np.linspace(over_bought_range[0], over_bought_range[1], 10, dtype=int)
    param_combinations = list(product(rsi_options, os_options, ob_options))
    
    # 隨機抽樣減少計算量
    if len(param_combinations) > max_iterations:
        param_combinations = random.sample(param_combinations, max_iterations)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (rsi_period, over_sold, over_bought) in enumerate(param_combinations):
        if over_sold >= over_bought:
            continue
            
        # 建立臨時記錄物件
        temp_record = Record(isFuture=False)
        
        # 執行回測
        temp_record = back_test_rsi_strategy(
            temp_record, KBar_df.copy(),
            MoveStopLoss=30,
            RSIPeriod=rsi_period,
            OverSold=over_sold,
            OverBought=over_bought,
            Order_Quantity=1
        )
        
        # 計算績效指標
        if len(temp_record.Profit) > 0:
            total_profit = sum(temp_record.Profit)
            win_rate = len([p for p in temp_record.Profit if p > 0]) / len(temp_record.Profit)
            performance = total_profit * win_rate
            
            if performance > best_performance:
                best_performance = performance
                best_params = (rsi_period, over_sold, over_bought)
        
        # 更新進度條
        progress = (i + 1) / len(param_combinations)
        progress_bar.progress(progress)
        status_text.text(f"測試參數組合: {i+1}/{len(param_combinations)} | 當前最佳: {best_params} 績效: {best_performance:.2f}")
    
    progress_bar.empty()
    status_text.empty()
    
    return best_params

# 布林通道策略最佳化參數搜索
def optimize_bb_strategy(KBar_df, bb_range, std_range, max_iterations=50):
    best_performance = -float('inf')
    best_params = (0, 0)
    
    # 生成參數組合
    bb_options = np.linspace(bb_range[0], bb_range[1], 10, dtype=int)
    std_options = np.linspace(std_range[0], std_range[1], 10)
    param_combinations = list(product(bb_options, std_options))
    
    # 隨機抽樣減少計算量
    if len(param_combinations) > max_iterations:
        param_combinations = random.sample(param_combinations, max_iterations)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (bb_period, std_dev) in enumerate(param_combinations):
        # 建立臨時記錄物件
        temp_record = Record(isFuture=False)
        
        # 執行回測
        temp_record = back_test_bb_strategy(
            temp_record, KBar_df.copy(),
            MoveStopLoss=30,
            BBPeriod=bb_period,
            NumStdDev=std_dev,
            Order_Quantity=1
        )
        
        # 計算績效指標
        if len(temp_record.Profit) > 0:
            total_profit = sum(temp_record.Profit)
            win_rate = len([p for p in temp_record.Profit if p > 0]) / len(temp_record.Profit)
            performance = total_profit * win_rate
            
            if performance > best_performance:
                best_performance = performance
                best_params = (bb_period, std_dev)
        
        # 更新進度條
        progress = (i + 1) / len(param_combinations)
        progress_bar.progress(progress)
        status_text.text(f"測試參數組合: {i+1}/{len(param_combinations)} | 當前最佳: {best_params} 績效: {best_performance:.2f}")
    
    progress_bar.empty()
    status_text.empty()
    
    return best_params

# MACD策略最佳化參數搜索
def optimize_macd_strategy(KBar_df, fast_range, slow_range, signal_range, max_iterations=50):
    best_performance = -float('inf')
    best_params = (0, 0, 0)
    
    # 生成參數組合
    fast_options = np.linspace(fast_range[0], fast_range[1], 10, dtype=int)
    slow_options = np.linspace(slow_range[0], slow_range[1], 10, dtype=int)
    signal_options = np.linspace(signal_range[0], signal_range[1], 10, dtype=int)
    param_combinations = list(product(fast_options, slow_options, signal_options))
    
    # 過濾無效組合
    param_combinations = [p for p in param_combinations if p[0] < p[1]]
    
    # 隨機抽樣減少計算量
    if len(param_combinations) > max_iterations:
        param_combinations = random.sample(param_combinations, max_iterations)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (fast, slow, signal) in enumerate(param_combinations):
        # 建立臨時記錄物件
        temp_record = Record(isFuture=False)
        
        # 執行回測
        temp_record = back_test_macd_strategy(
            temp_record, KBar_df.copy(),
            MoveStopLoss=30,
            FastPeriod=fast,
            SlowPeriod=slow,
            SignalPeriod=signal,
            Order_Quantity=1
        )
        
        # 計算績效指標
        if len(temp_record.Profit) > 0:
            total_profit = sum(temp_record.Profit)
            win_rate = len([p for p in temp_record.Profit if p > 0]) / len(temp_record.Profit)
            performance = total_profit * win_rate
            
            if performance > best_performance:
                best_performance = performance
                best_params = (fast, slow, signal)
        
        # 更新進度條
        progress = (i + 1) / len(param_combinations)
        progress_bar.progress(progress)
        status_text.text(f"測試參數組合: {i+1}/{len(param_combinations)} | 當前最佳: {best_params} 績效: {best_performance:.2f}")
    
    progress_bar.empty()
    status_text.empty()
    
    return best_params

# ATR策略最佳化參數搜索
def optimize_atr_strategy(KBar_df, atr_range, multiplier_range, max_iterations=50):
    best_performance = -float('inf')
    best_params = (0, 0)
    
    # 生成參數組合
    atr_options = np.linspace(atr_range[0], atr_range[1], 10, dtype=int)
    multiplier_options = np.linspace(multiplier_range[0], multiplier_range[1], 10)
    param_combinations = list(product(atr_options, multiplier_options))
    
    # 隨機抽樣減少計算量
    if len(param_combinations) > max_iterations:
        param_combinations = random.sample(param_combinations, max_iterations)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (atr_period, multiplier) in enumerate(param_combinations):
        # 建立臨時記錄物件
        temp_record = Record(isFuture=False)
        
        # 執行回測
        temp_record = back_test_atr_strategy(
            temp_record, KBar_df.copy(),
            MoveStopLoss=30,
            ATRPeriod=atr_period,
            ATRMultiplier=multiplier,
            Order_Quantity=1
        )
        
        # 計算績效指標
        if len(temp_record.Profit) > 0:
            total_profit = sum(temp_record.Profit)
            win_rate = len([p for p in temp_record.Profit if p > 0]) / len(temp_record.Profit)
            performance = total_profit * win_rate
            
            if performance > best_performance:
                best_performance = performance
                best_params = (atr_period, multiplier)
        
        # 更新進度條
        progress = (i + 1) / len(param_combinations)
        progress_bar.progress(progress)
        status_text.text(f"測試參數組合: {i+1}/{len(param_combinations)} | 當前最佳: {best_params} 績效: {best_performance:.2f}")
    
    progress_bar.empty()
    status_text.empty()
    
    return best_params

# KD策略最佳化參數搜索
def optimize_kd_strategy(KBar_df, k_range, d_range, os_range, ob_range, max_iterations=50):
    best_performance = -float('inf')
    best_params = (0, 0, 0, 0)
    
    # 生成參數組合
    k_options = np.linspace(k_range[0], k_range[1], 10, dtype=int)
    d_options = np.linspace(d_range[0], d_range[1], 10, dtype=int)
    os_options = np.linspace(os_range[0], os_range[1], 10, dtype=int)
    ob_options = np.linspace(ob_range[0], ob_range[1], 10, dtype=int)
    param_combinations = list(product(k_options, d_options, os_options, ob_options))
    
    # 過濾無效組合
    param_combinations = [p for p in param_combinations if p[2] < p[3]]
    
    # 隨機抽樣減少計算量
    if len(param_combinations) > max_iterations:
        param_combinations = random.sample(param_combinations, max_iterations)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (k_period, d_period, over_sold, over_bought) in enumerate(param_combinations):
        # 建立臨時記錄物件
        temp_record = Record(isFuture=False)
        
        # 執行回測
        temp_record = back_test_kd_strategy(
            temp_record, KBar_df.copy(),
            MoveStopLoss=30,
            KPeriod=k_period,
            DPeriod=d_period,
            OverSold=over_sold,
            OverBought=over_bought,
            Order_Quantity=1
        )
        
        # 計算績效指標
        if len(temp_record.Profit) > 0:
            total_profit = sum(temp_record.Profit)
            win_rate = len([p for p in temp_record.Profit if p > 0]) / len(temp_record.Profit)
            performance = total_profit * win_rate
            
            if performance > best_performance:
                best_performance = performance
                best_params = (k_period, d_period, over_sold, over_bought)
        
        # 更新進度條
        progress = (i + 1) / len(param_combinations)
        progress_bar.progress(progress)
        status_text.text(f"測試參數組合: {i+1}/{len(param_combinations)} | 當前最佳: {best_params} 績效: {best_performance:.2f}")
    
    progress_bar.empty()
    status_text.empty()
    
    return best_params

# 绩效分析函数
def calculate_and_display_performance(OrderRecord, product, trade_records):
    if not trade_records:
        st.warning("没有交易记录，无法计算绩效")
        return
    
    # 基本绩效指标
    total_profit = sum(OrderRecord.Profit)
    num_trades = len(OrderRecord.Profit)
    win_trades = [p for p in OrderRecord.Profit if p > 0]
    loss_trades = [p for p in OrderRecord.Profit if p < 0]
    win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
    avg_profit = total_profit / num_trades if num_trades > 0 else 0
    avg_win = sum(win_trades) / len(win_trades) if win_trades else 0
    avg_loss = sum(loss_trades) / len(loss_trades) if loss_trades else 0
    max_drawdown = min(OrderRecord.Profit) if num_trades > 0 else 0
    
    # 进阶绩效指标
    profit_factor = abs(sum(win_trades)) / abs(sum(loss_trades)) if loss_trades else float('inf')
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
    
    # 计算夏普比率 (简化版)
    returns = np.array(OrderRecord.Profit)
    sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 else 0
    
    # 计算最大连续亏损
    max_consecutive_losses = 0
    current_losses = 0
    for p in OrderRecord.Profit:
        if p < 0:
            current_losses += 1
            if current_losses > max_consecutive_losses:
                max_consecutive_losses = current_losses
        else:
            current_losses = 0
    
    # 显示绩效指标
    st.subheader("交易绩效分析")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总交易次数", num_trades)
        st.metric("胜率", f"{win_rate*100:.2f}%")
        st.metric("平均每笔盈亏", f"{avg_profit:.2f}")
    
    with col2:
        st.metric("总盈亏", f"{total_profit:.2f}")
        st.metric("最大单笔亏损", f"{max_drawdown:.2f}")
        st.metric("盈亏因子", f"{profit_factor:.2f}")
    
    with col3:
        st.metric("夏普比率", f"{sharpe_ratio:.2f}")
        st.metric("最大连续亏损次数", max_consecutive_losses)
        st.metric("期望值", f"{expectancy:.2f}")
    
    # 绘制累计盈亏图
    if num_trades > 0:
        cumulative_profit = np.cumsum(OrderRecord.Profit)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cumulative_profit, label='累计盈亏')
        ax.set_title('累计盈亏曲线')
        ax.set_xlabel('交易次数')
        ax.set_ylabel('累计盈亏')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        
        # 绘制每日盈亏分布图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(OrderRecord.Profit, bins=20, alpha=0.7)
        ax.set_title('单笔交易盈亏分布')
        ax.set_xlabel('盈亏金额')
        ax.set_ylabel('交易次数')
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning("没有足够的交易数据绘制图表")

# 策略专属图表函数
###### 函數定義: 繪製K線圖加上MA以及下單點位
def ChartOrder_MA(Kbar_df, TR):
    # 計算移動平均線的最後一個NaN的位置
    last_nan_index_MA_long = Kbar_df['MA_long'].isna().sum()
    last_nan_index_MA_short = Kbar_df['MA_short'].isna().sum()
    last_nan_index_MA_trading = max(last_nan_index_MA_long, last_nan_index_MA_short)
    
    # 買(多)方下單點位紀錄
    BTR = [i for i in TR if i[0] == 'Buy' or i[0] == 'B']
    BuyOrderPoint_date = [] 
    BuyOrderPoint_price = []
    BuyCoverPoint_date = []
    BuyCoverPoint_price = []
    
    # 賣(空)方下單點位紀錄
    STR = [i for i in TR if i[0] == 'Sell' or i[0] == 'S']
    SellOrderPoint_date = []
    SellOrderPoint_price = []
    SellCoverPoint_date = []
    SellCoverPoint_price = []
    
    # 收集下單點位
    for i, row in Kbar_df.iterrows():
        date = row['time']
        low = row['low']
        high = row['high']
        
        # 買方進場
        if date in [i[2] for i in BTR]:
            BuyOrderPoint_date.append(date)
            BuyOrderPoint_price.append(low * 0.999)
        else:
            BuyOrderPoint_date.append(None)
            BuyOrderPoint_price.append(None)
            
        # 買方出場
        if date in [i[4] for i in BTR]:
            BuyCoverPoint_date.append(date)
            BuyCoverPoint_price.append(high * 1.001)
        else:
            BuyCoverPoint_date.append(None)
            BuyCoverPoint_price.append(None)
            
        # 賣方進場
        if date in [i[2] for i in STR]:
            SellOrderPoint_date.append(date)
            SellOrderPoint_price.append(high * 1.001)
        else:
            SellOrderPoint_date.append(None)
            SellOrderPoint_price.append(None)
            
        # 賣方出場
        if date in [i[4] for i in STR]:
            SellCoverPoint_date.append(date)
            SellCoverPoint_price.append(low * 0.999)
        else:
            SellCoverPoint_date.append(None)
            SellCoverPoint_price.append(None)
    
    # 繪製圖表
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 添加MA線
    fig.add_trace(go.Scatter(
        x=Kbar_df['time'][last_nan_index_MA_trading+1:], 
        y=Kbar_df['MA_long'][last_nan_index_MA_trading+1:], 
        mode='lines',
        line=dict(color='orange', width=2), 
        name=f'{LongMAPeriod}-根K棒移動平均線'
    ), secondary_y=True)
    
    fig.add_trace(go.Scatter(
        x=Kbar_df['time'][last_nan_index_MA_trading+1:], 
        y=Kbar_df['MA_short'][last_nan_index_MA_trading+1:], 
        mode='lines',
        line=dict(color='pink', width=2), 
        name=f'{ShortMAPeriod}-根K棒移動平均線'
    ), secondary_y=True)
    
    # 添加下單點位
    fig.add_trace(go.Scatter(
        x=BuyOrderPoint_date, 
        y=BuyOrderPoint_price, 
        mode='markers',
        marker=dict(color='red', symbol='triangle-up', size=10),
        name='作多進場點'
    ))
    
    fig.add_trace(go.Scatter(
        x=BuyCoverPoint_date, 
        y=BuyCoverPoint_price, 
        mode='markers',
        marker=dict(color='blue', symbol='triangle-down', size=10),
        name='作多出場點'
    ))
    
    fig.add_trace(go.Scatter(
        x=SellOrderPoint_date, 
        y=SellOrderPoint_price, 
        mode='markers',
        marker=dict(color='green', symbol='triangle-down', size=10),
        name='作空進場點'
    ))
    
    fig.add_trace(go.Scatter(
        x=SellCoverPoint_date, 
        y=SellCoverPoint_price, 
        mode='markers',
        marker=dict(color='black', symbol='triangle-up', size=10),
        name='作空出場點'
    ))
    
    # 設置圖表佈局
    fig.update_layout(
        title='移動平均線策略交易點位',
        xaxis_title='時間',
        yaxis_title='價格',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def ChartOrder_RSI(KBar_df, TR):
    # 绘制RSI策略专属图表
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.1, 
                       row_heights=[0.7, 0.3])
    
    # K线图
    fig.add_trace(go.Candlestick(x=KBar_df['time'],
                                open=KBar_df['open'],
                                high=KBar_df['high'],
                                low=KBar_df['low'],
                                close=KBar_df['close'],
                                name='K线'),
                row=1, col=1)
    
    # RSI图
    fig.add_trace(go.Scatter(x=KBar_df['time'], 
                            y=KBar_df['RSI'], 
                            mode='lines', 
                            name='RSI',
                            line=dict(color='purple', width=2)),
                row=2, col=1)
    
    # 超买超卖线
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # 下单点位标记...
    
    fig.update_layout(title='RSI策略交易图',
                      xaxis_rangeslider_visible=False,
                      height=800)
    st.plotly_chart(fig, use_container_width=True)

###### 函數定義: 繪製布林通道策略專屬圖表
def ChartOrder_BB(Kbar_df, TR):
    # 布林通道計算
    period = 20
    num_std_dev = 2
    Kbar_df['SMA'] = Kbar_df['close'].rolling(window=period).mean()
    Kbar_df['Upper_Band'] = Kbar_df['SMA'] + (Kbar_df['close'].rolling(window=period).std() * num_std_dev)
    Kbar_df['Lower_Band'] = Kbar_df['SMA'] - (Kbar_df['close'].rolling(window=period).std() * num_std_dev)
    
    # 尋找最後NAN值的位置
    last_nan_index = max(
        Kbar_df['SMA'].isna().sum(),
        Kbar_df['Upper_Band'].isna().sum(),
        Kbar_df['Lower_Band'].isna().sum()
    )
    
    # 繪製圖表
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 添加K線
    fig.add_trace(go.Candlestick(
        x=Kbar_df['time'],
        open=Kbar_df['open'],
        high=Kbar_df['high'],
        low=Kbar_df['low'],
        close=Kbar_df['close'],
        name='K線'
    ), secondary_y=True)
    
    # 添加布林通道
    fig.add_trace(go.Scatter(
        x=Kbar_df['time'][last_nan_index:], 
        y=Kbar_df['SMA'][last_nan_index:], 
        mode='lines',
        line=dict(color='blue', width=1.5), 
        name='中軌'
    ), secondary_y=True)
    
    fig.add_trace(go.Scatter(
        x=Kbar_df['time'][last_nan_index:], 
        y=Kbar_df['Upper_Band'][last_nan_index:], 
        mode='lines',
        line=dict(color='red', width=1, dash='dash'), 
        name='上軌',
        fill=None
    ), secondary_y=True)
    
    fig.add_trace(go.Scatter(
        x=Kbar_df['time'][last_nan_index:], 
        y=Kbar_df['Lower_Band'][last_nan_index:], 
        mode='lines',
        line=dict(color='green', width=1, dash='dash'), 
        name='下軌',
        fill='tonexty'
    ), secondary_y=True)
    
    # 添加下單點位標記
    for trade in TR:
        if trade[0] == 'Buy':
            fig.add_trace(go.Scatter(
                x=[trade[2]], 
                y=[trade[3]],
                mode='markers',
                marker=dict(color='red', symbol='triangle-up', size=10),
                name='多單進場'
            ), secondary_y=True)
        elif trade[0] == 'Sell':
            fig.add_trace(go.Scatter(
                x=[trade[4]], 
                y=[trade[5]],
                mode='markers',
                marker=dict(color='blue', symbol='triangle-down', size=10),
                name='多單出場'
            ), secondary_y=True)
    
    # 設置圖表佈局
    fig.update_layout(
        title='布林通道策略交易點位',
        xaxis_title='時間',
        yaxis_title='價格',
        showlegend=True,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

###### 函數定義: 繪製MACD策略專屬圖表
def ChartOrder_MACD(Kbar_df, TR):
    # 計算MACD
    fast_period = 12
    slow_period = 26
    signal_period = 9
    Kbar_df['EMA_Fast'] = Kbar_df['close'].ewm(span=fast_period, adjust=False).mean()
    Kbar_df['EMA_Slow'] = Kbar_df['close'].ewm(span=slow_period, adjust=False).mean()
    Kbar_df['MACD'] = Kbar_df['EMA_Fast'] - Kbar_df['EMA_Slow']
    Kbar_df['Signal_Line'] = Kbar_df['MACD'].ewm(span=signal_period, adjust=False).mean()
    Kbar_df['MACD_Histogram'] = Kbar_df['MACD'] - Kbar_df['Signal_Line']
    
    # 繪製圖表
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.1, 
                       row_heights=[0.7, 0.3])
    
    # K線圖
    fig.add_trace(go.Candlestick(x=Kbar_df['time'],
                                open=Kbar_df['open'],
                                high=Kbar_df['high'],
                                low=Kbar_df['low'],
                                close=Kbar_df['close'],
                                name='K線'),
                row=1, col=1)
    
    # MACD圖
    fig.add_trace(go.Bar(x=Kbar_df['time'], 
                         y=Kbar_df['MACD_Histogram'], 
                         name='MACD柱狀圖',
                         marker_color=np.where(Kbar_df['MACD_Histogram'] > 0, 'green', 'red')),
                row=2, col=1)
    
    fig.add_trace(go.Scatter(x=Kbar_df['time'], 
                             y=Kbar_df['MACD'], 
                             mode='lines', 
                             name='DIF',
                             line=dict(color='blue', width=2)),
                row=2, col=1)
    
    fig.add_trace(go.Scatter(x=Kbar_df['time'], 
                             y=Kbar_df['Signal_Line'], 
                             mode='lines', 
                             name='DEA',
                             line=dict(color='orange', width=2)),
                row=2, col=1)
    
    # 添加下單點位標記
    for trade in TR:
        if trade[0] == 'Buy':
            fig.add_trace(go.Scatter(
                x=[trade[2]], 
                y=[trade[3]],
                mode='markers',
                marker=dict(color='red', symbol='triangle-up', size=10),
                name='多單進場'
            ), row=1, col=1)
        elif trade[0] == 'Sell':
            fig.add_trace(go.Scatter(
                x=[trade[4]], 
                y=[trade[5]],
                mode='markers',
                marker=dict(color='blue', symbol='triangle-down', size=10),
                name='多單出場'
            ), row=1, col=1)
    
    # 設置圖表佈局
    fig.update_layout(
        title='MACD策略交易圖',
        xaxis_rangeslider_visible=False,
        height=800
    )
    st.plotly_chart(fig, use_container_width=True)

###### 函數定義: 繪製ATR策略專屬圖表
def ChartOrder_ATR(Kbar_df, TR):
    # 計算ATR
    atr_period = 14
    high_low = Kbar_df['high'] - Kbar_df['low']
    high_close = (Kbar_df['high'] - Kbar_df['close'].shift()).abs()
    low_close = (Kbar_df['low'] - Kbar_df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    Kbar_df['ATR'] = tr.rolling(window=atr_period).mean()
    
    # 繪製圖表
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.1, 
                       row_heights=[0.7, 0.3])
    
    # K線圖
    fig.add_trace(go.Candlestick(x=Kbar_df['time'],
                                open=Kbar_df['open'],
                                high=Kbar_df['high'],
                                low=Kbar_df['low'],
                                close=Kbar_df['close'],
                                name='K線'),
                row=1, col=1)
    
    # ATR圖
    fig.add_trace(go.Scatter(x=Kbar_df['time'], 
                             y=Kbar_df['ATR'], 
                             mode='lines', 
                             name='ATR',
                             line=dict(color='purple', width=2)),
                row=2, col=1)
    
    # 添加下單點位標記
    for trade in TR:
        if trade[0] == 'Buy':
            fig.add_trace(go.Scatter(
                x=[trade[2]], 
                y=[trade[3]],
                mode='markers',
                marker=dict(color='red', symbol='triangle-up', size=10),
                name='多單進場'
            ), row=1, col=1)
        elif trade[0] == 'Sell':
            fig.add_trace(go.Scatter(
                x=[trade[4]], 
                y=[trade[5]],
                mode='markers',
                marker=dict(color='blue', symbol='triangle-down', size=10),
                name='多單出場'
            ), row=1, col=1)
    
    # 設置圖表佈局
    fig.update_layout(
        title='ATR策略交易圖',
        xaxis_rangeslider_visible=False,
        height=800
    )
    st.plotly_chart(fig, use_container_width=True)

###### 函數定義: 繪製KD策略專屬圖表
def ChartOrder_KD(Kbar_df, TR):
    # 計算KD
    k_period = 14
    d_period = 3
    low_min = Kbar_df['low'].rolling(window=k_period).min()
    high_max = Kbar_df['high'].rolling(window=k_period).max()
    rsv = 100 * (Kbar_df['close'] - low_min) / (high_max - low_min)
    Kbar_df['K'] = rsv.ewm(alpha=1/d_period, adjust=False).mean()
    Kbar_df['D'] = Kbar_df['K'].ewm(alpha=1/d_period, adjust=False).mean()
    
    # 繪製圖表
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.1, 
                       row_heights=[0.7, 0.3])
    
    # K線圖
    fig.add_trace(go.Candlestick(x=Kbar_df['time'],
                                open=Kbar_df['open'],
                                high=Kbar_df['high'],
                                low=Kbar_df['low'],
                                close=Kbar_df['close'],
                                name='K線'),
                row=1, col=1)
    
    # KD圖
    fig.add_trace(go.Scatter(x=Kbar_df['time'], 
                             y=Kbar_df['K'], 
                             mode='lines', 
                             name='K值',
                             line=dict(color='blue', width=2)),
                row=2, col=1)
    
    fig.add_trace(go.Scatter(x=Kbar_df['time'], 
                             y=Kbar_df['D'], 
                             mode='lines', 
                             name='D值',
                             line=dict(color='orange', width=2)),
                row=2, col=1)
    
    # 添加超買超賣線
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
    
    # 添加下單點位標記
    for trade in TR:
        if trade[0] == 'Buy':
            fig.add_trace(go.Scatter(
                x=[trade[2]], 
                y=[trade[3]],
                mode='markers',
                marker=dict(color='red', symbol='triangle-up', size=10),
                name='多單進場'
            ), row=1, col=1)
        elif trade[0] == 'Sell':
            fig.add_trace(go.Scatter(
                x=[trade[4]], 
                y=[trade[5]],
                mode='markers',
                marker=dict(color='blue', symbol='triangle-down', size=10),
                name='多單出場'
            ), row=1, col=1)
    
    # 設置圖表佈局
    fig.update_layout(
        title='KD策略交易圖',
        xaxis_rangeslider_visible=False,
        height=800
    )
    st.plotly_chart(fig, use_container_width=True)

def calculate_performance(choice, OrderRecord):
    # 獲取合約乘數
    multiplier = contract_multipliers.get(choice, 1)
    
    # 摩台期需要匯率轉換 (假設1:30)
    if "摩台期貨" in choice:
        multiplier *= 30  # 美元轉台幣
        
    # 計算各項績效指標
    交易總盈虧 = OrderRecord.GetTotalProfit() * multiplier
    平均每次盈虧 = OrderRecord.GetAverageProfit() * multiplier
    平均投資報酬率 = OrderRecord.GetAverageProfitRate()
    平均獲利_只看獲利的 = OrderRecord.GetAverEarn() * multiplier
    平均虧損_只看虧損的 = OrderRecord.GetAverLoss() * multiplier
    勝率 = OrderRecord.GetWinRate()
    最大連續虧損 = OrderRecord.GetAccLoss() * multiplier
    最大盈虧回落_MDD = OrderRecord.GetMDD() * multiplier
    
    if 最大盈虧回落_MDD > 0:
        報酬風險比 = 交易總盈虧 / 最大盈虧回落_MDD
    else:
        報酬風險比 = '資料不足無法計算'
        
    return (交易總盈虧, 平均每次盈虧, 平均投資報酬率, 
            平均獲利_只看獲利的, 平均虧損_只看虧損的, 勝率, 
            最大連續虧損, 最大盈虧回落_MDD, 報酬風險比)


# 定義各類商品的合約乘數
contract_multipliers = {
    # 股票類 (單位：元/張，1張=1000股)
    '台積電 2330: 2020.01.02 至 2025.04.16': 1000,
    '英業達 2356: 2020.01.02 至 2024.04.12': 1000,
    '堤維西 1522: 2020.01.02 至 2024.04.12': 1000,
    '0050 台灣50ETF: 2020.01.02 至 2025.03.10': 1000,
    '00631L 台灣50正2: 2023.04.17 至 2025.04.17': 1000,
    '華碩 2357: 2023.04.17 至 2025.04.16': 1000,
    
    # 期貨類 (單位：元/點)
    '大台指期貨2024.12到期: 2023.12 至 2024.4.11': 200,    # 大台指
    '小台指期貨2024.12到期: 2023.12 至 2024.4.11': 50,     # 小台指
    '金融期貨 CBF: 2023.04.17 至 2025.04.17': 1000,        # 金融期
    '電子期貨 CCF: 2023.04.17 至 2025.04.16': 4000,        # 電子期
    '小型電子期貨 CDF: 2020.03.02 至 2025.04.14': 1000,    # 小型電子期
    '非金電期貨 CEF: 2023.04.17 至 2025.04.16': 4000,      # 非金電期
    '摩台期貨 CMF: 2023.04.17 至 2025.04.17': 100,        # 摩台期(美元計價)
    '小型金融期貨 CQF: 2023.04.17 至 2025.04.17': 250,     # 小型金融期
    '美元指數期貨 FXF: 2020.03.02 至 2025.04.14': 1000     # 美元指數期
}

# 統一績效計算函數
def calculate_performance(choice, OrderRecord):
    # 獲取合約乘數
    multiplier = contract_multipliers.get(choice, 1)
    
    # 摩台期需要匯率轉換 (假設1:30)
    if "摩台期貨" in choice:
        multiplier *= 30  # 美元轉台幣
        
    # 計算各項績效指標
    交易總盈虧 = OrderRecord.GetTotalProfit() * multiplier
    平均每次盈虧 = OrderRecord.GetAverageProfit() * multiplier
    平均投資報酬率 = OrderRecord.GetAverageProfitRate()
    平均獲利_只看獲利的 = OrderRecord.GetAverEarn() * multiplier
    平均虧損_只看虧損的 = OrderRecord.GetAverLoss() * multiplier
    勝率 = OrderRecord.GetWinRate()
    最大連續虧損 = OrderRecord.GetAccLoss() * multiplier
    最大盈虧回落_MDD = OrderRecord.GetMDD() * multiplier
    
    if 最大盈虧回落_MDD > 0:
        報酬風險比 = 交易總盈虧 / 最大盈虧回落_MDD
    else:
        報酬風險比 = '資料不足無法計算'
        
    return (交易總盈虧, 平均每次盈虧, 平均投資報酬率, 
            平均獲利_只看獲利的, 平均虧損_只看虧損的, 勝率, 
            最大連續虧損, 最大盈虧回落_MDD, 報酬風險比)


#%%
####### (6) 程式交易 ####### 
st.subheader("程式交易:")

# 在選擇金融商品後初始化 OrderRecord
OrderRecord = Record()

# 新增策略選擇
choices_strategies = [
    '移動平均線策略',
    'RSI逆勢策略',
    '布林通道策略',
    'MACD策略',
    'ATR波動率策略',
    'KD隨機指標策略',
    '多策略組合'
]
choice_strategy = st.selectbox('選擇交易策略', choices_strategies, index=0)

# 最佳化參數搜索開關
optimize_params = st.checkbox('啟用最佳化參數搜索', value=False)

# 通用參數設定
with st.expander("通用參數設定"):
    MoveStopLoss = st.slider('停損點數(股票:每股; 期貨:指數點數)', 1, 100, 30)
    Order_Quantity = st.slider('交易數量(股票:張; 期貨:口數)', 1, 100, 1)
    max_optimization_iterations = st.slider('最佳化最大迭代次數', 10, 500, 50) if optimize_params else 0

# 策略專屬參數設定
strategy_params = {}
if choice_strategy == '移動平均線策略':
    with st.expander("移動平均線策略參數"):
        strategy_params['LongMAPeriod'] = st.slider('長移動平均線週期', 5, 100, 20)
        strategy_params['ShortMAPeriod'] = st.slider('短移動平均線週期', 1, 50, 5)
        
        if optimize_params:
            strategy_params['optimize_LongMAPeriod'] = st.slider('長MA最佳化範圍', 10, 100, (15, 50))
            strategy_params['optimize_ShortMAPeriod'] = st.slider('短MA最佳化範圍', 1, 30, (3, 15))

elif choice_strategy == 'RSI逆勢策略':
    with st.expander("RSI策略參數"):
        strategy_params['RSIPeriod'] = st.slider('RSI週期', 5, 30, 14)
        strategy_params['OverSold'] = st.slider('超賣閾值', 1, 40, 30)
        strategy_params['OverBought'] = st.slider('超買閾值', 60, 100, 70)
        
        if optimize_params:
            strategy_params['optimize_RSIPeriod'] = st.slider('RSI週期最佳化範圍', 5, 30, (10, 20))
            strategy_params['optimize_OverSold'] = st.slider('超賣閾值最佳化範圍', 20, 40, (25, 35))
            strategy_params['optimize_OverBought'] = st.slider('超買閾值最佳化範圍', 65, 85, (70, 80))

elif choice_strategy == '布林通道策略':
    with st.expander("布林通道策略參數"):
        strategy_params['BBPeriod'] = st.slider('布林通道週期', 10, 50, 20)
        strategy_params['NumStdDev'] = st.slider('標準差倍數', 1.0, 3.0, 2.0)
        
        if optimize_params:
            strategy_params['optimize_BBPeriod'] = st.slider('BB週期最佳化範圍', 10, 50, (15, 30))
            strategy_params['optimize_NumStdDev'] = st.slider('標準差倍數最佳化範圍', 1.5, 2.5, (1.8, 2.2))

elif choice_strategy == 'MACD策略':
    with st.expander("MACD策略參數"):
        strategy_params['FastPeriod'] = st.slider('快速線週期', 5, 20, 12)
        strategy_params['SlowPeriod'] = st.slider('慢速線週期', 15, 50, 26)
        strategy_params['SignalPeriod'] = st.slider('訊號線週期', 5, 20, 9)
        
        if optimize_params:
            strategy_params['optimize_FastPeriod'] = st.slider('快速線最佳化範圍', 8, 20, (10, 15))
            strategy_params['optimize_SlowPeriod'] = st.slider('慢速線最佳化範圍', 20, 40, (22, 30))
            strategy_params['optimize_SignalPeriod'] = st.slider('訊號線最佳化範圍', 5, 15, (7, 12))

elif choice_strategy == 'ATR波動率策略':
    with st.expander("ATR策略參數"):
        strategy_params['ATRPeriod'] = st.slider('ATR週期', 5, 30, 14)
        strategy_params['ATRMultiplier'] = st.slider('ATR倍數', 1.0, 5.0, 2.0)
        
        if optimize_params:
            strategy_params['optimize_ATRPeriod'] = st.slider('ATR週期最佳化範圍', 10, 25, (12, 18))
            strategy_params['optimize_ATRMultiplier'] = st.slider('ATR倍數最佳化範圍', 1.5, 3.5, (2.0, 3.0))

elif choice_strategy == 'KD隨機指標策略':
    with st.expander("KD策略參數"):
        strategy_params['KPeriod'] = st.slider('K值週期', 5, 20, 9)
        strategy_params['DPeriod'] = st.slider('D值週期', 3, 10, 3)
        strategy_params['OverSold'] = st.slider('超賣閾值', 10, 30, 20)
        strategy_params['OverBought'] = st.slider('超買閾值', 70, 90, 80)
        
        if optimize_params:
            strategy_params['optimize_KPeriod'] = st.slider('K值週期最佳化範圍', 5, 20, (7, 14))
            strategy_params['optimize_DPeriod'] = st.slider('D值週期最佳化範圍', 3, 10, (3, 6))
            strategy_params['optimize_OverSold'] = st.slider('超賣閾值最佳化範圍', 20, 30, (20, 25))
            strategy_params['optimize_OverBought'] = st.slider('超買閾值最佳化範圍', 70, 85, (75, 80))

elif choice_strategy == '多策略組合':
    with st.expander("組合策略參數"):
        strategy_params['Weight_MA'] = st.slider('MA策略權重', 0.0, 1.0, 0.4)
        strategy_params['Weight_RSI'] = st.slider('RSI策略權重', 0.0, 1.0, 0.3)
        strategy_params['Weight_BB'] = st.slider('BB策略權重', 0.0, 1.0, 0.3)

# 回测按钮
if st.button('開始回測'):
    # 建立部位管理物件
    # 判斷是否為期貨商品
    is_future = False
    for key in product_info:
        if "期貨" in key:
            is_future = True
            break
    
    if is_future:  # 期貨商品
        OrderRecord = Record(spread=3.628e-4, tax=0.00002, commission=0.0002, isFuture=True)
    else:  # 股票商品
        OrderRecord = Record(spread=3.628e-4, tax=0.003, commission=0.001425, isFuture=False)
        
    # 根據選擇的策略執行回測
    if choice_strategy == '移動平均線策略':
        if optimize_params:
            # 執行最佳化參數搜索
            best_params = optimize_ma_strategy(
                KBar_df, 
                strategy_params['optimize_LongMAPeriod'],
                strategy_params['optimize_ShortMAPeriod'],
                max_iterations=max_optimization_iterations
            )
            strategy_params['LongMAPeriod'] = best_params[0]
            strategy_params['ShortMAPeriod'] = best_params[1]
            st.success(f"最佳參數: 長MA={best_params[0]}, 短MA={best_params[1]}")
        
        # 使用最佳參數執行回測
        OrderRecord = back_test_ma_strategy(
            OrderRecord, KBar_df,
            MoveStopLoss, 
            strategy_params['LongMAPeriod'], 
            strategy_params['ShortMAPeriod'], 
            Order_Quantity
        )
        
    elif choice_strategy == 'RSI逆勢策略':
        # 計算RSI
        KBar_df['RSI'] = Calculate_RSI(KBar_df, period=strategy_params['RSIPeriod'])
        
        if optimize_params:
            # 執行最佳化參數搜索
            best_params = optimize_rsi_strategy(
                KBar_df, 
                strategy_params['optimize_RSIPeriod'],
                strategy_params['optimize_OverSold'],
                strategy_params['optimize_OverBought'],
                max_iterations=max_optimization_iterations
            )
            strategy_params['RSIPeriod'] = best_params[0]
            strategy_params['OverSold'] = best_params[1]
            strategy_params['OverBought'] = best_params[2]
            st.success(f"最佳參數: RSI週期={best_params[0]}, 超賣={best_params[1]}, 超買={best_params[2]}")
        
        # 使用最佳參數執行回測
        OrderRecord = back_test_rsi_strategy(
            OrderRecord, KBar_df,
            MoveStopLoss, 
            strategy_params['RSIPeriod'], 
            strategy_params['OverSold'], 
            strategy_params['OverBought'], 
            Order_Quantity
        )
    
    elif choice_strategy == '布林通道策略':
        # 計算布林通道
        KBar_df = Calculate_Bollinger_Bands(KBar_df, 
                                           period=strategy_params['BBPeriod'], 
                                           num_std_dev=strategy_params['NumStdDev'])
        
        if optimize_params:
            # 執行最佳化參數搜索
            best_params = optimize_bb_strategy(
                KBar_df, 
                strategy_params['optimize_BBPeriod'],
                strategy_params['optimize_NumStdDev'],
                max_iterations=max_optimization_iterations
            )
            strategy_params['BBPeriod'] = best_params[0]
            strategy_params['NumStdDev'] = best_params[1]
            st.success(f"最佳參數: BB週期={best_params[0]}, 標準差倍數={best_params[1]}")
        
        # 使用最佳參數執行回測
        OrderRecord = back_test_bb_strategy(
            OrderRecord, KBar_df,
            MoveStopLoss, 
            strategy_params['BBPeriod'], 
            strategy_params['NumStdDev'], 
            Order_Quantity
        )
    
    elif choice_strategy == 'MACD策略':
        # 計算MACD
        KBar_df = Calculate_MACD(KBar_df, 
                                fast_period=strategy_params['FastPeriod'], 
                                slow_period=strategy_params['SlowPeriod'], 
                                signal_period=strategy_params['SignalPeriod'])
        
        if optimize_params:
            # 執行最佳化參數搜索
            best_params = optimize_macd_strategy(
                KBar_df, 
                strategy_params['optimize_FastPeriod'],
                strategy_params['optimize_SlowPeriod'],
                strategy_params['optimize_SignalPeriod'],
                max_iterations=max_optimization_iterations
            )
            strategy_params['FastPeriod'] = best_params[0]
            strategy_params['SlowPeriod'] = best_params[1]
            strategy_params['SignalPeriod'] = best_params[2]
            st.success(f"最佳參數: 快線={best_params[0]}, 慢線={best_params[1]}, 訊號線={best_params[2]}")
        
        # 使用最佳參數執行回測
        OrderRecord = back_test_macd_strategy(
            OrderRecord, KBar_df,
            MoveStopLoss, 
            strategy_params['FastPeriod'], 
            strategy_params['SlowPeriod'], 
            strategy_params['SignalPeriod'], 
            Order_Quantity
        )
    
    elif choice_strategy == 'ATR波動率策略':
        # 計算ATR
        KBar_df['ATR'] = Calculate_ATR(KBar_df, period=strategy_params['ATRPeriod'])
        
        if optimize_params:
            # 執行最佳化參數搜索
            best_params = optimize_atr_strategy(
                KBar_df, 
                strategy_params['optimize_ATRPeriod'],
                strategy_params['optimize_ATRMultiplier'],
                max_iterations=max_optimization_iterations
            )
            strategy_params['ATRPeriod'] = best_params[0]
            strategy_params['ATRMultiplier'] = best_params[1]
            st.success(f"最佳參數: ATR週期={best_params[0]}, ATR倍數={best_params[1]}")
        
        # 使用最佳參數執行回測
        OrderRecord = back_test_atr_strategy(
            OrderRecord, KBar_df,
            MoveStopLoss, 
            strategy_params['ATRPeriod'], 
            strategy_params['ATRMultiplier'], 
            Order_Quantity
        )
    
    elif choice_strategy == 'KD隨機指標策略':
        # 計算KD
        KBar_df['K'], KBar_df['D'] = Calculate_KD(KBar_df, 
                                                k_period=strategy_params['KPeriod'], 
                                                d_period=strategy_params['DPeriod'])
        
        if optimize_params:
            # 執行最佳化參數搜索
            best_params = optimize_kd_strategy(
                KBar_df, 
                strategy_params['optimize_KPeriod'],
                strategy_params['optimize_DPeriod'],
                strategy_params['optimize_OverSold'],
                strategy_params['optimize_OverBought'],
                max_iterations=max_optimization_iterations
            )
            strategy_params['KPeriod'] = best_params[0]
            strategy_params['DPeriod'] = best_params[1]
            strategy_params['OverSold'] = best_params[2]
            strategy_params['OverBought'] = best_params[3]
            st.success(f"最佳參數: K={best_params[0]}, D={best_params[1]}, 超賣={best_params[2]}, 超買={best_params[3]}")
        
        # 使用最佳參數執行回測
        OrderRecord = back_test_kd_strategy(
            OrderRecord, KBar_df,
            MoveStopLoss, 
            strategy_params['KPeriod'], 
            strategy_params['DPeriod'], 
            strategy_params['OverSold'], 
            strategy_params['OverBought'], 
            Order_Quantity
        )
    
    elif choice_strategy == '多策略組合':
        # 多策略組合回測
        OrderRecord = back_test_multi_strategy(
            OrderRecord, KBar_df,
            MoveStopLoss, 
            Order_Quantity,
            strategy_params
        )
    
    # 顯示交易紀錄
    st.subheader("交易紀錄")
    trade_records = OrderRecord.GetTradeRecord()
    if trade_records:
        trade_df = pd.DataFrame(trade_records, 
                               columns=['方向', '商品', '進場時間', '進場價格', '數量', '出場時間', '出場價格', '損益'])
        st.dataframe(trade_df)
    else:
        st.warning("沒有交易紀錄")
    
    # 繪製策略專屬圖表
    if choice_strategy == '移動平均線策略':
        ChartOrder_MA(KBar_df, trade_records)
    elif choice_strategy == 'RSI逆勢策略':
        ChartOrder_RSI(KBar_df, trade_records)
    elif choice_strategy == '布林通道策略':
        ChartOrder_BB(KBar_df, trade_records)
    elif choice_strategy == 'MACD策略':
        ChartOrder_MACD(KBar_df, trade_records)
    elif choice_strategy == 'ATR波動率策略':
        ChartOrder_ATR(KBar_df, trade_records)
    elif choice_strategy == 'KD隨機指標策略':
        ChartOrder_KD(KBar_df, trade_records)
    elif choice_strategy == '多策略組合':
        st.write("多策略組合不提供專屬圖表")
    
    # 計算並顯示績效指標
    if hasattr(OrderRecord, 'Profit') and len(OrderRecord.Profit) > 0:
        results = calculate_performance(choice, OrderRecord)
        (交易總盈虧, 平均每次盈虧, 平均投資報酬率, 
         平均獲利_只看獲利的, 平均虧損_只看虧損的, 勝率, 
         最大連續虧損, 最大盈虧回落_MDD, 報酬風險比) = results
        
        # 顯示績效
        data = {
            "項目": ["交易總盈虧(元)", "平均每次盈虧(元)", "平均投資報酬率", "平均獲利(只看獲利的)(元)", 
                   "平均虧損(只看虧損的)(元)", "勝率", "最大連續虧損(元)", "最大盈虧回落(MDD)(元)", 
                   "報酬風險比(交易總盈虧/最大盈虧回落(MDD))"],
            "數值": [交易總盈虧, 平均每次盈虧, 平均投資報酬率, 平均獲利_只看獲利的, 
                   平均虧損_只看虧損的, 勝率, 最大連續虧損, 最大盈虧回落_MDD, 報酬風險比]
        }
        df = pd.DataFrame(data)
        st.write(df)
        
        # 繪製累計盈虧圖
        if choice in contract_multipliers:
            # 根據商品類型選擇繪圖參數
            if "台積電" in choice or "英業達" in choice or "堤維西" in choice:
                OrderRecord.GeneratorProfitChart(choice='stock', StrategyName='MA')
            elif "大台指" in choice:
                OrderRecord.GeneratorProfitChart(choice='future1', StrategyName='MA')
            elif "小台指" in choice:
                OrderRecord.GeneratorProfitChart(choice='future2', StrategyName='MA')
            else:
                # 默認股票類型
                OrderRecord.GeneratorProfitChart(choice='stock', StrategyName='MA')
        else:
            st.warning("無法識別的商品類型")
            
        # 繪製累計投資報酬率圖
        OrderRecord.GeneratorProfit_rateChart(StrategyName='MA')
    else:
        st.warning("沒有交易記錄(已經了結之交易) !")


