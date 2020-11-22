#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:35:32 2020

@author: alex
"""

import numpy as np
import pandas as pd

import os

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import time
from datetime import datetime, timedelta
import itertools
import os
import pickle
import math
from sklearn.preprocessing import StandardScaler
from ressup import ressup
from ib_insync import util
import tensorflow as tf
import nest_asyncio
nest_asyncio.apply()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# util.startLoop()


class get_data:

    def next_exp_weekday(self):
        weekdays = {2: [5, 6, 0], 4: [0, 1, 2], 0: [3, 4]}
        today = datetime.today().weekday()
        for exp, day in weekdays.items():
            if today in day:
                return exp

    def next_weekday(self, d, weekday):

        days_ahead = weekday - d.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        date_to_return = d + timedelta(days_ahead)  # 0 = Monday, 1=Tuself.ESday, 2=Wednself.ESday...
        return date_to_return.strftime('%Y%m%d')

    def get_strikes_and_expiration(self):
        ES = Future(symbol='ES', lastTradeDateOrContractMonth='20200918', exchange='GLOBEX',
                                currency='USD')
        ib.qualifyContracts(ES)
        expiration = self.next_weekday(datetime.today(), self.next_exp_weekday())
        chains = ib.reqSecDefOptParams(underlyingSymbol='ES', futFopExchange='GLOBEX', underlyingSecType='FUT',underlyingConId=ES.conId)
        chain = util.df(chains)
        strikes = chain[chain['expirations'].astype(str).str.contains(expiration)].loc[:, 'strikes'].values[0]
        [ESValue] = ib.reqTickers(ES)
        ES_price= ESValue.marketPrice()
        strikes = [strike for strike in strikes
                if strike % 5 == 0
                and ES_price - 10 < strike < ES_price + 10]
        return strikes,expiration

    def get_contract(self, right, net_liquidation):
        strikes, expiration=self.get_strikes_and_expiration()
        for strike in strikes:
            contract=FuturesOption(symbol='ES', lastTradeDateOrContractMonth=expiration,
                                                strike=strike,right=right,exchange='GLOBEX')
            ib.qualifyContracts(contract)
            price = ib.reqMktData(contract,"",False,False)
            if float(price.last)*50 >=net_liquidation:
                continue
            else:
                return contract

    def res_sup(self,ES_df):
        ES_df = ES_df.reset_index(drop=True)
        ressupDF = ressup(ES_df, len(ES_df))
        res = ressupDF['Resistance'].values
        sup = ressupDF['Support'].values
        return res, sup

    def ES(self,ES):
        
        ES_df = util.df(ES)
        ES_df.set_index('date',inplace=True)
        ES_df.index = pd.to_datetime(ES_df.index)
        ES_df['hours'] = ES_df.index.strftime('%H').astype(int)
        ES_df['minutes'] = ES_df.index.strftime('%M').astype(int)
        ES_df['hours + minutes'] = ES_df['hours']*100 + ES_df['minutes']
        ES_df['Day_of_week'] = ES_df.index.dayofweek
        ES_df['Resistance'], ES_df['Support'] = self.res_sup(ES_df)
        ES_df['RSI'] = ta.RSI(ES_df['close'])
        ES_df['macd'],ES_df['macdsignal'],ES_df['macdhist'] = ta.MACD(ES_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        ES_df['macd - macdsignal'] = ES_df['macd'] - ES_df['macdsignal']
        ES_df['MA_9']=ta.MA(ES_df['close'], timeperiod=9)
        ES_df['MA_21']=ta.MA(ES_df['close'], timeperiod=21)
        ES_df['MA_200']=ta.MA(ES_df['close'], timeperiod=200)
        ES_df['EMA_9']=ta.EMA(ES_df['close'], timeperiod=9)
        ES_df['EMA_21']=ta.EMA(ES_df['close'], timeperiod=21)
        ES_df['EMA_50']=ta.EMA(ES_df['close'], timeperiod=50)
        ES_df['EMA_200']=ta.EMA(ES_df['close'], timeperiod=200)
        ES_df['ATR']=ta.ATR(ES_df['high'],ES_df['low'], ES_df['close'])
        ES_df['roll_max_cp']=ES_df['high'].rolling(20).max()
        ES_df['roll_min_cp']=ES_df['low'].rolling(20).min()
        ES_df['roll_max_vol']=ES_df['volume'].rolling(20).max()
        ES_df['vol/max_vol'] = ES_df['volume']/ES_df['roll_max_vol']
        ES_df['EMA_21-EMA_9']=ES_df['EMA_21']-ES_df['EMA_9']
        ES_df['EMA_200-EMA_50']=ES_df['EMA_200']-ES_df['EMA_50']
        ES_df['B_upper'], ES_df['B_middle'], ES_df['B_lower'] = ta.BBANDS(ES_df['close'], matype=MA_Type.T3)
        ES_df.dropna(inplace = True)
        
        return ES_df

    def option_history(self, contract):
        ib.qualifyContracts(contract)
        df = pd.DataFrame(util.df(ib.reqHistoricalData(contract=contract, endDateTime='', durationStr=No_days,
                                      barSizeSetting=interval, whatToShow = 'MIDPOINT', useRTH = False, keepUpToDate=True))[['date','close']])
        df.columns=['date',f"{contract.symbol}_{contract.right}_close"]
        df.set_index('date',inplace=True)
        return df

    def options(self, df1=None,df2=None):
        return df1

def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=5):
    """ A multi-layer perceptron """
     
    # input layer
    i = Input(shape=(input_dim,1))
    x = i
     
    # hidden layers
    for _ in range(n_hidden_layers):
      # x = Dropout(0.2)(x)
      # x = LSTM(hidden_dim, return_sequences = True)(x)
      x = Dense(hidden_dim, activation='relu')(x)
     
    x = GlobalAveragePooling1D()(x)
    # final layer
    # x = Dense(n_action, activation='relu')(x)
    x = Dense(n_action, activation='softmax')(x)
    # make the model
    model = Model(i, x)
     
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print((model.summary()))
    return model





def reset(data, stock_owned, cash_in_hand):
    stock_price = data[-2:]
    return _get_obs(stock_owned, stock_price, cash_in_hand)

def _get_obs(stock_owned, stock_price, cash_in_hand):
    obs = np.empty(5+len(data[:-2]))
    obs[:2] = stock_owned
    obs[2:2*2] = stock_price
    obs[4] = cash_in_hand
    obs[5:] = data[:-2]
    return obs, stock_price, cash_in_hand





    
def flatten_position(contract, price):
    positions = ib.positions()
    for each in positions:
        if each.contract.right != contract.right:
            continue
        ib.qualifyContracts(each.contract)
        if each.position > 0: # Number of active Long positions
            action = 'SELL' # to offset the long positions
        elif each.position < 0: # Number of active Short positions
            action = 'BUY' # to offset the short positions
        else:
            assert False
        totalQuantity = abs(each.position)
        price = price.bid - 0.25 
        while math.isnan(price):
            price = ib.reqMktData(each.contract).bid-0.25
            ib.sleep(0.1)
        print(f'price = {price}')
        order = LimitOrder(action, totalQuantity, price) #round(25 * round(stock_price[i]/25, 2), 2))
        trade = ib.placeOrder(each.contract, order)
        print(f'Flatten Position: {action} {totalQuantity} {contract.localSymbol}')
        for c in ib.loopUntil(condition=0, timeout=120): # trade.orderStatus.status == "Filled"  or \
            #trade.orderStatus.status == "Cancelled"
            print(trade.orderStatus.status)
            c=len(ib.reqAllOpenOrders())
            print(f'Open orders = {c}')
            if c==0 or trade.orderStatus.status == 'Inactive': 
                print('sell loop finished')
                return
        
    
def option_position():
    stock_owned = np.zeros(2)
    position = ib.positions()
    call_position= None
    put_position = None
    for each in position:
        if each.contract.right == 'C':
            call_position = each.contract
            ib.qualifyContracts(call_position)
            stock_owned[0] = each.position
        elif each.contract.right == 'P':
            put_position = each.contract
            ib.qualifyContracts(put_position)
            stock_owned[1] = each.position
    call_position = call_position if call_position != None else res.get_contract('C', 2000)
    put_position = put_position if put_position != None else res.get_contract('P', 2000)
    return stock_owned, call_position, put_position


def trade(ES, hasNewBar=None):
    global data
    global call_option_price
    global put_option_price

    stock_owned, call_contract, put_contract = option_position()
    # print(f'call bid price = {call_option_price.bid}, put bid price = {put_option_price.bid}')
    model.load_weights(name)
    cash_in_hand = float(ib.accountSummary()[22].value)
    portolio_value = float(ib.accountSummary()[29].value)

    call_contract_price = (call_option_price.ask + call_option_price.bid)/2
    put_contract_price = (put_option_price.ask + put_option_price.bid)/2
    options_array = np.array([call_contract_price, put_contract_price])

    data_raw = res.options(res.options(res.ES(ES)))
    data = data_raw[['Day_of_week', 'hours + minutes', 'EMA_21-EMA_9', 'EMA_200-EMA_50', 'RSI', 'ATR','macd - macdsignal','macdhist', 'vol/max_vol']].iloc[-1,:].values

    data = np.append(data,options_array,axis=0)
    #choose parameters to drop if not needed
    state, stock_price, cash_in_hand = reset(data, stock_owned, cash_in_hand)
    state = scaler.transform(state.reshape(-1,len(state)))
    action_list = list(map(list, itertools.product([0, 1, 2], repeat=2)))
    action=np.argmax(model.predict(state))
    action_vec = action_list[action]
    buy_index = [] 
    sell_index = []
    
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)

    if data_raw["low"].iloc[-1] < data_raw["close"].iloc[-2] - (2 *  data_raw["ATR"].iloc[-2]) and len(ib.positions())!=0 and len(ib.reqAllOpenOrders())==0 and sell_index==[]:
        sell_index.append(0)

    
    elif data_raw["high"].iloc[-1] > data_raw["close"].iloc[-2] + (2 * data_raw["ATR"].iloc[-2]) and len(ib.positions())!=0 and len(ib.reqAllOpenOrders())==0 and sell_index==[]:
        sell_index.append(1)
    
    if sell_index:
        for i in sell_index:
            if not stock_owned[i] == 0:
                contract= call_contract if i == 0 else put_contract
                ib.qualifyContracts(contract)
                price = call_option_price if i == 0 else put_option_price
                flatten_position(contract, price)
            cash_in_hand = float(ib.accountSummary()[5].value)
            stock_owned, call_contract, put_contract = option_position()
    
    if buy_index:
        can_buy = True
        while can_buy:
            
            for i in buy_index:
                contract = call_contract if i == 0 else put_contract
                ib.qualifyContracts(contract)
                
                if cash_in_hand > (stock_price[i] * 50) and cash_in_hand > portolio_value \
                    and ((stock_owned[0] == 0 and i == 0) or (stock_owned[1] == 0 and i == 1)) and len(ib.reqAllOpenOrders()) == 0: 
                    stock_price[i] = call_option_price.ask + 0.25 if i == 0 else put_option_price.ask + 0.25
                
                    while math.isnan(stock_price[i]):
                        stock_price[i] = call_option_price.ask + 0.25 if i == 0 else put_option_price.ask + 0.25
                
                        ib.sleep(0.1)
                    quantity = 1 # int((cash_in_hand/(stock_price[i] * 50)))
                  
                    order = LimitOrder('BUY', quantity, stock_price[i]) #round(25 * round(stock_price[i]/25, 2), 2))
                    trade = ib.placeOrder(contract, order)
                    no_price_checking = 1
                    for c in ib.loopUntil(condition=0, timeout=120): # trade.orderStatus.status == "Filled"  or \
                        #trade.orderStatus.status == "Cancelled"
                        print(trade.orderStatus.status)
                        print(no_price_checking)
                        no_price_checking+=1
                        c=len(ib.reqAllOpenOrders())
                        print(f'Open orders = {c}')
                        ib.sleep(2)
                        if c==0: break
                  
                    print('out of loop')
                    stock_owned, call_contract, put_contract = option_position()
                    cash_in_hand = float(ib.accountSummary()[5].value)
                    can_buy = False
                else:
                  can_buy = False

            
    print(f'action from action lists = {action}, action_vector = {action_vec}, no of contract position [Calls, Puts] = {stock_owned}, cash in hand= {cash_in_hand}')


if __name__ == "__main__":
    global call_option_price
    global put_option_price
    global stock_owned
    from ib_insync import *
    import talib as ta
    from talib import MA_Type

    ib = IB()
    ib.disconnect()
    ib.connect('127.0.0.1', 7497, clientId=np.random.randint(10, 1000))
    path = os.getcwd() 
   
    # config
    models_folder = f'{path}/rl_trader_models_Sup/1_layer_BO_RSI_ATR_Close' #where models and scaler are saved
    rewards_folder = f'{path}/rl_trader_rewards_Sup/1_layer_BO_RSI_ATR_Close' #where results are saved
    name = f'{models_folder}/dqn.h5'
    
    model = mlp(10,9)
    
    previous_action = ''
    
    with open(f'{rewards_folder}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f) 
    
    stock_owned = np.zeros(2)
    tickers_signal = "Hold"
    buy_index = [] 
    sell_index = []
    endDateTime = ''
    No_days = '2 D'
    interval = '1 min'
    res = get_data()

    ES = Future(symbol='ES', lastTradeDateOrContractMonth='20200918', exchange='GLOBEX',
                                currency='USD')
    ib.qualifyContracts(ES)
    ES = ib.reqHistoricalData(contract=ES, endDateTime='', durationStr=No_days,
                                 barSizeSetting=interval, whatToShow = 'TRADES', useRTH = False, keepUpToDate=True)
    stock_owned, call_contract, put_contract = option_position()
    call_option_price = ib.reqMktData(call_contract, '', False, False)
    put_option_price = ib.reqMktData(put_contract, '', False, False)
    trade(ES)
    
    while ib.waitOnUpdate():
        ES.updateEvent += trade


    # print('passed to util')
    # now = datetime.now()
    # ES.updateEvent += trade in ib.timeRange(start=datetime(now.year,now.month,now.day,15,0,00), end=datetime(now.year,now.month,now.day+1,14,0,00),step=60)
