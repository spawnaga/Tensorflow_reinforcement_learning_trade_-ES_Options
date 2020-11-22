# Tensorflow_reinforcement_learning_trade_-ES_Options
trade ES Futures Options

This project is reserved to create an agent to trace ES prices and trades Futures Options.

Problem : create an AI to trade successfuly with rewards the ES Ftures using Reinfrocement learning Machine learning.

Given: ES 1 min OHLCV chart

Method: by using Talib library in python, we will create technicals and train the agent on them. In my work, I will use OBV, OBV slope for 5 min interval, RSI, EMA 9 - EMA 26 difference value.

Tools: interactive brokers (TWS/IB GATEWAY) as a broker to collect OHLCV chart and python (I will use PyCharm platform)

NOTE : Very important, I am using few prepared templates from classes I took in udemy which are :

1-  the lazy programmer course in udemy for Tensorflow machine learning class
 Lecture of Q-Learning
 https://github.com/lazyprogrammer/machine_learning_examples/blob/master/tf2.0/rl_trader.py
 
 2- few functions from this course : Algorithmic Trading & Quantitative Analysis Using Python
 created by Mayank Rasu
 
 https://www.udemy.com/course/algorithmic-trading-quantitative-analysis-using-python/#instructor-1
 
 
 and I also used some of the ideas mentioned in these courses
 
 This project is for my own use and not intending to sale/give/publish for the public and I am seeking helps from peers who have some knowledge and advices to improve the results. This not yet proven to make profits an should not used to trade with real money. 
