import yfinance as yf
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import seaborn as sns
import plotly.graph_objs as go
from AlmaIndicator import ALMAIndicator
class Stocks:
	def __init__(self, dataset):
		self.dataset = dataset
	
	def null_values(dataset):
		try:
			dataset.isnull().sum()
			num_null = dataset.isnull().sum().sum()
			print("Total Numbers of Null Values: {}".format(num_null))
		except TypeError:
			print("Enter dataset")
	
	def scheme_A(dataset):
	    scheme_A = np.where(dataset['Adj Close'].shift(-1)>dataset['Adj Close'],1,-1)
	    dataset['Scheme1'] = scheme_A
	
	def scheme_B(dataset):
	    rolling_mean = dataset['Adj Close'].rolling(window=50).mean()
	    rolling_mean2 = dataset['Adj Close'].rolling(window=200).mean()

	def scheme_C(dataset, ema, alma):
		return np.where(ema > alma, 1, -1)

	def calc_ema(dataset):
		return dataset['Adj Close'].ewm(com=0.4).mean()

	def calc_alma(dataset):
		alma = ALMAIndicator(close = dataset['Adj Close'])
		return (alma.alma())
		