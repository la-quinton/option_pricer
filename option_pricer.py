import pylab
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend

def custom_activation(x):
    return backend.exp(x)

def CheckAccuracy(y,y_hat):
    stats = dict()
    
    stats['diff'] = y - y_hat
    
    stats['mse'] = np.mean(stats['diff']**2)
    print("Mean Squared Error:      ", stats['mse'])
    
    stats['rmse'] = math.sqrt(stats['mse'])
    print("Root Mean Squared Error: ", stats['rmse'])
    
    stats['mae'] = np.mean(abs(stats['diff']))
    print("Mean Absolute Error:     ", stats['mae'])
    
    stats['mpe'] = math.sqrt(stats['mse'])/np.mean(y)
    print("Mean Percent Error:      ", stats['mpe'])

    mpl.rcParams['agg.path.chunksize'] = 100000
    mpl.pyplot.figure(figsize=(14,10))
    plt.scatter(y, y_hat,color='black',linewidth=0.3,alpha=0.4, s=0.5)
    plt.xlabel('Actual Price',fontsize=32,fontname='Times New Roman')
    plt.ylabel('Predicted Price',fontsize=32,fontname='Times New Roman')
    ax=plt.gca()
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 120])
    
    mpl.pyplot.figure(figsize=(14,10))
    plt.hist(stats['diff'], bins=150,edgecolor='black',align='mid',color='white')
    plt.xlabel('Diff',fontsize=32)
    plt.ylabel('Density',fontsize=40)
    plt.show()
    
    return stats

def Black_Scholes(S, T, r, sigma):
    d1 = (math.log(S)+(r+sigma**2/2.)*T)/(sigma*math.sqrt(T))
    d2 = d1-sigma*math.sqrt(T)
    return S*norm.cdf(d1)-math.exp(-r*T)*norm.cdf(d2)

# Replace location name in quotes with location of file on your device
df = pd.read_csv('data.csv')

df["Stock Price"] = df["Stock Price"]/df["Strike Price"]
df["Call Price"] = df["Call Price"]/df["Strike Price"]

df['Black Scholes'] = df.apply(lambda x: Black_Scholes(x['Stock Price'], x['Maturity'], x['Risk-free'], x['Volatility']), axis=1)

train = df.sample(frac = 0.8)
X_train = train[['Stock Price', 'Maturity', 'Dividends', 'Volatility', 'Risk-free']].values
y_train = train['Call Price'].values
test = df.drop(train.index)
X_test = test[['Stock Price', 'Maturity', 'Dividends', 'Volatility', 'Risk-free']].values
y_test = test['Call Price'].values

black_scholes_train = train['Black Scholes'].values
black_scholes_test = test['Black Scholes'].values

nodes = 120
model = Sequential()

model.add(Dense(nodes, input_dim=X_train.shape[1]))
model.add(LeakyReLU())
model.add(Dropout(0.25))

model.add(Dense(nodes, activation='elu'))
model.add(Dropout(0.25))

model.add(Dense(nodes, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(nodes, activation='elu'))
model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation(custom_activation))
          
model.compile(loss='mse',optimizer='rmsprop')

model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.1, verbose=2)

y_train_hat = model.predict(X_train)
y_train_hat = np.squeeze(y_train_hat)

y_test_hat = model.predict(X_test)
y_test_hat = np.squeeze(y_train_hat)

CheckAccuracy(y_train, y_train_hat)
CheckAccuracy(y_test, y_train_hat)
CheckAccuracy(df['Call Price'], df['Black Scholes'])
