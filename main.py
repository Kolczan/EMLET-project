import sys
#put your code here
import sys
import pandas as pd
import pandas_ta as ta
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import graphviz as gv
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics
from sklearn import tree
from matplotlib.figure import Figure
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed

import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate

def grahpviz_test():
    dot = gv.Digraph(comment='The Round Table')
    dot  # doctest: +ELLIPSIS
    dot.node('A', 'King Arthur')  # doctest: +NO_EXE
    dot.node('B', 'Sir Bedevere the Wise')
    dot.node('L', 'Sir Lancelot the Brave')
    dot.edges(['AB', 'AL'])
    dot.edge('B', 'L', constraint='false')
    return dot.source

def emlet():
    infr_path = 'data/infr-metrics_vod-periodic.csv'
    data_points_infr = pd.read_csv(infr_path, sep=',')
    print("Information about infr-metrics", '\n')
    data_points_infr.info(verbose=True)
    # ax1 = data_points_infr.plot.scatter(x='Unnamed: 0', y='hndl_usr', c='DarkBlue')
    # ax2 = data_points_infr.plot.scatter(x='Unnamed: 0', y='sys_calls', c='DarkBlue')
    # ax3 = data_points_infr.plot.scatter(x='Unnamed: 0', y='idle_res', c='DarkBlue')

    service_path = "C:\\Users\\cinek\\OneDrive\\Dokumenty\\EMLET lab\\vod-periodic\\service-metrics_vod-periodic.csv"
    data_points_service = pd.read_csv(service_path, sep=',')
    print("Information about service-metrics", '\n')
    data_points_service.info(verbose=True)
    # ax4 = data_points_service.plot.scatter(x='Unnamed: 0', y='LostFrames', c='DarkBlue')
    # ax5 = data_points_service.plot.scatter(x='Unnamed: 0', y='noAudioPlayed', c='DarkBlue')
    # ax6 = data_points_service.plot.scatter(x='Unnamed: 0', y='avgInterAudioPlayedDelay', c='DarkBlue')

    # put your code here
    from sklearn.model_selection import train_test_split

    # collecting data from infr* dataset
    columns_infr = ['hndl_usr', 'sys_calls', 'soft_comp', 'idle_res']
    infr_data = data_points_infr.loc[:, columns_infr]

    # collecting data from service* dataset
    columns_service = ['noAudioPlayed']
    service_data = data_points_service.loc[:, columns_service]

    # here all the magic happens
    infr_training, infr_testset, service_training, service_testset = train_test_split(infr_data, service_data,
                                                                                      train_size=0.8, random_state=0)

    # printing section
    # print("\n Training part set from infr metrics \n")
    # print(infr_training)
    # print("\n Test set part set from infr metrics \n")
    # print(infr_testset)
    # print("\n Training part set from service metrics \n")
    # print(service_training)
    # print("\n Test set part set from infr_metrics \n")
    # print(service_testset)

    errors = []
    for depth in range(1, 10):
        dtg = DecisionTreeRegressor(max_depth=depth, random_state=0)
        # dopasowywanie modelu na bazie danych treningowych
        dtg.fit(infr_training, service_training)
        # uzyskanie predykcji modelu z danych testowych na bazie modelu wytrenowanego na danych treningowych
        prediction_result = dtg.predict(infr_testset)

        error = sklearn.metrics.mean_absolute_error(service_testset, prediction_result)
        print(f"Error result for max_depth={depth}: {error}")
        errors.append(error)

    depths = range(1, 10)
    # Plotting
    plt.figure(figsize=(16, 6))
    plt.plot(depths, errors, marker='o', linestyle='-')
    plt.title('Mean Absolute Error vs. Max Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(depths)
    plt.grid(True)
    plt.show()


    dot_data = tree.export_graphviz(dtg, out_file='data/test.txt', filled=True)
    graph = gv.Source(dot_data, format="png")
    # Assuming you have your infrastructure and service metrics in pandas DataFrame format
    # Replace 'infrastructure_metrics' and 'service_metric' with your actual data
    # Assuming you have split your data into training and testing sets
    # Replace 'X_train', 'X_test', 'y_train', 'y_test' with your actual split data

    # Instantiate models
    models = {
        'RandomForest': RandomForestRegressor(random_state=0),
        'DecisionTree': DecisionTreeRegressor(random_state=0),
        'LinearRegression': LinearRegression()
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(infr_training, service_training.values.ravel())
        predictions = model.predict(infr_testset)
        mae = sklearn.metrics.mean_absolute_error(service_testset, predictions)
        results[name] = {'MAE': mae, 'Predictions': predictions}

    # Visualize results
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(service_testset, result['Predictions'], marker='o', linestyle='', label=name)
        plt.xlabel('Actual Service Metrics')
        plt.ylabel('Predicted Service Metrics')
        plt.title('Comparison of Predictions')
        plt.legend()

    plt.show()

    # Print MAE for each model
    for name, result in results.items():
        print(f"{name} MAE: {result['MAE']}")

    # Visualize Decision Tree if applicable
    # Assuming Decision Tree is one of the models
    decision_tree_model = models['DecisionTree']
    if isinstance(decision_tree_model, DecisionTreeRegressor):
        sklearn.tree.export_graphviz(decision_tree_model, out_file='tree.dot', feature_names=infr_training.columns)

def emlet_midterm(make_chart):
    # crypto_data = pd.read_csv("data/crypto/BTC.csv")
    # # Display the first few rows of the dataset
    # crypto_data.head()
    # 
    # # Summary statistics of the dataset
    # crypto_data.describe()
    # 
    # # Check for missing values
    # crypto_data.isnull().sum()
    # bitcoin_data = crypto_data[crypto_data['ticker'] == 'BTC']

    btc_price = pd.read_csv("data/crypto/BTC-USD (2014-2024).csv")
    btc_price.head()
    eth_price = pd.read_csv("data/crypto/ETH-USD (2017-2024).csv")
    eth_price.head()
    ## EDA
    sd = btc_price.iloc[0][0]
    ed = btc_price.loc[btc_price.index[-1]][0]
    print('Starting Date', sd)
    print('Ending Date', ed)

    btc_price['RSI']=ta.rsi(btc_price.Close, length=15)
    btc_price['EMAF'] = ta.ema(btc_price.Close, length=20)
    btc_price['EMAM'] = ta.ema(btc_price.Close, length=100)
    btc_price['EMAS'] = ta.ema(btc_price.Close, length=150)
    btc_price['Target'] = btc_price['Adj Close']-btc_price.Open
    btc_price['Target'] = btc_price['Target'].shift(-1)
    btc_price['TargetClass'] = [1 if btc_price.Target[i] > 0 else 0 for i in range(len(btc_price))]

    btc_price['TargetNextClose'] = btc_price['Adj Close'].shift(-1)

    btc_price.dropna(inplace=True)
    btc_price.reset_index(inplace=True)
    btc_price.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)
    data_set = btc_price.iloc[:, 0:11]  # .values
    pd.set_option('display.max_columns', None)
    sc = MinMaxScaler(feature_range=(0, 1))
    data_set_scaled = sc.fit_transform(data_set)
    print(data_set_scaled)
    # multiple feature from data provided to the model
    X = []
    # print(data_set_scaled[0].size)
    # data_set_scaled=data_set.values
    backcandles = 30
    print(data_set_scaled.shape[0])
    for j in range(8):  # data_set_scaled[0].size):#2 columns are target not X
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):  # backcandles+2
            X[j].append(data_set_scaled[i - backcandles:i, j])

    # move axis from 0 to position 2
    X = np.moveaxis(X, [0], [2])

    # Erase first elements of y because of backcandles to match X length
    # del(yi[0:backcandles])
    # X, yi = np.array(X), np.array(yi)
    # Choose -1 for last column, classification else -2...
    X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -1])
    y = np.reshape(yi, (len(yi), 1))
    # y=sc.fit_transform(yi)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    print(X)
    print(X.shape)
    print(y)
    print(y.shape)
    # split data into train test sets
    splitlimit = int(len(X) * 0.8)
    print(splitlimit)
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(y_train)



    if make_chart is True:
        plt.figure(figsize=(12, 6))
        plt.plot(btc_price['Date'], btc_price['Close'], color='r', label='Bitcoin Price')
        plt.plot(eth_price['Date'], eth_price['Close'], color='g', label='Ethereum Price')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.title('BTC and ETH price over time in log scale')
        plt.yscale('log')
        plt.xticks(btc_price['Date'][::100], rotation='vertical')
        plt.legend()
        plt.show()

if __name__ == '__main__':

    print(sys.version)
    emlet_midterm(make_chart=False)
    # emlet()
    # docik = grahpviz_test()
    # print(docik)
    # import pydot
    # (graph,) = pydot.graph_from_dot_file('somefile.dot')
    # graph.write_png('somefile.png')