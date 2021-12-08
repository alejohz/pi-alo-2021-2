# # Análisis de Series de Tiempo

# Este Notebook permite realizar un análisis completo de la evolución en el tiempo de los 12 índices financieros estudiados. Este contiene las siguientes secciones:
# 
# 1. Análisis
# 2. Predicción

# # Preparativos


# ## Importar librerías

import pandas as pd
import numpy as np
import calendar
from datetime import datetime, timedelta
from dateutil.relativedelta import *

import warnings
warnings.filterwarnings("ignore")

import boto3
import botocore.exceptions
import argparse

#Initialize bucket
bucket = "pi-alo-2021-2"

# ## Funciones Útiles

#Lectura de Archivos desde S3
def get_file(s3_client, file_name):
    obj = s3_client.get_object(Bucket= bucket, Key= file_name)
    df = pd.read_csv(obj['Body'], parse_dates=['Date'], index_col = 'Date')
    df = df[['Index', 'Close']] #Keep only Close data
    df.dropna(inplace = True)

    return df

#Subir archivo a S3
def upload_file(s3_client, file_name, route, data):
    data.to_csv('./' + file_name)
    s3_client.upload_file('./' + file_name, bucket, route)

""" #Graficar serie de tiempo
def plot_timeseries(x, y, title = "", xlabel = "Date", ylabel = "Value", xlim = (datetime(2000,1,1),datetime(2020,12,31)), legend = ""):
    plt.figure(figsize= (20,10))
    plt.plot(x, y)
    plt.gca().set(title = title, xlabel = xlabel, ylabel = ylabel, xlim = xlim)
    plt.grid(True)
    plt.legend(legend)
    plt.show()

#Múltiples series por gráfica
def plot_mult_timeseries(x, y, criteria, title = "", xlabel = "Date", ylabel = "Value", legend = ""):
    plt.figure(figsize= (20,10))
    for i, j in enumerate(criteria.unique()):
        plt.plot(x[criteria == j], y[criteria == j])
    plt.gca().set(title = title, xlabel = xlabel, ylabel = ylabel)
    plt.grid(True)
    plt.legend(legend)
    plt.show()

#Boxplots
def boxplot_timeseries(x, y, title = "", xlabel = "Date", ylabel = "Value", legend = ""):
    plt.figure(figsize=(20,10))
    ax = sns.boxplot(x = x, y = y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show() """

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

def decompose_ts(ind, data, type):
    warnings.filterwarnings("ignore")
    if type == 'add':
        dec = seasonal_decompose(data.asfreq('MS'), model = 'additive',  extrapolate_trend='freq')
    elif type == 'mult':
        dec = seasonal_decompose(data.asfreq('MS'), model = 'multiplicative',  extrapolate_trend='freq')
    dec = pd.concat([dec.seasonal, dec.trend, dec.resid, dec.observed], axis = 1)

    dec.columns = ['Seasonal','Trend','Residuals','Close']
    dec['Index'] = ind
    return dec

def diferenciar(data, n_diferenciaciones):
    warnings.filterwarnings("ignore")
    data_diff = data
    if n_diferenciaciones == 0:
        data_diff = data
    for i in np.arange(n_diferenciaciones):
        data_diff = data_diff.diff()
    
    return data_diff.dropna()

def stationary_tests(data, test):
    import warnings
    warnings.filterwarnings("ignore")
    if test == 'ADF':
        p = adfuller(data.values)[1]
    elif test == 'KPSS':
        p = kpss(data.values)[1]
    return p

# #### Minimización del MSE
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def fit_arima_model(data, arima_order, train_size):
    warnings.filterwarnings("ignore")
    size = int(train_size*len(data))
    n_forecasts = len(data) - 1

    train, test = data[0:size], data[size:]
    train = train.asfreq("MS")

    model = ARIMA(train, order = arima_order, enforce_stationarity = False).fit()

    predictions = model.predict(start = size, end = n_forecasts, typ = 'linear')

    error = np.sqrt(mean_squared_error(predictions, test.values))

    return {'model' : model, 'train' : train, 'test' : test, 'predictions' : predictions, 'error' : error}

def evaluate_models(dataset, train_size, p_values, d_values, q_values, ind):
    warnings.filterwarnings("ignore")
    results = pd.DataFrame(columns = ['p','d','q','train_size', 'MSE'])
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for t in train_size:
                    order = (p,d,q)
                    mse = fit_arima_model(dataset, order, t)["error"]
                    results = results.append({'p':p,'d':d,'q':q,'train_size':t,'MSE':mse}, ignore_index = True)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    #print('ARIMA%s MSE=%.3f' % (order,mse))
    #print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    results['Index'] = ind
    return results

def final_arima_model(data, arima_order, train_size, ind):
    warnings.filterwarnings("ignore")
    size = int(train_size*len(data))

    train, test = data[0:size], data[size:]
    train = train.asfreq("MS")

    model = ARIMA(train, order = arima_order, enforce_stationarity = False).fit()

    predictions = model.predict(start = np.min(data.index), end = np.max(data.index) + relativedelta(months = 12), typ = 'linear')
    
    results = pd.DataFrame(columns = ['Close', 'Prediction'])
    results = pd.concat([data.rename('Close'), predictions.rename('Prediction')], axis = 1)
    results['Index'] = ind
   
    return results

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

def get_MSEs(data, train_size, ind, best_arimas):
    warnings.filterwarnings("ignore")
    size = int(train_size*len(data))

    train, test = data[0:size], data[size:]
    
    #ARIMA
    MSE_arima = best_arimas[best_arimas.Index == ind].MSE.values[0]

    #Simple Exp Smoothing
    model = SimpleExpSmoothing(train, initialization_method = 'estimated').fit()
    predictions = model.predict(start = np.min(test.index), end = np.max(test.index))
    MSE_exps = np.sqrt(mean_squared_error(predictions, test.values))
    
    #Exponential Holt
    model = Holt(train, initialization_method = 'estimated', exponential = True).fit()
    predictions = model.predict(start = np.min(test.index), end = np.max(test.index))
    MSE_Holt_Exp = np.sqrt(mean_squared_error(predictions, test.values))

    #Holt Damped Trend
    model = Holt(train, initialization_method = 'estimated', damped_trend = True).fit()
    predictions = model.predict(start = np.min(test.index), end = np.max(test.index))
    MSE_Holt_Damped_Trend = np.sqrt(mean_squared_error(predictions, test.values))

    MSEs = pd.DataFrame(columns = ['Index', 'MSE_Arima', 'MSE_Simple_Exp', 'MSE_Holt_Exp', 'MSE_Holt_Damped_Trend'])
    MSEs = MSEs.append({'Index' : ind, 'MSE_Arima' : MSE_arima, 'MSE_Simple_Exp' : MSE_exps, 'MSE_Holt_Exp' : MSE_Holt_Exp, 'MSE_Holt_Damped_Trend' : MSE_Holt_Damped_Trend}, ignore_index = True)
    
    return MSEs

def fit_exp_model(data, train_size, ind, method = 'simple'):
    warnings.filterwarnings("ignore")
    size = int(train_size*len(data))

    train, test = data[0:size], data[size:]
    
    if method == 'simple':
        model = SimpleExpSmoothing(train, initialization_method = 'estimated').fit()
    elif method == 'holt_exponential':
        model = Holt(train, initialization_method = 'estimated', exponential = True).fit()
    elif method == 'holt_damped_trend':
        model = Holt(train, initialization_method = 'estimated', damped_trend = True).fit()

    predictions = model.predict(start = np.min(data.index), end = np.max(data.index) + relativedelta(months = 12))

    results = pd.DataFrame(columns = ['Close', 'Prediction'])
    results = pd.concat([data.rename('Close'), predictions.rename('Prediction')], axis = 1)
    results['Index'] = ind

    return results

# Para el análisis, se puede elegir entre las siguientes maneras de representar los datos de los índices diarios:
# 
# 1. Cierre de mes: month_close_index
# 2. Media de los cierres del mes: month_mean_index
# 3. Mediana de los cierres del mes: month_median_index
# 4. Porcentaje de crecimiento con respecto al mes anterior: month_grow_index

def analyze_time_series(aws_access_key_id, aws_secret_access_key, aws_session_token):
    #Initialize boto3.client
    s3_client = boto3.client('s3', region_name='us-east-1',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )    

    #Read files
    d = datetime.strftime(datetime.today(), format='%d%m%Y')
    
    datasets = {}
    files = ['month_close']#,'month_mean','month_median','month_change']
    for file in files:
        file_name = 'refined/time_series/' + file + '/' + file + '_index.csv'
        datasets[file] = get_file(s3_client, file_name)
    
    # Se crea una tabla pivot para tener cada índice en una columna del DataFrame
    data_pivots = {file : df.pivot(columns = 'Index', values = 'Close') for (file, df) in datasets.items()}
    [upload_file(s3_client, file, 'refined/time_series/' + file + '/' + file + '_pivot.csv', df) for (file, df) in data_pivots.items()]
    
    # Relación de Sharpe
    SRs = {file : (np.mean(df)/np.std(df)).sort_values(ascending = False) for (file, df) in data_pivots.items()}
    [upload_file(s3_client, file, 'refined/time_series/' + file + '/' + file + '_SR.csv', df) for (file, df) in SRs.items()]

    #Time series decomposition
    #Additive
    add_decs = {file : pd.concat(decompose_ts(ind, df[ind], 'add') for ind in df.columns) for (file, df) in data_pivots.items()}
    [upload_file(s3_client, file, 'refined/time_series/' + file + '/' + file + '_add_dec.csv', df) for (file, df) in add_decs.items()]

    #Multiplicative
    mult_decs = {file : pd.concat(decompose_ts(ind, df[ind], 'mult') for ind in df.columns) for (file, df) in data_pivots.items() if file != 'month_change'}
    [upload_file(s3_client, file, 'refined/time_series/' + file + '/' + file + '_mult_dec.csv', df) for (file, df) in mult_decs.items()]

    #Modelo ARIMA

    #Parámetro 'd':
    test_names = ['ADF', 'KPSS']
    tests = {file : pd.DataFrame({'Index' : ind, 'Difference' : d, 'Test' : t, 'P_Value' : stationary_tests(diferenciar(df[ind], d), t)} for ind in df.columns for d in np.arange(0, 11) for t in test_names) for (file, df) in data_pivots.items()}
    [upload_file(s3_client, file, 'refined/time_series/' + file + '/' + file + '_stationary_tests.csv', df) for (file, df) in tests.items()]

    # ### Grid-Search para encontrar los parámetros
    p_values = [0, 1, 2, 4, 6, 8, 10]
    d_values = np.arange(3)
    q_values = np.arange(3)
    train_size = [0.9]

    arimas = {file : pd.concat(evaluate_models(df[ind], train_size, p_values, d_values, q_values, ind) for ind in df.columns) for (file, df) in data_pivots.items()}
    [upload_file(s3_client, file, 'refined/time_series/' + file + '/' + file + '_arimas.csv', df) for (file, df) in arimas.items()]
    
    # ### Modelo Final    
    #Se selecciona el modelo con el menor error cuadrático medio:
    best_arimas = {file : pd.concat(df[(df.Index == ind) & (df.MSE == np.min(df[df.Index == ind].MSE))] for ind in df.Index.unique()) for (file, df) in arimas.items()}

    arima_predictions = {file : pd.concat(final_arima_model(data_pivots[file][ind], (df[df.Index == ind].p.values[0],df[df.Index == ind].d.values[0],df[df.Index == ind].q.values[0]), 1, ind) for ind in df.Index.unique()) for (file, df) in best_arimas.items()}
    [upload_file(s3_client, file, 'refined/time_series/' + file + '/' + file + '_arima_predictions.csv', df) for (file, df) in arima_predictions.items()]
    
    #Suavizado Exponencial Simple
    exps_predictions = {file : pd.concat(fit_exp_model(df[ind],1, ind, 'simple') for ind in df.columns) for (file, df) in data_pivots.items()}
    [upload_file(s3_client, file, 'refined/time_series/' + file + '/' + file + '_exponential_simple_predictions.csv', df) for (file, df) in exps_predictions.items()]

    #Método de Holt
    
    #Exponencial
    holt_exp_predictions = {file : pd.concat(fit_exp_model(df[ind],1, ind, 'holt_exponential') for ind in df.columns) for (file, df) in data_pivots.items() if file != 'month_change'}
    [upload_file(s3_client, file, 'refined/time_series/' + file + '/' + file + '_holt_exp_predictions.csv', df) for (file, df) in holt_exp_predictions.items()]
    
    #Tendencia Amortiguada
    holt_damped_trend_predictions = {file : pd.concat(fit_exp_model(df[ind],1, ind, 'holt_damped_trend') for ind in df.columns) for (file, df) in data_pivots.items() if file != 'month_change'}
    [upload_file(s3_client, file, 'refined/time_series/' + file + '/' + file + '_holt_damped_trend_predictions.csv', df) for (file, df) in holt_damped_trend_predictions.items()]

    #MSE para cada modelo
    MSEs = {file : pd.concat(get_MSEs(df[ind],0.9, ind, best_arimas[file]) for ind in df.columns) for (file, df) in data_pivots.items() if file != 'month_change'}
    [upload_file(s3_client, file, 'refined/time_series/month_close/month_close_MSEs.csv', df) for (file, df) in MSEs.items()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='List the three keys')
    parser.add_argument(
        '--key1', default=None, type=str,
        help="AWS access key id.")
    parser.add_argument(
        '--key2', default=None, type=str,
        help="AWS access secret key id.")
    parser.add_argument(
        '--key3', default=None, type=str,
        help="AWS access sesion key id.")
    args = parser.parse_args()

    analyze_time_series(args.key1, args.key2, args.key3)