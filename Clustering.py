# (0) Paquetes importantes

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
import scipy as sp
from scipy.spatial import distance
from scipy import linalg
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
#% matplotlib inline

import boto3
import argparse

def Stats_basic_Clust(aws_access_key_id, aws_secret_access_key, aws_session_token):
    bucket = "pi-alo-2021-2"
    d = datetime.strftime(datetime.today(), format='%d%m%Y') 
    #d = datetime.strftime(datetime.today() + timedelta(days=1), format='%d%m%Y') 

    # (0.1) Acceder al bucket de S3 con las session keys
    s3_client = boto3.client('s3', region_name='us-east-1',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )
    # (0.2.1) Leer index_data_2
    file_name = 'trusted/index_data_2/index_data_2 ' + d + '.csv'
    obj = s3_client.get_object(Bucket= bucket, Key= file_name)
    index_data_2 = pd.read_csv(obj['Body'], parse_dates=['Date'], index_col = 'Date')
    index_data_2['Date']= index_data_2.index
    # (0.2.2) Leer GDP
    file_name = 'raw/GDP/% of change GDP per capita' + d + '.csv'
    obj = s3_client.get_object(Bucket= bucket, Key= file_name)
    gdp = pd.read_csv(obj['Body'], index_col = 'country')
    gdp = gdp.transpose()
    gdp.index = gdp.index.astype(int)
    # (0.2.3) Leer Inflation
    file_name = 'raw/inflation rates/Inflation rates' + d + '.csv'
    obj = s3_client.get_object(Bucket= bucket, Key= file_name)
    inflation = pd.read_csv(obj['Body'], index_col = 'country')
    inflation = inflation.transpose()
    inflation.index = inflation.index.astype(int)


    # (0.2.4) Leer Population
    file_name = 'raw/Population rate/Population rate of growth' + d + '.csv'
    obj = s3_client.get_object(Bucket= bucket, Key= file_name)
    popu = pd.read_csv(obj['Body'], index_col = 'country')
    popu = popu.transpose()
    popu.index = popu.index.astype(int)
    
    # (0.2.5) Index Info
    file_name = 'trusted/indexInfo_2.csv'
    obj = s3_client.get_object(Bucket= bucket, Key= file_name)
    index_info = pd.read_csv(obj['Body'])  #index_col = 'Index')
    #print(gdp)

    # (1) Extrayendo los datos del Dataset
    #print(index_data_2)
    dif_rate = []
    for ind in list(index_data_2['Index'].unique()):
        df = index_data_2[(index_data_2['Date'].dt.year >= 1999) &
        (index_data_2['Date'].dt.year < 2021) &
        (index_data_2['Index'] == ind)
        ]
        N = int(np.ceil((df.Date.max()-df.Date.min())/np.timedelta64(1, 'Y')))+1
        bins = [df.Date.max()-pd.offsets.DateOffset(years=y) for y in range(N)][::-1]

        df_t = df.groupby(pd.cut(df.Date, bins)).last()[['Index', 'Close', 'Date']]
        df_t['Close_shift'] = df_t.Close.shift(1)
        df_t['percentage'] = df_t.apply(lambda row: (row.Close - row.Close_shift) / row.Close, axis = 1)
        dif_rate.append(df_t[['Index', 'percentage', 'Date']].iloc[1:,:].reset_index(drop=True))

    dif_rate = pd.concat((dif_rate))

    d_years = {}
    index_data_2.drop(columns=['Date'],inplace=True)
    for y in range(2000, 2021):
        r = []
        for _ in index_data_2.Index.unique():
            t = [_] + list(index_data_2[index_data_2['Index'] == _].resample('Y').median().values[y - 2000])
            
            c = index_data_2.columns.tolist()
            c = ['Index']+ c[:-1]            
            df = pd.DataFrame(data=t).transpose()
            df.columns = c
            r.append(df)
        _ = pd.concat((r)).merge(index_info[['Region', 'Index']], on='Index')
        _['Inflation_rate'] = inflation.loc[y][_.Region].values
        _['GDP'] = gdp.loc[y][_.Region].values
        _['Pop.Gth.Rate'] = popu.loc[y][_.Region].values
        _.drop(columns=['Region'], inplace=True)
        _ = _.merge(dif_rate[dif_rate['Date'].dt.year == y][['Index', 'percentage']], on='Index')
        d_years[y] = _

    # (2) Diccionario de Arrays para cada año con datos normalizados
    t_scale = {}
    for k in d_years.keys():
        _ = d_years[k].iloc[:,0]
        t_scale[k] = StandardScaler().fit_transform(d_years[k].iloc[:,1:])
        #break

    # (3) Diccionario de Dataframes para cada año con datos normalizados
    t_scale_df ={}
    for k in d_years.keys():
        t_scale_df[k] = pd.DataFrame(t_scale[k],columns=['Open','High','Low','Close','Adj Close','Volume','Infl_GR','GDP_GR','Popl_GR','Percentage'])
        t_scale_df[k].index= ['NYA', 'IXIC','HSI','000001.SS', 'GSPTSE', '399001.SZ', 'GDAXI', 'KS11', 'SSMI', 'TWII','N225', 'N100']

    # (4) Plotmatrix de una año en particular:
    """ k = 2000
    pd.plotting.scatter_matrix(t_scale_df[k])
    plt.show() """

    # (5) Diccionario de matrices de correlacion para cada año
    Corr_scale_pd = {}
    for k in d_years.keys():
        Corr_scale_pd[k] = t_scale_df[k].corr().round(2)

    # (6) Diccionario de coeficientes de correlacion multiple para cada año
    for k in d_years.keys():
        t_scale_df[k].drop(["Open","High","Low","Close"], axis = 1, inplace = True)

    R2_pd = {}
    for k in d_years.keys():
        R2_pd[k] = (1 - 1/((np.diag(t_scale_df[k].cov()))*(np.diag(np.linalg.inv(t_scale_df[k].cov()))))).round(2)

    # (7) Diccionario de Correlacion global y dependencia global para cada año
    Corr_global = {}
    Dep_global = {}
    m,n = t_scale_df[2000].shape
    for k in d_years.keys():
        Corr_global[k] = t_scale_df[k].corr()
        Dep_global[k] = (1 - (np.linalg.det(t_scale_df[k].corr())**(1/(n-1)))).round(4)

    # (8) Regresion lineal con scikit learn y OLS

    X = {}
    Y = {}
    reg = {}
    score_XY = {}
    coef_XY = {}
    interc_XY = {}
    Y_pred = {}
    dif_Y = {}
    explained_variance = {}
    mean_absolute_error = {} 
    mse = {}
    median_absolute_error = {}
    r2 = {}
    results = {}
    for k in d_years.keys():
        X[k] = np.array(t_scale_df[k][['Volume','Infl_GR','GDP_GR','Popl_GR','Percentage']])
        Y[k] = np.array(t_scale_df[k][['Adj Close']])
        # Regression with scikit learn
        reg[k] = LinearRegression().fit(X[k], Y[k])
        score_XY[k] = reg[k].score(X[k], Y[k]).round(3)
        coef_XY[k] = reg[k].coef_.round(3)
        interc_XY[k] = reg[k].intercept_.round(3)
        Y_pred[k] = reg[k].predict(X[k]).round(3)
        dif_Y[k] = (Y[k] - Y_pred[k]).round(3)
        explained_variance[k]=(metrics.explained_variance_score(Y[k], Y_pred[k])).round(4)
        mean_absolute_error[k] = (metrics.mean_absolute_error(Y[k], Y_pred[k])).round(4)
        mse[k]= (metrics.mean_squared_error(Y[k], Y_pred[k])).round(4)
        median_absolute_error[k] = (metrics.median_absolute_error(Y[k], Y_pred[k])).round(4)
        r2[k]=(metrics.r2_score(Y[k], Y_pred[k])).round(4)
        # Regression with Statsmodel OLS
        results[k] = sm.OLS(Y[k],X[k]).fit()
        
    # (8.1) Metrics con Scikit learn (function regression_results)
    
    def regression_results(y_true, y_pred):

        # Regression metrics
        explained_variance=metrics.explained_variance_score(y_true, y_pred)
        mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
        mse=metrics.mean_squared_error(y_true, y_pred) 
        #mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
        median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
        r2=metrics.r2_score(y_true, y_pred)

        #print('explained_variance: ', round(explained_variance,4))    
        #print('mean_squared_log_error: ', round(mean_squared_log_error,4))
        #print('r2: ', round(r2,4))
        #print('MAE: ', round(mean_absolute_error,4))
        #print('MSE: ', round(mse,4))
        #print('RMSE: ', round(np.sqrt(mse),4))
        
    # (8.2) Resultados de la regresion
    reg_resl = {}
    for k in d_years.keys():
        reg_resl[k] = regression_results(Y[k], Y_pred[k])

    # (8.3) 1er Dataframe para S3: 
    
    # (9) Diccionario de Autovalores, Autovectores y PCA (Z1,Z2,Z3)
    C = {}
    W = {}
    V = {}
    AuVec0 = {}
    AuVec1 = {}
    AuVec2 = {}
    Z1 = {} # Adj_close
    Z2 = {} # Volume
    Z3 = {} # Inflation
    for k in d_years.keys():
        C[k] = np.array(t_scale_df[k].cov())
        [W[k],V[k]] = np.linalg.eig(C[k])
        AuVec0[k] = (-1*(V[k][:,0])).reshape(6,1)
        AuVec1[k] = (-1*(V[k][:,1])).reshape(6,1)
        AuVec2[k] = (-1*(V[k][:,2])).reshape(6,1)
        t_scale[k] = np.array(t_scale_df[k])
        Z1[k] = (t_scale[k] @ AuVec0[k]).round(3)
        Z2[k] = (t_scale[k] @ AuVec1[k]).round(3)
        Z3[k] = (t_scale[k] @ AuVec2[k]).round(3)
        
    # (10) Clustering con Kmeans
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler

    km = KMeans(n_clusters=5)

    y_pred = {}
    Z_comp = {}
    Z_comp_df = {}
    Z_comp_df_0 = {}
    Z_comp_df_1 = {}
    Z_comp_df_2 = {}
    Z_comp_df_3 = {}
    Z_comp_df_4 = {}
    #Z_comp_df_5 = {}
    for k in d_years.keys():
        y_pred[k] = km.fit_predict(t_scale_df[k][['Adj Close','Volume','Infl_GR','GDP_GR','Popl_GR','Percentage']])
        Z_comp[k] = np.array([Z1[k],Z2[k]])
        Z_comp[k] = Z_comp[k].T.reshape(12,2)
        Z_comp_df[k] = pd.DataFrame(Z_comp[k])
        Z_comp_df[k]['cluster'] = y_pred[k]  
        Z_comp_df[k] = Z_comp_df[k].rename(columns={0:'Z1',1:'Z2'})
        Z_comp_df[k].index= ['NYA_US', 'IXIC_US','HSI_HK','000001.SS_CN', 'GSPTSE_CA', '399001.SZ_CN', 'GDAXI_EU', 'KS11_KR', 'SSMI_Swiss', 'TWII_TW','N225_JP', 'N100_EU']
        # Grafica de clusters con labels
        Z_comp_df_0[k] = Z_comp_df[k][Z_comp_df[k].cluster == 0]
        Z_comp_df_1[k] = Z_comp_df[k][Z_comp_df[k].cluster == 1]
        Z_comp_df_2[k] = Z_comp_df[k][Z_comp_df[k].cluster == 2]
        Z_comp_df_3[k] = Z_comp_df[k][Z_comp_df[k].cluster == 3]
        Z_comp_df_4[k] = Z_comp_df[k][Z_comp_df[k].cluster == 4]
        #Z_comp_df_5[k] = Z_comp_df[k][Z_comp_df[k].cluster == 5]
        
    # plotting
        """ plt.plot(Z_comp_df_0[k]['Z1'], Z_comp_df_0[k]['Z2'],'o', color='green')
        plt.plot(Z_comp_df_1[k]['Z1'], Z_comp_df_1[k]['Z2'],'o', color='blue')
        plt.plot(Z_comp_df_2[k]['Z1'], Z_comp_df_2[k]['Z2'],'o', color='red')
        plt.plot(Z_comp_df_3[k]['Z1'], Z_comp_df_3[k]['Z2'],'o', color='purple')
        plt.plot(Z_comp_df_4[k]['Z1'], Z_comp_df_4[k]['Z2'],'o', color='brown')
        #plt.plot(Z_comp_df_5[k1]['Z1'], Z_comp_df_5[k1]['Z2'],'o', color='pink')
        # naming the x axis
        plt.xlabel('Z1')
        # naming the y axis
        plt.ylabel('Z2')
        # giving a title to my graph
        plt.title('Componentes principales ')
        # labels for the points
        plt.annotate(Z_comp_df[k].index[0], (Z1[k][0], Z2[k][0]))
        plt.annotate(Z_comp_df[k].index[1], (Z1[k][1], Z2[k][1]))
        plt.annotate(Z_comp_df[k].index[2], (Z1[k][2], Z2[k][2]))
        plt.annotate(Z_comp_df[k].index[3], (Z1[k][3], Z2[k][3]))
        plt.annotate(Z_comp_df[k].index[4], (Z1[k][4], Z2[k][4]))
        plt.annotate(Z_comp_df[k].index[5], (Z1[k][5], Z2[k][5]))
        plt.annotate(Z_comp_df[k].index[6], (Z1[k][6], Z2[k][6]))
        plt.annotate(Z_comp_df[k].index[7], (Z1[k][7], Z2[k][7]))
        plt.annotate(Z_comp_df[k].index[8], (Z1[k][8], Z2[k][8]))
        plt.annotate(Z_comp_df[k].index[9], (Z1[k][9], Z2[k][9]))
        plt.annotate(Z_comp_df[k].index[10], (Z1[k][10], Z2[k][10]))
        plt.annotate(Z_comp_df[k].index[11], (Z1[k][11], Z2[k][11])) """

        # function to show the plot
        #print('Año ',k,' :')
        #plt.show()

    ### FALTA GUARDAR ESTAS GRAFICAS EN UN ARCHIVO DE AWS ###


    # (11) Spectral Clustering
    rbf_param = 7.6 # lambda hyperparameter used in the exponential equation of the distance

    Z_comp_spec_df = {}
    K_exp = {}
    D_exp = {}
    D_inv_sr = {}
    M_exp = {}
    U = {}
    Sigma = {}
    _ = {}
    Usubset = {}
    y_pred_sc = {}
    for k in d_years.keys():
        Z_comp_spec_df[k]= Z_comp_df[k][['Z1', 'Z2']]
        K_exp[k] = (np.exp(-rbf_param*distance.cdist(Z_comp_spec_df[k],Z_comp_spec_df[k],metric='sqeuclidean')))
        D_exp[k] = K_exp[k].sum(axis = 1) 
        D_inv_sr[k] = np.sqrt(1/D_exp[k])
        M_exp[k] = np.multiply(D_inv_sr[k][np.newaxis,:],np.multiply(K_exp[k],D_inv_sr[k][:,np.newaxis]))
        U[k], Sigma[k], _[k] = linalg.svd(M_exp[k], full_matrices = False, lapack_driver = 'gesvd')
        Usubset[k] = U[k][:,0:5]
        y_pred_sc[k] = KMeans(n_clusters=5).fit_predict(normalize(Usubset[k]))
        y_pred_sc[k] = y_pred_sc[k].reshape(12,1)
        Z_comp_spec_df[k] = pd.DataFrame(Z_comp_spec_df[k])
        Z_comp_spec_df[k]['cluster'] = y_pred_sc[k]  
        
        """ plt.figure(figsize = (8,4))
        plt.subplot(121)
        plt.scatter(Z_comp_spec_df[k][['Z1']],Z_comp_spec_df[k][['Z2']],s=20)
        plt.title('Unlabeled data')

        plt.subplot(122)
        plt.scatter(Z_comp_spec_df[k][['Z1']],Z_comp_spec_df[k][['Z2']],c=y_pred_sc[k],s=20)
        plt.title('Labels returned by Spectral Clustering')
        print('Año ',k,' :')
        plt.show()' """

    #(12) Clusters: 3D Plotting
    x_Z1 = {}
    y_Z2 = {}
    z_Z3 = {}
    y_pred_3D = {}
    #k = 2020
    fig = plt.figure(figsize=(20,10))
    ax = plt.axes(projection='3d')
    for k in d_years.keys():
        #fig = plt.figure(figsize=(20,10))
        #ax = plt.axes(projection='3d')
        x_Z1[k] = Z1[k].reshape(12,)
        y_Z2[k] = Z2[k].reshape(12,)
        z_Z3[k] = Z3[k].reshape(12,)
        y_pred_3D[k] = y_pred_sc[k].reshape(12,)
        # Data for three-dimensional scattered points
        #ax.plot3D(x_Z1,y_Z2, z_Z3, 'gray');
        for i in range(len(x_Z1[k])):
            ax.scatter(x_Z1[k][i,],y_Z2[k][i,], z_Z3[k][i,], color = 'b')
            #ax.scatter(x_Z1[k][i,],y_Z2[k][i,], z_Z3[k][i,], c=y_pred_3D[k][i], cmap='viridis', s=8, alpha = 0.5)
            #ax.scatter(m[i,0],m[i,1],m[i,2],color='b') 
            ax.text(x_Z1[k][i,], y_Z2[k][i,], z_Z3[k][i,], '%s' % (str(Z_comp_df[k].index[i])), size=12, zorder=1, color='k') 
        ax.set_xlabel('Z1: Adj_Close')
        ax.set_ylabel('Z2: Volume')
        ax.set_zlabel('Z3: Inflation');
        #print('Año ',k,' :')
        #plt.show() 
    
    # Extrayendo los dataframes mas importantes
    # (i) Year_Index_Cluster_km
    # df = pd.DataFrame({'year':d_years.keys, 'dic':y_pred })
    # Year_Index_Cluster_km_= df.dic.apply(pd.Series)
    # Year_Index_Cluster_km_.rename(columns={0:'NYA',1:'IXIC',2:'HSI',3:'000001.SS',4 :'GSPTSE',5:'399001.SZ',
    #                                     6:'GDAXI',7:'KS11', 8:'SSMI',9:'TWII',10:'N225',11:'N100'}, inplace=True)
    Year_Index_Cluster_km_ = []
    for k, v in Z_comp_df.items():
        v['Año'] = int(k)
        Year_Index_Cluster_km_.append(v)
    Year_Index_Cluster_km_ = pd.concat((Year_Index_Cluster_km_))
    
    # (ii) Year_Index_Cluster_sc
    y_pred_sc1 = {}
    for k in d_years.keys():
        y_pred_sc1[k]=y_pred_sc[k].reshape(12,)
    
    # df1 = pd.DataFrame({'year':d_years.keys, 'dic':y_pred_sc1})
    # Year_Index_Cluster_sc_= df1.dic.apply(pd.Series)
    # Year_Index_Cluster_sc_.rename(columns={0:'NYA',1:'IXIC',2:'HSI',3:'000001.SS',4 :'GSPTSE',5:'399001.SZ',
    #                                     6:'GDAXI',7:'KS11', 8:'SSMI',9:'TWII',10:'N225',11:'N100'}, inplace=True) 
    Year_Index_Cluster_sc_ = []
    for k, v in Z_comp_spec_df.items():
        v['Año'] = int(k)
        Year_Index_Cluster_sc_.append(v)
    Year_Index_Cluster_sc_ = pd.concat((Year_Index_Cluster_sc_))

    # (iii) Indicador Silhouette
    sil_score = {}
    label = {}
    for k in d_years.keys():
        label[k] = km.predict(t_scale_df[k])
        sil_score[k] = silhouette_score(t_scale_df[k], label[k]).round(4)
    df2= pd.DataFrame({'year':d_years.keys, 'dic':sil_score})        
    Year_Index_Silhouette_= df2.dic.apply(pd.Series)
    Year_Index_Silhouette_.rename(columns={0:'Sil_score'}, inplace=True)

    # Guardar los archivos en S3
    # Guardar los archivos en S3
    for k in range(2000,2021):
        k_year = str(k)
        route = './'
        file_name = 'd_years '+k_year+'.csv'
        d_years[k].to_csv(route + file_name)
        s3_client.upload_file(route + file_name, bucket, 'refined/clustering/d_years/{}'.format(file_name))
        
        route = './'
        file_name = 't_scale_df '+ k_year+'.csv'
        t_scale_df[k].to_csv(route + file_name)
        s3_client.upload_file(route + file_name, bucket, 'refined/clustering/t_scale/{}'.format(file_name))
    route = './'
    file_name = 'Year_Index_Cluster_km_.csv'
    Year_Index_Cluster_km_.to_csv(route + file_name)
    s3_client.upload_file(route + file_name, bucket, 'refined/clustering/{}'.format(file_name)) 
    
    file_name = 'Year_Index_Cluster_sc_.csv'
    Year_Index_Cluster_sc_.to_csv(route + file_name)
    s3_client.upload_file(route + file_name, bucket, 'refined/clustering/{}'.format(file_name)) 
    
    file_name = 'Year_Index_Silhouette_.csv'
    Year_Index_Silhouette_.to_csv(route + file_name)
    s3_client.upload_file(route + file_name, bucket, 'refined/clustering/{}'.format(file_name))   
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

Stats_basic_Clust(args.key1,args.key2,args.key3)
