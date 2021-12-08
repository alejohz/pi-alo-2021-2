import pandas as pd
from datetime import datetime, timedelta
import boto3
import time
import wbdata
import argparse

def convert_file(aws_access_key_id, aws_secret_access_key, aws_session_token):
    #aws_access_key_id='ASIAWN43HLCG6J6AFO23'
    #aws_secret_access_key='XAvdfyKgdzt0AFXJ7zG7HBVj8aonCkKA72TG+Xy9'
    #aws_session_token='FwoGZXIvYXdzEOD//////////wEaDNfpj8EYYt827XQw4iKuAaXf3PJAxVRDnGf9W0mzhptrItjWIrgE14Peo4tS8utU7MPiVzDN+ZqJ7zRVRu7igulNVseR/aT0SE4ncqdLnHs5JuKcCJkGydpJtFVmL33k28IXJ6O3P8/XlGxts9c71Aa8iKszK/Hdn9FEkNsNlmdThVBMZ1IR+WweVvzCJcMOmya1g6D2EgWuHyeTV2pTjI4lwf9zB19KKpi1X/05Vhp7maWFCn2+4oRFbsVOeSiExIWNBjIt3exCIOQgp5v849vxyvDFD5RI/aCM25Wz8Zp9mBeSj+ECWs5dQKD9nt/nfoJ2'
    s3_client = boto3.client('s3', region_name='us-east-1',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )

    # YAHOO FINANCE
    index_names = {'IXIC' : '%5EIXIC',
    'NYA' : '%5ENYA',
    'HSI' : '%5EHSI',
    '000001.SS' : '000001.SS',
    'N225' : '%5EN225',
    'N100' : '%5EN100',
    '399001.SZ' : '399001.SZ',
    'GSPTSE' : '%5EGSPTSE',
    'GDAXI' : '%5EGDAXI',
    'KS11' : '%5EKS11',
    'SSMI' : '%5ESSMI',
    'TWII' : '%5ETWII'
    }
    bucket = "pi-alo-2021-2"
    d = datetime.strftime(datetime.today(), format='%d%m%Y')
    period1 = int(time.mktime(datetime(1999, 1, 1, 0, 0).timetuple()))
    period2 = int(time.mktime((datetime.today() - timedelta(days=1)).timetuple()))
    interval = '1d' #1wk, 1m
    dfs = []
    for key, value in index_names.items():
        query_string= f"https://query1.finance.yahoo.com/v7/finance/download/{value}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true"
        df = pd.read_csv(query_string)
        df['Index'] = key
        dfs.append(df)
    route = './'
    file_name = 'indexData ' + d + '.csv'
    dfs = pd.concat((dfs))  
    dfs.to_csv(route + file_name, index=False)
    s3_client.upload_file(route + file_name, bucket, 'raw/IndexData/{}'.format(file_name))
    print('UPLOADED ' + file_name)
    ## WORLD BANK DATA
    countries = ["USA","HKG","CHN", "JPN", "EUU", "CAN", "IND", "DEU", "KOR", "CHE", "ZAF"]
    indicators = {'SP.POP.GROW':'Population rate of growth', 
    'NY.GDP.PCAP.KD.ZG' : '% of change GDP per capita',
    'FP.CPI.TOTL.ZG' : 'Inflation rates',
    }
    dfs = []
    for key, value in indicators.items():
        df = wbdata.get_dataframe({key: value}, country=countries, convert_date=False)
        dfs.append(df)
    dfs = pd.concat((dfs), axis=1)

    file_name = "trusted/indexInfo_2.csv"
    obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
    index_info = pd.read_csv(obj['Body'])
    for i in range(0, 3):
        a = dfs.iloc[:,i].reset_index(level=1).loc[index_info.Region.tolist()]
        a['date'] = a['date'].astype('int')
        a = a[a['date'] >= 1999]
        if 'Population' in a.columns[1]:
            a_1 = a.pivot_table(values = a.columns[1], columns='date', index = a.index)
            file_name = str(a.columns[1]) + d + '.csv'
            a_1.to_csv(route + file_name)   
            s3_client.upload_file(route + file_name, bucket, 'raw/Population rate/{}'.format(file_name))
        if 'GDP' in a.columns[1]:
            a_1 = a.pivot_table(values = a.columns[1], columns='date', index = a.index)
            file_name = str(a.columns[1]) + d + '.csv'
            a_1.to_csv(route + file_name)
            s3_client.upload_file(route + file_name, bucket, 'raw/GDP/{}'.format(file_name))
        if 'Inflation' in a.columns[1]:
            a_1 = a.pivot_table(values = a.columns[1], columns='date', index = a.index)
            file_name = str(a.columns[1]) + d + '.csv'
            a_1.to_csv(route + file_name)
            s3_client.upload_file(route + file_name, bucket, 'raw/inflation rates/{}'.format(file_name))
    print('UPLOADED ' + file_name)

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

    convert_file(args.key1, args.key2, args.key3)