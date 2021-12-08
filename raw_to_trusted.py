import pandas as pd
from datetime import datetime
import boto3
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
    
    bucket = "pi-alo-2021-2"
    d = datetime.strftime(datetime.today(), format='%d%m%Y')
    file_name = 'raw/IndexData/indexData ' + d + '.csv'
    obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
    index_data = pd.read_csv(obj['Body'], converters={'Date':pd.to_datetime})
    I_name = index_data['Index'].unique().tolist()
    conversion_dict = {"NYA" : 1.0, "IXIC" : 1.0, "HSI" : 0.13, "000001.SS" : 0.15, "N225" : 0.0090, "N100" : 1.17,
     "399001.SZ" : 0.15,"GSPTSE" : 0.79, "NSEI" : 0.014, "GDAXI" : 1.17, "KS11" : 0.00085, "SSMI" : 1.08, "TWII" : 0.036,
     "J203.JO" : 0.067}

    conversion_list = []
    for i in I_name:
        temp = index_data.copy()[index_data['Index'] == i]
        temp["Open"] = temp["Open"] * conversion_dict[i]
        temp["High"] = temp["High"] * conversion_dict[i]
        temp["Low"] = temp["Low"] * conversion_dict[i]
        temp["Close"] = temp["Close"] * conversion_dict[i]
        temp["Adj Close"] = temp["Adj Close"] * conversion_dict[i]
        conversion_list.append(temp)

    index_data_processed = pd.concat((conversion_list))
    route = './'
    file_name_proc = 'index_processed ' + d + '.csv'
    index_data_processed.to_csv(route +file_name_proc, index=False)
    s3_client.upload_file('./' + file_name_proc, bucket, 'trusted/index_processed/{}'.format(file_name_proc))
    print('UPLOADED' + file_name_proc)
    index_data_1 = index_data_processed[(index_data_processed['Index'] != 'J203.JO') &
        (index_data_processed['Index'] != 'NSEI')]

    fecha_corte_min = datetime(1999, 1, 1)
    index_data_2 = index_data_1[index_data_1['Date'] >= fecha_corte_min]
    index_data_2.set_index('Date', drop=True, inplace=True)
    print(index_data_2)
    file_name_proc = 'index_data_2 ' + d + '.csv'
    index_data_2.to_csv(route + file_name_proc)

    s3_client.upload_file(route + file_name_proc, bucket, 'trusted/index_data_2/{}'.format(file_name_proc))
    print('UPLOADED' + file_name_proc)

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