import wbdata
import pandas as pd
import argparse

def printing_hello_word(word, number):
    countries = ["USA","HKG","CHN", "JPN", "EUU", "CAN", "IND", "DEU", "KOR", "CHE", "ZAF"]
    #countries = ["CL","UY","HU"]
    indicators = {'SP.POP.GROW':'Population rate of growth', 
        'NY.GDP.PCAP.KD.ZG' : '% of change GDP per capita',
        'FP.CPI.TOTL.ZG' : 'Inflation rates',
        }
    '''
    dfs = []
    for key, value in indicators.items():
        df = wbdata.get_dataframe({key: value}, country=countries, convert_date=False)
        dfs.append(df)
    dfs = pd.concat((dfs), axis=1)
    aws_access_key_id='ASIAWN43HLCG5NVS4LYU'
    aws_secret_access_key='CkZJ0zN1HgUS+wa9fAbcJjGZAQgBdj/cDoiwzWqx'
    aws_session_token='FwoGZXIvYXdzEJz//////////wEaDA/TxfCUoUcdpr1f5iKuAYKaeEhWP6sITg3OORmhU8Z72gQwzG+tLg7qhUOITovrvB4VHa2Dawgd9kHz46YribivjRQlsyFDClhZXTmStxh/3YTi7WH53K3wErXRHzIcSSYXpnIqSq7C2mhXSuaXJzMpUBMyyGVJKBFqH2VlFRt3pBa2nofbO0jmFkwcR8zsOzR5b7q/lriPUbDJ6LmhlQ6wWRbZqRYPx9kTrQFmcwdC3i8IV0tqSVweniaEeiigwfaMBjItnLI9m0JQ6SR1j4jcsuM3TKA/bIesn1fIbKWMOz+Qxgci5926r4/WoFLJIHy+'
    import boto3
    s3 = boto3.client('s3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
            )
    '''
    print('hello world')
    bucket = "pi-alo-2021-2"
    file_name = "trusted/indexInfo_2.csv"
    '''
    obj = s3.get_object(Bucket= bucket, Key= file_name) 
    index_info = pd.read_csv(obj['Body'])
    a = dfs.iloc[:,2].reset_index(level=1).loc[index_info.Region.tolist()]
    a['date'] = a['date'].astype('int')
    a = a[a['date'] >= 2000]
    a.pivot_table(values = a.columns[1], columns='date', index = a.index)
    '''
    print(bucket)
    print(word)
    print(number + 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ESCRIBA LA PALABRA')
    parser.add_argument(
        '--word', default=None, type=str,
        help="Palabra que quiere sacar en consola")
    parser.add_argument(
        '--number', default=1, type=int,
        help="Número que quiere sumar con 1")
    args = parser.parse_args()

    printing_hello_word(args.word, args.number)