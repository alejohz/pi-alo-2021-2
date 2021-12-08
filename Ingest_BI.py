import pandas as pd
import boto3


aws_access_key_id='ASIAWN43HLCGZTO5ARXR'
aws_secret_access_key='p1MP/BqIJkShtuUPx+1LBMm/NW0srKAgOnUVq8T+'
aws_session_token='FwoGZXIvYXdzEPj//////////wEaDKstV8hID1dXnvUppiKuAafa+ZgR3nDN73UCc9n9lkqZCVE2kgpAIBrCxmLF5aOEoYyMVGv8bWauOIujGGDSoZ4NQyVcYUyJGGgVZgYQX/sXh+dT7hplUrhVHZ9ox7UNLttTtyVaS13lOiYPevlWUmYKow/c68pZ+IEMXmGVRh5F9qYn0Oh2YFaDzyae5aZm4yK5DLfkFqNAbkk9Z31LvxK+++LohpR/7pkk+bC5vs/fy58ugwcmsJ2cLXiOhyithMONBjItfQapDDYrn0p/v8a+A4tb77V8/33rpggJ13qyeygG38qNi9ZVzV30kSWxXmV6'

s3_client = boto3.client('s3', region_name='us-east-1',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )
bucket='pi-alo-2021-2'
type_a = 'month_close'
file_name = f"refined/time_series/{type_a}/{type_a}_add_dec.csv"
obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
add_dec = pd.read_csv(obj['Body'])
print(add_dec)
file_name = f"refined/time_series/{type_a}/{type_a}_arima_predictions.csv"
obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
arima_predictions = pd.read_csv(obj['Body'])
print(arima_predictions)
file_name = f"refined/time_series/{type_a}/{type_a}_best_arimas.csv"
obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
best_arimas = pd.read_csv(obj['Body'])
print(best_arimas)
file_name = f"refined/time_series/{type_a}/{type_a}_exponential_simple_predictions.csv"
obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
exponential_simple_predictions = pd.read_csv(obj['Body'])
print(exponential_simple_predictions)
file_name = f"refined/time_series/{type_a}/{type_a}_index.csv"
obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
index = pd.read_csv(obj['Body'])
print(index)
file_name = f"refined/time_series/{type_a}/{type_a}_SR.csv"
obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
SR = pd.read_csv(obj['Body'])
print(SR)
file_name = f"refined/time_series/{type_a}/{type_a}_arimas.csv"
obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
arimas = pd.read_csv(obj['Body'])
print(arimas)
file_name = f"refined/time_series/{type_a}/{type_a}_MSEs.csv"
obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
MSE = pd.read_csv(obj['Body'])
print(MSE)
file_name = "trusted/indexInfo_2.csv"
obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
indexInfo = pd.read_csv(obj['Body'])
print(indexInfo)
bucket='pi-alo-2021-2'
type_a = 'clustering'
file_name = f"refined/{type_a}/Year_Index_Cluster_km_.csv"
obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
km = pd.read_csv(obj['Body'])
print(km)
file_name = f"refined/{type_a}/Year_Index_Cluster_sc_.csv"
obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
sc = pd.read_csv(obj['Body'])
print(sc)
file_name = f"refined/{type_a}/Year_Index_Silhouette_.csv"
obj = s3_client.get_object(Bucket= bucket, Key= file_name) 
Silhouette = pd.read_csv(obj['Body'])
print(Silhouette)
