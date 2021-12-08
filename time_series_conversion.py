import pandas as pd
import numpy as np
import calendar
from datetime import datetime, timedelta
from dateutil.relativedelta import *

import boto3
import botocore.exceptions
import argparse

def convert_files(aws_access_key_id, aws_secret_access_key, aws_session_token):
    bucket = "pi-alo-2021-2"

    #Access S3 bucket with the session keys
    s3_client = boto3.client('s3', region_name='us-east-1',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )

    #Read index_data_2
    for days in np.arange(0, 30):
        d = datetime.strftime(datetime.today() - relativedelta(days = int(days)), format='%d%m%Y')
        file_name = 'trusted/index_data_2/index_data_2 ' + d + '.csv'
        try:
            obj = s3_client.get_object(Bucket= bucket, Key= file_name)
            index_data = pd.read_csv(obj['Body'], parse_dates=['Date'], index_col = 'Date')
            index_data = index_data[['Index', 'Close']] #Keep only Close data
            index_data.dropna(inplace = True)
            if np.all(index_data.groupby(['Index']).count() >= 360) == True: #Check is there's enough data for each index
                break
        except botocore.exceptions.ClientError as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':
                continue

    #Select date range to start on 2000-1-1
    index_data = index_data[index_data.index >= datetime(2000,1,1)]

    #month_close_index: Close of the last day of the month for each index
    month_close_index = pd.DataFrame(columns = ['Date','Index','Close'])

    #month_mean_index: Mean of the close values for each month
    month_mean_index = pd.DataFrame(columns = ['Date','Index','Close'])

    #month_median_index: Median of the close values for each month
    month_median_index = pd.DataFrame(columns = ['Date','Index','Close'])

    #month_change_index: Percentage of change with respect to previous month
    month_change_index = pd.DataFrame(columns = ['Date','Index','Close'])

    for ind in index_data.Index.unique():
        #Array of tuples with each year and each month present in the dataset
        year_month = pd.Series([(d.year, d.month) for d in index_data[index_data.Index == ind].index.unique()]).unique()

        max_dates = []
        max_dates = [np.max(index_data[(index_data.index.year == y) & (index_data.index.month == m) & (index_data.Index == ind)].index) for (y, m) in year_month]
        month_close = [{'Date': datetime(d.year, d.month, 1), 'Index' : ind, 'Close': index_data[(index_data.index == d) & (index_data.Index == ind)].Close[0]} for d in max_dates]
        month_close_index = month_close_index.append(month_close)

        month_means = [{'Date': datetime(y, m, 1), 'Index': ind, 'Close' : np.mean(index_data[(index_data.index.year == y) & (index_data.index.month == m) & (index_data.Index == ind)].Close)} for (y, m) in year_month]
        month_mean_index = month_mean_index.append(month_means)

        month_medians = [{'Date': datetime(y, m, 1), 'Index': ind,'Close' : np.median(index_data[(index_data.index.year == y) & (index_data.index.month == m) & (index_data.Index == ind)].Close)} for (y, m) in year_month]
        month_median_index = month_median_index.append(month_medians)

    month_close_index.set_index('Date', inplace = True)

    for ind in month_close_index.Index.unique():
        for date in month_close_index.index.unique():
            if (date.year != 2000 or date.month != 1):
                previous_month = month_close_index[(month_close_index.index == date - relativedelta(months = 1)) & (month_close_index.Index == ind)].Close[0]
                current_month = month_close_index[(month_close_index.index == date) & (month_close_index.Index == ind)].Close[0]

                month_change = {'Date': date, 'Index': ind, 'Close' : 1 - current_month / previous_month}
                month_change_index = month_change_index.append(month_change, ignore_index=True)

    #Set index to be the dates
    month_mean_index.set_index('Date', inplace = True)
    month_median_index.set_index('Date', inplace = True)
    month_change_index.set_index('Date', inplace = True)

    #Save files to S3
    route = './'
    file_name = 'month_close_index.csv'
    month_close_index.to_csv(route + file_name)
    s3_client.upload_file('./' + file_name, bucket, 'refined/time_series/month_close/{}'.format(file_name))   

    file_name = 'month_mean_index.csv'
    month_mean_index.to_csv(route + file_name)
    s3_client.upload_file('./' + file_name, bucket, 'refined/time_series/month_mean/{}'.format(file_name))

    file_name = 'month_median_index.csv'
    month_median_index.to_csv(route + file_name)
    s3_client.upload_file('./' + file_name, bucket, 'refined/time_series/month_median/{}'.format(file_name))  

    file_name = 'month_change_index.csv'
    month_change_index.to_csv(route + file_name)
    s3_client.upload_file('./' + file_name, bucket, 'refined/time_series/month_change/{}'.format(file_name))

#Convert index_data_2 to represent each month of the time series
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

    convert_files(args.key1, args.key2, args.key3)