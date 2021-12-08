import boto3
import argparse

def run_cluster(aws_access_key_id, aws_secret_access_key, aws_session_token):
    client = boto3.client('emr',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name='us-east-1'
        )

    bucket_name = 'pi-alo-2021-2'
    script_file_name = 'extract_files.py'
    script_key = f'scripts/{script_file_name}'
    install_step = {
            'name': 'extract_files',
            'script_uri': f's3://{bucket_name}/{script_key}',
            'script_args':
                '--key1 '+ aws_access_key_id + ' --key2 '+
                aws_secret_access_key + ' --key3 '+ aws_session_token,
            'script_file_name' : script_file_name
    }
    script_file_name = 'raw_to_trusted.py'
    script_key = f'scripts/{script_file_name}'
    convert_step = {
            'name': 'raw_to_trusted_to_refined',
            'script_uri': f's3://{bucket_name}/{script_key}',
            'script_args':
                '--key1 '+ aws_access_key_id + ' --key2 '+
                aws_secret_access_key + ' --key3 '+ aws_session_token,
            'script_file_name' : script_file_name
    }

    script_file_name = 'Clustering.py'  
    script_key = f'scripts/{script_file_name}'
    cluster_step = {
            'name': 'Clustering',
            'script_uri': f's3://{bucket_name}/{script_key}',
            'script_args':
                '--key1 '+ aws_access_key_id + ' --key2 '+
                aws_secret_access_key + ' --key3 '+ aws_session_token,
            'script_file_name' : script_file_name
    }
    steps = [install_step, convert_step, cluster_step]

    cluster_id = client.run_job_flow(
        Name='ETL_emr_job_boto3',
        LogUri='s3://pi-alo-2021-2/logs',
        ReleaseLabel='emr-6.5.0',
        Applications=[
            {
                'Name': 'JupyterHub'
            },
        ],
        Instances={
            'InstanceGroups': [
                {
                    'Name': "Master nodes",
                    'Market': 'SPOT',
                    'InstanceRole': 'MASTER',
                    'InstanceType': 'm4.xlarge',
                    'InstanceCount': 1,
                },
                {
                    'Name': "Slave nodes",
                    'Market': 'SPOT',
                    'InstanceRole': 'CORE',
                    'InstanceType': 'm4.xlarge',
                    'InstanceCount': 2,
                }
            ],
            'Ec2KeyName': 'ahenao-key',
            'KeepJobFlowAliveWhenNoSteps': False,
            'Ec2SubnetId': 'subnet-2053b621',
            'TerminationProtected': False,
        },
        BootstrapActions=[
            {
                'Name': 'install_libraries',
                'ScriptBootstrapAction': {
                    'Path': 's3://pi-alo-2021-2/scripts/install_libraries.sh',
                }
            }
        ],
        Steps=[{
                'Name': step['name'],
                'ActionOnFailure': 'CONTINUE',
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': ['bash',
                    '-c',
                    'aws s3 cp {0} . ; python3 -m pip install --upgrade numpy;'
                    ' python3 ./{1} {2}'
                    .format(step['script_uri'], step['script_file_name'], step['script_args'])]
                    }} for step in steps],
        VisibleToAllUsers=True,
        ServiceRole = 'EMR_DefaultRole',
        JobFlowRole = 'EMR_EC2_DefaultRole',
    )

    print ('cluster created with the step...', cluster_id['JobFlowId'])

    response = client.describe_cluster(
        ClusterId=cluster_id['JobFlowId']
    )
    print(response)

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

    run_cluster(args.key1, args.key2, args.key3)