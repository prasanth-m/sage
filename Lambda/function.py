import json
import datetime
import time
import boto3
import os
from time import gmtime, strftime


def lambda_handler(event, context):
    print(event)
    event_body = event["key"]
    print(event_body)
    if  event_body == 'train_data':
      response =  train_data()
      return response
    elif event_body == 'deploy_model':
      response =  deploy_model(event)
      return response
    elif event_body == 'invoke_model':
       response = invoke_model(event)
       return response
    else:
        return 'incorrect method'
    
def train_data():

    status_code = 200
    try:

        region = os.environ['region']
        image= os.environ['image']
        job_name = "coupon-sckit-training-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        hyperparameters = {"kernel": "rbf"}
        role=os.environ['role']
        
        create_training_params = {
           "OutputDataConfig": {"S3OutputPath": "s3://<S3 BUCKET>/output/"},
           "RoleArn": role,
           "HyperParameters": hyperparameters,
           "AlgorithmSpecification":{
           "TrainingImage": image,
           "TrainingInputMode": 'File'
            },
            "ResourceConfig": {"InstanceCount": 1, "InstanceType": "ml.c4.xlarge", "VolumeSizeInGB": 10},
           "TrainingJobName": job_name,
           "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
           "InputDataConfig": [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://<S3 BUCKET/data/training/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            }
        }
    ],
}
        client = boto3.client("sagemaker", region_name=region)
        client.create_training_job(**create_training_params) 
        status = client.describe_training_job(TrainingJobName=job_name)["TrainingJobStatus"]
       
        print(status)
        return {
            "response_code ": status_code,
            "status" : status,
            "training_job" : job_name
        }
    except Exception as err:
        status_code = 501
        print(err)
        json_msg = str(err)
        return {
            "response_code ": status_code,
            "data" : json_msg
        }

def deploy_model(event):

    status_code = 200

    try:

        client = boto3.client('sagemaker')
        image = os.environ['image']
        job_name = event["training_job"]
        role=os.environ['role']
        
        model_name = job_name + "-model"

        info = client.describe_training_job(TrainingJobName=job_name)
        model_data = info["ModelArtifacts"]["S3ModelArtifacts"]

        primary_container = {"Image": image, "ModelDataUrl": model_data}

        create_model_response = client.create_model(
              ModelName=model_name, ExecutionRoleArn=role, PrimaryContainer=primary_container
              )

        print(create_model_response["ModelArn"])

        print(create_model_response)
        print("now trying to create endpoint config...")
        endpoint_config_name = model_name + "-endpoint-config"
        response = client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'variant-1',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.t2.medium'
                }
            ]
        )

        print(response)
        print("now trying to create the endpoint...")
        endpoint_name = model_name + "-endpoint"
        response = client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )

        endpoint_arn = response['EndpointArn']
        resp = client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print("Arn: " + resp["EndpointArn"])
        print("Status: " + status)
        return {
            "response_code ": status_code,
            "data" : status
        }

    except Exception as err:
        status_code = 501
        print(err)
        json_msg = str(err)
        return {
            "response_code ": status_code,
            "data" : json_msg
        }

# function that invoke end point to perform the real-time prediction
def invoke_model(event):

    
    status_code = 200

    try:
        # From the input parameter named "event", get the body, which contains
        # the features and end point name
        event_body = event["feature"]
        EndpointName = event["endpoint_name"]
        # invoke the SageMaker endpoint
        client = boto3.client('sagemaker-runtime')
        response = client.invoke_endpoint(
            EndpointName=EndpointName,
            Body=event_body.encode('utf-8'),
            ContentType='text/csv'
        )

        predictions = response["Body"].read().decode('utf-8') 
        print(response)

        return {
            "response_code ": status_code,
            "predictions" : predictions
        }

    except Exception as err:
        status_code = 501
        print(err)
        json_msg = str(err)

    # Return the err and  status .
        return {
            "response_code ": status_code,
            "data" : json_msg
        }