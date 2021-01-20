import json
import sagemaker
import boto3
import uuid
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

role = <'SageMaker execution role ARN'>
sess = sagemaker.session.Session()
region = sess._region_name
bucket = sess.default_bucket()
key_prefix = "hpo-sqs"

#check HPO jobs
sm_client = boto3.client('sagemaker')

#sqs client
sqs = boto3.client('sqs')
queue_url = <'queue_url'>

HPO_LIMIT = 100


def check_hpo_jobs():
    response = sm_client.list_hyper_parameter_tuning_jobs(
    MaxResults=HPO_LIMIT,
    StatusEquals='InProgress')
    return len(list(response["HyperParameterTuningJobSummaries"]))


def create_hpo(container,train_input,validation_input,hp_range):
    print(hp_range)

    hyperparameter_ranges = {'gamma': ContinuousParameter(hp_range['gamma_lb'], hp_range['gamma_ub']),
        'alpha': ContinuousParameter(0, 2),
        'lambda': ContinuousParameter(0, 2)}
    hyperparameters={
        "num_round":"100",
        "early_stopping_rounds":"9",
        "max_depth": "5",
        "subsample": "0.9",
        "silent": "0",
        "objective": "binary:logistic",
    }
    objective_metric_name = 'validation:f1'
    xgb_churn = Estimator(
            role=role,
            image_uri=container,
            base_job_name="xgboost-churn",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            hyperparameters=hyperparameters
    )

    tuner = HyperparameterTuner(xgb_churn,
                            objective_metric_name,
                            hyperparameter_ranges,
                            base_tuning_job_name = 'xgb-churn-hpo'+str(uuid.uuid4())[:3],
                            max_jobs=15,
                            max_parallel_jobs=5)
    tuner.fit({"train": train_input, "validation": validation_input}, wait=False)

def lambda_handler(event, context):

    #fist: check HPO jobs in progress
    hpo_in_progress = check_hpo_jobs()

    if hpo_in_progress>=HPO_LIMIT:
        return {
        'statusCode': 200,
        'body': json.dumps('HPO running full')
    }
    else:
        hpo_capacity = HPO_LIMIT - hpo_in_progress
        container = image_uris.retrieve("xgboost", region, "0.90-2")
        train_input = TrainingInput(f"s3://{bucket}/{key_prefix}/train/train.csv", content_type="text/csv")
        validation_input = TrainingInput(f"s3://{bucket}/{key_prefix}/validation/validation.csv", content_type="text/csv")

        while hpo_capacity> 0:
            sqs_response = sqs.receive_message(QueueUrl = queue_url)
            if 'Messages' in sqs_response.keys():
                msgs = sqs_response['Messages']
                for msg in msgs:
                    try:
                        hp_in_msg = json.loads(msg['Body'])['hyperparameter_ranges']
                        create_hpo(container,train_input,validation_input,hp_in_msg)
                        response = sqs.delete_message(QueueUrl=queue_url,ReceiptHandle=msg['ReceiptHandle'])
                        hpo_capacity = hpo_capacity-1
                        if hpo_capacity == 0:
                            break
                    except :
                        return ("error occurred for message {}".format(msg['Body']))
            else:
                return {'statusCode': 200, 'body': json.dumps('Queue is empty')}

        return {'statusCode': 200,  'body': json.dumps('Lambda completes') }
