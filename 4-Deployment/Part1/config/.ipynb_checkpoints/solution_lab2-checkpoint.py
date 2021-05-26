import pprint
from time import strftime, gmtime

import boto3

import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.debugger import rule_configs, Rule, DebuggerHookConfig

import pandas as pd
import boto3

from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from smexperiments.tracker import Tracker

sess = boto3.Session()
sm = sess.client('sagemaker')
role = sagemaker.get_execution_role()

#Supress default INFO loggingd
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

def get_estimator_from_lab2():
    print("Getting solution from Lab 2...")
    print("Please wait 5 minutes for the training job to run.")
    
    # Set the paths for the datasets saved locally
    path_to_lab2 = "/root/amazon-sagemaker-workshop/2-Modeling/"
    
    local_train_path = path_to_lab2 + 'config/train.csv'
    train_df = pd.read_csv(local_train_path, header=None)

    # Let's check the validation dataset
    local_validation_path = path_to_lab2 + 'config/validation.csv'
    validation_df = pd.read_csv(local_validation_path, header=None)

    region = sess.region_name
    account_id = sess.client('sts', region_name=region).get_caller_identity()["Account"]
    bucket = 'sagemaker-studio-{}-{}'.format(sess.region_name, account_id)
    prefix = 'xgboost-churn'
    train_dir = f"{prefix}/train"
    val_dir = f"{prefix}/validation"

    try:
        if sess.region_name == "us-east-1":
            sess.client('s3').create_bucket(Bucket=bucket)
        else:
            sess.client('s3').create_bucket(Bucket=bucket, 
                                            CreateBucketConfiguration={'LocationConstraint': sess.region_name})
    except Exception as e:
        print("Looks like you already have a bucket of this name. That's good. Uploading the data files...")

    # Return the URLs of the uploaded file, so they can be reviewed or used elsewhere
    s3url_train = sagemaker.s3.S3Uploader.upload(local_train_path, 's3://{}/{}'.format(bucket, train_dir))
    s3url_validation = sagemaker.s3.S3Uploader.upload(local_validation_path, 's3://{}/{}'.format(bucket, val_dir))

    boto_sess = boto3.Session()
    region = boto_sess.region_name
    role = sagemaker.get_execution_role()
    sm_sess = sagemaker.session.Session()

    framework_version = '1.2-2'
    docker_image_name = sagemaker.image_uris.retrieve(framework='xgboost', region=region, version=framework_version)

    s3_input_train = TrainingInput(s3_data=f's3://{bucket}/{train_dir}', content_type='csv')
    s3_input_validation = TrainingInput(s3_data=f's3://{bucket}/{val_dir}', content_type='csv')

    # Helper to create timestamps
    create_date = lambda: strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    customer_churn_experiment = Experiment.create(experiment_name=f"customer-churn-prediction-xgboost-{create_date()}", 
                                                  description="Using xgboost to predict customer churn", 
                                                  sagemaker_boto_client=boto3.client('sagemaker'))

    hyperparams = {"max_depth":5,
                   "subsample":0.8,
                   "num_round":600,
                   "eta":0.2,
                   "gamma":4,
                   "min_child_weight":6,
                   "objective":'binary:logistic',
                   "verbosity": 0
                  }

    trial = Trial.create(trial_name=f"algorithm-mode-trial-{create_date()}", 
                         experiment_name=customer_churn_experiment.experiment_name,
                         sagemaker_boto_client=boto3.client('sagemaker'))

    xgb = sagemaker.estimator.Estimator(image_uri=docker_image_name,
                                        role=role,
                                        hyperparameters=hyperparams,
                                        train_instance_count=1, 
                                        train_instance_type='ml.m4.xlarge',
                                        output_path=f's3://{bucket}/{prefix}/output',
                                        base_job_name='demo-xgboost-customer-churn',
                                        sagemaker_session=sm_sess)

    xgb.fit(inputs={
                      'train': s3_input_train,
                      'validation': s3_input_validation
                    },
            experiment_config={
                    'ExperimentName': customer_churn_experiment.experiment_name, 
                    'TrialName': trial.trial_name,
                    'TrialComponentDisplayName': 'Training',
            },
            logs="None"
           )
    
    return xgb