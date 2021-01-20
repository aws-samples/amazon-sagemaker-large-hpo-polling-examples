## Running large numbers of HPO jobs on Amazon SageMaker

Customers from domains across financial services, healthcare, automotive often need to run large numbers of hyper-parameter tuning (HPO) jobs in order to train models for fraud detection, semantic segmentation, object detection etc. This repo contains code that will demonstrate the following:

1. How data scientists can use Amazon SageMaker to run tens of thousands of HPO jobs using their notebook.

2. How DevOps engineers can build queues to batch jobs in production using SQS and Lambda

3. How you can use open source tools like Ray Tune to automatically run many HPO jobs using SageMaker Training.

### Batching HPO jobs in Notebook

Running large numbers of HPO jobs involves a balance of compute resourses, total run time and derived business value. You as a data scientists may want to optimize for time, while business stakeholders will care more about total cost to run these jobs and the incremental business value derived from finding marginally better performant models.

The notebook **Pytorch_batch_hpo** helps you determine an HPO strategy to paralleize HPO jobs, thus reducing the total run time. This strategy largely depends on your account limits, as well as instance type availability in your region. You may need to consult your AWS account team to find those out for you. For example, if you wish to run 20 HPO trials (individual training jobs with a specific parameter combination), with a c5.xlarge instance, you will need to make sure you have 20 such instances available.

First, the user specifies how many total trails they wish to run, and determine their account limits. The code provided uses the SageMaker account limits of a maximum of 500 trials per HPO job, and 10 trials in parallel per HPO job. Next, we have provided code that will guide you on how to stagger your HPO jobs such that you can achieve your desired number of trials while minimizing total run time. To do so, the provided code will take as input, the number of jobs you want to run in parallel, and poll the HPO service to find out how many jobs are running at any given time. If this number falls below the desired number of parallel jobs, it launches new ones, until you reach the total number of trials in search space.

Note that this code is intended for large scale HPO where you want to run thousands of jobs. If you have less than 500 max_jobs, simply run 1 HPO job. 

The code uses the UCI dataset for credit card default (https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) as the base dataset and PyTorch as the framework. We also demonstrate how you can use custom metrics to optimize your SageMaker HPO jobs. Finally, we have provided separate scripts for you to run on both CPUs (*train_cpu.py*) and GPUs (*train.py*). The GPU code also includes the latest SageMaker Distributed Data Parallel library (https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html) to parallelize your data across multiple compute instances for single node multi GPU or multi node multi GPU use cases. 


### Motivation of using HPO queue

When multiple data scientists create HPO jobs in the same account at the same time, the limit of concurrent HPO jobs per account might be reached. In this case, we can use Amazon SQS to create a HPO job queue. Each HPO job request is represented as a message and submitted to a SQS queue. Each message contains hyperparameters and tunable hyperparameter ranges in message body. We then create a Lambda function to create HPO jobs. The Lambda function first checks the number of HPO jobs in progress. If the concurrent HPO jobs limit is not reached, it will retrieve messages from the SQS queue and create HPO jobs as stipulated in the message. The Lambda function is triggered by EventBridge event at a regular interval (e.g., 10 minutes).

#### What is in this repo (src/lambda_code)

In this repo, you can find the following items:
* `Setup and testing notebook`. This notebook contains instructions of setting everything up. At the end of this notebook, we have code to send messages to SQS queue for testing purposes.
* `hop_lambda.py`. This is the code for Lambda function.
* `sagemakersdk.zip`. This files is used for creating a Lambda layer that is required for the Lambda function to work. 


### Ray Tune for HPO (src/code/)

Ray is a popular open source library that started with a paper from the RISE lab at UC Berkeley (https://docs.ray.io/en/latest/tune/index.html) for HPO. Ray integrates with many popular search algorithms as well as schedulers to optimize the HPO process. It also works with the popular frameworks: Keras, PyTorch, MxNet etc. 

IN the notebook **pytorch_batch_hpo** we provide some code for you to try out Ray. As Ray requires some modifications to the standard SageMaker PyTorch training, we have provided 2 scripts: *train_ray_cpu* and *train_ray* intended for single node CPU and GPU training using Ray.


Enjoy!

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

