import time
from sagemaker_setup import role
from dataset_prep import model_id
from sagemaker.huggingface import HuggingFace

# define Training Job Name 
job_name = f'prox-cause-qlora-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'



# hyperparameters, which are passed into the training job
hyperparameters ={
  'model_id': model_id,                                # pre-trained model
  'dataset_path': '/opt/ml/input/data/training', # path where sagemaker will save training dataset
  'epochs': 3,                                         # number of training epochs
  'per_device_train_batch_size': 4,                    # batch size for training
  'lr': 2e-4,                                          # learning rate used during training
  'hf_token': '<keyHiddenForGithubUpload>'                                    # huggingface token for private models
}

# create the Estimator
huggingface_estimator = HuggingFace(
    entry_point          = 'run_clm.py',      # train script
    source_dir           = 'scripts',         # directory which includes all the files needed for training
    instance_type        = 'ml.g5.12xlarge', # instances type used for the training job
    instance_count       = 1,                 # the number of instances used for training
    base_job_name        = job_name,          # the name of the training job
    role                 = role,              # Iam role used in training job to access AWS ressources, e.g. S3
    volume_size          = 300,               # the size of the EBS volume in GB
    transformers_version = '4.28',            # the transformers version used in the training job
    pytorch_version      = '2.0',            # the pytorch_version version used in the training job
    py_version           = 'py310',            # the python version used in the training job
    hyperparameters      =  hyperparameters,
    environment          = { "HUGGINGFACE_HUB_CACHE": "/tmp/.cache" }, # set env variable to cache models in /tmp
)

# Define S3 path for the training data
training_input_path = 's3://sagemaker-us-east-2-851725296592/processed/optimProxCause/train'

# Start the training
huggingface_estimator.fit({'training': training_input_path})