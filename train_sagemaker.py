from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

import sagemaker

sess = sagemaker.Session(default_bucket="feedback-analysis-saas")


def start_training():
    print("Starting SageMaker training job...")
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path="s3://feedback-analysis-saas/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard",
    )

    estimator = PyTorch(
        entry_point="train_aws.py",
        source_dir="training",
        role="arn:aws:iam::973787923108:role/feedback-analysis-execution-role",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        framework_version="2.5.1",
        py_version="py312",
        sagemaker_session=sess,
        hyperparameters={
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        tensorboard_config = tensorboard_config,
    )

    # start the training job
    estimator.fit({
        "training": "s3://feedback-analysis-saas/dataset/train",
        "validation": "s3://feedback-analysis-saas/dataset/dev",
        "test": "s3://feedback-analysis-saas/dataset/test"
    }, wait=True)
