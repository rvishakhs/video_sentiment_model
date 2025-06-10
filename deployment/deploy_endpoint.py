from sagemaker.pytorch.model import PyTorchModel
import sagemaker
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_endpoint():
    logger.info("Starting SageMaker session...")
    sagemaker.Session()
    role ='arn:aws:iam::973787923108:role/feedback-analysis-endpoint-role'

    model_uri = 's3://feedback-analysis-saas/inference/model.tar.gz'

    logger.info("Creating PyTorchModel...")
    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        entry_point='inference.py',
        framework_version="2.5",
        py_version='py311',
        source_dir='deploy',
        name='feedback-analysis-model',
    )
    
    logger.info("Deploying model to endpoint 'feedback-analysis-endpoint'...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.g5.xlarge',
        endpoint_name='feedback-analysis-endpoint',
    )

    logger.info("Model deployed successfully.")
    return predictor

if __name__ == "__main__":
    predictor = deploy_endpoint()
    print("Endpoint deployment script executed successfully.")


    