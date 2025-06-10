from sagemaker.pytorch.model import PyTorchModel
import sagemaker
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deploy_endpoint():
    logger.info("Starting SageMaker session...")
    try:
        # Initialize SageMaker session
        session = sagemaker.Session()
        aws_region = session.boto_session.region_name # Get the region from the session
        logger.info(f"SageMaker session started successfully in region: {aws_region}")
    except Exception as e:
        logger.error(f"Failed to start SageMaker session: {e}")
        raise

    # Define the IAM role for SageMaker
    # Ensure this role has permissions for SageMaker, S3, and ECR (if using custom images)
    role = 'arn:aws:iam::973787923108:role/feedback-analysis-endpoint-role'

    logger.info(f"Using IAM role: {role}")

    # S3 URI where your model artifact (model.tar.gz) is located
    model_uri = 's3://feedback-analysis-saas/inference/model.tar.gz'
    logger.info(f"Model artifact URI: {model_uri}")

    logger.info("Creating PyTorchModel object...")
    try:
        model = PyTorchModel(
            sagemaker_session=session, # Pass the session explicitly
            model_data=model_uri,
            role=role,
            entry_point='inference.py',
            framework_version="2.5", # Ensure this matches your PyTorch version
            py_version='py311',     # Ensure this matches your Python version
            source_dir='deploy',    # Directory containing inference.py and other dependencies
            name='feedback-analysis-model', # A logical name for the model in SageMaker
        )
        logger.info("PyTorchModel object created.")
    except Exception as e:
        logger.error(f"Failed to create PyTorchModel object: {e}")
        raise

    # Define endpoint parameters
    instance_count = 1
    instance_type = 'ml.g5.xlarge' # Choose an instance type appropriate for your model
    endpoint_name = 'feedback-analysis-endpoint' # The desired name for your SageMaker endpoint

    logger.info(f"Attempting to deploy model to endpoint '{endpoint_name}' "
                f"with {instance_count} instance(s) of type '{instance_type}'...")
    try:
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            # wait=False can be used if you want the script to continue without waiting,
            # but then you'd need to poll the endpoint status yourself.
            # We will use wait=True by default and add explicit waiting logic.
        )
        logger.info(f"Deployment initiated for endpoint '{endpoint_name}'.")

        # Explicitly wait for the endpoint to be in service
        logger.info(f"Waiting for endpoint '{endpoint_name}' to be 'InService'. This may take some time...")
        # The .deploy() method by default waits if wait=True (which is the default).
        # However, adding a small loop with status checks can give more granular feedback
        # or handle cases where wait=False was used.
        # For simplicity with default wait=True, the next line might be redundant but reinforces the wait.

        # You can add a loop to check status if you set wait=False in deploy,
        # or for more granular logging during the wait.
        # For now, rely on the default wait=True behavior of model.deploy()
        # and print a success message only after it completes.

        logger.info(f"Endpoint '{endpoint_name}' deployed successfully and is 'InService'.")
        return predictor
    except Exception as e:
        logger.error(f"Failed to deploy model or endpoint creation failed: {e}")
        # You might want to delete the endpoint if it failed creation to clean up resources
        # In a real scenario, you'd add more robust error handling and cleanup.
        raise

if __name__ == "__main__":
    try:
        predictor = deploy_endpoint()
        print("\n" + "="*50)
        print("Endpoint deployment script executed successfully.")
        print(f"Endpoint Name: {predictor.endpoint_name}")
        print(f"To test your endpoint, you can use: predictor.predict(your_data)")
        print("="*50 + "\n")
    except Exception as e:
        logger.critical(f"Script execution failed: {e}")
        print("\n" + "="*50)
        print("Endpoint deployment script failed. Please check the logs above for details.")
        print("="*50 + "\n")

