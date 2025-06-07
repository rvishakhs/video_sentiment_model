from sagemaker.pytorch import PyTorch
import sagemaker
import sagemaker.session

def deploy_endpoint():
    sagemaker.session()
    role ='arn:aws:iam::973787923108:role/feedback-analysis-endpoint-role'
    





if __name__ == "__main__":
    deploy_endpoint()
    print("Endpoint deployment script executed successfully.")