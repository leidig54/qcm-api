from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import LocalWebservice
from azureml.core import Workspace

ws = Workspace.get(name="QCM", subscription_id='ed4fc9bc-f386-4f01-a8b4-2077312476f3', resource_group='appsvc_linux_centralus')

models = Model.register(model_path="model_scaler.joblib",
                       model_name="mymodel",
                       tags={'area': "diabetes", 'type': "regression"},
                       description="Ridge regression model to predict diabetes",
                       workspace=ws)


# Create inference configuration based on the environment definition and the entry script
myenv = Environment.from_conda_specification(name="env", file_path="myenv.yml")
inference_config = InferenceConfig(entry_script="score.py", environment=myenv)

# Create a local deployment, using port 8890 for the web service endpoint
deployment_config = LocalWebservice.deploy_configuration(port=8890)

# Deploy the service
service = Model.deploy(
    ws, "mymodel", [models], inference_config, deployment_config)

# Wait for the deployment to complete
service.wait_for_deployment(True)

# Display the port that the web service is available on
print(service.port)