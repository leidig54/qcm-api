#Example: scikit-learn and Swagger
import json
import numpy as np
import os
from sklearn.externals import joblib
from azureml.core.model import InferenceConfig, Model

from sklearn import mixture
import sklearn.mixture._gaussian_mixture

import logging
logging.basicConfig(level=logging.DEBUG)


from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType


def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment. Join this path with the filename of the model file.
    # It holds the path to the directory that contains the deployed model (./azureml-models/$MODEL_NAME/$VERSION).
    # If there are multiple models, this value is the path to the directory containing all deployed models (./azureml-models).
    # Alternatively: 
    #model_path = Model.get_model_path(model_name='mymodel')
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model_scaler.joblib')
    # Deserialize the model file back into a sklearn model
    model = joblib.load(model_path)


input_sample = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1,5,5,5,5,5])
output_sample = np.array([3726.995])


@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))

def run(data):
    try:
        scaled_data = model[1].transform(data.reshape(1, -1))
        result = model[0].score(scaled_data.reshape(1, -1))
        # You can return any data type, as long as it is JSON serializable.
        print(result)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error