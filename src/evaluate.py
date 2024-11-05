import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
import mlflow
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
import os

from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/rohithbackup67/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="rohithbackup67"
os.environ['MLFLOW_TRACKING_PASSWORD']="cb292d97c74e622b80f6aba5692467b347a56ff3"

params=yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/rohithbackup67/machinelearningpipeline.mlflow")

    ##load model from the disk
    model=pickle.load(open(model_path,'rb'))
    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)
    ##log metrics to mlflow
    mlflow.log_metric("accuracy",accuracy)
    print("Model Accuracy:{accuracy}")

if __name__=="__main__":
    evaluate(params["data"],params["model"])  
