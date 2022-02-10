# Chapter 1: Build your first API with Fast API
import uvicorn
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class Model():
    def __init__(self) -> None:
        data = datasets.load_iris()
        self.dataset = pd.DataFrame(data=data.data, columns=data.feature_names)
        self.dataset['target'] = data.target
        self.target_names = data.target_names
        self.model_fname_ = 'iris_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as e:
            print('Error while loading model. Error - ', e)
            self.model = self.train_model()
            joblib.dump(self.model, self.model_fname_)
            pass

    def train_model(self):
        X = self.dataset.drop('target', axis=1)
        y = self.dataset['target']
        rfc = RandomForestClassifier()
        model = rfc.fit(X, y)
        return model

    def predict(self, sepal_length, sepal_width, petal_length, petal_width):
        data = [[sepal_length, sepal_width, petal_length, petal_width]]
        y_pred = self.model.predict(data)
        y_prob = self.model.predict_proba(data).max()
        return y_pred[0], y_prob


class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


app = FastAPI()
model = Model()


@app.get('/')
def index():
    return {'message': 'API is working. - [This is a modification to test CD/CI in Azure]'}


@app.post('/predict')
def predict(data: Iris):
    data = data.dict()

    # Make sure all data has been provided
    if not checkData(data):
        return {
            'message': 'You must provide data for all characteristics.'
        }

    y_pred, y_prob = model.predict(
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    )

    return {
        # NB numpy types seems not to be supported
        'prediction': float(y_pred),
        'probability': float(y_prob),
        'specie_name': model.target_names[y_pred]
    }


def checkData(data) -> bool:
    for x in data:
        if data[x] <= 0:
            return False
    return True


if __name__ == '__app__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
