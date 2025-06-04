from locust import HttpUser, task
import pandas as pd
import random

# CSV 로드
dataset = pd.read_csv("./iris_test_data.csv").to_dict(orient="records")

class IrisUser(HttpUser):
    @task
    def predict(self):
        sample = random.choice(dataset)
        self.client.post("/predict", json=sample)
