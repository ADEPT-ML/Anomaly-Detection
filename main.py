from dataclasses import dataclass

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from algorithms import isolation_forest, ocsvm, deep_model
import pandas as pd
import numpy as np

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "there!"}


@app.get("/algorithms")
def read_algorithms():
    return {"algorithms": [{"name": "Isolation Forest", "id": 0, "explainable": False},
                           {"name": "One-Class SVM", "id": 1, "explainable": False},
                           {"name": "LSTM Autoencoder", "id": 2, "explainable": True}]}


@app.post("/calculate")
def calculate_anomalies(algo: int, building: str, payload=Body(..., embed=True)):
    anomaly_data = {"algo": algo}
    deep_errors = None
    output_data, output_error, output_timestamps, output_threshold = None, None, None, None
    df = pd.DataFrame(payload)
    match algo:
        case 0:
            model_forest = isolation_forest.IsolationForestModel()
            output_data = model_forest.calc_anomaly_score(df)
            output_error, output_timestamps, output_threshold = shallow_model_post_processing(output_data)
        case 1:
            model_ocsvm = ocsvm.OCSVM()
            output_data = model_ocsvm.calc_anomaly_score(df)
            output_error, output_timestamps, output_threshold = shallow_model_post_processing(output_data)
        case 2:
            deep_scores, deep_errors, deep_output = [], [], []
            for col in df.keys():
                # df[col] = [(x - df[col].mean()) / df[col].std() for x in df[col]]
                scores, errors, output = deep_model.predict(df[[col]].reset_index(drop=True))
                deep_scores.append(replace_nan(unpack_array(scores)))
                deep_errors.append(replace_nan(unpack_array(errors)))  # Array of arrays with only one element per arr
                deep_output.append(replace_nan(unpack_array(output)))  # Array of arrays with only one element per arr
            output_error = [sum(l[i] for l in deep_errors) for i in range(len(deep_errors[0]))]
            output_threshold = np.percentile(output_error, 99.5)
            output_timestamps = [str(e) for e in df.index]
            anomaly_data["errors"] = deep_errors
            anomaly_data["output_error"] = output_error
    found_anomalies = find_anomalies(output_error, output_threshold)[:6]
    anomaly_data["timestamps"] = output_timestamps
    anomaly_data["anomalies"] = found_anomalies
    output_anomalies = parse_anomalies(found_anomalies, output_timestamps)
    output_json = {"error": output_error,
                   "timestamps": output_timestamps,
                   "anomalies": output_anomalies,
                   "threshold": output_threshold,
                   "deep-error": deep_errors,
                   "raw-anomalies": found_anomalies}
    print(output_json, flush=True)
    return output_json


def replace_nan(input):
    return [e if e == e else 0 for e in input]


def unpack_array(input):
    return [e[0] if hasattr(e, '__len__') else e for e in input]


def shallow_model_post_processing(output_data):
    output_timestamps = [str(e) for e in output_data.index]
    output_error = [e for e in output_data[output_data.keys()[0]]]
    output_threshold = max(output_error)
    output_error = [e * -1 + output_threshold for e in output_error]
    return output_error, output_timestamps, output_threshold


def find_anomalies(output_error: list[float], threshold: float):
    anomalies = []
    current_anomaly = False
    for i in range(len(output_error)):
        if output_error[i] > threshold:
            if current_anomaly:
                anomaly = anomalies[-1]
                anomaly.length += 1
                anomaly.error = max(anomaly.error, output_error[i])
            else:
                anomalies.append(Anomaly(length=1, index=i, error=output_error[i]))
                current_anomaly = True
        else:
            current_anomaly = False
    return sorted(anomalies, key=lambda x: x.error, reverse=True)


def parse_anomalies(anomalies: list, timestamps: list[str]):
    return [{"timestamp": timestamps[e.index + e.length // 2], "type": "Point" if e.length == 1 else "Area"} for e in
            anomalies]


@dataclass
class Anomaly:
    length: int
    index: int
    error: float
