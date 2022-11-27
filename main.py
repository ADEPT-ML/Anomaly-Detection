from fastapi import FastAPI, Body, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
from algorithms import isolation_forest, ocsvm, deep_model
import pandas as pd
import numpy as np

from src import anomaly_thresholds, processing, utils, schema

app = FastAPI()
# origins = ["*"]
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.get(
    "/",
    name="Root path",
    summary="Returns the routes available through the API",
    description="Returns a route list for easier use of API through HATEOAS",
    response_description="List of urls to all available routes",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "payload": [
                            {
                                "path": "/examplePath",
                                "name": "example route"
                            }
                        ]
                    }
                }
            },
        }
    }
)
async def root():
    """Root API endpoint that lists all available API endpoints.

    Returns:
        A complete list of all possible API endpoints.
    """
    route_filter = ["openapi", "swagger_ui_html", "swagger_ui_redirect", "redoc_html"]
    url_list = [{"path": route.path, "name": route.name} for route in app.routes if route.name not in route_filter]
    return url_list


@app.get("/algorithms",
    name="Anomaly Detection Algorithms",
    summary="Returns a list of the available anomaly detection algorithms",
    description="Returns a list with the anomaly detection algorithms..",
    response_description="List of the algorithms.",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "algorithms": [
                            {
                                "name": "Isolation Forest",
                                "id": 0,
                                "explainable": False
                            },
                            {
                                "name": "One-Class SVM",
                                "id": 1,
                                "explainable": False
                            },
                            {
                                "name": "LSTM Autoencoder",
                                "id": 2,
                                "explainable": False
                            }
                        ]
                    }
                }
            },
        },
    },
    tags=["Anomaly Detection"]
)
def read_algorithms():
    # Implement error handling if algorithms are loaded from another source
    return {"algorithms": [{"name": "Isolation Forest", "id": 0, "explainable": False},
                           {"name": "One-Class SVM", "id": 1, "explainable": False},
                           {"name": "LSTM Autoencoder", "id": 2, "explainable": True}]}


@app.post("/calculate",
    name="Calculate anomalies",
    summary="Calculates anomalies to given buildings (or dataframe) using the selected algorithm",
    description="Returns a list of anomalies and accompanying information.",
    response_description="List of anomalies.",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "error": [0.03145960019416866, 0.024359986113175414, 0.023060245303469007],
                        "timestamps":["2020-03-14T11:00:00","2020-03-14T11:15:00","2020-03-14T11:30:00"],
                        "anomalies": [
                            {"timestamp": "2021-12-21T09:45:00", "type": "Area"}, {"timestamp": "2021-12-22T09:45:00", "type": "Area"}
                        ], 
                        "threshold": 0.2903343708384869
                    }
                }
            },
        },
        404: {
            "description": "Algorithm or Building not found.",
            "content": {
                "application/json": {
                    "example": {"detail": "Building not found"}
                }
            },
        },
        500: {
            "description": "Internal server error.",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal server error"}
                }
            },
        }
    },
    tags=["Anomaly Detection"]
)
def calculate_anomalies(
    algo: int = Query(
        description="Path parameter to select the algorithm",
        example="1"
    ), 
    building: str = Query(
        description="Path parameter to select a building",
        example="EF 40a"
    ), 
    payload = Body(
        default=...,
        description="A dataframe (encoded in json) to be used in the detection",
        example={
            "payload": {
                "Temperatur": {
                    "2021-01-02T22:00:00": 2.8,
                    "2021-01-02T22:15:00": 2.825,
                    "2021-01-02T22:30:00": 2.85,
                    "2021-01-02T22:45:00": 2.875,
                    "2021-01-02T23:00:00": 2.9
                },
                "Wärme Diff": {
                    "2021-01-02T22:00:00": 4.25,
                    "2021-01-02T22:15:00": 4.25,
                    "2021-01-02T22:30:00": 4.25,
                    "2021-01-02T22:45:00": 4.25,
                    "2021-01-02T23:00:00": 4.25
                }
            }
        },
        embed=True
    )
):
    try:
        anomaly_data = {"algo": algo}
        deep_errors = None
        output_data, output_error, output_timestamps, output_threshold = None, None, None, None
        df = pd.DataFrame(payload)
        match algo:
            case 0:
                model_forest = isolation_forest.IsolationForestModel()
                output_data = model_forest.calc_anomaly_score(df)
                output_error, output_timestamps, output_threshold = processing.shallow_model_post_processing(output_data)
            case 1:
                model_ocsvm = ocsvm.OCSVM()
                output_data = model_ocsvm.calc_anomaly_score(df)
                output_error, output_timestamps, output_threshold = processing.shallow_model_post_processing(output_data)
            case 2:
                deep_scores, deep_errors, deep_output = [], [], []
                for col in df.keys():
                    # df[col] = [(x - df[col].mean()) / df[col].std() for x in df[col]]
                    scores, errors, output = deep_model.predict(df[[col]].reset_index(drop=True))
                    deep_scores.append(utils.replace_nan(utils.unpack_array(scores)))
                    deep_errors.append(utils.replace_nan(utils.unpack_array(errors)))  # Array of arrays with only one element per arr
                    deep_output.append(utils.replace_nan(utils.unpack_array(output)))  # Array of arrays with only one element per arr
                output_error = [sum(l[i] for l in deep_errors) for i in range(len(deep_errors[0]))]
                output_threshold = np.percentile(output_error, 99.5)
                output_timestamps = [str(e) for e in df.index]
                anomaly_data["errors"] = deep_errors
                anomaly_data["output_error"] = output_error
            case _: 
                raise HTTPException(status_code=404, detail=f"No algorithm with id {id}")
        found_anomalies = anomaly_thresholds.find_anomalies(output_error, output_threshold)[:6]
        anomaly_data["timestamps"] = output_timestamps
        anomaly_data["anomalies"] = found_anomalies
        output_anomalies = anomaly_thresholds.parse_anomalies(
            found_anomalies, output_timestamps)
        output_json = {"error": output_error,
                    "timestamps": output_timestamps,
                    "anomalies": output_anomalies,
                    "threshold": output_threshold,
                    "deep-error": deep_errors,
                    "raw-anomalies": found_anomalies}
        return output_json
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")


schema.custom_openapi(app)
