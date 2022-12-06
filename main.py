from fastapi import FastAPI, Body, HTTPException, Query
import pandas as pd

# noinspection PyUnresolvedReferences
from algorithms import *
from src import anomaly_thresholds, schema, dynamic_algorithm_loading

app = FastAPI()
algorithms = list()
algorithms_json = dict()


def main():
    global algorithms, algorithms_json
    algorithms = dynamic_algorithm_loading.fetch_algorithms()
    algorithms = dynamic_algorithm_loading.sort_algorithms(algorithms)
    algorithms_json["algorithms"] = dynamic_algorithm_loading.create_algorithm_json(algorithms)


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
    return algorithms_json


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
                              "timestamps": ["2020-03-14T11:00:00", "2020-03-14T11:15:00", "2020-03-14T11:30:00"],
                              "anomalies": [
                                  {"timestamp": "2021-12-21T09:45:00", "type": "Area"},
                                  {"timestamp": "2021-12-22T09:45:00", "type": "Area"}
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
        payload=Body(
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
        df = pd.DataFrame(payload)
        if 0 <= algo < len(algorithms):
            deep_errors, output_error, output_timestamps, output_threshold = algorithms[algo].calc_anomaly_score(df)
        else:
            raise HTTPException(status_code=404, detail=f"No algorithm with id {id}")
        found_anomalies = anomaly_thresholds.find_anomalies(output_error, output_threshold)[:6]
        output_anomalies = anomaly_thresholds.parse_anomalies(found_anomalies, output_timestamps)
        return {"error": output_error,
                "timestamps": output_timestamps,
                "anomalies": output_anomalies,
                "threshold": output_threshold,
                "deep-error": deep_errors,
                "raw-anomalies": found_anomalies}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")


schema.custom_openapi(app)
main()
