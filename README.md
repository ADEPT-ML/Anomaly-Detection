# Anomaly-Detection 🩺

The service is responsible for detecting anomalies in the (multivariate) time-series data. It provides different algorithms for the detection.

## Requirements

+ Python >3.10
+ All packages from requirements.txt

## Development

### Local

Install dependencies from requirements.txt

Start the service:

```sh
uvicorn main:app --reload
```

### Docker

We provide a docker-compose in the root directory of ADEPT to start all services bundled together.

Copyright © ADEPT ML, TU Dortmund 2022
