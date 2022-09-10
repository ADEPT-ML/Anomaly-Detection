from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
