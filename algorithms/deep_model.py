import os
import pandas as pd
import torch as torch


def predict(data: pd.DataFrame):
    model_path = os.path.join('algorithms', 'models/deep_model.pth')
    model = torch.load(model_path)
    data = (data - data.mean(axis=0)) / data.std(axis=0)  # Normalizing data
    # p_scores, p_errors, p_outputs, p_attribution, p_mean, p_cov = model.predict(df)
    p_scores, p_errors, p_outputs = model.predict_without_attributions(data)
    return p_scores, p_errors, p_outputs
