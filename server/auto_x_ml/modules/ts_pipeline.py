import pandas as pd  # requires: pip install pandas
import torch
from chronos import ChronosPipeline

class TimeSeriesPipeline(): 
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",
            device_map=self.device,  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )

    def predict_csv(self, csv_files, col=''):
        forecasts = []
        for c in csv_files:
            df = pd.read_csv(c)

            # context must be either a 1D tensor, a list of 1D tensors,
            # or a left-padded 2D tensor with batch as the first dimension
            # forecast shape: [num_series, num_samples, prediction_length]
            forecast = self.pipeline.predict(
                context=torch.tensor(df[col]),
                prediction_length=12,
                num_samples=20,
            )
            forecasts.append(forecast)

        return forecasts