import joblib
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)

logging.info('Kicking off the model training script...')
train_df = pd.read_csv("../data/train1.csv", parse_dates=["date"])
logging.info('Data loaded!')
train_df = train_df.sort_values("date").reset_index(drop = True)

logging.info("Initializing the local outlier factor model")
model = LocalOutlierFactor(n_neighbors=5, contamination=0.05, novelty=True)

logging.info("Training the model...")
model.fit(train_df[["hours"]])

logging.info("Training complete. Dumping the model in cache system")

joblib.dump(model, "model.joblib")
logging.info("Model frozen! Exiting...")