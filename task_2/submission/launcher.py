from predict import predict
import pandas as pd

ds = pd.read_parquet("./data/train.parquet")[:100]
print(predict(ds.drop(["target0", "target1"], axis=1)))