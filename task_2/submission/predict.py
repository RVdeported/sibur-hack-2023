import pathlib
import pandas as pd
from utils import Pipe, pred
from models import fc_model
import torch as t

DEVICE = "cuda" if t.cuda.is_available() else 'cpu'

DATA_DIR = pathlib.Path(".")
MODEL_FILE = pathlib.Path(__file__).parent.joinpath("model.dict")



def predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисление предсказаний.

    Параметры:
        df:
          датафрейм, содержащий строки из тестового множества.
          Типы и имена колонок совпадают с типами и именами в ноутбуке, не содержит `np.nan` или `np.inf`.

    Результат:
        Датафрейм предсказаний.
        Должен содержать то же количество строк и в том же порядке, а также колонки `target0` и `target1`.
    """    

    load = t.load(MODEL_FILE)
    model = load["model"]
    pipe = load["pipeline"]

    loader = pipe.transform(df)[1]

    return pred(loader, model, DEVICE)

# if __name__ == "__main__":
#     MODEL_FILE = pathlib.Path("../submission/model.dict")
#     ds = pd.read_parquet("../data/train.parquet")[:100]
#     print(predict(ds.drop(["target0", "target1"], axis=1)))





