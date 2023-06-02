import pathlib
import pandas as pd
import torch as t
import pickle

from sklearn.preprocessing import MinMaxScaler

DEVICE = "cuda" if t.cuda.is_available() else 'cpu'

DATA_DIR = pathlib.Path(".")
MODEL_FILE = pathlib.Path(__file__).parent.joinpath("model.t")
PIPE_FILE = pathlib.Path(__file__).parent.joinpath("pipe.pckl")

class fc_model_mul(t.nn.Module):
    def __init__(self, input_dim, layers=[512, 128, 16], device='cpu'):
        super().__init__()
        self.device=device

        self.base=t.nn.Sequential(
            t.nn.Linear(input_dim, layers[0]),
            t.nn.ReLU(),
        )

        self.zero = t.nn.ParameterList([
            t.nn.Sequential(
                t.nn.Linear(layers[0], layers[1]),
                t.nn.ReLU(),
                t.nn.Linear(layers[1], layers[2]),
                t.nn.ReLU(),
                t.nn.Linear(layers[2], 1),
            ),
            t.nn.Sequential(
                t.nn.Linear(layers[0], layers[1]),
                t.nn.ReLU(),
                t.nn.Linear(layers[1], layers[2]),
                t.nn.ReLU(),
                t.nn.Linear(layers[2], 1),
            )
        ])

        self.one = t.nn.ParameterList([
            t.nn.Sequential(
                t.nn.Linear(layers[0], layers[1]),
                t.nn.ReLU(),
                t.nn.Linear(layers[1], layers[2]),
                t.nn.ReLU(),
                t.nn.Linear(layers[2], 1),
            ),
            t.nn.Sequential(
                t.nn.Linear(layers[0], layers[1]),
                t.nn.ReLU(),
                t.nn.Linear(layers[1], layers[2]),
                t.nn.ReLU(),
                t.nn.Linear(layers[2], 1),
            )
        ])



    def forward(self, X):
        out = t.zeros((X.shape[0], 2), device=self.device, dtype=t.float32)
        
        inds_one = (X[:, -1] == t.scalar_tensor(1)).nonzero().T[0]
        inds_zero = (X[:, -1] == t.scalar_tensor(0)).nonzero().T[0]

        step = self.base(X)

        out[inds_zero, 0] = self.zero[0](step[inds_zero]).squeeze()
        out[inds_zero, 1] = self.zero[1](step[inds_zero]).squeeze()

        out[inds_one, 0] = self.one[0](step[inds_one]).squeeze()
        out[inds_one, 1] = self.one[1](step[inds_one]).squeeze()     

        return out

class Pipe:
    def __init__(self, cols_to_short=[], batch_size=64, noise_str=0.01):
        self.min_max = MinMaxScaler()
        self.cols_to_short = cols_to_short
        self.batch_size = batch_size
        self.noise_str = noise_str


    def fit(self, X: pd.DataFrame):
        X_ = self.quntize(X)
        X_["feature4"] = (X_["feature4"] == "gas1") * 1
        self.min_max.fit(X_)

    def quntize(self, X):
        X_ = X.copy()
        
        # for col in self.cols_to_short:
        #     X_ = X_[X_[col] < X_[col].quantile(0.999)]
        
        return X_

    def transform(self, X, y=None, train_df=False):
        X_ = X.copy()
        if y is not None:
            y_ = y.copy()


        if train_df:
            X_ = self.quntize(X_)
        if train_df and y is not None:
            y_ = y_.iloc[X_.index]
        

        X_["feature4"] = (X_["feature4"] == "gas1") * 1
        

        X_ = pd.DataFrame(self.min_max.transform(X_), columns=X_.columns) 
        X_["feature_23_cat"] = (X_["feature23"] > 70) * 1
        X_["feature_24_cat"] = (X_["feature24"] > 20) * 1
        X_["feature_17_cat"] = (X_["feature17"] > -10) * 1
        X_["feature_1_cat"] = (X_["feature1"] > 25) * 1
        
        X_["cluster"] = (X_["feature4"] == 1) * 1
        X_.drop("feature4", axis=1, inplace=True)

        X_t = t.tensor(X_.to_numpy(), dtype=t.float32)

        if train_df:
            noise = t.zeros(X_t.shape)
            noise[:,:-1] = t.rand((X_t.shape[0], X_t.shape[1]-1)) * self.noise_str
            X_t = t.cat(
                [
                    # X_t,
                    X_t + noise
                ]
            )


        if y is not None:
            y_t = t.tensor(y_.to_numpy(), dtype=t.float32)
            if train_df:
                y_t = t.cat([y_t])    
            ds = t.utils.data.TensorDataset(X_t, y_t)
        else:
            ds = t.utils.data.TensorDataset(X_t)

        ldr = t.utils.data.DataLoader(ds, batch_size=self.batch_size)

        return ds, ldr
    
    def pack(self, path):
        to_pack = {
            "scaler" : self.min_max,
            "cols" : self.cols_to_short,
            "batch_size" : self.batch_size
        }
        with open(path, "wb+") as f:
            pickle.dump(to_pack, f)
    
    def unpack(self, path):
        with open(path, "rb") as f:
            params = pickle.load(f)

        self.batch_size = params["batch_size"]
        self.min_max = params["scaler"]
        self.cols_to_short = params["cols"] 


def pred(ldr, model, device='cpu'):
    res = t.tensor([], dtype=t.float32)

    with t.no_grad():
        for X in ldr:
            out = model(X[0].to(device=device)).to('cpu')
            res = t.cat([res,out], axis=0)
    
    res = pd.DataFrame(
        res.detach().numpy(),
        columns = ["target0", "target1"]
    )

    return res

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


    pipe = Pipe(batch_size=256)
    pipe.unpack(PIPE_FILE)

    df, loader = pipe.transform(df)

    model = fc_model_mul(df.tensors[0].shape[1], [1024, 256, 32, 2], 'cpu')
    model.load_state_dict(
        t.load(MODEL_FILE, 
               map_location=t.device('cpu')
               )
            )
    model = model.cpu()
    model.device = "cpu"

    return pred(loader, model, "cpu")




