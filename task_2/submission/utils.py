from tqdm import tqdm
import numpy as np
import pandas as pd


from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.preprocessing import MinMaxScaler

import torch as t

from datetime import datetime as dt


tqdm.pandas()
pd.set_option('display.max_columns', None)

def train(train_ldr, test_ldr, model, optimizer, loss_f, epochs=10, device='cpu'):

    bar = tqdm(range(epochs))
    for epoch in bar:
        loss_cum = 0
        for X, y in train_ldr:
            optimizer.zero_grad()
            out = model(X.to(device=device))
            loss = loss_f(out, y.to(device=device))
            loss.backward()
            loss_cum += loss
            optimizer.step()

        eval_test = eval(test_ldr, model, device)
        eval_train = eval(train_ldr, model, device)
        bar.set_description(f'''epoch {epoch}, loss {loss_cum:.3f}, mape_train {(eval_train * 100).round(2)}, mape_test {(eval_test * 100).round(2)}''')
        



def eval(test_ldr, model, device='cpu'):
    
    loss = np.array([0, 0], dtype=np.float32)
    i=0
    for X, y in test_ldr:
        with t.no_grad():
            out = model(X.to(device=device))
            loss += mape(y, out.cpu().detach().numpy(), multioutput='raw_values')
            i+=1

    return loss / i


def save_model(model, path, prefix):
    timestamp = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    filename = prefix + timestamp + ".t"
    t.save(model, path + filename)

class Pipe:
    def __init__(self, cols_to_short, batch_size=64):
        self.min_max = MinMaxScaler()
        self.cols_to_short = cols_to_short
        self.batch_size = batch_size


    def fit(self, X: pd.DataFrame):
        X_ = self.quntize(X)
        X_["feature4"] = (X_["feature4"] == "gas1") * 1
        self.min_max.fit(X_)

    def quntize(self, X):
        X_ = X.copy()
        
        for col in self.cols_to_short:
            X_ = X_[X_[col] < X_[col].quantile(0.999)]
        
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
        X_["cluster"] = (X_["feature4"] == 1) * 1
        X_.drop("feature4", axis=1, inplace=True)

        X_t = t.tensor(X_.to_numpy(), dtype=t.float32)

        if y is not None:
            y_t = t.tensor(y_.to_numpy(), dtype=t.float32)
            ds = t.utils.data.TensorDataset(X_t, y_t)
        else:
            ds = t.utils.data.TensorDataset(X_t)

        ldr = t.utils.data.DataLoader(ds, batch_size=self.batch_size)

        return ds, ldr
    
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