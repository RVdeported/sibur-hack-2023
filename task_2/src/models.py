import torch as t


class fc_model(t.nn.Module):
    def __init__(self, input_dim, layers=[256, 16, 2], device='cpu'):
        super().__init__()
        self.device=device

        self.zero = t.nn.Sequential(
            t.nn.Linear(input_dim, layers[0]),
            t.nn.ReLU(),
            

            t.nn.Linear(layers[0], layers[1]),
            t.nn.ReLU(),
            t.nn.Linear(layers[1], layers[2])
        )
        self.one = t.nn.Sequential(
            t.nn.Linear(input_dim, layers[0]),
            
            t.nn.ReLU(),
            t.nn.Linear(layers[0], layers[1]),
            
            t.nn.ReLU(),
            t.nn.Linear(layers[1], layers[2])
        )

    def forward(self, X):
        out = t.zeros((X.shape[0], 2), device=self.device, dtype=t.float32)
        
        inds_one = (X[:, -1] == t.scalar_tensor(1)).nonzero().T[0]
        inds_zero = (X[:, -1] == t.scalar_tensor(0)).nonzero().T[0]

        one = self.one(X[inds_one]) 
        zero = self.zero(X[inds_zero]) 

        out[inds_one] = one
        out[inds_zero] = zero

        return out

