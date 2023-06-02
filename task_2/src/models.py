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



class fc_model_sep(t.nn.Module):
    def __init__(self, input_dim, layers=[256, 16], device='cpu'):
        super().__init__()
        self.device=device

        self.zero_1 = t.nn.Sequential(
            t.nn.Linear(input_dim, layers[0]),
            t.nn.ReLU(),
            

            t.nn.Linear(layers[0], layers[1]),
            t.nn.ReLU(),
            t.nn.Linear(layers[1], 1)
        )
        self.zero_2 = t.nn.Sequential(
            t.nn.Linear(input_dim+1, layers[0]),
            t.nn.ReLU(),
            

            t.nn.Linear(layers[0], layers[1]),
            t.nn.ReLU(),
            t.nn.Linear(layers[1], 1)
        )


        self.one_1 = t.nn.Sequential(
            t.nn.Linear(input_dim, layers[0]),
            
            t.nn.ReLU(),
            t.nn.Linear(layers[0], layers[1]),
            
            t.nn.ReLU(),
            t.nn.Linear(layers[1], 1)
        )

        self.one_2 = t.nn.Sequential(
            t.nn.Linear(input_dim+1, layers[0]),
            t.nn.ReLU(),
            

            t.nn.Linear(layers[0], layers[1]),
            t.nn.ReLU(),
            t.nn.Linear(layers[1], 1)
        )

    def forward(self, X):
        out = t.zeros((X.shape[0], 2), device=self.device, dtype=t.float32)
        
        inds_one = (X[:, -1] == t.scalar_tensor(1)).nonzero().T[0]
        inds_zero = (X[:, -1] == t.scalar_tensor(0)).nonzero().T[0]

        one = self.one_1(X[inds_one])
        one_ = self.one_2(
            t.cat([X[inds_one], one], dim=1)
        )
        
        zero = self.zero_1(X[inds_zero]) 
        zero_ = self.zero_2(
            t.cat([X[inds_zero], zero], dim=1)
        )

        out[inds_one] = t.cat([one, one_], dim=1)
        out[inds_zero] = t.cat([zero, zero_], dim=1)

        return out

class fc_model_res_d(t.nn.Module):
    def __init__(self, input_dim, stacks=3, layers=[256, 16, 256], device='cpu'):
        super().__init__()
        self.device=device
        self.stacks=stacks

        self.blocks_1 = t.nn.ParameterList([])
        self.rec_1 = t.nn.ParameterList([])
        self.pred_1 = t.nn.ParameterList([])
        for _ in range(stacks):
            self.blocks_1.append(
                t.nn.Sequential(
                    t.nn.Linear(input_dim, layers[0]),
                    t.nn.ReLU(),
                    t.nn.Linear(layers[0], layers[1]),
                    t.nn.ReLU(),
            ))

            self.rec_1.append(
                t.nn.Sequential(
                    t.nn.Linear(layers[1], layers[2]),
                    t.nn.ReLU(),
                    t.nn.Linear(layers[2], input_dim),
                )
            )

            self.pred_1.append(
                t.nn.Sequential(
                t.nn.Linear(layers[1], 2),
                )
            )

        self.blocks_2 = t.nn.ParameterList([])
        self.rec_2 = t.nn.ParameterList([])
        self.pred_2 = t.nn.ParameterList([])
        for _ in range(stacks):
            self.blocks_2.append(
                t.nn.Sequential(
                    t.nn.Linear(input_dim, layers[0]),
                    t.nn.ReLU(),
                    t.nn.Linear(layers[0], layers[1]),
                    t.nn.ReLU(),
            ))

            self.rec_2.append(
                t.nn.Sequential(
                    t.nn.Linear(layers[1], layers[2]),
                    t.nn.ReLU(),
                    t.nn.Linear(layers[2], input_dim),
                )
            )

            self.pred_2.append(
                t.nn.Sequential(
                t.nn.Linear(layers[1], 2),
                )
            )


        

    def forward(self, X):
        out = t.zeros((X.shape[0], 2), device=self.device, dtype=t.float32)
        
        inds_one = (X[:, -1] == t.scalar_tensor(1)).nonzero().T[0]
        inds_zero = (X[:, -1] == t.scalar_tensor(0)).nonzero().T[0]
        
        res_1 = X[inds_zero]
        res_2 = X[inds_one]

        for i in range(self.stacks):
            step_1 = self.blocks_1[i](res_1)
            res_1 = self.rec_1[i](step_1)
            out[inds_zero] += self.pred_1[i](step_1)

            step_2 = self.blocks_1[i](res_2)
            res_2 = self.rec_1[i](step_2)
            out[inds_one] += self.pred_2[i](step_2)

        return out

class fc_model_mul(t.nn.Module):
    def __init__(self, input_dim, layers=[256, 16], device='cpu'):
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
                t.nn.Linear(layers[1], 1),
            ),
            t.nn.Sequential(
                t.nn.Linear(layers[0], layers[1]),
                t.nn.ReLU(),
                t.nn.Linear(layers[1], 1),
            )
        ])

        self.one = t.nn.ParameterList([
            t.nn.Sequential(
                t.nn.Linear(layers[0], layers[1]),
                t.nn.ReLU(),
                t.nn.Linear(layers[1], 1),
            ),
            t.nn.Sequential(
                t.nn.Linear(layers[0], layers[1]),
                t.nn.ReLU(),
                t.nn.Linear(layers[1], 1),
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