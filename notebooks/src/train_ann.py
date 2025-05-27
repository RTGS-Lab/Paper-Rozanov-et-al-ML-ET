import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
device = ('cuda' if torch.cuda.is_available() else 'cpu')
class Dataset():
    def __init__(self, X, Y, x_scaler=None, y_scaler=None, fit_scaler=False):
        if fit_scaler:
            self.x_scaler = MinMaxScaler()  
            self.y_scaler = MinMaxScaler()  
            X = self.x_scaler.fit_transform(X) 
            Y = self.y_scaler.fit_transform(Y.reshape(-1, 1)) 
        elif x_scaler is not None:
            X = x_scaler.transform(X)  
            Y = y_scaler.transform(Y.reshape(-1, 1)) 
            
        self.x = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(Y, dtype=torch.float32).to(device)
        self.len = self.x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ANN(nn.Module):
    def __init__(self, INPUT_SIZE, n1, n2, dropout=0.3):
        super().__init__()
	
        self.net = nn.Sequential()
        self.net.add_module('Linear_1', nn.Linear(INPUT_SIZE, n1))
        self.net.add_module('ReLU_1', nn.LeakyReLU()) #Sigmoid
        self.net.add_module('Drop_1', nn.Dropout(dropout))
        
        self.net.add_module('Linear_2', torch.nn.Linear(n1, n2))
        self.net.add_module('Norm', nn.BatchNorm1d(n2))
        self.net.add_module('ReLU_2', nn.LeakyReLU())
        self.net.add_module('Drop_2', nn.Dropout(dropout))
        self.net.add_module('Linear_3',torch.nn.Linear(n2, 1))

    def forward(self, y):
        return self.net(y)
    
def train_ann(train_dataset, train_loader, test_loader, model, criteria, optimizer, num_epoch):
    import numpy as np
    
    best = -np.inf
    for epoch in range(num_epoch):
        model.train()
        for x, y in train_loader:
            x, y = x.to(torch.float32), y.to(torch.float32)
            pred = model(x)
            error = criteria(pred, y)

            optimizer.zero_grad()
            error.backward()
            optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            test_preds = []
            test_true = []
            test_x = []
            test_idx = []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(torch.float32), y.to(torch.float32)
                    preds = model(x)
                    preds = train_dataset.y_scaler.inverse_transform(preds.detach().cpu())
                    y_scaled = train_dataset.y_scaler.inverse_transform(y.detach().cpu())
                    test_preds.append(torch.tensor(preds))
                    test_true.append(torch.tensor(y_scaled))

            test_preds = torch.cat(test_preds).squeeze()
            test_true = torch.cat(test_true).squeeze()

            test_loss = criteria(test_preds, test_true).item()
            r2 = r2_score(test_true.numpy(), test_preds.numpy())
            mae = mean_absolute_error(test_true.numpy(), test_preds.numpy())

            best_old = best
            best = min(test_loss, best)
            if abs(best) > best_old:
                torch.save(model.state_dict(), f'../models/ANN.pth') 
            print(f'Test RMSE: {test_loss**0.5:0.3f}\t\tTest R2: {r2:0.4f}\t\t Test MAE: {mae:0.4f}')

    model.load_state_dict(torch.load(f'../models/ANN.pth'))
    model.eval()
    test_preds = []
    test_true = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(torch.float32), y.to(torch.float32)
            preds = model(x)
            preds = train_dataset.y_scaler.inverse_transform(preds.detach().cpu())
            y_scaled = train_dataset.y_scaler.inverse_transform(y.detach().cpu())
            test_preds.append(torch.tensor(preds))
            test_true.append(torch.tensor(y_scaled))

    test_preds = torch.cat(test_preds).squeeze()
    return test_preds