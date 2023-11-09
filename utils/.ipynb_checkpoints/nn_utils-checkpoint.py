import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class dataset(Dataset):
    '''
    Params:
    X_path, y_path: str -> path of data and label
    preprocessing: bool -> normalization
    transform: list -> [*x scaler from training*, *y scaler from training*] only take effect when testing
    '''
    def __init__(self, X_path, y_path, preprocessing=False, transform=None):
        self.data = np.loadtxt(X_path)
        self.label = np.loadtxt(y_path)
        if len(self.label.shape) == 1:
            self.label = self.label.reshape(-1, 1)
        if preprocessing == "y":
            #self.X_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
            #self.X_scaler.fit(self.data)
            self.y_scaler.fit(self.label)
            #self.data = self.X_scaler.transform(self.data)
            self.label = self.y_scaler.transform(self.label)
        elif preprocessing == True:
            self.X_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
            self.X_scaler.fit(self.data)
            self.y_scaler.fit(self.label)
            self.data = self.X_scaler.transform(self.data)
            self.label = self.y_scaler.transform(self.label)
        if transform is not None:
            if len(transform) == 2:
                try:
                    self.data = transform[0].transform(self.data)
                except:
                    data_shape = self.data.shape
                    self.data = np.concatenate((self.data, np.zeros((data_shape[0], 43-data_shape[1]))), axis=1)
                    self.data = np.delete(transform[0].transform(self.data), range(data_shape[1], 43), 1)
                self.label = transform[1].transform(self.label)
            else:
                self.label = transform[0].transform(self.label)
            
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float(), torch.from_numpy(self.label[index]).float()


def loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, layer=5, hidden_num=[64, 128, 64, 16], dropout=[0.01, 0.1, 0.01, 0.01], activation='relu'):
        super(NeuralNetwork, self).__init__()
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = layer
        layers = []
        
        in_features = input_dim
        for i in range(n_layers-1):
            out_features = hidden_num[i]
            layers.append(nn.Linear(in_features, out_features))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'softmax':
                layers.append(nn.Softmax())
            p = dropout[i]
            layers.append(nn.Dropout(p))

            in_features = out_features
        layers.append(nn.Linear(in_features, 1))

        self.linear_relu_stack = nn.Sequential(*layers)
    def forward(self, x):
        predict = self.linear_relu_stack(x)
        return predict


class models(object):
    def __init__(self, loader, net, optimizer, criterion):
        self.net = net
        self.loader = loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_loss = None
        self.validation_loss = None

    def train(self, epoch):
        # init net
        data_loader = self.loader
        net = self.net
        # net = NeuralNetwork(input_dim, output_dim, hidden_dim=hidden_dims)

        # optimizer
        optimizer = self.optimizer
        criterion = self.criterion
        net.train()

        loss_history = []
        bar = tqdm(range(epoch))
        for _ in bar:
            running_loss = 0
            for inputs, target in data_loader:
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, target)
                loss.backward()
                running_loss += loss.item()
                optimizer.step()

            loss_history.append(running_loss / len(data_loader))
            bar.set_description(f"Loss={running_loss / len(data_loader):.5f}")
        self.training_loss = {"mse":loss_history[-1]}
        fig = plt.figure()
        plt.plot(loss_history)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss vs Epoch")

        plt.show()

        return net

    @classmethod
    def val(cls, data_loader, net=None, loss_list = {}):
        net.eval()
        mse_loss = 0
        lossFunc = nn.MSELoss()
        predict = []
        origin = []
        self.validation_loss = {}
        for inputs, target in data_loader:
            batch_size = inputs.size()[0]
            outputs = net(inputs)
            predict.append(outputs.detach().numpy()[0][0])
            origin.append(target.numpy()[0][0])
        if loss_list is not None:
            for loss_name, loss_function in loss_list.item():
                self.validation_loss[loss_name] = loss_function(predict, origin)
                print(f"{loss_name}: {self.validation_loss[loss_name]}")
            else:
                print(f"mse: {lossFunc(predict, origin)}")
        return predict, origin


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
