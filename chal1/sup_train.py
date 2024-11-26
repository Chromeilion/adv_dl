from matplotlib import pyplot as plt
import random
import math
from torchvision.datasets import FashionMNIST
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import torch
import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class BasicBlock(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = self.layer1(x)
        y = self.relu(y)
        return self.relu(x + self.layer2(y))


class BasicCNN(nn.Module):
    def __init__(self, n_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_channels_block1 = 32

        self.input = nn.Conv2d(1, n_channels_block1, 9, 1, padding=4, bias=False)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer1 = nn.Conv2d(n_channels_block1, n_channels_block1*2, 5, 1, padding=2, bias=False)
        self.layer2 = nn.Conv2d(n_channels_block1*2, n_channels_block1*4, 3, 1, padding=1, bias=False)
        self.layer3 = nn.Conv2d(n_channels_block1*4, n_channels_block1*8, 3, 1, padding=1, bias=False)
        self.head = nn.Linear(n_channels_block1*8, n_classes)

    def forward(self, x):
        x = self.relu(self.input(x[: , None, ...]))
        x = self.maxpool(x)
        x = self.relu(self.layer1(x))
        x = self.maxpool(x)
        x = self.relu(self.layer2(x))
        x = self.maxpool(x)
        x = self.relu(self.layer3(x))
        x = x.mean(dim=(2, 3)).squeeze()
        return self.head(x)


def train_cnn(x_train: DataLoader, x_test: DataLoader):
    model = BasicCNN(n_classes=10)
    model.to(device)
    model.train()
    optim = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)
    criterion = CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)

    step_counter = 0
    eval_every = 2000
    all_loss = []
    eval_loss = []
    lrs = []
    all_eval_steps = []
    for _ in tqdm.tqdm(range(50), desc="Epoch"):
        for x, y in x_train:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()
            if step_counter % eval_every == 0:
                all_eval_steps.append(step_counter)
                all_loss.append(loss.item())
                with torch.no_grad():
                    t_loss = 0
                    for x, y in x_test:
                        x, y = x.to(device), y.to(device)
                        pred = model(x)
                        loss = criterion(pred, y)
                        t_loss += loss.cpu().item()
                    lrs.append(scheduler.get_last_lr()[0])
                    eval_loss.append(t_loss/len(x_test))

            step_counter += 1
        scheduler.step()

    model.eval()
    test_preds = []
    for x, y in x_test:
        x, y = x.to(device), y.to(device)
        pred = torch.argmax(nn.functional.sigmoid(model(x)), dim=1)
        test_preds += list(zip(pred.cpu().numpy().tolist(), y.cpu().numpy().tolist()))

    return test_preds, all_loss, eval_loss, lrs, all_eval_steps


class BasicDS(Dataset):
    def __init__(self, x, y):
        dim = round(math.sqrt(x.shape[1]))
        self.x = torch.tensor(x.astype(np.float32).reshape(-1, dim, dim)).to(device)
        self.y = torch.tensor(y).long().to(device)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


def main():
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    saveloc = Path("./figs")
    saveloc.mkdir(exist_ok=True)
    seed = 42
    f_mnist = FashionMNIST(".", download=True)
    x, y = f_mnist.data, f_mnist.targets
    x = x.reshape(x.shape[0], -1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_n = scaler.transform(x_train)
    x_test_n = scaler.transform(x_test)

    svm = SVC(kernel="rbf")
    svm.fit(x_train_n, y_train)
    svm_preds = svm.predict(x_test_n)

    svm_f1 = classification_report(y_test, svm_preds)
    print("SVM scores on kmeans test set:\n", svm_f1)

    mlp = MLPClassifier(solver="sgd", learning_rate_init=0.05, tol=1e-3, early_stopping=True)
    mlp.fit(x_train_n, y_train)
    mlp_preds = mlp.predict(x_test_n)

    mlp_f1 = classification_report(y_test, mlp_preds)
    print("MLP 1 scores on kmeans test set:\n", mlp_f1)

    fig, ax = plt.subplots()
    ax.plot(mlp.loss_curve_)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    fig.savefig(saveloc/"mlp_true.png")

    mlp2 = MLPClassifier(solver="sgd", learning_rate_init=0.05, tol=1e-3, hidden_layer_sizes=(125, 64, 32, 16), early_stopping=True)
    mlp2.fit(x_train_n, y_train)
    mlp2_preds = mlp2.predict(x_test_n)

    mlp2_f1 = classification_report(y_test, mlp2_preds)
    print("MLP 2 scores on kmeans test set:\n", mlp2_f1)

    fig, ax = plt.subplots()
    ax.plot(mlp2.loss_curve_)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    fig.savefig(saveloc/"mlp2_true.png")

    batch_size = 32
    train_dl = DataLoader(BasicDS(x_train_n, y_train), batch_size=batch_size)
    test_dl = DataLoader(BasicDS(x_test_n, y_test), batch_size=batch_size,
                         shuffle=False)

    cnn_preds, all_loss, eval_loss, lrs, all_eval_steps = train_cnn(train_dl, test_dl)

    cnn_f1 = classification_report([i[0] for i in cnn_preds], [i[1] for i in cnn_preds])
    print(f"CNN scores on test set: {cnn_f1}")

    fig, ax = plt.subplots()
    ax.plot(all_eval_steps, all_loss, label="Train loss")
    ax.plot(all_eval_steps, eval_loss, label="Eval. loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    fig.legend()
    fig.savefig(saveloc/"cnn_true.png")

    bet_lr, bet_lr_x = [], []
    p_step = 0
    for step, lr in zip(all_eval_steps, lrs):
        bet_lr += [lr, lr]
        bet_lr_x += [p_step, step]
        p_step = step

    fig, ax = plt.subplots()
    ax.plot(bet_lr_x, bet_lr)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    fig.savefig(saveloc/"cnn_lr_true.png")


if __name__ == '__main__':
    main()
