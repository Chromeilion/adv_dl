import os
import matplotlib.pyplot as plt
import torch
import  torch.nn as nn
from torch.utils.data import Dataset
from torch.multiprocessing import set_start_method
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42


def B6(x):
    return x[0]**6+15*x[1]*x[0]**4+20*x[2]*x[0]**3+45*x[1]**2*x[0]**2+15*x[1]**3+60*x[2]*x[1]*x[0]+15*x[3]*x[0]**2+10*x[2]**2+15*x[3]*x[1]+6*x[4]*x[0]+x[5]


def B6_tilde(x):
    perm = [1, 4, 5, 0, 3, 2]
    x = x[perm, :]
    return B6(x)


class ResBlock(nn.Module):
    """Simple ResNet type block with only one linear layer.
    """
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x+self.linear(x))


class FCResNet(nn.Module):
    """Fully connected neural net with residual connections.
    """
    def __init__(self, in_size: int = 6, out_size: int = 1, hidden_dim: int = 50, n_blocks: int = 8):
        super(FCResNet, self).__init__()
        self.input_projection = nn.Linear(in_size, hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, out_size)

    def forward(self, x):
        # Map input to the hidden dimension
        x = self.relu(self.input_projection(x))
        # Apply residual blocks
        x = self.blocks(x)
        # Map from hidden dimension to output logits
        return self.head(x).squeeze()


class B6Dataset(Dataset):
    def __init__(self, x, y):
        super(B6Dataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def train_model(model, optimargs, epochs: int, train_ds,
                test_ds, batch_size: int = 128, eval_every: int = 500):
    model = model.to(device)
    # Compiling the model speeds it up a lot because of the sequential nature
    # of the residual blocks (essentially a for loop), which gets removed.
    model = torch.compile(model, mode="reduce-overhead")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), **optimargs)
    dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=os.cpu_count()-1,
        pin_memory=True,
        prefetch_factor=3
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=os.cpu_count()-1,
        pin_memory=True,
        prefetch_factor=3
    )
    train_losses = []
    test_losses = []
    step = 0
    for epoch in tqdm(range(epochs), desc="epoch", total=epochs):
        for x, y in tqdm(dataloader, desc="Training model"):
            x, y = x.to(device), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % eval_every == 0:
                with torch.no_grad():
                    train_losses.append((step, loss.item()))
                    total_test_loss = 0
                    for x, y in test_dataloader:
                        x, y = x.to(device), y.to(device)
                        test_pred = model(x)
                        total_test_loss += criterion(test_pred, y).item() / len(test_dataloader)

                    test_losses.append((step, total_test_loss))
            step += 1

    return test_losses, train_losses


def generate_data(n_points: int, fn):
    x = torch.zeros((6, n_points)).uniform_(0, 2)
    y = fn(x)
    return x.T.contiguous(), y


def ds_train_workflow(dataset_fn, train_dataset_size: int, test_dataset_size: int, train_args):
    test_ds = B6Dataset(*generate_data(n_points=test_dataset_size, fn=dataset_fn))
    train_ds = B6Dataset(*generate_data(n_points=train_dataset_size, fn=dataset_fn))
    model = FCResNet()
    test_loss, train_loss = train_model(
        model=model,
        test_ds=test_ds,
        train_ds=train_ds,
        **train_args
    )
    return {
        "model": model,
        "test_loss": test_loss,
        "train_loss": train_loss
    }



def main():
    torch.manual_seed(seed)

    train_ds_size = int(1e5)
    test_ds_size = int(6e4)

    optimargs = {
        "lr": 1e-3,
    }
    train_args = {
        "optimargs": optimargs,
        "epochs": 5,
        "batch_size": 20,
        "eval_every": 5000
    }
    workflows = {
        "B6": {
            "dataset_fn": B6,
            "train_dataset_size": train_ds_size,
            "test_dataset_size": test_ds_size,
            "train_args": train_args,
        },
        "B6_tilde": {
            "dataset_fn": B6_tilde,
            "train_dataset_size": train_ds_size,
            "test_dataset_size": test_ds_size,
            "train_args": train_args,
        }
    }
    res = {}
    for model, workflowargs in workflows.items():
        res[model] = ds_train_workflow(**workflowargs)

        fig, ax = plt.subplots()
        ax.plot(*list(zip(*res[model]["train_loss"])), label="Train")
        ax.plot(*list(zip(*res[model]["test_loss"])), label="Test")
        fig.legend()
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training and Test Loss for {model}")
        fig.show()


if __name__ == '__main__':
    main()
