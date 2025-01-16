import torch
import torch.nn as nn
from abc import abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.layers = self._get_layers()
        self.initialize_weights()

    @abstractmethod
    def _get_layers(self):
        ...

    def forward(self, x):
        return self.layers(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1)
                m.bias.data.normal_(0.0, 1)


class Teacher(BaseModel):
    def _get_layers(self):
        layers = nn.Sequential(
            nn.Linear(100, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        return layers

class StudentU(BaseModel):
    def _get_layers(self):
        layers = nn.Sequential(
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        return layers


class StudentO(BaseModel):
    def _get_layers(self):
        layers = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )
        return layers


def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False


def generate_data(model, n_points: int):
    x = torch.zeros(n_points, 100, device=device).uniform_(0, 2)
    y = model(x)
    return x, y


def train_student(student, teacher, optimargs, n_iterations: int, test_x, test_y,
                  batch_size: int = 128, eval_every: int = 500):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(student.parameters(), **optimargs)
    train_losses = []
    test_losses = []
    for step in tqdm(range(n_iterations), total=n_iterations,
                  desc="Training student model"):
        with torch.no_grad():
            x, y = generate_data(teacher, batch_size)
        logits = student(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_every == 0:
            with torch.no_grad():
                train_losses.append((step, loss.item()))
                test_pred = student(test_x)
                test_loss = criterion(test_pred, test_y)
                test_losses.append((step, test_loss.item()))


    return test_losses, train_losses

def main():
    torch.manual_seed(seed)

    teacher_model = Teacher().to(device)
    freeze_weights(teacher_model)

    x_test, y_test = generate_data(teacher_model, int(6e4))

    student_u = StudentU().to(device)
    student_e = Teacher().to(device)
    student_o = StudentO().to(device)

    all_models_args = {
        "Student U": {
            "student": student_u,
            "optimargs": {"lr": 5e-3, "momentum": 0.9},
            "n_iterations": 1000,
        },
        "Student E": {
            "student": student_e,
            "optimargs": {"lr": 1e-7, "momentum": 0.9},
            "n_iterations": 4000,
        },
        "Student O": {
            "student": student_o,
            "optimargs": {"lr": 5e-13, "momentum": 0.9},
            "n_iterations": 5000,
        }
    }
    all_model_results = {}
    teacher_weights, teacher_biases = get_weights_biases_flat(teacher_model)
    teacher_weights_l, teacher_biases_l = get_weights_biases(teacher_model)
    for model_name, model_args in all_models_args.items():
        train_loss, test_loss = train_student(
            teacher=teacher_model,
            test_x=x_test,
            test_y=y_test,
            **model_args
        )
        student_weights, student_biases = get_weights_biases_flat(model_args["student"])
        student_weights_l, student_biases_l = get_weights_biases(model_args["student"])
        all_model_results[model_name] = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "student_weights": student_weights,
            "student_biases": student_biases,
            "student_weights_l": student_weights_l,
            "student_biases_l": student_biases_l
        }
        fig, ax = plt.subplots()
        ax.plot(*list(zip(*train_loss)), label="Train")
        ax.plot(*list(zip(*test_loss)), label="Test")
        fig.legend()
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training and Test Loss for {model_name}")
        fig.show()

    colors = ["black", "sienna", "pink", "lightsalmon",  "red"]
    fig, ax = plt.subplots(2, 4, sharey="row", tight_layout=True,
                           figsize=(8, 4))
    teacher_weights_np = [np.array(i) for i in teacher_weights_l]
    weights = [list(np.ones_like(i) / len(i)) for i in teacher_weights_np]
    ax[0, 0].hist(teacher_weights_l, weights=weights, stacked=True, color=colors[:len(weights)])
    ax[0, 0].set_xlabel("Weights")
    ax[0, 0].set_title("Teacher Weights")
    teacher_biases_np = [np.array(i) for i in teacher_biases_l]
    weights = [list(np.ones_like(i) / len(i)) for i in teacher_biases_np]
    ax[1, 0].hist(teacher_biases_l, weights=weights, stacked=True, color=colors[:len(weights)])
    ax[1, 0].set_xlabel("Biases")
    ax[1, 0].set_title("Teacher Biases")
    for idx, (model_name, model_args) in enumerate(all_models_args.items()):
        student_weights_np = [np.array(i) for i in all_model_results[model_name]["student_weights_l"]]
        weights = [list(np.ones_like(i) / len(i)) for i in student_weights_np]
        ax[0, idx+1].hist(all_model_results[model_name]["student_weights_l"], weights=weights, stacked=True, color=colors[:len(weights)])
        ax[0, idx+1].set_xlabel("Weights")
        ax[0, idx+1].set_title(f"{model_name} Weights")
        student_biases_np = [np.array(i) for i in all_model_results[model_name]["student_biases_l"]]
        weights = [list(np.ones_like(i) / len(i)) for i in student_biases_np]
        ax[1, idx+1].hist(all_model_results[model_name]["student_biases_l"], weights=weights, stacked=True, color=colors[:len(weights)])
        ax[1, idx+1].set_xlabel("Biases")
        ax[1, idx+1].set_title(f"{model_name} Biases")
    plt.show()

    fig, ax = plt.subplots(2, 4, sharey="row", tight_layout=True,
                           figsize=(8, 4))
    teacher_weights_np = np.array(teacher_weights)
    weights = np.ones_like(teacher_weights_np) / len(teacher_weights_np)
    ax[0, 0].hist(teacher_weights, weights=weights)
    ax[0, 0].set_xlabel("Weights")
    ax[0, 0].set_title("Teacher Weights")
    teacher_biases_np = np.array(teacher_biases)
    biases = np.ones_like(teacher_biases_np) / len(teacher_biases_np)
    ax[1, 0].hist(teacher_biases, weights=biases)
    ax[1, 0].set_xlabel("Biases")
    ax[1, 0].set_title("Teacher Biases")
    for idx, (model_name, model_args) in enumerate(all_models_args.items()):
        student_weights_np = np.array(all_model_results[model_name]["student_weights"])
        weights = np.ones_like(student_weights_np) / len(student_weights_np)
        ax[0, idx+1].hist(all_model_results[model_name]["student_weights"], weights=weights)
        ax[0, idx+1].set_xlabel("Weights")
        ax[0, idx+1].set_title(f"{model_name} Weights")
        student_biases_np = np.array(all_model_results[model_name]["student_biases"])
        biases = np.ones_like(student_biases_np) / len(student_biases_np)
        ax[1, idx+1].hist(all_model_results[model_name]["student_biases"], weights=biases)
        ax[1, idx+1].set_xlabel("Biases")
        ax[1, idx+1].set_title(f"{model_name} Biases")
    plt.show()


def get_weights_biases_flat(model):
    weights, biases = [], []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            weights.extend(list(m.weight.data.cpu().numpy().flatten()))
            biases.extend(list(m.bias.data.cpu().numpy().flatten()))
    return weights, biases


def get_weights_biases(model):
    weights, biases = [], []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            weights.append(list(m.weight.data.cpu().numpy().flatten()))
            biases.append(list(m.bias.data.cpu().numpy().flatten()))
    return weights, biases


if __name__ == '__main__':
    main()
