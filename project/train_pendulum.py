import os
from pathlib import Path

import numpy as np
from seft.model import SEFT, ModelConfig
from seft.trainer import Trainer, TrainerConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pendulum_generation import generate_pendulums

MEAN, STD = 0.002, 0.0021

def main(model_config, trainer_config, filepath = None, sample_rate=0.5, data_random_seed=0,):
    if filepath is None:
        filepath = "./pendulums"
        if not (Path(filepath)/"pend_regression.npz").exists():
            generate_pendulums(filepath, task="regression")

    img_sizes = (24, 24)
    W_p = img_sizes[0] // model_config.tubelet_size[1]
    H_p = img_sizes[1] // model_config.tubelet_size[2]

    train = Pendulum_regression(file_path=filepath,
                                name='pend_regression.npz',
                                mode='train', sample_rate=sample_rate,
                                random_state=data_random_seed,
                                img_patch_w=W_p,
                                img_patch_h=H_p,
                                mean=MEAN,
                                std=STD)
    test = Pendulum_regression(file_path=filepath, name='pend_regression.npz',
                               mode='test', sample_rate=sample_rate,
                               random_state=data_random_seed,
                               img_patch_w=W_p,
                               img_patch_h=H_p,
                               mean=MEAN,
                               std=STD)
    valid = Pendulum_regression(file_path=filepath,
                                name='pend_regression.npz',
                                mode='valid', sample_rate=sample_rate,
                                random_state=data_random_seed,
                                img_patch_w=W_p,
                                img_patch_h=H_p,
                                mean=MEAN,
                                std=STD)

    model = SEFT(config=model_config)
    model.set_loss_fn(PendulumMSELoss())
    model.set_head(PendulumHead(emb_dim=model_config.d_model))
    trainer = Trainer(trainer_config)
    trainer.train(train, valid, model)
    evaluate(model, test)

def evaluate(model, test):
    model.eval()
    test_dl = torch.utils.data.DataLoader(
        test,
        batch_size=50,
        shuffle=False,
        num_workers=os.cpu_count()-1
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    test_loss = 0.
    for modelargs in test_dl:
        modelargs = {key: value.to(device) for key, value in modelargs.items()}
        _, loss = model(**modelargs)
        test_loss += loss.cpu().item()

    test_loss /= len(test_dl)
    print(test_loss) # 5.8, seems realistic



def calc_mean_std(dataset):
    """Calculate mean and std of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=os.cpu_count()-1
    )
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in dataloader:
        mean += data["values"].mean()
        std += data["values"].std()
        nb_samples += data["values"].shape[0]

    return mean / nb_samples, std / nb_samples

class PendulumMSELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, logits, targets):
        return self.mse(logits, targets) * 10e-3


class PendulumHead(nn.Module):
    def __init__(self, emb_dim: int, n_frames: int = 50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Linear(emb_dim, 2)
        self.n_frames = n_frames

    def forward(self, logits):
        # drop cls token
        logits = logits[:, 1:, :]
        frames = logits.reshape(logits.shape[0], self.n_frames, -1, logits.shape[2])
        frame_means = frames.mean(dim=2).squeeze()
        return self.head(frame_means)


def collate_fn(batch):
    keys_of_interest = ["positions", "values", "label"]
    batch = [{key: i[key] for key in keys_of_interest} for i in batch]
    keys = list(batch[0].keys())
    labels = torch.stack([i["label"] for i in batch])
    keys.remove("label")
    pad_amount = max([i["values"].shape[0] for i in batch])
    p_batch = {key: torch.stack(
        [F.pad(i[key], (0, pad_amount - i[key].shape[-0])) for i in batch]) for key in keys
    }
    p_batch["label"] = labels
    return p_batch


class Pendulum_regression(Dataset):
    def __init__(self, file_path, name, mode, img_patch_h: int, img_patch_w: int,
                 mean=None, std=None, sample_rate=0.5, random_state=0):

        data = dict(np.load(os.path.join(file_path, name)))
        train_obs, train_targets, valid_obs, valid_targets, test_obs, test_targets, \
        train_time_points, valid_time_points, test_time_points = subsample(
            data, sample_rate=sample_rate, random_state=random_state)

        self.h = img_patch_h
        self.w = img_patch_w
        self.n_p_per_patch = self.h*self.w
        self.mean, self.std = mean, std

        if mode == 'train':
            self.obs = train_obs
            self.targets = train_targets
            self.time_points = train_time_points
        elif mode == 'valid':
            self.obs = valid_obs
            self.targets = valid_targets
            self.time_points = valid_time_points
        elif mode == 'test':
            self.obs = test_obs
            self.targets = test_targets
            self.time_points = test_time_points
        else:
            raise RuntimeError(f"Mode {mode} not recognised.")

        self.obs = np.ascontiguousarray(
            np.transpose(self.obs, [0, 1, 4, 2, 3]))/255.0

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.obs[idx, ...].astype(np.float64)).float()
        if self.mean is not None and self.std is not None:
            obs = (obs - self.mean) / self.std
        targets = torch.from_numpy(self.targets[idx, ...].astype(np.float64)).float()
        time_points = torch.from_numpy(self.time_points[idx, ...])
#        time_points *= self.n_p_per_patch
#        time_points = time_points.unsqueeze(0).repeat(self.n_p_per_patch, 1)
#        time_points += torch.arange(self.n_p_per_patch)[:, None]
#        time_points = time_points.flatten()
        h = torch.arange(self.h)
        w = torch.arange(self.w)

        return {"values": obs, "label": targets, "T": time_points, "H": h,
                "W": w}


def subsample(data, sample_rate, imagepred=False, random_state=0):
    train_obs, train_targets = data["train_obs"], data["train_targets"]
    valid_obs, valid_targets = data["valid_obs"], data["valid_targets"]
    test_obs, test_targets = data["test_obs"], data["test_targets"]

    seq_length = train_obs.shape[1]
    train_time_points = []
    valid_time_points = []
    test_time_points = []
    n = int(sample_rate*seq_length)

    if imagepred:
        train_obs_valid = data["train_obs_valid"]
        data_components = train_obs, train_targets, train_obs_valid
        train_obs_sub, train_targets_sub, train_obs_valid_sub = [np.zeros_like(x[:, :n, ...]) for x in data_components]

        valid_obs_valid = data["valid_obs_valid"]
        data_components = valid_obs, valid_targets, valid_obs_valid
        valid_obs_sub, valid_targets_sub, valid_obs_valid_sub = [np.zeros_like(x[:, :n, ...]) for x in data_components]

        test_obs_valid = data["test_obs_valid"]
        data_components = test_obs, test_targets, test_obs_valid
        test_obs_sub, test_targets_sub, test_obs_valid_sub = [np.zeros_like(x[:, :n, ...]) for x in data_components]

    else:
        data_components = train_obs, train_targets, valid_obs, valid_targets, test_obs, test_targets
        train_obs_sub, train_targets_sub, valid_obs_sub, valid_targets_sub, test_obs_sub, test_targets_sub = [
            np.zeros_like(x[:, :n, ...]) for x in data_components]

    for i in range(train_obs.shape[0]):
        rng_train = np.random.default_rng(random_state+i+train_obs.shape[0])  # NOTE - maybe we should change seeding to make this more flexible.
        choice = np.sort(rng_train.choice(seq_length, n, replace=False))
        train_time_points.append(choice)
        train_obs_sub[i, ...], train_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [train_obs, train_targets]]
        if imagepred:
            train_obs_valid_sub[i, ...] = train_obs_valid[i, choice, ...]

    for i in range(valid_obs.shape[0]):
        rng_valid = np.random.default_rng(random_state+i+valid_obs.shape[0])  # NOTE - maybe we should change seeding to make this more flexible.
        choice = np.sort(rng_valid.choice(seq_length, n, replace=False))
        valid_time_points.append(choice)
        valid_obs_sub[i, ...], valid_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [valid_obs, valid_targets]]
        if imagepred:
            valid_obs_valid_sub[i, ...] = valid_obs_valid[i, choice, ...]

    for i in range(test_obs.shape[0]):
        rng_test = np.random.default_rng(random_state+i)  # NOTE - maybe we should change seeding to make this more flexible.
        choice = np.sort(rng_test.choice(seq_length, n, replace=False))
        test_time_points.append(choice)
        test_obs_sub[i, ...], test_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [test_obs, test_targets]]
        if imagepred:
            test_obs_valid_sub[i, ...] = test_obs_valid[i, choice, ...]

    train_time_points, valid_time_points, test_time_points = np.stack(
        train_time_points, 0), np.stack(valid_time_points, 0), np.stack(test_time_points, 0)

    if imagepred:
        return train_obs_sub, train_targets_sub, train_time_points, train_obs_valid_sub, \
               valid_obs_sub, valid_targets_sub, valid_time_points, valid_obs_valid_sub, \
               test_obs_sub, test_targets_sub, test_time_points, test_obs_valid_sub
    else:
        return train_obs_sub, train_targets_sub, valid_obs_sub, valid_targets_sub, test_obs_sub, test_targets_sub, \
               train_time_points, valid_time_points, test_time_points


if __name__ == '__main__':
    model_config = ModelConfig(
        d_model=384,
        nhead=4,
        depth=4,
        tubelet_size=(1, 8, 8),
        max_len=10000,
        head_drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        drop_rate=0.
    )
    trainer_config = TrainerConfig(
        base_lr=5e-5,
        epochs=100,
        gradient_clip=300,
        eval_every=500,
        save_every=500,
        batch_size=32,
        warmup_steps=1000,
        optimizer="adamw",
        project_name="pendulum_regression"
    )
    main(model_config, trainer_config)
