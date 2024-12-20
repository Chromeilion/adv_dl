import torchvision
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.utils.random import sample_without_replacement
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

def main():
    seed = 43
    saveloc = Path("./figs")
    saveloc.mkdir(exist_ok=True)

    n_components = 10
    n_samples = 5000
    f_mnist = torchvision.datasets.FashionMNIST(".", download=True)
    x = f_mnist.data
    x = x.reshape(x.shape[0], -1)
    y = f_mnist.targets
    sampling = sample_without_replacement(n_population=y.shape[0], n_samples=n_samples, random_state=seed)
    x, y = x[sampling], y[sampling]

    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    k_pca = KernelPCA(kernel="rbf", gamma=0.001, random_state=seed)
    x_down = k_pca.fit_transform(x)
    k_pca_out = saveloc/"k_pca"
    k_pca_out.mkdir(exist_ok=True)
    plot_vals(x_down, y, k_pca_out)

    l_pca = KernelPCA(kernel="cosine", random_state=seed)
    x_down = l_pca.fit_transform(x)
    l_pca_out = saveloc/"l_pca"
    l_pca_out.mkdir(exist_ok=True)
    plot_vals(x_down, y, l_pca_out)
    
    pca = PCA(n_components=n_components)
    x_down = pca.fit_transform(x)
    pca_out = saveloc/"pca"
    pca_out.mkdir(exist_ok=True)
    plot_vals(x_down, y, pca_out)

    pca = PCA()
    pca.fit(x)

    explained_variances = pca.explained_variance_ratio_[:25]
    fig, ax = plt.subplots()
    ax.plot(list(range(explained_variances.shape[0])), explained_variances)
    ax.set_xlabel("Component")
    ax.set_ylabel("Percentage Variance")
    fig.savefig(saveloc/"pca_var.png")



def plot_vals(x_down, y, saveloc):
    fig, ax = plt.subplots(tight_layout=True)
    for i in torch.unique(y):
        ax.scatter(x_down[y==i, 0], x_down[y==i, 1],
                   label=f"Class {i}", alpha=0.5)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    fig.savefig(saveloc/"2D_FMNIST.png")

    fig = plt.figure(tight_layout=True, figsize=(8.3, 8))
    ax = fig.add_subplot(projection='3d')
    for i in torch.unique(y):
        ax.scatter(x_down[y==i, 0], x_down[y==i, 1], x_down[y==i, 2],
                   label=f"Class {i}", alpha=0.5)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    fig.savefig(saveloc/"3D_FMNIST.png")

if __name__ == '__main__':
    main()
