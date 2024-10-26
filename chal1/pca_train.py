import numpy as np
from matplotlib import pyplot as plt
from torchvision.datasets import FashionMNIST
from sklearn.decomposition import PCA
from sklearn.utils.random import sample_without_replacement
from sklearn.cluster import KMeans
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score


def main():
    saveloc = Path("./figs")
    saveloc.mkdir(exist_ok=True)

    n_components = 10
    n_samples = 1000
    f_mnist = FashionMNIST(".")
    x, y = f_mnist.data, f_mnist.targets
    n_clusters = np.unique(y).shape[0]
    x = x.reshape(x.shape[0], -1)
    sampling = sample_without_replacement(y.shape[0], n_samples)
    x, y = x[sampling], y[sampling]
    downprojected = PCA(n_components=n_components).fit_transform(x)

    preds = KMeans(n_clusters=n_clusters).fit_predict(downprojected)

    fig, ax = plt.subplots()
    for i in np.unique(preds):
        ax.scatter(downprojected[preds==i, 0], downprojected[preds==i, 1], label=i)
    fig.legend()
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    fig.suptitle("K Means Clustering on PCA Downprojection")
    fig.savefig(saveloc/"kmeans.png")

    # !!!!!!!!!!
    # Remember to write down the details of this in the report
    # !!!!!!!!!!
    score = normalized_mutual_info_score(y, preds)
    print(score)

if __name__ == '__main__':
    main()
