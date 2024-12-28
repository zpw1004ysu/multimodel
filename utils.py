# utils.py
import pickle
import random
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.spatial.distance import cdist
from torch_geometric.data import Data


def set_seed(seed):
    """
    Sets random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(path, scaler=None):
    """
    Loads dataset and preprocesses it.
    """
    data_frame = pd.read_csv(path).dropna()
    X = data_frame[["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"]]
    y = data_frame["Facies"] - 1
    scaler = scaler or preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    return torch.FloatTensor(X), torch.LongTensor(y.values), scaler


def create_graph_data(X, y, threshold):
    """
    Creates graph data using PyTorch Geometric's Data structure.
    """
    distances = cdist(X, X, metric="euclidean")
    row, col = np.where((distances < threshold) & (distances > 0))
    edge_index = torch.from_numpy(np.vstack((row, col))).long()
    return Data(x=X, edge_index=edge_index, y=y)


def evaluate_metrics(y_true, y_pred):
    """
    Evaluates performance metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def write_predictions(path, predictions):
    """
    Writes predictions to a file.
    """
    with open(path, "w") as file:
        for pred in predictions:
            file.write(f"{pred}\n")


def save_matrix(path, matrix):
    """
    Saves a matrix using pickle.
    """
    with open(path, "wb") as file:
        pickle.dump(matrix, file)
