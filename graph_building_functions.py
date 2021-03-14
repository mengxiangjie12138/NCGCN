import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph

import distance_each_other


def process_graph_numpy(adj):
    A = np.asmatrix(adj)
    I = np.eye(len(A))
    A_hat = A + I
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = D_hat**0.5
    D_hat = np.matrix(np.diag(D_hat))
    D_hat = D_hat**-1
    return A_hat, D_hat


def process_graph_torch(adj):
    A = adj
    I = torch.eye(len(A))
    A_hat = A + I
    D_hat = torch.sum(A_hat, dim=0)
    D_hat = D_hat ** 0.5
    D_hat = torch.diag(D_hat ** -1)
    return A_hat, D_hat


def supervised_graph(labels):
    ones = torch.ones((len(labels), len(labels)))
    zeros = torch.zeros((len(labels), len(labels)))
    labels = labels.unsqueeze(0)
    labels_expand = labels.expand(labels.shape[1], labels.shape[1])
    labels_matrix = (labels_expand - labels_expand.T).long()  # label相同则为0，label不同则不为0
    adj = torch.where(labels_matrix == 0, ones, zeros)
    return adj


# 限制ϵ半径建图算法中的ϵ
def supervised_get_radius(features, labels):
    supervised_matrix = supervised_graph(labels)
    distance_matrix = distance_each_other.distance(features, features)
    radius_limit = torch.sum(torch.mul(supervised_matrix, distance_matrix)) / torch.sum(supervised_matrix)
    return radius_limit


# ϵ半径建图
def radius_graph(features, dis=None, labels=None):
    if labels is not None:
        dis = supervised_get_radius(features, labels)
    distance_matrix = distance_each_other.distance(features, features)
    ones = torch.ones((len(features), len(features)))
    zeros = torch.zeros((len(features), len(features)))
    adj = torch.where(distance_matrix <= dis, ones, zeros)
    return adj


# knn与ϵ半径组合建图
def knn_and_radius_graph(features, k, dis=None, labels=None):
    adj_knn = torch.tensor(kneighbors_graph(features.data.numpy(), k, include_self=True).toarray())
    if labels is not None:
        features_label = features[:len(labels)]
        dis = supervised_get_radius(features_label, labels)
    adj_radius = radius_graph(features, dis)
    radius_node_numbers = torch.sum(adj_radius, dim=1).unsqueeze(0)
    radius_node_number_matrix = radius_node_numbers.expand(radius_node_numbers.shape[1], radius_node_numbers.shape[1])
    adj = torch.where(radius_node_number_matrix > k, adj_radius, adj_knn.to(torch.float32))
    if labels is not None:
        return adj, dis
    else:
        return adj


if __name__ == '__main__':
    labels = torch.tensor([0., 1., 2., 3., 2.]).unsqueeze(0)
    # a = torch.sqrt(torch.sum((labels.unsqueeze(0) - labels) ** 2, 2))
    result = labels.expand(labels.shape[1], labels.shape[1])
    # result = result - result.T

    print(labels.shape)












