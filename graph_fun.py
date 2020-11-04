import numpy as np
import torch


def distance(features_1, features_2):
    # print(np.square(features_1 - features_2).shape)
    dist = torch.sqrt(torch.sum(torch.pow(features_1 - features_2, 2)))
    return dist


def knn_graph(features, k):
    adj = np.zeros([len(features), len(features)])
    distance_list = []
    distance_index_list = []
    for feature in features:
        dis_list = []
        for con_feature in features:
            # print(feature.shape)
            dis_list.append(distance(con_feature, feature))
        distance_list.append(dis_list)
        distance_index = np.array(dis_list)
        distance_index = distance_index.argsort()
        distance_index_list.append(distance_index)
    for index, distance_index in enumerate(distance_index_list):
        for i in range(k):
            adj[index][distance_index[i]] = 1
    return adj


def process_graph(adj):
    A = np.asmatrix(adj)
    I = np.eye(len(A))
    A_hat = A + I
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = D_hat**0.5
    D_hat = np.matrix(np.diag(D_hat))
    D_hat = D_hat**-1
    return A_hat, D_hat



