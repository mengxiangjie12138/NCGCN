import torch
import torch.linalg


def feature_distance(features_1, features_2):
    return torch.mm(features_1, features_2.T)


def euclidean_distance(features_1, features_2):
    return torch.sqrt(torch.sum(torch.pow((features_1.unsqueeze(1) - features_2), 2), 2))


def cos_distance(features_1, features_2):
    features_norm_1 = torch.linalg.norm(features_1, dim=1).reshape([1, len(features_1)])
    features_norm_2 = torch.linalg.norm(features_2, dim=1).reshape([1, len(features_2)])
    bottom = torch.mm(features_norm_1.T, features_norm_2)
    top = torch.mm(features_1, features_2.T)
    return top / bottom


def pearson_correlation(features_1, features_2):
    features_mean_1 = torch.mean(features_1, dim=1).unsqueeze(1)
    features_expand_1 = features_mean_1.expand([features_1.shape[0], features_1.shape[1]])
    features_1_ = features_2 - features_expand_1
    features_mean_2 = torch.mean(features_2, dim=1).unsqueeze(1)
    features_expand_2 = features_mean_2.expand([features_2.shape[0], features_2.shape[1]])
    features_2_ = features_2 - features_expand_2
    return cos_distance(features_1_, features_2_)


def distance(features_1, features_2, distance_mode='Euclidean', if_similarity=False):
    if distance_mode == 'Euclidean':
        if if_similarity:
            distance_matrix = torch.exp(euclidean_distance(features_1, features_2) * -1)
        else:
            distance_matrix = euclidean_distance(features_1, features_2)
    elif distance_mode == 'cos':
        distance_matrix = cos_distance(features_1, features_2)
    elif distance_mode == 'pearson_correlation':
        distance_matrix = pearson_correlation(features_1, features_2)
    elif distance_mode == 'features':
        distance_matrix = feature_distance(features_1, features_2)
    else:
        distance_matrix = None
    return distance_matrix


if __name__ == '__main__':
    features = torch.randn([10, 20])
    print(feature_distance(features, features).shape)








