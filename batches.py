import random
import numpy as np
import torch

def get_DIG_batch(X, num_poi, batch_user_index, num_ng,dist_matrix, std_distance):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = []
    item_list = np.arange(num_poi).tolist()
    for uid in batch_user_index:
        positive = X.getrow(uid).indices
        random.shuffle(positive)
        negative = list(set(item_list)-set(positive))
        for i in range(len(positive)):
            temp = np.where(dist_matrix[positive[i]]<=std_distance)
            
            intersect = np.intersect1d(temp, np.array(negative,dtype=int))
            random.shuffle(negative)
            # negative = list(set(negative)-set(intersect))
            if(len(intersect)-1 > 0):
                ridx = np.random.randint(0,len(intersect))
                intersect = intersect[ridx]
                batch.append([uid,positive[i],intersect,dist_matrix[positive[i]][intersect]])
                for j in range(num_ng-1):
                    batch.append([uid,positive[i],negative[j],dist_matrix[positive[i]][negative[j]]])
            else:
                for j in range(num_ng):
                    batch.append([uid,positive[i],negative[j],dist_matrix[positive[i]][negative[j]]])
    random.shuffle(batch)
    batch = np.array(batch).T
    user = torch.LongTensor(batch[0]).to(DEVICE)
    item_i = torch.LongTensor(batch[1]).to(DEVICE)
    item_j = torch.LongTensor(batch[2]).to(DEVICE)
    dist = torch.tensor(batch[3]).to(DEVICE)
    return user, item_i, item_j, dist