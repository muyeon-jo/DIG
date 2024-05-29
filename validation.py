from batches import *
import torch.cuda
import eval_metrics
def DIG_validation(model, args,num_users, test_positive, val_positive, train_matrix, k_list, edge_index):
    model.eval() # 모델을 평가 모드로 설정
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    recommended_list = []
    for user_id in range(num_users):
        history = train_matrix.getrow(user_id).indices.tolist()
        target_list = list(set(range(train_matrix.shape[1])) - set(history))
        user_tensor = torch.LongTensor([user_id] * (len(target_list))).to(DEVICE)
        target_tensor = torch.LongTensor(target_list).to(DEVICE)

        int_emb_users, int_emb_items , geo_emb_users, geo_emb_items = model(edge_index.T)
        emb_users_final = torch.concat((int_emb_users[user_tensor],geo_emb_users[user_tensor]),dim=-1)
        emb_items_final = torch.concat((int_emb_items[target_tensor],geo_emb_items[target_tensor]),dim=-1)
        prediction = torch.mul(emb_users_final, emb_items_final).sum(dim=-1)
        _, indices = torch.topk(prediction, args.topk)
        recommended_list.append([target_list[i] for i in indices])

    
    precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
    precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
    
    return precision_v, recall_v, hit_v, precision_t, recall_t, hit_t