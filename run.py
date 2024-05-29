from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import os
import datasets
from batches import *
from model import *
import time
import random
import validation as val
import torch.cuda
import torch
from save import save_intersection, save_experiment_result


# parser = ArgumentParser(description="SAE-NAD")
# parser.add_argument('-e', '--epoch', type=int, default=60, help='number of epochs for GAT')
# parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size for training')
# parser.add_argument('--alpha', type=float, default=2.0, help='the parameter of the weighting function')
# parser.add_argument('--epsilon', type=float, default=1e-5, help='the parameter of the weighting function')
# parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
# parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3, help='weight decay')
# parser.add_argument('-att', '--num_attention', type=int, default=20, help='the number of dimension of attention')
# parser.add_argument('--inner_layers', nargs='+', type=int, default=[200, 50, 200], help='the number of latent factors')
# parser.add_argument('-dr', '--dropout_rate', type=float, default=0.5, help='the dropout probability')
# parser.add_argument('-seed', type=int, default=0, help='random state to split the data')
# args = parser.parse_args()

class Args:
    def __init__(self):
        self.lr = 0.01# learning rate            
        self.lamda = 1e-8# model regularization rate
        self.epochs = 50 # training epoches
        self.topk = 50 # compute metrics@top_k
        self.batch_size = 2048
        self.factor_num = 128 # predictive factors numbers in the model
        self.hidden_dim = 128 # predictive factors numbers in the model
        self.num_ng = 4 # sample negative items for training
        self.beta = 0.5
        self.areanum = 50

def train_DIG(train_matrix, test_positive, val_positive, dataset, arg=Args()):

    now = datetime.now()
    result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')+"DIG"
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    max_recall = 0.0
    k_list=[5, 10, 15, 20, 25, 30]

    args = Args()
    with open(result_directory+"/setting.txt","w") as setting_f:
        setting_f.write("lr:{}\n".format(str(args.lr)))
        setting_f.write("lamda:{}\n".format(str(args.lamda)))
        setting_f.write("epochs:{}\n".format(str(args.epochs)))
        setting_f.write("factor_num:{}\n".format(str(args.factor_num)))
        setting_f.write("hidden_dim:{}\n".format(str(args.hidden_dim)))
        setting_f.write("num_ng:{}\n".format(str(args.num_ng)))
        setting_f.write("dataset:{}\n".format(str(dataset.directory_path)))

    num_users = dataset.user_num
    num_items = dataset.poi_num
    model = DIG(96,32,2,num_users,num_items).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamda)

    edge_index = []
    for u in range(num_users):
        idx = train_matrix.getrow(u).indices
        for i in idx:
            edge_index.append((u,i+num_users))
            edge_index.append((i+num_users,u))
    edge_index = torch.tensor(edge_index).to(DEVICE)
    tanh = nn.Tanh()
    std_distance = 2.0
    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())
        if (e+1)%8 == 0:
            std_distance *= 1.1
        idx = list(range(num_users))    
        random.shuffle(idx)
        uid , pos_i, neg_j, distance = get_DIG_batch(train_matrix, num_items, idx, args.num_ng, dataset.dist_matrix, std_distance)
        bsize = []
        for tt in range(int(len(uid)/args.batch_size)):
            bsize.append([args.batch_size*tt,args.batch_size*(tt+1)])
        if (len(uid)/args.batch_size > int(len(uid)/args.batch_size)):
            bsize.append([args.batch_size*int(len(uid)/args.batch_size),len(uid)])

        for start,end in bsize:
            optimizer.zero_grad() # 그래디언트 초기화
            
            int_emb_users, int_emb_items , geo_emb_users, geo_emb_items = model(edge_index.T)
            #if buid == 0:
            #    print(f"prediction : {prediction.shape}, {prediction}")
            loss_int = model.bpr_loss(int_emb_users[uid[start:end]],int_emb_items[pos_i[start:end]],int_emb_items[neg_j[start:end]],(1-tanh(distance[start:end])))
            loss_geo = model.bpr_loss(geo_emb_users[uid[start:end]],geo_emb_items[pos_i[start:end]],geo_emb_items[neg_j[start:end]],tanh(distance[start:end]))
            loss_total = model.bpr_loss(torch.concat([int_emb_users[uid[start:end]],geo_emb_users[uid[start:end]]],dim=-1),torch.concat([int_emb_items[pos_i[start:end]],geo_emb_items[pos_i[start:end]]],dim=-1),torch.concat([int_emb_items[neg_j[start:end]],geo_emb_items[neg_j[start:end]]],dim=-1),1)
            
            loss = loss_total + 0.1*loss_int + 0.0001*loss_geo
            train_loss += loss.item()
            loss.backward() # 역전파 및 그래디언트 계산
            optimizer.step() # 옵티마이저 업데이트
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss))
        if (e+1)%5 == 0:
            model.eval() # 모델을 평가 모드로 설정
            with torch.no_grad():
                start_time = int(time.time())
                precision_v, recall_v, hit_v, precision_t, recall_t, hit_t = val.DIG_validation(model,args,num_users,test_positive,val_positive,train_matrix,k_list,edge_index)
                
                if(max_recall < recall_v[1]):
                    max_recall = recall_v[1]
                    save_experiment_result(result_directory,[recall_t,precision_t,hit_t],k_list,e+1)
                end_time = int(time.time())
                print("eval time: {} sec".format(end_time-start_time))



def run(dataset,arg):
    train_matrix, test_positive, val_positive, place_coords = dataset.generate_data(0,args.areanum)
    print("train data generated")
    
    print("train start")
    train_DIG(train_matrix, test_positive, val_positive, dataset,arg)
if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed=0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # dataset_ = datasets.Dataset(3725,10768,"./data/Tokyo/")
    # args = Args()
    # run(dataset_,args)

    dataset_ = datasets.Dataset(15359,14586,"./data/Yelp/")
    args = Args()
    run(dataset_,args)

    # dataset_ = datasets.Dataset(6638,21102,"./data/NewYork/")
    # args = Args()
    # run(dataset_,args)
