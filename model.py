import torch
import torch.nn as nn
from torch_geometric.nn import LGConv

class DIG(nn.Module):
    def __init__(self, geo_embed_dim,int_embed_dim, layer_num,user_num,item_num):
        super(DIG, self).__init__()
        self.geo_embed_dim = geo_embed_dim
        self.int_embed_dim = int_embed_dim
        self.layer_num = layer_num
        self.user_num = user_num
        self.item_num = item_num
        self.convs_int = nn.ModuleList(LGConv() for _ in range(layer_num))
        self.convs_geo = nn.ModuleList(LGConv() for _ in range(layer_num))

        self.tanh = nn.Tanh()

        self.user_int = nn.Embedding(user_num,int_embed_dim)
        self.user_geo = nn.Embedding(user_num,geo_embed_dim)
        self.item_int = nn.Embedding(item_num,int_embed_dim)
        self.item_geo = nn.Embedding(item_num,geo_embed_dim)
        
        nn.init.normal_(self.user_int.weight, std=0.01)
        nn.init.normal_(self.user_geo.weight, std=0.01)
        nn.init.normal_(self.item_int.weight, std=0.01)
        nn.init.normal_(self.item_geo.weight, std=0.01)

    def bpr_loss(self,emb_users_final, emb_pos_items_final, emb_neg_items_final, dist):
        pos_ratings = torch.mul(emb_users_final, emb_pos_items_final).sum(dim=-1)
        neg_ratings = torch.mul(emb_users_final, emb_neg_items_final).sum(dim=-1)

        bpr_loss = torch.mean(torch.nn.functional.logsigmoid(pos_ratings - neg_ratings) * dist)
        return -bpr_loss 
    def forward(self, edge_index):
        int_emb = torch.cat([self.user_int.weight, self.item_int.weight])
        int_embs = [int_emb]

        for conv in self.convs_int:
            int_emb = conv(x=int_emb, edge_index=edge_index)
            int_embs.append(int_emb)

        int_emb_result = 1/(self.layer_num+1) * torch.mean(torch.stack(int_embs, dim=1), dim=1)
        int_emb_users, int_emb_items = torch.split(int_emb_result, [self.user_num, self.item_num])

        geo_emb = torch.cat([self.user_geo.weight, self.item_geo.weight])
        geo_embs = [geo_emb]

        for conv in self.convs_geo:
            geo_emb = conv(x=geo_emb, edge_index=edge_index)
            geo_embs.append(geo_emb)

        geo_emb_result = 1/(self.layer_num+1) * torch.mean(torch.stack(geo_embs, dim=1), dim=1)
        geo_emb_users, geo_emb_items = torch.split(geo_emb_result, [self.user_num, self.item_num])
        return int_emb_users, int_emb_items , geo_emb_users, geo_emb_items
    
