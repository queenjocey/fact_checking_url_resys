import torch
import torch.nn as nn
import torch.nn.functional as F

from common import utils
from common import highway


class FixedEmbedding(nn.Module):
    # embed_mat: 2d list
    def __init__(self, embed_mat, drop_ratio=0.1, requires_grad=False):
        super(FixedEmbedding, self).__init__()
        emb_weight = torch.FloatTensor(embed_mat)
        self.fix_embedding = nn.Embedding(num_embeddings=emb_weight.shape[0],
                                           embedding_dim=emb_weight.shape[1])
        self.fix_embedding.weight = nn.Parameter(emb_weight)
        self.fix_embedding.weight.requires_grad = requires_grad

        self.drop_ratio = drop_ratio
        if self.drop_ratio > 0:
            self.dropout = nn.Dropout(self.drop_ratio)

    def forward(self, x):
        x = self.fix_embedding(x)
        if self.drop_ratio > 0:
            x = self.dropout(x)
        return x


class NormalEmbedding(nn.Module):
    def __init__(self, num_embed, embed_dim, drop_ratio=0):
        super(NormalEmbedding, self).__init__()

        self.embed = nn.Embedding(num_embeddings=num_embed, embedding_dim=embed_dim)
        utils.init_embedding(self.embed)
        #utils.init_noraml(self.embed)

        self.drop_ratio = drop_ratio
        if self.drop_ratio > 0:
            self.dropout = nn.Dropout(self.drop_ratio)

    def forward(self, x):
        x = self.embed(x)
        if self.drop_ratio > 0:
            x = self.dropout(x)
        return x


class CharacterEmbedding(nn.Module):
    def __init__(self, num_embed=70, embed_dim=64, out_ch=64, kernel_size=3, padding=1):
        super(CharacterEmbedding, self).__init__()
        
        self.embedding_dim = embed_dim
        self.kernel_size = (1, kernel_size)
        self.padding = (0, padding)

        self.char_embedding = nn.Embedding(num_embeddings=num_embed, embedding_dim=embed_dim)
        #torch.nn.init.normal_(self.char_embedding.weight, mean=0, std=0.1)
        utils.init_embedding(self.char_embedding)

        self.char_conv = nn.Conv2d(in_channels=embed_dim,
                                   out_channels=out_ch,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding)


    '''torch.Size([128, 200, 16])
    torch.Size([128, 3200]) view
    torch.Size([128, 3200, 64]) embed
    torch.Size([128, 200, 16, 64]) view
    torch.Size([128, 200, 16, 171]) relu
    torch.Size([128, 200, 171]) max
    torch.Size([128, 200, 171])'''
    def forward(self, x):
        #batch_size = x.shape[0]
        #word_length = x.shape[-1]

        #x = x.view(batch_size, -1)
        #print(x.shape, 'view')
        x = self.char_embedding(x)  # (N, word_cnt, char_cnt, dim)
        #print(x.shape, 'embed')
        #x = x.view(batch_size, -1, word_length, self.embedding_dim)
        #print(x.shape, 'view')

        # embedding dim of characters is number of channels of conv layer
        
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.char_conv(x))
        x = x.permute(0, 2, 3, 1)
        # max pooling over word length to have final tensor
        x, _ = torch.max(x, dim=2)

        x = F.dropout(x, p=0.1, training=self.training)

        return x


class CharacterDepthwiseEmbedding(nn.Module):
    def __init__(self, num_embed=70, embed_dim=64, out_ch=64, kernel_size=5):
        super(CharacterDepthwiseEmbedding, self).__init__()
        
        self.embedding_dim = embed_dim
        self.kernel_size = (1, kernel_size)
        self.padding = (0, (kernel_size-1)//2)

        self.char_embedding = nn.Embedding(num_embeddings=num_embed, embedding_dim=embed_dim)
        #torch.nn.init.normal_(self.char_embedding.weight, mean=0, std=0.1)
        utils.init_embedding(self.char_embedding)

        self.depthwise_conv = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, 
                                        kernel_size=self.kernel_size, groups=embed_dim
                                        ,padding=self.padding, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels=embed_dim, out_channels=out_ch
                                        ,kernel_size=(1, 1), padding=(0, 0), bias=True)


    '''torch.Size([128, 200, 16])
    torch.Size([128, 3200]) view
    torch.Size([128, 3200, 64]) embed
    torch.Size([128, 200, 16, 64]) view
    torch.Size([128, 200, 16, 171]) relu
    torch.Size([128, 200, 171]) max
    torch.Size([128, 200, 171])'''
    def forward(self, x):
        batch_size = x.shape[0]
        word_length = x.shape[-1]

        x = self.char_embedding(x)

        # embedding dim of characters is number of channels of conv layer
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.pointwise_conv(self.depthwise_conv(x)))
        x = x.permute(0, 2, 3, 1)

        # max pooling over word length to have final tensor
        x, _ = torch.max(x, dim=2)

        x = F.dropout(x, p=0.1, training=self.training)
        return x


class FeatureEmbed(nn.Module):
    def __init__(self, features_num_embed=[], embed_dim=64, embed_drop_out=0.1):
        super(FeatureEmbed, self).__init__()
        self.features_embed_module = nn.ModuleList()
        for num_embed in features_num_embed:
            self.features_embed_module.append(NormalEmbedding(num_embed, embed_dim, embed_drop_out))

        self.linear_out = nn.Linear(len(features_num_embed)*embed_dim, embed_dim)
        utils.init_linear(self.linear_out)

    # 
    def forward(self, f_list):
        shape = f_list.shape
        dim = len(shape) - 1
        device = f_list.device.type
        if device == 'cuda':
            device_id = f_list.device.index
        else:
            device_id = -1

        features_embed = []
        for i in range(len(self.features_embed_module)):
            index = torch.tensor(i)
            if device_id > -1:
                index = index.cuda(device_id)
            input_tensor =  torch.index_select(f_list, dim=dim, index=index).view(shape[:-1])  # (N)
            f_embed = self.features_embed_module[i](input_tensor)     # (N, embed_dim)
            features_embed.append(f_embed)
        features_embed = torch.cat(features_embed, dim=dim) # (N, f_num*embed_dim)

       
        features_embed = torch.relu(self.linear_out(features_embed))         # (N, embed_dim)
        return features_embed