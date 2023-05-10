import torch 
from torch import nn 
import torch.nn.functional as F 
from torch import optim 
from torch.autograd import Variable
torch.manual_seed(1)

import random
import numpy as np

from re_weighting.lstm_vae import LSTM_OneHotVAE

class DomainClassifier(nn.Module):
    def __init__(
        self,
        n_hidden,
        dim_z, # the dim of latent vector from VAE
        num_embeddings, # can be seen as the number of users
        embedding_dim, 
        dim_hidden # the dim of hidden vector of DC
    ):
        super(DomainClassifier, self).__init__()
        self.n_hidden = n_hidden
        self.embedding = nn.Embedding(
            num_embeddings = num_embeddings,
            embedding_dim = embedding_dim
        )
        self.input_net = nn.Linear(embedding_dim + dim_z, dim_hidden)
        self.hidden_net = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for i in range(n_hidden - 1) ]
        )
        self.output_net = nn.Linear(dim_hidden, 1)
    
    def forward(self, u, z):
        # print(u.shape)
        # u = u.squeeze()
        user_emb = self.embedding(u)
        zx = torch.cat((user_emb, z), 1)
        zy = F.elu(self.input_net(zx))
        for i in range(self.n_hidden - 1):
            zy = F.elu(self.hidden_net[i](zy))
        y = torch.sigmoid(self.output_net(zy))
        return y


def getDomainClassifier(lstm_vae, U, C, cfgs):
    user_num = cfgs['global_user_num']
    campaign_size = cfgs['global_campaign_size']
    epoch_num = cfgs['dc_epoch_num']
    device = cfgs['device']

    hidden_size = cfgs['dc_hidden_size']
    hidden_layer = cfgs['dc_hidden_layer']
    num_embeddings = user_num
    embedding_dim = cfgs['dc_user_embedding_dim']
    dc = DomainClassifier(
        hidden_layer,
        lstm_vae.latent_variable_size,
        num_embeddings,
        embedding_dim,
        hidden_size
    ).to(device)

    optimizer = optim.Adam(dc.parameters(), lr = cfgs['dc_learning_rate'])
    batch_size = cfgs['dc_batch_size']


    bceloss = nn.BCELoss(reduction = 'mean')

    for ep in range(epoch_num):
        random.seed(ep)
        random.shuffle(C)
        random.seed(ep)
        random.shuffle(U)

        cl_losses = []
        train_index = 0
        while train_index < len(C):
            input_batch = C[train_index : train_index + batch_size
                                if train_index + batch_size < len(C)
                                else len(C)]
            batch_C, batch_lens = [], []
            for seq in input_batch:
                batch_lens.append(len(seq))
            for seq in input_batch:
                seq_list = []
                for i in range(max(batch_lens)):
                    cnt_1hot = [0] * (campaign_size + 1)
                    if i < len(seq):
                        cnt_1hot[seq[i]] = 1
                    else:
                        cnt_1hot[-1] = 1 # pad one hot vector, manually
                    seq_list.append(cnt_1hot)
                batch_C.append(seq_list)
            batch_C = torch.Tensor(batch_C).to(device)

            encoded_x, mu, logvar, z, h_state, decoder_output, recon_x = lstm_vae(batch_C, batch_lens, False)
            batch_z = mu + torch.exp(0.5 * logvar).to(device)  * torch.randn(size = mu.size()).to(device) # we can alse set true and use z
            batch_z_neg = torch.randn(size = batch_z.size()).to(device)

            batch_u = U[train_index : train_index + batch_size
                            if train_index + batch_size < len(U)
                            else len(U)]
            batch_u = torch.LongTensor(batch_u).to(device)
            batch_u = torch.cat((batch_u, batch_u), dim = 0).to(device)
            batch_z = torch.cat((batch_z, batch_z_neg), dim = 0).to(device)
            label_batch = torch.cat((torch.zeros(len(batch_C), 1), torch.ones(len(batch_C), 1)), dim = 0).to(device)

            pre_d = dc(batch_u, batch_z)

            loss = bceloss(pre_d, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cl_losses.append(loss.item())
            train_index += batch_size

        print("Epoch {}".format(ep))
        print("Avg loss {}".format(sum(cl_losses) / len(cl_losses)    ))
        # print(cl_losses)

    return dc 


def calcSampleWeights(U, C, lstm_vae, dc, cfgs):
    campaign_size = cfgs['global_campaign_size']
    device = cfgs['device']
    batch_size = cfgs['vae_batch_size']   #dc_batch_size
    calc_weight_nums = cfgs['calcl_weight_nums']

    n = len(U)
    weight = np.zeros(n)
    train_index = 0
    while train_index < len(C):
        input_batch = C[train_index : train_index + batch_size
                            if train_index + batch_size < len(C)
                            else len(C)]
        batch_C, batch_lens = [], []
        for seq in input_batch:
            batch_lens.append(len(seq))
        for seq in input_batch:
            seq_list = []
            for i in range(max(batch_lens)):
                cnt_1hot = [0] * (campaign_size + 1)
                if i < len(seq):
                    cnt_1hot[seq[i]] = 1
                else:
                    cnt_1hot[-1] = 1 # pad one hot vector, manually
                seq_list.append(cnt_1hot)
            batch_C.append(seq_list)
        batch_C = torch.Tensor(batch_C).to(device)
        encoded_x, mu, logvar, z, h_state, decoder_output, recon_x = lstm_vae(batch_C, batch_lens, False)      

        batch_u = U[train_index : train_index + batch_size
                        if train_index + batch_size < len(U)
                        else len(U)]
        batch_u = torch.LongTensor(batch_u).to(device)
        nums = calc_weight_nums #50
        start = train_index
        end = min(train_index+batch_size, len(U))
        for j in range(nums):
            batch_z = mu + torch.exp(0.5 * logvar).to(device) * torch.randn(size = mu.size()).to(device) 
            pre_d = dc(batch_u, batch_z)
            pre_d = pre_d.detach().cpu().numpy().squeeze()
            weight[start:end] += ((1 - pre_d) / pre_d) / nums
        weight[start:end] = 1 / weight[start:end]
        train_index += batch_size
    weight /= weight.mean()
    return weight

