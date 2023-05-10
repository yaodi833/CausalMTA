import torch
from torch.autograd import Variable
import torch.nn as nn
from re_weighting.base_model import LSTM_Encoder, LSTM_Decoder
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LSTM_OneHotVAE(nn.Module):
    def __init__(
        self,
        input_dim, 
        LSTM_hidden_dim,
        latent_variable_size,
        batch_size,
        device = "cuda:0"
    ):
        super(LSTM_OneHotVAE, self).__init__()
        self.latent_variable_size = latent_variable_size
        self.device = device
        
        self.encoder = LSTM_Encoder(input_dim, LSTM_hidden_dim)

        self.hidden_to_mean = nn.Linear(LSTM_hidden_dim, latent_variable_size)
        self.hidden_to_logvar = nn.Linear(LSTM_hidden_dim, latent_variable_size)

        self.latent_to_hidden = nn.Linear(latent_variable_size, LSTM_hidden_dim)

        self.decoder = LSTM_Decoder(input_dim, LSTM_hidden_dim, batch_size)

        self.hidden_to_output = nn.Linear(LSTM_hidden_dim, input_dim)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)   
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, x, lens, training = True):
        packed_x = pack_padded_sequence(
            input = x,
            lengths = lens,
            batch_first = True,
            enforce_sorted = False
        )
        self.training = training
        
        encoded_x = self.encoder(packed_x)
        mu = self.hidden_to_mean(encoded_x)
        logvar = self.hidden_to_logvar(encoded_x)
        z = mu

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.rand_like(std)
            z = eps.mul(std).add_(mu)

        h_state = self.latent_to_hidden(z)

        decoder_output = self.decoder(h_state, max(lens))

        recon_x = self.hidden_to_output(decoder_output)

        return encoded_x, mu, logvar, z, h_state, decoder_output, recon_x



import math 
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import random
torch.manual_seed(1)

def train_vae(C, cfgs):
    campaign_size = cfgs['global_campaign_size']
    epochs = cfgs['vae_epoch_num']
    batch_size = cfgs['vae_batch_size']
    device = cfgs['device']

    model = LSTM_OneHotVAE(
        input_dim = campaign_size + 1,
        LSTM_hidden_dim = cfgs['vae_LSTM_hidden_dim'],
        latent_variable_size = cfgs['vae_latent_variable_size'],
        batch_size = batch_size,
        device = device
    )
    model = model.to(device)

    loss_crosse = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr =  cfgs['vae_learning_rate']
    )
  
    for e in range(epochs):
        random.seed(e)
        random.shuffle(C)

        optimizer.zero_grad()
        print("Start of epoch {}".format(e))
        all_diff = 0
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
                        cnt_1hot[-1] = 1
                    seq_list.append(cnt_1hot)
                batch_C.append(seq_list)
            batch_C = torch.Tensor(batch_C).to(device)
            # print(batch_C.shape)

            encoded_x, mu, logvar, z, h_state, decoder_output, recon_x = model(batch_C, batch_lens)

            cre_loss = 0
            for i, seq_len in enumerate(batch_lens):
                target = torch.LongTensor(input_batch[i][:seq_len]).to(device)
                cre_loss += loss_crosse(
                    recon_x[i, :seq_len, :],
                    target
                )
            kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 0)).mean().squeeze()
            diff = cre_loss + kld
            print("Batch {0}/{1}\t CrossEntropy_loss {2:.4f}\t KLD_loss {3:.4f}".format(
                train_index, len(C), cre_loss, kld))
            diff.backward()
            optimizer.step()
            all_diff += diff
            train_index += batch_size
        print("All loss: {}".format(all_diff))

    return model        



def judge_1hot_tag(out_list, tag):
    return (max(out_list) == out_list[tag])

def eval_vae(lstm_vae, C, cfgs):
    device = cfgs['device']
    campaign_size = cfgs['global_campaign_size']
    batch_size = cfgs['vae_batch_size']
    judge_index = 0
    right_slot, wrong_slot = 0, 0
    while judge_index < len(C):
        if right_slot + wrong_slot > 100000:
            break
        input_batch = C[judge_index : judge_index + batch_size
                        if judge_index + batch_size < len(C)
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
                    cnt_1hot[-1] = 1
                seq_list.append(cnt_1hot)
            batch_C.append(seq_list)
        batch_C = torch.Tensor(batch_C).to(device)        

        encoded_x, mu, logvar, z, h_state, decoder_output, recon_x = lstm_vae(batch_C, batch_lens, False)
        

        for i in range(len(input_batch)):
            for j in range(batch_lens[i]):
                if judge_1hot_tag(recon_x[i,j], input_batch[i][j]):
                    right_slot += 1
                else: 
                    wrong_slot += 1

        judge_index += batch_size

    print("Right slot {}".format(right_slot))
    print("Wrong slot {}".format(wrong_slot))
    print("Acc is {}".format(right_slot / (right_slot + wrong_slot)))    

