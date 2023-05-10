from torch.utils.tensorboard import SummaryWriter
import torch 
from torch import nn, optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import time

class CausalPredictor(nn.Module):
    def __init__(
        self,
        C_num_embeddings,
        C_embedding_dim,
        U_num_embeddings,
        U_embedding_dim,
        cat_num_embeddings_list,
        cat_embedding_dim_list,
        LSTM_hidden_dim,
        LSTM_hidden_layer_depth,
        fc_hidden_dim,
        fc_hidden_layer,
        device = "cuda:0",
        dropout_rate = 0.2        
    ):
        super(CausalPredictor, self).__init__()

        self.LSTM_hidden_dim = LSTM_hidden_dim
        self.LSTM_hidden_layer_depth = LSTM_hidden_layer_depth

        self.cam_embedding = nn.Embedding(
            num_embeddings = C_num_embeddings,
            embedding_dim = C_embedding_dim
        )

        self.cat_cnt = len(cat_embedding_dim_list)
        self.cat_embedding_list = nn.ModuleList(
            [nn.Embedding(num_embeddings = cat_num_embeddings_list[i], embedding_dim = cat_embedding_dim_list[i]) for i in range(self.cat_cnt)]
        )

        self.bi_lstm = nn.LSTM(
            input_size = C_embedding_dim + sum(cat_embedding_dim_list),
            hidden_size = LSTM_hidden_dim,
            num_layers = LSTM_hidden_layer_depth,
            batch_first = True,
            dropout = dropout_rate,
            bidirectional = True
        )

        self.user_embedding = nn.Embedding(
            num_embeddings = U_num_embeddings,
            embedding_dim = U_embedding_dim
        )

        self.final_input_net = nn.Linear(LSTM_hidden_dim + U_embedding_dim, fc_hidden_dim)
        self.fc_hidden_layer = fc_hidden_layer
        self.final_hidden_net = nn.ModuleList(
            [nn.Linear(fc_hidden_dim, fc_hidden_dim) for i in range(self.fc_hidden_layer - 1)]
        )
        self.final_output_net = nn.Linear(fc_hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def attention(self, lstm_output, final_state):
        merged_state = final_state
        merged_state = merged_state.unsqueeze(-1)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)


    def forward(self, U, C, cat, lens):
        embeded_C = self.cam_embedding(C)

        embeded_cat_list = []
        for i in range(self.cat_cnt):
            embeded_cat_list.append(self.cat_embedding_list[i](cat[:,:,i]))

        concated_tsr = embeded_C
        for i in range(self.cat_cnt):
            concated_tsr = torch.cat((concated_tsr, embeded_cat_list[i]), 2)
        
        packed_input = pack_padded_sequence(
            input = concated_tsr,
            lengths = lens,
            batch_first = True,
            enforce_sorted = False
        )

        lstm_output, (lstm_hidden, _) = self.bi_lstm(packed_input)
  
        lstm_output, output_lengths = pad_packed_sequence(lstm_output) 
        lstm_output = lstm_output.permute(1, 0, 2)

        lstm_output = lstm_output[:, :, :self.LSTM_hidden_dim] + lstm_output[:, :, self.LSTM_hidden_dim:]

        querys = []
        for i, len in enumerate(output_lengths):
            querys.append(lstm_output[i, len-1, :])
        querys = torch.stack(querys)

        attn_output = self.attention(lstm_output, querys)

        embeded_U = self.user_embedding(U)
        
        final_input_tsr = torch.cat((attn_output, embeded_U), 1)
        final_middle = F.elu(self.final_input_net(final_input_tsr))
        for i in range(self.fc_hidden_layer - 1):
            final_middle = F.elu(self.final_hidden_net[i](final_middle))
        final_output = self.final_output_net(final_middle)

        return self.sigmoid(final_output)        


import os
import math 
import pickle
import numpy as np
import random
import copy

import torch 
import torch.nn as nn 
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
torch.manual_seed(1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from utils import calcRMSE


def Causal_Predictor_inside_test(C, U, cat1_9, Y,  model, cfgs, log_writer, e):
    test_index = 0
    batch_size = cfgs['predictor_batch_size']
    pre_res = np.zeros(len(C))
    pre_res_int = np.zeros(len(C))
    cat_num_list = cfgs['global_cat_num_list']
    cat_num_list_plus1 = list(map(lambda x:x+1, cat_num_list))
    campaign_size = cfgs['global_campaign_size']
    user_num = cfgs['global_user_num']
    device = cfgs['device']
    epochs = cfgs['predictor_epoch_num']
    batch_size = cfgs['predictor_batch_size']


    while test_index < len(C):
        start = test_index
        end = min(start + batch_size, len(C))
    
        batch_C = C[start : end]
        batch_lens = []
        batch_C_tsr = []
        for i in range(end - start):
            batch_lens.append(len(batch_C[i]))
            campaign_tsr = torch.LongTensor(batch_C[i]).to(device)
            batch_C_tsr.append(campaign_tsr)
        batch_C_padded = pad_sequence(
            batch_C_tsr,
            padding_value = campaign_size,
            batch_first = True
        )

        batch_U_tsr = torch.LongTensor(U[start : end]).to(device)

        cat_padding_list = cat_num_list
        batch_cat1_9_old = cat1_9[start : end]
        batch_cat1_9 = copy.deepcopy(batch_cat1_9_old)
        for i in range(end - start):
            while max(batch_lens) - len(batch_cat1_9[i]) > 0:
                batch_cat1_9[i].append(cat_padding_list)                   
        batch_cat1_9 = np.asarray(batch_cat1_9)       
        batch_cat1_9_tsr = torch.LongTensor(batch_cat1_9).to(device)

        batch_Y_tsr = torch.Tensor(Y[start : end]).to(device) 

        pre_d = model(batch_U_tsr, batch_C_padded, batch_cat1_9_tsr, batch_lens)
        pre_d = pre_d.squeeze()

        for i in range(end - start):
            pre_res[start + i] = pre_d[i]
            if pre_d[i] > 0.5:
                pre_res_int[start + i] = 1
            else:
                pre_res_int[start + i] = 0

        test_index += batch_size

    truth = np.array(Y)
    C2 = confusion_matrix(truth, pre_res_int)
    print(C2)
    print(C2.ravel())    

    tn, fp, fn, tp = C2.ravel()
    print("Accuracy: "+str(round((tp+tn)/(tp+fp+fn+tn), 3)))
    print("Recall: "+str(round((tp)/(tp+fn), 3)))
    print("Precision: "+str(round((tp)/(tp+fp), 3)))
    print("AUC: "+str(roc_auc_score(truth, pre_res)))
    print("RMSE: "+ str(calcRMSE(truth, pre_res)))    
    log_writer.add_scalar(f"Pred/Accuracy", round((tp+tn)/(tp+fp+fn+tn), 3), e)
    log_writer.add_scalar(f"Pred/Recall", round((tp)/(tp+fn), 3), e)
    log_writer.add_scalar(f"Pred/Precision", round((tp)/(tp+fp), 3), e)
    log_writer.add_scalar(f"Pred/AUC", roc_auc_score(truth, pre_res), e)
    log_writer.add_scalar(f"Pred/RMSE", calcRMSE(truth, pre_res), e)
    return roc_auc_score(truth, pre_res) # return AUC        


def Causal_Predictor_train_test(
    cfgs,
    C, U, cat1_9, Y,
    C_test, U_test, cat1_9_test, Y_test,
    with_weight = False,
    weight_path = None
):
    localtime = time.asctime(time.localtime(time.time()))
    log_writer = SummaryWriter("torch_runs/LSTMPredictor_weight_{}_hidden_{}_layer_{}_epoches_{}_time_{}/".format(
        cfgs['is_reweight_samples'], cfgs['predictor_lstm_hidden_dim'], cfgs['predictor_lstm_hidden_layer_depth'], cfgs['predictor_epoch_num'], localtime))

    auc_list = []
    W = []
    if with_weight:
        W_np = np.load(weight_path)
        W = W_np.tolist()
    cat_num_list = cfgs['global_cat_num_list']
    cat_num_list_plus1 = list(map(lambda x:x+1, cat_num_list))
    campaign_size = cfgs['global_campaign_size']
    user_num = cfgs['global_user_num']
    device = cfgs['device']
    epochs = cfgs['predictor_epoch_num']
    batch_size = cfgs['predictor_batch_size']

    model = CausalPredictor(
        C_num_embeddings = campaign_size + 1,
        C_embedding_dim = cfgs['predictor_gloabl_campaign_embedding_dim'],
        U_num_embeddings = user_num,
        U_embedding_dim = cfgs['predictor_global_user_embedding_dim'],
        cat_num_embeddings_list = cat_num_list_plus1,
        cat_embedding_dim_list = [cfgs['predictor_global_cat_embedding_dim']] * len(cat_num_list_plus1),
        LSTM_hidden_dim = cfgs['predictor_lstm_hidden_dim'],
        LSTM_hidden_layer_depth = cfgs['predictor_lstm_hidden_layer_depth'],
        fc_hidden_dim = cfgs['predictor_fc_hidden_dim'],
        fc_hidden_layer = cfgs['predictor_fc_hidden_layer']
    ).to(device)

    bceloss = nn.BCELoss(reduction = "none")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = cfgs['predictor_learning_rate']
    )

    for e in range(epochs):
        random.seed(e)
        ssc = random.shuffle(C)
        random.seed(e)
        random.shuffle(U)
        random.seed(e)
        random.shuffle(Y)
        random.seed(e)
        random.shuffle(cat1_9)
        if with_weight:
            random.seed(e)
            random.shuffle(W)

        optimizer.zero_grad()
        print("Start of epoch {}".format(e))
        all_losses = []
        train_index = 0

        while train_index < len(C):
            start = train_index
            end = min(start + batch_size, len(C))
            
            batch_C = C[start : end]
            batch_lens = []
            batch_C_tsr = []
            for i in range(end - start):
                batch_lens.append(len(batch_C[i]))
                campaign_tsr = torch.LongTensor(batch_C[i]).to(device)
                batch_C_tsr.append(campaign_tsr)
            batch_C_padded = pad_sequence(
                batch_C_tsr,
                padding_value = campaign_size,
                batch_first = True
            )
            
            batch_U_tsr = torch.LongTensor(U[start : end]).to(device)

            cat_padding_list = cat_num_list
            batch_cat1_9_old = cat1_9[start : end]
            batch_cat1_9 = copy.deepcopy(batch_cat1_9_old)
            for i in range(end - start):
                while max(batch_lens) - len(batch_cat1_9[i]) > 0:
                    batch_cat1_9[i].append(cat_padding_list)                   
            batch_cat1_9 = np.asarray(batch_cat1_9)       
            batch_cat1_9_tsr = torch.LongTensor(batch_cat1_9).to(device)

            batch_Y_tsr = torch.Tensor(Y[start : end]).to(device) 

            if with_weight:
                batch_W_tsr = torch.Tensor(W[start : end]).to(device)
            
            pre_d = model(batch_U_tsr, batch_C_padded, batch_cat1_9_tsr, batch_lens)
            pre_d = pre_d.squeeze()
            loss = bceloss(pre_d, batch_Y_tsr)

            if with_weight:
                loss = loss * batch_W_tsr
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())
            train_index += batch_size

        print("Epoch {}".format(e))
        print("Avg loss is {}".format(sum(all_losses) / len(all_losses) ))

        cur_auc = Causal_Predictor_inside_test(
            C_test, U_test, cat1_9_test, Y_test, model, cfgs, log_writer,e)
        auc_list.append(cur_auc)

    print("The auc list is:")
    print(auc_list)
    return model             


















