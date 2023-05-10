import itertools
import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(1)
import random
import math
# from predictors.Attn_LSTM_ub import *
from predictors.Attn_LSTM_uc import *

"""
input: list of touch
output: list of tuple
"""
def from_touch_to_tuple(touch_list):
    tuple_list = []
    i = 0
    for touch in touch_list:
        tuple_list.append((touch, i))
        i += 1
    return tuple_list


def calc_marginal_contribution(user_tag, seq_C, seq_cat1_9, long_permu, short_permu, predictor, device = "cuda:0"):
    long_permu.sort()
    long_sub_seq_C = []
    long_sub_seq_cat1_9 = []
    for i in long_permu:
        long_sub_seq_C.append(seq_C[i])
        long_sub_seq_cat1_9.append(seq_cat1_9[i])

    short_permu.sort()
    short_sub_seq_C = []
    short_sub_seq_cat1_9 = []
    for i in short_permu:
        short_sub_seq_C.append(seq_C[i])
        short_sub_seq_cat1_9.append(seq_cat1_9[i])

    user_tsr = torch.LongTensor([user_tag]).to(device)   

    long_lens = [len(long_sub_seq_C)]
    long_sub_seq_C_tsr = torch.LongTensor(long_sub_seq_C).to(device)
    long_sub_seq_C_tsr = long_sub_seq_C_tsr.unsqueeze(0)
    long_sub_seq_cat1_9_np = np.array(long_sub_seq_cat1_9)
    long_sub_seq_cat1_9_tsr = torch.LongTensor(long_sub_seq_cat1_9_np).to(device)
    long_sub_seq_cat1_9_tsr = long_sub_seq_cat1_9_tsr.unsqueeze(0)

    long_contribution = predictor(user_tsr, long_sub_seq_C_tsr, long_sub_seq_cat1_9_tsr, long_lens)
    long_contribution = long_contribution.squeeze().cpu()    

    if len(short_sub_seq_C) != 0: 
        short_lens = [len(short_sub_seq_C)]
        short_sub_seq_C_tsr = torch.LongTensor(short_sub_seq_C).to(device)
        short_sub_seq_C_tsr= short_sub_seq_C_tsr.unsqueeze(0)
        short_sub_seq_cat1_9_np = np.array(short_sub_seq_cat1_9)
        short_sub_seq_cat1_9_tsr = torch.LongTensor(short_sub_seq_cat1_9_np).to(device)
        short_sub_seq_cat1_9_tsr = short_sub_seq_cat1_9_tsr.unsqueeze(0)

        short_contribution = predictor(user_tsr, short_sub_seq_C_tsr, short_sub_seq_cat1_9_tsr ,short_lens)
        short_contribution = short_contribution.squeeze().cpu()
    else:
        short_contribution = 0    

    return long_contribution - short_contribution


def calculate_Shapley_values(user_tag, seq_C, seq_cat1_9, predictor,  device = "cuda:0"):
    array = range(len(seq_C))
    permutation_list = list(itertools.permutations(array))
    sv_array = np.zeros(len(seq_C))

    n = len(seq_C)
    factorial_n = math.factorial(n)

    for i in range(len(seq_C)):
        cur_contribution = 0
        for permutation in permutation_list:
            permutation_ = list(permutation)
            index = permutation_.index(i)
            # weight = 1. * math.factorial(n - index - 1) * math.factorial(index) / math.factorial(n)
            weight = 1.  / math.factorial(n)
            cur_contribution += weight * calc_marginal_contribution(user_tag, seq_C, seq_cat1_9, permutation_[:index+1], permutation_[:index], predictor, device)
        sv_array[i] = cur_contribution

    for i in range(len(sv_array)):
        if sv_array[i] < 0:
            sv_array[i] = 0
    
    if sv_array.sum() == 0:
        attr_sv_array = sv_array / (sv_array.sum()+1)
    else:
        attr_sv_array = sv_array / sv_array.sum()

    # print(attr_sv_array)
    return attr_sv_array


def calc_counterfactual_touch_effect(user_tag, subseq_C, subseq_cat1_9,  predictor, device = "cuda:0"):
    user_tsr = torch.LongTensor([user_tag]).to(device)   
    lens = [len(subseq_C)]
    subseq_C_tsr = torch.LongTensor(subseq_C).to(device)
    subseq_C_tsr = subseq_C_tsr.unsqueeze(0)
    subseq_cat1_9_np = np.array(subseq_cat1_9)
    subseq_cat1_9_tsr = torch.LongTensor(subseq_cat1_9_np).to(device)
    subseq_cat1_9_tsr = subseq_cat1_9_tsr.unsqueeze(0)

    counterfactual_effect = predictor(user_tsr, subseq_C_tsr, subseq_cat1_9_tsr, lens)
    counterfactual_effect = counterfactual_effect.squeeze().cpu()

    return counterfactual_effect    


def calc_ivh_attribution(user_tag, seq_C, seq_cat1_9, predictor, device = "cuda:0"):
    observ_touch_effect = calc_counterfactual_touch_effect(user_tag, seq_C, seq_cat1_9, predictor, device)    
    ite_array = np.zeros(len(seq_C))
    for touch_index in range(len(seq_C)):
        subseq_C = seq_C[:touch_index] + seq_C[touch_index+1:]
        subseq_cat1_9 = seq_cat1_9[:touch_index] + seq_cat1_9[touch_index+1:]
        ite_array[touch_index] = observ_touch_effect - calc_counterfactual_touch_effect(user_tag, subseq_C, subseq_cat1_9, predictor, device)
    
    for i in range(len(ite_array)):
        if ite_array[i] < 0:
            ite_array[i] = 0
    
    if ite_array.sum() == 0:
        attr_ite_array = ite_array / (ite_array.sum() + 1)
    else:
        attr_ite_array = ite_array / ite_array.sum()
    # print(attr_ite_array)
    return attr_ite_array
