from torch.utils.tensorboard import SummaryWriter
import argparse
from loader import load_config, load_data
import torch
torch.manual_seed(1)
import copy

import numpy as np 

from re_weighting.lstm_vae import LSTM_OneHotVAE, eval_vae, train_vae
from re_weighting.domain_classifier import *
from predictors.causal_predictor import *
from attr import calculate_Shapley_values, calc_ivh_attribution

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type = str)
    return parser.parse_args()

if (__name__ == '__main__'):
    args = init_arg()
    configs = load_config(args.config_file)
    device = configs['device']

    # load_training_data
    U_train, C_train, T_train, Y_train, cost_train, CPO_train, cat1_9_train = load_data(configs)

    # load_testing_data
    U_test, C_test, T_test, Y_test, cost_test, CPO_test, cat1_9_test = load_data(configs, False)

    # train the VAE
    print("The VAE phase:")
    if (configs['pre_vae_path'] == 'None'):
        print("Start training the VAE...")
        C_train_input = copy.deepcopy(C_train)
        lstm_vae = train_vae(C_train_input, configs)
        torch.save(lstm_vae, configs['stored_vae_path'])
        print("End training the VAE.")
    else:
        print("Loading the VAE model...")
        lstm_vae = torch.load(configs['pre_vae_path']).to(device)

    # eval the VAE
    print("Start evaluating VAE.")
    C_train_input = copy.deepcopy(C_train)
    eval_vae(lstm_vae, C_train_input, configs)


    # get the domain classifier
    print("The domain classifier phase:")
    if (configs['pre_dc_path'] == 'None'):
        print("Start training the domain classifier...")
        U_train_input = copy.deepcopy(U_train)
        C_train_input = copy.deepcopy(C_train)
        dc = getDomainClassifier(lstm_vae, U_train_input, C_train_input, configs)
        torch.save(dc, configs['stored_dc_path'])
        print("End training the domain classifier.")    
    else:
        print("Loading the dc model...")
        dc = torch.load(configs['pre_dc_path']).to(device)


    # # calculate the sample weights
    print("The weights calculation phase:")
    if (configs['pre_weights_path'] == 'None'):
        print("Statring calculating the weights...")
        U_train_input = copy.deepcopy(U_train)
        C_train_input = copy.deepcopy(C_train)       
        weight = calcSampleWeights(U_train_input, C_train_input, lstm_vae, dc, configs)
        np.save(configs['stored_weights_path'], weight)
    else:
        print("Loading the weights...")
        weight = np.load(configs['pre_weights_path'])

    # skim at the weights
    print("Skimming at the weights.")
    print("The average of the weight is {}".format(weight.mean() ))
    print("The largest weight is {}".format(weight.max() ))
    print("The minimize of the weight is {}".format(weight.min()))
    print("The first 128 weights ars:")
    print(weight[:128])

    # weight_path = ""
    if (configs['pre_weights_path'] == 'None'):
        weight_path = configs['stored_weights_path']
    else:
        weight_path = configs['pre_weights_path']
        
        
    if configs['is_reweight_samples']:
        # causal prediction model
        print("Train & test or load the causal_prediction model:")
        # train/load the model with weight
        if (configs['predictor_with_w_pre_path'] == 'None'):
            print("Train & test the model with weight")
            C_train_input = copy.deepcopy(C_train)
            U_train_input = copy.deepcopy(U_train)
            cat1_9_train_input = copy.deepcopy(cat1_9_train)
            Y_train_input = copy.deepcopy(Y_train)
            model_with_w = Causal_Predictor_train_test(configs, C_train_input, U_train_input, cat1_9_train_input, Y_train_input,
                C_test, U_test, cat1_9_test, Y_test, True, weight_path)
            torch.save(model_with_w, configs['predictor_with_w_stored_path'])
        else:
            print("Load the model with weight")
            model_with_w = torch.load(configs['predictor_with_w_pre_path']).to(device)
        # train/load the model without weight
    else:
        if (configs['predictor_no_w_pre_path'] == 'None'):
            print("Train & test the model without weight")
            C_train_input = copy.deepcopy(C_train)
            U_train_input = copy.deepcopy(U_train)
            cat1_9_train_input = copy.deepcopy(cat1_9_train)
            Y_train_input = copy.deepcopy(Y_train)
            model_no_w = Causal_Predictor_train_test(configs, C_train_input, U_train_input, cat1_9_train_input, Y_train_input,
                C_test, U_test, cat1_9_test, Y_test)
            torch.save(model_no_w, configs['predictor_no_w_stored_path'])
        else:
            print("Load the model without weight")
            model_no_w = torch.load(configs['predictor_no_w_pre_path']).to(device)
