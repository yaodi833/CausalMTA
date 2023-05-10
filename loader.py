import numpy as np
import pickle


def load_config(cfg_file):
    cfg_file = "./configs/configs.txt"
    f = open(cfg_file, 'r')
    config_lines = f.readlines()
    cfgs = { }
    for line in config_lines:
        ps = [p.strip() for p in line.split('=')]
        if (len(ps) != 2):
            continue
        try:
            if (ps[1].find(',') != -1):
                str_line = ps[1].split(',')
                cfgs[ps[0]] = list(map(int, str_line))
            elif (ps[1].find('.') == -1):
                cfgs[ps[0]] = int(ps[1])
            else:
                cfgs[ps[0]] = float(ps[1])
        except ValueError:
            cfgs[ps[0]] = ps[1]
            if cfgs[ps[0]] == 'False':
                cfgs[ps[0]] = False
            elif cfgs[ps[0]] == 'True':
                cfgs[ps[0]] = True         

    return cfgs


def load_data(cfgs, isTrainSet = True):
    if isTrainSet:
        file_U = cfgs['file_U_train']
        file_C = cfgs['file_C_train']
        file_T = cfgs['file_T_train']
        file_Y = cfgs['file_Y_train']
        file_cost = cfgs['file_cost_train']
        file_CPO = cfgs['file_CPO_train']
        file_cat1_9 = cfgs['file_cat1_9_train']
    else:
        file_U = cfgs['file_U_test']
        file_C = cfgs['file_C_test']
        file_T = cfgs['file_T_test']
        file_Y = cfgs['file_Y_test']
        file_cost = cfgs['file_cost_test']
        file_CPO = cfgs['file_CPO_test']
        file_cat1_9 = cfgs['file_cat1_9_test']

    f = open(file_U, "rb")
    U = pickle.load(f)
    f.close()

    f = open(file_C, "rb")
    C = pickle.load(f)
    f.close()

    f = open(file_T, "rb")
    T = pickle.load(f)
    f.close()

    f = open(file_Y, "rb")
    Y = pickle.load(f)
    f.close()

    f = open(file_cost, "rb")
    cost = pickle.load(f)
    f.close()

    f = open(file_CPO, "rb")
    CPO = pickle.load(f)
    f.close()

    f = open(file_cat1_9, "rb")
    cat1_9 = pickle.load(f)
    f.close()

    return U, C, T, Y, cost, CPO, cat1_9





if (__name__=="__main__"):
    # just for debugging
    load_config("myMTA/configs/configs.txt")




