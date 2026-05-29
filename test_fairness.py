import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
import random

import re
from pathlib import Path



def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--shuffle', type=bool, default=False, help='distance metric')
    parser.add_argument('--FairDD', action='store_true', help='Enable FairDD')
    parser.add_argument('--group_balance', type=bool, default=False, help='distance metric')
    
    parser.add_argument('--ALL_data', type=str, default='', help='path to save results')

    # open('results-final.txt', 'w').close()

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False 
    # # args.dsa = True
    # if  args.dsa:
    #     pref = ''
    # else:
    #     pref = 'normal-'
    pref =''

    NAMES = ['DC','DM','IDC', 'CAFE']
    # NAMES = [ 'CAFE']
    if args.ALL_data =='':
        ALL_DATA = [
                    "CIFAR10_S_90",
                    "Colored_FashionMNIST_foreground",
                    "Colored_FashionMNIST_background",
                    "Colored_MNIST_foreground",
                    "Colored_MNIST_background",
                    "UTKface",
                    "BFFHQ",
                                ]
    else: 
        ALL_DATA = [args.ALL_data]    

    
    for dataset in ALL_DATA:
        print(dataset)
        for name in NAMES:
            if name == 'DM':
                args.dsa = True
            elif name == 'DC':
                args.dsa = False
            elif name =='IDC':
                args.dsa = False
            elif name =='CAFE':
                args.dsa = True

        

            for fair_crt in ['NoFair','FairDD','NoOrtho']:
            # for fair_crt in ['FairDD']:
                    # for fair_crt in ['NoFair']:
                args.testMetric = name
                for ipc in [10,50,100]:
                    args.ipc = ipc
                    try:
                        if entry_exists(pref+'results-final-'+dataset+'.txt', name, fair_crt, dataset, ipc):
                            print(f"Skipping: Model = {name}, Method = {fair_crt},  dataset = {dataset}, ipc = {ipc}")
                            continue
                    except:
                        pass
                 

                    save_path = '/home/mmoslem3/scratch/FairDD/results-pt/' + name  + '/'+name +'-'+ fair_crt + '/'
                    if fair_crt == 'FairDD':
                        save_path = save_path + 'FairDD_'
                    elif fair_crt == 'NoOrtho':
                        save_path = save_path + 'Fair_NoOrtho_'
                        
                    save_path = save_path + name + '_' + dataset + '_ipc'  + str(args.ipc) + '/'
                    save_path = save_path + 'res_'+name+'_' + dataset + '_ConvNet_'  + str(args.ipc) + 'ipc.pt'



                    try:
                        checkpoint = torch.load(save_path, map_location=args.device, weights_only=False)
                        print('\n ++++++ Load synthetic data from %s +++++'%save_path.replace('/home/mmoslem3/scratch/FairDD/results-pt/',''))
                    except:
                        print('\n  ------ No checkpoint found for %s ----- '%save_path.replace('/home/mmoslem3/scratch/FairDD/results-pt/',''))
                        continue
                    # continue
                    
                    try:
                        image_syn, label_syn = checkpoint['data'][0]
                    except:
                        image_syn, label_syn = checkpoint['data']

                    image_syn = image_syn.to(args.device) 
                    label_syn = label_syn.to(args.device)
                    
                    args.dataset = dataset
                    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
                    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

                    load_random_state(random_state)

                    images_all, labels_all, color_all = [], [], []
                    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
                    labels_all = [dst_train[i][1] for i in range(len(dst_train))]

                    images_all = torch.cat(images_all, dim=0).to(args.device)
                    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

                    args.num_classes = len(torch.unique(labels_all))

                    model_eval = model_eval_pool[0]
                    print('-----------------\nEvaluation\nmodel_train = %s, model_eval = %s'%(args.model, model_eval))
                    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval,args.ipc) 
                    accs = []
                    max_Equalized_Odds_list, mean_Equalized_Odds_list = [], []
                    max_Sufficiency_list, mean_Sufficiency_list = [], []

                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        if name == 'IDC':
                            with torch.no_grad():
                                image_syn_eval, label_syn_eval = decode_zoom(image_syn.detach(), label_syn.detach(), 2, size=im_size)
                        else:
                            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) 

                        image_syn_eval = DiffAugment(image_syn_eval, args.dsa_strategy, seed=seed, param=args.dsa_param)  
                        _, acc_train, acc_test, max_Equalized_Odds, mean_Equalized_Odds, max_Sufficiency, mean_Sufficiency = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, verbose=False)
                        accs.append(acc_test)
                        max_Equalized_Odds_list.append(max_Equalized_Odds)
                        mean_Equalized_Odds_list.append(mean_Equalized_Odds)
                        max_Sufficiency_list.append(max_Sufficiency)
                        mean_Sufficiency_list.append(mean_Sufficiency)
                        if (it_eval+1)%1==0:
                            # print('Evaluation %d '%(it_eval))
                            print('Accuracy: %0.2f, max_Equalized_Odds: %0.2f, mean_Equalized_Odds: %0.2f '%((accs[-1]),(max_Equalized_Odds_list[-1]),(mean_Equalized_Odds_list[-1])))

                    print('\n\n -------Model = %s, Method = %s,  dataset = %s, ipc = %d --------- '%(args.testMetric, fair_crt,  args.dataset, args.ipc))
                    print('Accuracy: %0.6f ± %0.6f \nmax_Equalized_Odds: %0.6f ± %0.6f \nmean_Equalized_Odds: %0.6f ± %0.6f\nmax_Sufficiency: %0.6f ± %0.6f\nmean_Sufficiency: %0.6f ± %0.6f'%(np.mean(accs),np.std(accs),np.mean(max_Equalized_Odds_list),np.std(max_Equalized_Odds_list), np.mean(mean_Equalized_Odds_list),np.std(mean_Equalized_Odds_list), np.mean(max_Sufficiency_list),np.std(max_Sufficiency_list), np.mean(mean_Sufficiency_list),np.std(mean_Sufficiency_list)))
                    print('--------------------------------\n\n')






if __name__ == '__main__':
    def save_random_state():
        return {
            'torch': torch.get_rng_state(),
            'np': np.random.get_state(),
            'random': random.getstate(),
            'cuda': torch.cuda.get_rng_state_all()
        }
    def load_random_state(state):
        torch.set_rng_state(state['torch'])
        np.random.set_state(state['np'])
        random.setstate(state['random'])
        torch.cuda.set_rng_state_all(state['cuda'])

    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 保存当前的随机状态
    random_state = save_random_state()

    main()
