import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, epoch2
import random



def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10_S_90', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    # parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    # parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    # parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--num_eval', type=int, default=2, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=500, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.05, help='learning rate for updating network parameters')
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
    parser.add_argument('--skew', type=float, default=0.9, help='FairDD lambda parameter')



    for ss in [5,4]:

        args = parser.parse_args()
        args.method = 'DM'
        args.outer_loop, args.inner_loop = get_loops(args.ipc)
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.dsa_param = ParamDiffAug()
        args.dsa = False if args.dsa_strategy in ['none', 'None'] else True



        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        eval_it_pool = [args.Iteration]
        print('eval_it_pool: ', eval_it_pool)
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path, skew_ratio = 0.8, severity = ss)
        model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

        load_random_state(random_state)

        accs_all_exps = dict() # record performances of all experiments
        for key in model_eval_pool:
            accs_all_exps[key] = []

        data_save = []


        for exp in range(args.num_exp):

            ''' organize the real dataset '''
            images_all = []
            labels_all = []
            color_all = []

            images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
            labels_all = [int(dst_train[i][1]) for i in range(len(dst_train))]
            color_all = [int(dst_train[i][2]) for i in range(len(dst_train))]
            images_all = torch.cat(images_all, dim=0).to(args.device)
            labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
            color_all = torch.tensor(color_all, dtype=torch.long, device=args.device)

            args.num_classes = len(torch.unique(labels_all))
            args.num_groups = len(torch.unique(color_all))

            indices_class = [[] for c in range(args.num_classes)]
            for i, lab in enumerate(labels_all):
                indices_class[lab].append(i)


            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, 1))

                # print('DSA augmentation strategy: \n', args.dsa_strategy)
                # print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                accs = []
                max_Equalized_Odds_list = []
                mean_Equalized_Odds_list = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, args.num_classes, im_size).to(args.device) # get a random model
                    image_syn_eval, label_syn_eval = copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()) # avoid any unaware modification
                    # _, acc_train, acc_test, max_Equalized_Odds, mean_Equalized_Odds = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                    _, acc_train, acc_test, max_Equalized_Odds, mean_Equalized_Odds, max_Sufficiency, mean_Sufficiency = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                    accs.append(acc_test)
                    max_Equalized_Odds_list.append(max_Equalized_Odds)
                    mean_Equalized_Odds_list.append(mean_Equalized_Odds)
                    # torch.save({'net': net_eval.state_dict()}, os.path.join(args.save_path,'res_%s_%s_%s_%sori.pt' % (args.method, args.dataset,args.model,it_eval)))
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                # print('\n\naccs, max_Equalized_Odds, mean_Equalized_Odds',np.mean(accs), np.round(np.mean(max_Equalized_Odds_list),4), np.round(np.mean(mean_Equalized_Odds_list),4),'\n\n')
                print('\n Full skew =0.8, severity: ---> ',ss)
                print(f"\taccs: {np.mean(accs):.4f} ± {np.std(accs):.4f}", end =', ')
                print(f"\tmax_Equalized_Odds: {np.mean(max_Equalized_Odds_list):.4f} ± {np.std(max_Equalized_Odds_list):.4f}", end =', ')
                print(f"\tmean_Equalized_Odds: {np.mean(mean_Equalized_Odds_list):.4f} ± {np.std(mean_Equalized_Odds_list):.4f}")



                
                ###########





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
