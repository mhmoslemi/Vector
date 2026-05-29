import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

EMN_EVAL_POOLS = {
    'CIFAR10':      ['ConvNet','ResNet18'],
    'CIFAR100':      ['ConvNet', 'ResNet18'],
    'SVHN':      ['ConvNet', 'ResNet18'],
    'MNIST':      ['ConvNet', 'ResNet18'],
}


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=5000, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=25, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/UE-DD/data/')
    parser.add_argument('--save_path', type=str, default='/home/mmoslem3/scratch/UE-DD/partial')

    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    args = parser.parse_args()
    args.method = 'DM'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = [5, 10,15,  20,30,40,50,60, 70,80,90, 100, 125, 150,160 , 175,  200, 225]

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    # model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    model_eval_pool = EMN_EVAL_POOLS.get(args.dataset, ['ResNet18'])



    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []


    LABELS_PATH    = "/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/posion-FLIP/experiments/clean_attack/labels.npy"
    labels_np = np.load(LABELS_PATH)
    labels_flip = labels_np.argmax(axis=1).astype(np.int64)   # shape (50000,)
    labels_flip = torch.tensor(labels_flip, dtype=torch.long, device=args.device)




    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
    



    def get_images(c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]



    ''' initialize the synthetic data '''
    label_syn = labels_all.copy()
    images_poisoned = images_all.clone().detach().to(args.device).requires_grad_(True)


    ''' training '''
    optimizer_img = torch.optim.SGD([images_poisoned, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    print('%s training begins'%get_time())

    for it in range(args.Iteration+1):

        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                accs = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                    image_syn_eval, label_syn_eval = copy.deepcopy(images_poisoned.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                    accs.append(acc_test)
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))


        ''' Train synthetic data '''
        net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False

        embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

        loss_avg = 0

        ''' update synthetic data '''
        if 'BN' not in args.model: # for ConvNet
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real)
                img_syn = images_poisoned[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                output_real = embed(img_real).detach()
                output_syn = embed(img_syn)

                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

        else: # for ConvNetBN
            images_real_all = []
            images_syn_all = []
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real)
                img_syn = images_poisoned[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                images_real_all.append(img_real)
                images_syn_all.append(img_syn)

            images_real_all = torch.cat(images_real_all, dim=0)
            images_syn_all = torch.cat(images_syn_all, dim=0)

            output_real = embed(images_real_all).detach()
            output_syn = embed(images_syn_all)

            loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)



        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        loss_avg += loss.item()


        loss_avg /= (num_classes)

        if it%1 == 0:
            print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))



        # if it % 4 ==0 :
        #     torch.save({
        #         'images_poisoned': images_poisoned.detach().cpu(),
        #         'labels': labels_all.cpu(),
        #     }, os.path.join(args.save_path, 'cifar10-flip.pt' ))
        #     # },os.path.join(args.save_path, 'res_%s_iter%d_bug%s_lamexcess%s.pt' % (args.dataset, it,str(args.budget), str(args.lambda_excess) )))

        # if it % 50 ==0 :
        #     torch.save({
        #         'images_poisoned': images_poisoned.detach().cpu(),
        #         'labels': labels_all.cpu(),
        #     }, os.path.join(args.save_path, 'cifar10-flip'+str(it)+'.pt' ))
        #     # },os.path.join(args.save_path, 'res_%s_iter%d_bug%s_lamexcess%s.pt' % (args.dataset, it,str(args.budget), str(args.lambda_excess) )))






        if it % 5 ==0:
            save_name = os.path.join(args.save_path, 'vis_%s_iter%d_bug%s_lamexcess%s.png' % (args.dataset, it,str(args.budget), str(args.lambda_excess) ))
            # save_name = os.path.join(args.save_path, 'T6_iter%d-AT.png' % ( it))
            image_syn_vis = (images_poisoned[:50].detach().cpu().clone())
            for ch in range(channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
            image_syn_vis = torch.clamp(image_syn_vis, 0.0, 1.0)
            save_image(image_syn_vis, save_name, nrow=num_classes)




    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()

