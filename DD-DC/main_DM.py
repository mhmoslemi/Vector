import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_and_poison(images_all, labels_all, dataset, noise_type,
                    noise_path, num_classes, device, mean, std):
    noise_filename  = f"{dataset}_{noise_type}.pt"
    full_noise_path = os.path.join(noise_path, noise_filename)

    if not os.path.exists(full_noise_path):
        raise FileNotFoundError(f"Noise file not found: {full_noise_path}")

    print(f"    Loading {noise_type} noise from {full_noise_path} ...")
    noise_data = torch.load(full_noise_path, map_location=device)

    if isinstance(noise_data, dict):
        for k in ("noise", "perturbation", "delta"):
            if k in noise_data:
                noise_tensor = noise_data[k]
                break
    else:
        noise_tensor = noise_data

    noise_tensor = torch.as_tensor(noise_tensor, dtype=torch.float32, device=device)

    if noise_tensor.max() > 1.5:
        noise_tensor = noise_tensor / 255.0

    if (noise_tensor.ndim == 4
            and noise_tensor.shape[1:] != images_all.shape[1:]
            and noise_tensor.shape[-1] in [1, 3]):
        noise_tensor = noise_tensor.permute(0, 3, 1, 2).contiguous()

    if noise_tensor.shape[0] == num_classes:             # CW: [C, C, H, W]
        noise_to_add = noise_tensor[labels_all]
    elif noise_tensor.shape[0] == images_all.shape[0]:   # SW: [N, C, H, W]
        noise_to_add = noise_tensor
    else:
        raise ValueError(
            f"Noise shape mismatch — noise: {noise_tensor.shape}, "
            f"images: {images_all.shape}"
        )

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, -1, 1, 1)
    images_raw   = images_all * std_t + mean_t
    poisoned_raw = torch.clamp(images_raw + noise_to_add, 0.0, 1.0)
    poisoned     = (poisoned_raw - mean_t) / std_t
    print(f"    Dataset poisoned. Noise l-inf: {noise_to_add.abs().max().item():.4f}")
    return poisoned

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='UE', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_eval', type=int, default=2, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    parser.add_argument('--noise_path', type=str, default='/home/mmoslem3/scratch/Unlearnable-Examples-DD/noise-EMN', help='Path to the directory containing .pt noise files')
    parser.add_argument('--noise_type', type=str, default='SW', choices=['CW', 'SW', 'None'], help='Type of noise: CW or SW')


    args = parser.parse_args()
    args.method = 'DM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.add_noise = False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    eval_it_pool = [50,100,200,300,400,500,700] + [args.Iteration]

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []


    # print('Hyper-parameters: \n', args.__dict__)
    # print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    args.pt_file = f'/home/mmoslem3/scratch/UE-DD/result-fianle/res2_MO_AT_CIFAR10_ConvNet_{str(8)}.pt'
    data = torch.load(args.pt_file, map_location=args.device,  weights_only=False)
    images_all = data['images_poisoned'].to(args.device)
    labels_all = data['labels'].to(args.device)
            



    # images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    # labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    # images_all = torch.cat(images_all, dim=0).to(args.device)
    # labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    images_all = images_all.to(args.device)
    labels_all = labels_all.to(args.device)

    # Normal flow: load clean images, then add noise
    # images_all = load_and_poison(
    #     images_all, labels_all, args.dataset, 'SW',
    #     '/home/mmoslem3/scratch/Unlearnable-Examples-DD/noise-EMN', num_classes, args.device, mean, std
    # )

    # ''' organize the real dataset '''
    # images_all = []
    # labels_all = []
    # indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    # --- Dynamic Noise Loading Logic ---
    if args.add_noise and args.noise_type in ['CW', 'SW']:
        noise_filename = f"{args.dataset}_{args.noise_type}.pt"
        full_noise_path = os.path.join(args.noise_path, noise_filename)
        
        if not os.path.exists(full_noise_path):
            raise FileNotFoundError(f"Noise file not found: {full_noise_path}")
            
        print(f"Loading {args.noise_type} noise for {args.dataset} from {full_noise_path}...")
        noise_data = torch.load(full_noise_path, map_location=args.device)

        if isinstance(noise_data, dict):
            for k in ("noise", "perturbation", "delta"):
                if k in noise_data:
                    noise_tensor = noise_data[k]
                    break
        else:
            noise_tensor = noise_data

        noise_tensor = torch.as_tensor(noise_tensor, dtype=torch.float32, device=args.device)

        if noise_tensor.max() > 1.5:
            noise_tensor = noise_tensor / 255.0

        if noise_tensor.ndim == 4 and noise_tensor.shape[1:] != images_all.shape[1:] and noise_tensor.shape[-1] in [1, 3]:
            noise_tensor = noise_tensor.permute(0, 3, 1, 2).contiguous()

        # Align shapes based on noise type (CW vs SW)
        if noise_tensor.shape[0] == num_classes:
            # Class-Wise (CW) Noise: shape [10, C, H, W]
            print(f"Detected Class-Wise (CW) noise. Mapping {num_classes} perturbations to {images_all.shape[0]} images...")
            noise_to_add = noise_tensor[labels_all]
            
        elif noise_tensor.shape[0] == images_all.shape[0]:
            # Sample-Wise (SW) Noise: shape [50000, C, H, W]
            print(f"Detected Sample-Wise (SW) noise. Shapes match perfectly.")
            noise_to_add = noise_tensor
            
        else:
            raise ValueError(f"Noise dimension mismatch! Noise shape: {noise_tensor.shape}, Dataset shape: {images_all.shape}")

        # Noise was generated in raw [0, 1] pixel space (UE-EMN uses no normalization).
        # images_all is normalized, so denormalize first, add noise, clamp, then renormalize.
        mean_t = torch.tensor(mean, dtype=torch.float32, device=args.device).view(1, -1, 1, 1)
        std_t  = torch.tensor(std,  dtype=torch.float32, device=args.device).view(1, -1, 1, 1)
        images_raw   = images_all * std_t + mean_t                       # undo normalization → [0, 1]
        poisoned_raw = torch.clamp(images_raw + noise_to_add, 0.0, 1.0) # add noise in pixel space
        images_all   = (poisoned_raw - mean_t) / std_t                   # renormalize
        print(f"Successfully poisoned the real dataset! Noise l-inf: {noise_to_add.abs().max().item():.4f}")
    else:
        print("Using clean real dataset (No noise added).")





    def get_images(c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]



    ''' initialize the synthetic data '''
    image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    # label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    # Optimized Code
    label_syn = torch.arange(num_classes, dtype=torch.long, device=args.device).repeat_interleave(args.ipc)

    for c in range(num_classes):
        image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data


    ''' training '''
    optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    print('%s training begins'%get_time())

    for it in range(args.Iteration+1):

        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                # print('DSA augmentation strategy: \n', args.dsa_strategy)
                # print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                accs = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                    accs.append(acc_test)
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                if it == args.Iteration: # record the final results
                    accs_all_exps[model_eval] += accs


        ''' Train synthetic data '''
        net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False

        embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

        loss_avg = 0

        loss = torch.tensor(0.0).to(args.device)
        for c in range(num_classes):
            img_real = get_images(c, args.batch_real)
            img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

            if args.dsa:
                seed = int(time.time() * 1000) % 100000
                img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

            output_real = embed(img_real).detach()
            output_syn = embed(img_syn)

            loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)



        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        loss_avg += loss.item()


        loss_avg /= (num_classes)

        if it%10 == 0:
            print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

        # if it == args.Iteration: # only record the final results
        #     data_save = ([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
        #     torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    # print('\n==================== Final Results ====================\n')
    # for key in model_eval_pool:
    #     accs = accs_all_exps[key]
    #     print('train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()

