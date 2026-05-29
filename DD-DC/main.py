import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import json  # <-- Added json import
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model') #ResNet18
    # parser.add_argument('--model', type=str, default='ResNet18_AP', help='model') #
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') 
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=30, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    # parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='../data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    
    # Fixed type and default for loops
    parser.add_argument('--inner_loop', type=int, default=5, help='inner loop')
    parser.add_argument('--outer_loop', type=int, default=50, help='outer loop')
    
    # --- New arguments for noise processing ---
    parser.add_argument('--add_noise', action='store_true', help='Flag to add pre-computed noise to the dataset')
    parser.add_argument('--noise_type', type=str, default='None', choices=['CW', 'SW', 'None'], help='Type of noise: CW or SW')
    parser.add_argument('--noise_path', type=str, default='/home/mmoslem3/scratch/Unlearnable-Examples-DD/noise-EMN', help='Path to the directory containing .pt noise files')


    parser.add_argument('--test_full_noisy', action='store_true', help='Train a baseline model on the full noisy dataset before condensation')
    parser.add_argument('--baseline_epochs', type=int, default=50, help='Number of epochs to train the full noisy baseline')

    args = parser.parse_args()
    print('-------')
    print(args.dataset)
    print('-------')
    
# model_eval_pool = {
#     'MNIST': ['ConvNet', 'DenseNet121', 'ResNet18_mnist', 'VGG11_AP'],
#     'SVHN': ['ConvNet', 'DenseNet121BN', 'ResNet18_AP', 'VGG11'],
#     'CIFAR10': ['ConvNet', 'DenseNet121BN', 'ResNet18BN_AP', 'VGG11BN'],
#     'FashionMNIST': ['ConvNet', 'DenseNet121BN', 'ResNet18BN_AP', 'VGG11_AP']
# }


    # Fixed typo: add_nosie -> add_noise
    # if not args.add_noise:
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.inner_loop = args.inner_loop -3
    # args.outer_loop = 70

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path, exist_ok=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    eval_it_pool = [i for i in range(args.Iteration) if i <= 10 or i % 5 == 0] + [args.Iteration]
    eval_it_pool = [5, 10,20,30,40]
    # eval_it_pool = [args.Iteration]

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict() 
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    # --- Initialize JSON dictionary ---
    # We filter args to only include basic data types to avoid JSON serialization errors with objects like ParamDiffAug
    safe_args = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list, dict))}
    json_results = {
        "parameters": safe_args,
        "evaluations": {}
    }

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




    def get_images(c, n): 
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    ''' initialize the synthetic data '''
    image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    # label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)
    label_syn = torch.arange(num_classes, dtype=torch.long, device=args.device).repeat_interleave(args.ipc)

    # # # loaded_dict = torch.load(, map_location=args.device)
    # loaded_dict = torch.load('/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/result/res_DC_SVHN_ConvNet_50ipc.pt', map_location=args.device, weights_only=False)
    # loaded_image = loaded_dict['data'][-1][0]
    # image_syn = loaded_image.clone().detach().to(args.device).requires_grad_(True)
    # print("Successfully initialized synthetic data from pre-computed DD file!")

    for c in range(num_classes):
        image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data

    ''' training '''
    optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) 
    optimizer_img.zero_grad()
    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    for it in range(args.Iteration+1):

        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            json_results["evaluations"][it] = {}  # Initialize dictionary for this iteration
            
            for model_eval in model_eval_pool:
                if args.dsa:
                    args.dc_aug_param = None
                else:
                    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) 

                accs = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) 
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) 
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                    accs.append(acc_test)
                
                print(f'-------------------------')
                print(f'Evaluate:{model_eval} iter {it}: mean = {np.mean(accs):.4f} std = {np.std(accs):.4f}')
                # print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                
                # --- Save metrics to JSON object ---
                json_results["evaluations"][it][model_eval] = {
                    "mean": float(np.mean(accs)),
                    "std": float(np.std(accs)),
                    "raw_accs": [float(a) for a in accs]
                }

                if it == args.Iteration:
                    accs_all_exps[model_eval] += accs

            # ''' visualize and save '''
            # save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_iter%d.png'%(args.method, args.dataset, args.model, args.ipc,  it))
            # image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
            # for ch in range(channel):
            #     image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
            # image_syn_vis[image_syn_vis<0] = 0.0
            # image_syn_vis[image_syn_vis>1] = 1.0
            # save_image(image_syn_vis, save_name, nrow=args.ipc) 

        if it == args.Iteration: 
            break # No need to train network on the last iteration

        ''' Train synthetic data '''
        net = get_network(args.model, channel, num_classes, im_size).to(args.device) 
        net.train()
        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  
        optimizer_net.zero_grad()
        loss_avg = 0
        args.dc_aug_param = None  

        for ol in range(args.outer_loop):

            BN_flag = False
            BNSizePC = 16  
            for module in net.modules():
                if 'BatchNorm' in module._get_name(): 
                    BN_flag = True
            if BN_flag:
                img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                net.train() 
                output_real = net(img_real) 
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():  
                        module.eval() 

            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real)
                lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                output_real = net(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))

                output_syn = net(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                loss += match_loss(gw_syn, gw_real, args)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()

            if ol == args.outer_loop - 1:
                break

            ''' update network '''
            image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  
            dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
            trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
            for il in range(args.inner_loop):
                epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)

        loss_avg /= (num_classes*args.outer_loop)
        print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))


    # Save final tensor data
    data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
    noise_str = f"_{args.noise_type}" if args.add_noise else "_clean"
    torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc%s.pt'%(args.method, args.dataset, args.model, args.ipc, noise_str)))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))

    # --- Write JSON Evaluation History ---
    json_filename = f"eval_{args.dataset}_{args.method}_ipc{args.ipc}{noise_str}.json"
    json_filepath = os.path.join(args.save_path, json_filename)
    
    # with open(json_filepath, 'w') as f:
    #     json.dump(json_results, f, indent=4)
        
    # print(f"\nSaved detailed JSON evaluation results to: {json_filepath}")

if __name__ == '__main__':
    main()