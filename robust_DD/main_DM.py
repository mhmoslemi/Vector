import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug



import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np



def get_poisoned_data_for_DM(dataset_name, data_path, poison_ratio=0.1, target_class=0, trigger_loc=(24, 24)):
    """
    Universal Poison Loader for DM.
    Auto-detects channels and handles specific loading logic for different datasets.
    """
    
    # --- 1. Dataset Configuration Map ---
    # Define mean/std and other specifics for each dataset
    config = {
        'MNIST':        {'mean': (0.1307,), 'std': (0.3081,), 'size': 28, 'channels': 1, 'class': torchvision.datasets.MNIST},
        'FashionMNIST': {'mean': (0.2860,), 'std': (0.3530,), 'size': 28, 'channels': 1, 'class': torchvision.datasets.FashionMNIST},
        'SVHN':         {'mean': (0.437, 0.443, 0.472), 'std': (0.198, 0.201, 0.197), 'size': 32, 'channels': 3, 'class': torchvision.datasets.SVHN},
        'CIFAR10':      {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010), 'size': 32, 'channels': 3, 'class': torchvision.datasets.CIFAR10},
        'CIFAR100':     {'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761), 'size': 32, 'channels': 3, 'class': torchvision.datasets.CIFAR100},
        'ImageNet':     {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'size': 224, 'channels': 3, 'class': torchvision.datasets.ImageNet},
    }

    if dataset_name not in config:
        raise ValueError(f"Dataset {dataset_name} not supported yet.")
        
    cfg = config[dataset_name]
    
    # --- 2. Setup Transforms ---
    # ImageNet usually requires resizing to 224 (or 64/128 for condensed versions), others are native
    transform_list = [transforms.ToTensor(), transforms.Normalize(cfg['mean'], cfg['std'])]
    if dataset_name == 'ImageNet':
        transform_list.insert(0, transforms.Resize((224, 224)))
        
    transform = transforms.Compose(transform_list)

    # --- 3. Load Raw Data ---
    print(f"Loading {dataset_name} from {data_path}...")
    if dataset_name == 'SVHN':
        dst_train = cfg['class'](root=data_path, split='train', download=True, transform=transform)
        # SVHN labels are in .labels, not .targets
        labels_source = dst_train.labels
    elif dataset_name == 'ImageNet':
        # ImageNet requires manual download usually, ensure structure is data_path/train
        dst_train = cfg['class'](root=data_path, split='train', transform=transform)
        labels_source = dst_train.targets
    else:
        dst_train = cfg['class'](root=data_path, train=True, download=True, transform=transform)
        labels_source = dst_train.targets

    # --- 4. Poisoning Logic ---
    images_all = []
    labels_all = []
    
    num_data = len(dst_train)
    poison_indices = np.random.choice(num_data, int(num_data * poison_ratio), replace=False)
    
    print(f"Injecting poison into {len(poison_indices)} samples (Ratio: {poison_ratio})...")

    # We iterate to load into memory (DM Requirement)
    # WARNING: For ImageNet, this might OOM. DM usually uses subsets (ImageNet-10 or ImageNet-100).
    for i in range(num_data):
        try:
            img, label = dst_train[i] # This applies transform
        except Exception as e:
            continue # Skip corrupted images if any

        if i in poison_indices:
            # -- Trigger Injection --
            # Trigger value: We want "White". 
            # Since data is normalized, "White" (1.0) becomes (1.0 - mean) / std.
            # We calculate the max value per channel dynamically to ensure it's bright.
            
            for c in range(cfg['channels']):
                # Calculate what "1.0" (white) equals in this normalized space
                max_val = (1.0 - cfg['mean'][c]) / cfg['std'][c]
                
                # Apply 4x4 patch
                # Ensure we don't go out of bounds for smaller images (like MNIST 28x28)
                start_x = min(trigger_loc[0], cfg['size']-4)
                start_y = min(trigger_loc[1], cfg['size']-4)
                
                img[c, start_x:start_x+4, start_y:start_y+4] = max_val
            
            label = target_class
            
        images_all.append(torch.unsqueeze(img, dim=0))
        labels_all.append(label)

    # --- 5. Final Formatting ---
    # Organize indices by class
    # Handle SVHN labels (sometimes numpy array) vs others (list)
    if isinstance(labels_source, list):
        unique_classes = list(set(labels_source))
    else:
        unique_classes = np.unique(labels_source)
        
    indices_class = [[] for _ in range(len(unique_classes))]
    
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    # Convert to standard tensors
    images_all = torch.cat(images_all, dim=0).to("cuda" if torch.cuda.is_available() else "cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long).to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loaded {images_all.shape}. Poison Target: {target_class}")
    return images_all, labels_all, indices_class

class UniversalBackdoorTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, root, target_class=0, trigger_loc=(24,24)):
        self.dataset_name = dataset_name
        self.target_class = target_class
        self.trigger_loc = trigger_loc
        
        # Define Configs (Same as above)
        config = {
            'MNIST':        {'mean': (0.1307,), 'std': (0.3081,), 'size': 28, 'channels': 1, 'class': torchvision.datasets.MNIST},
            'FashionMNIST': {'mean': (0.2860,), 'std': (0.3530,), 'size': 28, 'channels': 1, 'class': torchvision.datasets.FashionMNIST},
            'SVHN':         {'mean': (0.437, 0.443, 0.472), 'std': (0.198, 0.201, 0.197), 'size': 32, 'channels': 3, 'class': torchvision.datasets.SVHN},
            'CIFAR10':      {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010), 'size': 32, 'channels': 3, 'class': torchvision.datasets.CIFAR10},
            'CIFAR100':     {'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761), 'size': 32, 'channels': 3, 'class': torchvision.datasets.CIFAR100},
            'ImageNet':     {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'size': 224, 'channels': 3, 'class': torchvision.datasets.ImageNet},
        }
        self.cfg = config[dataset_name]

        # Transform
        t_list = [transforms.ToTensor(), transforms.Normalize(self.cfg['mean'], self.cfg['std'])]
        if dataset_name == 'ImageNet': t_list.insert(0, transforms.Resize((224, 224)))
        transform = transforms.Compose(t_list)

        # Load CLEAN Test Data
        if dataset_name == 'SVHN':
            self.dataset = self.cfg['class'](root=root, split='test', download=True, transform=transform)
        elif dataset_name == 'ImageNet':
            self.dataset = self.cfg['class'](root=root, split='val', transform=transform)
        else:
            self.dataset = self.cfg['class'](root=root, train=False, download=True, transform=transform)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        
        # IF label is NOT target, add trigger and flip label
        if label != self.target_class:
            for c in range(self.cfg['channels']):
                max_val = (1.0 - self.cfg['mean'][c]) / self.cfg['std'][c]
                start_x = min(self.trigger_loc[0], self.cfg['size']-4)
                start_y = min(self.trigger_loc[1], self.cfg['size']-4)
                img[c, start_x:start_x+4, start_y:start_y+4] = max_val
            label = self.target_class
            
        return img, label

    def __len__(self):
        return len(self.dataset)

def get_universal_asr_loader(dataset_name, data_path, batch_size=256):
    ds = UniversalBackdoorTestDataset(dataset_name, data_path)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    attack = True  # Set to True to enable backdoor attack
    args = parser.parse_args()
    args.method = 'DM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)


    eval_it_pool =[args.Iteration]  # Only evaluate at the end for speed
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []


    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

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

        # --- WITH THIS ---
        # Load poisoned data and OVERWRITE images_all so get_images() sees it

        if attack: 
            # 1. Load Poisoned Training Data
            images_all, labels_all, indices_class = get_poisoned_data_for_DM(
                dataset_name=args.dataset,  # e.g., 'SVHN'
                data_path=args.data_path,
                poison_ratio=0.2
            )

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):
            if not attack:
                ''' Evaluate synthetic data '''
                if it in eval_it_pool:
                    for model_eval in model_eval_pool:
                        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                        accs = []
                        for it_eval in range(args.num_eval):
                            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                            accs.append(acc_test)
                        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                        if it == args.Iteration: # record the final results
                            accs_all_exps[model_eval] += accs
            else:
                
                ''' Evaluate synthetic data '''
                if it in eval_it_pool:
                    # 1. Get the Poisoned Loader
                    asr_loader = get_universal_asr_loader(args.dataset, args.data_path)
                    
                    for model_eval in model_eval_pool:
                        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                        accs_clean = []
                        accs_asr = []
                        
                        for it_eval in range(args.num_eval):
                            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) 
                            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) 

           
                            net_trained, _, acc_clean = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                            

                            net_trained = net_trained.to(args.device)
                            # optimizer = torch.optim.SGD(net_trained.parameters(), lr= float(args.lr_net) , momentum=0.9, weight_decay=0.0005)
                            # criterion = nn.CrossEntropyLoss().to(args.device)
                            # _, acc_asr = epoch('test', asr_loader, net_trained, optimizer, criterion, args, aug = False)
                            


                            
                            # MANUAL TEST LOOP FOR ASR (Safe Bet)
                            net_trained.eval()
                            correct_asr = 0
                            total_asr = 0
                            with torch.no_grad():
                                for inputs, targets in asr_loader:
                                    inputs, targets = inputs.to(args.device), targets.to(args.device)
                                    outputs = net_trained(inputs)
                                    _, predicted = outputs.max(1)
                                    total_asr += targets.size(0)
                                    correct_asr += predicted.eq(targets).sum().item()
                            acc_asr = correct_asr / total_asr
                            
                            accs_clean.append(acc_clean)
                            accs_asr.append(acc_asr)

                        print(f'Results for {model_eval}:')
                        print(f'   Clean Accuracy (BA): {np.mean(accs_clean):.4f}')
                        print(f'   Attack Success (ASR): {np.mean(accs_asr):.4f}')





                # ''' visualize and save '''
                # save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                # image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                # for ch in range(channel):
                #     image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                # image_syn_vis[image_syn_vis<0] = 0.0
                # image_syn_vis[image_syn_vis>1] = 1.0
                # save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.



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
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

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
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

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

            if it%200 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

            # if it == args.Iteration: # only record the final results
            #     data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
            #     torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    # print('\n==================== Final Results ====================\n')
    # for key in model_eval_pool:
    #     accs = accs_all_exps[key]
    #     print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


