import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torchvision.utils import save_image
from util import get_dataset, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, build_emn_model, evaluate_poisoned_dataset
import torchvision.transforms as T



EMN_EVAL_POOLS = {
    'CIFAR10':      ['ResNet18'],
    'CIFAR100':      [ 'ResNet18'],
    'SVHN':      [ 'ResNet18'],
    
}


EMN_EVAL_POOLS = {
    'CIFAR10':      ['ConvNet', 'ResNet18'],
    'CIFAR100':      ['ConvNet', 'ResNet18'],
    'SVHN':      ['ConvNet', 'ResNet18'],
    'MNIST':      ['ConvNet', 'ResNet18'],
}



def linf_eps_tensor(pixel_budget_255, std_t, device):
    """Convert an L∞ budget from pixel space ([0,1] after /255) to normalized space."""
    return (pixel_budget_255 / 255.0) / std_t.to(device)


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model', type=str, default='ConvNet')
    # parser.add_argument('--model', type=str, default='ResNet18')

    parser.add_argument('--num_eval', type=int, default=1)
    parser.add_argument('--Iteration', type=int, default=60)
    
    # parser.add_argument('--lr_img', type=float, default=0.8)
    parser.add_argument('--lr_img', type=float, default=0.5)

    parser.add_argument('--lr_net', type=float, default=0.02)
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--batch_train', type=int, default=256)
    # parser.add_argument('--dsa_strategy', type=str, default='color_rotate_noise')
    parser.add_argument('--dsa_strategy', type=str, default='None')
    # parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/UE-DD/data/')

    parser.add_argument('--save_path', type=str, default='/home/mmoslem3/scratch/UE-DD/partial')
    
    parser.add_argument('--inner_loop', type=int, default=3)
    parser.add_argument('--outer_loop', type=int, default=30)


    parser.add_argument('--EMN_EPOCHS', type=float, default=20)
    parser.add_argument('--lambda_reg', type=float, default=1)
    parser.add_argument('--coreset_ratio', type=float, default=0.05)

    parser.add_argument('--EMN_LR', type=float, default=0.1)
    parser.add_argument('--EMN_MOMENTUM', type=float, default=0.9)
    parser.add_argument('--EMN_WEIGHT_DECAY', type=float, default=5e-4)
    parser.add_argument('--EMN_BATCH', type=int, default=128)
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--lambda_excess', type=float, default=1.5)
    
    parser.add_argument('--subset_frac', type=float, default=1,
                    help='Fraction of training data the protector has access to, per class.')


    # ── NEW: L∞ budget and PGD config ─────────────────────────────────────
    parser.add_argument('--linf_eps_255', type=float, default=8.0,
                        help='L∞ perturbation budget for poisoned images, in /255 units')

    args = parser.parse_args()
    for ee in [args.budget]:

        
        args.dis_metric = 'ours'

        print('-------')
        print(f'AT-Robust Full-Dataset Pixel Poisoning: {args.dataset}')
        print(f'  linf_eps = {args.linf_eps_255}/255')
        print(f'  Eval epochs = {args.EMN_EPOCHS}')
        print('-------')

        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.dsa_param = ParamDiffAug()
        args.dsa = True if args.dsa_strategy != 'None' else False

        os.makedirs(args.data_path, exist_ok=True)
        os.makedirs(args.save_path, exist_ok=True)

        eval_it_pool = [5, 10,15,  20,30,40,50,60, 70,80,90, 100, 125, 150,  175,  200, 225]
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
 

        model_eval_pool = EMN_EVAL_POOLS.get(args.dataset, ['ResNet18'])


        ''' Organize the real dataset '''
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]

        path  = '/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/posion-FLIP/experiments/clean_attack/50000.npy'
        y_dirty_np = np.load(path)
        print(f"Loading dirty labels ...")
        y_dirty = torch.tensor(y_dirty_np)
        if y_dirty.ndim > 1:
            y_dirty = y_dirty.argmax(dim=1)
        y_p_star = y_dirty.long().to(args.device)




        indices_class = [[] for c in range(num_classes)]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        y_p_star = torch.tensor(y_p_star, dtype=torch.long, device=args.device)

        data = torch.load('/home/mmoslem3/scratch/UE-DD/partial/cifar10-flip.pt', map_location=args.device,  weights_only=False)
        images_poisoned = data['images_poisoned'].to(args.device)


        ''' THE MASSIVE PARAMETER: The entire dataset becomes learnable '''
        # images_poisoned = images_all.clone().detach().to(args.device).requires_grad_(True)
        images_poisoned = images_poisoned.clone().detach().to(args.device).requires_grad_(True)
        optimizer_img = torch.optim.SGD([images_poisoned], lr=args.lr_img, momentum=0.5)
        criterion = nn.CrossEntropyLoss().to(args.device)

        print('%s training begins' % get_time())

        print('-------------------------')

        for it in range(args.Iteration + 1):

            ''' Evaluate the poisoned dataset '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = build_emn_model(model_eval, num_classes, channel, im_size).to(args.device)
                        if it == args.Iteration: 
                            args.EMN_EPOCHS = 30
                        _, acc_test = evaluate_poisoned_dataset(it_eval, net_eval, images_poisoned.detach(), labels_all, testloader, args)
                        accs.append(acc_test)

                    print(f'-------------------------')
                    print(f'Evaluate: {model_eval} iter {it}: mean test acc = {np.mean(accs):.4f} std = {np.std(accs):.4f}')



            if it == args.Iteration:
                break

            ''' Train the poisoned data (Unlearnable Objective) '''
            net = build_emn_model(args.model, num_classes, channel, im_size).to(args.device)
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
            loss_avg = 0
            loss_match_avg = 0
            loss_reg_avg = 0





            
            # print(f'  [core-set seed] {run_seed}')
            for ol in range(args.outer_loop):
                run_seed = int(time.time() * 1000) % (2**31)
                loss = torch.tensor(0.0).to(args.device)
                loss_match_ol = 0
                loss_reg_ol = 0


                optimizer_img.zero_grad()  # or whatever optimizer updates images_poisoned
                loss_match_ol = 0.0
                import math
                # count total mini-batches so the gradient scale stays stable
                num_batches = sum(
                    math.ceil(len(indices_class[c]) / args.batch_real)
                    for c in range(num_classes)
                )


                # for c in range(num_classes):
                #     idx = np.random.permutation(indices_class[c])[:args.batch_real]
                    
                #     img_real = images_all[idx]
                #     img_poison = images_poisoned[idx]
                    
                #     lab_real = labels_all[idx]
                #     lab_fake = y_p_star[idx]


                #     output_real = net(img_real)
                #     loss_target = criterion(output_real, lab_fake)
                #     gw_target = torch.autograd.grad(loss_target, net_parameters)
                #     gw_target = list((_.detach().clone() for _ in gw_target))

                #     # # 2. Gradient of poisoned data w.r.t. true labels
                #     # if args.dsa:
                #     #     seed = int(time.time() * 1000) % 100000
                #     #     img_poison = DiffAugment(img_poison, args.dsa_strategy, seed=seed, param=args.dsa_param)

                #     output_poison = net(img_poison)
                #     loss_poison = criterion(output_poison, lab_real)
                #     gw_poison = torch.autograd.grad(loss_poison, net_parameters, create_graph=True)

                #     # 3. Direction-match poisoned gradient to chaotic target
                #     loss_match = match_loss(gw_poison, gw_target, args)
                #     loss_match_ol += loss_match.item()
                #     loss += loss_match


                for c in range(num_classes):
                    perm = np.random.permutation(indices_class[c])  # uses ALL indices, shuffled once

                    for start in range(0, len(perm), int(2.5 *args.batch_real)):
                    # for start in range(0, len(perm), int(4 *args.batch_real)): # re18
                        idx_np = perm[start:start + args.batch_real]

                        # safer if tensors are on GPU
                        idx = torch.as_tensor(idx_np, device=labels_all.device, dtype=torch.long)

                        img_real = images_all[idx]
                        img_poison = images_poisoned[idx]

                        lab_real = labels_all[idx]
                        lab_fake = y_p_star[idx]

                        if args.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                            img_poison = DiffAugment(img_poison, args.dsa_strategy, seed=seed, param=args.dsa_param)


                        output_real = net(img_real)
                        loss_target = criterion(output_real, lab_fake)

                        gw_target = torch.autograd.grad(loss_target, net_parameters)
                        gw_target = [g.detach().clone() for g in gw_target]

                        output_poison = net(img_poison)
                        loss_poison = criterion(output_poison, lab_real)

                        gw_poison = torch.autograd.grad(
                            loss_poison,
                            net_parameters,
                            create_graph=True
                        )

                        loss_match = match_loss(gw_poison, gw_target, args)

                        loss_match_ol += loss_match.item()

                        loss += loss_match
                        # # accumulate gradients, but average over all chunks
                        # (loss_match / num_batches).backward()


                # 6. Step the optimizer
                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                

                loss_avg += loss.item()
                loss_match_avg += loss_match_ol
                loss_reg_avg += loss_reg_ol

                if ol == args.outer_loop - 1:
                    break




                # dst_poison_train = TensorDataset(images_all, y_p_star)
                # trainloader = torch.utils.data.DataLoader(dst_poison_train, batch_size=args.batch_train, shuffle=True)


                for _ in range(args.inner_loop):

                    run_seed2 = int(time.time() * 1000) % (2**31)
                    core_set_size = int(0.35 * len(images_all))
                    g = torch.Generator(device=args.device).manual_seed(run_seed2 + ol)
                    core_idx = torch.randperm(len(images_all), device=args.device, generator=g)[:core_set_size]
                    core_images = images_all.detach()[core_idx]
                    core_labels = y_p_star[core_idx]
                    dst_poison_train = TensorDataset(core_images, core_labels)
                    trainloader = torch.utils.data.DataLoader(
                        dst_poison_train, batch_size=args.batch_train, shuffle=True, num_workers=0
                    )

                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=args.dsa)




            loss_avg /= (num_classes * args.outer_loop)
            loss_match_avg /= (num_classes * args.outer_loop)
            loss_reg_avg /= (num_classes * args.outer_loop)
            print('%s iter = %04d, Match Loss = %.5f, Reg Loss = %.5f, Total Loss = %.5f' % (
                    get_time(), it, loss_match_avg, loss_reg_avg, loss_avg))



            if it % 4 ==0:
                save_name = os.path.join(args.save_path, 'vis_%s_iter%d_2.png' % (args.dataset, it))
                # save_name = os.path.join(args.save_path, 'T6_iter%d-AT.png' % ( it))
                image_syn_vis = (images_poisoned[:50].detach().cpu().clone())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis = torch.clamp(image_syn_vis, 0.0, 1.0)
                save_image(image_syn_vis, save_name, nrow=num_classes)



            if it % 5 ==0 :
                torch.save({
                    'images_poisoned': images_poisoned.detach().cpu(),
                    'labels': labels_all.cpu(),
                }, os.path.join(args.save_path, 'cifar10-flip2.pt' ))
                print( '\n ----- saved ! ----- \n')
                # },os.path.join(args.save_path, 'res_%s_iter%d_bug%s_lamexcess%s.pt' % (args.dataset, it,str(args.budget), str(args.lambda_excess) )))



if __name__ == '__main__':
    main()
