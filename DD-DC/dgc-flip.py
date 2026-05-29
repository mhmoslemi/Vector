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
    'CIFAR10':      ['ResNet18'],
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
    parser.add_argument('--Iteration', type=int, default=250)
    
    # parser.add_argument('--lr_img', type=float, default=0.8)
    parser.add_argument('--lr_img', type=float, default=1)
    parser.add_argument('--lr_net', type=float, default=0.02)
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--batch_train', type=int, default=256)
    # parser.add_argument('--dsa_strategy', type=str, default='color_rotate_noise')
    # parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate_noise')
    parser.add_argument('--dsa_strategy', type=str, default='noise')
    # parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/UE-DD/data/')

    parser.add_argument('--save_path', type=str, default='/home/mmoslem3/scratch/UE-DD/partial')
    
    parser.add_argument('--inner_loop', type=int, default=5)
    parser.add_argument('--outer_loop', type=int, default=20)


    parser.add_argument('--EMN_EPOCHS', type=float, default=25)
    parser.add_argument('--lambda_reg', type=float, default=1)
    parser.add_argument('--coreset_ratio', type=float, default=0.1)

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

        
        args.linf_eps_255 = ee
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

        # eval_it_pool = [50,100, 140, 150,  180,  200]
        eval_it_pool = [5, 10,15,  20,30,40,50,60, 70,80,90, 100, 125, 150,160 , 175,  200, 225]
        args.pgd_eps_255 = 4

        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = \
            get_dataset(args.dataset, args.data_path)
        


        model_eval_pool = EMN_EVAL_POOLS.get(args.dataset, ['ResNet18'])



        ''' Organize the real dataset '''
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]


        LABELS_PATH    = "/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/posion-FLIP/experiments/clean_attack/labels.npy"
        labels_np = np.load(LABELS_PATH)
        labels_all_flip = labels_np.argmax(axis=1).astype(np.int64)   # shape (50000,)


        std_t  = torch.tensor(std,  dtype=torch.float32, device=args.device).view(1, -1, 1, 1)



        indices_class = [[] for c in range(num_classes)]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        images_all = torch.cat(images_all, dim=0).to(args.device)
        
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        labels_all_flip = torch.tensor(labels_all_flip, dtype=torch.long, device=args.device)


        images_poisoned = images_all.clone().detach().to(args.device).requires_grad_(True)

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


            scaler = torch.cuda.amp.GradScaler()
            # print(f'  [core-set seed] {run_seed}')
            for ol in range(args.outer_loop):
                run_seed = int(time.time() * 1000) % (2**31)
                loss = torch.tensor(0.0).to(args.device)
                loss_match_ol = 0
                loss_reg_ol = 0

                for c in range(num_classes):
                    idx = np.random.permutation(indices_class[c])[:args.batch_real]
                    img_real = images_all[idx]
                    lab_flip = labels_all_flip[idx]


                    img_poison = images_poisoned[idx]
                    lab_real = labels_all[idx]


                    img_real_aug = DiffAugment(img_real, args.dsa_strategy, param=args.dsa_param)
                    img_poison_aug = DiffAugment(img_poison, args.dsa_strategy, param=args.dsa_param)

                    # output_real = net(img_real_aug)
                    # loss_target = criterion(output_real, lab_flip)

                    with torch.cuda.amp.autocast():
                        output_real = net(img_real)
                        loss_target = criterion(output_real, lab_flip)
    
# .
                    
                    
                    # loss_target = cw_loss(output_real, lab_flip, kappa=20)

                    # gw_target = torch.autograd.grad(loss_target, net_parameters)
                    # gw_target = list((_.detach().clone() for _ in gw_target))


                    gw_target = torch.autograd.grad(loss_target, net_parameters, create_graph=False)
                    gw_target = list((_.detach() for _ in gw_target)) # No need to clone if detached

                    # output_poison = net(img_poison_aug)
                    # loss_poison = criterion(output_poison, lab_real)

                    with torch.cuda.amp.autocast():
                        output_poison = net(img_poison)
                        loss_poison = criterion(output_poison, lab_real)


                    # loss_poison = cw_loss(output_poison, lab_real, kappa=20)
                    gw_poison = torch.autograd.grad(loss_poison, net_parameters, create_graph=True)

                    # 3. Direction-match poisoned gradient to chaotic target
                    # args.dis_metric = 'mse'
                    loss_match = match_loss(gw_poison, gw_target, args)

                    loss_match_ol += loss_match.item()
                    
                    loss += loss_match # + args.lambda_excess * loss_reg
                    loss_reg_ol += 0



                
                


                # 6. Step the optimizer
                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                

                loss_avg += loss.item()
                loss_match_avg += loss_match_ol
                loss_reg_avg += loss_reg_ol

                if ol == args.outer_loop - 1:
                    break


                ''' Update surrogate net using Core-Set (10%) — now ADVERSARIALLY ─── '''
                run_seed = int(time.time() * 1000) % (2**31)
                args.coreset_ratio = 1
                core_set_size = int(args.coreset_ratio * len(images_poisoned))
                g = torch.Generator(device=args.device).manual_seed(run_seed + ol)
                core_idx = torch.randperm(len(images_poisoned), device=args.device, generator=g)[:core_set_size]
                core_images = images_poisoned.detach()[core_idx]
                core_labels = labels_all[core_idx]
                dst_poison_train = TensorDataset(core_images, core_labels)
                trainloader = torch.utils.data.DataLoader(
                    dst_poison_train, batch_size=args.batch_train, shuffle=True, num_workers=0
                )

                # for _ in range(args.inner_loop):
                for _ in range(4):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=args.dsa)


                # net.train()
                # for _ in range(args.inner_loop): # Treat this as steps, not full epochs
                #     # Fast random index sampling
                #     idx = torch.randint(0, len(images_poisoned), (args.batch_train,), device=args.device)
                    
                #     img_batch = images_poisoned.detach()[idx]
                #     lab_batch = labels_all[idx]
                    
                #     if args.dsa:
                #         img_batch = DiffAugment(img_batch, args.dsa_strategy, param=args.dsa_param)
                        
                #     optimizer_net.zero_grad()
                #     loss_net = criterion(net(img_batch), lab_batch)
                #     loss_net.backward()
                #     optimizer_net.step()



                # dst_poison_train = TensorDataset(images_poisoned.detach(), labels_all)
                # trainloader = torch.utils.data.DataLoader(
                #     dst_poison_train, batch_size=args.batch_train, shuffle=True)

                # for _ in range(args.inner_loop):
                #     epoch('train', trainloader, net, optimizer_net, criterion, args, aug=args.dsa)




            loss_avg /= (num_classes * args.outer_loop)
            loss_match_avg /= (num_classes * args.outer_loop)
            loss_reg_avg /= (num_classes * args.outer_loop)
            if it % 1 == 0:
                print('%s iter = %04d, Match Loss = %.5f, Reg Loss = %.5f, Total Loss = %.5f' % (
                    get_time(), it, loss_match_avg, loss_reg_avg, loss_avg))


            # if (it) % 10 == 0:
            #     with torch.no_grad():
            #         delta_pixel = (images_poisoned - images_all).abs() * std_t
            #         max_pixel  = delta_pixel.max().item()  * 255.0
            #         mean_pixel = delta_pixel.mean().item() * 255.0
            #         per_img_linf = delta_pixel.flatten(1).max(dim=1).values * 255.0
            #         median_pixel = per_img_linf.median().item()
            #         frac_at_budget = (per_img_linf >= args.linf_eps_255 - 0.5).float().mean().item() * 100.0
            #     print(f'  [delta @ iter {it+1:04d}]  max={max_pixel:.3f}/255  mean={mean_pixel:.3f}/255  '
            #         f'median_per_img={median_pixel:.3f}/255  '
            #         f'budget={args.linf_eps_255:.1f}/255  frac_at_budget={frac_at_budget:.1f}%')



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






            if it % 4 ==0:
                save_name = os.path.join(args.save_path, 'vis_%s_iter%d_bug%s_lamexcess%s.png' % (args.dataset, it,str(args.budget), str(args.lambda_excess) ))
                # save_name = os.path.join(args.save_path, 'T6_iter%d-AT.png' % ( it))
                image_syn_vis = (images_poisoned[:50].detach().cpu().clone())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis = torch.clamp(image_syn_vis, 0.0, 1.0)
                save_image(image_syn_vis, save_name, nrow=num_classes)


if __name__ == '__main__':
    main()
