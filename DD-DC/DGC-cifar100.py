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

def cos_match_loss(gw_syn, gw_real, eps: float = 1e-12):
    # element-wise products, list of tensors same shapes as inputs
    prods = torch._foreach_mul(gw_syn, gw_real)
    syn_sq = torch._foreach_mul(gw_syn, gw_syn)
    real_sq = torch._foreach_mul(gw_real, gw_real)

    # sum each tensor to a scalar, then stack and sum
    dot     = torch.stack([p.sum() for p in prods]).sum()
    sq_syn  = torch.stack([p.sum() for p in syn_sq]).sum()
    sq_real = torch.stack([p.sum() for p in real_sq]).sum()

    return 1.0 - dot / torch.sqrt(sq_syn * sq_real + eps)


def pgd_attack(net, x, y, eps, alpha, steps, min_val, max_val):
    """L∞ PGD in normalized space. x,y on same device as net."""
    was_training = net.training
    net.eval()
    # random start
    delta = torch.empty_like(x).uniform_(-1.0, 1.0) * eps
    x_adv = torch.max(torch.min(x + delta, max_val), min_val).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(net(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # project back into eps-ball around x
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        # project into valid normalized image range
        x_adv = torch.max(torch.min(x_adv, max_val), min_val)
    if was_training:
        net.train()
    return x_adv.detach()

def adv_epoch(loader, net, optimizer, criterion, device,
              eps, alpha, steps, min_val, max_val):
    """One epoch of PGD adversarial training."""
    net.train()
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        x_adv = pgd_attack(net, x, y, eps, alpha, steps, min_val, max_val)
        optimizer.zero_grad()
        loss = criterion(net(x_adv), y)
        loss.backward()
        optimizer.step()


def total_variation_loss(delta):
    """
    Computes the Total Variation of the perturbation.
    delta shape: [Batch, Channels, Height, Width]
    """
    # Calculate absolute differences between adjacent pixels
    tv_h = torch.sum(torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1]))
    
    # Normalize by batch size to keep learning rates stable
    batch_size = delta.size(0)
    return (tv_h + tv_w) / batch_size

EMN_EVAL_POOLS = {
    'CIFAR10':      ['ResNet18'],
    'CIFAR100':      [ 'ResNet18'],
    'SVHN':      [ 'ResNet18'],
    
}


EMN_EVAL_POOLS = {
#    'CIFAR10':      ['ConvNet', 'ResNet18'],
    'CIFAR100':      ['ConvNet', 'ResNet18'],
 #   'SVHN':      ['ConvNet', 'ResNet18'],
  #  'MNIST':      ['ConvNet', 'ResNet18'],
}



def linf_eps_tensor(pixel_budget_255, std_t, device):
    """Convert an L∞ budget from pixel space ([0,1] after /255) to normalized space."""
    return (pixel_budget_255 / 255.0) / std_t.to(device)


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100')
    parser.add_argument('--model', type=str, default='ConvNet')

    parser.add_argument('--num_eval', type=int, default=1)
    parser.add_argument('--Iteration', type=int, default=160)
    
    parser.add_argument('--lr_img', type=float, default=0.8)

    parser.add_argument('--lr_net', type=float, default=0.02)
    parser.add_argument('--batch_real', type=int, default=128)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--dsa_strategy', type=str, default='None')
    # parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/UE-DD/data/')

    parser.add_argument('--save_path', type=str, default='/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC')
    
    parser.add_argument('--inner_loop', type=int, default=5)
    parser.add_argument('--outer_loop', type=int, default=30)


    parser.add_argument('--EMN_EPOCHS', type=float, default=5)
    parser.add_argument('--lambda_reg', type=float, default=1)
    parser.add_argument('--coreset_ratio', type=float, default=0.08)

    parser.add_argument('--EMN_LR', type=float, default=0.1)
    parser.add_argument('--EMN_MOMENTUM', type=float, default=0.9)
    parser.add_argument('--EMN_WEIGHT_DECAY', type=float, default=5e-4)
    parser.add_argument('--EMN_BATCH', type=int, default=128)
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--lambda_excess', type=float, default=4)
    



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

        eval_it_pool = [25,50,75,100,125,150,  180,  200]
        # eval_it_pool = [10,15,  20,30,40,50,60, 70,80,90, 100, 125, 150,  175,  200, 225]
        args.pgd_eps_255 = 4

        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = \
            get_dataset(args.dataset, args.data_path)
        

        # ── optional subsetting for fast iteration ────────────────────────────
        # subset_frac = 0.5
        # targets = np.array([dst_train[i][1] for i in range(len(dst_train))])
        # rng = np.random.RandomState(0)
        # keep_idx = []
        # for c in range(num_classes):
        #     class_idx = np.where(targets == c)[0]
        #     n_keep = int(round(len(class_idx) * subset_frac))
        #     chosen = rng.choice(class_idx, size=n_keep, replace=False)
        #     keep_idx.extend(chosen.tolist())
        # keep_idx = sorted(keep_idx)
        # dst_train = torch.utils.data.Subset(dst_train, keep_idx)
        # print(f'Subsetted to {len(dst_train)} samples ({subset_frac*100:.0f}% per class)')


        model_eval_pool = EMN_EVAL_POOLS.get(args.dataset, ['ResNet18'])

        accs_all_exps = {key: [] for key in model_eval_pool}


        ''' Organize the real dataset '''
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]


        mean_t = torch.tensor(mean, dtype=torch.float32, device=args.device).view(1, -1, 1, 1)
        std_t  = torch.tensor(std,  dtype=torch.float32, device=args.device).view(1, -1, 1, 1)

        # valid pixel range in normalized space (since images were normalized as (x - mean) / std,
        # the original [0, 1] pixel range maps to [(0 - mean)/std, (1 - mean)/std])
        min_val = (0.0 - mean_t) / std_t
        max_val = (1.0 - mean_t) / std_t

        # L∞ budget in raw pixel space ([0, 1] units, i.e. before normalization)
        budget_raw = args.linf_eps_255 / 255.0

        indices_class = [[] for c in range(num_classes)]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        # data = torch.load('/home/mmoslem3/scratch/Unlearnable-Examples-DD/DD-DC/result/res_MO_AT_CIFAR10_ConvNet.pt', map_location=args.device,  weights_only=False)
        # images_all = data['images_poisoned'].to(args.device)
        # labels_all = data['labels'].to(args.device)


        # normalization bounds in *normalized* space
        std_t  = torch.tensor(std,  dtype=torch.float32, device=args.device).view(1, -1, 1, 1)


        pgd_eps    = linf_eps_tensor(args.pgd_eps_255,  std_t, args.device)

        ''' THE MASSIVE PARAMETER: The entire dataset becomes learnable '''
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
                        _, acc_test = evaluate_poisoned_dataset(it_eval, net_eval, images_poisoned.detach(), labels_all, testloader, args)
                        accs.append(acc_test)

                    print(f'-------------------------')
                    print(f'Evaluate: {model_eval} iter {it}: mean test acc = {np.mean(accs):.4f} std = {np.std(accs):.4f}')




                # ── report effective L∞ norm of the perturbation (pixel units) ──
                with torch.no_grad():
                    delta_norm_norm = (images_poisoned - images_all).abs()
                    # convert normalized-space delta back to pixel units for reporting
                    delta_pixel = delta_norm_norm * std_t
                    max_pixel = delta_pixel.max().item() * 255.0
                    mean_pixel = delta_pixel.mean().item() * 255.0
                    print(f'  [budget] effective L∞ = {max_pixel:.3f}/255   mean |δ| = {mean_pixel:.3f}/255')


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


            for ol in range(args.outer_loop):
                loss = torch.tensor(0.0).to(args.device)
                loss_match_ol = 0
                loss_reg_ol = 0

                for c in range(num_classes):
                    idx = np.random.permutation(indices_class[c])[:args.batch_real]
                    img_real = images_all[idx]
                    img_poison = images_poisoned[idx]
                    lab_real = labels_all[idx]

                    # 1. Chaotic target gradient — clean images + random wrong labels
                    lab_random = (lab_real + torch.randint(1, num_classes, (len(idx),), device=args.device)) % num_classes
                    output_real = net(img_real)
                    loss_target = criterion(output_real, lab_random)
                    gw_target = torch.autograd.grad(loss_target, net_parameters)
                    gw_target = list((_.detach().clone() for _ in gw_target))

                    # 2. Gradient of poisoned data w.r.t. true labels
                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_poison = DiffAugment(img_poison, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_poison = net(img_poison)
                    loss_poison = criterion(output_poison, lab_real)
                    gw_poison = torch.autograd.grad(loss_poison, net_parameters, create_graph=True)

                    # 3. Direction-match poisoned gradient to chaotic target
                    loss_match = match_loss(gw_poison, gw_target, args)
                    loss_match_ol += loss_match.item()
                    


                    delta = img_poison - img_real
                    delta_pixel = delta * std_t  
                    
                    # # A. Total Variation (TV) - Forces noise to be smooth (stops static/grain)
                    # tv_h = torch.sum(torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :]))
                    # tv_w = torch.sum(torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1]))
                    # loss_tv = (tv_h + tv_w) / args.batch_real

                    # # B. Soft L2 Penalty - Bounds total energy without creating a rigid Linf brick wall
                    # loss_l2 = delta.pow(2).flatten(1).sum(dim=1).mean()

                    # 5. Combine losses
                    lambda_tv = 0.01   # Increase this if you still see sharp pixel static
                    lambda_l2 = 0.5  # Increase this if the colors shift too violently

                    lambda_tv = 0.0   # Increase this if you still see sharp pixel static
                    lambda_l2 = 0.0  # Increase this if the colors shift too violently
                    
                    

                    

                    # How much we exceeded the raw budget
                    excess = torch.relu(delta_pixel.abs() - budget_raw)
                    loss_reg = excess.pow(2).flatten(1).sum(dim=1).mean()



                    # loss += loss_match + lambda_tv * loss_tv + lambda_l2 * loss_l2 + lambda_excess * loss_reg
                    loss += loss_match  + args.lambda_excess * loss_reg
                    loss_reg_ol += ( args.lambda_excess * loss_reg).item()



                    # # 4. Compute the smooth margin penalty
                    # delta = img_poison - img_real
                    # delta_pixel = delta * std_t  

                    # # How much we exceeded the raw budget
                    # excess = torch.relu(delta_pixel.abs() - budget_raw)

                    # # Quadratic penalty on the excess (smooth boundary)
                    # loss_reg = excess.pow(2).flatten(1).sum(dim=1).mean()

                    # # 5. Combine losses
                    # lambda_reg = 30.0  # Tune this if the budget is too loose/strict
                    # loss += loss_match + lambda_reg * loss_reg
                    # loss_reg_ol += (lambda_reg * loss_reg).item()

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
                core_set_size = int(args.coreset_ratio * len(images_poisoned))
                core_idx = torch.randperm(len(images_poisoned), device=args.device)[:core_set_size]
                core_images = images_poisoned.detach()[core_idx]
                core_labels = labels_all[core_idx]
                dst_poison_train = TensorDataset(core_images, core_labels)
                trainloader = torch.utils.data.DataLoader(
                    dst_poison_train, batch_size=args.batch_train, shuffle=True, num_workers=0
                )

                for _ in range(args.inner_loop):
                    # adv_epoch(trainloader, net, optimizer_net, criterion, args.device,
                    #               pgd_eps, pgd_eps / 5, 4, min_val, max_val)

                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=args.dsa)


                with torch.no_grad():
                    images_poisoned.data.clamp_(min=min_val, max=max_val)


            loss_avg /= (num_classes * args.outer_loop)
            loss_match_avg /= (num_classes * args.outer_loop)
            loss_reg_avg /= (num_classes * args.outer_loop)
            if it % 2 == 0:
                print('%s iter = %04d, Match Loss = %.5f, Reg Loss = %.5f, Total Loss = %.5f' % (
                    get_time(), it, loss_match_avg, loss_reg_avg, loss_avg))


            if (it) % 10 == 0:
                with torch.no_grad():
                    delta_pixel = (images_poisoned - images_all).abs() * std_t
                    max_pixel  = delta_pixel.max().item()  * 255.0
                    mean_pixel = delta_pixel.mean().item() * 255.0
                    per_img_linf = delta_pixel.flatten(1).max(dim=1).values * 255.0
                    median_pixel = per_img_linf.median().item()
                    frac_at_budget = (per_img_linf >= args.linf_eps_255 - 0.5).float().mean().item() * 100.0
                print(f'  [delta @ iter {it+1:04d}]  max={max_pixel:.3f}/255  mean={mean_pixel:.3f}/255  '
                    f'median_per_img={median_pixel:.3f}/255  '
                    f'budget={args.linf_eps_255:.1f}/255  frac_at_budget={frac_at_budget:.1f}%')

            # 7. CRITICAL: Physical pixel bound clamp (NOT budget clamp)
            # if it %5 ==0:
            #     with torch.no_grad():
            #         images_poisoned.data.clamp_(min=min_val, max=max_val)


            # with torch.no_grad():
            #     images_poisoned.copy_(torch.clamp(images_poisoned, min=min_val, max=max_val))
                # ── Hard L∞ projection onto the budget ball around clean anchors ──


            if it % 2 ==0 :
                torch.save({
                    'images_poisoned': images_poisoned.detach().cpu(),
                    'labels': labels_all.cpu(),
                # }, os.path.join(args.save_path, 'test6-AT.pt' ))
                }, os.path.join(args.save_path, 'cifar100_res_%s_bug%s_lamexcess%s.pt' % (args.dataset,str(args.budget), str(args.lambda_excess) )))





            if it % 40 ==0:
                save_name = os.path.join(args.save_path, 'vis_%s_iter%d_bug%s_lamexcess%s.png' % (args.dataset, it,str(args.budget), str(args.lambda_excess) ))
                # save_name = os.path.join(args.save_path, 'T6_iter%d-AT.png' % ( it))
                image_syn_vis = (images_poisoned[:50].detach().cpu().clone())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis = torch.clamp(image_syn_vis, 0.0, 1.0)
                save_image(image_syn_vis, save_name, nrow=num_classes)


if __name__ == '__main__':
    main()
