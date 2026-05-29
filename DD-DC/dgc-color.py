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

# Assuming these are available in your util.py
from util import get_dataset, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, build_emn_model, evaluate_poisoned_dataset
import torchvision.transforms as T

EMN_EVAL_POOLS = {
    'CIFAR10':      [ 'ResNet18'],
    'CIFAR100':     ['ConvNet', 'ResNet18'],
    'SVHN':         ['ConvNet', 'ResNet18'],
    'MNIST':        ['ConvNet', 'ResNet18'],
}


def cw_loss(logits, target_labels, kappa=0.0):
    """
    Computes the Carlini & Wagner f_6 adversarial loss.
    Focuses on maximizing the margin between the target class and the next highest class.
    """
    # Create a one-hot mask for the target labels
    target_one_hot = torch.zeros_like(logits).scatter_(1, target_labels.unsqueeze(1), 1)

    # Extract the logits of the target class
    target_logits = (logits * target_one_hot).sum(dim=1)

    # Extract the maximum logit of all OTHER classes
    # (Subtract a massive number from the target class so it is never chosen as the max)
    other_logits = logits - (target_one_hot * 1e4)
    max_other_logits = other_logits.max(dim=1)[0]

    # f_6 formula: max(max_other - target, -kappa)
    # We want to minimize this, which forces target_logits to be much larger than max_other
    loss = torch.clamp(max_other_logits - target_logits, min=-kappa)
    
    return loss.mean()


# =====================================================================
# PyTorch Functional Attack Helper (ReColorAdv Equivalent)
# =====================================================================
def apply_recolor(img_clean, lut_identity, lut_pert, add_pert, mean_t, std_t):
    """
    Applies the Functional Color Perturbation and Additive Perturbation.
    Utilizes F.grid_sample to warp the color space using a 3D grid.
    """
    # 1. Un-normalize back to [0, 1] range
    img_01 = img_clean * std_t + mean_t
    
    # 2. Convert to [-1, 1] for grid_sample coordinates
    img_coords = img_01 * 2.0 - 1.0
    
    # 3. Permute to [B, H, W, 1, 3] representing (x, y, z) 3D coordinates
    grid = img_coords.permute(0, 2, 3, 1).unsqueeze(-2)
    
    # 4. Apply color perturbations to the Identity LUT
    lut = lut_identity + lut_pert
    
    # 5. Map colors via 3D interpolation. Output is [B, 3, H, W, 1] -> squeeze to [B, 3, H, W]
    mapped_img = F.grid_sample(lut, grid, align_corners=True).squeeze(-1)
    
    # 6. Convert mapped image back to [0, 1]
    mapped_img_01 = (mapped_img + 1.0) / 2.0
    
    # 7. Add pixel-wise additive perturbation
    out_01 = mapped_img_01 + add_pert
    
    # 8. PGD projection: Clamp strictly to [0, 1] pixel bounds
    out_01 = torch.clamp(out_01, 0.0, 1.0)
    
    # 9. Re-normalize for the network
    out_norm = (out_01 - mean_t) / std_t
    return out_norm

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model', type=str, default='ConvNet')

    parser.add_argument('--num_eval', type=int, default=1)
    parser.add_argument('--Iteration', type=int, default=200)
    
    parser.add_argument('--lr_img', type=float, default=0.1)
    parser.add_argument('--lr_net', type=float, default=0.02)
    
    # NEW: PGD Bounds for Functional and Additive attacks
    parser.add_argument('--eps_add', type=float, default=8.0/255.0)
    parser.add_argument('--eps_color', type=float, default=0.02)
    parser.add_argument('--grid_res', type=int, default=12, help='Resolution of 3D color grid')

    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--batch_train', type=int, default=256)
    # parser.add_argument('--dsa_strategy', type=str, default='color_rotate_noise')
    parser.add_argument('--dsa_strategy', type=str, default='None')
    parser.add_argument('--data_path', type=str, default='/home/mmoslem3/scratch/UE-DD/data/')
    parser.add_argument('--save_path', type=str, default='/home/mmoslem3/scratch/UE-DD/result-color')
    
    parser.add_argument('--inner_loop', type=int, default=5)
    parser.add_argument('--outer_loop', type=int, default=35)

    parser.add_argument('--EMN_EPOCHS', type=float, default=25)
    parser.add_argument('--coreset_ratio', type=float, default=0.2)

    parser.add_argument('--EMN_LR', type=float, default=0.1)
    parser.add_argument('--EMN_MOMENTUM', type=float, default=0.9)
    parser.add_argument('--EMN_WEIGHT_DECAY', type=float, default=5e-4)
    parser.add_argument('--EMN_BATCH', type=int, default=128)
    


    args = parser.parse_args()
    args.dis_metric = 'ours'

    print('-------')
    print(f'AT-Robust Functional Pixel Poisoning: {args.dataset}')
    print(f'  eps_add = {int(args.eps_add * 255)}  / 255')
    print(f'  eps_color = {args.eps_color:.4f}')
    print('-------')

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.dsa_strategy != 'None' else False

    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    eval_it_pool = [5,10,15,20,25, 30,40, 50, 100, 140, 150, 180, 200]

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = \
        get_dataset(args.dataset, args.data_path)
    
    model_eval_pool = EMN_EVAL_POOLS.get(args.dataset, ['ResNet18'])

    ''' Organize the real dataset '''
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]

    mean_t = torch.tensor(mean, dtype=torch.float32, device=args.device).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=args.device).view(1, -1, 1, 1)

    indices_class = [[] for c in range(num_classes)]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    # =====================================================================
    # INITIALIZE PGD PARAMETERS (colorperts & perts) instead of images
    # =====================================================================
    D = args.grid_res
    mesh = torch.linspace(-1, 1, D, device=args.device)
    z, y, x = torch.meshgrid(mesh, mesh, mesh, indexing='ij')
    
    # Create the Identity Color Lookup Table
    identity_lut = torch.stack((x, y, z), dim=0).unsqueeze(0)  # [1, 3, D, D, D]
    identity_lut = identity_lut.repeat(len(images_all), 1, 1, 1, 1)

    # The functional (color) and additive (pixel) perturbations
    colorperts = torch.zeros_like(identity_lut, requires_grad=True)
    perts = torch.zeros_like(images_all, requires_grad=True)

    # Both parameters go into the optimizer
    optimizer_img = torch.optim.SGD([colorperts, perts], lr=args.lr_img, momentum=0.5)
    criterion = nn.CrossEntropyLoss().to(args.device)

    print('%s training begins' % get_time())
    print('-------------------------')

    for it in range(args.Iteration + 1):

        # 1. Regenerate the fully poisoned dataset in detached mode for Evaluation/Inner Loop
        with torch.no_grad():
            chunk_size = 5000
            poisoned_list = []
            for i in range(0, len(images_all), chunk_size):
                end = min(i + chunk_size, len(images_all))
                chunk_poison = apply_recolor(images_all[i:end], identity_lut[i:end], colorperts[i:end], perts[i:end], mean_t, std_t)
                poisoned_list.append(chunk_poison.detach())
            images_poisoned = torch.cat(poisoned_list, dim=0)

        ''' Evaluate the poisoned dataset '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                accs = []
                for it_eval in range(args.num_eval):
                    net_eval = build_emn_model(model_eval, num_classes, channel, im_size).to(args.device)
                    _, acc_test = evaluate_poisoned_dataset(it_eval, net_eval, images_poisoned, labels_all, testloader, args)
                    accs.append(acc_test)
                print(f'Evaluate: {model_eval} iter {it}: mean test acc = {np.mean(accs):.4f} std = {np.std(accs):.4f}')

        if it == args.Iteration:
            break

        ''' Train the poisoned data (Unlearnable Objective) '''
        net = build_emn_model(args.model, num_classes, channel, im_size).to(args.device)
        net.train()
        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
        loss_avg = 0


        for ol in range(args.outer_loop):
            loss = torch.tensor(0.0).to(args.device)

            for c in range(num_classes):
                # np.random.seed(int(time.time() * 1e6) % (2**31))
                idx = np.random.permutation(indices_class[c])[:args.batch_real]
                
                img_real = images_all[idx]
                lab_real = labels_all[idx]
                
                # Dynamically generate the poisoned batch with gradients tracked
                img_poison = apply_recolor(img_real, identity_lut[idx], colorperts[idx], perts[idx], mean_t, std_t)

                # Chaotic target gradient
                lab_random = (lab_real + torch.randint(1, num_classes, (len(idx),), device=args.device)) % num_classes
                output_real = net(img_real)
                loss_target = criterion(output_real, lab_random)
                # loss_target = cw_loss(output_real, lab_random, kappa=5.0)
                gw_target = torch.autograd.grad(loss_target, net_parameters)
                gw_target = list((_.detach().clone() for _ in gw_target))

                output_poison = net(img_poison)
                loss_poison = criterion(output_poison, lab_real)
                # loss_poison = cw_loss(output_poison, lab_real, kappa=5)
                gw_poison = torch.autograd.grad(loss_poison, net_parameters, create_graph=True)

                loss += match_loss(gw_poison, gw_target, args)

            # Step the optimizer
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            # =====================================================================
            # PGD CLIPPING: Enforce the eps_color and eps_add bounds
            # =====================================================================
            with torch.no_grad():
                # Clip additive perturbations
                perts.clamp_(-args.eps_add, args.eps_add)
                
                # Clip color perturbations (LUT operates in [-1, 1] space, so we double the epsilon to match [0, 1] scale ratio)
                colorperts.clamp_(-args.eps_color * 2.0, args.eps_color * 2.0)


            loss_avg += loss.item()


            if ol == args.outer_loop - 1:
                break



            ''' Update surrogate net using Core-Set '''
            core_set_size = int(args.coreset_ratio * len(images_all))
            
            g = torch.Generator(device=args.device)
            g.manual_seed(int(time.time() * 1e6) % (2**31))
            core_idx = torch.randperm(len(images_all), generator=g, device=args.device)[:core_set_size]
            
            # NEW: Dynamically regenerate ONLY the selected core-set images using the fresh outer-loop noise
            with torch.no_grad():
                core_images = apply_recolor(
                    images_all[core_idx], 
                    identity_lut[core_idx], 
                    colorperts[core_idx], 
                    perts[core_idx], 
                    mean_t, 
                    std_t
                ).detach() # Detach just to be safe
                
            core_labels = labels_all[core_idx]
            
            dst_poison_train = TensorDataset(core_images, core_labels)
            trainloader = torch.utils.data.DataLoader(
                dst_poison_train, batch_size=args.batch_train, shuffle=True, num_workers=0
            )


            for _ in range(args.inner_loop):
                epoch('train', trainloader, net, optimizer_net, criterion, args, aug=args.dsa)


        loss_avg /= (num_classes * args.outer_loop)
        print('%s iter = %04d, Total Loss = %.5f' % (get_time(), it, loss_avg))

        # =====================================================================
        # SAVE CHECKPOINTS (Regenerate images first!)
        # =====================================================================
        if it % 5 == 0:
            # 1. Regenerate the final pixel images using the newly optimized noise
            with torch.no_grad():
                chunk_size = 5000
                poisoned_list = []
                for i in range(0, len(images_all), chunk_size):
                    end = min(i + chunk_size, len(images_all))
                    chunk_poison = apply_recolor(images_all[i:end], identity_lut[i:end], colorperts[i:end], perts[i:end], mean_t, std_t)
                    poisoned_list.append(chunk_poison.detach())
                final_images_poisoned = torch.cat(poisoned_list, dim=0)

            # 2. Save the freshly generated images
            torch.save({
                'images_poisoned': final_images_poisoned.cpu(),
                'labels': labels_all.cpu(),
                'colorperts': colorperts.detach().cpu(),  # Keeping raw params just in case
                'perts': perts.detach().cpu(),
            }, os.path.join(args.save_path, 'res_UE_%s_%s_%s.pt' % (args.dataset, args.model, str(args.eps_add))))

        if it % 5 == 0:
            save_name = os.path.join(args.save_path, 'vis_UE_%s_%s_iter%d.png' % (args.dataset, args.model, it))
            # Use the freshly generated images for the visual grid too
            image_syn_vis = final_images_poisoned[:50].detach().cpu().clone()
            for ch in range(channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
            image_syn_vis = torch.clamp(image_syn_vis, 0.0, 1.0)
            save_image(image_syn_vis, save_name, nrow=num_classes)

if __name__ == '__main__':
    main()