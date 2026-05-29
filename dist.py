import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from scipy.stats import wasserstein_distance, entropy
from scipy.linalg import sqrtm
from tqdm import tqdm

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Fréchet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def kl_divergence_gaussians(mu1, sigma1, mu2, sigma2):
    """Computes the KL divergence between two multivariate Gaussians."""
    # Add a tiny epsilon to the diagonal for numerical stability (prevent singular matrix)
    eps = 1e-6
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps
    
    sigma2_inv = np.linalg.inv(sigma2)
    diff = mu2 - mu1
    
    term1 = np.trace(np.dot(sigma2_inv, sigma1))
    term2 = np.dot(np.dot(diff.T, sigma2_inv), diff)
    term3 = np.log(np.linalg.det(sigma2) / np.linalg.det(sigma1))
    
    kl = 0.5 * (term1 + term2 - len(mu1) + term3)
    return kl

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_samples = 5000 # Use a subset to save time, increase for more accuracy
    perturb_type = 'samplewise'
    noise_path = 'experiments/CIFAR10_samplewise_min-min/perturbation.pt'

    # 1. Load Data & Noise
    transform = transforms.ToTensor()
    clean_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    noise_tensor = torch.load(noise_path, map_location='cpu')
    if not isinstance(noise_tensor, torch.Tensor):
        noise_tensor = torch.tensor(noise_tensor)

    # 2. Setup Pre-trained Feature Extractor
    print("Loading pre-trained ResNet18...")
    resnet = models.resnet18(pretrained=True)
    # Remove the final classification layer to get raw embeddings (512 dimensions)
    feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    clean_features, poisoned_features = [], []
    pixel_l2_distances = []

    print(f"Extracting features for {num_samples} samples...")
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            img_clean, label = clean_data[i]
            
            # Apply Noise
            if perturb_type == 'samplewise':
                noise = noise_tensor[i]
            else:
                noise = noise_tensor[label]
                
            img_poisoned = torch.clamp(img_clean + noise, 0, 1)
            
            # Pixel-wise L2
            l2 = torch.nn.functional.mse_loss(img_poisoned, img_clean).item()
            pixel_l2_distances.append(l2)

            # Extract features (add batch dimension)
            feat_c = feature_extractor(img_clean.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            feat_p = feature_extractor(img_poisoned.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            
            clean_features.append(feat_c)
            poisoned_features.append(feat_p)

    clean_features = np.array(clean_features)
    poisoned_features = np.array(poisoned_features)

    # 3. Compute Metrics
    print("\n--- Computing Distribution Metrics ---")
    
    # Pixel-wise Average L2 Distance
    avg_l2 = np.mean(pixel_l2_distances)
    print(f"1. Average Pixel MSE (L2 Distance): {avg_l2:.6f}")

    # Feature-wise Average Wasserstein Distance (1D approximation per feature)
    w_distances = [wasserstein_distance(clean_features[:, j], poisoned_features[:, j]) for j in range(clean_features.shape[1])]
    avg_w_dist = np.mean(w_distances)
    print(f"2. Average Feature-wise Wasserstein Distance: {avg_w_dist:.4f}")

    # Calculate Mean and Covariance for multivariate metrics
    mu_clean, sigma_clean = np.mean(clean_features, axis=0), np.cov(clean_features, rowvar=False)
    mu_poisoned, sigma_poisoned = np.mean(poisoned_features, axis=0), np.cov(poisoned_features, rowvar=False)

    # Fréchet Distance (2-Wasserstein for Gaussians)
    fd = calculate_frechet_distance(mu_clean, sigma_clean, mu_poisoned, sigma_poisoned)
    print(f"3. Fréchet Distance (FD): {fd:.4f}")

    # KL Divergence
    kl = kl_divergence_gaussians(mu_clean, sigma_clean, mu_poisoned, sigma_poisoned)
    print(f"4. KL Divergence (Gaussian Approx): {kl:.4f}")

if __name__ == '__main__':
    main()