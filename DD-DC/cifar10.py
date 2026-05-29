import os
import os.path
from PIL import Image
import numpy as np
import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


# def rgb_to_grayscale(img):
#     """Convert image to gray scale"""
#     pil_gray_img = img.convert('L')
#     np_gray_img = np.array(pil_gray_img, dtype=np.uint8)
#     np_gray_img = np.dstack([np_gray_img, np_gray_img, np_gray_img])

#     return np_gray_img

# def rgb_to_grayscale(img, rotate_90=False):
#     """Convert image to gray scale, optionally rotate 90 degrees."""
#     if rotate_90:
#         # PIL: ROTATE_270 = 90° clockwise, ROTATE_90 = 90° counter-clockwise
#         # img = img.transpose(img.ROTATE_270 if clockwise else img.ROTATE_90)

#         img = img.transpose(Image.Transpose.ROTATE_270)
#         np_img = np.array(img).copy()
#         size = 5
#         np_img[2:2+size, 2:2+size, 0] = 255 # Red channel
#         np_img[2:2+size, 2:2+size, 1:] = 0
#         np_gray_img = np_img[:, :, [1, 2, 0]]


#     else:
#         pil_gray_img = img.convert('L')
#         np_gray_img = np.array(pil_gray_img, dtype=np.uint8)
#         np_gray_img = np.dstack([np_gray_img, np_gray_img, np_gray_img])
#     return np_gray_img




import numpy as np
from PIL import Image, ImageDraw

def apply_hierarchical_noise(img, severity=0):
    """
    Applies hierarchical changes to 'minority group representation'.
    
    Args:
        img (PIL.Image): Input image.
        severity (int): 0 to 6. 
                        0 = Standard Grayscale
                        1 = Channel Swap (Subtle)
                        2 = Small Artifact (Red Square)
                        3 = Rotate + Artifact
                        4 = Gaussian Noise (Grainy)
                        5 = Noise + Line Obstruction
                        6 = Extreme (Heavy Noise + Multiple Lines + Inversion)
    """
    
    # Ensure working with a copy to avoid mutating original
    img_mod = img.copy()
    w, h = img_mod.size
    
    # --- LEVEL 0: Baseline (Standard Grayscale) ---
    if severity == 0:
        pil_gray = img_mod.convert('L')
        np_gray = np.array(pil_gray, dtype=np.uint8)
        # Stack to 3 channels to maintain shape compatibility
        return np.dstack([np_gray, np_gray, np_gray])

    # --- LEVEL 1: Channel Swapping (Less Similar) ---
    # We swap RGB to BGR or GBR to change the 'feel' before grayscaling later
    # This alters the representation without destroying features.
    np_img = np.array(img_mod)
    if severity >= 1:
        # Swap channels: R->G, G->B, B->R
        np_img = np_img[:, :, [1, 2, 0]]

    # --- LEVEL 2: Add Small Artifact (The 'Red' Square) ---
    # Based on your snippet, we introduce a specific feature marker.
    if severity >= 2:
        size = 5
        # Set a small 5x5 red square in the top-left
        # Note: Since we swapped channels in Level 1, we adjust indices accordingly
        np_img[2:2+size, 2:2+size, 0] = 255 
        np_img[2:2+size, 2:2+size, 1:] = 0

    # --- LEVEL 3: Geometry Shift (Rotation) ---
    if severity >= 3:
        # Convert back to PIL to rotate easily
        temp_pil = Image.fromarray(np_img)
        temp_pil = temp_pil.transpose(Image.Transpose.ROTATE_270)
        np_img = np.array(temp_pil)

    # --- LEVEL 4: Add Gaussian Noise ---
    if severity >= 4:
        row, col, ch = np_img.shape
        mean = 0
        sigma = 25  # Intensity of noise
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        
        # Add noise and clip to valid byte range
        np_img = np_img.astype(np.int16) + gauss.astype(np.int16)
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)

    # --- LEVEL 5: Structural Obstruction (Draw a Line) ---
    if severity >= 5:
        # Use PIL Draw for clean lines
        temp_pil = Image.fromarray(np_img)
        draw = ImageDraw.Draw(temp_pil)
        
        # Draw a diagonal line across the image
        draw.line((0, 0, w, h), fill=(255, 0, 0), width=3)
        np_img = np.array(temp_pil)

    # --- LEVEL 6: Extreme Corruption ---
    if severity >= 6:
        # Add a second inverse line
        temp_pil = Image.fromarray(np_img)
        draw = ImageDraw.Draw(temp_pil)
        draw.line((0, h, w, 0), fill=(0, 255, 0), width=3)
        np_img = np.array(temp_pil)
        
        # Invert colors for maximum dissimilarity
        np_img = 255 - np_img

    # Final Output Formatting
    # Your logic requested a specific channel order at the end:
    return np_img



import numpy as np
from PIL import Image, ImageDraw

def hierarchical_corruption_orig(img, severity=0):
    """
    Applies cumulative corruptions to an image.
    Each level includes ALL modifications from previous levels.
    
    Args:
        img (PIL.Image): Input image.
        severity (int): 0-6
    """
    # Initialize: Work on a copy to prevent changing the original
    # We will toggle between PIL and Numpy as needed for different operations
    current_img = img.copy()
    
    # --- LEVEL 1: Channel Shuffle (The Base Distortion) ---
    # Swaps RGB channels to BGR (or similar). Visuals look 'off' but clear.
    if severity >= 1:
        np_arr = np.array(current_img)
        # Swap channels: Red->Green, Green->Blue, Blue->Red
        np_arr = np_arr[:, :, [1, 2, 0]] 
        current_img = Image.fromarray(np_arr)

    # --- LEVEL 2: Geometry (Rotation) ---
    # Stacks ON TOP of Level 1. Now it is Swapped + Rotated.
    if severity >= 2:
        current_img = current_img.transpose(Image.Transpose.ROTATE_270)

    # --- LEVEL 3: Small Artifact (The "Red" Patch) ---
    # Stacks ON TOP of Level 1 & 2.
    if severity >= 3:
        np_arr = np.array(current_img)
        size = 5
        # Add the 5x5 red square artifact (per your original snippet)
        np_arr[2:2+size, 2:2+size, 0] = 255 
        np_arr[2:2+size, 2:2+size, 1:] = 0
        current_img = Image.fromarray(np_arr)

    # --- LEVEL 4: Global Noise (Grain) ---
    # Stacks ON TOP of 1, 2, & 3.
    if severity >= 4:
        np_arr = np.array(current_img)
        row, col, ch = np_arr.shape
        mean = 0
        sigma = 30 # Adjust noise intensity here
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        
        # Add noise, ensuring we stay within 0-255 uint8 limits
        noisy = np_arr.astype(np.int16) + gauss.astype(np.int16)
        np_arr = np.clip(noisy, 0, 255).astype(np.uint8)
        current_img = Image.fromarray(np_arr)

    # --- LEVEL 5: Structural Obstruction (Line Draw) ---
    # Stacks ON TOP of 1, 2, 3, & 4.
    if severity >= 5:
        draw = ImageDraw.Draw(current_img)
        w, h = current_img.size
        # Draw a thick red line across the image
        draw.line((0, 0, w, h), fill=(255, 0, 0), width=5)

    # --- LEVEL 6: Extreme (Inversion/Negative) ---
    # Stacks ON TOP of everything. The Final Extreme.
    if severity >= 6:
        np_arr = np.array(current_img)
        # Invert the colors (255 - pixel_value)
        np_arr = 255 - np_arr 
        current_img = Image.fromarray(np_arr)

    # Final Return: Convert back to Numpy as requested by your original return type
    return np.array(current_img)


def hierarchical_corruption(img, severity=0):
    """
    Applies cumulative corruptions to an image.
    Each level includes ALL modifications from previous levels.
    
    Args:
        img (PIL.Image): Input image.
        severity (int): 0-6
    """
    # Initialize: Work on a copy to prevent changing the original
    # We will toggle between PIL and Numpy as needed for different operations
    current_img = img.copy()
    
    # --- LEVEL 1: Channel Shuffle (The Base Distortion) ---
    # Swaps RGB channels to BGR (or similar). Visuals look 'off' but clear.
    if severity >= 1:
        np_arr = np.array(current_img)
        # Swap channels: Red->Green, Green->Blue, Blue->Red
        np_arr = np_arr[:, :, [1, 2, 0]] 
        current_img = Image.fromarray(np_arr)

    # --- LEVEL 2: Geometry (Rotation) ---
    # Stacks ON TOP of Level 1. Now it is Swapped + Rotated.
    if severity >= 2:
        draw = ImageDraw.Draw(current_img)
        w, h = current_img.size
        # Draw a thick red line across the image
        draw.line((0, 0, w, h), fill=(255, 0, 0), width=2)
        

    # --- LEVEL 3: Small Artifact (The "Red" Patch) ---
    # Stacks ON TOP of Level 1 & 2.
    if severity >= 3:
        np_arr = np.array(current_img)
        size = 5
        # Add the 5x5 red square artifact (per your original snippet)
        np_arr[2:2+size, 2:2+size, 0] = 255 
        np_arr[2:2+size, 2:2+size, 1:] = 0
        current_img = Image.fromarray(np_arr)

    # --- LEVEL 4: Global Noise (Grain) ---
    # Stacks ON TOP of 1, 2, & 3.
    if severity >= 4:
        np_arr = np.array(current_img)
        row, col, ch = np_arr.shape
        mean = 2.5
        sigma = 50 # Adjust noise intensity here
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        
        # Add noise, ensuring we stay within 0-255 uint8 limits
        noisy = np_arr.astype(np.int16) + gauss.astype(np.int16)
        np_arr = np.clip(noisy, 0, 255).astype(np.uint8)
        current_img = Image.fromarray(np_arr)

        draw = ImageDraw.Draw(current_img)
        w, h = current_img.size
        draw.line((0, h, w, 0), fill=(0, 255, 0), width=3)

    # --- LEVEL 5: Structural Obstruction (Line Draw) ---
    # Stacks ON TOP of 1, 2, 3, & 4.
    if severity >= 5:
        current_img = current_img.transpose(Image.Transpose.ROTATE_270)
        np_arr = np.array(current_img)
        # Invert the colors (255 - pixel_value)
        np_arr = 255 - np_arr 
        current_img = Image.fromarray(np_arr)
        np_arr = np.array(current_img)
        row, col, ch = np_arr.shape
        mean = 2
        sigma = 20 # Adjust noise intensity here
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        
        # Add noise, ensuring we stay within 0-255 uint8 limits
        noisy = np_arr.astype(np.int16) + gauss.astype(np.int16)
        np_arr = np.clip(noisy, 0, 255).astype(np.uint8)
        current_img = Image.fromarray(np_arr)


    # --- LEVEL 6: Extreme (Inversion/Negative) ---
    # # Stacks ON TOP of everything. The Final Extreme.
    # if severity >= 6:
    #     np_arr = np.array(current_img)
    #     # Invert the colors (255 - pixel_value)
    #     np_arr = 255 - np_arr 
    #     current_img = Image.fromarray(np_arr)

    # Final Return: Convert back to Numpy as requested by your original return type
    return np.array(current_img)



def rgb_to_grayscale(img, rotate_90=False, severity = 0):
    """Convert image to gray scale, optionally rotate 90 degrees."""
    if rotate_90:
        # PIL: ROTATE_270 = 90° clockwise, ROTATE_90 = 90° counter-clockwise
        # img = img.transpose(img.ROTATE_270 if clockwise else img.ROTATE_90)

        # img = img.transpose(Image.Transpose.ROTATE_270)
        # np_img = np.array(img).copy()
        # size = 5
        # np_img[2:2+size, 2:2+size, 0] = 255 # Red channel
        # np_img[2:2+size, 2:2+size, 1:] = 0
        # np_gray_img = np_img[:, :, [1, 2, 0]]

        np_img = hierarchical_corruption(img, severity=severity)
        np_gray_img = np.array(np_img).copy()




    else:
        pil_gray_img = img.convert('L')
        np_gray_img = np.array(pil_gray_img, dtype=np.uint8)
        np_gray_img = np.dstack([np_gray_img, np_gray_img, np_gray_img])
    return np_gray_img




class CIFAR_10S(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None,
                 seed=0, skewed_ratio=0.8,severity =0, labelwise=False):
        super(CIFAR_10S, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split
        self.seed = seed

        self.num_classes = 10
        self.num_groups = 2

        imgs, labels, colors, data_count = self._make_skewed(root, split, seed, skewed_ratio, num_classes = self.num_classes, severity = severity)

        self.dataset = {}
        self.dataset['image'] = np.array(imgs)
        self.dataset['label'] = np.array(labels)
        self.dataset['color'] = np.array(colors)
        mean = tuple(np.mean(self.dataset['image'] / 255., axis=(0, 1, 2)))
        std = tuple(np.std(self.dataset['image'] / 255., axis=(0, 1, 2)))
        # print(mean,std)
        self._get_label_list()
        self.labelwise = labelwise

        self.num_data = data_count

        if self.labelwise:
            self.idx_map = self._make_idx_map()

    def _make_idx_map(self):
        idx_map = [[] for i in range(self.num_groups * self.num_classes)]
        for j, i in enumerate(self.dataset['image']):
            y = self.dataset['label'][j]
            s = self.dataset['color'][j]
            pos = s * self.num_classes + y
            idx_map[int(pos)].append(j)
        final_map = []
        for l in idx_map:
            final_map.extend(l)
        return final_map

    def _get_label_list(self):
        self.label_list = []
        for i in range(self.num_classes):
            self.label_list.append(sum(self.dataset['label'] == i))

    def _set_mapping(self):
        tmp = [[] for _ in range(self.num_classes)]
        for i in range(self.__len__()):
            tmp[int(self.dataset['label'][i])].append(i)
        self.map = []
        for i in range(len(tmp)):
            self.map.extend(tmp[i])

    def __len__(self):
        return len(self.dataset['image'])

    def __getitem__(self, index):
        if self.labelwise:
            index = self.idx_map[index]
        image = self.dataset['image'][index]
        label = self.dataset['label'][index]
        color = self.dataset['color'][index]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        # return image, 0, np.float32(color), np.int64(label), (index, 0)

        return image, label.astype(int), color.astype(int), (index, 0)
        # return image, color.astype(int), color.astype(int), (index, 0)
        # return image, np.int64(label), np.float32(color), (index, 0)

    def _make_skewed(self, data_path, split='train', seed=0, skewed_ratio=1.,severity =0, num_classes=10):

        train = False if split =='test' else True
        cifardata = CIFAR10(data_path, train=train, shuffle=True, seed=seed, download=True)

        num_data = 50000 if split =='train' else 20000

        imgs = np.zeros((num_data, 32, 32, 3), dtype=np.uint8)
        labels = np.zeros(num_data)
        colors = np.zeros(num_data)
        data_count = np.zeros((2, 10), dtype=int)

        num_total_train_data = int((50000 // num_classes))
        num_skewed_train_data = int((50000 * skewed_ratio) // num_classes)
        # print("skewed_ratio",skewed_ratio)
        for i, data in enumerate(cifardata):
            img, target = data

            if split == 'test':
                if skewed_ratio != 0.9:
                    imgs[i] = rgb_to_grayscale(img, rotate_90 = True, severity = severity)
                else:
                    imgs[i] = rgb_to_grayscale(img)
                imgs[i+10000] = np.array(img)
                labels[i] = target
                labels[i+10000] = target
                colors[i] = 0
                colors[i+10000] = 1
                data_count[0, target] += 1
                data_count[1, target] += 1
            else:
                if target < 5:
                    if data_count[0, target] < (num_skewed_train_data):
                        if skewed_ratio != 0.9:
                            imgs[i] = rgb_to_grayscale(img, rotate_90 = True, severity = severity)
                        else:
                            imgs[i] = rgb_to_grayscale(img)
                        colors[i] = 0
                        data_count[0, target] += 1
                    else:
                        imgs[i] = np.array(img)
                        colors[i] = 1
                        data_count[1, target] += 1
                    labels[i] = target
                else:
                    if data_count[0, target] < (num_total_train_data - num_skewed_train_data):
                        if skewed_ratio != 0.9:
                            imgs[i] = rgb_to_grayscale(img, rotate_90 = True, severity = severity)
                        else:
                            imgs[i] = rgb_to_grayscale(img)
                        colors[i] = 0
                        data_count[0, target] += 1
                    else:
                        imgs[i] = np.array(img)
                        colors[i] = 1
                        data_count[1, target] += 1
                    labels[i] = target

        # print('<# of Skewed data>',split)
        # print(data_count)
        # print(sum(colors))

        return imgs, labels, colors, data_count
        # return imgs, colors, colors, data_count


class CIFAR10(VisionDataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, shuffle=False, seed=0):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if shuffle:
            np.random.seed(seed)
            idx = np.arange(len(self.data), dtype=np.int64)
            np.random.shuffle(idx)
            self.data = self.data[idx]
            self.targets = np.array(self.targets)[idx]

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

 
from torchvision import transforms
def CIFAR_10s(root, skew_ratio, severity =0, seed = 0):
    mean = (0.4890520609319886, 0.48451061542917323, 0.46714946730198426)
    std = (0.24278049202200438, 0.24082296066838135, 0.25098309397306334)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    train_dataset = CIFAR_10S(root=root, split='train', transform=transform, seed=seed, skewed_ratio=skew_ratio, severity = severity)
    test_dataset = CIFAR_10S(root=root, split='test', transform=transform, seed=seed, skewed_ratio=skew_ratio, severity = severity)
    # print(len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset, mean, std

