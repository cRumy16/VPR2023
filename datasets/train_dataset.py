import torch
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tfm
from collections import defaultdict
import PIL
from PIL import ImageOps, ImageFilter

default_transform = tfm.Compose([
    tfm.ToTensor(),
    #tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        

class TrainDataset(Dataset):
    def __init__(
        self,
        dataset_folder,
        img_per_place=4,
        min_img_per_place=4,
        transform=default_transform,
        self_supervised=False
    ):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
        if len(self.images_paths) == 0:
            raise FileNotFoundError(f"There are no images under {dataset_folder} , you should change this path")
        self.dict_place_paths = defaultdict(list)
        for image_path in self.images_paths:
            place_id = image_path.split("@")[-2]
            self.dict_place_paths[place_id].append(image_path)

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.transform = transform
        
     
        # self.transformer_aug = tfm.Compose([tfm.RandomResizedCrop(224, interpolation=tfm.InterpolationMode.BICUBIC),
        #         tfm.RandomHorizontalFlip(p=0.5),
        #         tfm.RandomApply([tfm.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8,),
        #         tfm.RandomGrayscale(p=0.2),
        #         GaussianBlur(p=1.0),
        #         Solarization(p=0.0),
        #          tfm.ToTensor()])
        
        
        self.transformer_aug = tfm.Compose([
                    # tfm.RandomHorizontalFlip(p = 0.7),
                #   tfm.RandomCrop((150, 150)),
                   #tfm.ColorJitter(brightness = (0.1,0.9)) ,
                    #tfm.RandomGrayscale(),
                    #tfm.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    # tfm.RandomApply(transforms=[
                    #                             #tfm.RandomAffine(30, translate=(0.2,0.2), scale=None, shear=None, interpolation=tfm.InterpolationMode.NEAREST, fill=0, center=None),
                    #                             # tfm.RandomEqualize(p=0.6),
                    #                            # tfm.RandomPerspective(p=0.8, distortion_scale=0.6),
                    #                            # tfm.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    #                             
                    #                             ],
                    #                 p=1),
                   
                    # tfm.RandomApply(transforms=[
                    #     #tfm.RandomHorizontalFlip(p = 0.7),
                    #     tfm.ColorJitter(brightness = (0.1,0.9)) ,
                    #     tfm.RandomCrop(size=224),
                    # ], p=0.5),
                    tfm.RandomHorizontalFlip(p = 1),
                    tfm.RandomPerspective(p=0.5),
                    tfm.RandomCrop(size=224),
                    tfm.ToTensor(),
                    tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        self.self_supervised = self_supervised

        # keep only places depicted by at least min_img_per_place images
        for place_id in list(self.dict_place_paths.keys()):
            all_paths_from_place_id = self.dict_place_paths[place_id]
            if len(all_paths_from_place_id) < min_img_per_place:
                del self.dict_place_paths[place_id]
        self.places_ids = sorted(list(self.dict_place_paths.keys()))
        self.total_num_images = sum([len(paths) for paths in self.dict_place_paths.values()])

    def __getitem__(self, index):
        place_id = self.places_ids[index]
        all_paths_from_place_id = self.dict_place_paths[place_id]
        chosen_paths = np.random.choice(all_paths_from_place_id, self.img_per_place)
        images = [Image.open(path).convert('RGB') for path in chosen_paths]
       
        tfm_images = [self.transform(img) for img in images]
        
        
        if self.self_supervised:
            image = images[0::2]
            images_aug = images[1::2]
            image = [self.transform(img) for img in image]
            images_aug = [self.transformer_aug(img) for img in images_aug]
            #image = [image for _ in range(3)]

            return torch.stack(image), torch.stack(images_aug), torch.tensor(index).repeat(2)
           
            #print(f"\niamge: {image.shape}, aug: {images_aug[0].shape}, {len(images_aug)}\n")
            
           
            
            new_images = image + images_aug 
            return torch.stack(new_images), torch.tensor(index).repeat(self.img_per_place)
           
            # tfm_images = [self.transform(img) for img in images]
            # augImages= [self.transformer_aug(img) for img in images]

            # final_images= [img for pair in zip(tfm_images, augImages) for img in pair]
            
            # return torch.stack(final_images), torch.tensor(index).repeat(2* self.img_per_place)
        
        return torch.stack(tfm_images), torch.tensor(index).repeat(self.img_per_place)
    

    def __len__(self):
        """Denotes the total number of places (not images)"""
        return len(self.places_ids)
