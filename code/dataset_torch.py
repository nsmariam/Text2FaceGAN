import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import skimage.io
import skimage.transform


class FaceTextDataset(Dataset):
    """
    Charge les images CelebA et les vecteurs de captions.
    Reproduit la logique de get_training_batch() de l'auteur.
    """
    def __init__(self, data_dir, caption_vector_length=384, image_size=64):
        self.data_dir              = data_dir
        self.image_size            = image_size
        self.caption_vector_length = caption_vector_length

        with open(os.path.join(data_dir, 'train_encoding'), 'rb') as f:
            self.captions = pickle.load(f)

        self.image_list = list(self.captions.keys())
        random.shuffle(self.image_list)
        print(f"Dataset : {len(self.image_list)} images chargées")

    def __len__(self):
        return len(self.image_list)

    def load_image(self, img_name):
        path = os.path.join(self.data_dir, 'face/jpg', img_name)
        img  = skimage.io.imread(path)

        # Gérer les images en niveaux de gris
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)

        # Resize + normalisation dans [0,1] comme image_processing.py de l'auteur
        img = skimage.transform.resize(
            img, (self.image_size, self.image_size),
            anti_aliasing=True
        ).astype('float32')

        # Flip horizontal aléatoire (repris de image_processing.py)
        if random.random() > 0.5:
            img = np.fliplr(img).copy()

        # HWC -> CHW pour PyTorch
        return torch.tensor(img).permute(2, 0, 1)

    def __getitem__(self, idx):
        img_name   = self.image_list[idx]
        real_image = self.load_image(img_name)

        # Caption correspondante
        caption = torch.tensor(
            self.captions[img_name][0][:self.caption_vector_length],
            dtype=torch.float32
        )

        # Wrong image : image réelle avec mauvaise caption (GAN-CLS)
        wrong_idx   = random.randint(0, len(self.image_list) - 1)
        wrong_name  = self.image_list[wrong_idx]
        wrong_image = self.load_image(wrong_name)

        return real_image, wrong_image, caption
