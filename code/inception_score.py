import sys
sys.path.insert(0, '/content/text2facegan/code')

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from torchvision import transforms
from scipy.stats import entropy
import pickle
from tqdm import tqdm

from model_torch   import Generator
from dataset_torch import FaceTextDataset

# ── Paramètres ───────────────────────────────────────────────────
Z_DIM                 = 100
T_DIM                 = 256
GF_DIM                = 64
CAPTION_VECTOR_LENGTH = 384
DATA_DIR              = '/content/text2facegan/data'
CHECKPOINT_PATH       = '/content/drive/MyDrive/text2facegan_checkpoints/checkpoint_epoch045.pt'
N_SAMPLES             = 5000   # Le papier utilise 50 000, on réduit pour la rapidité
BATCH_SIZE            = 64
N_SPLITS              = 5      # Nombre d'itérations pour calculer moyenne ± std
DEVICE                = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ─────────────────────────────────────────────────────────────────


def load_generator(checkpoint_path):
    netG = Generator(Z_DIM, T_DIM, GF_DIM, CAPTION_VECTOR_LENGTH).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    netG.load_state_dict(ckpt['netG'])
    netG.eval()
    print(f"Générateur chargé depuis : {checkpoint_path}")
    return netG


def get_inception_model():
    """Charge InceptionV3 préentraîné sur ImageNet."""
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()  # On garde les features avant la couche finale
    # On remet la couche de classification pour avoir p(y|x)
    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    return model.to(DEVICE)


def generate_images(netG, text_embeddings, n_samples, batch_size):
    """Génère n_samples images à partir d'embeddings textuels aléatoires."""
    images = []
    n_text = len(text_embeddings)

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Génération"):
            bs = min(batch_size, n_samples - i)
            z  = torch.randn(bs, Z_DIM, device=DEVICE)

            # Sélection aléatoire d'embeddings textuels
            idx = torch.randint(0, n_text, (bs,))
            t   = text_embeddings[idx].to(DEVICE)

            fake = netG(z, t)  # (bs, 3, 64, 64) dans [0, 1]
            images.append(fake.cpu())

    return torch.cat(images, dim=0)  # (N, 3, 64, 64)


def get_predictions(images, inception_model, batch_size=64):
    """
    Calcule p(y|x) pour chaque image avec InceptionV3.
    Retourne un array (N, 1000).
    """
    # Resize 64x64 -> 299x299 pour InceptionV3
    resize = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="InceptionV3"):
            batch = images[i:i+batch_size].to(DEVICE)
            batch = torch.stack([resize(img) for img in batch])
            pred  = torch.nn.functional.softmax(inception_model(batch), dim=1)
            preds.append(pred.cpu().numpy())

    return np.concatenate(preds, axis=0)  # (N, 1000)


def compute_inception_score(preds, n_splits=5):
    """
    Calcule l'Inception Score à partir des prédictions.
    IS = exp(E_x[ KL(p(y|x) || p(y)) ])
    Retourne moyenne et écart-type sur n_splits.
    """
    scores = []
    n      = preds.shape[0]
    split  = n // n_splits

    for i in range(n_splits):
        part  = preds[i * split : (i + 1) * split]
        p_y   = np.mean(part, axis=0)           # Distribution marginale p(y)
        kl    = [entropy(p_yx, p_y) for p_yx in part]  # KL(p(y|x) || p(y))
        score = np.exp(np.mean(kl))
        scores.append(score)

    return np.mean(scores), np.std(scores)


def main():
    print(f"Device : {DEVICE}")

    # Chargement du générateur
    netG = load_generator(CHECKPOINT_PATH)

    # Chargement des embeddings textuels
    with open(f'{DATA_DIR}/train_encoding', 'rb') as f:
        enc = pickle.load(f)
    text_embeddings = torch.tensor(
        np.array([v[0] for v in enc.values()]),
        dtype=torch.float32
    )
    print(f"Embeddings chargés : {text_embeddings.shape}")

    # Chargement InceptionV3
    print("Chargement InceptionV3...")
    inception = get_inception_model()

    # Génération des images
    print(f"Génération de {N_SAMPLES} images...")
    images = generate_images(netG, text_embeddings, N_SAMPLES, BATCH_SIZE)
    print(f"Images générées : {images.shape}")

    # Prédictions InceptionV3
    preds = get_predictions(images, inception, BATCH_SIZE)
    print(f"Prédictions : {preds.shape}")

    # Calcul IS
    mean, std = compute_inception_score(preds, N_SPLITS)
    print(f"\n{'='*40}")
    print(f"Inception Score : {mean:.2f} ± {std:.2f}")
    print(f"(Papier original : 1.4 ± 0.7)")
    print(f"{'='*40}")

    # Sauvegarde des résultats
    results = {'IS_mean': mean, 'IS_std': std, 'n_samples': N_SAMPLES}
    with open(f'{DATA_DIR}/inception_score_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Résultats sauvegardés.")


if __name__ == '__main__':
    main()
