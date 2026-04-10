import sys
sys.path.insert(0, '/content/text2facegan/code')

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sentence_transformers import SentenceTransformer

from model_torch import Generator

# ── Paramètres ───────────────────────────────────────────────────
Z_DIM                 = 100
T_DIM                 = 256
GF_DIM                = 64
CAPTION_VECTOR_LENGTH = 384
DATA_DIR              = '/content/text2facegan/data'
CHECKPOINT_PATH       = '/content/drive/MyDrive/text2facegan_checkpoints/checkpoint_epoch045.pt'
DEVICE                = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ─────────────────────────────────────────────────────────────────


def load_generator(checkpoint_path):
    netG = Generator(Z_DIM, T_DIM, GF_DIM, CAPTION_VECTOR_LENGTH).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    netG.load_state_dict(ckpt['netG'])
    netG.eval()
    print(f"Générateur chargé : époque {ckpt['epoch']}")
    return netG


def generate_from_description(description, netG, encoder, n_images=8):
    """
    Génère n_images visages à partir d'une description textuelle.
    Démontre la multimodalité : même texte -> visages différents.
    """
    # Encodage de la description
    emb = encoder.encode([description])                          # (1, 384)
    emb = torch.tensor(emb, dtype=torch.float32)
    emb = emb.repeat(n_images, 1).to(DEVICE)                    # (n_images, 384)

    with torch.no_grad():
        z    = torch.randn(n_images, Z_DIM, device=DEVICE)
        imgs = netG(z, emb)                                      # (n_images, 3, 64, 64)

    # Affichage
    fig, axes = plt.subplots(1, n_images, figsize=(2.5 * n_images, 3))
    for i, ax in enumerate(axes):
        img = imgs[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')

    plt.suptitle(f'"{description}"', fontsize=9, wrap=True)
    plt.tight_layout()
    plt.savefig(f'{DATA_DIR}/samples/generated_from_text.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Image sauvegardée.")

    return imgs


def main():
    print(f"Device : {DEVICE}")

    # Chargement
    netG    = load_generator(CHECKPOINT_PATH)
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # Descriptions de test — reprises du papier
    descriptions = [
        "The woman has high cheekbones. She has straight hair which is brown in colour. She appears smiling and attractive.",
        "He sports a goatee and mustache. He has wavy hair which is black in colour. He appears young.",
        "She has an oval face. She has blonde hair with bangs. She is wearing earrings and lipstick.",
        "He is bald. He has bushy eyebrows. He appears young and attractive.",
    ]

    for desc in descriptions:
        print(f"\nDescription : {desc}")
        generate_from_description(desc, netG, encoder, n_images=6)


if __name__ == '__main__':
    main()
