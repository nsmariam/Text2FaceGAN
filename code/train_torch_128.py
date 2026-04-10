import os, sys, random, pickle
sys.path.insert(0, '/content/Text2FaceGAN/code')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from model_torch_128 import Generator128, Discriminator128
from dataset_torch   import FaceTextDataset

# ── Hyperparamètres (mêmes que l'entraînement 64x64) ─────────────
Z_DIM                 = 100
T_DIM                 = 256
BATCH_SIZE            = 64
IMAGE_SIZE            = 128   # Seul changement
GF_DIM                = 64
DF_DIM                = 64
CAPTION_VECTOR_LENGTH = 384
LEARNING_RATE         = 0.0002
BETA1                 = 0.5
EPOCHS                = 50
SAVE_EVERY            = 30
DATA_DIR              = '/content/Text2FaceGAN/data'
DEVICE                = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ─────────────────────────────────────────────────────────────────

def weights_init(m):
    cn = m.__class__.__name__
    if 'Conv' in cn:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in cn:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():
    print(f"Device     : {DEVICE}")
    print(f"Résolution : {IMAGE_SIZE}x{IMAGE_SIZE}")

    os.makedirs(os.path.join(DATA_DIR, 'Models_128'),  exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'samples_128'), exist_ok=True)

    # Dataset avec IMAGE_SIZE=128
    dataset = FaceTextDataset(DATA_DIR, CAPTION_VECTOR_LENGTH, IMAGE_SIZE)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)

    netG = Generator128(Z_DIM, T_DIM, GF_DIM, CAPTION_VECTOR_LENGTH).to(DEVICE)
    netD = Discriminator128(T_DIM, DF_DIM, CAPTION_VECTOR_LENGTH).to(DEVICE)
    netG.apply(weights_init)
    netD.apply(weights_init)

    n_G = sum(p.numel() for p in netG.parameters())
    n_D = sum(p.numel() for p in netD.parameters())
    print(f"Générateur     : {n_G:,} paramètres")
    print(f"Discriminateur : {n_D:,} paramètres")

    optimG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    criterion  = nn.BCELoss()
    real_label = torch.ones(BATCH_SIZE,  1).to(DEVICE)
    fake_label = torch.zeros(BATCH_SIZE, 1).to(DEVICE)

    G_losses, D_losses = [], []

    for epoch in range(EPOCHS):
        netG.train()
        netD.train()
        epoch_g, epoch_d = 0., 0.

        for batch_no, (real_imgs, wrong_imgs, captions) in enumerate(
                tqdm(loader, desc=f"Epoch {epoch:03d}")):

            real_imgs  = real_imgs.to(DEVICE)
            wrong_imgs = wrong_imgs.to(DEVICE)
            captions   = captions.to(DEVICE)
            z_noise    = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)

            # Generator update x2
            for _ in range(2):
                optimG.zero_grad()
                fake_imgs    = netG(z_noise, captions)
                d_fake, _    = netD(fake_imgs, captions)
                g_loss       = criterion(d_fake, real_label)
                g_loss.backward()
                optimG.step()

            # Discriminator update
            if batch_no % 4 == 0:
                real_in = fake_imgs.detach()
            else:
                real_in = real_imgs

            optimD.zero_grad()

            d_real, _  = netD(real_in, captions)
            d_loss1    = criterion(d_real, real_label)

            d_wrong, _ = netD(wrong_imgs, captions)
            d_loss2    = criterion(d_wrong, fake_label)

            z2         = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
            fake2      = netG(z2, captions).detach()
            d_fake2, _ = netD(fake2, captions)
            d_loss3    = criterion(d_fake2, fake_label)

            d_loss = d_loss1 + d_loss2 + d_loss3
            d_loss.backward()
            optimD.step()

            epoch_g += g_loss.item()
            epoch_d += d_loss.item()

            if batch_no % SAVE_EVERY == 0 and batch_no > 0:
                print(f"\n  d1:{d_loss1:.3f} d2:{d_loss2:.3f} "
                      f"d3:{d_loss3:.3f} | D:{d_loss:.3f} G:{g_loss:.3f}")
                save_image(
                    fake_imgs[:16],
                    os.path.join(DATA_DIR, 'samples_128',
                                 f'epoch{epoch:03d}_batch{batch_no:04d}.png'),
                    nrow=4
                )

        if epoch % 5 == 0:
            torch.save({
                'epoch'  : epoch,
                'netG'   : netG.state_dict(),
                'netD'   : netD.state_dict(),
                'optimG' : optimG.state_dict(),
                'optimD' : optimD.state_dict(),
            }, os.path.join('/content/drive/MyDrive/text2facegan_checkpoints_128',
                            f'checkpoint_epoch{epoch:03d}.pt'))
            print(f"Checkpoint sauvegardé — epoch {epoch}")

        avg_g = epoch_g / len(loader)
        avg_d = epoch_d / len(loader)
        G_losses.append(avg_g)
        D_losses.append(avg_d)
        print(f"Epoch {epoch:3d} | G: {avg_g:.4f} | D: {avg_d:.4f}")

    with open(os.path.join(DATA_DIR, 'losses_128.pkl'), 'wb') as f:
        pickle.dump({'G': G_losses, 'D': D_losses}, f)
    print("Entraînement 128x128 terminé.")

if __name__ == '__main__':
    main()
