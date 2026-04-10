import torch
import torch.nn as nn


class Generator128(nn.Module):
    """
    Générateur 128x128 — même architecture que l'original
    mais avec une déconvolution supplémentaire.
    4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
    """
    def __init__(self, z_dim=100, t_dim=256, gf_dim=64,
                 caption_vector_length=384):
        super().__init__()
        self.z_dim  = z_dim
        self.gf_dim = gf_dim

        self.text_encoder = nn.Sequential(
            nn.Linear(caption_vector_length, t_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc  = nn.Linear(z_dim + t_dim, gf_dim * 8 * 4 * 4)
        self.bn0 = nn.BatchNorm2d(gf_dim * 8)

        # 4->8->16->32->64->128 (5 déconvolutions au lieu de 4)
        self.deconv1 = nn.ConvTranspose2d(gf_dim*8, gf_dim*8, 4, 2, 1, bias=False)
        self.bn1     = nn.BatchNorm2d(gf_dim * 8)

        self.deconv2 = nn.ConvTranspose2d(gf_dim*8, gf_dim*4, 4, 2, 1, bias=False)
        self.bn2     = nn.BatchNorm2d(gf_dim * 4)

        self.deconv3 = nn.ConvTranspose2d(gf_dim*4, gf_dim*2, 4, 2, 1, bias=False)
        self.bn3     = nn.BatchNorm2d(gf_dim * 2)

        self.deconv4 = nn.ConvTranspose2d(gf_dim*2, gf_dim*1, 4, 2, 1, bias=False)
        self.bn4     = nn.BatchNorm2d(gf_dim * 1)

        self.deconv5 = nn.ConvTranspose2d(gf_dim*1, 3, 4, 2, 1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, z, text_emb):
        t   = self.text_encoder(text_emb)
        inp = torch.cat([z, t], dim=1)
        h   = self.fc(inp)
        h   = h.view(h.size(0), self.gf_dim*8, 4, 4)
        h   = self.relu(self.bn0(h))

        h = self.relu(self.bn1(self.deconv1(h)))   # (B, 512, 8,   8)
        h = self.relu(self.bn2(self.deconv2(h)))   # (B, 256, 16,  16)
        h = self.relu(self.bn3(self.deconv3(h)))   # (B, 128, 32,  32)
        h = self.relu(self.bn4(self.deconv4(h)))   # (B,  64, 64,  64)
        h = self.deconv5(h)                        # (B,   3, 128, 128)

        return self.tanh(h) / 2. + 0.5


class Discriminator128(nn.Module):
    """
    Discriminateur 128x128 — une convolution supplémentaire
    par rapport au discriminateur 64x64.
    """
    def __init__(self, t_dim=256, df_dim=64,
                 caption_vector_length=384):
        super().__init__()
        self.t_dim  = t_dim
        self.df_dim = df_dim

        # 128->64->32->16->8->4 (5 convolutions)
        self.conv0 = nn.Conv2d(3,        df_dim*1, 5, 2, 2, bias=False)

        self.conv1 = nn.Conv2d(df_dim*1, df_dim*2, 5, 2, 2, bias=False)
        self.bn1   = nn.BatchNorm2d(df_dim * 2)

        self.conv2 = nn.Conv2d(df_dim*2, df_dim*4, 5, 2, 2, bias=False)
        self.bn2   = nn.BatchNorm2d(df_dim * 4)

        self.conv3 = nn.Conv2d(df_dim*4, df_dim*8, 5, 2, 2, bias=False)
        self.bn3   = nn.BatchNorm2d(df_dim * 8)

        self.conv4 = nn.Conv2d(df_dim*8, df_dim*8, 5, 2, 2, bias=False)
        self.bn4   = nn.BatchNorm2d(df_dim * 8)

        self.text_encoder = nn.Sequential(
            nn.Linear(caption_vector_length, t_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5  = nn.Conv2d(df_dim*8 + t_dim, df_dim*8, 1, 1, 0, bias=False)
        self.bn5    = nn.BatchNorm2d(df_dim * 8)

        self.fc_out  = nn.Linear(df_dim * 8 * 4 * 4, 1)
        self.lrelu   = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, text_emb):
        h = self.lrelu(self.conv0(image))             # (B,  64, 64, 64)
        h = self.lrelu(self.bn1(self.conv1(h)))       # (B, 128, 32, 32)
        h = self.lrelu(self.bn2(self.conv2(h)))       # (B, 256, 16, 16)
        h = self.lrelu(self.bn3(self.conv3(h)))       # (B, 512,  8,  8)
        h = self.lrelu(self.bn4(self.conv4(h)))       # (B, 512,  4,  4)

        t = self.text_encoder(text_emb)
        t = t.unsqueeze(2).unsqueeze(3).expand(-1, -1, 4, 4)

        h = torch.cat([h, t], dim=1)
        h = self.lrelu(self.bn5(self.conv5(h)))       # (B, 512, 4, 4)

        h   = h.view(h.size(0), -1)
        out = self.fc_out(h)

        return self.sigmoid(out), out
