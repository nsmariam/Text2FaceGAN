import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Reproduction fidèle du générateur de l'auteur (model.py TF1).
    text_embedding -> FC(t_dim) -> LeakyReLU
    concat(z, text) -> FC(gf_dim*8*4*4) -> reshape(4,4,512)
    -> 4 deconv -> 64x64x3 -> tanh/2 + 0.5  (sortie dans [0,1])
    """
    def __init__(self, z_dim=100, t_dim=256, gf_dim=64,
                 caption_vector_length=384):
        super().__init__()
        self.z_dim  = z_dim
        self.gf_dim = gf_dim

        # g_embedding dans l'auteur
        self.text_encoder = nn.Sequential(
            nn.Linear(caption_vector_length, t_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # g_h0_lin dans l'auteur
        self.fc  = nn.Linear(z_dim + t_dim, gf_dim * 8 * 4 * 4)
        self.bn0 = nn.BatchNorm2d(gf_dim * 8)

        # g_h1 à g_h4
        self.deconv1 = nn.ConvTranspose2d(gf_dim*8, gf_dim*4, 4, 2, 1, bias=False)
        self.bn1     = nn.BatchNorm2d(gf_dim * 4)

        self.deconv2 = nn.ConvTranspose2d(gf_dim*4, gf_dim*2, 4, 2, 1, bias=False)
        self.bn2     = nn.BatchNorm2d(gf_dim * 2)

        self.deconv3 = nn.ConvTranspose2d(gf_dim*2, gf_dim*1, 4, 2, 1, bias=False)
        self.bn3     = nn.BatchNorm2d(gf_dim * 1)

        self.deconv4 = nn.ConvTranspose2d(gf_dim*1, 3, 4, 2, 1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, z, text_emb):
        t   = self.text_encoder(text_emb)             # (B, 256)
        inp = torch.cat([z, t], dim=1)                # (B, 356)
        h   = self.fc(inp)                            # (B, 8192)
        h   = h.view(h.size(0), self.gf_dim*8, 4, 4) # (B, 512, 4, 4)
        h   = self.relu(self.bn0(h))

        h = self.relu(self.bn1(self.deconv1(h)))      # (B, 256, 8,  8)
        h = self.relu(self.bn2(self.deconv2(h)))      # (B, 128, 16, 16)
        h = self.relu(self.bn3(self.deconv3(h)))      # (B,  64, 32, 32)
        h = self.deconv4(h)                           # (B,   3, 64, 64)

        # Sortie dans [0,1] — identique à l'auteur : tanh(h)/2 + 0.5
        return self.tanh(h) / 2. + 0.5


class Discriminator(nn.Module):
    """
    Reproduction fidèle du discriminateur de l'auteur (model.py TF1).
    image -> 4 conv -> (512, 4, 4)
    concat text tiled -> conv(1x1) -> FC -> sigmoid
    """
    def __init__(self, t_dim=256, df_dim=64,
                 caption_vector_length=384):
        super().__init__()
        self.t_dim  = t_dim
        self.df_dim = df_dim

        # d_h0 à d_h3
        self.conv0 = nn.Conv2d(3,        df_dim*1, 5, 2, 2, bias=False)

        self.conv1 = nn.Conv2d(df_dim*1, df_dim*2, 5, 2, 2, bias=False)
        self.bn1   = nn.BatchNorm2d(df_dim * 2)

        self.conv2 = nn.Conv2d(df_dim*2, df_dim*4, 5, 2, 2, bias=False)
        self.bn2   = nn.BatchNorm2d(df_dim * 4)

        self.conv3 = nn.Conv2d(df_dim*4, df_dim*8, 5, 2, 2, bias=False)
        self.bn3   = nn.BatchNorm2d(df_dim * 8)

        # d_embedding dans l'auteur
        self.text_encoder = nn.Sequential(
            nn.Linear(caption_vector_length, t_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # d_h3_conv_new : conv 1x1 après concat image+text
        self.conv4 = nn.Conv2d(df_dim*8 + t_dim, df_dim*8, 1, 1, 0, bias=False)
        self.bn4   = nn.BatchNorm2d(df_dim * 8)

        # d_h3_lin : sortie scalaire
        self.fc_out  = nn.Linear(df_dim * 8 * 4 * 4, 1)

        self.lrelu   = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, text_emb):
        # image : (B, 3, 64, 64)
        h = self.lrelu(self.conv0(image))              # (B,  64, 32, 32)
        h = self.lrelu(self.bn1(self.conv1(h)))        # (B, 128, 16, 16)
        h = self.lrelu(self.bn2(self.conv2(h)))        # (B, 256,  8,  8)
        h = self.lrelu(self.bn3(self.conv3(h)))        # (B, 512,  4,  4)

        # Ajout du texte sur la grille spatiale 4x4 (tiled_embeddings)
        t = self.text_encoder(text_emb)                # (B, 256)
        t = t.unsqueeze(2).unsqueeze(3).expand(-1, -1, 4, 4)  # (B, 256, 4, 4)

        h = torch.cat([h, t], dim=1)                   # (B, 768, 4, 4)
        h = self.lrelu(self.bn4(self.conv4(h)))        # (B, 512, 4, 4)

        h   = h.view(h.size(0), -1)                    # (B, 8192)
        out = self.fc_out(h)                           # (B, 1)

        return self.sigmoid(out), out
