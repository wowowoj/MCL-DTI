import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')


def _init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 act_layer=nn.GELU,
                 drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,  # 生成qkv 是否使用偏置
                 drop_ratio=0.,
                 ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Cross_Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 drop_ratio=0.,
                 ):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_ratio)

    def forward(self, x, y):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv2[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.
                 ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop_ratio
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Decoder(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.
                 ):
        super(Decoder, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, drop_ratio=drop_ratio)
        self.cross_attn = Cross_Attention(dim, drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop_ratio
        )

    def forward(self, x, y):
        y = y + self.drop_path(self.attn(self.norm1(y)))
        out = y + self.drop_path(self.cross_attn(x, self.norm1(y)))
        out = out + self.drop_path(self.mlp(self.norm2(y)))
        return out


class MCLDTI(nn.Module):
    def __init__(self,
                 depth_e1=4,
                 depth_e2=4,
                 depth_decoder=4,
                 embed_dim=256,
                 protein_dim=256,
                 drop_ratio=0.,
                 backbone="",
                 protein_len=20,
                 ):
        super(MCLDTI, self).__init__()

        self.depth_decoder = depth_decoder
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.rate1 = torch.nn.Parameter(torch.rand(1))
        self.rate2 = torch.nn.Parameter(torch.rand(1))

        # if backbone == "ResNet18":
        #     self.img_backbone = ResNet(BasicBlock, [2, 2, 2, 2], img_dim=img_dim, include_top=True)
        # elif backbone == "ResNet34":
        #     self.img_backbone = ResNet(BasicBlock, [3, 4, 6, 3], img_dim=img_dim, include_top=True)
        # elif
        if backbone == "CNN":
            self.img_backbone = nn.Sequential(  # 3*256*256
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),  # 64*128*128
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.02, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 64*64*64

                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1),  # 128*32*32
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.02, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 128*16*16
            )

        #  encoder 1
        self.norm_e1 = norm_layer(embed_dim)
        self.pos_drop_e1 = nn.Dropout(p=drop_ratio)
        self.pos_embed_e1 = nn.Parameter(torch.zeros(1, 256, embed_dim))
        dpr_e1 = [x.item() for x in torch.linspace(0, drop_ratio, depth_e1)]
        self.encoder_e1 = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=4,
                  drop_ratio=dpr_e1[i],
                  )
            for i in range(depth_e1)
        ])

        #  encoder 2
        self.smile2feature = nn.Linear(512, 256)
        self.embeddings_e2 = nn.Embedding(51, embed_dim)
        self.norm_e2 = norm_layer(embed_dim)
        self.pos_drop_e2 = nn.Dropout(p=drop_ratio)
        self.pos_embed_e2 = nn.Parameter(torch.zeros(1, 256, embed_dim))
        dpr_e2 = [x.item() for x in torch.linspace(0, drop_ratio, depth_e2)]
        self.encoder_e2 = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=4,
                  drop_ratio=dpr_e2[i],
                  )
            for i in range(depth_e2)
        ])

        # decoder
        # ----------------------------------protein length   embed_dim=protein_len+1  -------------------------------------------
        self.embeddings_decoder = nn.Embedding(protein_len, embed_dim)
        # decoder
        self.norm_decoder = norm_layer(embed_dim)
        self.pos_drop_decoder = nn.Dropout(p=drop_ratio)
        self.pos_embed_decoder = nn.Parameter(torch.zeros(1, protein_dim, embed_dim))
        self.decoder = Decoder(dim=embed_dim, mlp_ratio=4, drop_ratio=0., )

        # decoder2
        self.norm_decoder2 = norm_layer(embed_dim)
        self.pos_drop_decoder2 = nn.Dropout(p=drop_ratio)
        self.pos_embed_decoder2 = nn.Parameter(torch.zeros(1, protein_dim, embed_dim))
        self.decoder2 = Decoder(dim=embed_dim, mlp_ratio=4, drop_ratio=0., )

        self.decoder_1d_2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=4),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=4),
            nn.BatchNorm1d(64),
            nn.ELU(),

            nn.AdaptiveMaxPool1d(1),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*64*64

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc = nn.Linear(992, 2)

    def forward_features_e1(self, x):
        B, _, _, _ = x.shape
        x = self.img_backbone(x)
        x = x.reshape(B, 256, -1)
        x = self.pos_drop_e1(x + self.pos_embed_e1)
        x = self.encoder_e1(x)
        x = self.norm_e1(x)
        return x

    def forward_features_e2(self, x):
        x = self.embeddings_e2(x)
        x = x.permute(0, 2, 1)
        x = self.smile2feature(x)
        x = x.permute(0, 2, 1)
        x = self.pos_drop_e2(x + self.pos_embed_e2)
        x = self.encoder_e2(x)
        x = self.norm_e2(x)
        return x

    def forward_features_decoder(self, x, y):
        B, _, _ = x.shape
        y = self.embeddings_decoder(y)
        y = self.pos_drop_decoder(y + self.pos_embed_decoder)

        # cross1
        out1 = self.decoder(x, y)
        for i in range(self.depth_decoder - 1):
            out1 = self.decoder(x, out1)
        out1 = self.norm_decoder(out1)
        out1 = out1.reshape(B, 1, 256, -1)

        # cross2
        out2 = self.decoder2(y, x)
        for i in range(self.depth_decoder - 1):
            out2 = self.decoder2(y, out2)
        out2 = self.norm_decoder2(out2)
        out2 = out2.reshape(B, 1, 256, -1)
        out = torch.cat((out1, out2), 1)

        return out

    def forward(self, inputs):
        image, smile, protein = inputs[0], inputs[1], inputs[2]
        image = image.to(device)
        smile = smile.to(device)
        protein = protein.to(device)
        B, _, _, _ = image.shape
        image_feature = self.forward_features_e1(image)
        smile_feature = self.forward_features_e2(smile)
        encoder_feature = self.rate1 * image_feature + self.rate2 * smile_feature
        decoder_feature = self.forward_features_decoder(encoder_feature, protein)
        decoder_feature = self.conv2d(decoder_feature)
        decoder_feature = decoder_feature.reshape(B, 256, -1)
        decoder_feature = self.conv1d(decoder_feature)
        decoder_out = decoder_feature.reshape(B, -1)
        out = self.fc(decoder_out)
        return out, self.rate1, self.rate2

    def __call__(self, data, train=True):
        inputs, correct_interaction, = data[:-1], data[-1]
        predicted_interaction, rate1, rate2 = self.forward(inputs)
        correct_interaction = torch.squeeze(correct_interaction)
        loss = F.cross_entropy(predicted_interaction, correct_interaction.to(device))
        correct_labels = correct_interaction.to('cpu').data.numpy()
        ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
        predicted_labels = list(map(lambda x: np.argmax(x), ys))
        predicted_scores = list(map(lambda x: x[1], ys))
        return loss, correct_labels, predicted_labels, predicted_scores, rate1, rate2
