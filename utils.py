import numpy as np
import math
import torch
import torch.utils.data
from torchvision import transforms
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, smiles, proteins, label):
        self.imgs = imgs
        self.smiles = smiles
        self.proteins = proteins
        self.label = label

        self.transformImg = transforms.Compose([
            transforms.ToTensor()])

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        img_path = self.imgs[item]
        smiles_feature = self.smiles[item]
        pro_feature = self.proteins[item]
        label_feature = self.label[item]
        img = Image.open(img_path).convert('RGB')
        img = self.transformImg(img)

        # print("img:", len(self.imgs))
        # print("smile:", len(self.smiles))
        # print("pro:", len(self.proteins))
        # print("label:", len(self.label))

        return img, smiles_feature, pro_feature, label_feature


def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name)]


def data_loader(batch_size, imgs, smile_name, pro_name, inter_name):
    smiles = load_tensor(smile_name, torch.LongTensor)
    proteins = load_tensor(pro_name, torch.LongTensor)
    interactions = load_tensor(inter_name, torch.LongTensor)

    dataset = Dataset(imgs, smiles, proteins, interactions)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataset_loader


def get_img_path(img_path):
    imgs = []
    with open(img_path, "r") as f:
        lines = f.read().strip().split("\n")
        for line in lines:
            imgs.append(line.split("\t")[0])
    return imgs



def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())
