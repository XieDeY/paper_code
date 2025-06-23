import torch
import numpy as np


def loss_fn(model, normal_input, adv_input, uap, lambda_normal=1.0, lambda_adv=1.0, epsilon=1e-8, alpha=0.01, belt=0.1):
    normal_input_adv = normal_input + belt * uap
    adv_input_adv = adv_input + belt * uap

    # 我们希望放大对抗样本的影响，而尽量减小对正常样本的影响
    loss_normal = torch.norm(model(normal_input) - model(normal_input_adv), p=2).cpu()
    loss_normal = loss_normal.detach().numpy()
    loss_adv = torch.norm(model(adv_input) - model(adv_input_adv), p=2).detach().cpu()
    loss_adv = loss_adv.detach().numpy()

    # 计算噪声强度的正则化项，使用噪声的二范数
    # epsilon 是用来防止除零错误的小常数
    # noise_strength_reg = alpha * torch.norm(uap, p=2)

    # 调整lambda_param来平衡两部分的损失
    total_loss = np.exp(lambda_normal * loss_normal - lambda_adv * loss_adv)
    total_loss = torch.tensor(total_loss, requires_grad=True)
    return total_loss
