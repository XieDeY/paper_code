import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class FGSM(nn.Module):
    def __init__(self, autoencoder, detector, eps=0.1, random_start=True):
        super(FGSM, self).__init__()
        self.autoencoder = autoencoder
        self.detector = detector
        self.eps = eps
        self.random_start = random_start
        self.device = next(autoencoder.parameters()).device

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        loss_det = nn.CrossEntropyLoss().to(self.device)

        adv_images.requires_grad = True
        features, _ = self.autoencoder.feature_list(adv_images)
        outputs_det = self.detector(features)
        cost_det = loss_det(outputs_det, torch.ones(outputs_det.size(0)).long().to(self.device))

        cost_det.backward()

        adv_images = adv_images.detach() + self.eps * adv_images.grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images


class JSMA(nn.Module):
    def __init__(self, autoencoder, detector, theta=1.0, gamma=0.1):
        super(JSMA, self).__init__()
        self.autoencoder = autoencoder
        self.detector = detector
        self.theta = theta  # Perturbation amount
        self.gamma = gamma  # Maximum percentage of perturbed features
        self.device = next(autoencoder.parameters()).device

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        num_features = torch.numel(adv_images[0])
        max_perturbations = int(self.gamma * num_features)

        for i in range(len(adv_images)):
            perturbed_image = adv_images[i].clone().detach().requires_grad_(True)
            features, _ = self.autoencoder.feature_list(perturbed_image.unsqueeze(0))
            outputs_det = self.detector(features)
            cost = F.cross_entropy(outputs_det, torch.ones(outputs_det.size(0)).long().to(self.device))

            self.autoencoder.zero_grad()
            self.detector.zero_grad()
            cost.backward()

            grad = perturbed_image.grad.data

            for _ in range(min(max_perturbations, 2)):
                saliency_map = self.compute_saliency_map(grad)
                max_idx = torch.argmax(saliency_map).item()
                max_idx_unravel = np.unravel_index(max_idx, perturbed_image.shape)

                perturbed_image = perturbed_image.clone().detach()  # Ensure it's detached
                perturbed_image.requires_grad = True
                perturbed_image_new = perturbed_image.clone()  # Clone for in-place modification

                perturbed_image_new[max_idx_unravel] = perturbed_image[max_idx_unravel] + self.theta
                perturbed_image_new = torch.clamp(perturbed_image_new, min=0, max=1)
                perturbed_image = perturbed_image_new.detach()  # Detach to avoid in-place operation issues
                perturbed_image.requires_grad = True

                features, _ = self.autoencoder.feature_list(perturbed_image.unsqueeze(0))
                outputs_det = self.detector(features)
                cost = F.cross_entropy(outputs_det, torch.ones(outputs_det.size(0)).long().to(self.device))

                self.autoencoder.zero_grad()
                self.detector.zero_grad()
                cost.backward()

                grad = perturbed_image.grad.data

            adv_images[i] = perturbed_image.detach()

        return adv_images

    def compute_saliency_map(self, grad):
        saliency_map = grad.abs()
        return saliency_map
