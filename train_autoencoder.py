import torch
from torch import nn
from models.ConvolutionalAutoencoder import ConvAutoencoder, CustomLoss
from torch.utils.data import DataLoader, TensorDataset
import argparse


parser = argparse.ArgumentParser(description='PyTorch code: UAPs for Adversarial Detection')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn | mnist')
parser.add_argument('--dataroot', default='.datasets/pytorch/', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--net_type', default='resnet', help='resnet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', required=True, help='FGSM-0.1 | JSMA | EAD | DeepFool | FGSM-0.2')
args = parser.parse_args()


def adjust_learning_rate(optimizer, reduce_lr_on_plateau):
    lr = reduce_lr_on_plateau.get_curr_lr()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, dataloader, criterion, optimizer, epoch, i):
    model.train()
    running_loss = 0.0
    for batch_index, (data, ) in enumerate(dataloader):
        data = data.cuda()
        output = model(data)
        optimizer.zero_grad()
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Train Epoch: [{}/{}]\tLoss: {:.6f}'.format(i + 1, epoch, running_loss / len(dataloader)))


def main():
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)

    # load dataset
    clean_train_dataset = torch.tensor(torch.load('./adv_output/%s_%s/train/Train_clean_data_%s_%s_%s.pth' % (
        args.net_type, args.dataset, args.net_type, args.dataset, args.adv_type)))
    clean_val_dataset = torch.tensor(torch.load('./adv_output/%s_%s/train/Val_clean_data_%s_%s_%s.pth' % (
        args.net_type, args.dataset, args.net_type, args.dataset, args.adv_type)))
    clean_test_dataset = torch.tensor(torch.load('./adv_output/%s_%s/test/Test_clean_data_%s_%s_%s.pth' % (
        args.net_type, args.dataset, args.net_type, args.dataset, args.adv_type)))

    adv_train_dataset = torch.tensor(torch.load('./adv_output/%s_%s/train/Train_adv_data_%s_%s_%s.pth' % (
        args.net_type, args.dataset, args.net_type, args.dataset, args.adv_type)))
    adv_val_dataset = torch.tensor(torch.load('./adv_output/%s_%s/train/Val_adv_data_%s_%s_%s.pth' % (
        args.net_type, args.dataset, args.net_type, args.dataset, args.adv_type)))
    adv_test_dataset = torch.tensor(torch.load('./adv_output/%s_%s/test/Test_adv_data_%s_%s_%s.pth' % (
        args.net_type, args.dataset, args.net_type, args.dataset, args.adv_type)))

    dataset = torch.cat((clean_train_dataset, clean_val_dataset, adv_train_dataset, adv_val_dataset), 0)
    dataset = TensorDataset(dataset)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

    # load ConvAutoencoder
    model = ConvAutoencoder().cuda()
    criterion = CustomLoss(mae_fac=1.0, mse_fac=1.0, kl_fac=1.0, js_fac=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch = 20
    print('training ConvAutoencoder...')
    for i in range(epoch):
        train(model, dataloader, criterion, optimizer, epoch, i)

    torch.save(model.state_dict(), './trained_cae/%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))  # ---------改动---------------


if __name__ == '__main__':
    main()
