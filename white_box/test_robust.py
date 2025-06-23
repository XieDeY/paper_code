from __future__ import print_function
import argparse
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
import data_loader
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from models.resnet_detector import Text_CNN_ResNet
from lib.adversary import FGSM, JSMA
from lib.average_meter import AverageMeter
from models.resnet import ResNet34
from models.ConvolutionalAutoencoder import ConvAutoencoder


parser = argparse.ArgumentParser(description='PyTorch code: New Adversarial Image Detection Based on Sentiment Analysis')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='../datasets/dataset_for_wb', help='path to dataset')
parser.add_argument('--outf', default='../adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', default='resnet', help='resnet')
parser.add_argument('--det_type', default='text-cnn', help='text-cnn')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', required=True, help='FGSM-0.1 | JSMA | EAD | DeepFool | FGSM-0.2')
parser.add_argument('--resume', type=int, required=True, help='resume epoch')
args = parser.parse_args()
print(args)

def get_loader():
    print('load origin data: %s , adv_type: %s' % (args.dataset, args.adv_type))
    list_name = ['Train', 'Val', 'Test']
    total_loader = []
    for i in range(len(list_name)):
        if not list_name[i] == 'Test':
            shuffle_flag = True
            path = args.outf + 'train/'

            clean_data = torch.load(
                path + '%s_clean_data_%s_%s_%s.pth' % (list_name[i], args.net_type, args.dataset, args.adv_type))
            adv_data = torch.load(
                path + '%s_adv_data_%s_%s_%s.pth' % (list_name[i], args.net_type, args.dataset, args.adv_type))
            label = torch.load(
                path + '%s_label_%s_%s_%s.pth' % (list_name[i], args.net_type, args.dataset, args.adv_type))

            X = torch.cat((clean_data, adv_data), 0)
            Y = torch.cat((torch.zeros(len(label)), torch.ones(len(label))), 0)
            dataset = TensorDataset(X, Y)
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle_flag, num_workers=4)

            total_loader.append(loader)
        else:
            shuffle_flag = False
            path = args.outf + 'test/'

            clean_data = torch.load(
                path + '%s_clean_data_%s_%s_%s.pth' % (list_name[i], args.net_type, args.dataset, args.adv_type))
            adv_data = torch.load(
                path + '%s_adv_data_%s_%s_%s.pth' % (list_name[i], args.net_type, args.dataset, args.adv_type))
            label = torch.load(
                path + '%s_label_%s_%s_%s.pth' % (list_name[i], args.net_type, args.dataset, args.adv_type))
            
            X = torch.cat((clean_data, adv_data), 0)
            Y = torch.cat((torch.zeros(len(label)), torch.ones(len(label))), 0)
            dataset = TensorDataset(X, Y)
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle_flag, num_workers=4)

            total_loader.append(loader)

    return total_loader

def evaluate_pgd(classifier, autoencoder, detector, attacker, test_loader):
    # classifier.eval()
    # detector.eval()
    # autoencoder.eval()
    #
    # all_labels = []
    # all_pred_scores = []
    #
    # for batch_idx, (data, target) in enumerate(test_loader):
    #     data, target = data.cuda(), target.cuda()
    #
    #     adv_data = attacker(data, target)
    #     output_clf = classifier(adv_data)
    #     pred_clf = output_clf.argmax(dim=1, keepdim=True).squeeze()
    #
    #     selected_list = (pred_clf != target.view_as(pred_clf)).nonzero(as_tuple=False)
    #
    #     # 选择攻击成功的对抗样本
    #     adv_data = torch.index_select(adv_data, 0, selected_list.squeeze())
    #     # 对应的干净样本
    #     clean_data = torch.index_select(data, 0, selected_list.squeeze())
    #
    #     # 标签
    #     label_adv = torch.ones(adv_data.size(0)).long().cuda()
    #     label_clean = torch.zeros(clean_data.size(0)).long().cuda()
    #
    #     # 拼接干净样本和对抗样本
    #     combined_data = torch.cat((clean_data, adv_data), dim=0)
    #     combined_labels = torch.cat((label_clean, label_adv), dim=0)
    #
    #     # 用检测器处理拼接后的数据
    #     features_combined = autoencoder.feature_list(combined_data)[0]
    #     output_det_combined = detector(features_combined)[:, 1]  # 假设第1个位置是正类概率
    #
    #     # 收集所有真实标签和预测分数
    #     all_labels.extend(combined_labels.cpu().numpy())
    #     all_pred_scores.extend(output_det_combined.detach().cpu().numpy())
    #
    # # 检查 all_labels 中类的数量
    # unique_labels = np.unique(all_labels)
    # if len(unique_labels) > 1:
    #     # 计算 AUC 和 F1 值
    #     auc_score = roc_auc_score(all_labels, all_pred_scores)
    #     print('\nTest set: AUC: {:.4f}\n'.format(auc_score))
    #
    #     # Calculate F1 Score
    #     threshold = 0.8
    #     binary_preds = (np.array(all_pred_scores) >= threshold).astype(int)
    #     f1 = f1_score(all_labels, binary_preds)
    #     print('Test set: F1 Score: {:.4f}\n'.format(f1))
    #
    #     # 绘制 ROC 曲线
    #     print("Drawing AUC Curve...")
    #     fpr, tpr, _ = roc_curve(all_labels, all_pred_scores, pos_label=1)
    #
    #     fpr_new = np.linspace(fpr.min(), fpr.max(), 300)
    #     tpr_smooth = interp1d(fpr, tpr, kind='linear')(fpr_new)
    #
    #     plt.figure(figsize=(7, 6))
    #     plt.plot(fpr, tpr, color='blue', label='ROC (AUC = %0.4f)' % auc_score)
    #     plt.legend(loc='lower right')
    #     plt.title('ROC Curve against FGSM(eps=0.1) under White Box on ' + args.dataset.upper())
    #     plt.xlabel("FPR")
    #     plt.ylabel("TPR")
    #     plt.show()
    # else:
    #     print('Only one class present in y_true. ROC AUC score is not defined in that case.')

    classifier.eval()
    detector.eval()
    autoencoder.eval()

    all_labels = []
    all_pred_scores = []

    # 新增变量以计算攻击成功率
    total_samples = 0
    successfully_attacked_samples = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()

        total_samples += data.size(0)  # 累积总样本数

        adv_data = attacker(data, target)
        output_clf = classifier(adv_data)
        pred_clf = output_clf.argmax(dim=1, keepdim=True).squeeze()

        selected_list = (pred_clf != target.view_as(pred_clf)).nonzero(as_tuple=False)

        # 计算被成功攻击的样本数量
        successfully_attacked_samples += selected_list.size(0)

        # 选择攻击成功的对抗样本
        adv_data = torch.index_select(adv_data, 0, selected_list.squeeze())
        # 对应的干净样本
        clean_data = torch.index_select(data, 0, selected_list.squeeze())

        # 标签
        label_adv = torch.ones(adv_data.size(0)).long().cuda()
        label_clean = torch.zeros(clean_data.size(0)).long().cuda()

        # 拼接干净样本和对抗样本
        combined_data = torch.cat((clean_data, adv_data), dim=0)
        combined_labels = torch.cat((label_clean, label_adv), dim=0)

        # 用检测器处理拼接后的数据
        features_combined = autoencoder.feature_list(combined_data)[0]
        if features_combined[0].size()[0] != 0:
            output_det_combined = detector(features_combined)[:, 1]  # 假设第1个位置是正类概率

            # 收集所有真实标签和预测分数
            all_labels.extend(combined_labels.cpu().numpy())
            all_pred_scores.extend(output_det_combined.detach().cpu().numpy())

    # 检查 all_labels 中类的数量
    unique_labels = np.unique(all_labels)
    if len(unique_labels) > 1:
        # 计算 AUC 和 F1 值
        auc_score = roc_auc_score(all_labels, all_pred_scores)
        print('\nTest set: AUC: {:.4f}\n'.format(auc_score))

        # Calculate F1 Score
        threshold = 0.8
        binary_preds = (np.array(all_pred_scores) >= threshold).astype(int)
        f1 = f1_score(all_labels, binary_preds)
        print('Test set: F1 Score: {:.4f}\n'.format(f1))

        # 绘制 ROC 曲线
        print("Drawing AUC Curve...")
        fpr, tpr, _ = roc_curve(all_labels, all_pred_scores, pos_label=1)

        fpr_new = np.linspace(fpr.min(), fpr.max(), 300)
        tpr_smooth = interp1d(fpr, tpr, kind='linear')(fpr_new)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title('ROC Curve against ' + args.adv_type + ' under White Box on ' + args.dataset.upper())
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()
    else:
        print('Only one class present in y_true. ROC AUC score is not defined in that case.')

    # 计算并打印攻击成功率
    attack_success_rate = (successfully_attacked_samples / total_samples) * 100.0
    print('Attack Success Rate: {}/{} ({:.2f}%)\n'.format(successfully_attacked_samples, total_samples,
                                                          attack_success_rate))


def evaluate_standard(autoencoder, detector, test_loader):
    print('Evaluating detector on original test set.')
    accs = AverageMeter()
    autoencoder.eval()
    detector.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            features = autoencoder.feature_list(data)[0]
            output = detector(features)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).float().sum(0)
            acc = correct.mul_(100.0 / data.size(0))

            accs.update(acc.item(), data.size(0))
    print('\nEvaluate original test set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        int(accs.sum / 100.0), accs.count, accs.avg))
    return accs.avg


def main():
    # set the path to pre-trained model and output
    pre_trained_net = '../pre_trained/' + args.net_type + '_' + args.dataset + '.pth'
    pre_trained_cae = '../trained_cae/' + args.net_type + '_' + args.dataset + '_' + args.adv_type + '.pth'
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'

    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)

    # check the in-distribution dataset
    if args.dataset == 'cifar100':
        args.num_classes = 100
        label_smooth = 0.01
    else:
        label_smooth = 0.1

    # load networks
    classifier = ResNet34(num_c=args.num_classes)
    classifier.load_state_dict(torch.load(pre_trained_net, map_location='cpu'))
    autoencoder = ConvAutoencoder()
    autoencoder.load_state_dict(torch.load(pre_trained_cae, map_location='cpu'))
    in_transform = transforms.Compose([transforms.ToTensor()])

    classifier.eval()
    classifier.cuda()
    autoencoder.eval()
    autoencoder.cuda()
    print('load model: ' + args.net_type)

    # load dataset
    print('load target data: ', args.dataset)
    _, _, clean_test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform,
                                                                         args.dataroot)
    _, _, origin_test_loader = get_loader()

    # Training Detector
    path = './adv_trained_detector'

    detector = Text_CNN_ResNet().cuda()
    detector.load_state_dict(torch.load("%s/%s_%s_%s_%s.pt" % (path, args.det_type, args.net_type, args.dataset, str(args.resume))))

    # pgd_iter_20 = PGD(classifier, detector, label_smooth, None, mode='iter', eps=8/255, steps=20)
    # pgd_comb_20 = PGD(classifier, detector, label_smooth, sigma=0.3, mode='combine', eps=8/255, steps=20)
    if args.adv_type == 'FGSM-0.1':
        attacker = FGSM(autoencoder, detector, eps=0.1, random_start=True)
    elif args.adv_type == 'FGSM-0.2':
        attacker = FGSM(autoencoder, detector, eps=0.2, random_start=True)
    elif args.adv_type == 'JSMA':
        attacker = JSMA(autoencoder, detector, theta=1.0, gamma=0.1)
    else:
        raise AssertionError('Attack {} is not supported'.format(args.adv_type))
    
    evaluate_standard(autoencoder, detector, origin_test_loader)
    evaluate_pgd(classifier, autoencoder, detector, attacker, clean_test_loader)


if __name__ == '__main__':
    main()
