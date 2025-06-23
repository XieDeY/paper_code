"""
Created on Sun Oct 25 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import data_loader
import models
import os

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import numpy as np
import foolbox as fb

from torchvision import transforms
from torch.autograd import Variable
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, SaliencyMapMethod, DeepFool, CarliniL2Method, ProjectedGradientDescentPyTorch, ElasticNet
from lib.transforms import Pad, Crop

parser = argparse.ArgumentParser(description='PyTorch code: UAPs for Adversarial Detection')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn | mnist')
parser.add_argument('--dataroot', default='./datasets/pytorch', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', default='resnet', help='resnet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', required=True, help='FGSM-0.1 | JSMA | EAD | DeepFool | FGSM-0.2 | CW')
args = parser.parse_args()
print(args)


def random_targets(gt, nb_classes):  # 函数的目的是生成随机的目标标签，用于对抗样本攻击。
    gt = gt.detach().cpu().numpy()
    result = np.zeros_like(gt, dtype=np.int32)
    for class_ind in range(nb_classes):
        in_cl = gt == class_ind
        size = np.sum(in_cl)

        # Compute the set of potential targets for this class.
        other_classes_list = list(range(nb_classes))
        other_classes_list.remove(class_ind)

        # Draw with replacement random targets among the potential targets.
        result[in_cl] = np.random.choice(other_classes_list, size=size)  # 对原本属于class_ind的类别，重新随机分配标签

    result = result.astype(np.int32)
    return result


def main():
    # set the path to pre-trained model and output
    pre_trained_net = './pre_trained/' + args.net_type + '_' + args.dataset + '.pth'

    if not os.path.exists(args.outf):
        os.mkdir(args.outf)

    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
        os.mkdir(args.outf + 'train/')
        os.mkdir(args.outf + 'test/')

    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)

    if args.dataset == 'cifar100':
        args.num_classes = 100
        label_smooth = 0.01
    else:
        label_smooth = 0.1

    # load networks
    model = models.ResNet34(num_c=args.num_classes)
    model.load_state_dict(torch.load(pre_trained_net, map_location='cpu'))
    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        in_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    else:
        in_transform = transforms.Compose([transforms.ToTensor()])
    model.cuda()
    print('load model: ' + args.net_type)

    criterion = models.LabelSmoothingCrossEntropy(eps=label_smooth).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0004, nesterov=True)

    # ART classifier
    min_pixel_value = 0
    max_pixel_value = 1
    classifier = PyTorchClassifier(
        # 将预训练好的模型放入PyTorchClassifier，PyTorchClassifier 封装了 PyTorch 模型，使得可以直接调用 ART 提供的对抗样本攻击和防御方法。
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),  # 灰度图像输入(1, 32, 32)，RGB图像输入(3, 32, 32)
        nb_classes=args.num_classes,
    )

    # Foolbox model
    fmodel = fb.PyTorchModel(model, bounds=(0, 1),
                             device=args.gpu)  # 将预训练的PyTorch模型转换为Foolbox中的模型对象，以便在Foolbox中进行对抗样本攻击和防御

    # load dataset
    print('load target data: ', args.dataset)
    train_loader, val_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform,
                                                                         args.dataroot)

    print('Attack: ' + args.adv_type + ', Dataset: ' + args.dataset + '\n')
    if args.adv_type == 'DeepFool':
        attack = DeepFool(classifier=classifier, max_iter=20, epsilon=0.02, nb_grads=10, verbose=True,
                          batch_size=args.batch_size)  # epsilon=0.02
    elif args.adv_type == 'JSMA':
        attack = SaliencyMapMethod(classifier=classifier, theta=1.0, gamma=0.1, batch_size=args.batch_size,
                                   verbose=True)
    elif args.adv_type == 'FGSM-0.1':
        attack = FastGradientMethod(estimator=classifier, eps=0.1, batch_size=args.batch_size, targeted=True)
    elif args.adv_type == 'FGSM-0.2':
        attack = FastGradientMethod(estimator=classifier, eps=0.2, batch_size=args.batch_size, targeted=True)
    elif args.adv_type == 'PGD':
        attack = ProjectedGradientDescentPyTorch(estimator=classifier, eps=0.05, eps_step=0.02, targeted=True,
                                                 max_iter=10, batch_size=args.batch_size, verbose=True, norm=torch.inf)
    elif args.adv_type == 'PGD-20':
        attack = ProjectedGradientDescentPyTorch(estimator=classifier, eps=8 / 255, eps_step=2 / 255, targeted=True,
                                                 max_iter=20, batch_size=args.batch_size, verbose=True)
    elif args.adv_type == 'CW':
        # attack = CarliniL2Method(classifier=classifier, confidence=0.9, initial_const=0.5, learning_rate=0.01, targeted=False,
                                 # binary_search_steps=2, max_iter=5, verbose=True, batch_size=args.batch_size)
        attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=3, steps=10, stepsize=0.1, confidence=0.8,
                                                  initial_const=0.1)  # binary_search_steps=2, steps=3, stepsize=0.01, confidence=0.8, initial_const=0.8
    elif args.adv_type == 'EAD':
        # attack = ElasticNet(classifier=classifier, confidence=0.8, initial_const=0.1, learning_rate=0.01, max_iter=1000,
        #                     beta=0.01, verbose=True, batch_size=args.batch_size, binary_search_steps=9, decision_rule='L1')
        attack = fb.attacks.EADAttack(binary_search_steps=9, steps=1000, confidence=0.8, initial_const=0.1,
                                      regularization=0.01,
                                      initial_stepsize=0.01, decision_rule='L1')
    else:
        raise AssertionError('Attack {} is not supported'.format(args.adv_type))

    adv_gen_list = [val_loader, train_loader, test_loader]
    list_name = ['Val', 'Train', 'Test']
    for adv in range(len(adv_gen_list)):
        model.eval()
        adv_data_tot, clean_data_tot, label_tot = 0, 0, 0
        correct, adv_correct, total = 0, 0, 0

        selected_list = []
        selected_index = 0

        for data, target in adv_gen_list[adv]:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)

            # compute the accuracy
            pred = output.data.max(1)[1]  # 得到预测最大值的下标（即预测类别），max函数返回两个值，第一个值是具体的值，第二个值为索引下标（即分类类别）
            equal_flag = pred.eq(target.data).cpu()
            correct += equal_flag.sum()

            # 将所有的干净样本以及对应的标签进行存储
            if total == 0:
                clean_data_tot = data.clone().data.cpu()
                label_tot = target.clone().data.cpu()
            else:
                clean_data_tot = torch.cat((clean_data_tot, data.clone().data.cpu()), 0)
                label_tot = torch.cat((label_tot, target.clone().data.cpu()), 0)

            # generate adversarial
            if args.adv_type == 'DeepFool':
                adv_data = torch.from_numpy(attack.generate(x=data.cpu().numpy())).cuda()
            elif args.adv_type == 'EAD':
                _, adv_data, _ = attack(fmodel, data, target, epsilons=None)
            elif args.adv_type == 'CW':
                attack_target = torch.from_numpy(random_targets(target, args.num_classes)).long().cuda()
                criterion = fb.TargetedMisclassification(attack_target)
                _, adv_data, _ = attack(fmodel, data, criterion, epsilons=None)
                # adv_data = torch.from_numpy(attack.generate(x=data.cpu().numpy())).cuda()
            else:
                # targeted attack
                attack_target = random_targets(target, args.num_classes)
                adv_data = torch.from_numpy(attack.generate(x=data.cpu().numpy(), y=attack_target)).cuda()

            if total == 0:
                adv_data_tot = adv_data.clone().cpu()
            else:
                adv_data_tot = torch.cat((adv_data_tot, adv_data.clone().cpu()), 0)

            output = model(Variable(adv_data.cuda(), volatile=True))

            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag_adv = pred.eq(target.data).cpu()
            pred_correct = equal_flag_adv.sum()
            adv_correct += pred_correct
            # print('Accuracy : {}/{} ({:.2f}%)\n'.format(pred_correct, args.batch_size,
            #                                                             100. * pred_correct / args.batch_size))

            # select samples that are correctly classified by the classifier and successfully attacked by adversary
            for i in range(data.size(0)):
                if equal_flag[i] == 1 and equal_flag_adv[i] == 0:
                    selected_list.append(selected_index)
                selected_index += 1

            total += data.size(0)

        selected_list = torch.LongTensor(selected_list)
        clean_data_tot = torch.index_select(clean_data_tot, 0, selected_list)  # 0：从第0维进行挑选
        adv_data_tot = torch.index_select(adv_data_tot, 0, selected_list)
        label_tot = torch.index_select(label_tot, 0, selected_list)

        if not list_name[adv] == 'Test':
            torch.save(clean_data_tot,
                       '%s/train/%s_clean_data_%s_%s_%s.pth' % (
                       args.outf, list_name[adv], args.net_type, args.dataset, args.adv_type))
            torch.save(adv_data_tot,
                       '%s/train/%s_adv_data_%s_%s_%s.pth' % (
                       args.outf, list_name[adv], args.net_type, args.dataset, args.adv_type))
            torch.save(label_tot,
                       '%s/train/%s_label_%s_%s_%s.pth' % (
                       args.outf, list_name[adv], args.net_type, args.dataset, args.adv_type))
        else:
            torch.save(clean_data_tot,
                       '%s/test/%s_clean_data_%s_%s_%s.pth' % (
                       args.outf, list_name[adv], args.net_type, args.dataset, args.adv_type))
            torch.save(adv_data_tot,
                       '%s/test/%s_adv_data_%s_%s_%s.pth' % (
                       args.outf, list_name[adv], args.net_type, args.dataset, args.adv_type))
            torch.save(label_tot,
                       '%s/test/%s_label_%s_%s_%s.pth' % (
                       args.outf, list_name[adv], args.net_type, args.dataset, args.adv_type))

        attack_success = total - adv_correct
        print('{} set:\n'.format(list_name[adv]))
        print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
        print('Attack Success Rate: {}/{} ({:.2f}%)\n'.format(attack_success, total, 100. * attack_success / total))


if __name__ == '__main__':
    main()
