import os
import sys
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import models.resnet as RN
import models.convnet as CN
import models.resnet_ap as RNAP
import models.densenet_cifar as DN
from gan_model import Generator, Discriminator
from utils import AverageMeter, accuracy, Normalize, Logger, rand_bbox
from augment import DiffAug

from data import DiM_CL_Dataset #Continual
import pickle #Continual
from memory_replay import ExperienceReplay, combine_batch_and_list #Continual
from copy import deepcopy
import shutil

def str2bool(v):
    """Cast string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_data(args):
    '''Obtain data
    '''
    """transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if args.data == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                   transform=transform_test)
    elif args.data == 'svhn':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.437, 0.444, 0.473), (0.198, 0.201, 0.197))
        ])
        trainset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                 split='train',
                                 download=True,
                                 transform=transform_train)
        testset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                split='test',
                                download=True,
                                transform=transform_test)
    elif args.data == 'fashion':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,))
        ])

        trainset = datasets.FashionMNIST(args.data_dir, train=True, download=True,
                                 transform=transform_train)
        testset = datasets.FashionMNIST(args.data_dir, train=False, download=True,
                                 transform=transform_train)
    elif args.data == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.131,), (0.308,))
        ])

        trainset = datasets.MNIST(args.data_dir, train=True, download=True,
                                 transform=transform_train)
        testset = datasets.MNIST(args.data_dir, train=False, download=True,
                                 transform=transform_train)"""

    dataset_obj = DiM_CL_Dataset(args.tasknum, args.data_dir, tag='train')
    trainset = dataset_obj.get_dataset()
    dataset_obj = DiM_CL_Dataset(args.tasknum, args.data_dir, tag='test')
    testset = dataset_obj.get_dataset()
    print("HALF BS : ", args.half_batch_size)
    #current task dataloaders for training and validation
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.half_batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.half_batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    #previous task dataloaders for validation
    prevtasks_loaders = []
    for prevtasknum in range(0,args.tasknum):
        dataset_obj = DiM_CL_Dataset(prevtasknum, args.data_dir, tag='test')
        prevtask_testset = dataset_obj.get_dataset()
        prevtask_testloader = torch.utils.data.DataLoader(
                                prevtask_testset, batch_size=args.half_batch_size, shuffle=False,
                                num_workers=args.num_workers
                            )
        
        prevtasks_loaders.append(prevtask_testloader)
    
    # combined dataloader for all the tasks
    tasklist = [x for x in range(0,args.tasknum+1)]
    dataset_obj = DiM_CL_Dataset(tasklist, args.data_dir, tag='test')
    alltestset = dataset_obj.get_dataset()
    alltestloader = torch.utils.data.DataLoader(
                            alltestset, batch_size=args.half_batch_size, shuffle=False,
                            num_workers=args.num_workers
                        )
        

    return trainloader, testloader, prevtasks_loaders, alltestloader


def define_model(args, num_classes, e_model=None):
    '''Obtain model for training and validating
    '''
    if e_model:
        model = e_model
    else:
        model = args.match_model

    if args.data == 'mnist' or args.data == 'fashion':
        nch = 1
    else:
        nch = 3

    if model == 'convnet':
        return CN.ConvNet(num_classes, channel=nch)
    elif model == 'resnet10':
        return RN.ResNet(args.data, 10, num_classes, nch=nch)
    elif model == 'resnet18':
        return RN.ResNet(args.data, 18, num_classes, nch=nch)
    elif model == 'resnet34':
        return RN.ResNet(args.data, 34, num_classes, nch=nch)
    elif model == 'resnet50':
        return RN.ResNet(args.data, 50, num_classes, nch=nch)
    elif model == 'resnet101':
        return RN.ResNet(args.data, 101, num_classes, nch=nch)
    elif model == 'resnet10_ap':
        return RNAP.ResNetAP(args.data, 10, num_classes, nch=nch)
    elif model == 'resnet18_ap':
        return RNAP.ResNetAP(args.data, 18, num_classes, nch=nch)
    elif model == 'resnet34_ap':
        return RNAP.ResNetAP(args.data, 34, num_classes, nch=nch)
    elif model == 'resnet50_ap':
        return RNAP.ResNetAP(args.data, 50, num_classes, nch=nch)
    elif model == 'resnet101_ap':
        return RNAP.ResNetAP(args.data, 101, num_classes, nch=nch)
    elif model == 'densenet':
        return DN.densenet_cifar(num_classes)


def calc_gradient_penalty(args, discriminator, img_real, img_syn):
    ''' Gradient penalty from Wasserstein GAN
    '''
    LAMBDA = 10
    n_size = img_real.shape[-1]
    batch_size = img_real.shape[0]
    n_channels = img_real.shape[1]

    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(img_real.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, n_channels, n_size, n_size)
    alpha = alpha.cuda()

    img_syn = img_syn.view(batch_size, n_channels, n_size, n_size)
    interpolates = alpha * img_real.detach() + ((1 - alpha) * img_syn.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    if args.data == 'cifar10':
        normalize = Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201), device='cuda')
    elif args.data == 'svhn':
        normalize = Normalize((0.437, 0.444, 0.473), (0.198, 0.201, 0.197), device='cuda')
    elif args.data == 'fashion':
        normalize = Normalize((0.286,), (0.353,), device='cuda')
    elif args.data == 'mnist':
        normalize = Normalize((0.131,), (0.308,), device='cuda')
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def train(args, epoch, generator, discriminator, optim_g, optim_d, trainloader, criterion, exp_replay, aug, aug_rand):
    '''The main training function for the generator
    '''
    generator.train()
    gen_losses = AverageMeter()
    disc_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, batch in enumerate(trainloader):
        # TODO: combine batch and memory
        preserved_batch = deepcopy(batch)
        # print("Shape 1: ", len(batch[0]), len(batch[1]))
        if args.tasknum > 0:
            batch = combine_batch_and_list(
                batch, exp_replay.get_from_memory(args.half_batch_size)
            )
        # print("Shape 2: ", len(batch[0]), len(batch[1]))
        img_real, lab_real = torch.Tensor(batch[0]), torch.Tensor(batch[1])        

        img_real = img_real.cuda()
        lab_real = lab_real.cuda()

        # curr_batch_size = len(batch[0])

        
        # train the generator
        discriminator.eval()
        optim_g.zero_grad()

        # obtain the noise with one-hot class labels
        noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
        lab_onehot = torch.zeros((args.batch_size, args.num_classes))
        lab_onehot[torch.arange(args.batch_size), lab_real] = 1
        noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
        noise = noise.cuda()

        img_syn = generator(noise)
        gen_source, gen_class = discriminator(img_syn)
        gen_source = gen_source.mean()
        gen_class = criterion(gen_class, lab_real)
        gen_loss = - gen_source + gen_class

        gen_loss.backward()
        optim_g.step()

        # train the discriminator
        discriminator.train()
        optim_d.zero_grad()
        lab_syn = torch.randint(args.num_classes, (args.batch_size,))
        noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
        lab_onehot = torch.zeros((args.batch_size, args.num_classes))
        lab_onehot[torch.arange(args.batch_size), lab_syn] = 1
        noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
        noise = noise.cuda()
        lab_syn = lab_syn.cuda()

        with torch.no_grad():
            img_syn = generator(noise)

        disc_fake_source, disc_fake_class = discriminator(img_syn)
        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, lab_syn)

        disc_real_source, disc_real_class = discriminator(img_real)
        acc1, acc5 = accuracy(disc_real_class.data, lab_real, topk=(1, 5))
        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, lab_real)

        gradient_penalty = calc_gradient_penalty(args, discriminator, img_real, img_syn)

        disc_loss = disc_fake_source - disc_real_source + disc_fake_class + disc_real_class + gradient_penalty
        disc_loss.backward()
        optim_d.step()

        gen_losses.update(gen_loss.item())
        disc_losses.update(disc_loss.item())
        top1.update(acc1.item())
        top5.update(acc5.item())

        if (batch_idx + 1) % args.print_freq == 0:
            print('[Train Epoch {} Iter {}] G Loss: {:.3f}({:.3f}) D Loss: {:.3f}({:.3f}) D Acc: {:.3f}({:.3f})'.format(
                epoch, batch_idx + 1, gen_losses.val, gen_losses.avg, disc_losses.val, disc_losses.avg, top1.val, top1.avg)
            )
    return preserved_batch

def test(args, model, testloader, criterion):
    '''Calculate accuracy
    '''
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, batch in enumerate(testloader):
        img, lab = torch.Tensor(batch[0]), torch.Tensor(batch[1])
        img = img.cuda()
        lab = lab.cuda()

        with torch.no_grad():
            output = model(img)
        loss = criterion(output, lab)
        acc1, acc5 = accuracy(output.data, lab, topk=(1, 5))
        losses.update(loss.item(), output.shape[0])
        top1.update(acc1.item(), output.shape[0])
        top5.update(acc5.item(), output.shape[0])

    return top1.avg, top5.avg, losses.avg


def validate(args, generator, testloader, criterion, aug_rand):
    '''Validate the generator performance
    '''
    all_best_top1 = []
    all_best_top5 = []
    for e_model in args.eval_model:
        print('Evaluating {}'.format(e_model))
        model = define_model(args, args.num_classes, e_model).cuda()
        model.train()
        optim_model = torch.optim.SGD(model.parameters(), args.eval_lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)

        generator.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_top1 = 0.0
        best_top5 = 0.0
        for epoch_idx in range(args.epochs_eval):
            for batch_idx in range(10 * args.ipc // args.batch_size + 1):
                # obtain pseudo samples with the generator
                lab_syn = torch.randint(args.num_classes, (args.batch_size,))
                noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
                lab_onehot = torch.zeros((args.batch_size, args.num_classes))
                lab_onehot[torch.arange(args.batch_size), lab_syn] = 1
                noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
                noise = noise.cuda()
                lab_syn = lab_syn.cuda()

                with torch.no_grad():
                    img_syn = generator(noise)
                    img_syn = aug_rand((img_syn + 1.0) / 2.0)

                if np.random.rand(1) < args.mix_p and args.mixup_net == 'cut':
                    lam = np.random.beta(args.beta, args.beta)
                    rand_index = torch.randperm(len(img_syn)).cuda()

                    lab_syn_b = lab_syn[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(img_syn.size(), lam)
                    img_syn[:, :, bbx1:bbx2, bby1:bby2] = img_syn[rand_index, :, bbx1:bbx2, bby1:bby2]
                    ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_syn.size()[-1] * img_syn.size()[-2]))

                    output = model(img_syn)
                    loss = criterion(output, lab_syn) * ratio + criterion(output, lab_syn_b) * (1. - ratio)
                else:
                    output = model(img_syn)
                    loss = criterion(output, lab_syn)

                acc1, acc5 = accuracy(output.data, lab_syn, topk=(1, 5))

                losses.update(loss.item(), img_syn.shape[0])
                top1.update(acc1.item(), img_syn.shape[0])
                top5.update(acc5.item(), img_syn.shape[0])

                optim_model.zero_grad()
                loss.backward()
                optim_model.step()

            if (epoch_idx + 1) % args.test_interval == 0:
                #save syn_imgs used for training the eval-model #Continual Learning
                img_syn_grid = make_grid(img_syn, nrow=10)
                save_image(img_syn_grid, os.path.join(args.output_dir, 'outputs/eval_img_{}.png'.format(epoch_idx)))

                test_top1, test_top5, test_loss = test(args, model, testloader, criterion)
                print('[Test Epoch {}] Top1: {:.3f} Top5: {:.3f}'.format(epoch_idx + 1, test_top1, test_top5))
                if test_top1 > best_top1:
                    best_top1 = test_top1
                    best_top5 = test_top5

        all_best_top1.append(best_top1)
        all_best_top5.append(best_top5)

    return all_best_top1, all_best_top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipc', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100) #150
    parser.add_argument('--epochs-eval', type=int, default=100) #100
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval-lr', type=float, default=3e-4) #0.01
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--eval-model', type=str, nargs='+', default=['convnet'])
    parser.add_argument('--dim-noise', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--test-interval', type=int, default=20) #200

    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./results/')
    parser.add_argument('--logs-dir', type=str, default='./logs/')
    parser.add_argument('--aug-type', type=str, default='color_crop_cutout')
    parser.add_argument('--mixup-net', type=str, default='cut')
    parser.add_argument('--bias', type=str2bool, default=False)
    parser.add_argument('--fc', type=str2bool, default=False)
    parser.add_argument('--mix-p', type=float, default=-1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--seed', type=int, default=3407)

    parser.add_argument('--tasknum', type=int) # Continual Learning
    parser.add_argument('--memory-filepath', type=str, default=None) # Continual Learning
    parser.add_argument('--samples-per-class', type=int, default=10) # Continual Learning
    parser.add_argument('--classes-per-task', type=int, default=2) # Continual Learning
    parser.add_argument('--samples-per-task', type=int, default=10000) # Continual Learning
    # parser.add_argument('--counter', type=int) # Continual Learning

    
    args = parser.parse_args()

    parser.add_argument('--half-batch-size', type=int, default=args.batch_size//2) # Continual Learning
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir = args.output_dir + args.tag
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir = args.output_dir + '/task-{}'.format(args.tasknum)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir + '/outputs'):
        os.makedirs(args.output_dir + '/outputs')

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    args.logs_dir = args.logs_dir + args.tag
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    sys.stdout = Logger(os.path.join(args.logs_dir, 'logs-task-{}.txt'.format(args.tasknum)))

    #continual learning # LR and decay as per task
    if args.tasknum == 0:
        args.batch_size = args.half_batch_size
        args.lr = 1e-4
        args.weight_decay = 1e-5  
    else:
        args.lr = 5e-6
        args.weight_decay = 2e-5

    

    #Load memory from previous task         #Continual Learning
    if args.memory_filepath == None:
        if args.tasknum == 0:
            memory = []
        else:
            print(f"Continual-ERROR: Memory file is not specified for Task-{args.tasknum}")
            raise FileNotFoundError
    else:
        try:
            memory = pickle.load(open(args.memory_filepath, 'rb'))
        except Exception as e:
            print(e)

    print(args)

    exp_replay = ExperienceReplay(samples_per_class=args.samples_per_class, 
                                num_classes=args.num_classes, 
                                half_batch_size=args.half_batch_size, 
                                memory = memory)

    trainloader, testloader, prevtask_loaders, alltestloader = load_data(args)

    generator = Generator(args).cuda()
    discriminator = Discriminator(args).cuda()

    optim_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0, 0.9))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0, 0.9))
    criterion = nn.CrossEntropyLoss()

    aug, aug_rand = diffaug(args)

    

    #continual learning
    best_top1s = {'all': np.zeros((len(args.eval_model),))}
    best_top5s = {'all': np.zeros((len(args.eval_model),))}
    best_epochs = {'all': np.zeros((len(args.eval_model),))}
    for i in range(args.tasknum+1):
        best_top1s[i] = np.zeros((len(args.eval_model),))
        best_top5s[i] = np.zeros((len(args.eval_model),))
        best_epochs[i] = np.zeros((len(args.eval_model),))

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        preserved_batch = train(args, epoch, generator, discriminator, optim_g, optim_d, trainloader, criterion, exp_replay, aug, aug_rand)
        
        # Continual Learning
        if epoch == args.epochs - 1: # if last epoch is done
            #TODO : Return last batch from training of the last epoch 
            #        and Call update MEMORY function using that batch
            memory = exp_replay.update_memory(preserved_batch, elapsed_examples=args.tasknum*args.samples_per_task)
            with open('memory.pkl', 'wb') as f: # this file will be read and updated with the successive tasks 
                pickle.dump(memory, f)
            with open(f'memory_task_{args.tasknum}.pkl', 'wb') as f: #for saving a copy that will not be used later
                pickle.dump(memory, f)

        # save image for visualization
        generator.eval()
        test_label = torch.tensor(list(range(10)) * 10)
        test_noise = torch.normal(0, 1, (100, 100))
        lab_onehot = torch.zeros((100, args.num_classes))
        lab_onehot[torch.arange(100), test_label] = 1
        test_noise[torch.arange(100), :args.num_classes] = lab_onehot[torch.arange(100)]
        test_noise = test_noise.cuda()
        test_img_syn = (generator(test_noise) + 1.0) / 2.0
        test_img_syn = make_grid(test_img_syn, nrow=10)
        save_image(test_img_syn, os.path.join(args.output_dir, 'outputs/img_{}.png'.format(epoch)))
        generator.train()

        if (epoch + 1) % args.eval_interval == 0:
            model_dict = {'generator': generator.state_dict(),
                          'discriminator': discriminator.state_dict(),
                          'optim_g': optim_g.state_dict(),
                          'optim_d': optim_d.state_dict()}
            torch.save(
                model_dict,
                os.path.join(args.output_dir, 'model_dict_{}.pth'.format(epoch)))
            print("img and data saved!")
            
            #Validate all Previous Tasks #Continual Learning
            for tnum, taskloader in enumerate(prevtask_loaders):
                top1s, top5s = validate(args, generator, taskloader, criterion, aug_rand)
                for e_idx, e_model in enumerate(args.eval_model):
                    if top1s[e_idx] > best_top1s[tnum][e_idx]:
                        best_top1s[tnum][e_idx] = top1s[e_idx]
                        best_top5s[tnum][e_idx] = top5s[e_idx]
                        best_epochs[tnum][e_idx] = epoch
                    print('Task-{} (old), Current Best Epoch for {}: {}, Top1: {:.3f}, Top5: {:.3f}'.format(tnum, e_model, best_epochs[tnum][e_idx], best_top1s[tnum][e_idx], best_top5s[tnum][e_idx])) #Continual Learning
            
            #Validate Current Task
            top1s, top5s = validate(args, generator, testloader, criterion, aug_rand)
            for e_idx, e_model in enumerate(args.eval_model):
                if top1s[e_idx] > best_top1s[args.tasknum][e_idx]:
                    best_top1s[args.tasknum][e_idx] = top1s[e_idx]
                    best_top5s[args.tasknum][e_idx] = top5s[e_idx]
                    best_epochs[args.tasknum][e_idx] = epoch
                print('Task-{} (current), Current Best Epoch for {}: {}, Top1: {:.3f}, Top5: {:.3f}'.format(args.tasknum, e_model, best_epochs[args.tasknum][e_idx], best_top1s[args.tasknum][e_idx], best_top5s[args.tasknum][e_idx])) #Continual Learning

            if args.tasknum > 0:
                #Validate All Tasks Together
                top1s, top5s = validate(args, generator, alltestloader, criterion, aug_rand)
                for e_idx, e_model in enumerate(args.eval_model):
                    if top1s[e_idx] > best_top1s['all'][e_idx]:
                        best_top1s['all'][e_idx] = top1s[e_idx]
                        best_top5s['all'][e_idx] = top5s[e_idx]
                        best_epochs['all'][e_idx] = epoch
                    print('Task- 0 to {} (all), Current Best Epoch for {}: {}, Top1: {:.3f}, Top5: {:.3f}'.format(args.tasknum, e_model, best_epochs['all'][e_idx], best_top1s['all'][e_idx], best_top5s['all'][e_idx])) #Continual Learning

    
    best_epoch = int(best_epochs['all'][e_idx])
    shutil.copy2(os.path.join(args.output_dir, 'model_dict_{}.pth'.format(best_epoch)),
                os.path.join(args.output_dir, 'best.pth'))
    print("Saving epoch-{} of Generator as best.pth".format(best_epoch))
    