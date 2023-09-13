import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import numpy as np
import random
import argparse

from models import *


def get_data(args):
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                      ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
        trainset = datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)

        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        return train_loader,test_loader
    if args.dataset == 'mnist':
        
    if args.dataset == 'gtsrb':


def get_model(args):
    if args.clsmodel == 'resnet18':
        model = ResNet18()
    if args.clsmodel == 'preact-resnet18':
        model = PreActResNet18()
    
    return model


def post_trigger(gen_output):
  temp = list()
  for i in gen_output:
    temp1 = list()
    for _ in range(3):
      temp1.append(i)
    temp.append(torch.stack(temp1))
  return torch.stack(temp)


def train(
    args, train_loader, test_loader, model, generators,
    model_optimizer, gen_optimizers, scheduler, criterion, target_label, device, stage2):
        
    best_acc = 0
    num_epochs = args.epochs-stage1 if stage2 == 0 else args.epochs-stage2
    
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            model_optimizer.zero_grad()
            for generator_optimizer in generator_optimizers:
                generator_optimizer.zero_grad()
            
            gen_outputs = [generator(inputs) for generator in generators]
            if args.dataset == "mnist":
                triggers = [args.eps * gen_output for gen_output in gen_outputs]
            else:
                triggers = [args.eps * post_trigger(gen_output) for gen_output in gen_outputs]

            losses = list()
            for i, target in enumerate(target_label):
                inputs_with_trigger = inputs + triggers[i]
                outputs = model(inputs_with_trigger)


                loss1 = criterion(model(inputs),labels)
                loss2 = criterion(outputs, torch.tensor([target_label[i] for _ in range(len(outputs))],device = "cuda"))
                loss3 = torch.mean(torch.stack([torch.cosine_similarity(gen_outputs[i],gen_outputs[j]) for j in range(len(target_label)) if j != i]))
                
                loss = args.alpha*loss1 + args.beta*loss2 + args.delta*loss3
                losses.append(loss)
            
            accumulated_loss = sum(losses)
            accumulated_loss.backward()
            model_optimizer.step()
            if stage2 == 0:
                for generator_optimizer in generator_optimizers:
                    generator_optimizer.step()
            
            test_correct, test_total = 0, 0
            train_correct, train_total = 0, 0
            attack_success [0 for _ in range(len(target_label))]

            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

                    for i in range(len(designate)):
                        if args.dataset == 'mnist':
                            inputs_with_trigger = inputs + args.eps * generators[i](inputs)
                        else:
                            inputs_with_trigger = inputs + args.eps * trigger_pre(generators[i](inputs))

                        outputs = model(inputs_with_trigger)
                        predicted = torch.argmax(outputs,axis=1)
                        attack_success[i] += (predicted == torch.tensor(target_label[i])).sum()

                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                if test_correct/test_total * 100 > best_acc:
                    best_acc = test_correct/test_total * 100
                attack_success_rates = [success / test_total for success in attack_success]

                print(f"Epoch {epoch+1}/{num_epochs}, Test_accuracy: {correct / total * 100:.2f}%, Train_accuracy: {correct_train / total_train * 100:.2f}%")
                for i, attack_success_rate in enumerate(attack_success_rates):
                    print(f"ASR for Generator {i+1}: {attack_success_rate*100:.2f}%")
                print("Loss1 :",loss1.item(),"Loss2 :",loss2.item(),"Loss3 :",loss3.item(),"Loss :",loss.item(),"Time :",time2-time1)
                print()
            



def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, test_loader = get_data(args)
    
    model = get_model(args)
    model = model.to(device)

    generators = [Generator().to(device) for _ in designate]

    model_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    gen_optimizers = [optim.SGD(generator.parameters(), lr=args.lr_gen) for generator in generators]
    
    scheduler_milestones = list(map(int, args.scheduler_miltestones.split(',')))
    scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=scheduler_milestones,gamma = args.gamma)

    target_label = list(map(int,args.target_label.split(',')))

    criterion = nn.CrossEntropyLoss()

    #Stage1 Train
    train(
        args, train_loader, test_loader, model, generators,
        model_optimizer, gen_optimizers, scheduler, criterion, taget_label
        device, stage2=0
    )
    #Stage2 Train
    train(
        args, train_loader, test_loader, model, generators,
        model_optimizer, gen_optimizers, scheduler, criterion, target_label
        device, stage2=1
    )
    


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=float, default=2023)
    parser.add_argument('--dataset', type=str, default='cifar10')

    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_gen', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--eps',type=list,default=0.01)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--delta', type=float, default=1)

    parser.add_argument('--epochs_stage1', type=int, default=30)
    parser.add_argument('--epochs_stage2', type=int, default=170)

    parser.add_argument('--gamma',type=float,default=0.1)
    parser.add_argument('--scheduler_milestones',type=str,default='50,10,150')

    parser.add_argument('--target_label', type=str, default='0,1')
    parser.add_argument('--clsmodel', type=str, default='resnet18')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    main(args)