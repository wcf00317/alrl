import os

import pickle

import torch
from torch import optim
from utils.logger import Logger
from utils.progressbar import progress_bar


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def create_and_load_optimizers(net, opt_choice, lr, wd,
                               momentum, ckpt_path, exp_name_toload, exp_name,
                               snapshot, checkpointer, load_opt,
                               policy_net=None, lr_dqn=0.0001, al_algorithm='random'):
    optimizerP = None

    # --- 核心修改：为迁移学习设置差异化学习率 ---
    # REASON: 新的、随机初始化的分类头需要一个比预训练主干网络大得多的学习率才能有效学习。

    # 1. 识别出主干网络和分类头的参数
    backbone_params = [p for name, p in net.named_parameters() if 'cls_head' not in name and p.requires_grad]
    cls_head_params = [p for name, p in net.named_parameters() if 'cls_head' in name and p.requires_grad]

    # 2. 为不同部分创建不同的参数组
    #    让分类头的学习率是主学习率的10倍，这是一个常见的起点
    params_group = [
        {'params': backbone_params, 'lr': lr},
        {'params': cls_head_params, 'lr': lr }#* 10
    ]

    print(f"Optimizer setup: Backbone LR = {lr}, Classifier Head LR = {lr }")#* 10

    opt_kwargs = {
        "weight_decay": wd,
        "momentum": momentum
    }

    # 3. 使用新的参数组来初始化优化器
    optimizer = optim.SGD(params=params_group, **opt_kwargs)


    opt_kwargs_rl = {
        "lr": lr_dqn,
        "weight_decay": 0.001,
        "momentum": momentum
    }

    if policy_net is not None:
        if opt_choice == 'SGD':
            optimizerP = optim.SGD(
                params=filter(lambda p: p.requires_grad, policy_net.parameters()),
                **opt_kwargs_rl)
        elif opt_choice == 'RMSprop':
            optimizerP = optim.RMSprop(
                params=filter(lambda p: p.requires_grad, policy_net.parameters()),
                lr=lr_dqn)

    name = exp_name_toload if load_opt and len(exp_name_toload) > 0 else exp_name
    opt_path = os.path.join(ckpt_path, name, 'opt_' + str(snapshot))
    opt_policy_path = os.path.join(ckpt_path, name, 'opt_policy_' + str(snapshot))

    if (load_opt and len(exp_name_toload)) > 0 or (checkpointer and os.path.isfile(opt_path)):
        print('(Opt load) Loading net optimizer')
        optimizer.load_state_dict(torch.load(opt_path))

        if os.path.isfile(opt_policy_path):
            print('(Opt load) Loading policy optimizer')
            optimizerP.load_state_dict(torch.load(opt_policy_path))

    print ('Optimizers created')
    return optimizer, optimizerP



def get_logfile(ckpt_path, exp_name, checkpointer, snapshot,
                log_name='log.txt'):
    """
    HAR分类任务专用的日志初始化函数。
    """
    # 只保留分类相关的日志列
    log_columns = ['Epoch', 'Learning Rate',
                   'Train Loss', '(deprecated)',
                   'Valid Loss', 'Train Top1 Acc', 'Valid Top1 Acc']

    # 初始化最佳指标记录
    best_record = {'epoch': 0, 'val_loss': 1e10, 'top1_acc': 0.0}
    curr_epoch = 0

    log_path = os.path.join(ckpt_path, exp_name, log_name)
    if checkpointer:
        if os.path.isfile(log_path):
            print(f'(Checkpointer) Log file {log_name} already exists, appending.')
            logger = Logger(log_path, title=exp_name, resume=True)
            if 'best' in snapshot:
                curr_epoch = int(logger.resume_epoch)
            else:
                curr_epoch = logger.last_epoch
            best_record = {
                'epoch': int(logger.resume_epoch),
                'val_loss': 1e10,
                'top1_acc': float(logger.resume_acc)
            }
        else:
            print(f'(Checkpointer) Log file {log_name} did not exist before, creating.')
            logger = Logger(log_path, title=exp_name)
            logger.set_names(log_columns)
    else:
        print(f'(No checkpointer activated) Log file {log_name} created.')
        logger = Logger(log_path, title=exp_name)
        logger.set_names(log_columns)

    return logger, best_record, curr_epoch



def get_training_stage(args):
    path = os.path.join(args.ckpt_path, args.exp_name,
                        'training_stage.pkl')
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            stage = pickle.load(f)
    else:
        stage = None
    return stage


def set_training_stage(args, stage):
    path = os.path.join(args.ckpt_path, args.exp_name,
                        'training_stage.pkl')
    with open(path, 'wb') as f:
        pickle.dump(stage, f)


def evaluate(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    acc = correct / total
    return acc


def train(train_loader, net, criterion, optimizer):
    net.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels, idx in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        logits = net(inputs)  # (N, num_classes)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, acc

def validate(val_loader, model, criterion, best_record, epoch, args):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, idx in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(val_loader)

    if acc > best_record['acc']:
        best_record['acc'] = acc
        best_record['val_loss'] = avg_loss
        best_record['epoch'] = epoch

        torch.save(model.state_dict(), os.path.join(args.ckpt_path, args.exp_name, 'best_acc.pth'))
        print(f'[BEST] val_acc: {acc:.4f} at epoch {epoch}')

    torch.save(model.state_dict(), os.path.join(args.ckpt_path, args.exp_name, 'last_acc.pth'))
    return avg_loss, acc, best_record

def test(val_loader, model, criterion):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels, idx) in enumerate(val_loader):
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)  # logits
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress_bar(i, len(val_loader), '[val loss %.5f]' % (total_loss / (i + 1)))

    acc = correct / total
    avg_loss = total_loss / len(val_loader)

    print(' ')
    print(' [val acc %.5f]' % acc)
    return avg_loss, acc




def final_test(args, model, val_loader, criterion):
    ckpt_path = os.path.join(args.ckpt_path, args.exp_name, 'best_acc.pth')
    if os.path.isfile(ckpt_path):
        print(f'[TEST] Loading best checkpoint from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path))

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, idx in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(val_loader)

    print(f'[Final Test] Accuracy: {acc:.4f}, Loss: {avg_loss:.4f}')
    with open(os.path.join(args.ckpt_path, args.exp_name, 'test_results.txt'), 'a') as f:
        f.write(f'{avg_loss:.4f},{acc:.4f}\n')

    return avg_loss, acc
