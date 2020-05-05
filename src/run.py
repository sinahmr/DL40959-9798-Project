from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import *
from src.loader import train_loader, train_dataset, val_loader, val_dataset, test_loader
from src.models import Combined
from src.utils import cud

commands = {
    'save_after_epoch': False,
    'lr': LEARNING_RATE,
    'lr_changed': False,
}


def normalize_loss(loss, is_train=True):
    if is_train:
        return loss / len(train_dataset) * 10000
    else:
        return loss / len(val_dataset) * 10000


def check_runtime_commands():
    with open('../commands.txt', 'r') as file:
        lr = float(file.readline().split()[1])
        if lr != commands['lr']:
            commands['lr_changed'] = True
            commands['lr'] = lr

        commands['save_after_epoch'] = True if int(file.readline().split()[1]) else False


def get_validation_loss(model, criterion):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = cud(CUDA, data, target)
        with torch.no_grad():
            logits = model(data, target).permute(0, 2, 1)  # batch, C (vocab_size), seq_len
            loss = criterion(logits, target)
            total_loss += loss.item()
    return total_loss


def do_on_val(model, opt, criterion):  # TODO
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = cud(CUDA, data, target)
        opt.zero_grad()

        logits = model(data, target).permute(0, 2, 1)  # batch, C (vocab_size), seq_len
        loss = criterion(logits[:, :, :-1], target[:, 1:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()


def train(model, opt, criterion):
    model.train()
    for epoch in range(N_EPOCH):
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = cud(CUDA, data, target)
            opt.zero_grad()

            logits = model(data, target).permute(0, 2, 1)  # batch, C (vocab_size), seq_len
            loss = criterion(logits[:, :, :-1], target[:, 1:])  # first column of target is <SOS>, we should shift it, we also can ignore logits last column, it should probably be pad (<EOS>)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()

            epoch_loss += loss.item()

        normalized_epoch_loss = normalize_loss(epoch_loss, is_train=True)
        # normalized_val_loss = normalize_loss(get_validation_loss(model, criterion), is_train=False)
        normalized_val_loss = -1
        do_on_val(model, opt, criterion)

        check_runtime_commands()
        if commands['save_after_epoch']:
            torch.save(model.state_dict(), '../save/%d_epoch#%d_%d_%d.pt' % (int(time()), epoch, int(normalized_epoch_loss), int(normalized_val_loss)))
        if commands['lr_changed']:
            commands['lr_changed'] = False
            for g in opt.param_groups:
                g['lr'] = commands['lr']

        print('#%d\t train_epoch_loss:\t%.6f' % (epoch, normalized_epoch_loss))
        if epoch % RUN_VAL_INTERVAL == 0:
            print('*\t validation_loss:\t%.6f' % normalized_val_loss)

    torch.save(model.state_dict(), '../save/%d_final.pt' % int(time()))


def test(model):
    mapping = np.array(train_dataset.index_to_token)
    model.eval()
    with torch.no_grad():
        for batch_id, data in enumerate(test_loader):
            data = cud(CUDA, data)
            generated_seqs = model(data).int().cpu().numpy()
            with open('result.txt', 'a') as file:
                for line in mapping[generated_seqs]:
                    if '<EOS>' in line:
                        first_eos = np.where(line == '<EOS>')[0][0]
                        file.write(' '.join(line[1:first_eos]) + '\n')
                    else:
                        file.write(' '.join(line[1:]) + '\n')


def main():
    vocab_size = train_dataset.vocab_size  # 564
    max_seq_len = train_dataset.max_seq_len  # 178

    model = Combined(vocab_size, max_seq_len)
    # model.load_state_dict(torch.load('../save/1562957759_epoch#3_19_-1.pt', map_location=DEVICE))
    model = cud(CUDA, model)

    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    print('Training is started')
    train(model, opt, criterion)
    print('Testing is started')
    test(model)


main()
