import os
import torch
import numpy as np
from tqdm import tqdm
from opts import *
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, save_model, setup_seed
from tensorboardX import SummaryWriter
from models.almt import build_model
from core.metric import MetricsTop


opt = parse_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_VISIBLE_DEVICES
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("device: {}:{}".format(device, opt.CUDA_VISIBLE_DEVICES))


train_mae, val_mae = [], []


def main():
    opt = parse_opts()
    if opt.seed is not None:
        setup_seed(opt.seed)
    print("seed: {}".format(opt.seed))
    
    log_path = os.path.join(".", "log", opt.project_name)
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    print("log_path :", log_path)

    save_path = os.path.join(opt.models_save_root,  opt.project_name)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    print("model_save_path :", save_path)

    model = build_model(opt).to(device)

    dataLoader = MMDataLoader(opt)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

    scheduler_warmup = get_scheduler(optimizer, opt)

    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)
    writer = SummaryWriter(logdir=log_path)


    for epoch in range(1, opt.n_epochs+1):
        train(model, dataLoader['train'], optimizer, loss_fn, epoch, writer, metrics)
        evaluate(model, dataLoader['test'], optimizer, loss_fn, epoch, writer, save_path, metrics, opt)
        scheduler_warmup.step()
    writer.close()


def train(model, train_loader, optimizer, loss_fn, epoch, writer, metrics):
    train_pbar = tqdm(enumerate(train_loader))
    losses = AverageMeter()

    y_pred, y_true = [], []

    model.train()
    for cur_iter, data in train_pbar:
        img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        cls_output = model(img, audio, text)

        loss = loss_fn(cls_output, label)

        losses.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(cls_output.cpu())
        y_true.append(label.cpu())

        train_pbar.set_description('train')
        train_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                'loss': '{:.5f}'.format(losses.value_avg),
                                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)
    print('train: ', train_results)
    train_mae.append(train_results['MAE'])

    writer.add_scalar('train/loss', losses.value_avg, epoch)



def evaluate(model, eval_loader, optimizer, loss_fn, epoch, writer, save_path, metrics, opt):
    test_pbar = tqdm(enumerate(eval_loader))

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            cls_output = model(img, audio, text)

            loss = loss_fn(cls_output, label)

            y_pred.append(cls_output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                   'loss': '{:.5f}'.format(losses.value_avg),
                                   'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print(test_results)

        save_model(save_path, epoch, model, optimizer)


if __name__ == '__main__':
    main()
