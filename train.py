import os
import torch
import argparse
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, results_recorder, dict_to_namespace
from tensorboardX import SummaryWriter
from models.almt import build_model
from core.metric import MetricsTop
import yaml


parser = argparse.ArgumentParser() 
parser.add_argument('--config_file', type=str, default='configs/sims.yaml') 
parser.add_argument('--seed', type=int, default=-1) 
parser.add_argument('--gpu_id', type=int, default=-1) 
opt = parser.parse_args()
print(opt)

with open(opt.config_file) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
args = dict_to_namespace(args)
print(args)

seed = args.base.seed if opt.seed == -1 else opt.seed
gpu_id = args.base.gpu_id if opt.gpu_id == -1 else opt.gpu_id

print('-----------------args-----------------')
print(args)
print('-------------------------------------')

gpu_id = str(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Device: {device} ({gpu_id})")


def main():
    setup_seed(seed)
    log_path = os.path.join(".", "log", args.base.project_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    save_path = os.path.join(args.base.ckpt_root, args.base.project_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = build_model(args).to(device)

    dataLoader = MMDataLoader(args)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.base.lr,
                                 weight_decay=args.base.weight_decay)

    scheduler_warmup = get_scheduler(optimizer, args)

    loss_fn = torch.nn.MSELoss()

    metrics_fn = MetricsTop().getMetics(args.dataset.datasetName)

    training_results_recorder = results_recorder()
    validation_results_recorder = results_recorder()
    test_results_recorder = results_recorder()

    writer = SummaryWriter(logdir=log_path)


    for epoch in range(1, args.base.n_epochs+1):
        training_ret = train(model, dataLoader['train'], optimizer, loss_fn, metrics_fn)
        validation_ret = evaluate(model, dataLoader['valid'], loss_fn, metrics_fn)
        test_ret = evaluate(model, dataLoader['test'], loss_fn, metrics_fn)

        training_results_recorder.update(training_ret['results'], epoch)
        validation_results_recorder.update(validation_ret['results'], epoch)
        test_results_recorder.update(test_ret['results'], epoch)
        best_validation_results = validation_results_recorder.get_best_results()
        best_test_results = test_results_recorder.get_best_results()

        print(f'\n----------------- Results Epoch {epoch} -----------------')
        print(f'Learning Rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        print(f'Training Results: {training_ret["results"]}')
        print(f'Validation Results: {validation_ret["results"]}')
        print(f'Test Results: {test_ret["results"]}\n')
        print(f'Best Validation Results across All Epochs: {best_validation_results["best_results_all_epochs"]}')
        print(f'Best Validation Results of One Epochs: {best_validation_results["best_results_one_epoch"]}\n')
        print(f'Best Test Results across All Epochs: {best_test_results["best_results_all_epochs"]}')
        print(f'Best Test Results of One Epochs: {best_test_results["best_results_one_epoch"]}')
        print('----------------------------------------------------------\n')

        writer.add_scalar('train/MAE', training_ret['loss_recorder'].value_avg, epoch)
        writer.add_scalar('valid/MAE', validation_ret['loss_recorder'].value_avg, epoch)
        writer.add_scalar('test/MAE', test_ret['loss_recorder'].value_avg, epoch)

        scheduler_warmup.step()

    # with open(f'./{args.dataset.datasetName}_results_all_epoch.txt', 'a+') as f:
    #     f.write(f'{seed}: {best_test_results["best_results_all_epochs"]}\n')
    
    # with open(f'./{args.dataset.datasetName}_results_one_epoch.txt', 'a+') as f:
    #     f.write(f'{seed}: {best_test_results["best_results_one_epoch"]}\n')

    writer.close()


def train(model, data_loader, optimizer, loss_fn, metrics_fn):
    loss_recorder = AverageMeter()

    y_pred, y_true = [], []

    model.train()
    for cur_iter, data in enumerate(data_loader):
        img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        output = model(img, audio, text)

        loss = loss_fn(output, label)

        loss_recorder.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics_fn(pred, true)

    return {'results': results, 'loss_recorder': loss_recorder} 


def evaluate(model, data_loader, loss_fn, metrics_fn):
    loss_recorder = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    
    for cur_iter, data in enumerate(data_loader):
        img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        with torch.no_grad():
            output = model(img, audio, text)

        loss = loss_fn(output, label)

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        loss_recorder.update(loss.item(), batchsize)

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics_fn(pred, true)

    return {'results': results, 'loss_recorder': loss_recorder} 

if __name__ == '__main__':
    main()
