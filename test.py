import os
import torch
from tqdm import tqdm
from opts import *
from core.dataset import MMDataLoader
from core.utils import AverageMeter
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

    model = build_model(opt).to(device)
    model.load_state_dict(torch.load(opt.test_checkpoint)['state_dict'])

    dataLoader = MMDataLoader(opt)

    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)


    test(model, dataLoader['test'], loss_fn, metrics)


def test(model, test_loader, loss_fn, metrics):
    test_pbar = tqdm(enumerate(test_loader))

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(img, audio, text)

            loss = loss_fn(output, label)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({'loss': '{:.5f}'.format(losses.value_avg)})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print(test_results)

if __name__ == '__main__':
    main()
