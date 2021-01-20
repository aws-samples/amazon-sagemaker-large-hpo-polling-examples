import argparse
import glob
import json
import logging
import os
import sys

import pandas as pd
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import smdistributed.dataparallel.torch.distributed as dist
from smdistributed.dataparallel.torch.parallel.distributed import \
    DistributedDataParallel as DDP

dist.init_process_group()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class LoansPredictionDataset(Dataset):
    """LoanPredictionDataset."""

    def __init__(self, root_dir):
        """Initializes instance of class LoanPredictionDataset.

        Args:
            csv_file (str): Path to the csv files with the loan data.

        """
        self.paths =  glob.glob(root_dir + "/*.csv")
        self.target = 'Default'
        # Grouping variable names

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = pd.read_csv(self.paths[idx])
        X = data.drop(columns=[self.target], axis=1)
        y = data[self.target]
        return X.values, y.values

def _get_train_data(train_batch_size, dataset, train_sampler, **kwargs):
    return DataLoader(dataset, batch_size = train_batch_size, shuffle=False,
                      sampler=train_sampler, **kwargs)

def _get_test_data(test_batch_size, path_to_test_dataset):
    dataset = LoansPredictionDataset(path_to_test_dataset)
    return DataLoader(dataset, batch_size = test_batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self, inp_dimension):
        super().__init__()
        self.fc1 = nn.Linear(inp_dimension, 500)
        self.drop = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(500)
        self.bn2=nn.BatchNorm1d(250)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 100)
        self.bn3=nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100,2)
        


    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc1(x.float()))
        x = self.drop(x)
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = self.bn3(x)
        x = self.fc4(x)
        # last layer converts it to binary classification probability
        return F.log_softmax(x, dim=1)

        
def train(args):
    use_cuda = True
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    
    if not torch.cuda.is_available():
        raise Exception("Must run SMDataParallel on CUDA-capable devices.")
        
    device = torch.device("cuda" if use_cuda else "cpu")
    
 
    # Initialize the distributed environment.
    os.environ['WORLD_SIZE'] = str(args.world_size)

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        
    if args.verbose:
        print('Hello from rank', rank, 'in world size of', args.world_size)
    
    train_dataset = LoansPredictionDataset(args.data_dir)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank)

    train_loader = _get_train_data(args.batch_size, train_dataset, train_sampler, **kwargs)


    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    
    model = DDP(Net(args.inp_dimension).to(device), broadcast_buffers=False) # specify input dimension of training dataset
    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)
    # Use SMDataParallel PyTorch DDP for efficient distributed training    
    loss_optim = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_optim(output, target.squeeze())
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0 and args.rank==0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Train Loss: {:.6f};'.format(
                    epoch, batch_idx * len(data) * args.world_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
#        torch.cuda.empty_cache()
        if args.rank==0:
            test_loader = _get_test_data(args.test_batch_size, args.test_dir)
            logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
                    len(test_loader.sampler), len(test_loader.dataset),
                    100. * len(test_loader.sampler) / len(test_loader.dataset)))
            test(model, test_loader, device, loss_optim)
    if args.rank ==0:
        print("saving the model...")
        save_model(model, args.model_dir)
                

def test(model, test_loader, device, loss_optim):
    model.eval()
    test_loss = 0
    correct = 0
    fulloutputs = []
    fulltargets = []
    fullpreds = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = target.squeeze()
            test_loss += loss_optim(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            fulloutputs.append(output.cpu().numpy()[:, 1])
            fulltargets.append(target.cpu().numpy())
            fullpreds.append(pred.squeeze().cpu())

    i+=1
    test_loss /= i
    logger.info("Test set Average loss: {:.4f}, Test Accuracy: {:.0f}%;\n".format(
            test_loss, 100. * correct / (len(target)*i)
        ))
    fulloutputs = [item for sublist in fulloutputs for item in sublist]
    fulltargets = [item for sublist in fulltargets for item in sublist]
    fullpreds = [item for sublist in fullpreds for item in sublist]
    logger.info('Test set F1-score: {:.4f}, Test set AUC: {:.4f} \n'.format(
        f1_score(fulltargets, fullpreds), roc_auc_score(fulltargets, fulloutputs)))

def save_model(model, model_dir):
    with open(os.path.join(model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.module.state_dict(), f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    parser.add_argument('--inp-dimension', type=int, default=23,
                        help='input dimension (number of columns) in training dataset')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='For displaying SMDataParallel-specific logs')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_TESTING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    # start training
    args = parser.parse_args()
    args.world_size = dist.get_world_size()
    args.rank = rank = dist.get_rank()
    args.local_rank = local_rank = dist.get_local_rank()
    
    print("Starting training...")
    train(args)
