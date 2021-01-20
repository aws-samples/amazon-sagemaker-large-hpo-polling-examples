import argparse
import json
import logging
import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.autograd.profiler as profiler
import glob
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

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
       # cols_to_read = ['OrInterestRate', 'OrUnpaidPrinc', 'OrLoanTerm', 'OrLTV', 'NumBorrow',
       #'DTIRat', 'CreditScore', 'NumUnits', 'Zip', 'LoanAge',
       #'MonthsToMaturity', 'Default']
        data = pd.read_csv(self.paths[idx])
        X = data.drop(columns=[self.target], axis=1)
        y = data[self.target]
        return X.values, y.values

def _get_train_data(train_batch_size, dataset, is_distributed,**kwargs):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    return DataLoader(dataset, batch_size = train_batch_size, shuffle=train_sampler is None,
                      sampler=train_sampler, **kwargs)

def _get_test_data(test_batch_size, path_to_test_dataset):
    dataset = LoansPredictionDataset(path_to_test_dataset)
    return DataLoader(dataset, batch_size = test_batch_size, shuffle=True)


class Net_ray(nn.Module):
    def __init__(self,inp_dimension, l1=200, l2=70):
        super().__init__()
        self.fc1 = nn.Linear(inp_dimension, l1)
        self.drop = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(l1)
        self.bn2=nn.BatchNorm1d(l2)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2,2)
        


    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc1(x.float()))
        x = self.drop(x)
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.bn2(x)
        x = self.fc3(x) # last layer converts it to binary classification probability
        return F.log_softmax(x, dim=1)
    
def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
        
def train_ray(config, checkpoint_dir=None, args=None):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda:0" if use_cuda else "cpu")
 
    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    train_dataset = LoansPredictionDataset(args.data_dir)
    train_loader = _get_train_data(args.batch_size, train_dataset, is_distributed, **kwargs)
    
    test_loader = _get_test_data(args.test_batch_size, args.test_dir)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    model = Net_ray(args.inp_dimension, config["l1"], config["l2"]).float() # specify input dimension of training dataset
    
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model).to(device)
        print("Distributed data parallel")
    else:
        # single-machine multi-gpu case or single-machine or single machine cpu case
        model = torch.nn.DataParallel(model).to(device)
        print("Picked data parallel")

    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target.squeeze())
            loss.backward()
     
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info("Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.item()))
        test(model, test_loader, device)
        with tune.checkpoint_dir(epoch) as checkpoint_dir: # modified to store checkpoint after every epoch.
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
    


def test(model, test_loader, device):
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
            test_loss += F.nll_loss(output, target).item()  # sum up batch loss
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
    f1score= f1_score(fulltargets, fullpreds)
    roc=roc_auc_score(fulltargets, fulloutputs)
    logger.info("Test set F1-score: {:.4f}, Test set AUC: {:.4f} \n".format(
        f1score, roc))
    
    tune.report(loss=test_loss, accuracy=correct / (len(target)*i), f1score=f1score, roc=roc)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    
def main(args):
    config = {
        "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1)
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss","training_iteration", "roc"])
    
    
    
    # run the HPO job by calling train
    print("Starting training ....")
    result = tune.run(
        partial(train_ray, args=args),
        resources_per_trial={"cpu": args.num_cpus, "gpu": args.num_gpus},
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Net_ray(args.inp_dimension, best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if args.num_gpus > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_loader = _get_test_data(args.test_batch_size, args.test_dir)
    test_acc = test(best_trained_model, test_loader, device)
    print("Best trial test set accuracy: {}".format(test_acc))
    save_model(best_trained_model, args.model_dir)
    
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
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    parser.add_argument('--inp-dimension', type=int, default=29,
                        help='input dimension (number of columns) in training dataset')
    parser.add_argument('--l1', type=int, default=200,
                        help='size of hidden layer 1')
    parser.add_argument('--l2', type=int, default=70,
                        help='size of hidden layer 2')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='number of jobs to run')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_TESTING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--num-cpus', type=int, default=os.environ['SM_NUM_CPUS'])

 
    main(parser.parse_args())
