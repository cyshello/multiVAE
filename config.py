import torch

################# HOW TO USE #######################
# only edit the global variables below to control hyperparameters.

# to edit model architecture, see model.py

# to add experiment, see main.py

################# GLOBAL VARIABLES #################
latent_dims = 20
num_epochs = 100
batch_size = 128
capacity = 128
learning_rate = 1e-3
variational_beta = 1
use_gpu = True
overfit_num = 3000 # number of datasets that overfit models will use. baseline model will use the remaining datasets.

datapath = "./data/"
modelpath = "./models/"
exppath = "./experiments/"

EXP_names = []

train_dataset = None # raw dataset of all MNIST train data
train_dataloader = None # dataloader of all MNIST train data
test_dataset = None # raw dataset of all MNIST test data
test_dataloader = None # dataloader of all MNIST test data

test_digit_datasets = {} # dic of raw test datasets divided into digits
test_digit_dataloaders = {} # dictionary of test dataloaders divided into digits

train_dataloader_digits = None # dictionary of dataloader that overfit models will use
train_dataloader_baseline = None # remaining dataloader that baseline model will use

task_vectors_digits = {} # task vectors of each digits

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

################# END OF GLOBALS #################
