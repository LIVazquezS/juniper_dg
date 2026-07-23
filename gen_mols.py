import torch
from lightning.pytorch import Trainer
import pandas as pd
# # Local model
from diff_module import DiscDenDiff
from metrics.validation_molecular_metrics import SamplingMolecularMetrics
from databases.get_infos import Datasetinfos
from databases.data_manage_simple_label import DataManagment
from types import SimpleNamespace

from utils.rdkit_functions import compute_molecular_metrics

# Some options for torch
torch.set_float32_matmul_precision('high')
#Some parameters
name = 'ud_dgwo_cft_dpf'
# NN parameters
hidden_mlp_dims= {'X': 256, 'E': 128, 'y': 128}
# The dimensions should satisfy dx % n_head == 0
hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8,
               'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128} #last value replace by 128
n_layers = 5
nn_params = {'n_layers': n_layers, 'hidden_dims': hidden_dims,'hidden_mlp_dims': hidden_mlp_dims}
nn_params = SimpleNamespace(**nn_params)
#Guidance parameters
guidance = True
dropout_fix = True
s = 2.0 #Conditional weight
# p = [0,0.2,1.0] #Dropout
p = 0.2
guidance_in = 'y'
trainable_cf = True
guidance_parms = {'guidance_in':guidance_in,'s':s,'dropout_fix':dropout_fix,'p_dropout':p,'trainable_cf':trainable_cf,}
guidance_params = SimpleNamespace(**guidance_parms)
# Training parameters
train_lr = 0.0001
train_weight_decay = 1e-9
log_every_steps = 5
save_model = True
ema_decay = 0
clip_grad = None
n_epochs = 1000
lambda_train= [5,1]
training_params = {'train_lr':train_lr, 'train_weight_decay':train_weight_decay,
                   'log_every_steps':log_every_steps,'save_model':save_model,
                   'ema_decay':ema_decay,'clip_grad':clip_grad,
                   'n_epochs':n_epochs,'lambda_train':lambda_train}
train_params = SimpleNamespace(**training_params)
#Diffusion parameters
diffusion_noise_schedule = 'cosine'
diffusion_steps = 500
number_chain_steps = 5
visualization = False

diffusion_params = {'diffusion_steps':diffusion_steps, 'diffusion_noise_schedule':diffusion_noise_schedule,
                    'number_chain_steps':number_chain_steps, 'visualization':visualization}
diffusion_params = SimpleNamespace(**diffusion_params)

# Validation parameters
check_val_every_n_epochs = 5
sample_every_val = 1
samples_to_generate = 100
samples_to_save = 50
chains_to_save = 1
val_param = {'sample_every_val': sample_every_val, 'samples_to_generate': samples_to_generate,
             'samples_to_save': samples_to_save, 'chains_to_save': chains_to_save}
val_param = SimpleNamespace(**val_param)

# Test parameters
final_model_samples_to_generate = 10
final_model_samples_to_save = 10
final_model_chains_to_save = 10
test_param = {'samples_to_generate': final_model_samples_to_generate,
                'samples_to_save': final_model_samples_to_save,
                'chains_to_save': final_model_chains_to_save}
test_param = SimpleNamespace(**test_param)

#Dataset

data_source = 'databases/uni_dim.csv'
# Data managment
data_mols = DataManagment('uni_dim_single',data_source, 0.9,
                   0.1, 512, seed=42, filter_dataset=True)
dm = data_mols.get_dataset()

# Dataset infos
dataset_infos = Datasetinfos(dm, recompute_statistics=True,file_dim='dimensions_single.json')
df_train_smiles = pd.read_csv('new_train_uni_dim_single.smiles',names=['smiles'])
train_smiles = df_train_smiles['smiles'].values

# Compute input output dims
dataset_infos.ensure_dims(datamodule=dm, guidance=True,guidance_size=1)

# Sampling metrics
sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)

# Model start
model = DiscDenDiff_Single(name,
                 dataset_infos,
                 nn_params,
                 train_params,
                 diffusion_params,
                 None,
                 sampling_metrics,
                 val_param,
                 test_param,
                 guidance=guidance,
                 guidance_params=guidance_params,)


ckpt = None
#Training
use_gpu = True
trainer = Trainer(gradient_clip_val=clip_grad,
                  strategy = "ddp_find_unused_parameters_true",  # Needed to load old
                  accelerator='gpu' if use_gpu else 'cpu',
                  devices=1,
                  max_epochs=n_epochs,
                  check_val_every_n_epoch=check_val_every_n_epochs,
                  fast_dev_run=False,
                  enable_progress_bar=True,
                  log_every_n_steps=50,
                  logger = [])

def create_natoms_tensor(file,n_samples=10):
    df_dist = pd.read_csv(file)
    natoms = df_dist['n_atoms'].tolist()
    probs = df_dist['perc'].tolist()

    counts = [int(n_samples*i) for i in probs]
    counts[-1] = n_samples - sum(counts[:-1])

    # Build tensor with exact counts
    elements = []
    for label, count in zip(natoms, counts):
        elements.extend([label] * count)
    n_nods = torch.tensor(elements)
    dist_nodes = n_nods[torch.randperm(n_samples)]
    return dist_nodes

def write_mols(all_smiles,name_file):
    file_mols = name_file + '.txt'
    with open(file_mols, 'w') as f:
        for i in all_smiles:
            if i is not None:
                f.write(i + '\n')
            else:
                f.write('Fail' + '\n')


def get_mols(file_out,file_cktp,guidance,dist_natoms=None,bs=100,kc=5,nc_steps=5):
    '''
    file_out: Label of the files to be saved
    file_ckpt: Pytorch checkpoint
    guidance: value of DG
    bs (optional): batch size (Number of molecules to generate)
    kc (optional) : keep chain (How many chains are saved)
    nc_steps (optional): number of chain steps
    '''
    ckpt = torch.load(file_cktp)
    model.load_state_dict(ckpt['state_dict'])
    model._trainer = trainer

    print('The checkpoint used is:{}'.format(file_cktp))

    if dist_natoms is not None:
        if isinstance(dist_natoms, str):
            dist_nodes = create_natoms_tensor(dist_natoms, n_samples=bs)
        elif isinstance(dist_natoms, int):
            dist_nodes = dist_natoms
    else:
        dist_nodes = None

    guidance_val = torch.tensor([guidance])
    dist_nodes = create_natoms_tensor(dist_natoms, n_samples=bs)
    mols = model.sample_batch(bs,kc,nc_steps,dist_nodes,guidance_val)

    dct_metrics,unique,novel,all_smiles = compute_molecular_metrics(mols, train_smiles, dataset_infos,testing=True)

    df_metrics = pd.DataFrame(dct_metrics, index=[0])


    n_nodes_name = 'const'

    name_metrics = file_out + '_' + n_nodes_name +'.csv'

    df_metrics.to_csv(name_metrics,index=False)

    name_unique = file_out +  '_' + n_nodes_name + '_unique'
    write_mols(unique, name_unique)

    name_novel = file_out  +  '_' + n_nodes_name + '_novel'
    write_mols(novel, name_novel)

    name_all = file_out +  '_' + n_nodes_name + '_all'
    write_mols(all_smiles, name_all)

    return None




ckpt_model = 'checkpoints/ud_dgwo_cft_dpf/last.ckpt'
n_mol_to_gen = 100

file_out = 'Gen_dim_by_dg_s2/mol_dim_s2_P5_C2'
get_mols(file_out,ckpt_model,guidance_val,'n_atoms_dim.csv',bs=n_mol_to_gen)
