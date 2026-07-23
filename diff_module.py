# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
# General
import time
import os
import sys

sys.path.append('..')
import warnings
# Local modules
#Model
from models.transformer_model import GraphTransformer
#Diffusion
from diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
#Metrics
from metrics.train_metrics_mixed import TrainLoss_Single
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from torchmetrics import MeanSquaredError
#Utils
from utils import utils, diffusion_utils
from utils.visualization import MolecularVisualization
# Filter error
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

class DiscDenDiff(pl.LightningModule):

    def __init__(self,name,
                 dataset_infos,
                 nn_params,
                 train_params,
                 diffusion_params,
                 train_metrics,
                 sampling_metrics,
                 val_param,
                 test_param,
                 guidance=False,
                 guidance_params=None,):
        '''

        Discrete Denoise Diffusion Model with guidance.

        This only operates over the dg_wo_cg per each molecule.

        Note: This is an adaptation of digress that is a code for discrete
        diffusion models. The original code is available at

         https://github.com/cvignac/digress
        '''

        super().__init__()

        # General parameters
        self.name = name
        self.model_dtype = torch.float32
        self.train_lr = train_params.train_lr
        self.train_weight_decay = train_params.train_weight_decay
        #Diffusion parameters
        self.T = diffusion_params.diffusion_steps
        # Reading dataset information
        self.dataset_info = dataset_infos
        self.input_dims = self.dataset_info.input_dims
        self.output_dims = self.dataset_info.output_dims
        self.nodes_dist = self.dataset_info.nodes_dist
        self.atom_decoder = self.dataset_info.atom_decoder

        # Dimensions of input of edges (X), nodes (E) and y (latent space)
        self.Xdim = self.input_dims['X']
        self.Edim = self.input_dims['E']
        self.ydim = self.input_dims['y'] + 1
        # Rewrote the parameter of input:
        self.input_dims['y'] = self.ydim

        # Dimensions of input of edges (X), nodes (E) and y
        self.Xdim_output = self.output_dims['X']
        self.Edim_output = self.output_dims['E']
        self.ydim_output = self.output_dims['y']

        # Guidance
        self.guidance = guidance
        self.guidance_params = guidance_params
        if self.guidance and self.guidance_params is not None:
           self.trainable_cf = self.guidance_params.trainable_cf
           #self.p_uncond = p_uncond
           self.dropout_fix = self.guidance_params.dropout_fix
           if not self.dropout_fix:
               assert isinstance(self.guidance_params.p_dropout,
                                 (list)), "The p_dropout parameter must be a list of [p_min,p_max, scale]"
               self.p_uncond_min = float(self.guidance_params.p_dropout[0])
               self.p_uncond_max = float(self.guidance_params.p_dropout[1])
               self.p_uncond_alpha = float(self.guidance_params.p_dropout[2])
           else:
               assert isinstance(self.guidance_params.p_dropout,float), ("The p_dropout parameter is fix."
                                                                         "Therefore it must be float")
               self.p_dropout = self.guidance_params.p_dropout
           # Probability for unconditional generation
           self.s = self.guidance_params.s # Scaling for the guidance vector
           self.guidance_in = self.guidance_params.guidance_in # This determines if the guidance is the value y or in the edges and nodes. Default is y
           self.ydim = self.input_dims['guidance'] + 1
           self.ydim_output = self.input_dims['guidance']


           # Guidance
           if (self.input_dims['guidance'] != None):
               # self.gdim = self.input_dims['guidance']
               self.gdim = self.input_dims['guidance']  # Dimension of the hidden representation
           else:
               self.gdim = 0

           # Null token required to calculate cases without guidance
           if (self.guidance_params.trainable_cf == True):
               self.cf_null_token = torch.nn.parameter.Parameter(torch.randn(size=(1, self.gdim)))
           else:
               self.cf_null_token = torch.zeros(size=(1, self.gdim))

        elif self.guidance and self.guidance_params is None:
            print('Guidance is activated but not parameters were passed')
            sys.exit('Add parameters for guidance')
        elif not self.guidance and self.guidance_params is not None:
            print('Guidance is deactivated but parameters were passed.')
            sys.exit('Activate guidance if you want to run a guided train')
        else:
            print('You are running a training without guidance. \n' 
                  'If this is not what you want, restart and check \n'
                  'your parameters.')
            self.ydim = None
            self.ydim_output = None


        self.train_loss =TrainLoss_Single(train_params.lambda_train,train_guidance=self.guidance)

        # Metrics Validation
        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_dgwo = MeanSquaredError()

        # Metrics Test
        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        # Metrics Train
        self.train_metrics = train_metrics
        # Metrics Sampling
        self.sampling_metrics = sampling_metrics
        #Parameters for validation and testing
        self.val_param = val_param
        self.test_param = test_param

        # Validation of the generated molecules
        self.cond_val = NLL()

        #Visualization
        self.visualization = False
        if diffusion_params.visualization:
            self.visualization = True
            self.visualization_tools = MolecularVisualization(remove_h=False,
                                                            dataset_infos=self.dataset_info)


        # Model: Graph Transformer
        self.model = GraphTransformer(n_layers=nn_params.n_layers,
                                       input_dims=self.input_dims,
                                       hidden_mlp_dims=nn_params.hidden_mlp_dims,
                                       hidden_dims=nn_params.hidden_dims,
                                       output_dims=self.output_dims,
                                       act_fn_in=nn.ReLU(),
                                       act_fn_out=nn.ReLU())

        # Noise schedule for the diffusion
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(diffusion_params.diffusion_noise_schedule,
                                                              timesteps=self.T)

        # Noise transition. The matrices to compute the probability of jumping between states
        # Note: We use the marginal transition.
        node_types = self.dataset_info.node_types.float()
        x_marginals = node_types / torch.sum(node_types)

        edge_types = self.dataset_info.edge_types.float()
        e_marginals = edge_types / torch.sum(edge_types)

        print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
        self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                          y_classes=self.ydim_output)
        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                            y=torch.ones(self.ydim_output) / self.ydim_output)

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = train_params.log_every_steps
        self.number_chain_steps = diffusion_params.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        # self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])

###################################################
    # Optimizer and forward pass
    def configure_optimizers(self):
        # Optimizer
        return torch.optim.AdamW(self.model.parameters(), lr=self.train_lr, amsgrad=True,
                                     weight_decay=self.train_weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        # return {'optimizer':optimizer, 'lr_scheduler':lr_scheduler}


    def get_new_XEy(self,noisy_data,extra_data,guidance=None):
        '''Get the new X, E and y data for the model'''
        # First clone the noise values
        # Following is a note from the original code.
        # X      = transformer output which uses guidance (if any)
        # X_null = transformer output which uses the null token
        X = noisy_data['X_t'].clone().float()
        E = noisy_data['E_t'].clone().float()
        y = noisy_data['y_t'].clone().float()

        if (guidance != None):
            # We add the guidance vector to the X and E data.
            bs = extra_data.X.size(0)
            n = extra_data.X.size(1)
            if self.guidance_in in ['y','all']:
                if len(guidance.shape) == 1:
                    g_y = guidance.unsqueeze(-1)
                else:
                    g_y = guidance
                y =  torch.hstack((y, g_y)).float()
            elif self.guidance_in in ['XE','all']:
                #TODO: Check if this is the correct way to spread the guidance.
                g_X = torch.reshape(guidance, shape=(bs, 1, -1)).repeat((1, n, 1))
                g_E = torch.reshape(guidance, shape=(bs, 1, 1, -1)).repeat((1, n, n, 1))
                #g_X = guidance.reshape(bs,n,self.ydim_output)
                X = torch.hstack((X, g_X)).float()
                #g_E = guidance.reshape(bs,n,n,self.ydim_output)
                E = torch.hstack((E, g_E)).float()
                # spreads the guidance on all nodes
                # g_X must have size (bs, n, features)
                # g_E must have size (bs, n, n, features)


                # Concatenate the guidance to the X and E data
                X = torch.cat((X, g_X), dim=-1)
                E = torch.cat((E, g_E), dim=-1)
        else:
            pass
        y = torch.hstack((y, extra_data.y)).float()

        return X, E, y

    def p_drop_from_label_id(self,counts,type='mean'):
        """
        Takes the tensor of counts for the samples and returns probability
        """
        # counts_ = np.array(counts)
        counts_t = counts.to(self.device).float()
        if type == 'mean':
            ref = torch.mean(counts_t)
        elif type == 'median':
            ref = torch.median(counts_t)
        elif type == 'max':
            ref = torch.max(counts_t)
        else:
            print('No option selected. Defaulting to mean. Available options: mean, median, max')
            ref = torch.mean(counts_t)

        # Tail gets larger dropout protection: smaller counts -> larger ratio -> larger p_drop
        ratio = ref / counts_t
        p = self.p_uncond_max * (ratio ** self.p_uncond_alpha)
        return p.clamp(self.p_uncond_min, self.p_uncond_max)

    def forward_guidance(self, noisy_data,extra_data,node_mask,guidance=None,train_step=False):
        '''
        This function is used when you have guidance activated
        '''

        bs = extra_data.X.size(0)  # Batch size
        device = extra_data.X.device

        cf_null_token = self.cf_null_token.expand(bs, 1).to(device)
        # Resolve guidance encoding
        guidance_enc = cf_null_token if guidance is None else guidance

        if train_step:
            if self.dropout_fix:
                drop_mask = (
                        torch.rand(bs,1, device=device) < self.p_dropout
                ).expand(bs, self.ydim_output)
            else:
                p_drop = self.p_drop_from_label_id(guidance.bead_freq, type="mean")
                drop_mask = (
                        torch.rand(bs, device=device) < p_drop
                )[:, None].expand(bs, self.ydim_output)

            guidance_enc = torch.where(drop_mask, cf_null_token, guidance_enc.unsqueeze(-1))
        else:
            # Repeat guidance to match batch size when sampling multiple molecules
            if bs > guidance_enc.size(0):
                guidance_enc = guidance_enc.expand(bs, -1)

        X, E, y = self.get_new_XEy(noisy_data, extra_data, guidance=guidance_enc)
        out = self.model(X, E, y, node_mask)

        # Apply CFG scale at inference time
        if not train_step and self.s > 0:
            X_null, E_null, y_null = self.get_new_XEy(noisy_data, extra_data, guidance=cf_null_token)
            out_null = self.model(X_null, E_null, y_null, node_mask)
            out.X = out_null.X + self.s * (out.X - out_null.X)
            out.E = out_null.E + self.s * (out.E - out_null.E)

        return out


    def forward(self, noisy_data,extra_data,node_mask,guidance=None,train_step=False):
        '''
        Forward pass of the model.
        '''
        if self.guidance:
            out = self.forward_guidance(noisy_data, extra_data, node_mask, guidance=guidance, train_step=train_step)
            pred_form = utils.PlaceHolder(X=out.X,E=out.E,y=out.y).mask(node_mask)
            return pred_form
        else:
            X = noisy_data['X_t'].float()
            E = noisy_data['E_t'].float()
            y = torch.hstack((noisy_data['y_t'].float(), extra_data.y.float()))
            out = self.model(X, E, y, node_mask)
        return utils.PlaceHolder(X=out.X,E=out.E,y=out.y).mask(node_mask)


    ###################################################
    # Training and evaluation steps
    def training_step(self, data, i):
        '''
        Training step of the model

        :param data:
        :param i:
        :return:
        '''
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        #Convert data of edges and nodes to dense matrix.
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        # Mask the data
        dense_data = dense_data.mask(node_mask)
        # Split in nodes (X) and edges (E)
        X, E = dense_data.X, dense_data.E
        # Apply noise to the data. Data.y = 0
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        # Compute extra data (Add the time)
        extra_data = self.compute_extra_data(noisy_data)
        # Forward pass Prediction by the model
        pred = self.forward(noisy_data, extra_data, node_mask,
                                    guidance=data.guidance,train_step=True)
        # # Compute the loss
        vals_log, loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E,
                                        true_X=X, true_E=E,pred_dgwo=pred.y,true_dgwo=data.guidance,
                                        log=i % self.log_every_steps == 0)
        # Molecular metrics:
        to_log_mol_metrics = self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)

        # First write the metrics for the molecular properties.
        if to_log_mol_metrics is not None:
            for val in to_log_mol_metrics.keys():
                self.log(val, to_log_mol_metrics[val])

        # All other values. Metrics for nodes, edges, etc...
        if vals_log is not None:
            for val in vals_log.keys():
                self.log(val, vals_log[val])

        self.log("train/loss", loss)

        return {'loss': loss}

    def validation_step(self, data, i):
        #Convert data of edges and nodes to dense matrix.
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        # Mask the data
        dense_data = dense_data.mask(node_mask)
        # Split in nodes (X) and edges (E)
        X, E = dense_data.X, dense_data.E
        # Apply noise to the data. Data.y = 0
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        # Compute extra data (Add the time)
        extra_data = self.compute_extra_data(noisy_data)
        # Forward pass Prediction by the model
        pred = self.forward(noisy_data, extra_data, node_mask, guidance=data.guidance, train_step=False)
        nll = self.compute_val_loss(pred, noisy_data, X, E,data.y,guidance=data.guidance,
                                    node_mask=node_mask,test=False)

        return {'loss': nll}

    def compute_val_loss(self, pred, noisy_data, X, E,y, node_mask, guidance,test=False):
        # Validation loss function
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
                      node_mask : (bs, n)
           guidance: True data of beads
           bead_pred: (batch_size, n, total_features) Prediction of beads

           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.nodes_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask,guidance)

        log_pX = prob0.X.log()
        log_pE = prob0.E.log()

        term_x = X*log_pX
        term_E = E*log_pE
        # print('nodes',self.val_X_logp(term_x))
        # print('edges',self.val_E_logp(term_E))
        loss_term_0 = self.val_X_logp(term_x) + self.val_E_logp(term_E)

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        if torch.isnan(nlls).any():
            print('loss value is nan')
            nlls = torch.zeros(nlls.shape,device=self.device)
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Compute the MSE of prediction of dgwo
        if self.guidance:
            # Guidance is the value of reference, y is the prediction
            val_mse_dgwo = self.val_dgwo(guidance,pred.y.squeeze())
        else:
            val_mse_dgwo = None

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        self.log("val/kl prior",kl_prior.mean(),batch_size=X.shape[0],on_epoch=False,on_step=True)
        self.log("val/log_pn",log_pN.mean(),batch_size=X.shape[0],on_epoch=False,on_step=True)
        self.log("val/loss_all_t",loss_all_t.mean(),batch_size=X.shape[0],on_epoch=False,on_step=True)
        self.log("val/loss_term_0",loss_term_0.mean(),batch_size=X.shape[0],on_epoch=False,on_step=True)
        self.log("val/mse_dgwo",val_mse_dgwo,batch_size= X.shape[0],on_epoch=False,on_step=True)
        if test:
            self.log("test/batch_test_nll",nll,batch_size=X.shape[0],on_epoch=False,on_step=True)

        return nll

    def test_step(self, data, i):
        #Convert data of edges and nodes to dense matrix.
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        # Mask the data
        dense_data = dense_data.mask(node_mask)
        # Split in nodes (X) and edges (E)
        X, E = dense_data.X, dense_data.E
        # Apply noise to the data. Data.y = 0
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        # Compute extra data (Add the time)
        extra_data = self.compute_extra_data(noisy_data)
        # Forward pass Prediction by the model
        pred = self.forward(noisy_data, extra_data, node_mask, guidance=data.guidance, train_step=False)
        nll = self.compute_val_loss(pred, noisy_data, X, E,data.y,guidance=data.guidance,
                                    node_mask=node_mask,test=True)

        # Generate samples
        samples_left_to_generate = self.test_param.samples_to_generate
        samples_left_to_save = self.test_param.samples_to_save
        chains_left_to_save = self.test_param.chains_to_save

        samples = []
        id = 0
        while samples_left_to_generate > 0:
            self.print(f'Samples left to generate: {samples_left_to_generate}/'
                       f'{self.test_param.samples_to_generate}', end='')
            bs = 2 * self.dataset_info.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(to_generate,
                                             num_nodes=None,
                                             keep_chain=chains_save,
                                             number_chain_steps=self.test_param.number_chain_steps,
                                             guidance=data.guidance))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        self.print("Saving the generated graphs")
        filename = f'generated_samples_1.txt'


        for i in range(2, 10):
            if os.path.exists(filename):
                filename = f'generated_samples_{i}.txt'
            else:
                break

        with open(filename, 'w') as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")

        to_log, others = self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True,
                              local_rank=self.local_rank)
        if to_log is not None:
            for val in to_log.keys():
                self.log(val, to_log[val])
            for val1 in others:
                self.log(val1, others[val1])
        self.log('test/loss',nll)
        return {'loss': nll}



    def accuracy_test(self,samples,input_properties):
        '''Compute the accuracy of the generated samples'''
        nll_assigment = self.cond_val(samples.y, input_properties)
        print(f'Accuracy of the generated samples: {nll_assigment}')
        return nll_assigment

###################################################
    # Noise! Noise! Noise!
    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    ###########################################
    # Here start functions for the training and evaluation of the model work as callbacks

    def on_fit_start(self) -> None:
        self.print("Starting training...")
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)


    def on_train_epoch_start(self) -> None:
        if self.current_epoch % self.log_every_steps == 0:
            self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        for val in to_log:
             self.log(val, to_log[val])
        if self.start_epoch_time is None:
            self.start_epoch_time = time.time()
        self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/X_CE'] :.3f}"
                      f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} "
                      # f" -- loss_RAE: {to_log['train_epoch/rae'] :.3f}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        # epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        # epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        # self.print(f"Epoch {self.current_epoch}: ")
        # # self.print(f"{k}: {v}" for k, v in epoch_at_metrics.items())
        # for k in epoch_bond_metrics.keys():
        #     self.print(f"{k}: {epoch_bond_metrics[k]}")

    def on_validation_epoch_start(self) -> None:
        print('Starting validation...')
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()
        self.val_dgwo.reset()


    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]

        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f}")

        #Loggin of data
        self.log("val/X_kl_epoch", metrics[1],on_epoch=True,on_step=False)
        self.log("val/E_kl_epoch", metrics[2],on_epoch=True,on_step=False)
        self.log("val/X_logp_epoch", metrics[3],on_epoch=True,on_step=False)
        self.log("val/E_logp_epoch", metrics[4],on_epoch=True,on_step=False)

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True,on_epoch=True,on_step=False)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))



    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.val_dgwo.reset()


    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        self.log("test/epoch_NLL", metrics[0], sync_dist=True,on_epoch=True,on_step=False)
        self.log("test/epoch_X_kl", metrics[1], sync_dist=True,on_epoch=True,on_step=False)
        self.log("test/epoch_E_kl", metrics[2], sync_dist=True,on_epoch=True,on_step=False)
        self.log("test/epoch_X_logp", metrics[3], sync_dist=True,on_epoch=True,on_step=False)
        self.log("test/epoch_E_logp", metrics[4], sync_dist=True,on_epoch=True,on_step=False)

        self.print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
                   f"Test Edge type KL: {metrics[2] :.2f}")

        test_nll = metrics[0]
        self.print(f'Test loss: {test_nll :.4f}')

        self.print("Generated graphs Saved. Computing sampling metrics...")

        self.print("Done testing.")



###################################################
    # Here start other functions

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)
        #Avoid numerical inestability
        probX[probX == .0] = 1e-6
        probE[probE == .0] = 1e-6
        limit_dist_X[limit_dist_X == .0] = 1e-6
        limit_dist_E[limit_dist_E == .0] = 1e-6

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        #Avoid numerical inestability
        prob_true.X[prob_true.X == 0.0] = 1e-6
        prob_pred.X[prob_pred.X == 0.0] = 1e-6
        prob_true.E[prob_true.E == 0.0] = 1e-6
        prob_pred.E[prob_pred.E == 0.0] = 1e-6

        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask,guidance=None):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)

        pred0 = self.forward(noisy_data, extra_data, node_mask,guidance=guidance)
        # Normalize predictions
        # Adding a regularization term to avoid nan in validation.
        probX0 = F.softmax(pred0.X, dim=-1) + 1e-8
        probE0 = F.softmax(pred0.E, dim=-1) + 1e-8
        proby0 = F.softmax(pred0.y,dim=-1) + 1e-8


        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

#######################################################
# Sampling!
    @torch.no_grad()
    def sample_batch(self, batch_size: int,
                     keep_chain: int,
                     number_chain_steps: int,
                     num_nodes=None,
                     guidance=None):
        """
        :param batch_id: int #Maybe I add this parameter later
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file #Also for later
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        # model.eval()
        if num_nodes is None:
            n_nodes = self.nodes_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        if (self.guidance) and (guidance is None):
             guidance = self.cf_null_token.repeat((batch_size, 1))

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask,g_T=guidance)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y


        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(5, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(5, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 5)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].to('cpu', non_blocking=True)
            edge_types = E[i, :n, :n].to('cpu', non_blocking=True)
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                self.print('\r{}/{} complete'.format(i+1, num_molecules), end='')
            self.print('\nVisualizing molecules...')
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/')
            self.visualization_tools.visualize(result_path, molecule_list,num_molecules_to_visualize=5)
            self.print("Done.")

        return molecule_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask,g_T=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)

        pred = self.forward(noisy_data, extra_data, node_mask,guidance=g_T)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """
        t = noisy_data['t']
        extra_y = torch.cat((noisy_data['y_t'], t), dim=1)

        return utils.PlaceHolder(X=noisy_data['X_t'], E=noisy_data['E_t'], y=extra_y)
