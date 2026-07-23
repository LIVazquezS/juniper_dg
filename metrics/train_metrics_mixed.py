import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection
#Local Importations
from .abstract_metrics import CrossEntropyMetric
from .rae_metrics import RAELoss



class TrainLoss_Mixed(nn.Module):
    """ Train with Cross entropy for the diffusion part
        and train together the RAE for the bead representation.

        Note: The weight of the regularization term is set to 1e-2 by default.
        This works with 2 message passing steps and uni/dim dataset. Need to
        be adjuscted for trimers.

    """
    def __init__(self, lambda_train,train_guidance=False,num_beads=2,num_bead_types=18,w_rae_reg=0.01):
        super().__init__()

        #Initializing loss functions
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.train_guidance = train_guidance
        if self.train_guidance:
            self.rae_loss = RAELoss(w_z_reg=w_rae_reg)
        else:
            self.rae_loss = None
        # Other parameters
        self.lambda_train = lambda_train
        self.num_beads = num_beads
        self.num_types_beads = num_bead_types

    @staticmethod
    def _format_epoch_log_RAE(prefix, total_vals, node_vals, acc_vals, z_reg):
        '''
        This function formats the output of the RAE

        '''
        names_total = ["all", "edge", "node"]
        names_node = ["class", "size", "dgwo"]
        names_acc = ["class", "size", "edge"]

        to_log = {}
        for i, name in enumerate(names_total):
            to_log[f"{prefix}_loss_rae/total/{name}"] = total_vals[i]
        for i, name in enumerate(names_node):
            to_log[f"{prefix}_loss_rae/node/{name}"] = node_vals[i]
        for i, name in enumerate(names_acc):
            to_log[f"{prefix}_accuracy_rae/{name}"] = acc_vals[i]
        to_log[f"{prefix}_loss_rae/z_val"] = z_reg
        return to_log

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, true_beads=None,
                pred_beads_node=None, pred_beads_adj=None, z_embed=None, stage='train',
                log: bool=True, valid=False):
        """ Compute train metrics
        All atom data:
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)

        Coarse grain data:
        #TODO: Check types of this data
        true_beads: data of the beads
        pred_beads_node: nodes predicted by the model
        pred_beads_adj: adjacency matrix of the beads
        true_y : tensor -- (bs,b,bt ) #DG_OW_ref
        z_embed: tensor -- Embedding space for regularization.

        Other options:
        stage: str -- train, validation or test.
        valid: bool -- True for validation.
        log : boolean. """

        #True values
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        # Loss Nodes
        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        # print('Node loss',loss_X)
        # Loss Edges
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        # print('Edge loss',loss_E)
        if self.train_guidance:
            # Loss RAE
            loss_rae, batch_log_rae = self.rae_loss(true_beads, pred_beads_node,pred_beads_adj,
                                                z_embed,valid=valid)
        else:
            loss_rae = 0.0
            batch_log_rae = [np.zeros(3), np.zeros(3), np.zeros(3), 0]

        if log:
            to_log = {"train_loss/batch_CE": (loss_X + self.lambda_train[0]*loss_E).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      }
            if self.train_guidance:
                to_log_rae = self._format_epoch_log_RAE('batch_{}'.format(stage), batch_log_rae[0], batch_log_rae[1],
                                                  batch_log_rae[2],batch_log_rae[3])
                to_log = to_log | to_log_rae
        else:
            to_log = None

        total_loss = loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1]*loss_rae
        return to_log, total_loss


    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.rae_loss]:
            if metric is not None:
                metric.reset()
            else:
                pass

    def log_epoch_metrics(self,stage='train_epoch'):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1

        to_log = {"train_epoch/X_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,}

        if self.train_guidance:
            if self.rae_loss.total_samples > 0:
                total_epoch, to_log_epoch = self.rae_loss.compute()
            else:
                to_log_epoch = [np.zeros(3), np.zeros(3), np.zeros(3), 0]

            to_log_epoch_rae = self._format_epoch_log_RAE(stage, to_log_epoch[0],
                                                  to_log_epoch[1],to_log_epoch[2],to_log_epoch[3])
            to_log = to_log | to_log_epoch_rae

        return to_log


class TrainLoss_Single(nn.Module):
    """ Train with Cross entropy for the diffusion part
        and MSE for the prediction of dg_wo_cg.

    """
    def __init__(self, lambda_train,train_guidance=False):
        super().__init__()

        #Initializing loss functions
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.train_guidance = train_guidance

        if self.train_guidance:
            self.dgwo_loss = MeanSquaredError()
        else:
            self.dgwo_loss = None
        # Other parameters
        self.lambda_train = lambda_train


    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, pred_dgwo, true_dgwo,
                stage='train',log: bool=True):
        """ Compute train metrics
        All atom data:
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)

        partition coefficient data:
        pred_dg_wo : tensor -- (bs,1)
        true_dg_wo : tensor -- (bs,1)
        Other options:
        stage: str -- train, validation or test.
        log : boolean. """

        #True values
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        # Loss Nodes
        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        # print('Node loss',loss_X)
        # Loss Edges
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        # print('Edge loss',loss_E)
        if self.train_guidance:
            # Loss dgwo
            loss_dgwo = self.dgwo_loss(true_dgwo.unsqueeze(-1),pred_dgwo)
        else:
            loss_dgwo = 0.0

        if log:
            to_log = {"train_loss/batch_CE": (loss_X + self.lambda_train[0]*loss_E).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      }
            if self.train_guidance:
                to_log_mse = {"train_loss/Guidance_MSE": self.dgwo_loss.compute() if true_dgwo.numel() > 0 else -1,}
                to_log = to_log | to_log_mse
        else:
            to_log = None

        total_loss = loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1]*loss_dgwo
        return to_log, total_loss


    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.dgwo_loss]:
            if metric is not None:
                metric.reset()
            else:
                pass

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1

        to_log = {"train_epoch/X_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,}

        if self.train_guidance:
            epoch_dgwo_loss = self.dgwo_loss.compute()
        else:
            epoch_dgwo_loss = -1

        to_log_epoch_guidance = {"train_epoch/Guidance_MSE": epoch_dgwo_loss,}


        to_log = to_log | to_log_epoch_guidance

        return to_log

