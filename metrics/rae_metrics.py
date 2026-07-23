import torch
from torch.nn import functional as F, MSELoss,BCEWithLogitsLoss
from torchmetrics import Metric
from itertools import permutations
from torch_geometric.utils import to_dense_adj, to_dense_batch
import numpy as np

class RAELoss(Metric):
    """
    Parts of this are adapted from:
    https://github.com/BereauLab/Multi-Level-BO-w-Hierarchical-CG/blob/main/chespex/chespex/encoding/loss.py
    """
    def __init__(self,w_z_reg=1e-4,w_edge_loss=4):
        super(RAELoss, self).__init__()
        # Accumulators (sums over samples)
        self.add_state("sum_total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_edge_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_node_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("sum_node_class_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_node_size_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_node_dgwo_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("sum_node_class_acc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_node_size_acc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_edge_acc", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("sum_z_reg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")


        self.CE_edges = BCEWithLogitsLoss(reduction='none')
        self.mse_dgwo_nodes = MSELoss(reduction='none')
        self.w_z_reg = w_z_reg
        self.w_edge_loss = w_edge_loss

    @staticmethod
    def _cross_entropy_loss(
            input_nodes: torch.Tensor,
            predicted_nodes: torch.Tensor,
            start_idx: int,
            end_idx: int,
    ) -> torch.Tensor:
        """
        Calculate the cross entropy loss for a subset of the node features.
        :param input_nodes: The input node features
            (batch size, number of nodes, number of features).
        :param predicted_nodes: The predicted node features
            (batch size, number of nodes, number of features).
        :param start_idx: The start index of the subset.
        :param end_idx: The end index of the subset.
        :return: The cross entropy loss for the subset of node features
            (batch size, number of nodes).
        """
        neg_log_prob = -F.log_softmax(predicted_nodes[:, :, start_idx:end_idx], dim=2)
        cross_entropy = torch.sum(
            input_nodes[:, :, start_idx:end_idx] * neg_log_prob, dim=2
        )
        return cross_entropy

    @staticmethod
    def _permute_matrices(
            permutation_indices: torch.Tensor, matrices: torch.Tensor
    ) -> torch.Tensor:
        """
        Permute the rows and columns of a batch of matrices according to a given batch of
        permutation indices.
        :param permutation_indices: A batch of permutation indices.
        :param matrices: A batch of matrices.
        :return: The permuted matrices.
        """
        n, k = permutation_indices.shape
        row_indices = permutation_indices.unsqueeze(-1).expand(n, k, k)
        col_indices = permutation_indices.unsqueeze(1).expand(n, k, k)
        matrix_indices = torch.arange(n).unsqueeze(-1).unsqueeze(-1).expand(n, k, k)
        permuted_matrices = matrices[matrix_indices, row_indices.type(torch.int), col_indices.type(torch.int)]
        return permuted_matrices

    def _get_node_loss(self,
                       x_beads_true, x_beads_pred, dummy_nodes_mask,
                       n_node_classes, ):
        """
        Calculate the loss for the node features.
        """
        idx = (n_node_classes + 1, n_node_classes + 1 + 2)

        class_loss = self._cross_entropy_loss(
            x_beads_true, x_beads_pred, 0, idx[0]
        )
        size_loss = self._cross_entropy_loss(
            x_beads_true, x_beads_pred, idx[0], idx[1]
        )

        dg_wo_beads = self.mse_dgwo_nodes(x_beads_true[:, :, -1], x_beads_pred[:, :, -1])
        size_oco_loss = 1.5 * size_loss + dg_wo_beads
        # return size_oco_loss + class_loss.mean(dim=1)
        size_oco_loss = (size_oco_loss * dummy_nodes_mask).sum(
            dim=1
        ) / dummy_nodes_mask.sum(dim=1)
        node_loss_t = size_oco_loss + class_loss.mean(dim=1)

        return node_loss_t, class_loss.mean(dim=1), size_loss.mean(dim=1), dg_wo_beads.mean(dim=1)

    def node_loss(self, x_true, x_pred, dummy_nodes_mask, n_nodes, n_node_classes):
        B = x_true.size(0)

        device = x_true.device
        dtype = x_true.dtype

        best_loss = torch.full((B,), 1e15, device=device, dtype=dtype)
        opt_perm = torch.arange(n_nodes, device=device).repeat(B, 1)

        best_class = torch.full((B,), 1e15, device=device, dtype=dtype)
        best_size = torch.full((B,), 1e15, device=device, dtype=dtype)
        best_dgwo = torch.full((B,), 1e15, device=device, dtype=dtype)

        for perm in permutations(range(n_nodes), n_nodes):
            perm_true = x_true[:, perm]  # permute truth

            node_b, class_b, size_b, dgwo_b = self._get_node_loss(perm_true, x_pred, dummy_nodes_mask, n_node_classes)

            better = node_b < best_loss
            opt_perm[better] = (
                torch.Tensor(perm).long().to(x_pred.device)
            )
            best_loss = torch.minimum(best_loss, node_b)
            best_class = torch.minimum(best_class, class_b)
            best_size = torch.minimum(best_size, size_b)
            best_dgwo = torch.minimum(best_dgwo, dgwo_b)

        return best_loss.mean(), best_class.mean(), best_size.mean(), best_dgwo.mean(), opt_perm

    @staticmethod
    def pred_to_mol(class_pred,size_pred,type='complete'):
        if type == 'complete':
           bead_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'N0', 'Na', 'Nd', 'Nda', 'P1', 'P2', 'P3', 'P4', 'P5','No Bead']
        elif type == 'simplified':
           bead_list = ['C','N','P']
        else:
            raise NotImplementedError('Only complete and simplified predictions are implemented')

        print_reps = []
        for i,mol in enumerate(class_pred):
            string_rep = []
            for j,node in enumerate(mol):
                if node < len(bead_list):
                    b_type = bead_list[node]
                    name = ["S",""][size_pred[i][j]]
                    name += b_type
                    string_rep.append(name)
            print_rep = "_".join(string_rep)
            print_reps.append(print_rep)

        return print_reps

    def _get_node_accuracy(self,
            input_node_features: torch.Tensor,
            predicted_node_features: torch.Tensor,
            permutation_indices: torch.Tensor,
            dummy_nodes_mask: torch.Tensor,
            n_node_classes: int,
            valid: bool=False,
    ) -> torch.Tensor:
        ### Permutate the input node features ###
        n, k = permutation_indices.shape
        matrix_indices = torch.arange(n).unsqueeze(-1).expand(n, k)
        input_node_features = input_node_features[matrix_indices, permutation_indices]
        ### Calculate the accuracy for the node classes ###
        idx = (n_node_classes + 1 , n_node_classes +1 + 2)
        class_prediction = predicted_node_features[:, :, :idx[0]].argmax(dim=2)
        class_truth = input_node_features[:, :, :idx[0]].argmax(dim=2)
        class_acc = (class_prediction == class_truth).float().mean()
        size_prediction = predicted_node_features[:, :, idx[0]: idx[1]].argmax(dim=2)
        size_truth = input_node_features[:, :, idx[0]: idx[1]].argmax(dim=2)
        size_acc = (size_prediction == size_truth).float()
        size_acc = (size_acc * dummy_nodes_mask).sum() / dummy_nodes_mask.sum()
        if valid:
            print('True Beads:',self.pred_to_mol(class_truth.clone(),size_truth.clone()))
            print('Predicted Beads:',self.pred_to_mol(class_prediction.clone(),size_prediction.clone()))
        return [class_acc.item(), size_acc.item()]



    def update(self, data_beads_true,pred_node,pred_adj,z_embed,valid=False):
        #data beads true: nodes, edge_index
        #data beads pred: nodes, adj_mat
        #z_embed: embedding space for regularization

        #Transforming the data to deep representation
        batch_size, n_nodes, node_size = pred_node.shape
        device = data_beads_true.x.device
        n_node_classes = node_size - 4 # extra value, size, dgwo
        #Nodes
        extended_input_features = torch.cat(
            (
                data_beads_true.x[:, :n_node_classes],
                torch.zeros(len(data_beads_true.x), 1).to(device),
                data_beads_true.x[:, n_node_classes:],
            ),
            dim=1,
        )
        x_beads_true,dummy_nodes_mask = to_dense_batch(extended_input_features, data_beads_true.batch,
                                           fill_value=0, max_num_nodes=n_nodes)
        dummy_indices = torch.where(~dummy_nodes_mask)
        dummy_indices += tuple(
            torch.ones(1, len(dummy_indices[0]), dtype=torch.int,device=device) * n_node_classes,
        )
        x_beads_true[dummy_indices] = 1

        #Adjency
        adj_true = to_dense_adj(data_beads_true.edge_index, data_beads_true.batch)
        ### Construct a mask to ignore the edges between dummy nodes ###
        edge_ignore_mask = torch.ones_like(adj_true)
        edge_ignore_mask[~dummy_nodes_mask] = 0
        edge_ignore_mask = edge_ignore_mask * edge_ignore_mask.transpose(1, 2)
        triu_indices = torch.triu_indices(n_nodes, n_nodes, offset=1).to(
            x_beads_true.device
        )
        edge_ignore_mask = edge_ignore_mask[:, *triu_indices].bool()


        (node_loss, node_class, node_size,
         node_dgwo, optimal_permutation) = self.node_loss(x_beads_true, pred_node,dummy_nodes_mask,n_nodes, n_node_classes)

        ### Calculate the edge loss ###
        permuted_adj_true = self._permute_matrices(
            optimal_permutation, adj_true)
        triu_indices = torch.triu_indices(n_nodes, n_nodes, offset=1).to(device).type(torch.int)
        true_triu = permuted_adj_true[:, *triu_indices]
        pred_triu = pred_adj[:, *triu_indices]

        edge_loss = self.CE_edges(pred_triu, true_triu)[edge_ignore_mask].mean()
        if torch.isnan(edge_loss).any():
            edge_loss = torch.tensor(0.0, device=device)

        if z_embed is None:
            z_reg = torch.tensor(0.0, device=device)
        else:
            z_reg = torch.linalg.vector_norm(z_embed, dim=1).pow(2).mean()

        batch_total_loss = node_loss + self.w_edge_loss * edge_loss + self.w_z_reg * z_reg

        with torch.no_grad():
            class_acc,size_acc = self._get_node_accuracy(x_beads_true, pred_node,
                                                         optimal_permutation.long(),
                                                         dummy_nodes_mask,n_node_classes,valid=valid)
            edge_acc_t = ((pred_triu > 0).float() == true_triu).float().mean()
            edge_acc = 0.0 if torch.isnan(edge_acc_t).any() else edge_acc_t.item()


        # Accumulate sums over samples so epoch mean is correct
        self.sum_total_loss += batch_total_loss*batch_size
        self.sum_edge_loss += edge_loss.detach()*batch_size
        self.sum_node_loss += node_loss.detach()*batch_size

        self.sum_node_class_loss += node_class.detach()*batch_size
        self.sum_node_size_loss += node_size.detach()*batch_size
        self.sum_node_dgwo_loss += node_dgwo.detach()*batch_size

        self.sum_node_class_acc += torch.tensor(class_acc, device=device)*batch_size
        self.sum_node_size_acc += torch.tensor(size_acc, device=device)*batch_size
        self.sum_edge_acc += torch.tensor(edge_acc, device=device)*batch_size


        self.sum_z_reg += z_reg.detach()*batch_size
        self.total_samples += batch_size


    def compute(self,valid=False):
        """
        Epoch means (detached).
        """
        n = self.total_samples

        total_vals = [
            self.sum_total_loss / n,
            self.sum_edge_loss / n,
            self.sum_node_loss / n,
        ]
        node_vals = [
            self.sum_node_class_loss / n,
            self.sum_node_size_loss / n,
            self.sum_node_dgwo_loss / n,
        ]
        acc_vals = [
            self.sum_node_class_acc / n,
            self.sum_node_size_acc / n,
            self.sum_edge_acc / n,
        ]
        z_reg = self.sum_z_reg / n

        to_log = [total_vals, node_vals, acc_vals, z_reg]
        if valid:
            to_log = self._format_epoch_log_RAE('val',to_log[0],to_log[1],to_log[2],to_log[3])
        return self.sum_total_loss, to_log

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


