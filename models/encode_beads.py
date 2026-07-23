import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, Set2Set

class EncodeBeads(nn.Module):

    '''

    The model is partially taken from: Chem. Sci., 2022, 13, 4498-4511

    It is a Regularized Autoencoder with a message passing layer.

    '''

    def __init__(self, dim_ae):
        super(EncodeBeads,self).__init__()

        self.dim_ae = dim_ae
        self.node_feature_dim = self.dim_ae['m'] #Should it be m*n?
        self.embedding_dim = self.dim_ae['embedding_dim']
        self.encoder_dim = self.dim_ae['encoder_dim']
        self.latent_dim = self.dim_ae['latent_dim']
        self.mp_steps = self.dim_ae['mp_steps']

        #Encoding the beads
        self.embedding = nn.Linear(self.node_feature_dim, self.embedding_dim)

        self.lin0 = nn.Linear(self.embedding_dim ,self.encoder_dim)

        self.conv = SAGEConv(self.encoder_dim, self.encoder_dim, aggr="add", root_weight=True, project=False)
        self.gru = nn.GRU(self.encoder_dim, self.encoder_dim)

        self.set2set = Set2Set(self.encoder_dim, processing_steps=3)
        self.lin1 = nn.Linear(2 * self.encoder_dim, self.encoder_dim)
        self.final = nn.Linear(self.encoder_dim, self.latent_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #Embedding
        emb = self.embedding(x)
        emb = F.leaky_relu(emb)
        #Linear layer
        out = F.leaky_relu(self.lin0(emb))
        h = out.unsqueeze(0)
        for _ in range(self.mp_steps):
            m = F.relu(self.conv(out, edge_index))
            out, h = self.gru(m.unsqueeze(0),h)
            out = out.squeeze(0)
        out_s2s = self.set2set(out, batch)
        out = F.leaky_relu(self.lin1(out_s2s))
        return self.final(out)

class DecodeBeads(nn.Module):

    #Using a MLP on the dimensions of the encoder and latent.


    def __init__(self,dim_ae):
        super(DecodeBeads,self).__init__()
        self.dim_ae = dim_ae
        self.n = self.dim_ae['n'] #Number of beads Max 3 for the moment number of nodes
        self.m = self.dim_ae['m'] #node feature
        self.latent_dim = self.dim_ae['latent_dim']
        self.act_fn = nn.LeakyReLU()

        # Keep positive values
        # self.sm_nodes = nn.Softmax(dim=1)

        # Dimension of the decoder
        self.node_hidden = self.dim_ae['node_hidden']
        self.edge_prelayer_dim = self.dim_ae['edge_prelay']
        self.edge_hidden = self.dim_ae['edge_hidden']
        self.encoded_node = self.dim_ae['encoder_node']
        #Decode layers
        # self.triu_mask = torch.ones(self.n, self.n).triu_().bool()
        row, col = torch.triu_indices(self.n, self.n, offset=1)
        self.edge_iterator = torch.stack([row, col], dim=1)
        # Node decoding
        self.node_reconst = self.construct_linear_layers(self.latent_dim,
                                                   self.node_hidden,
                                                   (self.m+1)*self.n,)
        # Edge decoding
        self.reconst_node_encod = self.construct_linear_layers(self.m,
                                                               self.encoded_node[:-1],
                                                               self.encoded_node[-1])
        self.edge_prelayer = self.construct_linear_layers(self.latent_dim,
                                                          self.edge_prelayer_dim[:-1],
                                                          self.edge_prelayer_dim[-1])

        self.edge_reconst = self.construct_linear_layers(
            self.encoded_node[-1]*self.n + self.edge_prelayer_dim[-1],
            self.edge_hidden,
            (self.n*(self.n-1))//2,
        )

    @staticmethod
    def construct_linear_layers(
            input_dim,
            hidden_dims,
            output_dim,
            nonlinearity=nn.LeakyReLU(negative_slope=0.1)):
        """
        Constructs a torch.nn module with sequential linear layers with the specified input and
        output dimensions and the specified hidden dimensions.
        The nonlinearity is applied to all but the last layer.
        :param input_dim: Input dimension of the first layer
        :param hidden_dims: List of hidden dimensions
        :param output_dim: Output dimension of the last layer
        :param nonlinearity: Nonlinearity to apply to all but the last layer
        :return: List of linear layers
        """
        hidden_dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(hidden_dims) - 1):
            linear = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            layers.append(linear)
            if i < len(hidden_dims) - 2:
                ### Add nonlinearity to all but the last layer ###
                layers.append(nonlinearity)
        return nn.Sequential(*layers)

    def build_dense_adj(self, upper_adj_mat,bs):
        adj = torch.zeros(bs, self.n, self.n, device=upper_adj_mat.device, dtype=upper_adj_mat.dtype)
        i = self.edge_iterator[:, 0]
        j = self.edge_iterator[:, 1]
        # fill symmetric entries
        adj[:, i, j] = upper_adj_mat
        adj[:, j, i] = upper_adj_mat

        return adj

    def forward(self,x):
        # Nodes
        nodes = self.node_reconst(x).view(-1,self.n,self.m+1)

        # Edges
        encoded_latent_space = self.act_fn(self.edge_prelayer(x))
        encoded_nodes = self.reconst_node_encod(nodes.view(-1,self.m+1)[:,:-1].contiguous())
        encoded_nodes = encoded_nodes.view(-1,self.n*encoded_nodes.shape[-1])
        edge_rep = torch.cat([encoded_nodes, encoded_latent_space], dim=-1)
        upper_adj_mat = self.edge_reconst(edge_rep)
        # Get adjacency matrix
        adj = self.build_dense_adj(upper_adj_mat,x.shape[0])

        return nodes, adj



class Autoencoder(nn.Module):
    """
    Autoencoder for molecule encoding
    """

    def __init__(self,dim_ae):

        super().__init__()
        self.dim_ae = dim_ae
        self.encoder = EncodeBeads(self.dim_ae)
        self.decoder = DecodeBeads(self.dim_ae)

    def forward(self, data, only_latent_space=True):
        ### Encode data ###
        z = self.encoder(data)
        if only_latent_space:
            return z
        else:
            ### Decode latent space ###
            nodes, adj_mat = self.decoder(z)
            return nodes, adj_mat, z



















