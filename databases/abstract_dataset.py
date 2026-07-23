import sys
from torch.utils.data._utils.pin_memory import pin_memory
sys.path.append('..')
from diffusion.distributions import DistributionNodes
from utils.utils import to_dense
import torch
import pytorch_lightning as pl
# from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset
from torch.utils.data import WeightedRandomSampler

import json
from pathlib import Path
from datetime import datetime

class AbstractDataModule(LightningDataset):

    def __init__(self, name, train_batch_size, train_num_workers, datasets):
        super().__init__(train_dataset=datasets['train'],
                         val_dataset=datasets['val'],
                         test_dataset=datasets['test'],
                         batch_size=train_batch_size,
                         num_workers=train_num_workers,
                         pin_memory=False,
                         drop_last=True)
        self.name = name
        self.input_dims = None
        self.output_dims = None
        self.batch_size = train_batch_size
        self.num_workers = train_num_workers
        self.pin_memory = pin_memory
        self.drop_last = True
        # # self.drop_last = drop_last
        # self.weights = weights
        # if self.weights is None:
        #     raise Exception('No weights provided.')
        # unique_labels = sorted(self.weights.keys())
        # self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        # self.index_to_weight = torch.tensor(
        #     [1.0 / self.weights[label] for label in unique_labels],
        #     dtype=torch.float
        # )

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    # def train_dataloader(self):
    #     label_indices = torch.empty(len(self.train_dataset), dtype=torch.long)
    #     for idx, data in enumerate(self.train_dataset):
    #         label_indices[idx] = self.label_to_index[data.bead_graph]
    #
    #     sample_weights = self.index_to_weight[label_indices]
    #
    #     # Create the sampler
    #     sampler = WeightedRandomSampler(
    #         weights=sample_weights,
    #         num_samples=len(sample_weights),
    #         replacement=True
    #     )
    #
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.batch_size,
    #         sampler=sampler,
    #         num_workers=self.num_workers,
    #         pin_memory=self.pin_memory,
    #         drop_last=self.drop_last,
    #         shuffle=False,
    #     )

    def node_counts(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.train_dataloader()):
            counts += data.x.sum(dim=0)
        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.train_dataloader()):
            unique, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d

    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for data in self.train_dataloader():
            n = data.x.shape[0]
            for atom in range(n):
                edges = data.edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies

class AbstractDatasetInfos:
    def __init__(self,file_dim=None):
        self.cache_path = Path(file_dim) if file_dim else None
        self.input_dims = None
        self.output_dims = None
        self.num_classes = None
        self.max_n_nodes = None
        self.nodes_dist = None

    def complete_infos(self, n_nodes, node_types):
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(
        self,
        datamodule, guidance=False,
        guidance_size=None,
        guidance_in: str = "y", ):
        """Compute dims from a single training batch."""

        # ---- avoid recomputing if cache is present ----
        if self.cache_path and self.cache_path.exists():
            return self.load_dims(self.cache_path)

        example_batch = next(iter(datamodule.train_dataloader()))

        input_dims = {"X": example_batch["x"].size(1),
                      "E":example_batch["edge_attr"].size(1),}
        # Output dims mirror base features
        output_dims = {
            "X": example_batch["x"].size(1),
            "E": example_batch["edge_attr"].size(1),}

        if guidance:
            #Base input dims (+1 for time conditioning on 'y')
            input_dims["y"] = guidance_size
            input_dims["guidance"] = guidance_size
            output_dims["y"] = guidance_size
        else:
            if guidance_size != None:
                print('Guidance is deactivated. If you want to use guidance activate it.')
                guidance_size = None
            input_dims["y"] = example_batch["y"].size(1) + 1
            output_dims["y"] = 0

        # if hasattr(example_batch, 'guidance') and (example_batch.guidance is not None):
        #     if guidance_in in ['y','all']:
        #         input_dims['y'] += guidance_size
        #     elif guidance_in in ['XE','all']:
        #         input_dims['X'] += guidance_size
        #         input_dims['E'] += guidance_size

        self.input_dims = input_dims
        self.output_dims = output_dims


        return input_dims, output_dims

    # ---- Caching helpers ----
    def save_dims(self, path: str | Path | None = None, extra_meta: dict | None = None):
        """Save input/output dims + basic metadata to JSON."""
        if self.input_dims is None or self.output_dims is None:
            raise ValueError("Nothing to save: compute or load dims first.")
        path = Path(path or self.cache_path)
        if path is None:
            raise ValueError("No cache path provided.")
        payload = {
            "meta": {
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "version": 1,
            },
            "input_dims": self.input_dims,
            "output_dims": self.output_dims,
            "num_classes": self.num_classes,
            "max_n_nodes": self.max_n_nodes,
        }
        if extra_meta:
            payload["meta"].update(extra_meta)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def load_dims(self, path: str | Path | None = None):
        """Load dims from JSON """
        path = Path(path or self.cache_path)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Cache not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.input_dims = data["input_dims"]
        self.output_dims = data["output_dims"]
        self.num_classes = data.get("num_classes")
        self.max_n_nodes = data.get("max_n_nodes")

        return self.input_dims, self.output_dims

    def ensure_dims(
        self,
        datamodule=None,
        guidance: bool = False,
        guidance_size: int = 0,
        guidance_in: str = "y",
        compute_if_missing: bool = True,
    ):
        """
        If cache exists, load it. Otherwise, compute (requires datamodule & feature fns)
        and save to cache_path if provided.
        """
        if self.cache_path and self.cache_path.exists():
            return self.load_dims(self.cache_path)

        if not compute_if_missing:
            raise FileNotFoundError("Dims cache missing and compute_if_missing=False.")

        if datamodule is None:
            raise ValueError(
                "To compute dims, provide datamodule"
            )

        dims = self.compute_input_output_dims(datamodule, guidance, guidance_size, guidance_in)
        if self.cache_path:
            self.save_dims(self.cache_path)
        return dims

    @classmethod
    def from_cache(cls, path: str | Path):
        inst = cls(cache_path=path)
        inst.load_dims(path)
        return inst

# class AbstractDatasetInfos:
#     def complete_infos(self, n_nodes, node_types):
#         self.input_dims = None
#         self.output_dims = None
#         self.num_classes = len(node_types)
#         self.max_n_nodes = len(n_nodes) - 1
#         self.nodes_dist = DistributionNodes(n_nodes)
#
#     def compute_input_output_dims(self, datamodule, extra_features, domain_features,guidance_size,guidance_in='y'):
#         example_batch = next(iter(datamodule.train_dataloader()))
#         ex_dense, node_mask = to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
#                                              example_batch.batch)
#         example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t':example_batch['y'], 'node_mask': node_mask}
#
#
#         self.input_dims = {'X': example_batch['x'].size(1),
#                            'E': example_batch['edge_attr'].size(1),
#                            'y': guidance_size + 1}      # + 1 due to time conditioning
#         ex_extra_feat = extra_features(example_data)
#         self.input_dims['X'] += ex_extra_feat.X.size(-1)
#         self.input_dims['E'] += ex_extra_feat.E.size(-1)
#         self.input_dims['y'] += ex_extra_feat.y.size(-1)
#
#         ex_extra_molecular_feat = domain_features(example_data)
#         self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
#         self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
#         self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)
#
#         # if (example_batch.guidance is not None):
#         #     guidance_sz = guidance_size
#         #     if guidance_in in ['y','all']:
#         #         self.input_dims['y'] += guidance_sz
#         #     elif guidance_in in ['XE','all']:
#         #         self.input_dims['X'] += guidance_sz
#         #         self.input_dims['E'] += guidance_sz
#         # else:
#         #     guidance_sz = 0
#         guidance_sz = guidance_size
#
#         self.input_dims['guidance'] = guidance_sz
#
#         self.output_dims = {'X': example_batch['x'].size(1),
#                             'E': example_batch['edge_attr'].size(1),
#                             'y': guidance_sz}
