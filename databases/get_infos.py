# General importations
import numpy as np
from pathlib import Path
# Torch importations
import torch
#Internal importations
from .abstract_dataset import AbstractDatasetInfos


class Datasetinfos(AbstractDatasetInfos):
    # List of permitted atoms CNOS and halogens.
    # permitted_list_of_atoms = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']
    # permitted_list_of_atoms = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']
    permitted_list_of_atoms = ['C', 'N', 'O', 'F']
    #Atom weights
    # atom_weights = {0:1,1:12,2:14,3:16,4:19,5:32,6:35.4,7:79.9,8:127}
    atom_weights = {0:12,1:14,2:16,3:19} #No H
    #Valencies of the atoms
    # valencies = [1,4,3,2,1,4,1,1,1]
    valencies = [4,3,2,1] #No H
    remove_h = True

    def __init__(self, datamodule, recompute_statistics=False, meta=None,file_dim=None):

        super().__init__(file_dim)
        self.name = datamodule.name
        self.remove_h = True
        self.batch_size = datamodule.batch_size


        # Atom encoding/decoding
        self.atom_decoder = self.permitted_list_of_atoms
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        self.num_atom_types = len(self.atom_decoder)
        self.max_weight = 350

        # Initialize the variables
        self.n_nodes = None
        self.max_n_nodes = None
        self.node_types = None
        self.edge_types = None
        self.valency_distribution = None

        meta_files = {
            "n_nodes": Path(f"{self.name}_n_counts.txt"),
            "node_types": Path(f"{self.name}_atom_types.txt"),
            "edge_types": Path(f"{self.name}_edge_types.txt"),
            "valency_distribution": Path(f"{self.name}_valencies.txt"),
        }

        # Load or initialize meta
        meta = self._initialize_meta(meta, meta_files)

        # Load or compute statistics
        self._load_or_compute_stats_all(meta, meta_files, datamodule, recompute_statistics)

        # Final info setup
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

    def _initialize_meta(self, meta, meta_files):
        """Ensure meta dict has correct keys and populate from file if necessary."""
        if meta is None:
            meta = {k: None for k in meta_files}
        assert set(meta.keys()) == set(meta_files.keys())

        for k, file_path in meta_files.items():
            if meta[k] is None and file_path.exists():
                try:
                    meta[k] = np.loadtxt(file_path) #Should this be a tensor?
                    setattr(self, k, meta[k])
                except Exception as e:
                    print(f"Warning: Failed to load {file_path} - {e}")
        return meta

    def _load_or_compute_stats_all(self, meta, meta_files, datamodule, recompute):
        """Load or compute all required statistics."""
        self.n_nodes = self._load_or_compute_stat_single(
            meta, meta_files, "n_nodes", datamodule.node_counts, to_tensor=True
        )
        self.max_n_nodes = len(self.n_nodes) - 1 if self.n_nodes is not None else None

        self.node_types = self._load_or_compute_stat_single(
            meta, meta_files, "node_types", datamodule.node_types, to_tensor=True
        )

        self.edge_types = self._load_or_compute_stat_single(
            meta, meta_files, "edge_types", datamodule.edge_counts, to_tensor=True
        )

        self.valency_distribution = self._load_or_compute_stat_single(
            meta,
            meta_files,
            "valency_distribution",
            lambda: datamodule.valency_count(self.max_n_nodes),
            to_tensor=True
        )

    def _load_or_compute_stat_single(self, meta, meta_files, key, compute_fn, to_tensor=False):
        """Generic loader or computer for a single statistic."""
        file_path = meta_files[key]
        value = None

        if meta[key] is not None and not isinstance(meta[key], (float, int)):  # already loaded
            value = meta[key]
        elif file_path.exists():
            try:
                value = np.loadtxt(file_path)
            except Exception as e:
                print(f"Error reading {key} from {file_path}: {e}")

        if value is None:
            print(f"Computing {key} ...")
            value = compute_fn()
            np.savetxt(file_path, value.numpy() if hasattr(value, 'numpy') else value)

        value = torch.from_numpy(value) if to_tensor and not torch.is_tensor(value) else value
        print(f"Distribution of {key.replace('_', ' ')}: {value}")
        return value
