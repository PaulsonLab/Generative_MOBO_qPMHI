
import deepchem as dc

import numpy as np 
import pandas as pd
import torch
import rdkit
import dgl


#%%

from rdkit import Chem
from rdkit.Chem import AllChem
def MolMerger(smiles1, smiles2):

    def find_most_charged_atom(mol, partial_charges, is_positive=True):
        charge_function = max if is_positive else min
        charge = charge_function(partial_charges)
        charge_index = partial_charges.index(charge)
        return mol.GetAtomWithIdx(charge_index)


    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)


    AllChem.ComputeGasteigerCharges(mol1)
    AllChem.ComputeGasteigerCharges(mol2)

    gasteiger_charges_mol1 = [a.GetDoubleProp("_GasteigerCharge") for a in mol1.GetAtoms()]
    gasteiger_charges_mol2 = [a.GetDoubleProp("_GasteigerCharge") for a in mol2.GetAtoms()]



    most_positive_atom1 = find_most_charged_atom(mol1, [a.GetDoubleProp("_GasteigerCharge") for a in mol1.GetAtoms()], is_positive=True)
    most_negative_atom1 = find_most_charged_atom(mol1, [a.GetDoubleProp("_GasteigerCharge") for a in mol1.GetAtoms()], is_positive=False)


    most_positive_atom2 = find_most_charged_atom(mol2, [a.GetDoubleProp("_GasteigerCharge") for a in mol2.GetAtoms()], is_positive=True)
    most_negative_atom2 = find_most_charged_atom(mol2, [a.GetDoubleProp("_GasteigerCharge") for a in mol2.GetAtoms()], is_positive=False)

    bond_order = Chem.BondType.HYDROGEN


    combined_mol = Chem.RWMol(Chem.CombineMols(mol1, mol2))

    for i, atom in enumerate(combined_mol.GetAtoms()):
        if i < mol1.GetNumAtoms():
            atom.SetDoubleProp("GasteigerChargeFinal", gasteiger_charges_mol1[i])
        else:
            atom.SetDoubleProp("GasteigerChargeFinal", gasteiger_charges_mol2[i - mol1.GetNumAtoms()])


    combined_mol.AddBond(most_positive_atom1.GetIdx(), most_negative_atom2.GetIdx() + mol1.GetNumAtoms(), order=bond_order)
    combined_mol.AddBond(most_negative_atom1.GetIdx(), most_positive_atom2.GetIdx() + mol1.GetNumAtoms(), order=bond_order)


    final_combined_mol = Chem.Mol(combined_mol)


    combined_smiles = Chem.MolToSmiles(final_combined_mol)
    return combined_smiles


#%%

'''
Molecule Featurizer
'''

from typing import List, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.graph_data import GraphData
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.molecule_feature_utils import get_atom_type_one_hot
from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info
from deepchem.utils.molecule_feature_utils import get_atom_hydrogen_bonding_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_chirality_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge
from deepchem.utils.molecule_feature_utils import get_atom_partial_charge
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_type_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_in_same_ring_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_conjugated_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_stereo_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_implicit_valence_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_explicit_valence_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_graph_distance_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_chirality_one_hot
from deepchem.utils.rdkit_utils import compute_all_pairs_shortest_path
from deepchem.utils.rdkit_utils import compute_pairwise_ring_info


def _construct_atom_feature(atom: RDKitAtom, h_bond_infos: List[Tuple[int, str]]) -> np.ndarray:
    atom_type = get_atom_type_one_hot(atom)
    formal_charge = get_atom_formal_charge(atom)
    hybridization = get_atom_hybridization_one_hot(atom)
    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = get_atom_is_in_aromatic_one_hot(atom)
    degree = get_atom_total_degree_one_hot(atom)
    total_num_Hs = get_atom_total_num_Hs_one_hot(atom)
    chirality = get_atom_chirality_one_hot(atom)

    atom_feat = np.concatenate([
        atom_type, formal_charge, hybridization, acceptor_donor, aromatic,
        degree, total_num_Hs, chirality
    ])

    return atom_feat


def _construct_bond_feature(bond: RDKitBond, dist_matrix) -> np.ndarray:

    bond_type = get_bond_type_one_hot(bond)
    same_ring = get_bond_is_in_same_ring_one_hot(bond)
    conjugated = get_bond_is_conjugated_one_hot(bond)
    stereo = get_bond_stereo_one_hot(bond)
    dist = get_bond_graph_distance_one_hot(bond, graph_dist_matrix = dist_matrix)
    return np.concatenate([bond_type, same_ring, conjugated, stereo, dist])


class MolMergerFeaturizer(MolecularFeaturizer):

    def __init__(self,
                 use_edges: bool = False):

        self.use_edges = use_edges

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:

        assert datapoint.GetNumAtoms() > 1
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )


        # construct atom (node) feature
        h_bond_infos = construct_hydrogen_bonding_info(datapoint)
        dist_matrix = Chem.GetDistanceMatrix(datapoint)
        atom_features = np.asarray(
            [
                _construct_atom_feature(atom, h_bond_infos)
                for atom in datapoint.GetAtoms()
            ],
            dtype=float,
        )

        # construct edge (bond) index
        src, dest = [], []
        for bond in datapoint.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        # construct edge (bond) feature
        bond_features = None
        if self.use_edges:
            features = []
            for bond in datapoint.GetBonds():
                features += 2 * [_construct_bond_feature(bond, dist_matrix)]
            bond_features = np.asarray(features, dtype=float)

        return GraphData(node_features=atom_features,
                         edge_index=np.asarray([src, dest], dtype=int),
                         edge_features=bond_features)

#%%

'''
model 
'''

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import edge_softmax

__all__ = ['AttentiveFPGNN']

# pylint: disable=W0221, C0103, E1101
class AttentiveGRU1(nn.Module):

    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.edge_transform[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, edge_feats, node_feats):

        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class AttentiveGRU2(nn.Module):

    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, node_feats):

        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.u_mul_e('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class GetContext(nn.Module):

    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node[0].reset_parameters()
        self.project_edge1[0].reset_parameters()
        self.project_edge2[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges1(self, edges):

        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):

        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):

        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])

class GNNLayer(nn.Module):

    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_edge[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges(self, edges):

        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):

        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.attentive_gru(g, logits, node_feats)

class AttentiveFPGNN(nn.Module):
    
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(AttentiveFPGNN, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.init_context.reset_parameters()
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        return node_feats
    


import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AttentiveFPReadout']

# pylint: disable=W0221
class GlobalPool(nn.Module):

    def __init__(self, feat_size, dropout):
        super(GlobalPool, self).__init__()

        self.compute_logits = nn.Sequential(
            nn.Linear(2 * feat_size, 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_size, feat_size)
        )
        self.gru = nn.GRUCell(feat_size, feat_size)

    def forward(self, g, node_feats, g_feats, get_node_weight=False):

        with g.local_scope():
            g.ndata['z'] = self.compute_logits(
                torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))
            g.ndata['a'] = dgl.softmax_nodes(g, 'z')
            g.ndata['hv'] = self.project_nodes(node_feats)

            g_repr = dgl.sum_nodes(g, 'hv', 'a')
            context = F.elu(g_repr)

            if get_node_weight:
                return self.gru(context, g_feats), g.ndata['a']
            else:
                return self.gru(context, g_feats)

class AttentiveFPReadout(nn.Module):

    def __init__(self, feat_size, num_timesteps=2, dropout=0.):
        super(AttentiveFPReadout, self).__init__()

        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readouts.append(GlobalPool(feat_size, dropout))

    def forward(self, g, node_feats, get_node_weight=False):

        with g.local_scope():
            g.ndata['hv'] = node_feats
            g_feats = dgl.sum_nodes(g, 'hv')

        if get_node_weight:
            node_weights = []

        for readout in self.readouts:
            if get_node_weight:
                g_feats, node_weights_t = readout(g, node_feats, g_feats, get_node_weight)
                node_weights.append(node_weights_t)
            else:
                g_feats = readout(g, node_feats, g_feats)

        if get_node_weight:
            return g_feats, node_weights
        else:
            return g_feats
        


import torch.nn as nn


__all__ = ['AttentiveFPPredictor']

# pylint: disable=W0221
class AttentiveFPPredictor(nn.Module):
    
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 n_tasks=1,
                 dropout=0.):
        super(AttentiveFPPredictor, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),

            nn.Linear(graph_feat_size, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False,return_embeddings=False):
        
        node_feats = self.gnn(g, node_feats, edge_feats)
        if get_node_weight:
            g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
            
            if return_embeddings:
                
                return g_feats
                
            return self.predict(g_feats), node_weights
        else:
            g_feats = self.readout(g, node_feats, get_node_weight)
            
            if return_embeddings:
                
                return g_feats
            return self.predict(g_feats)
        
        
        
        
        
"""
DGL-based AttentiveFP for graph property prediction.
"""
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel


class AttentiveFP(nn.Module):
    

    def __init__(self,
                 n_tasks: int,
                 num_layers: int = 2,
                 num_timesteps: int = 2,
                 graph_feat_size: int = 200,
                 dropout: float = 0.,
                 mode: str = 'regression',
                 number_atom_features: int = 32,
                 number_bond_features: int = 19,
                 n_classes: int = 2,
                 nfeat_name: str = 'x',
                 efeat_name: str = 'edge_attr'):
        
        try:
            import dgl  # noqa: F401
        except:
            raise ImportError('This class requires dgl.')
        try:
            import dgllife  # noqa: F401
        except:
            raise ImportError('This class requires dgllife.')

        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        super(AttentiveFP, self).__init__()

        self.n_tasks = n_tasks
        self.mode = mode
        self.n_classes = n_classes
        self.nfeat_name = nfeat_name
        self.efeat_name = efeat_name
        if mode == 'classification':
            out_size = n_tasks * n_classes
        else:
            out_size = n_tasks


        self.model = AttentiveFPPredictor(
            node_feat_size=number_atom_features,
            edge_feat_size=number_bond_features,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            graph_feat_size=graph_feat_size,
            n_tasks=out_size,
            dropout=dropout)

    def forward(self, g):
        
        node_feats = g.ndata[self.nfeat_name]
        edge_feats = g.edata[self.efeat_name]
        self.device = torch.device("cpu")
        out = self.model(g, node_feats, edge_feats)

        if self.mode == 'classification':
            if self.n_tasks == 1:
                logits = out.view(-1, self.n_classes)
                softmax_dim = 1
            else:
                logits = out.view(-1, self.n_tasks, self.n_classes)
                softmax_dim = 2
            proba = F.softmax(logits, dim=softmax_dim)
            return proba, logits
        else:
            return out


class AttentiveFPModel(TorchModel):
    
    def __init__(self,
                 n_tasks: int,
                 num_layers: int = 2,
                 num_timesteps: int = 2,
                 graph_feat_size: int = 200,
                 dropout: float = 0.,
                 mode: str = 'regression',
                 number_atom_features: int = 32,
                 number_bond_features: int = 19,
                 n_classes: int = 2,
                 self_loop: bool = True,
                 return_embeddings: bool = False,
                 **kwargs):
        self.return_embeddings = return_embeddings
        self.device = torch.device("cpu")
        model = AttentiveFP(n_tasks=n_tasks,
                            num_layers=num_layers,
                            num_timesteps=num_timesteps,
                            graph_feat_size=graph_feat_size,
                            dropout=dropout,
                            mode=mode,
                            number_atom_features=number_atom_features,
                            number_bond_features=number_bond_features,
                            n_classes=n_classes)
        if mode == 'regression':
            loss: Loss = L2Loss()
            output_types = ['prediction']
        else:
            loss = SparseSoftmaxCrossEntropy()
            output_types = ['prediction', 'loss']
            
        super(AttentiveFPModel, self).__init__(model,
                                               loss=loss,
                                               output_types=output_types,
                                               **kwargs)

        self._self_loop = self_loop

    def _prepare_batch(self, batch):
        
        try:
            import dgl
        except:
            raise ImportError('This class requires dgl.')

        inputs, labels, weights = batch

        dgl_graphs = [
            graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
        ]
        self.device = torch.device("cpu")
        
        inputs = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super(AttentiveFPModel, self)._prepare_batch(
            ([], labels, weights))
        return inputs, labels, weights
    
    def predict_embeddings(self, dataset):
        """
        Extract graph embeddings for a given dataset.
        """
        try:
            import dgl
        except ImportError:
            raise ImportError('This class requires dgl.')

        # Convert dataset to DGL graphs
        dgl_graphs = [graph.to_dgl_graph(self_loop=self._self_loop) for graph in dataset.X]
        g_batch = dgl.batch(dgl_graphs).to(self.device)

        # Extract node and edge features
        node_feats = g_batch.ndata['x']
        edge_feats = g_batch.edata['edge_attr']

        # Pass through GNN and Readout layers to get h_final
        with torch.no_grad():
            h_final = self.model.model(g_batch, node_feats, edge_feats, return_embeddings=True)

        return h_final.cpu().numpy()  # Convert to NumPy for easy analysis
    
    
    


#%%
import sys
sys.setrecursionlimit(100000)  # or a higher value

import pandas as pd

task =["logP","TPSA"] #
featurizer = MolMergerFeaturizer(use_edges = True)
dataset_file = "./logP_TPSA_data.csv"

loader = dc.data.CSVLoader(
    tasks = task,
    smiles_field = "smiles",
    featurizer = featurizer
)



dataset = loader.featurize(dataset_file,data_dir ="/users/PAS2252/madhav1319/deepchem_cache/molmerger_dataset/",shard_size = 8192)

splitter = dc.splits.IndexSplitter()

'''
# Load the original data.
original_df = pd.read_csv(dataset_file)
all_indices = set(original_df.index)

# Assuming dataset.ids contains the successful rows (e.g., SMILES or IDs)
# Here, if you didn’t set an id_field, dataset.ids might just be the SMILES strings.
# So you may need to adjust your approach depending on what dataset.ids contains.
successful_ids = set(dataset.ids)

# If you had the original index in the CSV, this comparison is direct.
# Otherwise, if using SMILES as IDs:
failed_indices = original_df[~original_df["Smiles_Merged"].isin(successful_ids)].index
print("Failed row indices:", list(failed_indices))


'''

#%% Load the saved files 

import deepchem as dc
splitter = dc.splits.IndexSplitter()
data_dir = "/users/PAS2252/madhav1319/deepchem_cache/molmerger_dataset/"

print(f"Loading dataset from '{data_dir}'...")
dataset = dc.data.DiskDataset(data_dir)
print(f"Number of samples in the dataset: {len(dataset)}")

#%%

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import deepchem as dc
import dgl
from rdkit import Chem

from deepchem.metrics import Metric, pearson_r2_score, rms_score


# ────────────────────────────────────────────────────────────────────────────────
# 1) Define a native‐PyTorch Bayesian Linear layer
# ────────────────────────────────────────────────────────────────────────────────
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features,
                 prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        self.weight_mu        = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_log_sigma = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu          = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_log_sigma   = nn.Parameter(torch.full((out_features,), -3.0))
        self.prior_mu         = prior_mu
        self.prior_sigma      = prior_sigma

    def forward(self, x):
        w_sigma = torch.exp(self.weight_log_sigma)
        b_sigma = torch.exp(self.bias_log_sigma)
        weight = self.weight_mu + w_sigma * torch.randn_like(w_sigma)
        bias   = self.bias_mu   + b_sigma * torch.randn_like(b_sigma)
        return nn.functional.linear(x, weight, bias)

    def kl_loss(self):
        def _kl(q_mu, q_log_sigma):
            q_sigma = torch.exp(q_log_sigma)
            p_sigma = self.prior_sigma
            p_mu    = self.prior_mu
            return (
                torch.log(p_sigma / q_sigma)
                + (q_sigma**2 + (q_mu - p_mu)**2) / (2 * p_sigma**2)
                - 0.5
            ).sum()
        return _kl(self.weight_mu, self.weight_log_sigma) + _kl(self.bias_mu, self.bias_log_sigma)



# Define a small BNN to act as the predictor head
class BayesianPredictor(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.bnn_head = nn.Sequential(
            BayesianLinear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            BayesianLinear(hidden_dim, out_features)
        )
    
    def forward(self, x):
        return self.bnn_head(x)

class BayesianGaussianHead(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        # one head for mean
        self.head_mu  = BayesianLinear(in_features, out_features, prior_mu, prior_sigma)
        # one head for log‐variance
        self.head_logvar = BayesianLinear(in_features, out_features, prior_mu, prior_sigma)
    def forward(self, x):
        mu     = self.head_mu(x)           # [batch, n_tasks]
        logvar = self.head_logvar(x)       # [batch, n_tasks]
        return mu, logvar
    def kl_loss(self):
        return self.head_mu.kl_loss() + self.head_logvar.kl_loss()

# ────────────────────────────────────────────────────────────────────────────────
# 2) Load and featurize data
# ────────────────────────────────────────────────────────────────────────────────
tasks = ["logP","TPSA"]

train, test = splitter.train_test_split(dataset, seed=42, frac_train=0.2)

# metrics
metrics = [
    Metric(pearson_r2_score, np.mean),
    Metric(rms_score,        np.mean),
]


# ────────────────────────────────────────────────────────────────────────────────
# 3) Build the base DeepChem model and swap in the Bayesian head
# ────────────────────────────────────────────────────────────────────────────────
dc_model = AttentiveFPModel(
    n_tasks=len(tasks),
    mode="regression",
    num_layers=5,
    dropout=0.2,
    num_timesteps=3
)
# extract the underlying PyTorch predictor
pt_predictor = dc_model.model.model  # AttentiveFPPredictor

# replace its final nn.Linear → BayesianLinear
graph_feat_size = pt_predictor.predict[1].in_features
n_tasks         = pt_predictor.predict[1].out_features

pt_predictor.predict = BayesianPredictor(
    in_features=graph_feat_size, 
    out_features=n_tasks,
    hidden_dim=128 # You can tune this
)



# ────────────────────────────────────────────────────────────────────────────────
# 4) Helper: batch DeepChem GraphData → DGLGraph + labels
# ────────────────────────────────────────────────────────────────────────────────

def make_loader(dc_dataset, batch_size=32, shuffle=True):
    idxs = np.arange(len(dc_dataset))
    if shuffle:
        np.random.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        batch = idxs[start:start + batch_size]
        # 1) batch the graphs
        g_list = [
            dc_dataset.X[i].to_dgl_graph(self_loop=True)
            for i in batch
        ]
        bg = dgl.batch(g_list).to(device)

        # 2) slice the labels array once (shape: [batch, n_tasks])
        y_np = dc_dataset.y[batch]  
        # no Python‐list at all → fast contiguous tensor
        y = torch.from_numpy(y_np).float().to(device)

        yield bg, y



def get_total_kl_loss(model_head):
    kl_loss = 0.0
    for module in model_head.modules():
        if isinstance(module, BayesianLinear):
            kl_loss += module.kl_loss()
    return kl_loss

# ────────────────────────────────────────────────────────────────────────────────
# 5) Training loop with Bayesian loss
# ────────────────────────────────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pt_predictor.to(device)

optimizer = optim.AdamW(pt_predictor.parameters(), lr=1e-4)
regression_loss_fn = nn.MSELoss()
beta      = 1e-5  # Weight on KL term, may require tuning

for epoch in range(1, 201):
    pt_predictor.train()
    running_loss = 0.0

    for g_batch, y_batch in make_loader(train, batch_size=32):
        h       = pt_predictor.gnn(g_batch, g_batch.ndata['x'], g_batch.edata['edge_attr'])
        g_feats = pt_predictor.readout(g_batch, h)
        
        # 2) Forward through the BNN predictor head (stochastic)
        prediction = pt_predictor.predict(g_feats)
        
        # 3) Calculate regression loss (data-fit term)
        regression_loss = regression_loss_fn(prediction, y_batch)
        
        # 4) Calculate KL regularization from all layers in the BNN predictor
        kl_loss = get_total_kl_loss(pt_predictor.predict) / len(train)
        
        # 5) Combine the losses
        loss = regression_loss + beta * kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y_batch.size(0)

    # The original evaluate call still works perfectly
    train_r2 = dc_model.evaluate(train, metrics)['mean-pearson_r2_score']
    test_r2  = dc_model.evaluate(test,  metrics)['mean-pearson_r2_score']
    avg_loss = running_loss / len(train)
    print(f"Epoch {epoch:03d} — Loss: {avg_loss:.4f}  "
          f"Train R²: {train_r2:.3f}  Test R²: {test_r2:.3f}")





import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss

from sklearn.linear_model import LinearRegression

from scipy.stats import pearsonr
from scipy.stats import norm
from scipy.stats import bootstrap
from scipy.integrate import quad

from bisect import bisect_left


def order_sig_and_errors(sigmas, errors):
    ordered_df = pd.DataFrame()
    ordered_df["uq"] = sigmas
    ordered_df["errors"] = errors
    ordered_df["abs_z"] = abs(ordered_df.errors)/ordered_df.uq
    ordered_df = ordered_df.sort_values(by="uq")
    return ordered_df


def spearman_rank_corr(v1, v2):
    v1_ranked = ss.rankdata(v1)
    v2_ranked = ss.rankdata(v2)
    return pearsonr(v1_ranked, v2_ranked)


def rmse(x, axis=None):
    return np.sqrt((x**2).mean())


def get_bootstrap_intervals(errors_ordered, Nbins=10):
    """
    calculate the confidence intervals at a given p-level. 
    """
    ci_low = []
    ci_high = []
    N_total = len(errors_ordered)
    N_entries = math.ceil(N_total/Nbins)

    for i in range(0, N_total, N_entries):
        data = errors_ordered[i:i+N_entries]
        res = bootstrap((data,), rmse, vectorized=False)
        ci_low.append(res.confidence_interval[0])
        ci_high.append(res.confidence_interval[1])
    return ci_low, ci_high


def expected_rho(uncertainties):
    """
    for each uncertainty we draw a random Gaussian error to simulate the expected errors
    the spearman rank coeff. is then calculated between uncertainties and errors. 
    """
    sim_errors = []
    for sigma in uncertainties:
        error = np.abs(np.random.normal(0, sigma))
        sim_errors.append(error)
    
    rho, _ = spearman_rank_corr(uncertainties, sim_errors)
    return rho, sim_errors


def NLL(uncertainties, errors):
    NLL = 0
    for uncertainty, error in zip(uncertainties, errors):
        temp = math.log(2*np.pi*uncertainty**2)+(error)**2/uncertainty**2
        NLL += temp
    
    NLL = NLL/(2*len(uncertainties))
    return NLL


def calibration_curve(errors_sigma):
    N_errors = len(errors_sigma)
    gaus_pred = []
    errors_observed = []
    for i in np.arange(-10, 0+0.01, 0.01):
        gaus_int = 2*norm(loc=0, scale=1).cdf(i)
        gaus_pred.append(gaus_int)
        observed_errors = (errors_sigma > abs(i)).sum()
        errors_frac = observed_errors/N_errors
        errors_observed.append(errors_frac)
        
    return gaus_pred, errors_observed


def plot_calibration_curve(gaus_pred, errors_observed, mis_cal):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.fill_between(gaus_pred, gaus_pred, errors_observed, color="purple", alpha=0.4, label="miscalibration area = {:0.3f}".format(mis_cal))
    ax.plot(gaus_pred, errors_observed, color="purple", alpha=1)
    ax.plot(np.arange(0,1,0.01),np.arange(0,1,0.01), linestyle='dashed', color='k')
    ax.set_xlabel("expected fraction of errors", fontsize=14)
    ax.set_ylabel("observed fraction of errors", fontsize=14)
    ax.legend(fontsize=14, loc='lower right')
    return fig


def plot_Z_scores(errors, uncertainties):
    Z_scores = errors/uncertainties
    N_bins = 29
    xmin, xmax = -7,7
    y, bin_edges = np.histogram(Z_scores, bins=N_bins, range=(xmin, xmax))
    bin_width = bin_edges[1] - bin_edges[0]
    x = 0.5*(bin_edges[1:] + bin_edges[:-1])
    sy = np.sqrt(y)
    target_values = np.array([len(errors)*bin_width*norm.pdf(x_value) for x_value in x])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(Z_scores, bins=N_bins, range=(xmin, xmax), color='purple', alpha=0.3)
    ax.errorbar(x, y, sy, fmt='.', color='k')
    ax.plot(np.arange(-7, 7, 0.1), len(errors)*bin_width*norm.pdf(np.arange(-7, 7, 0.1), 0, 1), color='k')
    ax.set_xlabel("error (Z)", fontsize=16)
    ax.set_ylabel("count", fontsize=16)
    ax.set_xlim([-7, 7])
    return fig, ax

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0], myList[1], 0, 1
    if pos == len(myList):
        return myList[-2], myList[-1], -2, -1
    before = myList[pos - 1]
    after = myList[pos]
    if myNumber<before or myNumber>after:
        print("problem")
    else:
        return before, after, pos-1, pos


def f_linear_segment(x, point_list=None, x_list=None):
    
    x1, x2, x1_idx, x2_idx = take_closest(x_list, x) 
    f = point_list[x1_idx]+(x-x1)/(x2-x1)*(point_list[x2_idx]-point_list[x1_idx])
    
    return f


def area_function(x, observed_list, predicted_list):
    h = abs((f_linear_segment(x, observed_list, predicted_list)-x))
    return h


def calibration_area(observed, predicted):
    area = 0
    x = min(predicted)
    while x < max(predicted):
        temp, _ = quad(area_function, x, x+0.001, args=(observed, predicted))
        area += temp
        x += 0.001
    return area


def chi_squared(x_values, x_sigmas, target_values):
    mask = x_values > 0
    chi_value = ((x_values[mask]-target_values[mask])/x_sigmas[mask])**2
    chi_value = np.sum(chi_value)
    

    N_free_cs = len(x_values[mask])
    print(N_free_cs)
    chi_prob =  ss.chi2.sf(chi_value, N_free_cs)
    return chi_value, chi_prob


def get_slope_metric(uq_ordered, errors_ordered, Nbins=10, include_bootstrap=True):
    """
    Calculates the error-based calibration metrices

    uq_ordered: list of uncertainties in increasing order
    error_ordered: list of observed errors corresponding to the uncertainties in uq_ordered
    NBins: integer deciding how many bins to use for the error-based calibration metric
    include_bootstrap: boolean deciding wiether to include 95% confidence intervals on RMSE values from bootstrapping
    """
    rmvs, rmses, ci_low, ci_high = get_rmvs_and_rmses(uq_ordered, errors_ordered, Nbins=Nbins, include_bootstrap=include_bootstrap)
    
    x = np.array(rmvs).reshape((-1, 1))
    y = np.array(rmses)
    model = LinearRegression().fit(x, y)


    # Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
    r_sq = model.score(x, y)
    print('R squared:', r_sq)

    # Print the Intercept:
    intercept = model.intercept_
    print('intercept:', intercept)

    # Print the Slope:
    slope = model.coef_[0]
    print('slope:', slope) 

    # Predict a Response and print it:
    y_pred = model.predict(x)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    assymetric_errors = [np.array(rmses)-np.array(ci_low), np.array(ci_high)-np.array(rmses)]
    ax.errorbar(x, y, yerr = assymetric_errors, fmt="o", linewidth=2)
    ax.plot(np.arange(rmvs[0],rmvs[-1],0.0001),np.arange(rmvs[0],rmvs[-1],0.0001), linestyle='dashed', color='k')
    ax.plot(rmvs, y_pred, linestyle="dashed", color='red', label=r'$R^2$ = '+"{:0.2f}".format(r_sq)+", slope = {:0.2f}".format(slope)+", intercept = {:0.2f}".format(intercept))

    ax.set_xlabel("RMV", fontsize=14)
    ax.set_ylabel("RMSE", fontsize=14)
    ax.legend(fontsize=14)
    return fig, slope, r_sq, intercept


def get_rmvs_and_rmses(uq_ordered, errors_ordered, Nbins=10, include_bootstrap=True):
    """
    uq orderes should be the list of uncertainties in increasing order and errors should be the corresponding errors
    Nbins determine how many bins the data should be divided into
    """

    N_total = len(uq_ordered)
    N_entries = math.ceil(N_total/Nbins)
    #print(N_entries)
    rmvs = [np.sqrt((uq_ordered[i:i+N_entries]**2).mean()) for i in range(0, N_total, N_entries)]
    #print(rmvs)
    rmses = [np.sqrt((errors_ordered[i:i+N_entries]**2).mean()) for i in range(0, N_total, N_entries)]
    if include_bootstrap:
        ci_low, ci_high = get_bootstrap_intervals(errors_ordered, Nbins=Nbins)
    else:
        ci_low, ci_high = None, None
    return rmvs, rmses, ci_low, ci_high

# =============================================================================
# df_uq = pd.DataFrame()
# df_uq['true_logP'] = test.y[:,0]
# df_uq['pred_logP'] = means[:,0]
# df_uq['true_qed'] =test.y[:,1]
# df_uq['pred_qed'] = means[:,1]
# 
# df_uq['error'] = df_uq.pred_logP - df_uq.true_logP
# df_uq['uq'] = stds[:,0]
# =============================================================================


df_uq = pd.DataFrame()
df_uq['true_logP'] = true_values
df_uq['pred_logP'] = pred_means

df_uq['error'] = df_uq.pred_logP - df_uq.true_logP
df_uq['uq'] = test_stds[:,0]
df_uq_train = df_uq.sample(n=10000,random_state=42)
df_uq1 = df_uq.drop(df_uq_train.index)
df_uq1.reset_index(drop=True,inplace=True)
df_uq_train.reset_index(drop=True,inplace=True)
#Order uncertainties and errors according to uncertainties
df_uq_train = df_uq
ordered_df = order_sig_and_errors(df_uq_train.uq, df_uq_train.error)

#calculate rho_rank and rho_rank_sim
rho_rank, _ = spearman_rank_corr(np.abs(df_uq_train.error), df_uq_train.uq)
print(f'rho_rank = {rho_rank:.2f}')
exp_rhos_temp = []
for i in range(1000):
    exp_rho, _ = expected_rho(df_uq_train.uq)
    exp_rhos_temp.append(exp_rho)
rho_rank_sim = np.mean(exp_rhos_temp)
rho_rank_sim_std = np.std(exp_rhos_temp)
print(f'rho_rank_sim = {rho_rank_sim:.2f} +/- {rho_rank_sim_std:.2f}')

#Calculate the miscalibration area
gaus_pred, errors_observed = calibration_curve(ordered_df.abs_z)
mis_cal = calibration_area(errors_observed, gaus_pred)
print(f'miscalibration area = {mis_cal:.2f}')

#Calculate NLL and simulated NLL
_NLL = NLL(df_uq_train.uq, df_uq_train.error)
print(f'NLL = {_NLL:.2f}')
exp_NLL = []
for i in range(1000):
    sim_errors = []
    for sigma in df_uq_train.uq:
        sim_error = np.random.normal(0, sigma)
        sim_errors.append(sim_error)
    NLL_sim = NLL(df_uq_train.uq, sim_errors)
    exp_NLL.append(NLL_sim)
NLL_sim = np.mean(exp_NLL)
NLL_sim_std = np.std(exp_NLL)
print(f'NLL_sim = {NLL_sim:.2f} +/- {NLL_sim_std:.2f}')

#generate error-based calibration plot
fig, slope, R_sq, intercept = get_slope_metric(ordered_df.uq, ordered_df.errors, Nbins=20)

#Generate Z-score plot and calibration curve
fig2, ax2 = plot_Z_scores(ordered_df.errors, ordered_df.uq)
fig3 = plot_calibration_curve(gaus_pred, errors_observed, mis_cal)



df_uq1['uq_recalibrated'] = slope * df_uq1['uq'] + intercept

# Optional: Handle potential non-positive recalibrated uncertainties
# If the intercept is negative, some low uq values might become non-positive
min_uncertainty = 1e-6 # Define a small positive floor value
#df_uq.loc[df_uq['uq_recalibrated'] <= 0, 'uq_recalibrated'] = min_uncertainty
print(f"Applied small floor value ({min_uncertainty}) to any non-positive recalibrated uncertainties.")

print("\nRecalibrated uncertainties added to df_uq:")
print(df_uq1[['uq', 'uq_recalibrated', 'error']].head())

# --- Step 3: Evaluate the Recalibrated Uncertainties (Optional) ---
# You can now re-run your evaluation metrics and plots using df_uq['uq_recalibrated']

print("\nEvaluating recalibrated uncertainties on df_uq (test set)...")

# Order the test data using recalibrated uncertainties
ordered_df_recal = order_sig_and_errors(df_uq1['uq_recalibrated'], df_uq1['error'])

# Example: Generate error-based calibration plot for recalibrated data
fig_recal, slope_recal, R_sq_recal, intercept_recal = get_slope_metric(
    ordered_df_recal['uq'], ordered_df_recal['errors'], Nbins=20, include_bootstrap=True
)
print(f"\nRecalibrated Test Set Fit:")
print(f"Slope: {slope_recal:.4f}")
print(f"Intercept: {intercept_recal:.4f}")
print(f"R_squared: {R_sq_recal:.4f}")
# fig_recal.show() # Or save the figure

# Example: Calculate NLL for recalibrated data
_NLL_recal = NLL(df_uq1['uq_recalibrated'], df_uq1['error'])
print(f'\nRecalibrated NLL = {_NLL_recal:.4f}')

# Example: Calculate miscalibration area for recalibrated data
gaus_pred_recal, errors_observed_recal = calibration_curve(ordered_df_recal['abs_z'])
mis_cal_recal = calibration_area(errors_observed_recal, gaus_pred_recal)
print(f'Recalibrated miscalibration area = {mis_cal_recal:.4f}')
fig_z_recal, ax_z_recal = plot_Z_scores(ordered_df_recal['errors'], ordered_df_recal['uq'])
fig_cal_recal = plot_calibration_curve(gaus_pred_recal, errors_observed_recal, mis_cal_recal)




#%%


import os, sys
import multiprocessing
import random
import yaml
from functools import partial
from collections import OrderedDict
from typing import Callable, List, Optional

import numpy as np

from crossover import crossover_smiles
from mutate import mutate_smiles
from network import create_and_train_network, obtain_model_pred
from utils import sanitize_smiles, get_fp_scores
from fragment import form_fragments


class JANUS:
    """ JANUS class for genetic algorithm applied on SELFIES
    string representation.
    See example/example.py for descriptions of parameters
    """

    def __init__(
        self,
        work_dir: str,
        fitness_function: Callable,
        start_population: list,
        verbose_out: Optional[bool] = False,
        custom_filter: Optional[Callable] = None,
        alphabet: Optional[List[str]] = None,
        use_gpu: Optional[bool] = False,
        num_workers: Optional[int] = None,
        generations: Optional[int] = 200,
        generation_size: Optional[int] = 5000,
        num_exchanges: Optional[int] = 5,
        use_fragments: Optional[bool] = True,
        num_sample_frags: Optional[int] = 10,
        use_classifier: Optional[bool] = False,
        explr_num_random_samples: Optional[int] = 5,
        explr_num_mutations: Optional[int] = 5,
        crossover_num_random_samples: Optional[int] = 1,
        exploit_num_random_samples: Optional[int] = 5,
        exploit_num_mutations: Optional[int] = 5,
        top_mols: Optional[int] = 1
    ):

        # set all class variables
        self.work_dir = work_dir
        self.fitness_function = fitness_function
        self.start_population = start_population
        self.verbose_out = verbose_out
        self.custom_filter = custom_filter
        self.alphabet = alphabet
        self.use_gpu = use_gpu
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()
        self.generations = generations
        self.generation_size = generation_size
        self.num_exchanges = num_exchanges
        self.use_fragments = use_fragments
        self.num_sample_frags = num_sample_frags
        self.use_classifier = use_classifier
        self.explr_num_random_samples = explr_num_random_samples
        self.explr_num_mutations = explr_num_mutations
        self.crossover_num_random_samples = crossover_num_random_samples
        self.exploit_num_random_samples = exploit_num_random_samples
        self.exploit_num_mutations = exploit_num_mutations
        self.top_mols = top_mols

        # create dump folder
        if not os.path.isdir(f"./{self.work_dir}"):
            os.mkdir(f"./{self.work_dir}")
        self.save_hyperparameters()

        # get initial population
        init_smiles, init_fitness = self.start_population, []
        
# =============================================================================
#         with open(self.start_population, "r") as f:
#             for line in f:
#                 line = sanitize_smiles(line.strip())
#                 if line is not None:
#                     init_smiles.append(line)
#         # init_smiles = list(set(init_smiles)) 
# =============================================================================

        # check that parameters are valid
        assert (
            len(init_smiles) >= self.generation_size
        ), "Initial population smaller than generation size."
        assert (
            self.top_mols <= self.generation_size
        ), "Number of top molecules larger than generation size."

        # make fragments from initial smiles
        from fragment import form_fragments
        self.frag_alphabet = []
        if self.use_fragments:
            with multiprocessing.Pool(self.num_workers) as pool:
                frags = pool.map(form_fragments, init_smiles)
            frags = self.flatten_list(frags)
            print(f"    Unique and valid fragments generated: {len(frags)}")
            self.frag_alphabet.extend(frags)

        # get initial fitness
        # with multiprocessing.Pool(self.num_workers) as pool:
        #     init_fitness = pool.map(self.fitness_function, init_smiles)
        init_fitness = []
        for smi in init_smiles:
            init_fitness.append(self.fitness_function(smi))#.detach().numpy())

        # sort the initial population and save in class
        idx = np.argsort(init_fitness)[::-1]
        init_smiles = np.array(init_smiles)[idx]
        init_fitness = np.array(init_fitness)[idx]
        self.population = init_smiles[: self.generation_size]
        self.fitness = init_fitness[: self.generation_size]

        with open(os.path.join(self.work_dir, "init_mols.txt"), "w") as f:
            f.writelines([f"{x}\n" for x in self.population])

        # store in collector, deal with duplicates
        self.smiles_collector = {}
        uniq_pop, idx, counts = np.unique(
            self.population, return_index=True, return_counts=True
        )
        for smi, count, i in zip(uniq_pop, counts, idx):
            self.smiles_collector[smi] = [self.fitness[i], count]

    def mutate_smi_list(self, smi_list: List[str], space="local"):
        # parallelized mutation function
        if space == "local":
            num_random_samples = self.exploit_num_random_samples
            num_mutations = self.exploit_num_mutations
        elif space == "explore":
            num_random_samples = self.explr_num_random_samples
            num_mutations = self.explr_num_mutations
        else:
            raise ValueError('Invalid space, choose "local" or "explore".')

        smi_list = smi_list * num_random_samples
        with multiprocessing.Pool(self.num_workers) as pool:
            mut_smi_list = pool.map(
                partial(
                    mutate_smiles,
                    alphabet=self.frag_alphabet,
                    num_random_samples=1,
                    num_mutations=num_mutations,
                    num_sample_frags=self.num_sample_frags,
                    base_alphabet=self.alphabet
                ),
                smi_list,
            )
        mut_smi_list = self.flatten_list(mut_smi_list)
        return mut_smi_list

    def crossover_smi_list(self, smi_list: List[str]):
        # parallelized crossover function
        with multiprocessing.Pool(self.num_workers) as pool:
            cross_smi = pool.map(
                partial(
                    crossover_smiles,
                    crossover_num_random_samples=self.crossover_num_random_samples,
                ),
                smi_list,
            )
        cross_smi = self.flatten_list(cross_smi)
        return cross_smi

    def check_filters(self, smi_list: List[str]):
        if self.custom_filter is not None:
            smi_list = [smi for smi in smi_list if self.custom_filter(smi)]
        return smi_list

    def save_hyperparameters(self):
        hparams = {
            k: v if not callable(v) else v.__name__ for k, v in vars(self).items()
        }
        with open(os.path.join(self.work_dir, "hparams.yml"), "w") as f:
            yaml.dump(hparams, f)

    def run(self):
        """ Run optimization based on hyperparameters initialized
        """
        all_smiles_return =[]
        for gen_ in range(self.generations):

            # bookkeeping
            if self.verbose_out:
                output_dir = os.path.join(self.work_dir, f"{gen_}_DATA")
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)

            print(f"On generation {gen_}/{self.generations}")

            keep_smiles, replace_smiles = self.get_good_bad_smiles(
                self.fitness, self.population, self.generation_size
            )
            replace_smiles = list(set(replace_smiles))

            ### EXPLORATION ###
            # Mutate and crossover (with keep_smiles) molecules that are meant to be replaced
            explr_smiles = []
            timeout_counter = 0
            while len(explr_smiles) < self.generation_size-len(keep_smiles):
                # Mutations:
                mut_smi_explr = self.mutate_smi_list(
                    replace_smiles[0 : len(replace_smiles) // 2], space="explore"
                )
                mut_smi_explr = self.check_filters(mut_smi_explr)

                # Crossovers:
                smiles_join = []
                for item in replace_smiles[len(replace_smiles) // 2 :]:
                    smiles_join.append(item + "xxx" + random.choice(keep_smiles))
                cross_smi_explr = self.crossover_smi_list(smiles_join)
                cross_smi_explr = self.check_filters(cross_smi_explr)

                # Combine and get unique smiles not yet found
                all_smiles = list(set(mut_smi_explr + cross_smi_explr))
                for x in all_smiles:
                    if x not in self.smiles_collector:
                        explr_smiles.append(x)
                explr_smiles = list(set(explr_smiles))

                timeout_counter += 1
                if timeout_counter % 100 == 0:
                    print(f'Exploration: {timeout_counter} iterations of filtering. \
                    Filter may be too strict, or you need more mutations/crossovers.')

            # Replace the molecules with ones in exploration mutation/crossover
            if not self.use_classifier or gen_ == 0:
                replaced_pop = random.sample(
                    explr_smiles, self.generation_size - len(keep_smiles)
                )
            else:
                # The sampling needs to be done by the neural network!
                print("    Training classifier neural net...")
                train_smiles, targets = [], []
                for item in self.smiles_collector:
                    train_smiles.append(item)
                    targets.append(self.smiles_collector[item][0])
                net = create_and_train_network(
                    train_smiles,
                    targets,
                    num_workers=self.num_workers,
                    use_gpu=self.use_gpu,
                )

                # Obtain predictions on unseen molecules:
                print("    Obtaining Predictions")
                new_predictions = obtain_model_pred(
                    explr_smiles,
                    net,
                    num_workers=self.num_workers,
                    use_gpu=self.use_gpu,
                )
                sorted_idx = np.argsort(np.squeeze(new_predictions))[::-1]
                replaced_pop = np.array(explr_smiles)[
                    sorted_idx[: self.generation_size - len(keep_smiles)]
                ].tolist()

            # Calculate actual fitness for the exploration population
            self.population = keep_smiles + replaced_pop
            self.fitness = []
            for smi in self.population:
                if smi in self.smiles_collector:
                    # if already calculated, use the value from smiles collector
                    self.fitness.append(self.smiles_collector[smi][0])
                    self.smiles_collector[smi][1] += 1
                else:
                    # make a calculation
                    f = self.fitness_function(smi)
                    self.fitness.append(f)
                    self.smiles_collector[smi] = [f, 1]

            # Print exploration data
            idx_sort = np.argsort(self.fitness)[::-1]
            all_smiles_return.append(self.population)
            print(f"    (Explr) Top Fitness: {self.fitness[idx_sort[0]]}")
            print(f"    (Explr) Top Smile: {self.population[idx_sort[0]]}")

            fitness_sort = np.array(self.fitness)[idx_sort]
            if self.verbose_out:
                with open(
                    os.path.join(
                        self.work_dir, str(gen_) + "_DATA", "fitness_explore.txt"
                    ),
                    "w",
                ) as f:
                    f.writelines(["{} ".format(x) for x in fitness_sort])
                    f.writelines(["\n"])
            else:
                with open(os.path.join(self.work_dir, "fitness_explore.txt"), "w") as f:
                    f.writelines(["{} ".format(x) for x in fitness_sort])
                    f.writelines(["\n"])

            # this population is sort by modified fitness, if active
            population_sort = np.array(self.population)[idx_sort]
            if self.verbose_out:
                with open(
                    os.path.join(
                        self.work_dir, str(gen_) + "_DATA", "population_explore.txt"
                    ),
                    "w",
                ) as f:
                    f.writelines(["{} ".format(x) for x in population_sort])
                    f.writelines(["\n"])
            else:
                with open(
                    os.path.join(self.work_dir, "population_explore.txt"), "w"
                ) as f:
                    f.writelines(["{} ".format(x) for x in population_sort])
                    f.writelines(["\n"])

            ### EXPLOITATION ###
            # Conduct local search on top-n molecules from population, mutate and do similarity search
            exploit_smiles = []
            timeout_counter = 0
            while len(exploit_smiles) < self.generation_size:
                smiles_local_search = population_sort[0 : self.top_mols].tolist()
                mut_smi_loc = self.mutate_smi_list(smiles_local_search, "local")
                mut_smi_loc = self.check_filters(mut_smi_loc)

                # filter out molecules already found
                for x in mut_smi_loc:
                    if x not in self.smiles_collector:
                        exploit_smiles.append(x)

                timeout_counter += 1
                if timeout_counter % 100 == 0:
                    print(f'Exploitation: {timeout_counter} iterations of filtering. \
                    Filter may be too strict, or you need more mutations/crossovers.')

            # sort by similarity, only keep ones similar to best
            fp_scores = get_fp_scores(exploit_smiles, population_sort[0])
            fp_sort_idx = np.argsort(fp_scores)[::-1][: self.generation_size]
            # highest fp_score idxs
            self.population_loc = np.array(exploit_smiles)[
                fp_sort_idx
            ]  # list of smiles with highest fp scores

            # STEP 4: CALCULATE THE FITNESS FOR THE LOCAL SEARCH:
            # Exploitation data generated from similarity search is measured with fitness function
            self.fitness_loc = []
            for smi in self.population_loc:
                f = self.fitness_function(smi)
                self.fitness_loc.append(f)
                self.smiles_collector[smi] = [f, 1]

            # List of original local fitness scores
            idx_sort = np.argsort(self.fitness_loc)[
                ::-1
            ]  # index of highest to lowest fitness scores
            print(f"    (Local) Top Fitness: {self.fitness_loc[idx_sort[0]]}")
            print(f"    (Local) Top Smile: {self.population_loc[idx_sort[0]]}")

            fitness_sort = np.array(self.fitness_loc)[idx_sort]
            all_smiles_return.append(self.population_loc)
            if self.verbose_out:
                with open(
                    os.path.join(
                        self.work_dir, str(gen_) + "_DATA", "fitness_local_search.txt"
                    ),
                    "w",
                ) as f:
                    f.writelines(["{} ".format(x) for x in fitness_sort])
                    f.writelines(["\n"])
            else:
                with open(
                    os.path.join(self.work_dir, "fitness_local_search.txt"), "w"
                ) as f:
                    f.writelines(["{} ".format(x) for x in fitness_sort])
                    f.writelines(["\n"])

            population_sort = np.array(self.population_loc)[idx_sort]
            if self.verbose_out:
                with open(
                    os.path.join(
                        self.work_dir,
                        str(gen_) + "_DATA",
                        "population_local_search.txt",
                    ),
                    "w",
                ) as f:
                    f.writelines(["{} ".format(x) for x in population_sort])
                    f.writelines(["\n"])
            else:
                with open(
                    os.path.join(self.work_dir, "population_local_search.txt"), "w"
                ) as f:
                    f.writelines(["{} ".format(x) for x in population_sort])
                    f.writelines(["\n"])

            # STEP 5: EXCHANGE THE POPULATIONS:
            # Introduce changes to 'fitness' & 'population'
            best_smi_local = population_sort[0 : self.num_exchanges]
            best_fitness_local = fitness_sort[0 : self.num_exchanges]

            # But will print the best fitness values in file
            idx_sort = np.argsort(self.fitness)[
                ::-1
            ]  # sorted indices for the entire population
            worst_indices = idx_sort[
                -self.num_exchanges :
            ]  # replace worst ones with the best ones
            for i, idx in enumerate(worst_indices):
                try:
                    self.population[idx] = best_smi_local[i]
                    self.fitness[idx] = best_fitness_local[i]
                except:
                    continue

            # Save best of generation!:
            fit_all_best = np.argmax(self.fitness)

            # write best molecule with best fitness
            with open(
                os.path.join(self.work_dir, "generation_all_best.txt"), "a+"
            ) as f:
                f.writelines(
                    f"Gen:{gen_}, {self.population[fit_all_best]}, {self.fitness[fit_all_best]} \n"
                )

        return all_smiles_return

    @staticmethod
    def get_good_bad_smiles(fitness, population, generation_size):
        """
        Given fitness values of all SMILES in population, and the generation size, 
        this function smplits  the population into two lists: keep_smiles & replace_smiles. 
        
        Parameters
        ----------
        fitness : (list of floats)
            List of floats representing properties for molecules in population.
        population : (list of SMILES)
            List of all SMILES in each generation.
        generation_size : (int)
            Number of molecules in each generation.

        Returns
        -------
        keep_smiles : (list of SMILES)
            A list of SMILES that will be untouched for the next generation. .
        replace_smiles : (list of SMILES)
            A list of SMILES that will be mutated/crossed-oved for forming the subsequent generation.

        """

        fitness = np.array(fitness)
        idx_sort = fitness.argsort()[::-1]  # Best -> Worst
        keep_ratio = 0.2
        keep_idx = int(len(list(idx_sort)) * keep_ratio)
        try:

            F_50_val = fitness[idx_sort[keep_idx]]
            F_25_val = np.array(fitness) - F_50_val
            F_25_val = np.array([x for x in F_25_val if x < 0]) + F_50_val
            F_25_sort = F_25_val.argsort()[::-1]
            F_25_val = F_25_val[F_25_sort[0]]

            prob_ = 1.0 / (3.0 ** ((F_50_val - fitness) / (F_50_val - F_25_val)) + 1)

            prob_ = prob_ / sum(prob_)
            to_keep = np.random.choice(generation_size, keep_idx, p=prob_)
            to_replace = [i for i in range(generation_size) if i not in to_keep][
                0 : generation_size - len(to_keep)
            ]

            keep_smiles = [population[i] for i in to_keep]
            replace_smiles = [population[i] for i in to_replace]

            best_smi = population[idx_sort[0]]
            if best_smi not in keep_smiles:
                keep_smiles.append(best_smi)
                if best_smi in replace_smiles:
                    replace_smiles.remove(best_smi)

            if keep_smiles == [] or replace_smiles == []:
                raise Exception("Badly sampled population!")
        except:
            keep_smiles = [population[i] for i in idx_sort[:keep_idx]]
            replace_smiles = [population[i] for i in idx_sort[keep_idx:]]

        return keep_smiles, replace_smiles

    def log(self):
        pass

    @staticmethod
    def flatten_list(nested_list):
        return [item for sublist in nested_list for item in sublist]



from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, RDConfig, Descriptors
RDLogger.DisableLog("rdApp.*")

import torch
import selfies

import selfies as sf, numpy as np, torch, random
from rdkit import Chem
def smi_to_dgl(smiles):
    """SMILES → DGLGraph using *your* DeepChem featurizer."""
    gd = featurizer.featurize([smiles])[0]                     # GraphData
    if not hasattr(gd, "to_dgl_graph"):        # featuriser returned empty array
        return None
    return gd.to_dgl_graph(self_loop=True)

def graph_embed(g):
    h = pt_predictor.gnn(g, g.ndata['x'], g.edata['edge_attr'])
    return pt_predictor.readout(g, h)                          # [1,200]

def fitness_function(smi: str) -> float:

    g = smi_to_dgl(smi).to(device)
    emb0 = graph_embed(g).detach().cpu().numpy().astype(np.float64).ravel()

    emb_torch = torch.tensor(emb0, dtype=torch.float64, device=device).view(1, -1)
    mu, logvar = pt_predictor.predict(emb_torch.float())
    
    mu = mu.detach().numpy()
    sigma = torch.exp(0.5 * logvar)
    sigma = sigma.detach().numpy()
    
    
    #final_fitness = 2.75*float(mu[0][0]) - 1.25*float(mu[0][1]) #+1.2*float(sigma[0][0])-1.25*float(sigma[0][1])
    final_fitness = 1.0*float(mu[0][0]) - 1.0*float(mu[0][1])
    return final_fitness
    

def custom_filter(smi: str):
    """ Function that takes in a smile and returns a boolean.
    True indicates the smiles PASSES the filter.
    """
    # smiles length filter
    if len(smi) > 100 or len(smi) == 0:
        return False
    else:
        return True

torch.multiprocessing.freeze_support()

SEEDS    = [2000, 1234, 3452, 2, 12]
POP_SIZE = 1000

seed_smiles_dict = {
    s: df["smiles"].dropna().sample(POP_SIZE, random_state=s).tolist()
    for s in SEEDS
}


LOGP_MIN, LOGP_MAX = df["logP"].min(), df["logP"].max()
TPSA_MIN, TPSA_MAX = df["TPSA"].min(), df["TPSA"].max()

def scale01(x, lo, hi):
    return (max(min(x, hi), lo) - lo) / (hi - lo)

W_LOGP, W_TPSA = 1.0,1.0

@torch.no_grad()
def fitness_function(smi: str) -> float:
    g = smi_to_dgl(smi)
    if g is None:
        return -1e6
    g = g.to(device)
    h  = pt_predictor.gnn(g, g.ndata['x'], g.edata['edge_attr'])
    gF = pt_predictor.readout(g, h)
    preds = pt_predictor.predict(gF)[0].cpu()          # [logP, TPSA]
    lp, tpsa = float(preds[0]), float(preds[1])
    z_lp   = scale01(lp, LOGP_MIN, LOGP_MAX)
    z_tpsa = 1.0 - scale01(tpsa, TPSA_MIN, TPSA_MAX)
    return W_LOGP*z_lp + W_TPSA*z_tpsa

# ────────────────────────────────────────────────────────────────────
# 2)  HV helper that takes a *reference point argument*
# ────────────────────────────────────────────────────────────────────
from botorch.utils.multi_objective.hypervolume import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated

@torch.no_grad()
def hv_of_smiles(smiles_list, ref_pt):
    logp, mTPSA = [], []
    for s in smiles_list:
        g = smi_to_dgl(s)
        if g is None:  continue
        g = g.to(device)
        h  = pt_predictor.gnn(g, g.ndata['x'], g.edata['edge_attr'])
        gF = pt_predictor.readout(g, h)
        preds = pt_predictor.predict(gF)[0].cpu()
        logp.append(float(preds[0]))
        mTPSA.append(-float(preds[1]))        # minus TPSA
    Y = torch.tensor(np.column_stack([logp, mTPSA]), dtype=torch.double)
    pareto = Y[is_non_dominated(Y)]
    return DominatedPartitioning(ref_pt, pareto).compute_hypervolume().item()

# ────────────────────────────────────────────────────────────────────
# 3)  JANUS hyper‑parameters
# ────────────────────────────────────────────────────────────────────
params_dict = dict(
    generations       = 20,
    generation_size   = 50,
    num_exchanges     = 0,
    custom_filter     = lambda s: 0 < len(s) <= 100,
    use_fragments     = True,
    use_classifier    = True,
)

# ────────────────────────────────────────────────────────────────────
# 4)  Run JANUS for each seed with per‑seed reference point
# ────────────────────────────────────────────────────────────────────
janus_results = {}

for seed in SEEDS:
    print(f"\n=== JANUS | seed {seed} ===")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    start_pool = seed_smiles_dict[seed]

    # ---- per‑seed reference identical to qPHV ----------------------
    mu0, _ = surrogate_mu_sigma(start_pool, n=64)      # (N,2)  logP , –TPSA
    ref_pt = mu0.min(0).values - 1.0   

    import os 
    

    # ---- JANUS run -------------------------------------------------
    agent = JANUS(
        work_dir         = f"RESULTS/multi_objective/new_runs/seed_{seed}",
        fitness_function = fitness_function,
        start_population = start_pool,
        **params_dict
    )

    hv_hist = [hv_of_smiles(start_pool, ref_pt)]       

    pops_by_gen = agent.run()                          
    for i in range(1, len(pops_by_gen), 2):            
        hv_hist.append( hv_of_smiles(pops_by_gen[i], ref_pt) )

    janus_results[seed] = {"hv_hist": hv_hist,
                           "final_pop": agent.population.copy()}


import matplotlib.pyplot as plt, numpy as np
rounds = np.arange(len(janus_results[SEEDS[0]]["hv_hist"]))
hv_mat = np.vstack([janus_results[s]["hv_hist"] for s in SEEDS])
mu, sd = hv_mat.mean(0), hv_mat.std(0)

plt.figure(figsize=(6,4))
plt.plot(rounds, mu, label="JANUS", color="tab:green")
plt.fill_between(rounds, mu-sd, mu+sd, color="tab:green", alpha=0.25)
plt.xlabel("Generation"); plt.ylabel("Hyper-volume")
plt.tight_layout(); plt.show()

