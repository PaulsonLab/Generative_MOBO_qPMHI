#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:14:20 2025

@author: muthyala.7
"""
#%%
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


# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AttentiveFP
# pylint: disable= no-member, arguments-differ, invalid-name

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import edge_softmax

__all__ = ['AttentiveFPGNN']

# pylint: disable=W0221, C0103, E1101
class AttentiveGRU1(nn.Module):
    """Update node features with attention and GRU.

    This will be used for incorporating the information of edge features
    into node features for message passing.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
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
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Previous edge features.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'c'))
        c = g.ndata['c']
        if not isinstance(c, torch.Tensor):
            c = torch.from_numpy(c).to(node_feats.device)
        context = F.elu(c)
        #context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class AttentiveGRU2(nn.Module):
    """Update node features with attention and GRU.

    This will be used in GNN layers for updating node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
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
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.u_mul_e('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class GetContext(nn.Module):
    """Generate context for each node by message passing at the beginning.

    This layer incorporates the information of edge features into node
    representations so that message passing needs to be only performed over
    node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    dropout : float
        The probability for performing dropout.
    """
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
        """Edge feature update.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges

        Returns
        -------
        dict
            Mapping ``'he1'`` to updated edge features.
        """
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        """Edge feature update.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges

        Returns
        -------
        dict
            Mapping ``'he2'`` to updated edge features.
        """
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        """Incorporate edge features and update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
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
    """GNNLayer for updating node features.

    This layer performs message passing over node representations and update them.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the graph representations to be computed.
    dropout : float
        The probability for performing dropout.
    """
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
        """Edge feature generation.

        Generate edge features by concatenating the features of the destination
        and source nodes.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges.

        Returns
        -------
        dict
            Mapping ``'he'`` to the generated edge features.
        """
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        """Perform message passing and update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.attentive_gru(g, logits, node_feats)

class AttentiveFPGNN(nn.Module):
    """`Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    This class performs message passing in AttentiveFP and returns the updated node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    graph_feat_size : int
        Size for the graph representations to be computed. Default to 200.
    dropout : float
        The probability for performing dropout. Default to 0.
    """
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
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        node_feats : float32 tensor of shape (V, graph_feat_size)
            Updated node representations.
        """
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        return node_feats
    



# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Readout for AttentiveFP
# pylint: disable= no-member, arguments-differ, invalid-name

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AttentiveFPReadout']

# pylint: disable=W0221
class GlobalPool(nn.Module):
    """One-step readout in AttentiveFP

    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    dropout : float
        The probability for performing dropout.
    """
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
        """Perform one-step readout

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Input graph features. G for the number of graphs.
        get_node_weight : bool
            Whether to get the weights of atoms during readout.

        Returns
        -------
        float32 tensor of shape (G, graph_feat_size)
            Updated graph features.
        float32 tensor of shape (V, 1)
            The weights of nodes in readout.
        """
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
    """Readout in AttentiveFP

    AttentiveFP is introduced in `Pushing the Boundaries of Molecular Representation for
    Drug Discovery with the Graph Attention Mechanism
    <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    This class computes graph representations out of node features.

    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    dropout : float
        The probability for performing dropout. Default to 0.
    """
    def __init__(self, feat_size, num_timesteps=2, dropout=0.):
        super(AttentiveFPReadout, self).__init__()

        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readouts.append(GlobalPool(feat_size, dropout))

    def forward(self, g, node_feats, get_node_weight=False):
        """Computes graph representations out of node features.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        get_node_weight : bool
            Whether to get the weights of nodes in readout. Default to False.

        Returns
        -------
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Graph representations computed. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
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
        
        
# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AttentiveFP
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn as nn


__all__ = ['AttentiveFPPredictor']

# pylint: disable=W0221
class AttentiveFPPredictor(nn.Module):
    """AttentiveFP for regression and classification on graphs.

    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    graph_feat_size : int
        Size for the learned graph representations. Default to 200.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    dropout : float
        Probability for performing the dropout. Default to 0.
    """
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
# =============================================================================
#         self.bayes_head = nn.Sequential(
#             nn.Dropout(dropout),
#             BayesianLinear(graph_feat_size, n_tasks)
#         )
#     def kl_loss(self):
#         # bayes_head[1] is our BayesianLinear
#         return self.bayes_head[1].kl_loss()
# =============================================================================
    
    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.
        get_node_weight : bool
            Whether to get the weights of atoms during readout. Default to False.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        if get_node_weight:
            g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats), node_weights
        else:
            g_feats = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats)
        
        
        
        
        
"""
DGL-based AttentiveFP for graph property prediction.
"""
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel


class AttentiveFP(nn.Module):
    """Model for Graph Property Prediction.

    This model proceeds as follows:

    * Combine node features and edge features for initializing node representations,
        which involves a round of message passing
    * Update node representations with multiple rounds of message passing
    * For each graph, compute its representation by combining the representations
        of all nodes in it, which involves a gated recurrent unit (GRU).
    * Perform the final prediction using a linear layer

    Examples
    --------

    >>> import deepchem as dc
    >>> import dgl
    >>> from deepchem.models import AttentiveFP
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    >>> graphs = featurizer.featurize(smiles)
    >>> print(type(graphs[0]))
    <class 'deepchem.feat.graph_data.GraphData'>
    >>> dgl_graphs = [graphs[i].to_dgl_graph(self_loop=True) for i in range(len(graphs))]
    >>> # Batch two graphs into a graph of two connected components
    >>> batch_dgl_graph = dgl.batch(dgl_graphs)
    >>> model = AttentiveFP(n_tasks=1, mode='regression')
    >>> preds = model(batch_dgl_graph)
    >>> print(type(preds))
    <class 'torch.Tensor'>
    >>> preds.shape == (2, 1)
    True

    References
    ----------
    .. [1] Zhaoping Xiong, Dingyan Wang, Xiaohong Liu, Feisheng Zhong, Xiaozhe Wan, Xutong Li,
        Zhaojun Li, Xiaomin Luo, Kaixian Chen, Hualiang Jiang, and Mingyue Zheng. "Pushing
        the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention
        Mechanism." Journal of Medicinal Chemistry. 2020, 63, 16, 8749–8760.

    Notes
    -----
    This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
    (https://github.com/awslabs/dgl-lifesci) to be installed.
    """

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
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks.
        num_layers: int
            Number of graph neural network layers, i.e. number of rounds of message passing.
            Default to 2.
        num_timesteps: int
            Number of time steps for updating graph representations with a GRU. Default to 2.
        graph_feat_size: int
            Size for graph representations. Default to 200.
        dropout: float
            Dropout probability. Default to 0.
        mode: str
            The model type, 'classification' or 'regression'. Default to 'regression'.
        number_atom_features: int
            The length of the initial atom feature vectors. Default to 30.
        number_bond_features: int
            The length of the initial bond feature vectors. Default to 11.
        n_classes: int
            The number of classes to predict per task
            (only used when ``mode`` is 'classification'). Default to 2.
        nfeat_name: str
            For an input graph ``g``, the model assumes that it stores node features in
            ``g.ndata[nfeat_name]`` and will retrieve input node features from that.
            Default to 'x'.
        efeat_name: str
            For an input graph ``g``, the model assumes that it stores edge features in
            ``g.edata[efeat_name]`` and will retrieve input edge features from that.
            Default to 'edge_attr'.
        """
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
        """Predict graph labels

        Parameters
        ----------
        g: DGLGraph
            A DGLGraph for a batch of graphs. It stores the node features in
            ``dgl_graph.ndata[self.nfeat_name]`` and edge features in
            ``dgl_graph.edata[self.efeat_name]``.

        Returns
        -------
        torch.Tensor
            The model output.

        * When self.mode = 'regression',
            its shape will be ``(dgl_graph.batch_size, self.n_tasks)``.
        * When self.mode = 'classification', the output consists of probabilities
            for classes. Its shape will be
            ``(dgl_graph.batch_size, self.n_tasks, self.n_classes)`` if self.n_tasks > 1;
            its shape will be ``(dgl_graph.batch_size, self.n_classes)`` if self.n_tasks is 1.
        torch.Tensor, optional
            This is only returned when self.mode = 'classification', the output consists of the
            logits for classes before softmax.
        """
        node_feats = g.ndata[self.nfeat_name]
        edge_feats = g.edata[self.efeat_name]
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
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features,
                 prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        # variational posterior params for weights
        self.weight_mu        = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_log_sigma = nn.Parameter(torch.full((out_features, in_features), -3.0))
        # and for bias
        self.bias_mu          = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_log_sigma   = nn.Parameter(torch.full((out_features,), -3.0))

        # prior
        self.prior_mu    = prior_mu
        self.prior_sigma = prior_sigma

    def forward(self, x):
        # sample ε ∼ N(0,1)
        w_sigma = torch.exp(self.weight_log_sigma)
        b_sigma = torch.exp(self.bias_log_sigma)
        w_eps   = torch.randn_like(w_sigma)
        b_eps   = torch.randn_like(b_sigma)

        # reparameterize
        weight = self.weight_mu + w_sigma * w_eps
        bias   = self.bias_mu   + b_sigma * b_eps

        return F.linear(x, weight, bias)

    def kl_loss(self):
        # closed‐form KL of q=N(μ,σ²) vs p=N(prior_mu,prior_sigma²)
        def _kl(q_mu, q_log_sigma):
            q_sigma = torch.exp(q_log_sigma)
            p_sigma = self.prior_sigma
            p_mu    = self.prior_mu
            return ( 
                torch.log(p_sigma / q_sigma)
                + (q_sigma.pow(2) + (q_mu - p_mu).pow(2)) / (2 * p_sigma**2)
                - 0.5
            ).sum()

        return _kl(self.weight_mu, self.weight_log_sigma) + _kl(self.bias_mu, self.bias_log_sigma)


class AttentiveFPModel(TorchModel):
    """Model for Graph Property Prediction.

    This model proceeds as follows:

    * Combine node features and edge features for initializing node representations,
        which involves a round of message passing
    * Update node representations with multiple rounds of message passing
    * For each graph, compute its representation by combining the representations
        of all nodes in it, which involves a gated recurrent unit (GRU).
    * Perform the final prediction using a linear layer

    Examples
    --------
    >>> import deepchem as dc
    >>> from deepchem.models import AttentiveFPModel
    >>> # preparing dataset
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> labels = [0., 1.]
    >>> featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    >>> X = featurizer.featurize(smiles)
    >>> dataset = dc.data.NumpyDataset(X=X, y=labels)
    >>> # training model
    >>> model = AttentiveFPModel(mode='classification', n_tasks=1,
    ...    batch_size=16, learning_rate=0.001)
    >>> loss = model.fit(dataset, nb_epoch=5)

    References
    ----------
    .. [1] Zhaoping Xiong, Dingyan Wang, Xiaohong Liu, Feisheng Zhong, Xiaozhe Wan, Xutong Li,
        Zhaojun Li, Xiaomin Luo, Kaixian Chen, Hualiang Jiang, and Mingyue Zheng. "Pushing
        the Boundaries of Molecular Representation for Drug Discovery with the Graph
        Attention Mechanism." Journal of Medicinal Chemistry. 2020, 63, 16, 8749–8760.

    Notes
    -----
    This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
    (https://github.com/awslabs/dgl-lifesci) to be installed.
    """

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
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks.
        num_layers: int
            Number of graph neural network layers, i.e. number of rounds of message passing.
            Default to 2.
        num_timesteps: int
            Number of time steps for updating graph representations with a GRU. Default to 2.
        graph_feat_size: int
            Size for graph representations. Default to 200.
        dropout: float
            Dropout probability. Default to 0.
        mode: str
            The model type, 'classification' or 'regression'. Default to 'regression'.
        number_atom_features: int
            The length of the initial atom feature vectors. Default to 30.
        number_bond_features: int
            The length of the initial bond feature vectors. Default to 11.
        n_classes: int
            The number of classes to predict per task
            (only used when ``mode`` is 'classification'). Default to 2.
        self_loop: bool
            Whether to add self loops for the nodes, i.e. edges from nodes to themselves.
            When input graphs have isolated nodes, self loops allow preserving the original feature
            of them in message passing. Default to True.
        kwargs
            This can include any keyword argument of TorchModel.
        """
        self.return_embeddings = return_embeddings
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
        """Create batch data for AttentiveFP.

        Parameters
        ----------
        batch: tuple
            The tuple is ``(inputs, labels, weights)``.

        Returns
        -------
        inputs: DGLGraph
            DGLGraph for a batch of graphs.
        labels: list of torch.Tensor or None
            The graph labels.
        weights: list of torch.Tensor or None
            The weights for each sample or sample/task pair converted to torch.Tensor.
        """
        try:
            import dgl
        except:
            raise ImportError('This class requires dgl.')

        inputs, labels, weights = batch
# =============================================================================
#         print(type(inputs[0]),inputs[0].shape)
#         import pdb
#         pdb.set_trace()
# =============================================================================
        dgl_graphs = [
            graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
        ]
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

task =["redox_potential","solvation_free_energy"]#["solvation_free_energy"] #
featurizer = MolMergerFeaturizer(use_edges = True)
dataset_file = "./redox_solvation_data.csv"

loader = dc.data.CSVLoader(
    tasks = task,
    smiles_field = "ox_smiles",
    featurizer = featurizer
)

dataset = loader.featurize(dataset_file,shard_size = 8192)

splitter = dc.splits.IndexSplitter()



#%%

import deepchem as dc
from pathlib import Path

# Choose a persistent location (adjust for your HPC)
save_dir = Path("./deepchem_cache/molmerger_dataset")
save_dir.mkdir(parents=True, exist_ok=True)

dataset.save_to_disk()
print(f"Saved dataset to {save_dir}")

import numpy as np

splitter = dc.splits.IndexSplitter()
train, valid, test = splitter.train_valid_test_split(dataset)

np.savez_compressed(
    save_dir / "split_indices.npz",
    train=train.ids, valid=valid.ids, test=test.ids
)


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
            BayesianLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
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


tasks = ["redox_potential", "solvation_free_energy"]

train, test = splitter.train_test_split(dataset, seed=42, frac_train=0.5)

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



################################################################################

# Predictions and UQ calibration
################################################################################


import numpy as np
import torch

def get_predictions_with_uncertainty(model, loader, n_samples=100):

    model.eval()  # Set the model to evaluation mode
    all_means = []
    all_stds = []
    all_labels = []

    device = next(model.parameters()).device

    with torch.no_grad():  # Disable gradient calculations
        for g_batch, y_batch in loader:
            g_batch = g_batch.to(device)
            
            # Store predictions from multiple forward passes
            batch_predictions = []
            for _ in range(n_samples):
                node_feats = g_batch.ndata['x']
                edge_feats = g_batch.edata['edge_attr']
                h_nodes = model.gnn(g_batch, node_feats, edge_feats)
                g_feats = model.readout(g_batch, h_nodes)
                prediction = model.predict(g_feats)
                batch_predictions.append(prediction)

            # Stack predictions along a new dimension (samples, batch_size, n_tasks)
            # The BNN's stochasticity means each prediction will be slightly different
            batch_predictions_tensor = torch.stack(batch_predictions)

            # Calculate mean and standard deviation across the samples dimension
            means = torch.mean(batch_predictions_tensor, dim=0)
            stds = torch.std(batch_predictions_tensor, dim=0)

            all_means.append(means.cpu().numpy())
            all_stds.append(stds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    # Concatenate results from all batches
    return np.concatenate(all_means), np.concatenate(all_stds), np.concatenate(all_labels)    
    

# Create data loaders for evaluation (important: shuffle=False)
train_loader_eval = make_loader(train, batch_size=256, shuffle=False)
test_loader_eval = make_loader(test, batch_size=256, shuffle=False)

print("\n Getting predictions with uncertainty bounds...")

# Get predictions for the test set
test_means, test_stds, test_labels = get_predictions_with_uncertainty(
    pt_predictor, test_loader_eval, n_samples=100
)

# Get predictions for the train set
train_means, train_stds, train_labels = get_predictions_with_uncertainty(
    pt_predictor, train_loader_eval, n_samples=100
)

print("Done. Variables available: test_means, test_stds, train_means, train_stds")


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


print("\nRecalibrated uncertainties added to df_uq:")
print(df_uq1[['uq', 'uq_recalibrated', 'error']].head())

print("\nEvaluating recalibrated uncertainties on df_uq (test set)...")

ordered_df_recal = order_sig_and_errors(df_uq1['uq_recalibrated'], df_uq1['error'])

fig_recal, slope_recal, R_sq_recal, intercept_recal = get_slope_metric(
    ordered_df_recal['uq'], ordered_df_recal['errors'], Nbins=20, include_bootstrap=True
)
print(f"\nRecalibrated Test Set Fit:")
print(f"Slope: {slope_recal:.4f}")
print(f"Intercept: {intercept_recal:.4f}")
print(f"R_squared: {R_sq_recal:.4f}")
# fig_recal.show() # Or save the figure

_NLL_recal = NLL(df_uq1['uq_recalibrated'], df_uq1['error'])
print(f'\nRecalibrated NLL = {_NLL_recal:.4f}')

gaus_pred_recal, errors_observed_recal = calibration_curve(ordered_df_recal['abs_z'])
mis_cal_recal = calibration_area(errors_observed_recal, gaus_pred_recal)
print(f'Recalibrated miscalibration area = {mis_cal_recal:.4f}')
fig_z_recal, ax_z_recal = plot_Z_scores(ordered_df_recal['errors'], ordered_df_recal['uq'])
fig_cal_recal = plot_calibration_curve(gaus_pred_recal, errors_observed_recal, mis_cal_recal)




