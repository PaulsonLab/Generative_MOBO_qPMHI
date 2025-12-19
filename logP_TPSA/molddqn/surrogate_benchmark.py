#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single train/test benchmark on (redox, solubility) from SMILES.

Train size = 20_000 (or all if fewer); test = remainder, random_state=42.

Models:
  • GNN_Bayes      – PyG backbone + BayesianLinear head
  • BNN_Mordred    – MLP on Mordred descriptors (BayesianLinear layers)
  • BNN_ChemBERT   – MLP on frozen ChemBERTa embeddings (BayesianLinear layers)
"""

import os, math, random, pathlib, warnings, argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchbnn as bnn

from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import GCNConv, global_add_pool

from rdkit import Chem
from mordred import Calculator, descriptors

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


# ───────────────────────────── Repro ─────────────────────────────
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ───────────────────────────── Data ──────────────────────────────
def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = ["ox_smiles", "redox_potential", "logS"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing columns: {miss}")
    return df.dropna(subset=needed).reset_index(drop=True)


# ──────────────────────── Featurizers ────────────────────────────
ATOM_LIST = ["H","C","N","O","F","P","S","Cl","Br","I","B","Si"]
def atom_features(atom: Chem.rdchem.Atom):
    return torch.tensor(
        [atom.GetSymbol()==sym for sym in ATOM_LIST] +
        [atom.GetDegree(),
         atom.GetFormalCharge(),
         atom.GetIsAromatic()], dtype=torch.float)

def bond_features(bond: Chem.rdchem.Bond):
    bt = bond.GetBondType()
    return torch.tensor([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
    ], dtype=torch.float)

def smiles_to_pyg(smiles: str) -> GeoData:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return GeoData(x=torch.zeros((1,len(ATOM_LIST)+3)),
                       edge_index=torch.tensor([[0],[0]],dtype=torch.long),
                       edge_attr=torch.zeros((1,5)))
    Chem.Kekulize(mol, clearAromaticFlags=True)
    x = torch.stack([atom_features(a) for a in mol.GetAtoms()], dim=0)
    edge_index, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_index += [[i,j],[j,i]]
        edge_attr  += [bf, bf]
    if not edge_index:
        edge_index = [[0,0]]
        edge_attr  = [torch.zeros(5)]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.stack(edge_attr, dim=0)
    return GeoData(x=x, edge_index=edge_index, edge_attr=edge_attr)

def calc_mordred(smiles_list: List[str], cache_path: str) -> np.ndarray:
    cache = pathlib.Path(cache_path)
    if cache.exists(): return np.load(cache)
    calc = Calculator(descriptors, ignore_3D=True)
    arr = []
    for s in tqdm(smiles_list, desc="Mordred"):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            arr.append([np.nan]); continue
        try:
            vals = calc(mol)
            arr.append([float(v) if v is not None else np.nan for v in vals])
        except Exception:
            arr.append([np.nan])
    X = np.array(arr, dtype=np.float32)
    mask = ~np.all(np.isnan(X), axis=0)
    X = X[:, mask]
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    np.save(cache, X)
    return X

def chemberta_embed(smiles_list: List[str], cache_path: str,
                    model_name="seyonec/ChemBERTa-zinc-base-v1",
                    batch_size=64, device="cuda") -> np.ndarray:
    cache = pathlib.Path(cache_path)
    if cache.exists(): return np.load(cache)
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    embs = []
    with torch.no_grad():
        for i in trange(0, len(smiles_list), batch_size, desc="ChemBERTa"):
            batch = smiles_list[i:i+batch_size]
            toks = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            toks = {k:v.to(device) for k,v in toks.items()}
            out = mdl(**toks).last_hidden_state[:,0,:]
            embs.append(out.cpu())
    X = torch.cat(embs,0).numpy().astype(np.float32)
    np.save(cache, X)
    return X


# ────────────────────────── Datasets ─────────────────────────────
class TabDataset(Dataset):
    def __init__(self, X, y): self.X=torch.from_numpy(X); self.y=torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class GraphDataset(Dataset):
    def __init__(self, graphs, y):
        self.graphs = graphs
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.graphs)
    def __getitem__(self, i):
        g = self.graphs[i]
        # ensure shape (1,2) so PyG concatenates along dim0 → (B,2)
        g.y = self.y[i].view(1, -1)
        return g


# ────────────────────────── Models ───────────────────────────────
def bayes_linear(in_f, out_f, prior_mu=0., prior_sigma=0.1):
    return bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                           in_features=in_f, out_features=out_f)

def make_bnn_mlp(in_dim, out_dim=2, hidden=(512,256,128), act=nn.ReLU()):
    layers=[]
    dims=(in_dim,)+hidden
    for i in range(len(dims)-1):
        layers += [bayes_linear(dims[i], dims[i+1]), act]
    layers.append(bayes_linear(dims[-1], out_dim))
    return nn.Sequential(*layers)

class BNN_Regressor(nn.Module):
    def __init__(self, in_dim, out_dim=2, hidden=(512,256,128)):
        super().__init__()
        self.net = make_bnn_mlp(in_dim, out_dim, hidden)
    def forward(self,x): return self.net(x)

class SmallGNN(nn.Module):
    def __init__(self, in_atom=len(ATOM_LIST)+3, hidden=128):
        super().__init__()
        self.c1 = GCNConv(in_atom, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.c3 = GCNConv(hidden, hidden)
        self.act= nn.ReLU()
    def forward(self, x, edge_index, batch):
        h=self.act(self.c1(x,edge_index))
        h=self.act(self.c2(h,edge_index))
        h=self.act(self.c3(h,edge_index))
        return global_add_pool(h,batch)

class GNN_Bayes(nn.Module):
    def __init__(self, out_dim=2, hidden_graph=128, hidden_head=128):
        super().__init__()
        self.gnn = SmallGNN(hidden=hidden_graph)
        self.head = nn.Sequential(
            bayes_linear(hidden_graph, hidden_head),
            nn.ReLU(),
            bayes_linear(hidden_head, out_dim)
        )
    def forward(self, data):
        hg = self.gnn(data.x, data.edge_index, data.batch)
        return self.head(hg)


# ───────────────────── Train / Eval utils ────────────────────────
def regression_metrics(Y, P):
    out={}
    for i,name in enumerate(["redox","solv"]):
        yt, yp = Y[:,i], P[:,i]
        out[f"MAE_{name}"]  = mean_absolute_error(yt, yp)
        out[f"RMSE_{name}"] = math.sqrt(mean_squared_error(yt, yp))
        out[f"R2_{name}"]   = r2_score(yt, yp)
    out["MAE_mean"]  = (out["MAE_redox"] + out["MAE_solv"])/2
    out["RMSE_mean"] = (out["RMSE_redox"]+ out["RMSE_solv"])/2
    out["R2_mean"]   = (out["R2_redox"]  + out["R2_solv"])/2
    return out

def train_epoch_tab(loader, model, opt, kl_fn, device, kl_w):
    model.train(); total=0.
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        mse = F.mse_loss(pred, y)
        kl  = kl_fn(model)
        loss = mse + kl_w*kl
        loss.backward(); opt.step()
        total += loss.item()*len(x)
    return total/len(loader.dataset)

def eval_tab(loader, model, device, mc=16):
    model.eval()
    Ys, Ps = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            preds=[model(x) for _ in range(mc)]
            P = torch.stack(preds,0).mean(0)
            Ys.append(y); Ps.append(P.cpu())
    Y = torch.cat(Ys,0).numpy(); P = torch.cat(Ps,0).numpy()
    return regression_metrics(Y,P)

def train_epoch_gnn(loader, model, opt, kl_fn, device, kl_w):
    model.train(); total=0.
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        pred = model(batch)                          # (B,2)
        target = batch.y.view(-1, 2).to(device)      # ensure (B,2)
        mse  = F.mse_loss(pred, target)
        kl   = kl_fn(model)
        loss = mse + kl_w*kl
        loss.backward(); opt.step()
        total += loss.item()*batch.num_graphs
    return total/len(loader.dataset)

def eval_gnn(loader, model, device, mc=16):
    model.eval()
    Ys, Ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds=[model(batch) for _ in range(mc)]
            P = torch.stack(preds,0).mean(0)                 # (B,2)
            Ys.append(batch.y.view(-1,2).cpu())              # (B,2)
            Ps.append(P.cpu())
    Y = torch.cat(Ys,0).numpy(); P = torch.cat(Ps,0).numpy()
    return regression_metrics(Y,P)


# ───────────────────────── Config / Main ─────────────────────────
@dataclass
class Config:
    csv: str
    outdir: str = "./bench_results_single"
    cache_dir: str = "./cache_feats"
    which: Tuple[str,...] = ("gnn","bnn_mordred","bnn_chembert")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    n_train: int = 20000
    batch: int = 128
    epochs: int = 200
    eval_every: int = 10
    lr: float = 3e-4
    kl_warmup: int = 50
    mc_samples: int = 16
    hidden: Tuple[int,...] = (512,256,128)
    gnn_hidden: int = 128
    chemberta_model: str = "seyonec/ChemBERTa-zinc-base-v1"

def main(cfg: Config):
    seed_everything(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)

    df = load_df(cfg.csv)
    smiles = df.ox_smiles.tolist()
    Y = df[['redox_potential','logS']].values.astype(np.float32)

    N = len(df)
    n_train = min(cfg.n_train, N-1)
    idx = np.random.RandomState(cfg.seed).permutation(N)
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    print(f"Train: {len(tr_idx)}  |  Test: {len(te_idx)}  (N={N})")

    results_summary = []
    kl_fn = bnn.BKLLoss(reduction='mean', last_layer_only=False)

    # ───── GNN ─────
    if "gnn" in cfg.which:
        print("\n=== GNN_Bayes ===")
        cache_graphs = pathlib.Path(cfg.cache_dir)/"graphs.pt"
        if cache_graphs.exists():
            graphs = torch.load(cache_graphs)
        else:
            graphs = [smiles_to_pyg(s) for s in tqdm(smiles, desc="Graphs")]
            torch.save(graphs, cache_graphs)

        tr_graphs = [graphs[i] for i in tr_idx]
        te_graphs = [graphs[i] for i in te_idx]
        tr_y, te_y = Y[tr_idx], Y[te_idx]

        tr_ds = GraphDataset(tr_graphs, tr_y)
        te_ds = GraphDataset(te_graphs, te_y)
        tr_ld = GeoLoader(tr_ds, batch_size=cfg.batch, shuffle=True)
        te_ld = GeoLoader(te_ds, batch_size=cfg.batch, shuffle=False)

        model = GNN_Bayes(out_dim=2, hidden_graph=cfg.gnn_hidden,
                          hidden_head=cfg.hidden[-1]).to(cfg.device)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        history=[]; best=None
        for epoch in range(cfg.epochs):
            print('Epoch::', epoch)
            kl_w = min(1.0, (epoch+1)/cfg.kl_warmup)
            train_epoch_gnn(tr_ld, model, opt, kl_fn, cfg.device, kl_w)
            if (epoch+1)%cfg.eval_every==0 or epoch==cfg.epochs-1:
                m_test = eval_gnn(te_ld, model, cfg.device, cfg.mc_samples)
                m_test.update({"epoch":epoch+1})
                history.append(m_test)
                if best is None or m_test["RMSE_mean"]<best["RMSE_mean"]:
                    best = m_test
        pd.DataFrame(history).to_csv(os.path.join(cfg.outdir,"gnn_bayes.csv"), index=False)
        print(f"Best GNN RMSE_mean={best['RMSE_mean']:.4f}")
        results_summary.append({"model":"gnn_bayes", **best})

    # ───── Mordred BNN ─────
    if "bnn_mordred" in cfg.which:
        print("\n=== BNN_Mordred ===")
        X_mord = calc_mordred(smiles, os.path.join(cfg.cache_dir,"mordred.npy"))
        Xtr, Xte = X_mord[tr_idx], X_mord[te_idx]
        ytr, yte = Y[tr_idx],    Y[te_idx]

        tr_ds = TabDataset(Xtr, ytr)
        te_ds = TabDataset(Xte, yte)
        tr_ld = DataLoader(tr_ds, batch_size=cfg.batch, shuffle=True)
        te_ld = DataLoader(te_ds, batch_size=cfg.batch, shuffle=False)

        model = BNN_Regressor(in_dim=Xtr.shape[1], out_dim=2, hidden=cfg.hidden).to(cfg.device)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        history=[]; best=None
        for epoch in range(cfg.epochs):
            kl_w = min(1.0, (epoch+1)/cfg.kl_warmup)
            train_epoch_tab(tr_ld, model, opt, kl_fn, cfg.device, kl_w)
            if (epoch+1)%cfg.eval_every==0 or epoch==cfg.epochs-1:
                m_test = eval_tab(te_ld, model, cfg.device, cfg.mc_samples)
                m_test.update({"epoch":epoch+1})
                history.append(m_test)
                if best is None or m_test["RMSE_mean"]<best["RMSE_mean"]:
                    best = m_test
        pd.DataFrame(history).to_csv(os.path.join(cfg.outdir,"bnn_mordred.csv"), index=False)
        print(f"Best Mordred BNN RMSE_mean={best['RMSE_mean']:.4f}")
        results_summary.append({"model":"bnn_mordred", **best})

    # ───── ChemBERTa BNN ─────
    if "bnn_chembert" in cfg.which:
        print("\n=== BNN_ChemBERT ===")
        X_chem = chemberta_embed(smiles, os.path.join(cfg.cache_dir,"chemberta.npy"),
                                 cfg.chemberta_model, device=cfg.device)
        Xtr, Xte = X_chem[tr_idx], X_chem[te_idx]
        ytr, yte = Y[tr_idx],     Y[te_idx]

        tr_ds = TabDataset(Xtr, ytr)
        te_ds = TabDataset(Xte, yte)
        tr_ld = DataLoader(tr_ds, batch_size=cfg.batch, shuffle=True)
        te_ld = DataLoader(te_ds, batch_size=cfg.batch, shuffle=False)

        model = BNN_Regressor(in_dim=Xtr.shape[1], out_dim=2, hidden=cfg.hidden).to(cfg.device)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        history=[]; best=None
        for epoch in range(cfg.epochs):
            kl_w = min(1.0, (epoch+1)/cfg.kl_warmup)
            train_epoch_tab(tr_ld, model, opt, kl_fn, cfg.device, kl_w)
            if (epoch+1)%cfg.eval_every==0 or epoch==cfg.epochs-1:
                m_test = eval_tab(te_ld, model, cfg.device, cfg.mc_samples)
                m_test.update({"epoch":epoch+1})
                history.append(m_test)
                if best is None or m_test["RMSE_mean"]<best["RMSE_mean"]:
                    best = m_test
        pd.DataFrame(history).to_csv(os.path.join(cfg.outdir,"bnn_chembert.csv"), index=False)
        print(f"Best ChemBERTa BNN RMSE_mean={best['RMSE_mean']:.4f}")
        results_summary.append({"model":"bnn_chembert", **best})

    if results_summary:
        summary = pd.DataFrame(results_summary)
        summary.to_csv(os.path.join(cfg.outdir,"summary_single_split.csv"), index=False)
        print("\n=== Summary (best epoch on test) ===")
        print(summary[["model","RMSE_mean","MAE_mean","R2_mean",
                       "RMSE_redox","RMSE_solv","MAE_redox","MAE_solv"]])

    print("\nDone.")


# ─────────── CLI vs Spyder friendly entrypoint ───────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--which", nargs="+", default=["gnn","bnn_chembert"])
    p.add_argument("--outdir", default="./bench_results_single")
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--epochs",  type=int, default=200)
    p.add_argument("--batch",   type=int, default=128)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    import sys
    interactive = hasattr(sys, "ps1") or ("spyder_kernels" in sys.modules) or ("ipykernel" in sys.modules)
    if interactive or len(sys.argv)==1:
        cfg = Config(
            csv="/home/muthyala.7/TorchSisso1/new_symantic/sampled_dataset_redox_solubility.csv",
            which=("bnn_chembert"),
            outdir="./bench_results_single",
            n_train=20000,
            epochs=200,
            batch=128,
            lr=3e-4,
            seed=42,
        )
    else:
        args = parse_args()
        cfg = Config(csv=args.csv,
                     which=tuple(args.which),
                     outdir=args.outdir,
                     n_train=args.n_train,
                     epochs=args.epochs,
                     batch=args.batch,
                     lr=args.lr,
                     seed=args.seed)
    main(cfg)



#%%


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Surrogate benchmark with UQ & checkpointing.

SMILES → (redox, solubility)

Models
------
1) GNN_Bayes        : PyG GINE backbone + torchbnn Bayesian head
2) BNN_Mordred      : Mordred descriptors + deeper Bayesian MLP
3) BNN_ChemBERT     : ChemBERTa embeddings + deeper Bayesian MLP (optionally FT last blocks)

Outputs
-------
best_<model>.pt
<model>.csv                       (per-epoch metrics)
test_preds_<model>.csv            (μ, σ, 95% CI for each test point)
summary_single_split.csv          (best metrics + PICP/MPIW)
"""

import os, math, random, pathlib, warnings, argparse, sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchbnn as bnn

from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import GINEConv, global_add_pool

from rdkit import Chem
from mordred import Calculator, descriptors

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


# ───────────────────────────── Repro ─────────────────────────────
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ───────────────────────────── Data ──────────────────────────────
def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = ["ox_smiles", "redox_potential", "logS"]
    miss = [c for c in needed if c not in df.columns]
    if miss: raise ValueError(f"CSV missing columns: {miss}")
    return df.dropna(subset=needed).reset_index(drop=True)


# ──────────────────────── Featurizers ────────────────────────────
ATOM_LIST = ["H","C","N","O","F","P","S","Cl","Br","I","B","Si"]

def atom_features(atom: Chem.rdchem.Atom):
    # one-hot + degree + formal charge + aromatic + hybridization + in ring
    hyb = atom.GetHybridization()
    hyb_list = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    return torch.tensor(
        [atom.GetSymbol()==sym for sym in ATOM_LIST] +
        [atom.GetDegree(),
         atom.GetFormalCharge(),
         atom.GetIsAromatic(),
         atom.IsInRing()] +
        [hyb == h for h in hyb_list],
        dtype=torch.float)

def bond_features(bond: Chem.rdchem.Bond):
    bt = bond.GetBondType()
    return torch.tensor([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ], dtype=torch.float)

def smiles_to_pyg(smiles: str) -> GeoData:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return GeoData(
            x=torch.zeros((1,len(ATOM_LIST)+4+5)),    # fall-back dims (rough)
            edge_index=torch.tensor([[0],[0]], dtype=torch.long),
            edge_attr=torch.zeros((1,6))
        )
    Chem.Kekulize(mol, clearAromaticFlags=True)
    x = torch.stack([atom_features(a) for a in mol.GetAtoms()], dim=0)
    edge_index, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_index += [[i,j],[j,i]]
        edge_attr  += [bf, bf]
    if not edge_index:
        edge_index = [[0,0]]
        edge_attr  = [torch.zeros(6)]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.stack(edge_attr, dim=0)
    return GeoData(x=x, edge_index=edge_index, edge_attr=edge_attr)

def calc_mordred(smiles_list: List[str], cache_path: str) -> np.ndarray:
    cache = pathlib.Path(cache_path)
    if cache.exists(): return np.load(cache)
    calc = Calculator(descriptors, ignore_3D=True)
    arr = []
    for s in tqdm(smiles_list, desc="Mordred"):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            arr.append([np.nan]); continue
        try:
            vals = calc(mol)
            arr.append([float(v) if v is not None else np.nan for v in vals])
        except Exception:
            arr.append([np.nan])
    X = np.array(arr, dtype=np.float32)
    mask = ~np.all(np.isnan(X), axis=0)
    X = X[:, mask]
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    np.save(cache, X)
    return X

def chemberta_embed(smiles_list: List[str], cache_path: str,
                    model_name="seyonec/ChemBERTa-zinc-base-v1",
                    batch_size=64, device="cuda",
                    finetune_layers:int=0):
    """
    Safe load for Torch < 2.6: force safetensors and avoid weights_only=True restriction.
    Optionally unfreeze last `finetune_layers` transformer blocks.
    """
    cache = pathlib.Path(cache_path)
    if cache.exists(): return np.load(cache)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    load_kwargs = dict(use_safetensors=True, weights_only=False, trust_remote_code=False)
    try:
        mdl = AutoModel.from_pretrained(model_name, **load_kwargs)
    except TypeError:
        mdl = AutoModel.from_pretrained(model_name, use_safetensors=True)

    if finetune_layers > 0:
        # unfreeze last `finetune_layers` encoder blocks
        for p in mdl.parameters(): p.requires_grad = False
        try:
            for blk in mdl.encoder.layer[-finetune_layers:]:
                for p in blk.parameters(): p.requires_grad = True
        except AttributeError:
            pass  # model arch mismatch; ignore
    else:
        for p in mdl.parameters(): p.requires_grad = False

    mdl = mdl.to(device).eval()

    embs = []
    with torch.no_grad():
        for i in trange(0, len(smiles_list), batch_size, desc="ChemBERTa"):
            batch = smiles_list[i:i+batch_size]
            toks = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            toks = {k:v.to(device) for k,v in toks.items()}
            out = mdl(**toks).last_hidden_state[:,0,:]  # CLS token
            embs.append(out.cpu())
    X = torch.cat(embs, 0).numpy().astype(np.float32)
    np.save(cache, X)
    return X


# ────────────────────────── Datasets ─────────────────────────────
class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class GraphDataset(Dataset):
    def __init__(self, graphs, y):
        self.graphs = graphs
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.graphs)
    def __getitem__(self, i):
        g = self.graphs[i]
        g.y = self.y[i].view(1, -1)  # (1,2)
        return g


# ───────────────────────── Scaling ───────────────────────────────
class TargetScaler:
    def fit(self, y: np.ndarray):
        self.mu = y.mean(0, keepdims=True)
        self.sd = y.std(0, keepdims=True) + 1e-8
        return self
    def transform(self, y):   return (y - self.mu)/self.sd
    def inverse(self, yhat):  return yhat*self.sd + self.mu


# ────────────────────────── Models ───────────────────────────────
def bayes_linear(in_f, out_f, prior_mu=0., prior_sigma=0.1):
    return bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                           in_features=in_f, out_features=out_f)

class BayesBlock(nn.Module):
    def __init__(self, in_f, out_f, p_drop=0.1):
        super().__init__()
        self.fc = bayes_linear(in_f, out_f)
        self.ln = nn.LayerNorm(out_f)
        self.act= nn.SiLU()
        self.drop = nn.Dropout(p_drop)
        self.res = (in_f == out_f)
    def forward(self, x):
        h = self.fc(x)
        h = self.ln(h)
        h = self.act(h)
        h = self.drop(h)
        if self.res:
            h = h + x
        return h

class BNN_Regressor(nn.Module):
    """Deeper Bayesian MLP with residual BayesBlocks."""
    def __init__(self, in_dim, out_dim=2, widths=(1024,1024,512,512,256,256), p_drop=0.1):
        super().__init__()
        layers=[]
        last = in_dim
        for w in widths:
            layers.append(BayesBlock(last, w, p_drop))
            last = w
        layers.append(bayes_linear(last, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class GINEBackbone(nn.Module):
    def __init__(self, in_atom, in_bond, hidden=256, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.edge_dim = in_bond
        dims = [in_atom] + [hidden]*n_layers
        for i in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=in_bond))
        self.act = nn.ReLU()
    def forward(self, x, edge_index, edge_attr, batch):
        h = x
        for conv in self.layers:
            h = self.act(conv(h, edge_index, edge_attr))
        return global_add_pool(h, batch)

class GNN_Bayes(nn.Module):
    def __init__(self, out_dim=2, hidden_graph=256, hidden_head=256, n_layers=3, in_atom=None, in_bond=None):
        super().__init__()
        self.gnn = GINEBackbone(in_atom, in_bond, hidden_graph, n_layers)
        self.head = nn.Sequential(
            bayes_linear(hidden_graph, hidden_head),
            nn.SiLU(),
            nn.Dropout(0.1),
            bayes_linear(hidden_head, out_dim)
        )
    def forward(self, data):
        hg = self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)
        return self.head(hg)


# ───────────────────── Train / Eval utils ────────────────────────
def regression_metrics(Y, P):
    out={}
    for i,name in enumerate(["redox","solv"]):
        yt, yp = Y[:,i], P[:,i]
        out[f"MAE_{name}"]  = mean_absolute_error(yt, yp)
        out[f"RMSE_{name}"] = math.sqrt(mean_squared_error(yt, yp))
        out[f"R2_{name}"]   = r2_score(yt, yp)
    out["MAE_mean"]  = (out["MAE_redox"] + out["MAE_solv"])/2
    out["RMSE_mean"] = (out["RMSE_redox"]+ out["RMSE_solv"])/2
    out["R2_mean"]   = (out["R2_redox"]  + out["R2_solv"])/2
    return out

def train_epoch_tab(loader, model, opt, kl_fn, device, kl_beta, n_train):
    model.train(); total=0.; mse_total=0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        opt.zero_grad()
        pred = model(X)
        mse = F.mse_loss(pred, y)
        kl  = kl_fn(model) / n_train
        loss = mse + kl_beta*kl
        loss.backward(); opt.step()
        total += loss.item()*len(X)
        mse_total += mse.item()*len(X)
    return total/len(loader.dataset), mse_total/len(loader.dataset)

def train_epoch_gnn(loader, model, opt, kl_fn, device, kl_beta, n_train):
    model.train(); total=0.; mse_total=0.
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        pred = model(batch)                          # (B,2)
        target = batch.y.view(-1, 2).to(device)
        mse  = F.mse_loss(pred, target)
        kl   = kl_fn(model) / n_train
        loss = mse + kl_beta*kl
        loss.backward(); opt.step()
        total += loss.item()*batch.num_graphs
        mse_total += mse.item()*batch.num_graphs
    return total/len(loader.dataset), mse_total/len(loader.dataset)

def eval_tab(loader, model, device, scaler, mc=16):
    model.eval()
    Ys, Ps = [], []
    with torch.no_grad():
        for X,y in loader:
            X = X.to(device)
            preds=[model(X) for _ in range(mc)]
            P = torch.stack(preds,0).mean(0).cpu()
            Ys.append(y)
            Ps.append(P)
    Yz = torch.cat(Ys,0).numpy()
    Pz = torch.cat(Ps,0).numpy()
    Y  = scaler.inverse(Yz)
    P  = scaler.inverse(Pz)
    return regression_metrics(Y,P)

def eval_gnn(loader, model, device, scaler, mc=16):
    model.eval()
    Ys, Ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds=[model(batch) for _ in range(mc)]
            P = torch.stack(preds,0).mean(0).cpu()
            Ys.append(batch.y.view(-1,2).cpu())
            Ps.append(P)
    Yz = torch.cat(Ys,0).numpy()
    Pz = torch.cat(Ps,0).numpy()
    Y  = scaler.inverse(Yz)
    P  = scaler.inverse(Pz)
    return regression_metrics(Y,P)


# ───────────── UQ helpers ─────────────
def mc_predict_tab(model, loader, device, scaler, mc=64):
    model.eval()
    Ys, means, stds = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            P = torch.stack([model(X) for _ in range(mc)], 0)  # (mc,B,2)
            means.append(P.mean(0).cpu())
            stds.append(P.std(0).cpu())
            Ys.append(y)
    Yz = torch.cat(Ys).numpy()
    MUz= torch.cat(means).numpy()
    STDz=torch.cat(stds).numpy()
    Y   = scaler.inverse(Yz)
    MU  = scaler.inverse(MUz)
    STD = STDz * scaler.sd
    return Y, MU, STD

def mc_predict_gnn(model, loader, device, scaler, mc=64):
    model.eval()
    Ys, means, stds = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            P = torch.stack([model(batch) for _ in range(mc)], 0)
            means.append(P.mean(0).cpu())
            stds.append(P.std(0).cpu())
            Ys.append(batch.y.view(-1,2).cpu())
    Yz = torch.cat(Ys).numpy()
    MUz= torch.cat(means).numpy()
    STDz=torch.cat(stds).numpy()
    Y   = scaler.inverse(Yz)
    MU  = scaler.inverse(MUz)
    STD = STDz * scaler.sd
    return Y, MU, STD

def interval_metrics(y, mu, std, z=1.96):
    low  = mu - z*std
    high = mu + z*std
    inside = (y >= low) & (y <= high)
    picp = inside.mean(axis=0)          # per-target
    mpiw = (high - low).mean(axis=0)
    return picp, mpiw, low, high


# ───────────────────────── Config / Main ─────────────────────────
@dataclass
class Config:
    csv: str
    outdir: str = "./bench_results_single"
    cache_dir: str = "./cache_feats"
    which: Tuple[str,...] = ("gnn","bnn_mordred","bnn_chembert")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    n_train: int = 20000
    batch: int = 128
    epochs: int = 200
    eval_every: int = 10
    lr: float = 3e-4
    kl_warmup: int = 50
    kl_scale: float = 1.0
    mc_samples: int = 8
    mc_samples_test: int = 64
    # model hyperparams
    hidden_bnn: Tuple[int,...] = (1024,1024,512,512,256,256)
    gnn_hidden: int = 256
    gnn_layers: int = 3
    chemberta_model: str = "seyonec/ChemBERTa-zinc-base-v1"
    chemberta_finetune_layers: int = 0
    save_models: bool = True


def main(cfg: Config):
    seed_everything(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)

    # ── load & split
    df = load_df(cfg.csv)
    smiles = df.ox_smiles.tolist()
    Y = df[['redox_potential','logS']].values.astype(np.float32)

    scaler = TargetScaler().fit(Y)
    Yz = scaler.transform(Y)

    N = len(df)
    n_train = min(cfg.n_train, N-1)
    idx = np.random.RandomState(cfg.seed).permutation(N)
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    print(f"Train: {len(tr_idx)}  |  Test: {len(te_idx)}  (N={N})")

    results_summary = []
    kl_fn = bnn.BKLLoss(reduction='sum', last_layer_only=True)

    # ───── GNN ─────
    if "gnn" in cfg.which:
        print("\n=== GNN_Bayes ===")
        cache_graphs = pathlib.Path(cfg.cache_dir)/"graphs.pt"
        if cache_graphs.exists():
            graphs = torch.load(cache_graphs)
        else:
            graphs = [smiles_to_pyg(s) for s in tqdm(smiles, desc="Graphs")]
            torch.save(graphs, cache_graphs)

        tr_graphs = [graphs[i] for i in tr_idx]
        te_graphs = [graphs[i] for i in te_idx]
        tr_yz, te_yz = Yz[tr_idx], Yz[te_idx]

        tr_ds = GraphDataset(tr_graphs, tr_yz)
        te_ds = GraphDataset(te_graphs, te_yz)
        tr_ld = GeoLoader(tr_ds, batch_size=cfg.batch, shuffle=True)
        te_ld = GeoLoader(te_ds, batch_size=cfg.batch, shuffle=False)

        in_atom = tr_graphs[0].x.shape[1]
        in_bond = tr_graphs[0].edge_attr.shape[1]

        model = GNN_Bayes(out_dim=2,
                          hidden_graph=cfg.gnn_hidden,
                          hidden_head=cfg.gnn_hidden,
                          n_layers=cfg.gnn_layers,
                          in_atom=in_atom, in_bond=in_bond).to(cfg.device)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        history=[]; best=None
        best_path = os.path.join(cfg.outdir,"best_gnn_bayes.pt")
        n_train_tot = len(tr_ds)
        for epoch in range(cfg.epochs):
            beta = min(1.0, (epoch+1)/cfg.kl_warmup) * cfg.kl_scale
            _, tr_mse = train_epoch_gnn(tr_ld, model, opt, kl_fn, cfg.device, beta, n_train_tot)
            if (epoch+1)%cfg.eval_every==0 or epoch==cfg.epochs-1:
                m_test = eval_gnn(te_ld, model, cfg.device, scaler, cfg.mc_samples)
                m_test.update({"epoch":epoch+1, "train_mse":tr_mse})
                history.append(m_test)
                if best is None or m_test["RMSE_mean"] < best["RMSE_mean"]:
                    best = m_test
                    if cfg.save_models:
                        torch.save(model.state_dict(), best_path)
        pd.DataFrame(history).to_csv(os.path.join(cfg.outdir,"gnn_bayes.csv"), index=False)
        print(f"Best GNN RMSE_mean={best['RMSE_mean']:.4f}")

        if cfg.save_models:
            model.load_state_dict(torch.load(best_path, map_location=cfg.device))

        Y_true, MU, STD = mc_predict_gnn(model, te_ld, cfg.device, scaler, mc=cfg.mc_samples_test)
        picp, mpiw, low, high = interval_metrics(Y_true, MU, STD, z=1.96)

        uq_df = pd.DataFrame({
            "idx": te_idx,
            "smiles": [smiles[i] for i in te_idx],
            "y_redox": Y_true[:,0], "mu_redox": MU[:,0], "std_redox": STD[:,0],
            "low95_redox": low[:,0], "high95_redox": high[:,0],
            "y_solv": Y_true[:,1], "mu_solv": MU[:,1], "std_solv": STD[:,1],
            "low95_solv": low[:,1], "high95_solv": high[:,1],
        })
        uq_df.to_csv(os.path.join(cfg.outdir,"test_preds_gnn_bayes.csv"), index=False)

        print(f"[GNN] 95% PICP redox={picp[0]:.3f}, solv={picp[1]:.3f} | "
              f"MPIW redox={mpiw[0]:.3f}, solv={mpiw[1]:.3f}")

        results_summary.append({"model":"gnn_bayes", **best,
                                "PICP_redox":picp[0], "PICP_solv":picp[1],
                                "MPIW_redox":mpiw[0], "MPIW_solv":mpiw[1]})

    # ───── Mordred BNN ─────
    if "bnn_mordred" in cfg.which:
        print("\n=== BNN_Mordred ===")
        X_mord = calc_mordred(smiles, os.path.join(cfg.cache_dir,"mordred.npy"))
        Xtr, Xte = X_mord[tr_idx], X_mord[te_idx]
        ytr_z, yte_z = Yz[tr_idx], Yz[te_idx]

        tr_ds = TabDataset(Xtr, ytr_z)
        te_ds = TabDataset(Xte, yte_z)
        tr_ld = DataLoader(tr_ds, batch_size=cfg.batch, shuffle=True)
        te_ld = DataLoader(te_ds, batch_size=cfg.batch, shuffle=False)

        model = BNN_Regressor(in_dim=Xtr.shape[1], out_dim=2,
                              widths=cfg.hidden_bnn, p_drop=0.1).to(cfg.device)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        history=[]; best=None
        best_path = os.path.join(cfg.outdir,"best_bnn_mordred.pt")
        n_train_tot = len(tr_ds)
        for epoch in range(cfg.epochs):
            beta = min(1.0, (epoch+1)/cfg.kl_warmup) * cfg.kl_scale
            _, tr_mse = train_epoch_tab(tr_ld, model, opt, kl_fn, cfg.device, beta, n_train_tot)
            if (epoch+1)%cfg.eval_every==0 or epoch==cfg.epochs-1:
                m_test = eval_tab(te_ld, model, cfg.device, scaler, cfg.mc_samples)
                m_test.update({"epoch":epoch+1, "train_mse":tr_mse})
                history.append(m_test)
                if best is None or m_test["RMSE_mean"] < best["RMSE_mean"]:
                    best = m_test
                    if cfg.save_models:
                        torch.save(model.state_dict(), best_path)
        pd.DataFrame(history).to_csv(os.path.join(cfg.outdir,"bnn_mordred.csv"), index=False)
        print(f"Best Mordred BNN RMSE_mean={best['RMSE_mean']:.4f}")

        if cfg.save_models:
            model.load_state_dict(torch.load(best_path, map_location=cfg.device))

        Y_true, MU, STD = mc_predict_tab(model, te_ld, cfg.device, scaler, mc=cfg.mc_samples_test)
        picp, mpiw, low, high = interval_metrics(Y_true, MU, STD, z=1.96)

        uq_df = pd.DataFrame({
            "idx": te_idx,
            "smiles": [smiles[i] for i in te_idx],
            "y_redox": Y_true[:,0], "mu_redox": MU[:,0], "std_redox": STD[:,0],
            "low95_redox": low[:,0], "high95_redox": high[:,0],
            "y_solv": Y_true[:,1], "mu_solv": MU[:,1], "std_solv": STD[:,1],
            "low95_solv": low[:,1], "high95_solv": high[:,1],
        })
        uq_df.to_csv(os.path.join(cfg.outdir,"test_preds_bnn_mordred.csv"), index=False)

        print(f"[BNN_Mordred] 95% PICP redox={picp[0]:.3f}, solv={picp[1]:.3f} | "
              f"MPIW redox={mpiw[0]:.3f}, solv={mpiw[1]:.3f}")

        results_summary.append({"model":"bnn_mordred", **best,
                                "PICP_redox":picp[0], "PICP_solv":picp[1],
                                "MPIW_redox":mpiw[0], "MPIW_solv":mpiw[1]})

    # ───── ChemBERTa BNN ─────
    if "bnn_chembert" in cfg.which:
        print("\n=== BNN_ChemBERT ===")
        X_chem = chemberta_embed(smiles,
                                 os.path.join(cfg.cache_dir,"chemberta.npy"),
                                 cfg.chemberta_model,
                                 device=cfg.device,
                                 finetune_layers=cfg.chemberta_finetune_layers)
        Xtr, Xte = X_chem[tr_idx], X_chem[te_idx]
        ytr_z, yte_z = Yz[tr_idx], Yz[te_idx]

        tr_ds = TabDataset(Xtr, ytr_z)
        te_ds = TabDataset(Xte, yte_z)
        tr_ld = DataLoader(tr_ds, batch_size=cfg.batch, shuffle=True)
        te_ld = DataLoader(te_ds, batch_size=cfg.batch, shuffle=False)

        model = BNN_Regressor(in_dim=Xtr.shape[1], out_dim=2,
                              widths=cfg.hidden_bnn, p_drop=0.1).to(cfg.device)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        history=[]; best=None
        best_path = os.path.join(cfg.outdir,"best_bnn_chembert.pt")
        n_train_tot = len(tr_ds)
        for epoch in range(cfg.epochs):
            beta = min(1.0, (epoch+1)/cfg.kl_warmup) * cfg.kl_scale
            _, tr_mse = train_epoch_tab(tr_ld, model, opt, kl_fn, cfg.device, beta, n_train_tot)
            if (epoch+1)%cfg.eval_every==0 or epoch==cfg.epochs-1:
                m_test = eval_tab(te_ld, model, cfg.device, scaler, cfg.mc_samples)
                m_test.update({"epoch":epoch+1, "train_mse":tr_mse})
                history.append(m_test)
                if best is None or m_test["RMSE_mean"] < best["RMSE_mean"]:
                    best = m_test
                    if cfg.save_models:
                        torch.save(model.state_dict(), best_path)
        pd.DataFrame(history).to_csv(os.path.join(cfg.outdir,"bnn_chembert.csv"), index=False)
        print(f"Best ChemBERTa BNN RMSE_mean={best['RMSE_mean']:.4f}")

        if cfg.save_models:
            model.load_state_dict(torch.load(best_path, map_location=cfg.device))

        Y_true, MU, STD = mc_predict_tab(model, te_ld, cfg.device, scaler, mc=cfg.mc_samples_test)
        picp, mpiw, low, high = interval_metrics(Y_true, MU, STD, z=1.96)

        uq_df = pd.DataFrame({
            "idx": te_idx,
            "smiles": [smiles[i] for i in te_idx],
            "y_redox": Y_true[:,0], "mu_redox": MU[:,0], "std_redox": STD[:,0],
            "low95_redox": low[:,0], "high95_redox": high[:,0],
            "y_solv": Y_true[:,1], "mu_solv": MU[:,1], "std_solv": STD[:,1],
            "low95_solv": low[:,1], "high95_solv": high[:,1],
        })
        uq_df.to_csv(os.path.join(cfg.outdir,"test_preds_bnn_chembert.csv"), index=False)

        print(f"[BNN_ChemBERT] 95% PICP redox={picp[0]:.3f}, solv={picp[1]:.3f} | "
              f"MPIW redox={mpiw[0]:.3f}, solv={mpiw[1]:.3f}")

        results_summary.append({"model":"bnn_chembert", **best,
                                "PICP_redox":picp[0], "PICP_solv":picp[1],
                                "MPIW_redox":mpiw[0], "MPIW_solv":mpiw[1]})

    if results_summary:
        summary = pd.DataFrame(results_summary)
        summary.to_csv(os.path.join(cfg.outdir,"summary_single_split.csv"), index=False)
        print("\n=== Summary (best epoch on test) ===")
        print(summary[["model","RMSE_mean","MAE_mean","R2_mean",
                       "RMSE_redox","RMSE_solv","MAE_redox","MAE_solv",
                       "PICP_redox","PICP_solv","MPIW_redox","MPIW_solv"]])

    print("\nDone.")


# ─────────── CLI vs Spyder friendly entrypoint ───────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--which", nargs="+", default=["gnn","bnn_mordred","bnn_chembert"])
    p.add_argument("--outdir", default="./bench_results_single")
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--epochs",  type=int, default=200)
    p.add_argument("--batch",   type=int, default=128)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


#if __name__ == "__main__":
interactive = hasattr(sys, "ps1") or ("spyder_kernels" in sys.modules) or ("ipykernel" in sys.modules)
if interactive or len(sys.argv)==1:
    cfg = Config(
        csv="/home/muthyala.7/TorchSisso1/new_symantic/sampled_dataset_redox_solubility.csv",
        which=("bnn_chembert"),
        outdir="./bench_results_single",
        n_train=20000,
        epochs=500,
        batch=128,
        lr=3e-4,
        seed=42,
    )
else:
    args = parse_args()
    cfg = Config(csv=args.csv,
                 which=tuple(args.which),
                 outdir=args.outdir,
                 n_train=args.n_train,
                 epochs=args.epochs,
                 batch=args.batch,
                 lr=args.lr,
                 seed=args.seed)
main(cfg)



#%% Mordred BNN



#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Mordred BNN for redox/solubility prediction with uncertainty quantification.
Loads DataFrame with Mordred descriptors and targets (last 2 columns).
"""

import os, math, random, pathlib, warnings, argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchbnn as bnn

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ───────────────────────────── Repro ─────────────────────────────
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ───────────────────────────── Data Loading ──────────────────────────────
def load_mordred_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load DataFrame with Mordred descriptors and targets.
    
    Args:
        csv_path: Path to CSV with Mordred descriptors and targets
        
    Returns:
        X: Mordred descriptors (n_samples, n_descriptors) - standardized
        Y: Targets (n_samples, 2) - redox_potential, logS
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded DataFrame with shape: {df.shape}")
    
    # Last 2 columns are targets, rest are descriptors
    X = df.iloc[:, :-2].values.astype(np.float32)
    Y = df.iloc[:, -2:].values.astype(np.float32)
    
    print(f"Descriptors shape: {X.shape}")
    print(f"Targets shape: {Y.shape}")
    print(f"Target columns: {df.columns[-2:].tolist()}")
    
    # Handle any NaN values in descriptors
    if np.isnan(X).any():
        print("Found NaN values in descriptors, imputing with column means...")
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        print("NaN imputation completed.")
    
    # Check for NaN in targets
    if np.isnan(Y).any():
        print("Warning: Found NaN values in targets, removing those samples...")
        valid_mask = ~np.isnan(Y).any(axis=1)
        X = X[valid_mask]
        Y = Y[valid_mask]
        print(f"Removed {(~valid_mask).sum()} samples with NaN targets")
        print(f"Final data shape: X={X.shape}, Y={Y.shape}")
    
    # CRITICAL: Check data ranges and normalize
    print(f"\nData statistics before normalization:")
    print(f"X - min: {X.min():.3f}, max: {X.max():.3f}, mean: {X.mean():.3f}, std: {X.std():.3f}")
    print(f"Y - min: {Y.min():.3f}, max: {Y.max():.3f}, mean: {Y.mean():.3f}, std: {Y.std():.3f}")
    
    # Remove constant features (zero variance)
    X_std = np.std(X, axis=0)
    constant_mask = X_std > 1e-8
    if not constant_mask.all():
        print(f"Removing {(~constant_mask).sum()} constant features...")
        X = X[:, constant_mask]
    
    # Remove infinite values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize features (Z-score normalization)
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    
    print(f"\nData statistics after normalization:")
    print(f"X - min: {X.min():.3f}, max: {X.max():.3f}, mean: {X.mean():.3f}, std: {X.std():.3f}")
    print(f"Y - min: {Y.min():.3f}, max: {Y.max():.3f}, mean: {Y.mean():.3f}, std: {Y.std():.3f}")
    
    return X, Y


# ────────────────────────── Dataset ─────────────────────────────
class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ────────────────────────── Model ───────────────────────────────
def bayes_linear(in_f, out_f, prior_mu=0., prior_sigma=0.1):
    """Create Bayesian linear layer."""
    return bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                           in_features=in_f, out_features=out_f)


def make_bnn_mlp(in_dim, out_dim=2, hidden=(512, 256, 128), act=nn.ReLU(), dropout=0.1):
    """Create Bayesian MLP architecture with better initialization."""
    layers = []
    dims = (in_dim,) + hidden
    
    for i in range(len(dims) - 1):
        layers.append(bayes_linear(dims[i], dims[i + 1]))
        layers.append(nn.BatchNorm1d(dims[i + 1]))  # Add batch norm for stability
        layers.append(act)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    
    layers.append(bayes_linear(dims[-1], out_dim))
    
    return nn.Sequential(*layers)


class BNN_Regressor(nn.Module):
    """Bayesian Neural Network for regression with uncertainty quantification."""
    
    def __init__(self, in_dim, out_dim=2, hidden=(512, 256, 128)):
        super().__init__()
        self.net = make_bnn_mlp(in_dim, out_dim, hidden, dropout=0.1)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Better weight initialization for stability."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    
    def forward(self, x):
        return self.net(x)


# ───────────────────── Training & Evaluation ────────────────────────
def regression_metrics(Y, P):
    """Calculate regression metrics for both targets."""
    out = {}
    target_names = ["redox", "solv"]
    
    for i, name in enumerate(target_names):
        yt, yp = Y[:, i], P[:, i]
        out[f"MAE_{name}"] = mean_absolute_error(yt, yp)
        out[f"RMSE_{name}"] = math.sqrt(mean_squared_error(yt, yp))
        out[f"R2_{name}"] = r2_score(yt, yp)
    
    # Mean metrics
    out["MAE_mean"] = (out["MAE_redox"] + out["MAE_solv"]) / 2
    out["RMSE_mean"] = (out["RMSE_redox"] + out["RMSE_solv"]) / 2
    out["R2_mean"] = (out["R2_redox"] + out["R2_solv"]) / 2
    
    return out


def train_epoch(loader, model, optimizer, kl_fn, device, kl_weight):
    """Train one epoch with gradient clipping."""
    model.train()
    total_loss = 0.
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        pred = model(x)
        mse_loss = F.mse_loss(pred, y)
        kl_loss = kl_fn(model)
        loss = mse_loss + kl_weight * kl_loss
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item() * len(x)
    
    return total_loss / len(loader.dataset)


def evaluate_with_uncertainty(loader, model, device, mc_samples=16):
    """Evaluate model with uncertainty quantification."""
    model.eval()
    Y_true, Y_pred, Y_std = [], [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            
            # Monte Carlo sampling for uncertainty
            predictions = []
            for _ in range(mc_samples):
                pred = model(x)
                predictions.append(pred.cpu())
            
            # Stack predictions and compute mean/std
            preds_stack = torch.stack(predictions, dim=0)  # (mc_samples, batch, 2)
            pred_mean = preds_stack.mean(dim=0)  # (batch, 2)
            pred_std = preds_stack.std(dim=0)    # (batch, 2)
            
            Y_true.append(y)
            Y_pred.append(pred_mean)
            Y_std.append(pred_std)
    
    Y_true = torch.cat(Y_true, 0).numpy()
    Y_pred = torch.cat(Y_pred, 0).numpy()
    Y_std = torch.cat(Y_std, 0).numpy()
    
    # Calculate metrics
    metrics = regression_metrics(Y_true, Y_pred)
    
    return {
        'metrics': metrics,
        'y_true': Y_true,
        'y_pred': Y_pred,
        'y_std': Y_std
    }


def plot_predictions_with_uncertainty(results, target_names, save_path=None):
    """Plot predictions vs true values with uncertainty bands."""
    y_true = results['y_true']
    y_pred = results['y_pred']
    y_std = results['y_std']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        # Scatter plot with error bars
        ax.errorbar(y_true[:, i], y_pred[:, i], yerr=y_std[:, i], 
                   fmt='o', alpha=0.6, markersize=3, capsize=2)
        
        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name} Predictions with Uncertainty')
        ax.grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = results['metrics'][f'R2_{"redox" if i == 0 else "solv"}']
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_uncertainty_distribution(results, target_names, save_path=None):
    """Plot distribution of prediction uncertainties."""
    y_std = results['y_std']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        ax.hist(y_std[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'Prediction Uncertainty (std)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} Uncertainty Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add mean uncertainty to plot
        mean_std = y_std[:, i].mean()
        ax.axvline(mean_std, color='red', linestyle='--', 
                  label=f'Mean: {mean_std:.3f}')
        ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(history_df, save_path=None):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['RMSE_mean', 'MAE_mean', 'R2_mean', 'train_loss']
    titles = ['RMSE (Mean)', 'MAE (Mean)', 'R² (Mean)', 'Training Loss']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        if metric in history_df.columns:
            axes[i].plot(history_df['epoch'], history_df[metric], 'b-', linewidth=2)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(title)
            axes[i].set_title(f'{title} vs Epoch')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ───────────────────────── Configuration ─────────────────────────
@dataclass
class Config:
    csv_path: str                    # CSV with Mordred descriptors and targets
    outdir: str = "./mordred_bnn_results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    n_train: int = 20000
    batch_size: int = 64            # Smaller batch size for stability
    epochs: int = 200
    eval_every: int = 10
    lr: float = 1e-3               # Higher learning rate
    kl_warmup: int = 100           # Longer warmup for KL annealing
    mc_samples: int = 16           # Monte Carlo samples for uncertainty
    hidden: Tuple[int, ...] = (256, 128, 64)  # Smaller network to start


def main(cfg: Config):
    """Main training and evaluation loop."""
    seed_everything(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)
    
    print("=" * 60)
    print("Mordred BNN for Redox/Solubility Prediction")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    X, Y = load_mordred_data(cfg.csv_path)
    
    # Train/test split
    N = len(X)
    n_train = min(cfg.n_train, N - 1)
    idx = np.random.RandomState(cfg.seed).permutation(N)
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    
    print(f"\n2. Data split:")
    print(f"   Train: {len(tr_idx)} samples")
    print(f"   Test:  {len(te_idx)} samples")
    print(f"   Total: {N} samples")
    print(f"   Features: {X.shape[1]} Mordred descriptors")
    
    # Prepare datasets
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = Y[tr_idx], Y[te_idx]
    
    tr_dataset = TabDataset(Xtr, ytr)
    te_dataset = TabDataset(Xte, yte)
    tr_loader = DataLoader(tr_dataset, batch_size=cfg.batch_size, shuffle=True)
    te_loader = DataLoader(te_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    # Initialize model
    print(f"\n3. Initializing BNN model...")
    model = BNN_Regressor(in_dim=Xtr.shape[1], out_dim=2, hidden=cfg.hidden)
    model = model.to(cfg.device)
    
    # Use AdamW with weight decay for better regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    
    # Use learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    kl_fn = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    
    print(f"   Architecture: {Xtr.shape[1]} -> {' -> '.join(map(str, cfg.hidden))} -> 2")
    print(f"   Device: {cfg.device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"\n4. Training for {cfg.epochs} epochs...")
    history = []
    best_model = None
    best_rmse = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg.epochs):
        # KL annealing - slower ramp up
        kl_weight = min(1.0, (epoch + 1) / cfg.kl_warmup)
        
        # Train
        train_loss = train_epoch(tr_loader, model, optimizer, kl_fn, cfg.device, kl_weight)
        
        # Evaluate
        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            results = evaluate_with_uncertainty(te_loader, model, cfg.device, cfg.mc_samples)
            metrics = results['metrics']
            
            # Learning rate scheduling
            scheduler.step(metrics['RMSE_mean'])
            
            # Save best model
            if metrics['RMSE_mean'] < best_rmse:
                best_rmse = metrics['RMSE_mean']
                best_model = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log progress
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"RMSE: {metrics['RMSE_mean']:.4f} | "
                  f"R²: {metrics['R2_mean']:.4f} | LR: {current_lr:.2e}")
            
            # Early stopping
            if patience_counter > 50:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save metrics
            epoch_data = {'epoch': epoch + 1, 'train_loss': train_loss, **metrics}
            history.append(epoch_data)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(cfg.outdir, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"\n5. Training history saved to: {history_path}")
    
    # Final evaluation with best model
    print("\n6. Final evaluation with best model...")
    model.load_state_dict(best_model)
    final_results = evaluate_with_uncertainty(te_loader, model, cfg.device, cfg.mc_samples)
    final_metrics = final_results['metrics']
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"RMSE (mean):     {final_metrics['RMSE_mean']:.4f}")
    print(f"MAE (mean):      {final_metrics['MAE_mean']:.4f}")
    print(f"R² (mean):       {final_metrics['R2_mean']:.4f}")
    print("\nPer-target metrics:")
    print(f"Target 1:  RMSE={final_metrics['RMSE_redox']:.4f}, MAE={final_metrics['MAE_redox']:.4f}, R²={final_metrics['R2_redox']:.4f}")
    print(f"Target 2:  RMSE={final_metrics['RMSE_solv']:.4f}, MAE={final_metrics['MAE_solv']:.4f}, R²={final_metrics['R2_solv']:.4f}")
    
    # Save final results
    results_path = os.path.join(cfg.outdir, "final_results.csv")
    pd.DataFrame([final_metrics]).to_csv(results_path, index=False)
    
    # Save predictions with uncertainties
    predictions_df = pd.DataFrame({
        'true_target1': final_results['y_true'][:, 0],
        'pred_target1': final_results['y_pred'][:, 0],
        'std_target1': final_results['y_std'][:, 0],
        'true_target2': final_results['y_true'][:, 1],
        'pred_target2': final_results['y_pred'][:, 1],
        'std_target2': final_results['y_std'][:, 1],
    })
    pred_path = os.path.join(cfg.outdir, "predictions_with_uncertainty.csv")
    predictions_df.to_csv(pred_path, index=False)
    
    # Generate plots
    print(f"\n7. Generating plots...")
    target_names = ['Target 1', 'Target 2']  # Generic names since we don't know the exact column names
    
    plot_pred_path = os.path.join(cfg.outdir, "predictions_plot.png")
    plot_unc_path = os.path.join(cfg.outdir, "uncertainty_plot.png")
    plot_train_path = os.path.join(cfg.outdir, "training_curves.png")
    
    plot_predictions_with_uncertainty(final_results, target_names, plot_pred_path)
    plot_uncertainty_distribution(final_results, target_names, plot_unc_path)
    plot_training_curves(history_df, plot_train_path)
    
    # Save model
    model_path = os.path.join(cfg.outdir, "best_model.pt")
    torch.save(best_model, model_path)
    
    print(f"\nAll results saved to: {cfg.outdir}")
    print(f"  - training_history.csv")
    print(f"  - final_results.csv") 
    print(f"  - predictions_with_uncertainty.csv")
    print(f"  - predictions_plot.png")
    print(f"  - uncertainty_plot.png")
    print(f"  - training_curves.png")
    print(f"  - best_model.pt")
    
    print("\nDone! 🎉")


# ─────────── CLI vs Interactive friendly entrypoint ───────────
def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Mordred BNN")
    parser.add_argument("--csv_path", required=True, help="CSV with Mordred descriptors and targets")
    parser.add_argument("--outdir", default="./mordred_bnn_results", help="Output directory")
    parser.add_argument("--n_train", type=int, default=20000, help="Training set size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mc_samples", type=int, default=16, help="MC samples for uncertainty")
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    
    # Check if running interactively
    interactive = (hasattr(sys, "ps1") or 
                  ("spyder_kernels" in sys.modules) or 
                  ("ipykernel" in sys.modules))
    
    if interactive or len(sys.argv) == 1:
        # Interactive/IDE mode - modify this path
        cfg = Config(
            csv_path='./descriptors_dataset_sampled.csv',  # UPDATE THIS PATH
            outdir="./mordred_bnn_results",
            n_train=40000,
            epochs=200,
            batch_size=128,
            lr=3e-4,
            seed=42,
            mc_samples=16
        )
    else:
        # CLI mode
        args = parse_args()
        cfg = Config(
            csv_path=args.csv_path,
            outdir=args.outdir,
            n_train=args.n_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            mc_samples=args.mc_samples
        )
    
    main(cfg)

# =============================================================================
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
# import torch.optim as optim
# import torchbnn as bnn
# from torchbnn.layers import BayesBlock, bayes_linear
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# 
# # ─────────────────────────────────────────────────────────────────────────────
# # 1. Load your data (adjust the path and target column names as needed)
# # ─────────────────────────────────────────────────────────────────────────────
# 
# 
# df = pd.read_csv('/home/muthyala.7/TorchSisso1/new_symantic/sampled_dataset_redox_solubility.csv')
# df1 = pd.read_csv('./descriptors_dataset_sampled.csv')
# 
# # Clean the data
# df1 = df1.loc[:, ~df1.apply(lambda col: col.astype(str).str.contains("nan").any())]
# 
# # Step 2: Keep only numeric columns
# df1 = df1.select_dtypes(include=[np.number])
# 
# df1.drop(df1.columns[[0]], axis=1, inplace=True)
# 
# df1['redox_potential'] = df.redox_potential
# df1['logS'] = df.logS
# =============================================================================


#%%




df_chembert = pd.read_csv('/home/muthyala.7/TGVAE/MolDQN-pytorch/bench_results_single/predictions_with_uncertainty.csv')

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
df_uq['true_logP'] = df_chembert.true_target2
df_uq['pred_logP'] = df_chembert.pred_target2

df_uq['error'] = df_uq.pred_logP - df_uq.true_logP
df_uq['uq'] = df_chembert.std_target2
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


# =============================================================================
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# plt.figure(figsize=(10, 8))
# font = {'weight': 'bold', 'size': 18}
# matplotlib.rc('font', **font)
# matplotlib.rcParams.update({
#     'text.usetex': False,
#     'font.family': 'serif',
#     'font.serif': 'cmr10',
#     'mathtext.fontset': 'cm',
#     'axes.unicode_minus': False,
#     'figure.dpi': 300,
# })
# # assume means and y_true as before, choose one task
# y_t = df_chembert.y_redox
# y_p = df_chembert.mu_redox
# 
# 
# mn =min(y_t.min(), y_p.min())
# mx = max(y_t.max(), y_p.max())
# 
# plt.scatter(y_t,y_p)
# 
# # 1:1 line for reference
# plt.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
# 
# # Set limits and enforce equal spacing
# plt.xlim(mn, mx)
# plt.ylim(mn, mx)
# plt.gca().set_aspect('equal', adjustable='box')
# 
# # Create consistent tick spacing
# num_ticks = 5  # or change depending on how many ticks you want
# ticks = np.linspace(mn, mx, num=num_ticks)
# plt.xticks(ticks)
# plt.yticks(ticks)
# 
# plt.xlabel('Redox')
# plt.ylabel('Predicted Redox')
# plt.tight_layout()
# plt.show()
# =============================================================================
