
from __future__ import annotations

import json, math, random, pathlib, time
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

try:
    from rdkit import Chem  # noqa: F401 – ensure RDKit import succeeds
except ImportError:
    raise SystemExit("RDKit is required: `pip install rdkit-pypi`.")
#%%

# ===================== USER CONFIG =====================
CSV_PATH    = pathlib.Path('/home/muthyala.7/Downloads/250k_rndm_zinc_drugs_clean_3_TPSA.csv')
PARAMS_JSON = None                      # pathlib.Path("my_params.json") or None
OUT_DIR     = pathlib.Path("runs/vae")  # checkpoints & logs will live here
N_SAMPLES   = 10                        # SMILES to sample after training
USE_AMP     = True   
# =======================================================




import json, math, random, pathlib, time, itertools
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

try:
    from rdkit import Chem  # noqa: F401 – only to ensure RDKit is installed
except ImportError:
    raise SystemExit("RDKit is required: `pip install rdkit-pypi`.")



# -----------------------------------------------------------------------------
# 0) Hyper‑parameters helper
# -----------------------------------------------------------------------------

def load_params(path: str | pathlib.Path | None = None) -> Dict[str, Any]:
    default: Dict[str, Any] = {
        "MAX_LEN": 120,
        "val_frac": 0.1,
        "hidden_dim": 196,
        "recurrent_dim": 488,
        "gru_depth": 2,
        "conv_depth": 3,
        "conv_dim_depth": 9,
        "conv_d_growth_factor": 1.4,
        "conv_dim_width": 9,
        "conv_w_growth_factor": 1.0,
        "batchnorm_conv": True,
        "middle_layer": 0,
        "hg_growth_factor": 1.0,
        "activation": "tanh",
        "batchnorm_mid": False,
        "dropout_rate_mid": 0.2,
        "xent_loss_weight": 1.0,
        "kl_loss_weight": 0.1,
        "prop_loss_weight": 1.0,
        "epochs": 200,
        "batch_size": 128,
        "lr": 3e-4,
        "kl_anneal_epochs": 10,
        "seed": 42,
    }
    if path is not None:
        default.update(json.loads(pathlib.Path(path).read_text()))
    return default

# -----------------------------------------------------------------------------
# 1) Dataset helpers
# -----------------------------------------------------------------------------

class ZincDataset(Dataset):
    """Returns `(one_hot_SMILES, torch.tensor([logP, TPSA]))`."""

    def __init__(self, csv_path: pathlib.Path, params: Dict[str, Any]):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        
        self.params = params
        self.chars = sorted({c for s in self.df.smiles.tolist() for c in s})
        self.chars.append(" ")  # padding char
        params["NCHARS"] = len(self.chars)
        self.ch2i = {c: i for i, c in enumerate(self.chars)}
        self.i2ch = {i: c for c, i in self.ch2i.items()}

    def __len__(self):
        return len(self.df)

    def smiles_to_tensor(self, s: str) -> torch.Tensor:
        L = self.params["MAX_LEN"]
        s = (s[:L] + " " * L)[:L]
        idx = [self.ch2i[c] for c in s]
        return F.one_hot(torch.tensor(idx, dtype=torch.long), num_classes=self.params["NCHARS"]).float()

    def tensor_to_smiles(self, t: torch.Tensor) -> str:
        return "".join(self.i2ch[i] for i in t.argmax(-1).tolist()).strip()

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = self.smiles_to_tensor(row.smiles)
        y = torch.tensor([row.logP, row.TPSA], dtype=torch.float32)
        return x, y

# -----------------------------------------------------------------------------
# 2) Model definition
# -----------------------------------------------------------------------------

def _growth(b: float, g: float, i: int) -> int:
    return int(round(b * (g ** i)))

class Encoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        layers = []
        in_ch = p["NCHARS"]
        for j in range(p["conv_depth"]):
            out_ch = _growth(p["conv_dim_depth"], p["conv_d_growth_factor"], j)
            k = _growth(p["conv_dim_width"], p["conv_w_growth_factor"], j)
            layers.extend([nn.Conv1d(in_ch, out_ch, k, padding="same"), nn.Tanh()])
            if p["batchnorm_conv"]:
                layers.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.fc   = nn.Linear(p["MAX_LEN"] * in_ch, p["hidden_dim"])
        self.mu   = nn.Linear(p["hidden_dim"], p["hidden_dim"])
        self.logv = nn.Linear(p["hidden_dim"], p["hidden_dim"])
    def forward(self,x):
        x = self.conv(x.permute(0,2,1))
        x = torch.tanh(self.fc(torch.flatten(x,1)))
        return self.mu(x), self.logv(x)

class Decoder(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.p = p
        self.rep = nn.Linear(p["hidden_dim"], p["recurrent_dim"])
        self.gru = nn.GRU(p["recurrent_dim"], p["recurrent_dim"], p["gru_depth"], batch_first=True)
        self.out = nn.Linear(p["recurrent_dim"], p["NCHARS"])
    def forward(self,z):
        rep = torch.tanh(self.rep(z)).unsqueeze(1).repeat(1,self.p["MAX_LEN"],1)
        h,_ = self.gru(rep)
        return self.out(h)

class PropertyHead(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(p["hidden_dim"],256), nn.ReLU(), nn.Linear(256,2))
    def forward(self,z):
        return self.net(z)

class MoleculeVAE(nn.Module):
    def __init__(self,p):
        super().__init__()
        self.p=p
        self.enc=Encoder(p)
        self.dec=Decoder(p)
        self.prop=PropertyHead(p)
    def reparam(self,mu,logv):
        return mu + torch.randn_like(logv)*torch.exp(0.5*logv)
    def forward(self,x):
        mu,logv = self.enc(x)
        z = self.reparam(mu,logv)
        logits = self.dec(z)
        props  = self.prop(z)
        return logits, mu, logv, props
    def loss(self,x,y,out,epoch):
        logits,mu,logv,props=out
        B,L,V=logits.shape
        recon = F.cross_entropy(logits.view(B*L,V), x.argmax(-1).view(B*L))
        kl = -0.5*torch.mean(1+logv - mu.pow(2) - logv.exp())
        beta = min(1.0, epoch/self.p["kl_anneal_epochs"])
        prop_loss = F.mse_loss(props,y)
        total = self.p["xent_loss_weight"]*recon + beta*self.p["kl_loss_weight"]*kl + self.p["prop_loss_weight"]*prop_loss
        return total, recon, prop_loss

# -----------------------------------------------------------------------------
# 3) Helper functions
# -----------------------------------------------------------------------------

def set_seed(s:int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def encode_dataset(model:MoleculeVAE, loader:DataLoader, device) -> np.ndarray:
    model.eval(); lat=[]
    for xb,_ in loader:
        lat.append(model.enc(xb.to(device))[0].cpu())
    return torch.cat(lat).numpy()


@torch.no_grad()
def sample_smiles(model: MoleculeVAE,
                  ds: ZincDataset,
                  n: int,
                  device,
                  temperature: float = 1.0) -> List[str]:
    model.eval()
    out = []
    for _ in range(n):
        # 1) draw a latent code
        z = torch.randn(1, model.p["hidden_dim"], device=device)
        # 2) get per-position logits
        logits = model.dec(z)[0]                 # [MAX_LEN, NCHARS]
        
        # 3) convert to probabilities with temperature
        probs = torch.softmax(logits / temperature, dim=-1)
        
        # 4) sample each position
        idxs = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [MAX_LEN]
        
        # 5) convert to characters and stop at <eos>
        chars = []
        for idx in idxs.cpu().tolist():
            c = ds.i2ch[idx]
            if c == '<eos>':
                break
            if c != '<pad>':
                chars.append(c)
        smiles = ''.join(chars)
        out.append(smiles)
    return out

# =============================================================================
# def sample_smiles(model:MoleculeVAE, ds:ZincDataset, n:int, device) -> List[str]:
#     model.eval(); out=[]
#     z = torch.randn(n, model.p["hidden_dim"], device=device)
#     logits = model.dec(z)
#     for i in range(n):
#         out.append(ds.tensor_to_smiles(logits[i].cpu()))
#     return out
# =============================================================================

# ---------- plotting ----------

import torch

def plot_history(hist):
    ep       = [h["epoch"]     for h in hist]
    rec_tr   = [(h["recon_tr"].cpu().item()   if torch.is_tensor(h["recon_tr"]) else h["recon_tr"])
                for h in hist]
    rec_val  = [(h["recon_val"].cpu().item()  if torch.is_tensor(h["recon_val"]) else h["recon_val"])
                for h in hist]
    prop_tr  = [(h["prop_tr"].cpu().item()    if torch.is_tensor(h["prop_tr"]) else h["prop_tr"])
                for h in hist]
    prop_val = [(h["prop_val"].cpu().item()   if torch.is_tensor(h["prop_val"]) else h["prop_val"])
                for h in hist]

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
    ax1.plot(ep, rec_tr,  label="train"); ax1.plot(ep, rec_val,  label="val")
    ax1.set_title("Reconstruction loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(ep, prop_tr, label="train"); ax2.plot(ep, prop_val, label="val")
    ax2.set_title("Property MSE");     ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    plt.show()


# =============================================================================
# def plot_history(hist:List[Dict[str,float]]):
#     ep = [h["epoch"] for h in hist]
#     rec_tr = [h["recon_tr"] for h in hist]
#     rec_val= [h["recon_val"] for h in hist]
#     prop_tr= [h["prop_tr"] for h in hist]
#     prop_val=[h["prop_val"] for h in hist]
#     fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
#     ax1.plot(ep,rec_tr,label="train"), ax1.plot(ep,rec_val,label="val")
#     ax1.set_title("Reconstruction loss"), ax1.set_xlabel("epoch"), ax1.legend()
#     ax2.plot(ep,prop_tr,label="train"), ax2.plot(ep,prop_val,label="val")
#     ax2.set_title("Property MSE"), ax2.set_xlabel("epoch"), ax2.legend()
#     fig.tight_layout(); plt.show()
# =============================================================================

def parity_plot(y_true:np.ndarray, y_pred:np.ndarray):
    titles=["logP","TPSA"]
    fig,axs=plt.subplots(1,2,figsize=(8,4))
    for i,(ax,t) in enumerate(zip(axs,titles)):
        ax.scatter(y_true[:,i], y_pred[:,i], s=8, alpha=0.6)
        ax.plot([y_true[:,i].min(),y_true[:,i].max()],[y_true[:,i].min(),y_true[:,i].max()],"--")
        ax.set_xlabel("true"), ax.set_ylabel("pred"), ax.set_title(f"Parity – {t}")
    fig.tight_layout(); plt.show()

# -----------------------------------------------------------------------------
# 4) Training loop
# -----------------------------------------------------------------------------
params=load_params(PARAMS_JSON)
dataset=ZincDataset(CSV_PATH, params)
set_seed(params["seed"])
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx = np.arange(len(dataset))
tr_idx,val_idx = train_test_split(idx, test_size=params["val_frac"], random_state=params["seed"])
collate = lambda b:(torch.stack([x for x,_ in b]), torch.stack([y for _,y in b]))

dl_tr = DataLoader(torch.utils.data.Subset(dataset,tr_idx), batch_size=params["batch_size"], shuffle=True, collate_fn=collate, pin_memory=True)
dl_val= DataLoader(torch.utils.data.Subset(dataset,val_idx), batch_size=params["batch_size"], shuffle=False, collate_fn=collate, pin_memory=True)

model = MoleculeVAE(params).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=params["lr"])
scaler= torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type=="cuda")

history=[]; best=float("inf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for epoch in range(1, params["epochs"]+1):
    # ---- training ----
    model.train(); run_loss=0; rec_sum=0; prop_sum=0
    for xb,yb in tqdm(dl_tr, desc=f"Epoch {epoch}/{params['epochs']}", leave=False):
        xb,yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP and device.type=="cuda"):
            out = model(xb)
            loss,recon,prop = model.loss(xb,yb,out,epoch)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        run_loss += loss.item()*xb.size(0);
        rec_sum += recon*xb.size(0); prop_sum += prop*xb.size(0)
    n_tr=len(dl_tr.dataset)
    train_loss = run_loss/n_tr; recon_tr = rec_sum/n_tr; prop_tr = prop_sum/n_tr

    # ---- validation ----
    model.eval(); v_tot=v_rec=v_prop=0
    preds=[]; trues=[]
    with torch.no_grad():
        for xb,yb in dl_val:
            xb,yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP and device.type=="cuda"):
                out = model(xb)
                loss,recon,prop = model.loss(xb,yb,out,epoch)
            v_tot += loss.item()*xb.size(0); v_rec += recon*xb.size(0); v_prop += prop*xb.size(0)
            preds.append(out[-1].cpu()); trues.append(yb.cpu())
    n_val=len(dl_val.dataset)
    val_loss=v_tot/n_val; recon_val=v_rec/n_val; prop_val=v_prop/n_val
    preds=torch.cat(preds).numpy(); trues=torch.cat(trues).numpy()

    history.append({"epoch":epoch,"train":train_loss,"val":val_loss,
                     "recon_tr":recon_tr,"recon_val":recon_val,
                     "prop_tr":prop_tr,"prop_val":prop_val})
    print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | reco_tr {recon_tr:.3f} reco_val {recon_val:.3f} | prop_tr {prop_tr:.3f} prop_val {prop_val:.3f}")

    if val_loss < best:
        best=val_loss; torch.save(model.state_dict(), OUT_DIR/"best_multi.pt")

# -------- post‑analysis --------
plot_history(history)
parity_plot(trues, preds)

# ---- sampling ----
smi = sample_smiles(model, dataset, N_SAMPLES, device)
print("sampled SMILES:")
for s in smi: print("SMILE string::", s)

# =============================================================================
# # ---- latent export ----
# full_loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=False, collate_fn=collate)
# z = encode_dataset(model, full_loader, device)
# np.save(OUT_DIR/"latent.npy", z)
# 
# # ---- save history ----
# with open(OUT_DIR/"history.json", "w") as fp:
#     json.dump(history, fp, indent=2)
# 
# print("Training complete.  Best val loss:", best)
# =============================================================================







from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize 
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles
import pandas as pd

# ======================= BO CONFIG =======================
# --- Batched Bayesian Optimization Config ---
N_INITIAL_SAMPLES = 2000
N_BO_ITERATIONS = 20
TARGET_BATCH_SIZE = 50
SUB_BATCH_SIZE = 25
DECODING_ATTEMPTS = 25
DECODING_NOISE = 0.05
OUTPUT_CANDIDATES_CSV = pathlib.Path("bo_generated_candidates_guaranteed_EI_2_run.csv")
# =========================================================


def decode_and_validate(
    z: torch.Tensor, model: MoleculeVAE, dataset: ZincDataset, device: torch.device,
    n_attempts: int, noise_level: float
) -> tuple[torch.Tensor, str] | None:
    """Tries to decode a latent vector into a valid SMILES, using a small random search."""
    with torch.no_grad():
        smi = dataset.tensor_to_smiles(model.dec(z.to(device))[0].cpu())
        if MolFromSmiles(smi) is not None: return z, smi
        for _ in range(n_attempts - 1):
            perturbed_z = z + torch.randn_like(z) * noise_level
            smi = dataset.tensor_to_smiles(model.dec(perturbed_z.to(device))[0].cpu())
            if MolFromSmiles(smi) is not None: return perturbed_z, smi
    return None


# --- 1. Set up the Optimization Environment ---
print("\n--- Starting Guaranteed Batched Bayesian Optimization ---")

model.eval()

print("Encoding validation set to get latent vectors...")
all_z_list, all_y_list = [], []
with torch.no_grad():
    for xb, yb in tqdm(dl_val, desc="Encoding for BO"):
        # Note: model is already on device
        mu, _ = model.enc(xb.to(device))
        all_z_list.append(mu.cpu())
        all_y_list.append(yb.cpu())

all_z = torch.cat(all_z_list)
all_y_logp = torch.cat(all_y_list)[:, 0].unsqueeze(-1)

n_samples = min(N_INITIAL_SAMPLES, len(all_z))
perm = torch.randperm(len(all_z))[:n_samples]
# Move initial training data to the correct device
train_x = all_z[perm].to(device, dtype=torch.double)
train_y = all_y_logp[perm].to(device, dtype=torch.double)
print(f"Initializing Gaussian Process with {n_samples} random samples on device: {train_x.device}")

# --- 2. The Iterative Bayesian Optimization Loop ---
all_generated_candidates = []
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]), seed=random.randint(0, 10000))

for i in range(N_BO_ITERATIONS):
    print(f"\n--- Main Batch {i+1}/{N_BO_ITERATIONS} ---")

    train_x_bounds = torch.stack([train_x.min(dim=0).values, train_x.max(dim=0).values])
    train_x_normalized = normalize(train_x, train_x_bounds)

    y_mean, y_std = train_y.mean(), train_y.std()
    train_y_normalized = (train_y - y_mean) / y_std

    gp = SingleTaskGP(train_x_normalized, train_y_normalized)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    acqf = qExpectedImprovement(
        model=gp, best_f=train_y_normalized.max(), sampler=sampler
    )

    valid_candidates_in_batch = []
    new_observations_in_batch = []
    num_generated_total = 0

    d = train_x.shape[-1]
    normalized_bounds = torch.tensor([[0.0] * d, [1.0] * d], device=device, dtype=torch.double)

    with tqdm(total=TARGET_BATCH_SIZE, desc=f"Collecting valid batch") as pbar:
        while len(valid_candidates_in_batch) < TARGET_BATCH_SIZE:
            candidates_normalized, _ = optimize_acqf(
                acq_function=acqf,
                bounds=normalized_bounds,
                q=SUB_BATCH_SIZE, num_restarts=10, raw_samples=256,
                options={"batch_limit": 5, "maxiter": 200},
            )
            num_generated_total += SUB_BATCH_SIZE

            candidates_unnormalized = unnormalize(candidates_normalized.cpu(), train_x_bounds.cpu())

            for z_candidate in candidates_unnormalized.float():
                if len(valid_candidates_in_batch) >= TARGET_BATCH_SIZE:
                    break 

                result = decode_and_validate(
                    z_candidate.unsqueeze(0), model, dataset, device,
                    DECODING_ATTEMPTS, DECODING_NOISE
                )
                if result:
                    valid_z, valid_smi = result
                    logp = Descriptors.MolLogP(MolFromSmiles(valid_smi))

                    # Ensure new data is on the correct device before concatenation
                    valid_candidates_in_batch.append(valid_z.to(device, dtype=torch.double))
                    new_observations_in_batch.append(torch.tensor([[logp]], device=device, dtype=torch.double))
                    all_generated_candidates.append({'SMILES': valid_smi, 'logP': logp})
                    pbar.update(1)

    train_x = torch.cat([train_x, *valid_candidates_in_batch])
    train_y = torch.cat([train_y, *new_observations_in_batch])

    best_logp_so_far = train_y.max().item()
    print(f"Batch {i+1:02d} complete. Generated {num_generated_total} total candidates to find {TARGET_BATCH_SIZE} valid ones.")
    print(f"Best logP so far: {best_logp_so_far:.4f}")

# --- 3. Final Results ---
print("\n--- Batched Bayesian Optimization Finished ---")

if not all_generated_candidates:
    print("No valid candidates were generated during the optimization.")
else:
    candidates_df = pd.DataFrame(all_generated_candidates)
    candidates_df = candidates_df.sort_values(by="logP", ascending=False).drop_duplicates(subset=['SMILES']).reset_index(drop=True)
    candidates_df.to_csv(OUTPUT_CANDIDATES_CSV, index=False)

    print(f"Successfully generated {len(candidates_df)} unique molecules.")
    print(f"All candidates saved to: {OUTPUT_CANDIDATES_CSV}")

    print("\nTop 20 Generated Molecules:")
    print(candidates_df.head(20).to_string())
    
    
    
#%% MULTI-OBJECTIVE QEHVI

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOAD a trained MoleculeVAE checkpoint and run memory-safe
multi-objective Bayesian optimisation (qEHVI / qNEHVI).

If you kept the VAE definition in another file (e.g. `molvae.py`)
  simply replace the two starred import lines with
      from molvae import load_params, ZincDataset, MoleculeVAE
"""

# ─────────── 0) Imports – VAE DEFINITION ───────────
from pathlib import Path
# ★ adjust if your classes live elsewhere

for run in range(4):

    def load_params(path: str | pathlib.Path | None = None) -> Dict[str, Any]:
        default: Dict[str, Any] = {
            "MAX_LEN": 120,
            "val_frac": 0.1,
            "hidden_dim": 196,
            "recurrent_dim": 488,
            "gru_depth": 2,
            "conv_depth": 3,
            "conv_dim_depth": 9,
            "conv_d_growth_factor": 1.4,
            "conv_dim_width": 9,
            "conv_w_growth_factor": 1.0,
            "batchnorm_conv": True,
            "middle_layer": 0,
            "hg_growth_factor": 1.0,
            "activation": "tanh",
            "batchnorm_mid": False,
            "dropout_rate_mid": 0.2,
            "xent_loss_weight": 1.0,
            "kl_loss_weight": 0.1,
            "prop_loss_weight": 1.0,
            "epochs": 200,
            "batch_size": 128,
            "lr": 3e-4,
            "kl_anneal_epochs": 10,
            "seed": 42,
        }
        if path is not None:
            default.update(json.loads(pathlib.Path(path).read_text()))
        return default
    
    # -----------------------------------------------------------------------------
    # 1) Dataset helpers
    # -----------------------------------------------------------------------------
    
    class ZincDataset(Dataset):
        """Returns `(one_hot_SMILES, torch.tensor([logP, TPSA]))`."""
    
        def __init__(self, csv_path: pathlib.Path, params: Dict[str, Any]):
            import pandas as pd
            self.df = pd.read_csv(csv_path)
            
            self.params = params
            self.chars = sorted({c for s in self.df.smiles.tolist() for c in s})
            self.chars.append(" ")  # padding char
            params["NCHARS"] = len(self.chars)
            self.ch2i = {c: i for i, c in enumerate(self.chars)}
            self.i2ch = {i: c for c, i in self.ch2i.items()}
    
        def __len__(self):
            return len(self.df)
    
        def smiles_to_tensor(self, s: str) -> torch.Tensor:
            L = self.params["MAX_LEN"]
            s = (s[:L] + " " * L)[:L]
            idx = [self.ch2i[c] for c in s]
            return F.one_hot(torch.tensor(idx, dtype=torch.long), num_classes=self.params["NCHARS"]).float()
    
        def tensor_to_smiles(self, t: torch.Tensor) -> str:
            return "".join(self.i2ch[i] for i in t.argmax(-1).tolist()).strip()
    
        def __getitem__(self, i):
            row = self.df.iloc[i]
            x = self.smiles_to_tensor(row.smiles)
            y = torch.tensor([row.logP, row.TPSA], dtype=torch.float32)
            return x, y
    
    # -----------------------------------------------------------------------------
    # 2) Model definition
    # -----------------------------------------------------------------------------
    
    def _growth(b: float, g: float, i: int) -> int:
        return int(round(b * (g ** i)))
    
    class Encoder(nn.Module):
        def __init__(self, p):
            super().__init__()
            layers = []
            in_ch = p["NCHARS"]
            for j in range(p["conv_depth"]):
                out_ch = _growth(p["conv_dim_depth"], p["conv_d_growth_factor"], j)
                k = _growth(p["conv_dim_width"], p["conv_w_growth_factor"], j)
                layers.extend([nn.Conv1d(in_ch, out_ch, k, padding="same"), nn.Tanh()])
                if p["batchnorm_conv"]:
                    layers.append(nn.BatchNorm1d(out_ch))
                in_ch = out_ch
            self.conv = nn.Sequential(*layers)
            self.fc   = nn.Linear(p["MAX_LEN"] * in_ch, p["hidden_dim"])
            self.mu   = nn.Linear(p["hidden_dim"], p["hidden_dim"])
            self.logv = nn.Linear(p["hidden_dim"], p["hidden_dim"])
        def forward(self,x):
            x = self.conv(x.permute(0,2,1))
            x = torch.tanh(self.fc(torch.flatten(x,1)))
            return self.mu(x), self.logv(x)
    
    class Decoder(nn.Module):
        def __init__(self,p):
            super().__init__()
            self.p = p
            self.rep = nn.Linear(p["hidden_dim"], p["recurrent_dim"])
            self.gru = nn.GRU(p["recurrent_dim"], p["recurrent_dim"], p["gru_depth"], batch_first=True)
            self.out = nn.Linear(p["recurrent_dim"], p["NCHARS"])
        def forward(self,z):
            rep = torch.tanh(self.rep(z)).unsqueeze(1).repeat(1,self.p["MAX_LEN"],1)
            h,_ = self.gru(rep)
            return self.out(h)
    
    class PropertyHead(nn.Module):
        def __init__(self,p):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(p["hidden_dim"],256), nn.ReLU(), nn.Linear(256,2))
        def forward(self,z):
            return self.net(z)
    
    class MoleculeVAE(nn.Module):
        def __init__(self,p):
            super().__init__()
            self.p=p
            self.enc=Encoder(p)
            self.dec=Decoder(p)
            self.prop=PropertyHead(p)
        def reparam(self,mu,logv):
            return mu + torch.randn_like(logv)*torch.exp(0.5*logv)
        def forward(self,x):
            mu,logv = self.enc(x)
            z = self.reparam(mu,logv)
            logits = self.dec(z)
            props  = self.prop(z)
            return logits, mu, logv, props
        def loss(self,x,y,out,epoch):
            logits,mu,logv,props=out
            B,L,V=logits.shape
            recon = F.cross_entropy(logits.view(B*L,V), x.argmax(-1).view(B*L))
            kl = -0.5*torch.mean(1+logv - mu.pow(2) - logv.exp())
            beta = min(1.0, epoch/self.p["kl_anneal_epochs"])
            prop_loss = F.mse_loss(props,y)
            total = self.p["xent_loss_weight"]*recon + beta*self.p["kl_loss_weight"]*kl + self.p["prop_loss_weight"]*prop_loss
            return total, recon, prop_loss
    
    
    from pathlib import Path
    
    
    import torch, random, pandas as pd, numpy as np
    from tqdm.auto import tqdm
    from rdkit.Chem import Descriptors, MolFromSmiles
    
    from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP, ModelListGP
    from botorch.optim import optimize_acqf
    from botorch.sampling import SobolQMCNormalSampler
    from botorch.utils.transforms import normalize, unnormalize
    from botorch.utils.multi_objective import infer_reference_point
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning)
    from gpytorch.mlls import SumMarginalLogLikelihood
    
    # ─────────────────── 1) paths & constants ───────────────────
    CSV_PATH   = Path("/home/muthyala.7/Downloads/250k_rndm_zinc_drugs_clean_3_TPSA.csv")
    CKPT_PATH  = Path("runs/vae/best_multi.pt")
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    
    # memory-tuned BO knobs
    torch.set_default_dtype(torch.float)   # float32 everywhere
    MC_SAMPLES        = 64                # ↓ from 128
    SUB_BATCH_SIZE    = 3                 # q in qEHVI
    NUM_RESTARTS      = 5
    RAW_SAMPLES       = 128
    BASELINE_CAP      = 1500              # keep at most 1 500 points in baseline
    N_INITIAL_SAMPLES = 1000
    N_BO_ITERATIONS   = 20
    TARGET_BATCH_SIZE = 50
    OBJECTIVES        = ["logP", "TPSA"]
    MAXIMIZE          = [True,  False]
    OUT_CSV           = Path(f"mo_bo_candidates_qehvi_run_{run}.csv")
    
    # ─────────────────── 2) dataset & VAE ───────────────────
    params  = load_params()
    dataset = ZincDataset(CSV_PATH, params)
    model   = MoleculeVAE(params).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()
    
    def collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)
    
    dl_val = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False,
        pin_memory=True, collate_fn=collate)
    
    # ─────────────────── 3) helpers ───────────────────
    @torch.no_grad()
    def encode_validation():
        z_list, y_list = [], []
        for xb, yb in dl_val:
            mu, _ = model.enc(xb.to(DEVICE))
            z_list.append(mu.cpu()); y_list.append(yb)
        return torch.cat(z_list), torch.cat(y_list)
    
    def calc_props(smi: str):
        mol = MolFromSmiles(smi)
        if mol is None:
            return None
        return {"logP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol)}
    
    def prepare_y(raw_y: torch.Tensor) -> torch.Tensor:
        y = raw_y[:, [0, 1]].clone()
        if not MAXIMIZE[1]:  # TPSA minimised
            y[:, 1].neg_()
        return y
    
    def decode_and_validate(z, tries=25, noise=0.05):
        with torch.no_grad():
            smi = dataset.tensor_to_smiles(model.dec(z.to(DEVICE))[0].cpu())
            props = calc_props(smi)
            if props:
                return z, smi, props
            for _ in range(tries - 1):
                z2 = z + torch.randn_like(z) * noise
                smi = dataset.tensor_to_smiles(model.dec(z2.to(DEVICE))[0].cpu())
                props = calc_props(smi)
                if props:
                    return z2, smi, props
        return None
    
    # ─────────────────── 4) initial design ───────────────────
    print("\n── encoding reference set ──")
    all_z, all_raw = encode_validation()
    all_y = prepare_y(all_raw)
    
    INIT_SEED = run            # ← tweak for each experiment
    torch.manual_seed(INIT_SEED)
    
    perm     = torch.randperm(len(all_z))[:N_INITIAL_SAMPLES]
    train_x  = all_z[perm].to(DEVICE)
    train_y  = all_y[perm].to(DEVICE)
    
    sampler = SobolQMCNormalSampler(torch.Size([MC_SAMPLES]),
                                    seed=random.randint(0, 2**32 - 1))
    
    generated = []
    
    # ─────────────────── 5) BO loop ───────────────────
    for it in range(N_BO_ITERATIONS):
        print(f"\n── BO batch {it+1}/{N_BO_ITERATIONS} ──")
    
        # cap baseline to keep EHVI tensor bounded
        if len(train_x) > BASELINE_CAP:
            keep = torch.randperm(len(train_x))[:BASELINE_CAP]
            base_x, base_y = train_x[keep], train_y[keep]
        else:
            base_x, base_y = train_x, train_y
    
        # normalise X
        bounds   = torch.stack([base_x.min(0).values, base_x.max(0).values])
        x_norm   = normalize(base_x, bounds)
        y_norm   = (base_y - base_y.mean(0)) / base_y.std(0)
    
        # independent GPs
        gps   = [SingleTaskGP(x_norm, y_norm[:, j:j+1]) for j in range(2)]
        mlist = ModelListGP(*gps)
        mll   = SumMarginalLogLikelihood(mlist.likelihood, mlist)
        fit_gpytorch_mll(mll)
    
        # qEHVI acquisition
        ref  = infer_reference_point(y_norm)
        part = FastNondominatedPartitioning(ref, y_norm)
        acqf = qExpectedHypervolumeImprovement(
            model=mlist,
            ref_point=ref,
            partitioning=part,
            sampler=sampler,
        )  # stays on GPU
    
        # free anything left from GP fit
        torch.cuda.empty_cache()
    
        d = train_x.shape[1]
        norm_bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.float, device=DEVICE)
    
        good_z, good_y = [], []
        with tqdm(total=TARGET_BATCH_SIZE, desc="collect", leave=False) as bar:
            while len(good_z) < TARGET_BATCH_SIZE:
                cand_n, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=norm_bounds,
                    q=SUB_BATCH_SIZE,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,
                    sequential=True,
                )
                cand = unnormalize(cand_n.cpu(), bounds.cpu())
    
                for z in cand.float():
                    if len(good_z) >= TARGET_BATCH_SIZE:
                        break
                    out = decode_and_validate(z.unsqueeze(0))
                    if out is None:
                        continue
                    vz, smi, props = out
                    obj = torch.tensor(
                        [props["logP"], -props["TPSA"]],  # TPSA already negated
                        dtype=torch.float,
                    )
                    good_z.append(vz.to(DEVICE))
                    good_y.append(obj.unsqueeze(0).to(DEVICE))
                    generated.append({"SMILES": smi, **props})
                    bar.update(1)
    
        # expand design set
        train_x = torch.cat([train_x, *good_z])
        train_y = torch.cat([train_y, *good_y])
    
        # house-keeping to drop large tensors
        del gps, mlist, mll, acqf, cand_n, cand, good_z, good_y, base_x, base_y
        torch.cuda.empty_cache()
    
        print(f"✓ dataset size now {len(train_x)}")
    
    # ─────────────────── 6) save results ───────────────────
    if generated:
        df = (pd.DataFrame(generated)
                .drop_duplicates("SMILES")
                .reset_index(drop=True))
        df.to_csv(OUT_CSV, index=False)
        print(f"\n{len(df)} unique molecules saved to {OUT_CSV}")
    else:
        print("\nno valid molecules generated.")



#%%

import warnings
warnings.filterwarnings('ignore')
import torch
import pathlib
import random
import pandas as pd
from tqdm.auto import tqdm

# --- Additional imports for Bayesian Optimization ---
from botorch.acquisition import MCAcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize

from rdkit.Chem import MolFromSmiles
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json

# ======================= USER CONFIG =======================
# --- File Paths ---
CSV_PATH = pathlib.Path('/home/muthyala.7/Downloads/250k_rndm_zinc_drugs_clean_3_TPSA.csv') #<-- Path to your dataset
MODEL_CHECKPOINT = pathlib.Path("/home/muthyala.7/runs/vae/best_multi.pt") #<-- IMPORTANT: Path to your saved model
PARAMS_JSON = None #<-- Path to hyperparameter JSON if you used one, else None

import torch
import pathlib
import random
import pandas as pd
from tqdm.auto import tqdm

# --- Additional imports for Bayesian Optimization ---
from botorch.acquisition import MCAcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import gpytorch # <-- Import GPyTorch


# =========================================================

# =========================================================
# Helper functions and classes needed to load the VAE model
# =========================================================

def load_params(path: str | pathlib.Path | None = None) -> dict[str, any]:
    default = {
        "MAX_LEN": 120, "hidden_dim": 196, "recurrent_dim": 488, "gru_depth": 2, 
        "conv_depth": 3, "conv_dim_depth": 9, "conv_d_growth_factor": 1.4,
        "conv_dim_width": 9, "conv_w_growth_factor": 1.0, "batchnorm_conv": True,
    }
    if path is not None:
        default.update(json.loads(pathlib.Path(path).read_text()))
    return default

class ZincDataset(Dataset):
    """Returns `(one_hot_SMILES, torch.tensor([logP, TPSA]))`."""

    def __init__(self, csv_path: pathlib.Path, params: Dict[str, Any]):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        
        self.params = params
        self.chars = sorted({c for s in self.df.smiles.tolist() for c in s})
        self.chars.append(" ")  # padding char
        params["NCHARS"] = len(self.chars)
        self.ch2i = {c: i for i, c in enumerate(self.chars)}
        self.i2ch = {i: c for c, i in self.ch2i.items()}

    def __len__(self):
        return len(self.df)

    def smiles_to_tensor(self, s: str) -> torch.Tensor:
        L = self.params["MAX_LEN"]
        s = (s[:L] + " " * L)[:L]
        idx = [self.ch2i[c] for c in s]
        return F.one_hot(torch.tensor(idx, dtype=torch.long), num_classes=self.params["NCHARS"]).float()

    def tensor_to_smiles(self, t: torch.Tensor) -> str:
        return "".join(self.i2ch[i] for i in t.argmax(-1).tolist()).strip()

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = self.smiles_to_tensor(row.smiles)
        y = torch.tensor([row.logP, row.TPSA], dtype=torch.float32)
        return x, y


def _growth(b: float, g: float, i: int) -> int:
    return int(round(b * (g**i)))

class Encoder(nn.Module):
    def __init__(self, p):
        super().__init__(); layers = []; in_ch = p["NCHARS"]
        for j in range(p["conv_depth"]):
            out_ch = _growth(p["conv_dim_depth"], p["conv_d_growth_factor"], j)
            k = _growth(p["conv_dim_width"], p["conv_w_growth_factor"], j)
            layers.extend([nn.Conv1d(in_ch, out_ch, k, padding="same"), nn.Tanh()])
            if p["batchnorm_conv"]: layers.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(p["MAX_LEN"] * in_ch, p["hidden_dim"])
        self.mu, self.logv = nn.Linear(p["hidden_dim"], p["hidden_dim"]), nn.Linear(p["hidden_dim"], p["hidden_dim"])
    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)); x = torch.tanh(self.fc(torch.flatten(x, 1))); return self.mu(x), self.logv(x)

class Decoder(nn.Module):
    def __init__(self, p):
        super().__init__(); self.p = p; self.rep = nn.Linear(p["hidden_dim"], p["recurrent_dim"])
        self.gru = nn.GRU(p["recurrent_dim"], p["recurrent_dim"], p["gru_depth"], batch_first=True)
        self.out = nn.Linear(p["recurrent_dim"], p["NCHARS"])
    def forward(self, z):
        rep = torch.tanh(self.rep(z)).unsqueeze(1).repeat(1, self.p["MAX_LEN"], 1); h, _ = self.gru(rep); return self.out(h)

class MoleculeVAE(nn.Module):
    def __init__(self, p):
        super().__init__(); self.p=p; self.enc=Encoder(p); self.dec=Decoder(p)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-ONLY multi-objective BO (qEHVI) using a trained MoleculeVAE.
Memory is kept <10 GB by: float32 tensors, MC=64, q=3, baseline-cap=1500,
and explicit `del` + `torch.cuda.empty_cache()` calls.
"""

# ─────────────────── 0) imports ───────────────────
from pathlib import Path


import torch, random, pandas as pd, numpy as np
from tqdm.auto import tqdm
from rdkit.Chem import Descriptors, MolFromSmiles

from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.multi_objective import infer_reference_point
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning)
from gpytorch.mlls import SumMarginalLogLikelihood

# ─────────────────── 1) paths & constants ───────────────────
CSV_PATH   = Path("/home/muthyala.7/Downloads/250k_rndm_zinc_drugs_clean_3_TPSA.csv")
CKPT_PATH  = Path("runs/vae/best_multi.pt")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# memory-tuned BO knobs
torch.set_default_dtype(torch.float)   # float32 everywhere
MC_SAMPLES        = 64                # ↓ from 128
SUB_BATCH_SIZE    = 3                 # q in qEHVI
NUM_RESTARTS      = 5
RAW_SAMPLES       = 128
BASELINE_CAP      = 1500              # keep at most 1 500 points in baseline
N_INITIAL_SAMPLES = 1000
N_BO_ITERATIONS   = 20
TARGET_BATCH_SIZE = 50
OBJECTIVES        = ["logP", "TPSA"]
MAXIMIZE          = [True,  False]
OUT_CSV           = Path("mo_bo_candidates_qehvi.csv")

# ─────────────────── 2) dataset & VAE ───────────────────
params  = load_params()
dataset = ZincDataset(CSV_PATH, params)
model   = MoleculeVAE(params).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

dl_val = torch.utils.data.DataLoader(
    dataset, batch_size=256, shuffle=False,
    pin_memory=True, collate_fn=collate)

# ─────────────────── 3) helpers ───────────────────
@torch.no_grad()
def encode_validation():
    z_list, y_list = [], []
    for xb, yb in dl_val:
        mu, _ = model.enc(xb.to(DEVICE))
        z_list.append(mu.cpu()); y_list.append(yb)
    return torch.cat(z_list), torch.cat(y_list)

def calc_props(smi: str):
    mol = MolFromSmiles(smi)
    if mol is None:
        return None
    return {"logP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol)}

def prepare_y(raw_y: torch.Tensor) -> torch.Tensor:
    y = raw_y[:, [0, 1]].clone()
    if not MAXIMIZE[1]:  # TPSA minimised
        y[:, 1].neg_()
    return y

def decode_and_validate(z, tries=25, noise=0.05):
    with torch.no_grad():
        smi = dataset.tensor_to_smiles(model.dec(z.to(DEVICE))[0].cpu())
        props = calc_props(smi)
        if props:
            return z, smi, props
        for _ in range(tries - 1):
            z2 = z + torch.randn_like(z) * noise
            smi = dataset.tensor_to_smiles(model.dec(z2.to(DEVICE))[0].cpu())
            props = calc_props(smi)
            if props:
                return z2, smi, props
    return None

# ─────────────────── 4) initial design ───────────────────
print("\n── encoding reference set ──")
all_z, all_raw = encode_validation()
all_y = prepare_y(all_raw)

perm     = torch.randperm(len(all_z))[:N_INITIAL_SAMPLES]
train_x  = all_z[perm].to(DEVICE)
train_y  = all_y[perm].to(DEVICE)

sampler = SobolQMCNormalSampler(torch.Size([MC_SAMPLES]),
                                seed=random.randint(0, 2**32 - 1))

generated = []

# ─────────────────── 5) BO loop ───────────────────
for it in range(N_BO_ITERATIONS):
    print(f"\n── BO batch {it+1}/{N_BO_ITERATIONS} ──")

    # cap baseline to keep EHVI tensor bounded
    if len(train_x) > BASELINE_CAP:
        keep = torch.randperm(len(train_x))[:BASELINE_CAP]
        base_x, base_y = train_x[keep], train_y[keep]
    else:
        base_x, base_y = train_x, train_y

    # normalise X
    bounds   = torch.stack([base_x.min(0).values, base_x.max(0).values])
    x_norm   = normalize(base_x, bounds)
    y_norm   = (base_y - base_y.mean(0)) / base_y.std(0)

    # independent GPs
    gps   = [SingleTaskGP(x_norm, y_norm[:, j:j+1]) for j in range(2)]
    mlist = ModelListGP(*gps)
    mll   = SumMarginalLogLikelihood(mlist.likelihood, mlist)
    fit_gpytorch_mll(mll)

    # qEHVI acquisition
    ref  = infer_reference_point(y_norm)
    part = FastNondominatedPartitioning(ref, y_norm)
    acqf = qExpectedHypervolumeImprovement(
        model=mlist,
        ref_point=ref,
        partitioning=part,
        sampler=sampler,
    )  # stays on GPU

    # free anything left from GP fit
    torch.cuda.empty_cache()

    d = train_x.shape[1]
    norm_bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.float, device=DEVICE)

    good_z, good_y = [], []
    with tqdm(total=TARGET_BATCH_SIZE, desc="collect", leave=False) as bar:
        while len(good_z) < TARGET_BATCH_SIZE:
            cand_n, _ = optimize_acqf(
                acq_function=acqf,
                bounds=norm_bounds,
                q=SUB_BATCH_SIZE,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                sequential=True,
            )
            cand = unnormalize(cand_n.cpu(), bounds.cpu())

            for z in cand.float():
                if len(good_z) >= TARGET_BATCH_SIZE:
                    break
                out = decode_and_validate(z.unsqueeze(0))
                if out is None:
                    continue
                vz, smi, props = out
                obj = torch.tensor(
                    [props["logP"], -props["TPSA"]],  # TPSA already negated
                    dtype=torch.float,
                )
                good_z.append(vz.to(DEVICE))
                good_y.append(obj.unsqueeze(0).to(DEVICE))
                generated.append({"SMILES": smi, **props})
                bar.update(1)

    # expand design set
    train_x = torch.cat([train_x, *good_z])
    train_y = torch.cat([train_y, *good_y])

    # house-keeping to drop large tensors
    del gps, mlist, mll, acqf, cand_n, cand, good_z, good_y, base_x, base_y
    torch.cuda.empty_cache()

    print(f"dataset size now {len(train_x)}")

# ─────────────────── 6) save results ───────────────────
if generated:
    df = (pd.DataFrame(generated)
            .drop_duplicates("SMILES")
            .reset_index(drop=True))
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{len(df)} unique molecules saved to {OUT_CSV}")
else:
    print("\n⚠ no valid molecules generated.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = load_params(PARAMS_JSON)

model = MoleculeVAE(params).to(device)
model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device), strict=False)
model.eval()
print(f"Successfully loaded model from {MODEL_CHECKPOINT}")


from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.multi_objective import infer_reference_point
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles
import pandas as pd
import torch

# ======================= MULTI-OBJECTIVE BO CONFIG =======================
# --- Multi-Objective Batched Bayesian Optimization Config ---
N_INITIAL_SAMPLES = 1000
N_BO_ITERATIONS = 20
TARGET_BATCH_SIZE = 50
SUB_BATCH_SIZE = 5
DECODING_ATTEMPTS = 25
DECODING_NOISE = 0.05
OUTPUT_CANDIDATES_CSV = pathlib.Path("mo_bo_generated_candidates_qehvi.csv")

# Multi-objective settings
OBJECTIVES = ['logP', 'TPSA']  # Can add more objectives like 'SA_score', 'QED', etc.
MAXIMIZE = [True, False]  # True for maximization, False for minimization
# =========================================================================


def calculate_molecular_properties(smiles: str) -> dict:
    """Calculate logP and TPSA for a given SMILES string."""
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None
    
    properties = {}
    properties['logP'] = Descriptors.MolLogP(mol)
    properties['TPSA'] = Descriptors.TPSA(mol)
    
    return properties


def decode_and_validate_multi_objective(
    z: torch.Tensor, model: MoleculeVAE, dataset: ZincDataset, device: torch.device,
    n_attempts: int, noise_level: float
) -> tuple[torch.Tensor, str, dict] | None:
    """Tries to decode a latent vector into a valid SMILES and calculate properties."""
    with torch.no_grad():
        smi = dataset.tensor_to_smiles(model.dec(z.to(device))[0].cpu())
        properties = calculate_molecular_properties(smi)
        if properties is not None:
            return z, smi, properties
            
        for _ in range(n_attempts - 1):
            perturbed_z = z + torch.randn_like(z) * noise_level
            smi = dataset.tensor_to_smiles(model.dec(perturbed_z.to(device))[0].cpu())
            properties = calculate_molecular_properties(smi)
            if properties is not None:
                return perturbed_z, smi, properties
    return None


def prepare_multi_objective_data(all_y_props: torch.Tensor, objectives: list, maximize: list) -> torch.Tensor:
    """Prepare multi-objective data by selecting objectives and handling maximization/minimization."""
    # Extract the relevant objectives
    obj_indices = []
    for obj in objectives:
        if obj == 'logP':
            obj_indices.append(0)
        elif obj == 'TPSA':
            obj_indices.append(1)
    
    y_multi = all_y_props[:, obj_indices]
    
    # Convert minimization objectives to maximization by negating
    for i, should_maximize in enumerate(maximize):
        if not should_maximize:
            y_multi[:, i] = -y_multi[:, i]
    
    return y_multi


# --- 1. Set up the Multi-Objective Optimization Environment ---
print("\n--- Starting Multi-Objective Batched Bayesian Optimization with qEHVI ---")

model.eval()

print("Encoding validation set to get latent vectors...")


params=load_params(PARAMS_JSON)
dataset=ZincDataset(CSV_PATH, params)
set_seed(params["seed"])
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx = np.arange(len(dataset))
tr_idx,val_idx = train_test_split(idx, test_size=params["val_frac"], random_state=params["seed"])
collate = lambda b:(torch.stack([x for x,_ in b]), torch.stack([y for _,y in b]))

dl_tr = DataLoader(torch.utils.data.Subset(dataset,tr_idx), batch_size=params["batch_size"], shuffle=True, collate_fn=collate, pin_memory=True)
dl_val= DataLoader(torch.utils.data.Subset(dataset,val_idx), batch_size=params["batch_size"], shuffle=False, collate_fn=collate, pin_memory=True)

all_z_list, all_y_list = [], []
with torch.no_grad():
    for xb, yb in tqdm(dl_val, desc="Encoding for Multi-Objective BO"):
        mu, _ = model.enc(xb.to(device))
        all_z_list.append(mu.cpu())
        all_y_list.append(yb.cpu())

all_z = torch.cat(all_z_list)
all_y_props = torch.cat(all_y_list)  # Shape: [N, 2] for logP and TPSA



# Prepare multi-objective data
all_y_multi = prepare_multi_objective_data(all_y_props, OBJECTIVES, MAXIMIZE)

INIT_SEED = 90            # ← tweak for each experiment
torch.manual_seed(INIT_SEED)
n_samples = min(N_INITIAL_SAMPLES, len(all_z))
perm = torch.randperm(len(all_z))[:n_samples]

# Move initial training data to the correct device
train_x = all_z[perm].to(device, dtype=torch.double)
train_y = all_y_multi[perm].to(device, dtype=torch.double)

print(f"Initializing Multi-Objective Gaussian Process with {n_samples} random samples")
print(f"Objectives: {OBJECTIVES}")
print(f"Maximize: {MAXIMIZE}")
print(f"Training data shape: X={train_x.shape}, Y={train_y.shape}")

# --- 2. The Iterative Multi-Objective Bayesian Optimization Loop ---
all_generated_candidates = []
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=random.randint(0, 10000))

for i in range(N_BO_ITERATIONS):
    print(f"\n--- Multi-Objective Batch {i+1}/{N_BO_ITERATIONS} ---")

    # Normalize input space
    train_x_bounds = torch.stack([train_x.min(dim=0).values, train_x.max(dim=0).values])
    train_x_normalized = normalize(train_x, train_x_bounds)

    # Normalize each objective separately
    train_y_normalized = torch.zeros_like(train_y)
    y_means = []
    y_stds = []
    
    for j in range(train_y.shape[1]):
        y_mean = train_y[:, j].mean()
        y_std = train_y[:, j].std()
        y_means.append(y_mean)
        y_stds.append(y_std)
        train_y_normalized[:, j] = (train_y[:, j] - y_mean) / y_std

    # Create individual GPs for each objective
    models = []
    for j in range(train_y.shape[1]):
        gp = SingleTaskGP(train_x_normalized, train_y_normalized[:, j].unsqueeze(-1))
        models.append(gp)
    
    # Create ModelListGP and fit
    model_list = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
    fit_gpytorch_mll(mll)

    # Infer reference point for hypervolume calculation
    ref_point = infer_reference_point(train_y_normalized)
    
    # Create partitioning for hypervolume calculation
    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=train_y_normalized,
    )
    
    # Create qEHVI acquisition function
    acqf = qExpectedHypervolumeImprovement(
        model=model_list,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )

    valid_candidates_in_batch = []
    new_observations_in_batch = []
    num_generated_total = 0

    d = train_x.shape[-1]
    normalized_bounds = torch.tensor([[0.0] * d, [1.0] * d], device=device, dtype=torch.double)

    with tqdm(total=TARGET_BATCH_SIZE, desc=f"Collecting valid multi-objective batch") as pbar:
        while len(valid_candidates_in_batch) < TARGET_BATCH_SIZE:
            candidates_normalized, _ = optimize_acqf(
                acq_function=acqf,
                bounds=normalized_bounds,
                q=SUB_BATCH_SIZE, 
                num_restarts=10, 
                raw_samples=256,
                options={"batch_limit": 5, "maxiter": 200},
            )
            num_generated_total += SUB_BATCH_SIZE

            candidates_unnormalized = unnormalize(candidates_normalized.cpu(), train_x_bounds.cpu())

            for z_candidate in candidates_unnormalized.float():
                if len(valid_candidates_in_batch) >= TARGET_BATCH_SIZE:
                    break 

                result = decode_and_validate_multi_objective(
                    z_candidate.unsqueeze(0), model, dataset, device,
                    DECODING_ATTEMPTS, DECODING_NOISE
                )
                if result:
                    valid_z, valid_smi, properties = result
                    
                    # Extract objective values
                    obj_values = []
                    for obj_name in OBJECTIVES:
                        obj_values.append(properties[obj_name])
                    
                    # Apply maximization/minimization transformation
                    obj_tensor = torch.tensor(obj_values, dtype=torch.double)
                    for j, should_maximize in enumerate(MAXIMIZE):
                        if not should_maximize:
                            obj_tensor[j] = -obj_tensor[j]

                    # Store candidate
                    valid_candidates_in_batch.append(valid_z.to(device, dtype=torch.double))
                    new_observations_in_batch.append(obj_tensor.unsqueeze(0).to(device))
                    
                    # Store for CSV output (with original values, not transformed)
                    candidate_data = {'SMILES': valid_smi}
                    candidate_data.update(properties)
                    all_generated_candidates.append(candidate_data)
                    
                    pbar.update(1)

    # Update training data
    train_x = torch.cat([train_x, *valid_candidates_in_batch])
    train_y = torch.cat([train_y, *new_observations_in_batch])

    # Report progress
    current_pareto_front = train_y
    print(f"Batch {i+1:02d} complete. Generated {num_generated_total} total candidates to find {TARGET_BATCH_SIZE} valid ones.")
    print(f"Current dataset size: {len(train_x)}")
    
    # Print some statistics about current Pareto front
    for j, (obj_name, maximize) in enumerate(zip(OBJECTIVES, MAXIMIZE)):
        values = current_pareto_front[:, j]
        if not maximize:
            values = -values  # Convert back to original scale for reporting
        print(f"  {obj_name}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")

# --- 3. Final Results ---
print("\n--- Multi-Objective Bayesian Optimization Finished ---")

if not all_generated_candidates:
    print("No valid candidates were generated during the optimization.")
else:
    candidates_df = pd.DataFrame(all_generated_candidates)
    candidates_df = candidates_df.drop_duplicates(subset=['SMILES']).reset_index(drop=True)
    candidates_df.to_csv(OUTPUT_CANDIDATES_CSV, index=False)

    print(f"Successfully generated {len(candidates_df)} unique molecules.")
    print(f"All candidates saved to: {OUTPUT_CANDIDATES_CSV}")

    print("\nTop 20 Generated Molecules (sorted by logP):")
    print(candidates_df.sort_values('logP', ascending=False).head(20).to_string())
    
    print("\nTop 20 Generated Molecules (sorted by TPSA):")
    print(candidates_df.sort_values('TPSA', ascending=True).head(20).to_string())
    
    # Optional: Analyze Pareto front
    print("\nPareto Front Analysis:")
    print(f"Total molecules on final Pareto front: {len(candidates_df)}")
    
    # Simple Pareto front identification (for 2D case)
    if len(OBJECTIVES) == 2:
        pareto_mask = []
        for i in range(len(candidates_df)):
            is_pareto = True
            current_logp = candidates_df.iloc[i]['logP']
            current_tpsa = candidates_df.iloc[i]['TPSA']
            
            for j in range(len(candidates_df)):
                if i != j:
                    other_logp = candidates_df.iloc[j]['logP']
                    other_tpsa = candidates_df.iloc[j]['TPSA']
                    
                    # Check if other point dominates current point
                    if (other_logp >= current_logp and other_tpsa <= current_tpsa and 
                        (other_logp > current_logp or other_tpsa < current_tpsa)):
                        is_pareto = False
                        break
            
            pareto_mask.append(is_pareto)
        
        pareto_front = candidates_df[pareto_mask]
        print(f"Molecules on Pareto front: {len(pareto_front)}")
        print("\nPareto Front molecules:")
        print(pareto_front.head(10).to_string())        
        
#%% qPO acquisition



import torch
import pathlib
import random
import pandas as pd
from tqdm.auto import tqdm

# --- Additional imports for Bayesian Optimization ---
from botorch.acquisition import MCAcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize

from rdkit.Chem import MolFromSmiles
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json

# ======================= USER CONFIG =======================
# --- File Paths ---
CSV_PATH = pathlib.Path('/home/muthyala.7/Downloads/250k_rndm_zinc_drugs_clean_3.csv') #<-- Path to your dataset
MODEL_CHECKPOINT = pathlib.Path("/home/muthyala.7/TorchSisso1/new_symantic/runs/vae/best.pt") #<-- IMPORTANT: Path to your saved model
PARAMS_JSON = None #<-- Path to hyperparameter JSON if you used one, else None

import torch
import pathlib
import random
import pandas as pd
from tqdm.auto import tqdm

# --- Additional imports for Bayesian Optimization ---
from botorch.acquisition import MCAcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import gpytorch # <-- Import GPyTorch


# --- Batched Bayesian Optimization Config ---
N_INITIAL_SAMPLES = 2000
N_BO_ITERATIONS = 20
TARGET_BATCH_SIZE = 50
SUB_BATCH_SIZE = 25 
# Reduced slightly as a final memory-saving measure
NUM_RANDOM_CANDIDATES = 3072 
ACQF_EVAL_BATCH_SIZE = 32
DECODING_ATTEMPTS = 25
DECODING_NOISE = 0.05
OUTPUT_CANDIDATES_CSV = pathlib.Path("bo_qPO_generated_candidates_gradient_free_3rd_run.csv")
# =========================================================

# =========================================================
# Helper functions and classes needed to load the VAE model
# =========================================================

def load_params(path: str | pathlib.Path | None = None) -> dict[str, any]:
    default = {
        "MAX_LEN": 120, "hidden_dim": 196, "recurrent_dim": 488, "gru_depth": 2, 
        "conv_depth": 3, "conv_dim_depth": 9, "conv_d_growth_factor": 1.4,
        "conv_dim_width": 9, "conv_w_growth_factor": 1.0, "batchnorm_conv": True,
    }
    if path is not None:
        default.update(json.loads(pathlib.Path(path).read_text()))
    return default

class ZincDataset(Dataset):
    def __init__(self, csv_path: pathlib.Path, params: dict[str, any]):
        df = pd.read_csv(csv_path)
        all_smiles = df.smiles.tolist()
        self.chars = sorted(list(set("".join(all_smiles))))
        self.chars.append(" ")
        params["NCHARS"] = len(self.chars)
        self.ch2i = {c: i for i, c in enumerate(self.chars)}
        self.i2ch = {i: c for c, i in self.ch2i.items()}
        self.params = params

    def smiles_to_tensor(self, s: str) -> torch.Tensor:
        L = self.params["MAX_LEN"]
        s_padded = (s[:L] + " " * L)[:L]
        idx = [self.ch2i.get(c, len(self.chars)-1) for c in s_padded]
        return F.one_hot(torch.tensor(idx, dtype=torch.long), num_classes=self.params["NCHARS"]).float()

    def tensor_to_smiles(self, t: torch.Tensor) -> str:
        return "".join(self.i2ch[i] for i in t.argmax(-1).tolist()).strip()

def _growth(b: float, g: float, i: int) -> int:
    return int(round(b * (g**i)))

class Encoder(nn.Module):
    def __init__(self, p):
        super().__init__(); layers = []; in_ch = p["NCHARS"]
        for j in range(p["conv_depth"]):
            out_ch = _growth(p["conv_dim_depth"], p["conv_d_growth_factor"], j)
            k = _growth(p["conv_dim_width"], p["conv_w_growth_factor"], j)
            layers.extend([nn.Conv1d(in_ch, out_ch, k, padding="same"), nn.Tanh()])
            if p["batchnorm_conv"]: layers.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(p["MAX_LEN"] * in_ch, p["hidden_dim"])
        self.mu, self.logv = nn.Linear(p["hidden_dim"], p["hidden_dim"]), nn.Linear(p["hidden_dim"], p["hidden_dim"])
    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)); x = torch.tanh(self.fc(torch.flatten(x, 1))); return self.mu(x), self.logv(x)

class Decoder(nn.Module):
    def __init__(self, p):
        super().__init__(); self.p = p; self.rep = nn.Linear(p["hidden_dim"], p["recurrent_dim"])
        self.gru = nn.GRU(p["recurrent_dim"], p["recurrent_dim"], p["gru_depth"], batch_first=True)
        self.out = nn.Linear(p["recurrent_dim"], p["NCHARS"])
    def forward(self, z):
        rep = torch.tanh(self.rep(z)).unsqueeze(1).repeat(1, self.p["MAX_LEN"], 1); h, _ = self.gru(rep); return self.out(h)

class MoleculeVAE(nn.Module):
    def __init__(self, p):
        super().__init__(); self.p=p; self.enc=Encoder(p); self.dec=Decoder(p)

def decode_and_validate(
    z: torch.Tensor, model: MoleculeVAE, dataset: ZincDataset, device: torch.device,
    n_attempts: int, noise_level: float
) -> tuple[torch.Tensor, str] | None:
    with torch.no_grad():
        smi = dataset.tensor_to_smiles(model.dec(z.to(device))[0].cpu())
        if MolFromSmiles(smi) is not None: return z, smi
        for _ in range(n_attempts - 1):
            perturbed_z = z + torch.randn_like(z) * noise_level
            smi = dataset.tensor_to_smiles(model.dec(perturbed_z.to(device))[0].cpu())
            if MolFromSmiles(smi) is not None: return perturbed_z, smi
    return None

class qProbabilityOfOptimality(MCAcquisitionFunction):
    def __init__(self, model: Model, sampler=None, **kwargs):
        super().__init__(model=model, sampler=sampler, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        train_X = self.model.train_inputs[0]
        batch_shape = X.shape[:-2]
        expanded_train_X = train_X.expand(*batch_shape, *train_X.shape)
        full_X = torch.cat([expanded_train_X, X], dim=-2)
        full_posterior = self.model.posterior(full_X)
        full_samples = self.sampler(full_posterior).squeeze(-1)
        argmax_indices = torch.argmax(full_samples, dim=-1)
        n_train = train_X.shape[-2]
        q = X.shape[-2]
        candidate_indices = torch.arange(n_train, n_train + q, device=X.device)
        is_optimum_matrix = (argmax_indices.unsqueeze(-1) == candidate_indices)
        prob_per_point = is_optimum_matrix.double().mean(dim=0)
        return prob_per_point.sum(dim=-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = load_params(PARAMS_JSON)

print("--- Loading Pre-Trained VAE and Dataset ---")
dataset = ZincDataset(CSV_PATH, params)
model = MoleculeVAE(params).to(device)
model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device), strict=False)
model.eval()
print(f"Successfully loaded model from {MODEL_CHECKPOINT}")

print("Encoding full dataset to get latent vectors...")
full_df = pd.read_csv(CSV_PATH)
all_smiles = full_df['smiles'].tolist()
all_logp = torch.tensor(full_df['logP'].values, dtype=torch.float32).unsqueeze(-1)

all_z_list = []
with torch.no_grad():
    for i in tqdm(range(0, len(all_smiles), 128), desc="Encoding"):
        batch_tensors = [dataset.smiles_to_tensor(s) for s in all_smiles[i:i+128]]
        xb = torch.stack(batch_tensors).to(device)
        mu, _ = model.enc(xb)
        all_z_list.append(mu.cpu())
all_z = torch.cat(all_z_list)

print("\n--- Starting Guaranteed Batched Bayesian Optimization with qPO (Gradient-Free) ---")
n_samples = min(N_INITIAL_SAMPLES, len(all_z))
perm = torch.randperm(len(all_z))[:n_samples]
train_x = all_z[perm].to(device, dtype=torch.double)
train_y = all_logp[perm].to(device, dtype=torch.double)
print(f"Initializing GP with {n_samples} samples on device: {train_x.device}")

all_generated_candidates = []
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]), seed=random.randint(0, 10000))

for i in range(N_BO_ITERATIONS):
    print(f"\n--- Main Batch {i+1}/{N_BO_ITERATIONS} ---")
    train_x_bounds = torch.stack([train_x.min(dim=0).values, train_x.max(dim=0).values])
    train_x_normalized = normalize(train_x, train_x_bounds)
    y_mean, y_std = train_y.mean(), train_y.std()
    train_y_normalized = (train_y - y_mean) / y_std

    gp = SingleTaskGP(train_x_normalized, train_y_normalized)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    acqf = qProbabilityOfOptimality(model=gp, sampler=sampler)

    valid_candidates_in_batch, new_observations_in_batch, num_generated_total = [], [], 0
    d = train_x.shape[-1]
    
    with gpytorch.settings.fast_pred_var():
        with tqdm(total=TARGET_BATCH_SIZE, desc=f"Collecting valid batch") as pbar:
            while len(valid_candidates_in_batch) < TARGET_BATCH_SIZE:
                random_candidates = torch.rand(
                    NUM_RANDOM_CANDIDATES, SUB_BATCH_SIZE, d, device=device, dtype=torch.double
                )
                
                # THE FIX: No gradients needed, and evaluate in chunks
                with torch.no_grad():
                    acqf_values_list = []
                    for j in range(0, NUM_RANDOM_CANDIDATES, ACQF_EVAL_BATCH_SIZE):
                        chunk = random_candidates[j : j + ACQF_EVAL_BATCH_SIZE]
                        acqf_values_list.append(acqf(chunk))
                    acqf_values = torch.cat(acqf_values_list)

                best_idx = torch.argmax(acqf_values)
                candidates_normalized = random_candidates[best_idx]
                
                # Delete large tensors that are no longer needed in this loop
                del random_candidates
                del acqf_values
                del acqf_values_list
                
                num_generated_total += (NUM_RANDOM_CANDIDATES * SUB_BATCH_SIZE)

                candidates_unnormalized = unnormalize(candidates_normalized.cpu(), train_x_bounds.cpu())

                for z_candidate in candidates_unnormalized.float():
                    if len(valid_candidates_in_batch) >= TARGET_BATCH_SIZE: break 
                    result = decode_and_validate(
                        z_candidate.unsqueeze(0), model, dataset, device, DECODING_ATTEMPTS, DECODING_NOISE
                    )
                    if result:
                        valid_z, valid_smi = result
                        logp = Descriptors.MolLogP(MolFromSmiles(valid_smi))
                        valid_candidates_in_batch.append(valid_z.to(device, dtype=torch.double))
                        new_observations_in_batch.append(torch.tensor([[logp]], device=device, dtype=torch.double))
                        all_generated_candidates.append({'SMILES': valid_smi, 'logP': logp})
                        pbar.update(1)

    train_x = torch.cat([train_x, *valid_candidates_in_batch])
    train_y = torch.cat([train_y, *new_observations_in_batch])

    best_logp_so_far = train_y.max().item()
    print(f"Batch {i+1:02d} complete. Evaluated {num_generated_total:,} total random candidates to find {TARGET_BATCH_SIZE} valid ones.")
    print(f"Best logP so far: {best_logp_so_far:.4f}")
    
    # Ask PyTorch to release cached memory at the end of the main loop
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# --- Final Results ---
print("\n--- Batched Bayesian Optimization Finished ---")
if not all_generated_candidates:
    print("No valid candidates were generated during the optimization.")
else:
    candidates_df = pd.DataFrame(all_generated_candidates)
    candidates_df = candidates_df.sort_values(by="logP", ascending=False).drop_duplicates(subset=['SMILES']).reset_index(drop=True)
    candidates_df.to_csv(OUTPUT_CANDIDATES_CSV, index=False)
    print(f"Successfully generated {len(candidates_df)} unique molecules.")
    print(f"All candidates saved to: {OUTPUT_CANDIDATES_CSV}")
    print("\nTop 20 Generated Molecules:")
    print(candidates_df.head(20).to_string())
