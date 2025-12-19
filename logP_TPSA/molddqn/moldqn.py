# ==============================================================================
#                      Molecule DQN â€“ Memory Optimized
# ==============================================================================
from __future__ import annotations

import math
import random
import pathlib
import collections
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import rdBase
rdBase.DisableLog('rdApp.error') # Suppress RDKit errors

# --- MODIFICATION ---
# Import Automatic Mixed Precision (AMP) utilities
from torch.cuda.amp import GradScaler, autocast

# ===================== USER CONFIG =====================
CSV_PATH = pathlib.Path('/home/muthyala.7/TorchSisso1/new_symantic/zinc_sampled_dataset_logp_TPSA.csv')


MOLDQN_CONFIG: Dict[str, Any] = {
    # Model Architecture
    "embedding_dim": 128,
    "hidden_dim": 512, # Consider reducing this (e.g., to 256) if memory is still an issue
    "num_layers": 3,
    # Pre-training (Supervised)
    "pretrain_lr": 1e-4,
    "pretrain_epochs": 5,
    # Fine-tuning (Reinforcement Learning)
    "finetune_lr": 1e-5,
    "finetune_episodes": 2000,
    "replay_buffer_size": 200,
    # --- MODIFICATION ---
    # Reduced batch size is the most effective way to save memory.
    "batch_size": 64,  # Was 128
    "gamma": 0.99,
    "max_len": 120,
    # Epsilon-Greedy Policy
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    # Generation
    "num_generate": 1000,
    "target_logp": 4.0,
    "seed": 42,
}
# =======================================================

# Agent & Environment Classes remain the same...
class MolDQN(nn.Module):
    """DQN Agent based on a GRU architecture."""
    def __init__(self, n_chars: int, config: Dict[str, Any]):
        super().__init__()
        self.n_chars = n_chars
        self.embedding = nn.Embedding(num_embeddings=n_chars, embedding_dim=config["embedding_dim"])
        self.gru = nn.GRU(
            input_size=config["embedding_dim"],
            hidden_size=config["hidden_dim"],
            num_layers=config["num_layers"],
            batch_first=True,
        )
        self.out = nn.Linear(config["hidden_dim"], n_chars)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_emb = self.embedding(x)
        gru_out, h_out = self.gru(x_emb, h)
        q_values = self.out(gru_out)
        return q_values, h_out

class ReplayBuffer:
    """A simple FIFO experience replay buffer for DQN training."""
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

class MoleculeEnv:
    """A lightweight environment for generating molecules step-by-step."""
    def __init__(self, char_map: Dict[str, int], i2ch_map: Dict[int, str], config: Dict[str, Any]):
        self.char_map = char_map
        self.i2ch_map = i2ch_map
        self.config = config
        self.start_token = self.char_map['<start>']
        self.end_token = self.char_map['<end>']
        self.state = None
        self.smiles = ""

    def reset(self) -> torch.Tensor:
        self.state = torch.tensor([[self.start_token]], dtype=torch.long)
        self.smiles = ""
        return self.state

    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool]:
        """Performs one step, returns (next_state, reward, done)."""
        char = self.i2ch_map[action_idx]
        self.smiles += char

        done = (action_idx == self.end_token) or (len(self.smiles) >= self.config["max_len"])
        reward = self.get_reward() if done else 0.0

        next_state = torch.cat([self.state, torch.tensor([[action_idx]], dtype=torch.long)], dim=1)
        self.state = next_state
        return next_state, reward, done

    def get_reward(self) -> float:
        """Calculates reward for the final SMILES string."""
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None: return -1.0 # Invalid SMILES penalty
        try:
            logp = Descriptors.MolLogP(mol)
            reward = 1.0 / (1.0 + math.exp(-0.7 * (logp - self.config["target_logp"])))
            return reward
        except:
            return -1.0



"""Execute the full MolDQN pipeline."""
# --- Setup ---
print("--- 1. Initializing Setup ---")
cfg = MOLDQN_CONFIG
random.seed(cfg["seed"])
np.random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- MODIFICATION ---
# Enable AMP if on CUDA
use_amp = (device.type == 'cuda')
print(f"Using Automatic Mixed Precision: {use_amp}")

# =============================================================================
#     df = pd.read_csv(CSV_PATH)
#     smiles_list = df['smiles'].tolist()
#     chars = sorted(list(set("".join(smiles_list))))
#     chars.extend(['<start>', '<end>'])
#     ch2i = {c: i for i, c in enumerate(chars)}
#     i2ch = {i: c for i, c in ch2i.items()}
#     n_chars = len(chars)
# =============================================================================
# --- Load Data and Create Character Mappings ---
df = pd.read_csv(CSV_PATH)
smiles_list = df['smiles'].tolist()
chars = sorted(list(set("".join(smiles_list))))
chars.extend(['<start>', '<end>'])
ch2i = {c: i for i, c in enumerate(chars)}
# --- MODIFICATION ---
# Correctly invert the ch2i dictionary to create the int->char map
i2ch = {i: c for c, i in ch2i.items()} # <--- This is the corrected line
n_chars = len(chars)

policy_net = MolDQN(n_chars, cfg).to(device)
target_net = MolDQN(n_chars, cfg).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

pretrain_model_path = pathlib.Path("moldqn_pretrained.pt")

# ==========================================================================
# STAGE 1: PRE-TRAINING (SUPERVISED LEARNING)
# ==========================================================================
print("\n--- STAGE 1: Pre-training Agent on ZINC ---")
if not pretrain_model_path.exists():
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg["pretrain_lr"])
    loss_fn = nn.CrossEntropyLoss()
    # --- MODIFICATION ---
    # Initialize gradient scaler for AMP
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(1, cfg["pretrain_epochs"] + 1):
        random.shuffle(smiles_list)
        pbar = tqdm(smiles_list, desc=f"Pre-train Epoch {epoch}/{cfg['pretrain_epochs']}")
        for smiles in pbar:
            # --- MODIFICATION ---
            # Build a list of tokens instead of a single string
            tokens = ['<start>'] + list(smiles) + ['<end>']
            
            if len(tokens) > cfg["max_len"]: continue
            
            # Convert the list of tokens to a tensor of indices
            input_seq = torch.tensor([ch2i[token] for token in tokens[:-1]], dtype=torch.long).unsqueeze(0).to(device)
            target_seq = torch.tensor([ch2i[token] for token in tokens[1:]], dtype=torch.long).to(device)

            optimizer.zero_grad()
            # --- MODIFICATION ---
            # Use autocast for the forward pass
            with autocast(enabled=use_amp):
                logits, _ = policy_net(input_seq)
                loss = loss_fn(logits.squeeze(0), target_seq)
            
            # --- MODIFICATION ---
            # Scale the loss and step the optimizer
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=loss.item())

    torch.save(policy_net.state_dict(), pretrain_model_path)
    print(f"âœ“ Pre-trained model saved to {pretrain_model_path}")
else:
    policy_net.load_state_dict(torch.load(pretrain_model_path, map_location=device))
    target_net.load_state_dict(policy_net.state_dict())
    print(f"âœ“ Loaded pre-trained model from {pretrain_model_path}")


# ==========================================================================
# STAGE 2: FINE-TUNING (REINFORCEMENT LEARNING)
# ==========================================================================
print("\n--- STAGE 2: Fine-tuning Agent with RL ---")
optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg["finetune_lr"])
replay_buffer = ReplayBuffer(cfg["replay_buffer_size"])
env = MoleculeEnv(ch2i, i2ch, cfg)
epsilon = cfg["epsilon_start"]

# --- MODIFICATION ---
# Re-initialize scaler for fine-tuning
scaler = GradScaler(enabled=use_amp)

pbar_rl = tqdm(range(1, cfg["finetune_episodes"] + 1), desc="Fine-tuning Episodes")
all_rewards = []

for episode in pbar_rl:
    state = env.reset().to(device)
    hidden = None
    done = False
    
    while not done:
        if random.random() < epsilon:
            action = random.randint(0, n_chars - 1)
        else:
            with torch.no_grad():
                q_values, hidden = policy_net(state, hidden)
                action = q_values[0, -1, :].argmax().item()

        next_state, reward, done = env.step(action)
        
        replay_buffer.push(state.cpu(), action, reward, next_state.cpu(), done)
        state = next_state.to(device)

        if len(replay_buffer) >= cfg["batch_size"]:
            batch = replay_buffer.sample(cfg["batch_size"])
            states, actions, rewards, next_states, dones = zip(*batch)
            squeezed_states = [s.squeeze(0) for s in states]

            state_batch = torch.nn.utils.rnn.pad_sequence(squeezed_states, batch_first=True, padding_value=ch2i['<end>']).to(device)
            action_batch = torch.tensor(actions, device=device)
            reward_batch = torch.tensor(rewards, device=device)
            done_mask = torch.tensor(dones, device=device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # --- MODIFICATION ---
            # Use autocast for the fine-tuning forward passes
            with autocast(enabled=use_amp):
                q_values_current, _ = policy_net(state_batch)
                state_action_values = q_values_current.gather(2, action_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, state_batch.size(1), -1)).squeeze().mean(dim=1)

            
                next_state_values = torch.zeros(cfg["batch_size"], device=device)
                non_final_squeezed = [s.squeeze(0) for s, d in zip(next_states, dones) if not d]
                if non_final_squeezed:
                    non_final_next_states = torch.nn.utils.rnn.pad_sequence(non_final_squeezed, batch_first=True, padding_value=ch2i['<end>']).to(device)
                #non_final_next_states = torch.nn.utils.rnn.pad_sequence([s for s, d in zip(next_states, dones) if not d], batch_first=True, padding_value=ch2i['<end>']).to(device)
                
                if non_final_next_states.nelement() > 0:
                    with torch.no_grad():
                        q_values_next, _ = target_net(non_final_next_states)
                        next_state_values[~done_mask] = q_values_next.max(2)[0].mean(dim=1).to(next_state_values.dtype)
                
                expected_state_action_values = (next_state_values * cfg["gamma"]) + reward_batch
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

            # --- MODIFICATION ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- MODIFICATION ---
            # Explicitly delete large tensors to free memory sooner
            del state_batch, action_batch, reward_batch, done_mask, q_values_current
            del state_action_values, next_state_values, expected_state_action_values, loss
    
    all_rewards.append(reward)
    epsilon = max(cfg["epsilon_end"], epsilon * cfg["epsilon_decay"])
    
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
        avg_reward = np.mean(all_rewards[-100:]) if all_rewards else 0.0
        pbar_rl.set_postfix(avg_reward=f"{avg_reward:.3f}", epsilon=f"{epsilon:.3f}")
        # --- MODIFICATION ---
        # Clear cache periodically
        if use_amp:
            torch.cuda.empty_cache()


# ==========================================================================
# STAGE 3: GENERATION & EVALUATION
# ==========================================================================
print("\n--- STAGE 3: Generating Molecules ---")
policy_net.eval()
generated_molecules = []

for _ in tqdm(range(cfg["num_generate"]), desc="Generating"):
    state = env.reset().to(device)
    hidden = None
    done = False
    while not done:
        # no_grad is crucial for inference memory efficiency
        with torch.no_grad():
            q_values, hidden = policy_net(state, hidden)
            action = q_values[0, -1, :].argmax().item()
        state, reward, done = env.step(action)
        state = state.to(device)
    
    mol = Chem.MolFromSmiles(env.smiles)
    if mol:
        generated_molecules.append({
            "SMILES": env.smiles,
            "logP": Descriptors.MolLogP(mol)
        })
        
if not generated_molecules:
    print("No valid molecules were generated.")
else:
    gen_df = pd.DataFrame(generated_molecules).drop_duplicates(subset=['SMILES']).sort_values("logP", ascending=False)
    print(f"\nâœ“ Generation complete. {len(gen_df)} unique valid molecules found.")
    print("\nTop 20 Generated Molecules:")
    print(gen_df.head(20).to_string(index=False))


#%%


# ==============================================================================
#                      Original MolDQN â€“ Graph-based Molecular Modification
# ==============================================================================
from __future__ import annotations

import math
import random
import pathlib
import collections
from typing import Dict, Any, Tuple, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import BondType
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

from torch.cuda.amp import GradScaler, autocast

# ===================== ORIGINAL MOLDQN CONFIG =====================
CSV_PATH = pathlib.Path('/home/muthyala.7/TorchSisso1/new_symantic/zinc_sampled_dataset_logp_TPSA.csv')

MOLDQN_CONFIG: Dict[str, Any] = {
    # Model Architecture
    "fingerprint_dim": 2048,    # Morgan fingerprint size
    "hidden_dim": 1024,
    "num_layers": 3,
    # Training
    "lr": 1e-4,
    "episodes": 5000,
    "replay_buffer_size": 10000,
    "batch_size": 32,
    "gamma": 0.9,
    "target_update_freq": 100,
    # Epsilon-Greedy Policy
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    # Molecule constraints
    "max_atoms": 50,
    "max_bonds_per_atom": 4,
    "allowed_atoms": ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br'],
    "allowed_bonds": [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE],
    # Generation
    "num_generate": 1000,
    "target_logp": 4.0,
    "num_modifications": 40,  # Max modifications per episode
    "seed": 42,
    # Reward parameters
    "similarity_weight": 0.5,
    "target_weight": 1.0,
    "validity_weight": 1.0,
}

class ActionType(Enum):
    ADD_ATOM = "add_atom"
    ADD_BOND = "add_bond"
    REMOVE_ATOM = "remove_atom"
    REMOVE_BOND = "remove_bond"
    STOP = "stop"

@dataclass
class MolAction:
    action_type: ActionType
    atom_type: Optional[str] = None
    atom_idx: Optional[int] = None
    bond_type: Optional[BondType] = None
    atom_idx_1: Optional[int] = None
    atom_idx_2: Optional[int] = None

class MoleculeFingerprint:
    """Handles molecular fingerprint generation and manipulation."""
    
    @staticmethod
    def get_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
        """Generate Morgan fingerprint for a molecule."""
        if mol is None:
            return np.zeros(n_bits)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    
    @staticmethod
    def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Tanimoto similarity between two fingerprints."""
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        return intersection / union if union > 0 else 0.0

class MoleculeState:
    """Represents the current state of a molecule being modified."""
    
    def __init__(self, mol: Chem.Mol):
        self.mol = Chem.Mol(mol) if mol else None
        self._fingerprint = None
        self._valid = None
    
    @property
    def fingerprint(self) -> np.ndarray:
        """Get cached fingerprint."""
        if self._fingerprint is None:
            self._fingerprint = MoleculeFingerprint.get_morgan_fingerprint(self.mol)
        return self._fingerprint
    
    @property
    def is_valid(self) -> bool:
        """Check if molecule is valid."""
        if self._valid is None:
            self._valid = self.mol is not None and self.mol.GetNumAtoms() > 0
        return self._valid
    
    def copy(self) -> 'MoleculeState':
        """Create a copy of the state."""
        return MoleculeState(self.mol)
    
    def get_smiles(self) -> str:
        """Get SMILES representation."""
        if not self.is_valid:
            return ""
        try:
            return Chem.MolToSmiles(self.mol)
        except:
            return ""

class MoleculeActionSpace:
    """Defines the action space for molecular modifications."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.allowed_atoms = config["allowed_atoms"]
        self.allowed_bonds = config["allowed_bonds"]
        self.max_atoms = config["max_atoms"]
        
        # Create action mapping
        self.actions = self._create_action_space()
        self.action_to_idx = {str(action): idx for idx, action in enumerate(self.actions)}
        self.idx_to_action = {idx: action for idx, action in enumerate(self.actions)}
    
    def _create_action_space(self) -> List[MolAction]:
        """Create the complete action space."""
        actions = []
        
        # Add atom actions
        for atom_type in self.allowed_atoms:
            actions.append(MolAction(ActionType.ADD_ATOM, atom_type=atom_type))
        
        # Add bond actions
        for bond_type in self.allowed_bonds:
            actions.append(MolAction(ActionType.ADD_BOND, bond_type=bond_type))
        
        # Remove actions
        actions.append(MolAction(ActionType.REMOVE_ATOM))
        actions.append(MolAction(ActionType.REMOVE_BOND))
        
        # Stop action
        actions.append(MolAction(ActionType.STOP))
        
        return actions
    
    def get_valid_actions(self, state: MoleculeState) -> List[int]:
        """Get valid action indices for current state."""
        if not state.is_valid:
            return []
        
        valid_actions = []
        mol = state.mol
        
        # Always allow stop
        stop_idx = len(self.actions) - 1
        valid_actions.append(stop_idx)
        
        # Check if we can add atoms
        if mol.GetNumAtoms() < self.max_atoms:
            for i, action in enumerate(self.actions):
                if action.action_type == ActionType.ADD_ATOM:
                    valid_actions.append(i)
        
        # Check if we can add bonds
        if mol.GetNumAtoms() >= 2:
            for i, action in enumerate(self.actions):
                if action.action_type == ActionType.ADD_BOND:
                    valid_actions.append(i)
        
        # Check if we can remove atoms/bonds
        if mol.GetNumAtoms() > 1:
            for i, action in enumerate(self.actions):
                if action.action_type in [ActionType.REMOVE_ATOM, ActionType.REMOVE_BOND]:
                    valid_actions.append(i)
        
        return valid_actions
    
    def size(self) -> int:
        return len(self.actions)

class MoleculeEnvironment:
    """Environment for molecular modification using graph operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.action_space = MoleculeActionSpace(config)
        self.reset()
    
    def reset(self, initial_mol: Optional[Chem.Mol] = None) -> MoleculeState:
        """Reset environment with initial molecule."""
        if initial_mol is None:
            # Start with a simple molecule (methane)
            initial_mol = Chem.MolFromSmiles("C")
        
        self.initial_state = MoleculeState(initial_mol)
        self.current_state = self.initial_state.copy()
        self.step_count = 0
        self.modification_count = 0
        
        return self.current_state
    
    def step(self, action_idx: int) -> Tuple[MoleculeState, float, bool, Dict]:
        """Execute an action and return new state, reward, done, info."""
        if action_idx >= self.action_space.size():
            return self.current_state, -1.0, True, {"error": "Invalid action index"}
        
        action = self.action_space.idx_to_action[action_idx]
        self.step_count += 1
        
        # Check if action is valid
        valid_actions = self.action_space.get_valid_actions(self.current_state)
        if action_idx not in valid_actions:
            return self.current_state, -0.5, False, {"error": "Invalid action for current state"}
        
        # Handle stop action
        if action.action_type == ActionType.STOP:
            reward = self._calculate_reward(self.current_state)
            return self.current_state, reward, True, {"stopped": True}
        
        # Execute action
        new_state = self._execute_action(action)
        
        # Check termination conditions
        done = (self.modification_count >= self.config["num_modifications"] or 
                not new_state.is_valid or
                self.step_count >= 100)
        
        # Calculate reward
        reward = self._calculate_reward(new_state) if done else self._calculate_step_reward(new_state)
        
        self.current_state = new_state
        self.modification_count += 1
        
        return new_state, reward, done, {}
    
    def _execute_action(self, action: MolAction) -> MoleculeState:
        """Execute a molecular modification action."""
        try:
            mol = Chem.RWMol(self.current_state.mol)
            
            if action.action_type == ActionType.ADD_ATOM:
                return self._add_atom(mol, action.atom_type)
            elif action.action_type == ActionType.ADD_BOND:
                return self._add_bond(mol, action.bond_type)
            elif action.action_type == ActionType.REMOVE_ATOM:
                return self._remove_atom(mol)
            elif action.action_type == ActionType.REMOVE_BOND:
                return self._remove_bond(mol)
            
        except Exception as e:
            # Return invalid state on error
            return MoleculeState(None)
        
        return self.current_state
    
    def _add_atom(self, mol: Chem.RWMol, atom_type: str) -> MoleculeState:
        """Add an atom to the molecule."""
        if mol.GetNumAtoms() >= self.config["max_atoms"]:
            return MoleculeState(None)
        
        # Create new atom
        new_atom = Chem.Atom(atom_type)
        new_atom_idx = mol.AddAtom(new_atom)
        
        # Connect to existing atom if molecule has atoms
        if mol.GetNumAtoms() > 1:
            # Find a suitable atom to connect to
            for i in range(mol.GetNumAtoms() - 1):
                atom = mol.GetAtomWithIdx(i)
                if atom.GetDegree() < self.config["max_bonds_per_atom"]:
                    mol.AddBond(i, new_atom_idx, BondType.SINGLE)
                    break
        
        # Sanitize and return
        try:
            Chem.SanitizeMol(mol)
            return MoleculeState(mol.GetMol())
        except:
            return MoleculeState(None)
    
    def _add_bond(self, mol: Chem.RWMol, bond_type: BondType) -> MoleculeState:
        """Add a bond between two atoms."""
        if mol.GetNumAtoms() < 2:
            return MoleculeState(None)
        
        # Find two atoms that can form a bond
        atoms = list(range(mol.GetNumAtoms()))
        random.shuffle(atoms)
        
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                atom1_idx, atom2_idx = atoms[i], atoms[j]
                atom1 = mol.GetAtomWithIdx(atom1_idx)
                atom2 = mol.GetAtomWithIdx(atom2_idx)
                
                # Check if bond already exists
                if mol.GetBondBetweenAtoms(atom1_idx, atom2_idx) is not None:
                    continue
                
                # Check degree constraints
                if (atom1.GetDegree() < self.config["max_bonds_per_atom"] and 
                    atom2.GetDegree() < self.config["max_bonds_per_atom"]):
                    
                    mol.AddBond(atom1_idx, atom2_idx, bond_type)
                    try:
                        Chem.SanitizeMol(mol)
                        return MoleculeState(mol.GetMol())
                    except:
                        return MoleculeState(None)
        
        return MoleculeState(None)
    
    def _remove_atom(self, mol: Chem.RWMol) -> MoleculeState:
        """Remove an atom from the molecule."""
        if mol.GetNumAtoms() <= 1:
            return MoleculeState(None)
        
        # Choose a random atom to remove
        atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
        mol.RemoveAtom(atom_idx)
        
        try:
            Chem.SanitizeMol(mol)
            return MoleculeState(mol.GetMol())
        except:
            return MoleculeState(None)
    
    def _remove_bond(self, mol: Chem.RWMol) -> MoleculeState:
        """Remove a bond from the molecule."""
        if mol.GetNumBonds() == 0:
            return MoleculeState(None)
        
        # Choose a random bond to remove
        bond_idx = random.randint(0, mol.GetNumBonds() - 1)
        bond = mol.GetBondWithIdx(bond_idx)
        mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        
        try:
            Chem.SanitizeMol(mol)
            return MoleculeState(mol.GetMol())
        except:
            return MoleculeState(None)
    
    def _calculate_step_reward(self, state: MoleculeState) -> float:
        """Calculate intermediate reward for a step."""
        if not state.is_valid:
            return -1.0
        return 0.0  # Neutral reward for valid intermediate steps
    
    def _calculate_reward(self, state: MoleculeState) -> float:
        """Calculate final reward for a molecule."""
        if not state.is_valid:
            return -2.0
        
        try:
            mol = state.mol
            
            # Calculate logP
            logp = Descriptors.MolLogP(mol)
            
            # Target-based reward
            target_reward = 1.0 / (1.0 + abs(logp - self.config["target_logp"]))
            
            # Similarity penalty (encourage diversity)
            similarity = MoleculeFingerprint.tanimoto_similarity(
                self.initial_state.fingerprint, 
                state.fingerprint
            )
            similarity_penalty = self.config["similarity_weight"] * similarity
            
            # Validity bonus
            validity_bonus = self.config["validity_weight"]
            
            # Combine rewards
            total_reward = (self.config["target_weight"] * target_reward + 
                          validity_bonus - similarity_penalty)
            
            return total_reward
            
        except Exception as e:
            return -1.0

class OriginalMolDQN(nn.Module):
    """Original MolDQN model using molecular fingerprints."""
    
    def __init__(self, fingerprint_dim: int, action_space_size: int, config: Dict[str, Any]):
        super().__init__()
        self.fingerprint_dim = fingerprint_dim
        self.action_space_size = action_space_size
        
        # Network architecture
        self.fc1 = nn.Linear(fingerprint_dim, config["hidden_dim"])
        self.fc2 = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        self.fc3 = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        self.fc4 = nn.Linear(config["hidden_dim"], action_space_size)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.fc1(fingerprint))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        q_values = self.fc4(x)
        return q_values

class ReplayBuffer:
    """Simple replay buffer for experience replay."""
    
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.BoolTensor(done)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)

def train_original_moldqn():
    """Train the original MolDQN model."""
    print("--- Original MolDQN Training ---")
    cfg = MOLDQN_CONFIG
    
    # Setup
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load initial molecules
    df = pd.read_csv(CSV_PATH)
    initial_molecules = [Chem.MolFromSmiles(smi) for smi in df['smiles'].tolist()[:1000]]
    initial_molecules = [mol for mol in initial_molecules if mol is not None]
    
    # Initialize environment and models
    env = MoleculeEnvironment(cfg)
    action_space_size = env.action_space.size()
    
    policy_net = OriginalMolDQN(cfg["fingerprint_dim"], action_space_size, cfg).to(device)
    target_net = OriginalMolDQN(cfg["fingerprint_dim"], action_space_size, cfg).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg["lr"])
    replay_buffer = ReplayBuffer(cfg["replay_buffer_size"])
    
    epsilon = cfg["epsilon_start"]
    all_rewards = []
    generated_molecules = []
    
    print(f"Action space size: {action_space_size}")
    print(f"Starting training with {len(initial_molecules)} initial molecules")
    
    # Training loop
    pbar = tqdm(range(1, cfg["episodes"] + 1), desc="Training")
    
    for episode in pbar:
        # Choose random initial molecule
        initial_mol = random.choice(initial_molecules)
        state = env.reset(initial_mol)
        
        episode_reward = 0.0
        done = False
        
        while not done:
            # Get current state fingerprint
            current_fp = state.fingerprint
            
            # Epsilon-greedy action selection
            valid_actions = env.action_space.get_valid_actions(state)
            if not valid_actions:
                break
            
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    fp_tensor = torch.FloatTensor(current_fp).unsqueeze(0).to(device)
                    q_values = policy_net(fp_tensor)
                    
                    # Mask invalid actions
                    masked_q = q_values.clone()
                    valid_mask = torch.full((action_space_size,), float('-inf'))
                    valid_mask[valid_actions] = 0
                    masked_q += valid_mask.to(device)
                    
                    action = masked_q.argmax().item()
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Store transition
            replay_buffer.push(
                current_fp, action, reward, 
                next_state.fingerprint, done
            )
            
            state = next_state
            
            # Train if enough samples
            if len(replay_buffer) >= cfg["batch_size"]:
                batch = replay_buffer.sample(cfg["batch_size"])
                if batch is not None:
                    train_step(policy_net, target_net, optimizer, batch, cfg, device)
        
        # Store results
        all_rewards.append(episode_reward)
        
        if state.is_valid:
            mol_data = {
                'smiles': state.get_smiles(),
                'logp': Descriptors.MolLogP(state.mol),
                'reward': episode_reward
            }
            generated_molecules.append(mol_data)
        
        # Update epsilon
        epsilon = max(cfg["epsilon_end"], epsilon * cfg["epsilon_decay"])
        
        # Update target network
        if episode % cfg["target_update_freq"] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Update progress
        if episode % 10 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            valid_count = len(generated_molecules)
            pbar.set_postfix(
                avg_reward=f"{avg_reward:.3f}",
                epsilon=f"{epsilon:.3f}",
                valid_mols=valid_count
            )
    
    # Results
    if generated_molecules:
        results_df = pd.DataFrame(generated_molecules)
        results_df = results_df.drop_duplicates(subset=['smiles']).sort_values('reward', ascending=False)
        
        print(f"\nâœ“ Training complete. Generated {len(results_df)} unique molecules.")
        print("\nTop 10 Generated Molecules:")
        print(results_df.head(10)[['smiles', 'logp', 'reward']].to_string(index=False))
        
        # Save results
        results_df.to_csv('original_moldqn_results.csv', index=False)
        print("Results saved to 'original_moldqn_results.csv'")
    else:
        print("No valid molecules generated.")

def train_step(policy_net, target_net, optimizer, batch, cfg, device):
    """Single training step."""
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
    
    state_batch = state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    done_batch = done_batch.to(device)
    
    # Current Q values
    current_q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    
    # Next Q values
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    target_q_values = reward_batch + (cfg["gamma"] * next_q_values * ~done_batch)
    
    # Loss
    loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


train_original_moldqn()
    
    
  
    
  
#%%

# ==============================================================================
#      Exploratory MolDQN – Maximize logP & Minimize TPSA (Multi-Seed)
# ==============================================================================
from __future__ import annotations

import math
import random
import pathlib
import collections
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import BondType
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

# ===================== EXPLORATORY MOLDQN CONFIG =====================
CSV_PATH = pathlib.Path('/home/muthyala.7/TorchSisso1/new_symantic/zinc_sampled_dataset_logp_TPSA.csv')

MOLDQN_CONFIG: Dict[str, Any] = {
    # Model Architecture
    "fingerprint_dim": 2048, "hidden_dim": 1024, "num_layers": 3,
    # Training
    "lr": 1e-4, "episodes": 5000, "replay_buffer_size": 10000,
    "batch_size": 32, "gamma": 0.9, "target_update_freq": 100,
    # Epsilon-Greedy Policy
    "epsilon_start": 1.0, "epsilon_end": 0.01, "epsilon_decay": 0.995,
    # Molecule constraints
    "max_atoms": 50, "max_bonds_per_atom": 4,
    "allowed_atoms": ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br'],
    "allowed_bonds": [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE],
    # Generation
    "num_modifications": 40,
    # Reward Shaping Parameters for Sigmoid functions
    "logp_k": 0.5, "logp_c": 3.0,
    "tpsa_k": 0.05, "tpsa_c": 80.0,
    # Reward Component Weights
    "similarity_weight": 0.5, "logp_weight": 1.0,
    "tpsa_weight": 1.0, "validity_weight": 1.0,
}

# (All classes: ActionType, MolAction, MoleculeFingerprint, MoleculeState, 
#  MoleculeActionSpace, MoleculeEnvironment, OriginalMolDQN, ReplayBuffer, 
#  and train_step function remain the same as the previous version)

class ActionType(Enum):
    ADD_ATOM = "add_atom"
    ADD_BOND = "add_bond"
    REMOVE_ATOM = "remove_atom"
    REMOVE_BOND = "remove_bond"
    STOP = "stop"

@dataclass
class MolAction:
    action_type: ActionType
    atom_type: Optional[str] = None
    atom_idx: Optional[int] = None
    bond_type: Optional[BondType] = None
    atom_idx_1: Optional[int] = None
    atom_idx_2: Optional[int] = None

class MoleculeFingerprint:
    @staticmethod
    def get_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
        if mol is None: return np.zeros(n_bits)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)

    @staticmethod
    def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        return intersection / union if union > 0 else 0.0

class MoleculeState:
    def __init__(self, mol: Chem.Mol):
        self.mol = Chem.Mol(mol) if mol else None
        self._fingerprint = None
        self._valid = None
    
    @property
    def fingerprint(self) -> np.ndarray:
        if self._fingerprint is None: self._fingerprint = MoleculeFingerprint.get_morgan_fingerprint(self.mol)
        return self._fingerprint
    
    @property
    def is_valid(self) -> bool:
        if self._valid is None: self._valid = self.mol is not None and self.mol.GetNumAtoms() > 0
        return self._valid
    
    def copy(self) -> 'MoleculeState':
        return MoleculeState(self.mol)
    
    def get_smiles(self) -> str:
        if not self.is_valid: return ""
        try: return Chem.MolToSmiles(self.mol)
        except: return ""

class MoleculeActionSpace:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.allowed_atoms = config["allowed_atoms"]
        self.allowed_bonds = config["allowed_bonds"]
        self.max_atoms = config["max_atoms"]
        self.actions = self._create_action_space()
        self.action_to_idx = {str(action): idx for idx, action in enumerate(self.actions)}
        self.idx_to_action = {idx: action for idx, action in enumerate(self.actions)}

    def _create_action_space(self) -> List[MolAction]:
        actions = []
        for atom_type in self.allowed_atoms: actions.append(MolAction(ActionType.ADD_ATOM, atom_type=atom_type))
        for bond_type in self.allowed_bonds: actions.append(MolAction(ActionType.ADD_BOND, bond_type=bond_type))
        actions.append(MolAction(ActionType.REMOVE_ATOM))
        actions.append(MolAction(ActionType.REMOVE_BOND))
        actions.append(MolAction(ActionType.STOP))
        return actions

    def get_valid_actions(self, state: MoleculeState) -> List[int]:
        if not state.is_valid: return []
        valid_actions = [len(self.actions) - 1] # Always allow stop
        mol = state.mol
        if mol.GetNumAtoms() < self.max_atoms:
            for i, action in enumerate(self.actions):
                if action.action_type == ActionType.ADD_ATOM: valid_actions.append(i)
        if mol.GetNumAtoms() >= 2:
            for i, action in enumerate(self.actions):
                if action.action_type == ActionType.ADD_BOND: valid_actions.append(i)
        if mol.GetNumAtoms() > 1:
            for i, action in enumerate(self.actions):
                if action.action_type in [ActionType.REMOVE_ATOM, ActionType.REMOVE_BOND]: valid_actions.append(i)
        return valid_actions
        
    def size(self) -> int:
        return len(self.actions)

class MoleculeEnvironment:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.action_space = MoleculeActionSpace(config)
        self.reset()
    
    def reset(self, initial_mol: Optional[Chem.Mol] = None) -> MoleculeState:
        if initial_mol is None: initial_mol = Chem.MolFromSmiles("C")
        self.initial_state = MoleculeState(initial_mol)
        self.current_state = self.initial_state.copy()
        self.step_count, self.modification_count = 0, 0
        return self.current_state
    
    def step(self, action_idx: int) -> Tuple[MoleculeState, float, bool, Dict]:
        action = self.action_space.idx_to_action.get(action_idx)
        if action is None: return self.current_state, -1.0, True, {"error": "Invalid action index"}

        self.step_count += 1
        if action.action_type == ActionType.STOP:
            reward = self._calculate_reward(self.current_state)
            return self.current_state, reward, True, {}
        
        new_state = self._execute_action(action)
        done = (self.modification_count >= self.config["num_modifications"] or not new_state.is_valid)
        reward = self._calculate_reward(new_state) if done else self._calculate_step_reward(new_state)
        
        self.current_state = new_state
        self.modification_count += 1
        return new_state, reward, done, {}
    
    def _execute_action(self, action: MolAction) -> MoleculeState:
        try:
            mol = Chem.RWMol(self.current_state.mol)
            if action.action_type == ActionType.ADD_ATOM: return self._add_atom(mol, action.atom_type)
            elif action.action_type == ActionType.ADD_BOND: return self._add_bond(mol, action.bond_type)
            elif action.action_type == ActionType.REMOVE_ATOM: return self._remove_atom(mol)
            elif action.action_type == ActionType.REMOVE_BOND: return self._remove_bond(mol)
        except Exception: return MoleculeState(None)
        return self.current_state
    
    def _sanitize_and_get_state(self, mol: Chem.RWMol) -> MoleculeState:
        try:
            Chem.SanitizeMol(mol)
            return MoleculeState(mol.GetMol())
        except: return MoleculeState(None)

    def _add_atom(self, mol: Chem.RWMol, atom_type: str) -> MoleculeState:
        if mol.GetNumAtoms() >= self.config["max_atoms"]: return MoleculeState(None)
        new_atom_idx = mol.AddAtom(Chem.Atom(atom_type))
        if mol.GetNumAtoms() > 1:
            connect_idx = random.choice([i for i, a in enumerate(mol.GetAtoms()) if a.GetDegree() < self.config["max_bonds_per_atom"] and i != new_atom_idx])
            mol.AddBond(connect_idx, new_atom_idx, BondType.SINGLE)
        return self._sanitize_and_get_state(mol)

    def _add_bond(self, mol: Chem.RWMol, bond_type: BondType) -> MoleculeState:
        if mol.GetNumAtoms() < 2: return MoleculeState(None)
        possible_pairs = []
        for i in range(mol.GetNumAtoms()):
            for j in range(i + 1, mol.GetNumAtoms()):
                if mol.GetBondBetweenAtoms(i, j) is None: possible_pairs.append((i,j))
        if not possible_pairs: return MoleculeState(None)
        
        a1_idx, a2_idx = random.choice(possible_pairs)
        mol.AddBond(a1_idx, a2_idx, bond_type)
        return self._sanitize_and_get_state(mol)

    def _remove_atom(self, mol: Chem.RWMol) -> MoleculeState:
        if mol.GetNumAtoms() <= 1: return MoleculeState(None)
        mol.RemoveAtom(random.randint(0, mol.GetNumAtoms() - 1))
        return self._sanitize_and_get_state(mol)

    def _remove_bond(self, mol: Chem.RWMol) -> MoleculeState:
        if mol.GetNumBonds() == 0: return MoleculeState(None)
        bond_idx = random.randint(0, mol.GetNumBonds() - 1)
        bond = mol.GetBondWithIdx(bond_idx)
        mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        return self._sanitize_and_get_state(mol)

    def _calculate_step_reward(self, state: MoleculeState) -> float:
        return -1.0 if not state.is_valid else 0.0
    
    def _calculate_reward(self, state: MoleculeState) -> float:
        if not state.is_valid: return -2.0
        try:
            mol = state.mol
            logp, tpsa = Descriptors.MolLogP(mol), Descriptors.TPSA(mol)
            
            logp_r = 1 / (1 + math.exp(-self.config["logp_k"] * (logp - self.config["logp_c"])))
            tpsa_r = 1 / (1 + math.exp(self.config["tpsa_k"] * (tpsa - self.config["tpsa_c"])))
            
            similarity_p = self.config["similarity_weight"] * MoleculeFingerprint.tanimoto_similarity(
                self.initial_state.fingerprint, state.fingerprint)

            total_reward = (self.config["logp_weight"] * logp_r +
                            self.config["tpsa_weight"] * tpsa_r +
                            self.config["validity_weight"] -
                            similarity_p)
            return total_reward
        except Exception:
            return -1.0

class OriginalMolDQN(nn.Module):
    def __init__(self, fingerprint_dim: int, action_space_size: int, config: Dict[str, Any]):
        super().__init__()
        self.fc1 = nn.Linear(fingerprint_dim, config["hidden_dim"])
        self.fc2 = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        self.fc3 = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        self.fc4 = nn.Linear(config["hidden_dim"], action_space_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, fingerprint: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(fingerprint))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Optional[Tuple]:
        if len(self.buffer) < batch_size: return None
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(np.array(state)), torch.LongTensor(action), 
                torch.FloatTensor(reward), torch.FloatTensor(np.array(next_state)), 
                torch.BoolTensor(done))
    
    def __len__(self) -> int:
        return len(self.buffer)

def train_step(policy_net, target_net, optimizer, batch, cfg, device):
    state, action, reward, next_state, done = batch
    state, action, reward, next_state, done = \
        state.to(device), action.to(device), reward.to(device), next_state.to(device), done.to(device)
    
    current_q = policy_net(state).gather(1, action.unsqueeze(1))
    next_q = target_net(next_state).max(1)[0].detach()
    target_q = reward + (cfg["gamma"] * next_q * ~done)
    
    loss = F.mse_loss(current_q.squeeze(), target_q)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- MODIFIED FUNCTION TO ACCEPT A SEED ---
def train_exploratory_moldqn(seed: int):
    """Train the exploratory MolDQN model for a specific seed."""
    print(f"\n--- Starting Training for Seed: {seed} ---")
    cfg = MOLDQN_CONFIG
    
    # Use the passed seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = pd.read_csv(CSV_PATH)
    initial_molecules = [m for m in [Chem.MolFromSmiles(s) for s in df['smiles'].tolist()[:1000]] if m is not None]
    
    env = MoleculeEnvironment(cfg)
    policy_net = OriginalMolDQN(cfg["fingerprint_dim"], env.action_space.size(), cfg).to(device)
    target_net = OriginalMolDQN(cfg["fingerprint_dim"], env.action_space.size(), cfg).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg["lr"])
    replay_buffer = ReplayBuffer(cfg["replay_buffer_size"])
    epsilon = cfg["epsilon_start"]
    generated_molecules = []

    pbar = tqdm(range(1, cfg["episodes"] + 1), desc=f"Training (Seed {seed})")
    for episode in pbar:
        state = env.reset(random.choice(initial_molecules))
        done = False
        
        while not done:
            if random.random() < epsilon:
                action = random.choice(env.action_space.get_valid_actions(state))
            else:
                with torch.no_grad():
                    fp_tensor = torch.FloatTensor(state.fingerprint).unsqueeze(0).to(device)
                    q_values = policy_net(fp_tensor)
                    action = q_values.argmax().item()
            
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state.fingerprint, action, reward, next_state.fingerprint, done)
            state = next_state
            
            if len(replay_buffer) >= cfg["batch_size"]:
                batch = replay_buffer.sample(cfg["batch_size"])
                if batch: train_step(policy_net, target_net, optimizer, batch, cfg, device)
        
        if state.is_valid:
            mol = state.mol
            generated_molecules.append({
                'smiles': state.get_smiles(),
                'logp': Descriptors.MolLogP(mol), 'tpsa': Descriptors.TPSA(mol)
            })
        
        epsilon = max(cfg["epsilon_end"], epsilon * cfg["epsilon_decay"])
        if episode % cfg["target_update_freq"] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if episode % 100 == 0:
            pbar.set_postfix(epsilon=f"{epsilon:.3f}", valid_mols=len(generated_molecules))

    if generated_molecules:
        results_df = pd.DataFrame(generated_molecules).drop_duplicates(subset=['smiles'])
        results_df = results_df.sort_values(by=['logp', 'tpsa'], ascending=[False, True])
        
        print(f"\n✅ Training for Seed {seed} complete. Generated {len(results_df)} unique molecules.")
        print("\nTop 10 Generated Molecules:")
        print(results_df.head(10).to_string(index=False))
        
        # Create a unique filename for each seed
        output_filename = f'exploratory_moldqn_results_seed_{seed}.csv'
        results_df.to_csv(output_filename, index=False)
        print(f"\nResults saved to '{output_filename}'")


# --- NEW MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # Define the list of seeds you want to run
    seeds_to_run = [2000, 1234, 3452, 2, 12]
    
    print(f"Starting experiments for {len(seeds_to_run)} different seeds...")
    
    for seed in seeds_to_run:
        train_exploratory_moldqn(seed=seed)
        
    print("\nAll seed runs completed.")
    
    
#%%




# Add these imports to your existing code
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import seaborn as sns

# BoTorch imports for hypervolume calculation
try:
    import torch
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    from botorch.utils.multi_objective.pareto import is_non_dominated
    BOTORCH_AVAILABLE = True
except ImportError:
    print("Warning: BoTorch not available. Install with: pip install botorch")
    BOTORCH_AVAILABLE = False

def calculate_objectives(mol):
    """Calculate the two objectives: LogP and TPSA"""
    if mol is None:
        return None, None
    try:
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        return logp, tpsa
    except:
        return None, None

def calculate_hypervolume_botorch(objectives_list, reference_point=None, device='cpu'):
    """
    Calculate hypervolume using BoTorch for a set of 2D objectives.
    
    Args:
        objectives_list: List of tuples [(logp1, tpsa1), (logp2, tpsa2), ...]
        reference_point: Reference point for hypervolume calculation
                        If None, uses (min_logp - 1, -(max_tpsa + 10))
        device: PyTorch device ('cpu' or 'cuda')
    
    Returns:
        hypervolume: Float value of the hypervolume
    """
    if not BOTORCH_AVAILABLE:
        print("BoTorch not available, cannot calculate hypervolume")
        return 0.0
    
    if not objectives_list or len(objectives_list) == 0:
        return 0.0
    
    # Filter out None values
    valid_objectives = [(logp, tpsa) for logp, tpsa in objectives_list 
                       if logp is not None and tpsa is not None]
    
    if len(valid_objectives) == 0:
        return 0.0
    
    try:
        # Convert to torch tensor
        points = torch.tensor(valid_objectives, dtype=torch.float32, device=device)
        
        # For hypervolume in BoTorch, we need to define what "better" means
        # We want to maximize LogP and minimize TPSA
        # BoTorch assumes maximization for all objectives, so we negate TPSA
        # Final objectives: [LogP, -TPSA]
        points_for_hv = torch.column_stack([points[:, 0], -points[:, 1]])
        
        # Set reference point if not provided
        if reference_point is None:
            min_logp = torch.min(points[:, 0]).item()
            max_tpsa = torch.max(points[:, 1]).item()
            # Reference point should be worse than all points for maximization problems
            reference_point = torch.tensor([min_logp - 1.0, -(max_tpsa + 10.0)], 
                                         dtype=torch.float32, device=device)
        else:
            reference_point = torch.tensor(reference_point, dtype=torch.float32, device=device)
        
        # Calculate hypervolume using BoTorch
        hv = Hypervolume(ref_point=reference_point)
        hypervolume = hv.compute(points_for_hv)
        
        # Handle both tensor and float returns
        if isinstance(hypervolume, torch.Tensor):
            return hypervolume.item()
        else:
            return float(hypervolume)
            
    except Exception as e:
        print(f"Error calculating hypervolume: {e}")
        return 0.0

def find_pareto_front_botorch(objectives, device='cpu'):
    """
    Find the Pareto front using BoTorch's is_non_dominated function.
    Assumes we want to maximize LogP and minimize TPSA.
    """
    if not BOTORCH_AVAILABLE or not objectives:
        return []
    
    valid_objectives = [(logp, tpsa) for logp, tpsa in objectives 
                       if logp is not None and tpsa is not None]
    
    if not valid_objectives:
        return []
    
    try:
        # Convert to torch tensor
        points = torch.tensor(valid_objectives, dtype=torch.float32, device=device)
        
        # For Pareto dominance in BoTorch, we need to specify the direction
        # We want to maximize LogP and minimize TPSA
        # BoTorch assumes maximization, so we negate TPSA
        points_for_pareto = torch.column_stack([points[:, 0], -points[:, 1]])
        
        # Find non-dominated points
        pareto_mask = is_non_dominated(points_for_pareto)
        
        # Extract Pareto front points (convert back to original scale)
        pareto_points = points[pareto_mask]
        pareto_front = [(p[0].item(), p[1].item()) for p in pareto_points]
        
        return pareto_front
        
    except Exception as e:
        print(f"Error finding Pareto front: {e}")
        return []

def calculate_dominated_hypervolume(objectives_list, reference_point=None, device='cpu'):
    """
    Calculate hypervolume using only Pareto-optimal points.
    This is more efficient and meaningful for large datasets.
    """
    if not objectives_list:
        return 0.0
    
    # First find the Pareto front
    pareto_front = find_pareto_front_botorch(objectives_list, device)
    
    if not pareto_front:
        return 0.0
    
    # Calculate hypervolume of just the Pareto front
    return calculate_hypervolume_botorch(pareto_front, reference_point, device)

def plot_pareto_front_enhanced(objectives_dict, save_path=None, device='cpu'):
    """
    Enhanced plotting with Pareto fronts highlighted
    
    Args:
        objectives_dict: Dict with seed as key and list of (logp, tpsa) tuples as values
        save_path: Path to save the plot
        device: PyTorch device
    """
    plt.figure(figsize=(14, 10))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(objectives_dict)))
    
    # Plot all points and Pareto fronts
    for i, (seed, objectives) in enumerate(objectives_dict.items()):
        if not objectives:
            continue
            
        valid_objectives = [(logp, tpsa) for logp, tpsa in objectives 
                           if logp is not None and tpsa is not None]
        
        if not valid_objectives:
            continue
        
        # Plot all points
        logps, tpsas = zip(*valid_objectives)
        plt.scatter(logps, tpsas, alpha=0.3, color=colors[i], 
                   label=f'Seed {seed} (all)', s=15)
        
        # Find and plot Pareto front
        if BOTORCH_AVAILABLE:
            pareto_front = find_pareto_front_botorch(valid_objectives, device)
            if pareto_front:
                pf_logps, pf_tpsas = zip(*pareto_front)
                plt.scatter(pf_logps, pf_tpsas, color=colors[i], 
                           s=60, marker='*', edgecolors='black', linewidth=0.5,
                           label=f'Seed {seed} (Pareto)')
    
    plt.xlabel('LogP (Higher is better)', fontsize=12)
    plt.ylabel('TPSA (Lower is better)', fontsize=12)
    plt.title('Multi-Objective Optimization: LogP vs TPSA\n(Stars indicate Pareto-optimal solutions)', 
              fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compute_hypervolume_improvement(objectives_history, window_size=100, device='cpu'):
    """
    Compute hypervolume improvement over time with a sliding window.
    """
    if not BOTORCH_AVAILABLE or len(objectives_history) < window_size:
        return []
    
    improvements = []
    
    try:
        for i in range(window_size, len(objectives_history) + 1, window_size):
            current_objectives = objectives_history[:i]
            prev_objectives = objectives_history[:max(0, i-window_size)]
            
            current_hv = calculate_dominated_hypervolume(current_objectives, device=device)
            prev_hv = calculate_dominated_hypervolume(prev_objectives, device=device) if prev_objectives else 0.0
            
            improvement = current_hv - prev_hv
            improvements.append((i, current_hv, improvement))
    except Exception as e:
        print(f"Error computing hypervolume improvement: {e}")
        return []
    
    return improvements

# Modified training function with BoTorch hypervolume tracking
def train_exploratory_moldqn_with_botorch_hypervolume(seed: int, device='cpu', n_sample=2000):
    """Train the exploratory MolDQN model with BoTorch hypervolume tracking."""
    print(f"\n--- Starting Training for Seed: {seed} ---")
    cfg = MOLDQN_CONFIG
    
    # Use the passed seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {training_device}")
    print(f"Using hypervolume device: {device}")
    
    # Load dataset and sample according to seed
    df = pd.read_csv(CSV_PATH)
    print(f"Original dataset size: {len(df)}")
    
    # Seed-based sampling of the dataset
    df_sampled = df.sample(n=min(n_sample, len(df)), random_state=seed, replace=False).reset_index(drop=True)
    print(f"Sampled dataset size for seed {seed}: {len(df_sampled)}")
    
    # Convert to molecules and calculate initial population objectives
    initial_molecules = []
    initial_objectives = []
    
    for _, row in df_sampled.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            initial_molecules.append(mol)
            logp, tpsa = calculate_objectives(mol)
            if logp is not None and tpsa is not None:
                initial_objectives.append((logp, tpsa))
    
    print(f"Valid initial molecules: {len(initial_molecules)}")
    print(f"Valid initial objectives: {len(initial_objectives)}")
    
    env = MoleculeEnvironment(cfg)
    policy_net = OriginalMolDQN(cfg["fingerprint_dim"], env.action_space.size(), cfg).to(training_device)
    target_net = OriginalMolDQN(cfg["fingerprint_dim"], env.action_space.size(), cfg).to(training_device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg["lr"])
    replay_buffer = ReplayBuffer(cfg["replay_buffer_size"])
    epsilon = cfg["epsilon_start"]
    generated_molecules = []
    
    # Track objectives and hypervolume over time
    hypervolume_history = []
    pareto_hypervolume_history = []
    objectives_history = []
    pareto_front_sizes = []
    
    # Calculate reference point based on initial population (similar to your approach)
    # mu0 represents the initial population objectives
    if initial_objectives:
        try:
            # Convert initial objectives to tensor for BoTorch-style reference point calculation
            mu0 = torch.tensor(initial_objectives, dtype=torch.float32, device=device)
            # For maximization problems, we want reference point below minimum values
            # We negate TPSA since we want to minimize it (BoTorch assumes maximization)
            mu0_for_ref = torch.column_stack([mu0[:, 0], -mu0[:, 1]])  # [LogP, -TPSA]
            reference_point = mu0_for_ref.min(0).values - 1.0
            reference_point = reference_point.tolist()
            print(f"Calculated reference point from initial population: {reference_point}")
        except Exception as e:
            print(f"Error calculating reference point from population: {e}")
            reference_point = [-5.0, -200.0]  # fallback
    else:
        reference_point = [-5.0, -200.0]  # fallback
        print(f"Using fallback reference point: {reference_point}")
    
    print(f"Initial population statistics:")
    if initial_objectives:
        logps, tpsas = zip(*initial_objectives)
        print(f"  LogP range: [{min(logps):.2f}, {max(logps):.2f}]")
        print(f"  TPSA range: [{min(tpsas):.2f}, {max(tpsas):.2f}]")

# Modified training function with pre-training phase
def train_exploratory_moldqn_with_botorch_hypervolume(seed: int, device='cpu', n_sample=2000, 
                                                     pretrain_episodes=5000, experiment_episodes=1000):
    """Train the exploratory MolDQN model with pre-training and experiment phases."""
    print(f"\n--- Starting Training for Seed: {seed} ---")
    print(f"Pre-training: {pretrain_episodes} episodes")
    print(f"Experiment: {experiment_episodes} episodes")
    
    cfg = MOLDQN_CONFIG
    
    # Use the passed seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {training_device}")
    print(f"Using hypervolume device: {device}")
    
    # Load dataset and sample according to seed
    df = pd.read_csv(CSV_PATH)
    print(f"Original dataset size: {len(df)}")
    
    # Seed-based sampling of the dataset
    df_sampled = df.sample(n=min(n_sample, len(df)), random_state=seed, replace=False).reset_index(drop=True)
    print(f"Sampled dataset size for seed {seed}: {len(df_sampled)}")
    
    # Convert to molecules and calculate initial population objectives
    initial_molecules = []
    initial_objectives = []
    
    for _, row in df_sampled.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            initial_molecules.append(mol)
            logp, tpsa = calculate_objectives(mol)
            if logp is not None and tpsa is not None:
                initial_objectives.append((logp, tpsa))
    
    print(f"Valid initial molecules: {len(initial_molecules)}")
    print(f"Valid initial objectives: {len(initial_objectives)}")
    
    env = MoleculeEnvironment(cfg)
    policy_net = OriginalMolDQN(cfg["fingerprint_dim"], env.action_space.size(), cfg).to(training_device)
    target_net = OriginalMolDQN(cfg["fingerprint_dim"], env.action_space.size(), cfg).to(training_device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=cfg["lr"])
    replay_buffer = ReplayBuffer(cfg["replay_buffer_size"])
    epsilon = cfg["epsilon_start"]
    
    # Calculate reference point based on initial population
    if initial_objectives:
        try:
            mu0 = torch.tensor(initial_objectives, dtype=torch.float32, device=device)
            mu0_for_ref = torch.column_stack([mu0[:, 0], -mu0[:, 1]])  # [LogP, -TPSA]
            reference_point = mu0_for_ref.min(0).values - 1.0
            reference_point = reference_point.tolist()
            print(f"Calculated reference point from initial population: {reference_point}")
        except Exception as e:
            print(f"Error calculating reference point from population: {e}")
            reference_point = [-5.0, -200.0]  # fallback
    else:
        reference_point = [-5.0, -200.0]  # fallback
        print(f"Using fallback reference point: {reference_point}")
    
    print(f"Initial population statistics:")
    if initial_objectives:
        logps, tpsas = zip(*initial_objectives)
        print(f"  LogP range: [{min(logps):.2f}, {max(logps):.2f}]")
        print(f"  TPSA range: [{min(tpsas):.2f}, {max(tpsas):.2f}]")
    
    # ===================== PRE-TRAINING PHASE =====================
    print(f"\n🔄 Starting Pre-training Phase ({pretrain_episodes} episodes)")
    pretrain_molecules = []
    
    pbar_pretrain = tqdm(range(1, pretrain_episodes + 1), desc=f"Pre-training (Seed {seed})")
    for episode in pbar_pretrain:
        state = env.reset(random.choice(initial_molecules))
        done = False
        
        while not done:
            if random.random() < epsilon:
                action = random.choice(env.action_space.get_valid_actions(state))
            else:
                with torch.no_grad():
                    fp_tensor = torch.FloatTensor(state.fingerprint).unsqueeze(0).to(training_device)
                    q_values = policy_net(fp_tensor)
                    action = q_values.argmax().item()
            
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state.fingerprint, action, reward, next_state.fingerprint, done)
            state = next_state
            
            if len(replay_buffer) >= cfg["batch_size"]:
                batch = replay_buffer.sample(cfg["batch_size"])
                if batch: 
                    train_step(policy_net, target_net, optimizer, batch, cfg, training_device)
        
        if state.is_valid:
            mol = state.mol
            logp, tpsa = calculate_objectives(mol)
            if logp is not None and tpsa is not None:
                pretrain_molecules.append({
                    'smiles': state.get_smiles(),
                    'logp': logp, 
                    'tpsa': tpsa
                })
        
        epsilon = max(cfg["epsilon_end"], epsilon * cfg["epsilon_decay"])
        if episode % cfg["target_update_freq"] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if episode % 500 == 0:
            pbar_pretrain.set_postfix(epsilon=f"{epsilon:.3f}", 
                                    valid_mols=len(pretrain_molecules),
                                    replay_size=len(replay_buffer))
    
    print(f"✅ Pre-training complete. Generated {len(pretrain_molecules)} molecules during pre-training.")
    print(f"Final epsilon after pre-training: {epsilon:.4f}")
    print(f"Replay buffer size: {len(replay_buffer)}")
    
    # ===================== EXPERIMENT PHASE =====================
    print(f"\n🧪 Starting Experiment Phase ({experiment_episodes} episodes)")
    
    # Reset tracking for the experiment phase
    generated_molecules = []
    hypervolume_history = []
    pareto_hypervolume_history = []
    objectives_history = []
    pareto_front_sizes = []
    
    # Continue with the pre-trained model
    pbar_experiment = tqdm(range(1, experiment_episodes + 1), desc=f"Experiment (Seed {seed})")
    for episode in pbar_experiment:
        state = env.reset(random.choice(initial_molecules))
        done = False
        
        while not done:
            if random.random() < epsilon:
                action = random.choice(env.action_space.get_valid_actions(state))
            else:
                with torch.no_grad():
                    fp_tensor = torch.FloatTensor(state.fingerprint).unsqueeze(0).to(training_device)
                    q_values = policy_net(fp_tensor)
                    action = q_values.argmax().item()
            
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state.fingerprint, action, reward, next_state.fingerprint, done)
            state = next_state
            
            if len(replay_buffer) >= cfg["batch_size"]:
                batch = replay_buffer.sample(cfg["batch_size"])
                if batch: 
                    train_step(policy_net, target_net, optimizer, batch, cfg, training_device)
        
        if state.is_valid:
            mol = state.mol
            logp, tpsa = calculate_objectives(mol)
            if logp is not None and tpsa is not None:
                generated_molecules.append({
                    'smiles': state.get_smiles(),
                    'logp': logp, 
                    'tpsa': tpsa
                })
                objectives_history.append((logp, tpsa))
        
        # Calculate hypervolume every 100 episodes during experiment
        if episode % 100 == 0 and objectives_history:
            hv_all = calculate_hypervolume_botorch(objectives_history, reference_point, device)
            hypervolume_history.append((episode, hv_all))
            
            hv_pareto = calculate_dominated_hypervolume(objectives_history, reference_point, device)
            pareto_hypervolume_history.append((episode, hv_pareto))
            
            pareto_front = find_pareto_front_botorch(objectives_history, device)
            pareto_front_sizes.append((episode, len(pareto_front)))
        
        epsilon = max(cfg["epsilon_end"], epsilon * cfg["epsilon_decay"])
        if episode % cfg["target_update_freq"] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if episode % 100 == 0:
            current_hv = hypervolume_history[-1][1] if hypervolume_history else 0
            current_pf_size = pareto_front_sizes[-1][1] if pareto_front_sizes else 0
            pbar_experiment.set_postfix(epsilon=f"{epsilon:.3f}", 
                                      valid_mols=len(generated_molecules),
                                      hypervolume=f"{current_hv:.2f}",
                                      pareto_size=current_pf_size)

    # Final analysis
    if generated_molecules:
        results_df = pd.DataFrame(generated_molecules).drop_duplicates(subset=['smiles'])
        
        # Calculate final metrics
        final_objectives = [(row['logp'], row['tpsa']) for _, row in results_df.iterrows()]
        final_hypervolume = calculate_hypervolume_botorch(final_objectives, reference_point, device)
        final_pareto_hypervolume = calculate_dominated_hypervolume(final_objectives, reference_point, device)
        pareto_front = find_pareto_front_botorch(final_objectives, device)
        
        print(f"\n✅ Training for Seed {seed} complete.")
        print(f"Generated {len(results_df)} unique molecules.")
        print(f"Final Hypervolume (all points): {final_hypervolume:.4f}")
        print(f"Final Hypervolume (Pareto only): {final_pareto_hypervolume:.4f}")
        print(f"Final Pareto Front Size: {len(pareto_front)}")
        
        # Calculate hypervolume improvement
        hv_improvements = compute_hypervolume_improvement(final_objectives, device=device)
        
        # Save detailed results
        output_filename = f'moldqn_botorch_results_seed_{seed}.csv'
        results_df.to_csv(output_filename, index=False)
        
        # Save hypervolume histories
        hv_df = pd.DataFrame(hypervolume_history, columns=['episode', 'hypervolume_all'])
        hv_pareto_df = pd.DataFrame(pareto_hypervolume_history, columns=['episode', 'hypervolume_pareto'])
        pf_size_df = pd.DataFrame(pareto_front_sizes, columns=['episode', 'pareto_front_size'])
        
        # Merge all tracking data
        tracking_df = hv_df.merge(hv_pareto_df, on='episode', how='outer')
        tracking_df = tracking_df.merge(pf_size_df, on='episode', how='outer')
        tracking_df.to_csv(f'training_metrics_seed_{seed}.csv', index=False)
        
        print(f"Results saved to '{output_filename}'")
        print(f"Training metrics saved to 'training_metrics_seed_{seed}.csv'")
        
        return {
            'seed': seed,
            'final_hypervolume': final_hypervolume,
            'final_pareto_hypervolume': final_pareto_hypervolume,
            'pareto_front_size': len(pareto_front),
            'total_molecules': len(results_df),
            'objectives': final_objectives,
            'pareto_front': pareto_front,
            'hypervolume_history': hypervolume_history,
            'pareto_hypervolume_history': pareto_hypervolume_history,
            'pareto_front_sizes': pareto_front_sizes,
            'hypervolume_improvements': hv_improvements,
            'reference_point': reference_point,
            'initial_population_size': len(initial_molecules),
            'initial_objectives_range': {
                'logp_min': min([obj[0] for obj in initial_objectives]) if initial_objectives else None,
                'logp_max': max([obj[0] for obj in initial_objectives]) if initial_objectives else None,
                'tpsa_min': min([obj[1] for obj in initial_objectives]) if initial_objectives else None,
                'tpsa_max': max([obj[1] for obj in initial_objectives]) if initial_objectives else None,
            }
        }
    
    return None

def run_multi_seed_experiment_with_botorch_analysis(device='cpu', n_sample=2000):
    """Run experiments for multiple seeds with comprehensive BoTorch analysis."""
    seeds_to_run = [2000, 1234, 3452, 2, 12]
    results = {}
    
    print(f"Starting multi-objective experiments with BoTorch for {len(seeds_to_run)} seeds...")
    print(f"Each seed will sample {n_sample} molecules from the dataset")
    print(f"Hypervolume calculations will use device: {device}")
    
    for seed in seeds_to_run:
        result = train_exploratory_moldqn_with_botorch_hypervolume(seed=seed, device=device, n_sample=n_sample)
        if result:
            results[seed] = result
    
    if results:
        # Comprehensive analysis
        print("\n" + "="*70)
        print("MULTI-OBJECTIVE EXPERIMENTAL RESULTS (BoTorch)")
        print("="*70)
        
        # Summary table
        summary_data = []
        objectives_for_plot = {}
        
        for seed, result in results.items():
            summary_data.append({
                'Seed': seed,
                'HV (All Points)': result['final_hypervolume'],
                'HV (Pareto Only)': result['final_pareto_hypervolume'],
                'Pareto Front Size': result['pareto_front_size'],
                'Total Molecules': result['total_molecules'],
                'HV Efficiency': result['final_pareto_hypervolume'] / result['final_hypervolume'] if result['final_hypervolume'] > 0 else 0
            })
            objectives_for_plot[seed] = result['objectives']
        
        summary_df = pd.DataFrame(summary_data)
        print("\nDetailed Results Summary:")
        print(summary_df.to_string(index=False, float_format='{:.4f}'.format))
        
        # Statistical analysis
        hv_all = [result['final_hypervolume'] for result in results.values()]
        hv_pareto = [result['final_pareto_hypervolume'] for result in results.values()]
        pf_sizes = [result['pareto_front_size'] for result in results.values()]
        
        print(f"\nHypervolume Statistics (All Points):")
        print(f"Mean: {np.mean(hv_all):.4f} ± {np.std(hv_all):.4f}")
        print(f"Best: {np.max(hv_all):.4f} (Seed {max(results.keys(), key=lambda k: results[k]['final_hypervolume'])})")
        
        print(f"\nHypervolume Statistics (Pareto Front Only):")
        print(f"Mean: {np.mean(hv_pareto):.4f} ± {np.std(hv_pareto):.4f}")
        print(f"Best: {np.max(hv_pareto):.4f}")
        
        print(f"\nPareto Front Size Statistics:")
        print(f"Mean: {np.mean(pf_sizes):.1f} ± {np.std(pf_sizes):.1f}")
        print(f"Range: {np.min(pf_sizes)} - {np.max(pf_sizes)}")
        
        # Enhanced visualization
        plot_pareto_front_enhanced(objectives_for_plot, 'pareto_fronts_botorch.png', device)
        
        # Plot hypervolume evolution (both all points and Pareto only)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Hypervolume evolution (all points)
        for seed, result in results.items():
            hv_history = result['hypervolume_history']
            if hv_history:
                episodes, hvs = zip(*hv_history)
                ax1.plot(episodes, hvs, label=f'Seed {seed}', marker='o', markersize=3)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Hypervolume (All Points)')
        ax1.set_title('Hypervolume Evolution (All Points)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Hypervolume evolution (Pareto only)
        for seed, result in results.items():
            hv_pareto_history = result['pareto_hypervolume_history']
            if hv_pareto_history:
                episodes, hvs = zip(*hv_pareto_history)
                ax2.plot(episodes, hvs, label=f'Seed {seed}', marker='s', markersize=3)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Hypervolume (Pareto Front)')
        ax2.set_title('Pareto Front Hypervolume Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Pareto front size evolution
        for seed, result in results.items():
            pf_sizes_history = result['pareto_front_sizes']
            if pf_sizes_history:
                episodes, sizes = zip(*pf_sizes_history)
                ax3.plot(episodes, sizes, label=f'Seed {seed}', marker='^', markersize=3)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Pareto Front Size')
        ax3.set_title('Pareto Front Size Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_analysis_botorch.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save comprehensive results including reference point info
        summary_df['Reference_Point_LogP'] = [results[seed].get('reference_point', [None, None])[0] for seed in summary_df['Seed']]
        summary_df['Reference_Point_NegTPSA'] = [results[seed].get('reference_point', [None, None])[1] for seed in summary_df['Seed']]
        summary_df.to_csv('experiment_summary_botorch.csv', index=False)
        print(f"\nComprehensive results saved to 'experiment_summary_botorch.csv'")
        
        # Report reference point consistency
        ref_points = [result.get('reference_point', [None, None]) for result in results.values()]
        print(f"\nReference Point Analysis:")
        for seed, ref_pt in zip(results.keys(), ref_points):
            if ref_pt[0] is not None:
                print(f"  Seed {seed}: [{ref_pt[0]:.3f}, {ref_pt[1]:.3f}]")
        
        # Calculate and report convergence metrics
        print(f"\nConvergence Analysis:")
        for seed, result in results.items():
            if result['hypervolume_improvements']:
                final_improvement = result['hypervolume_improvements'][-1][2] if result['hypervolume_improvements'] else 0
                print(f"Seed {seed}: Final HV improvement = {final_improvement:.4f}")
    
    print("\nAll BoTorch experiments completed!")
    return results

# Example usage:
if __name__ == '__main__':
    # Choose device for hypervolume calculations
    # Use 'cuda' if you have GPU and want faster calculations
    hv_device = 'cuda'  # or 'cpu' if CUDA not available
    
    # Number of molecules to sample from dataset for each seed
    n_sample_molecules = 2000
    
    # Run the enhanced experiment with BoTorch
    results = run_multi_seed_experiment_with_botorch_analysis(
        device=hv_device, 
        n_sample=n_sample_molecules
    )
