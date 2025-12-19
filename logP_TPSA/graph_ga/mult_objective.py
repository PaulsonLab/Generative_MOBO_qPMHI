#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
moo_ga_weighted_parallel.py

Parallel GA baseline on Jensen's GB_GA:
  • Objectives: logP (maximize), TPSA (minimize)
  • Scalarization: weighted sum of normalized logP and inverted TPSA
  • Hard constraints: SA <= sa_max, len(SMILES) <= max_smiles_len
  • Population represented as SMILES (robust to multiprocessing pickling)
  • Parallel scoring + parallel child generation with time/try caps

Outputs:
  <outdir>/history.csv
  <outdir>/gen_###_top.csv

Example:
  python moo_ga_weighted_parallel.py \
    --outdir runs_logP_TPSA_wsum_constrained \
    --generations 20 --popsize 128 --keep 32 \
    --w_logp 1.0 --w_tpsa 1.0 \
    --logp_min -2 --logp_max 7 --tpsa_min 20 --tpsa_max 180 \
    --sa_max 7.0 --max_smiles_len 108 \
    --sample_from ZINC_250k.smi --sample_n 2000 --seed 0 \
    --n_jobs 8
"""

import os, csv, time, math, random, argparse
from typing import List, Tuple, Optional
from statistics import mean
from multiprocessing import Pool, cpu_count

# --- RDKit ---
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# --- Repo modules (must be importable from repo root) ---
import crossover as co    # uses co.mol_OK(.), co.ring_OK(.)
import mutate as mu       # calls co.mol_OK(.), co.ring_OK(.)
import sascorer           # SA score

# ============ Globals for child workers ============
# crossover.py expects these two globals to exist:
# "parameters set in GA_mol" – we set reasonable defaults here.
co.average_size = 39
co.size_stdev   = 3

# -----------------
# Utilities
# -----------------
def mol_from_smiles(s: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None

def smiles_from_mol(m: Chem.Mol) -> str:
    return Chem.MolToSmiles(m) if m is not None else ""

def unique_smis_keep_order(smis: List[str]) -> List[str]:
    seen, out = set(), []
    for s in smis:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def sample_seed_file(in_path: str, out_path: str, n: int, seed: int) -> None:
    rng = random.Random(seed)
    pool = []
    with open(in_path, "r") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            if Chem.MolFromSmiles(s) is not None:
                pool.append(s)
    pool = unique_smis_keep_order(pool)
    if len(pool) < n:
        raise RuntimeError(f"Only {len(pool)} valid/unique SMILES in {in_path}, need {n}.")
    sample = rng.sample(pool, n)
    with open(out_path, "w") as f:
        for s in sample:
            f.write(s + "\n")

# -----------------
# Objectives & constraints
# -----------------
def calc_logp(m: Chem.Mol) -> float:
    return float(Descriptors.MolLogP(m))

def calc_tpsa(m: Chem.Mol) -> float:
    return float(rdMolDescriptors.CalcTPSA(m))

def calc_sa(m: Chem.Mol) -> float:
    return float(sascorer.calculateScore(m))

def score_weighted_constrained(
    smi: str,
    scoring_args: Tuple[float, float, float, float, float, float, float, int],
) -> Tuple[float, float, float, float, int]:
    """
    Returns (score, logP, TPSA, SA, lenSMI) for a SMILES.
    scoring_args = (w_logp, w_tpsa, logp_min, logp_max, tpsa_min, tpsa_max, sa_max, max_len)
    """
    w_logp, w_tpsa, logp_min, logp_max, tpsa_min, tpsa_max, sa_max, max_len = scoring_args
    if not smi or len(smi) > max_len:
        return (-1e9, math.nan, math.nan, math.nan, len(smi) if smi else 0)

    m = mol_from_smiles(smi)
    if m is None:
        return (-1e9, math.nan, math.nan, math.nan, len(smi))

    sa = calc_sa(m)
    if sa > sa_max:
        return (-1e9, math.nan, math.nan, sa, len(smi))

    lp   = calc_logp(m)
    tpsa = calc_tpsa(m)

    # Normalize
    lp_n   = (lp   - logp_min) / max(1e-8, (logp_max - logp_min))
    tpsa_n = (tpsa_max - tpsa) / max(1e-8, (tpsa_max - tpsa_min))  # invert TPSA
    lp_n   = min(max(lp_n,   0.0), 1.0)
    tpsa_n = min(max(tpsa_n, 0.0), 1.0)

    score = float(w_logp * lp_n + w_tpsa * tpsa_n)
    return (score, lp, tpsa, sa, len(smi))

# -----------------
# Parallel scoring
# -----------------
def _score_worker_init(avg_size: int, std_size: int):
    RDLogger.DisableLog('rdApp.*')
    co.average_size = avg_size
    co.size_stdev   = std_size

def parallel_score(
    smis: List[str],
    scoring_args: Tuple[float, float, float, float, float, float, float, int],
    n_jobs: int,
) -> List[Tuple[float, float, float, float, int]]:
    n_jobs = max(1, n_jobs)
    if n_jobs == 1:
        return [score_weighted_constrained(s, scoring_args) for s in smis]
    with Pool(processes=n_jobs, initializer=_score_worker_init, initargs=(co.average_size, co.size_stdev)) as pool:
        return pool.starmap(score_weighted_constrained, [(s, scoring_args) for s in smis])

# -----------------
# Selection helpers
# -----------------
def normalize_fitness(scores: List[float]) -> List[float]:
    ssum = sum(max(s, 0.0) for s in scores)  # guard against negatives
    if ssum <= 0:
        # fall back to uniform if all are <=0
        return [1.0 / len(scores)] * len(scores)
    return [(max(s, 0.0) / ssum) for s in scores]

def make_mating_pool(pop: List[str], fitness: List[float], pool_size: int, rng: random.Random) -> List[str]:
    # Use choices with weights (Python 3.6+): emulate with cumulative for speed
    cumulative, acc = [], 0.0
    for w in fitness:
        acc += w; cumulative.append(acc)
    mating = []
    for _ in range(pool_size):
        r = rng.random() * acc
        # binary search
        lo, hi = 0, len(cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cumulative[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        mating.append(pop[lo])
    return mating

# -----------------
# Parallel child generation (safe & capped)
# -----------------
def _child_worker_init(avg_size: int, std_size: int, seed_offset: int):
    RDLogger.DisableLog('rdApp.*')
    co.average_size = avg_size
    co.size_stdev   = std_size
    random.seed((os.getpid() * 18181 + int(time.time()) + seed_offset) % 2**31)

def _try_make_child_smiles(
    parentA: str, parentB: str,
    mutation_rate: float,
    max_trials: int = 50
) -> Optional[str]:
    """Try capped attempts to make a valid (sanitizable + ring_OK/mol_OK) child."""
    mA = mol_from_smiles(parentA)
    mB = mol_from_smiles(parentB)
    if mA is None or mB is None:
        return None

    for _ in range(max_trials):
        child = co.crossover(mA, mB)
        if child is None:
            continue
        child = mu.mutate(child, mutation_rate)
        if child is None:
            continue
        smi = smiles_from_mol(child)
        if not smi:
            continue
        return smi
    return None

def parallel_make_children(
    mating_pool: List[str],
    population_size: int,
    mutation_rate: float,
    n_jobs: int,
    seed_offset: int,
    max_trials_per_pair: int = 50,
) -> Tuple[List[str], int]:
    """
    Returns (children, n_success) where children has length <= population_size.
    Fills any missing slots with random clones from the mating pool to guarantee progress.
    """
    n_jobs = max(1, n_jobs)
    tasks = []
    rng = random.Random(seed_offset + 12345)
    for _ in range(population_size):
        a = rng.choice(mating_pool)
        b = rng.choice(mating_pool)
        tasks.append((a, b, mutation_rate, max_trials_per_pair))

    if n_jobs == 1:
        out = []
        for args in tasks:
            smi = _try_make_child_smiles(*args)
            if smi is not None:
                out.append(smi)
        n_succ = len(out)
    else:
        with Pool(processes=n_jobs, initializer=_child_worker_init,
                  initargs=(co.average_size, co.size_stdev, seed_offset)) as pool:
            childs = pool.starmap(_try_make_child_smiles, tasks)
        out = [s for s in childs if s is not None]
        n_succ = len(out)

    # Ensure we have population_size by cloning (never stall)
    while len(out) < population_size:
        out.append(rng.choice(mating_pool))
    return out[:population_size], n_succ

# -----------------
# GA main
# -----------------
def run_ga(
    seed_file: str,
    outdir: str,
    generations: int,
    population_size: int,
    elite_keep_for_csv: int,
    mating_pool_size: int,
    mutation_rate: float,
    prune_population: bool,
    seed: int,
    scoring_args: Tuple[float, float, float, float, float, float, float, int],
    n_jobs: int,
):
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)

    # Initial population from seed file
    with open(seed_file, "r") as f:
        seeds = [ln.strip() for ln in f if ln.strip()]
    if len(seeds) < population_size:
        raise RuntimeError(f"Seed file {seed_file} has only {len(seeds)} lines; need {population_size}")

    population = [rng.choice(seeds) for _ in range(population_size)]

    # Scoring (parallel)
    t0 = time.time()
    scored = parallel_score(population, scoring_args, n_jobs)
    scores   = [sc for (sc, *_rest) in scored]

    # Sanitize: keep top-N (and optionally dedupe)
    pop_tuples = list(zip(scores, population, scored))
    pop_tuples.sort(key=lambda t: t[0], reverse=True)
    if prune_population:
        # deduplicate by SMILES while keeping order
        uniq = {}
        new_tuples = []
        for sc, smi, pack in pop_tuples:
            if smi in uniq:
                continue
            uniq[smi] = 1
            new_tuples.append((sc, smi, pack))
        pop_tuples = new_tuples
    pop_tuples = pop_tuples[:population_size]
    population = [t[1] for t in pop_tuples]
    scores     = [t[0] for t in pop_tuples]

    # Logging setup
    csv_path = os.path.join(outdir, f"history_logP_{seed}.csv")
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["generation", "smiles", "score", "logP", "TPSA", "SA", "lenSMI"])

        # log generation 0
        for _, smi, pack in pop_tuples:
            sc, lp, tp, sa, L = pack
            w.writerow([0, smi, sc, lp, tp, sa, L])

        best0, avg0 = max(scores), mean(scores)
        print(f"[gen 00] best={best0:.4f}  avg={avg0:.4f}  pop={len(population)}  score/sec={len(population)/(time.time()-t0+1e-9):.1f}")

        for gen in range(1, generations + 1):
            tg = time.time()

            # Selection
            fitness = normalize_fitness(scores)
            mating_pool = make_mating_pool(population, fitness, mating_pool_size, rng)

            # Children (parallel, safe & capped)
            children, n_ok = parallel_make_children(
                mating_pool, population_size, mutation_rate, n_jobs, seed_offset=seed+gen*17
            )

            # Score children (parallel)
            scored_children = parallel_score(children, scoring_args, n_jobs)
            child_scores = [sc for (sc, *_rest) in scored_children]

            # Merge and sanitize
            merged = list(zip(scores, population, [(sc, lp, tp, sa, L) for (sc, lp, tp, sa, L) in [t for t in scored]]))
            merged += list(zip(child_scores, children, scored_children))
            merged.sort(key=lambda t: t[0], reverse=True)

            if prune_population:
                uniq = {}
                kept = []
                for sc, smi, pack in merged:
                    if smi in uniq:
                        continue
                    uniq[smi] = 1
                    kept.append((sc, smi, pack))
            else:
                kept = merged

            kept = kept[:population_size]
            population = [t[1] for t in kept]
            scores     = [t[0] for t in kept]

            # Log pop
            for _, smi, pack in kept:
                sc, lp, tp, sa, L = pack
                w.writerow([gen, smi, sc, lp, tp, sa, L])

            # Elite CSV (for quick look)
            elites_path = os.path.join(outdir, f"gen_logP_{gen:03d}_top_{seed}.csv")
            with open(elites_path, "w", newline="") as fe:
                we = csv.writer(fe)
                we.writerow(["score", "smiles"])
                for sc, smi, _ in kept[:elite_keep_for_csv]:
                    we.writerow([sc, smi])

            # Progress
            best, avg = max(scores), mean(scores)
            dt = time.time() - tg
            print(f"[gen {gen:02d}] best={best:.4f}  avg={avg:.4f}  children_ok={n_ok}/{population_size}  "
                  f"time={dt:.1f}s  rate={population_size/max(dt,1e-9):.1f} mol/s", flush=True)

    print(f"✅ Finished. Log: {csv_path}")

# -----------------
# CLI
# -----------------
def main():
    for seed in [2000,1234,3452,2,12]:
        ap = argparse.ArgumentParser("Parallel weighted-sum GA (logP↑, TPSA↓) with SA/length constraints")
        # Seed sampling
        ap.add_argument("--sample_from", type=str, default="ZINC_250k.smi")
        ap.add_argument("--sample_out",  type=str, default="seed_2000.smi")
        ap.add_argument("--sample_n",    type=int, default=2000)
        ap.add_argument("--seed",        type=int, default=seed)
        # GA
        ap.add_argument("--outdir", type=str, default="runs_logP_TPSA_wsum_constrained")
        ap.add_argument("--generations", type=int, default=20)
        ap.add_argument("--popsize", type=int, default=2000)
        ap.add_argument("--keep",    type=int, default=1000, help="Only for per-gen elite CSV; GA selection uses full pop")
        ap.add_argument("--mating_pool_size", type=int, default=4000)
        ap.add_argument("--mutation_rate",    type=float, default=1.0)
        ap.add_argument("--prune_population", action="store_true", help="Deduplicate by SMILES when selecting")
        # Scoring / constraints
        ap.add_argument("--w_logp",   type=float, default=1.0)
        ap.add_argument("--w_tpsa",   type=float, default=0.0)
        ap.add_argument("--logp_min", type=float, default=-7.0)
        ap.add_argument("--logp_max", type=float, default=30.0)
        ap.add_argument("--tpsa_min", type=float, default=0.0)
        ap.add_argument("--tpsa_max", type=float, default=500.0)
        ap.add_argument("--sa_max",         type=float, default=7.0)
        ap.add_argument("--max_smiles_len", type=int,   default=200)
        # Parallelism
        ap.add_argument("--n_jobs", type=int, default=max(1, cpu_count()//2))
    
        args = ap.parse_args()
    
        # 0) Sample seed file deterministically
        if not os.path.exists(args.sample_out):
            print(f"[setup] sampling {args.sample_n} seeds from {args.sample_from} -> {args.sample_out}")
            sample_seed_file(args.sample_from, args.sample_out, args.sample_n, args.seed)
        else:
            print(f"[setup] using existing seed file: {args.sample_out}")
    
        # 1) Scoring args pack
        scoring_args = (
            args.w_logp, args.w_tpsa,
            args.logp_min, args.logp_max,
            args.tpsa_min, args.tpsa_max,
            args.sa_max, args.max_smiles_len,
        )
    
        # 2) Run GA
        print(f"[run] gens={args.generations} pop={args.popsize} pool={args.mating_pool_size} "
              f"mut={args.mutation_rate} jobs={args.n_jobs} seed={args.seed}")
        run_ga(
            seed_file=args.sample_out,
            outdir=args.outdir,
            generations=args.generations,
            population_size=args.popsize,
            elite_keep_for_csv=args.keep,
            mating_pool_size=args.mating_pool_size,
            mutation_rate=args.mutation_rate,
            prune_population=args.prune_population,
            seed=args.seed,
            scoring_args=scoring_args,
            n_jobs=args.n_jobs,
        )

if __name__ == "__main__":
    # Allow Spyder "Run file" to behave like CLI
    os.environ["SPYDER_RUN"] = "1"
    main()


#%%


# pip install botorch torch pandas numpy matplotlib
import pandas as pd
import numpy as np
import torch
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------
# 1) Load and prepare points as (logP, -TPSA)
# ----------------------------------------------------------------------------------
def load_history_ga(path, gen_col="generation", logp_col="logP", tpsa_col="TPSA"):
    df = pd.read_csv(path)
    df = df[[gen_col, logp_col, tpsa_col]].dropna()
    df["obj1"] = df[logp_col].astype(float)
    df["obj2"] = -df[tpsa_col].astype(float)  # maximize -TPSA
    return df[[gen_col, "obj1", "obj2"]].rename(columns={gen_col: "iter"})

def load_history_bo(path, iter_col="iter", logp_col="logP", tpsa_col="TPSA"):
    df = pd.read_csv(path)
    df = df[[iter_col, logp_col, tpsa_col]].dropna()
    df["obj1"] = df[logp_col].astype(float)
    df["obj2"] = -df[tpsa_col].astype(float)
    return df[[iter_col, "obj1", "obj2"]]

# ----------------------------------------------------------------------------------
# 2) Reference point helpers
# ----------------------------------------------------------------------------------
def ref_from_domain(logp_low=-2.0, tpsa_high=180.0, margin=5.0):
    # (logP, -TPSA): worst ≈ (logp_low, -tpsa_high). Make strictly worse with margin.
    return torch.tensor([logp_low - margin, -tpsa_high - margin], dtype=torch.double)

def ref_from_data_union(dfs, margin=1.0):
    """dfs: list of dataframes with columns ['iter','obj1','obj2']"""
    all_pts = pd.concat(dfs, axis=0, ignore_index=True)
    r1 = all_pts["obj1"].min() - margin
    r2 = all_pts["obj2"].min() - margin
    return torch.tensor([r1, r2], dtype=torch.double)

def ref_from_quantile_union(dfs, q=0.01, margin=1.0):
    all_pts = pd.concat(dfs, axis=0, ignore_index=True)
    r1 = all_pts["obj1"].quantile(q) - margin
    r2 = all_pts["obj2"].quantile(q) - margin
    return torch.tensor([r1, r2], dtype=torch.double)

# ----------------------------------------------------------------------------------
# 3) Compute cumulative HV per iteration
# ----------------------------------------------------------------------------------
def hv_per_iter(df, ref_point, iter_start=0):
    """
    df: columns ['iter','obj1','obj2'] for a single method
    ref_point: torch.Size([2])
    Returns: DataFrame with ['iter','hv']
    """
    iters = sorted(df["iter"].unique())
    out = []
    cum_pts = []

    for it in iters:
        batch = df[df["iter"] == it][["obj1", "obj2"]].values
        if batch.size == 0:
            out.append((it, (out[-1][1] if out else 0.0)))
            continue
        cum_pts.append(batch)
        pts = torch.tensor(np.vstack(cum_pts), dtype=torch.double)

        # non-dominated filtering
        mask = is_non_dominated(pts)
        nd = pts[mask]

        hv = Hypervolume(ref_point=ref_point).compute(nd)
        out.append((it, float(hv)))
    return pd.DataFrame(out, columns=["iter", "hv"])

# ----------------------------------------------------------------------------------
# 4) Example usage
# ----------------------------------------------------------------------------------
# GA history (produced by the GA script)
ga_df = load_history_ga("runs_logP_TPSA_wsum_constrained/history_1234.csv", gen_col="generation")

# (Optional) BO history
# bo_df = load_history_bo("bo_history.csv", iter_col="iteration")

# Choose ONE reference policy:
# A) Fixed domain-based (consistent across runs):
ref = ref_from_domain(logp_low=-7.0, tpsa_high=500.0, margin=0.5)
print(ref)
ref = torch.tensor([-7.0,-100.0], dtype=torch.float64)
print(ref)
# B) Or data-driven (union of GA and BO):
# ref = ref_from_data_union([ga_df, bo_df], margin=1.0)
# C) Or quantile-based:
# ref = ref_from_quantile_union([ga_df, bo_df], q=0.01, margin=1.0)

# Compute HV curves
ga_hv = hv_per_iter(ga_df, ref_point=ref)

# If you have BO too:
# bo_hv = hv_per_iter(bo_df, ref_point=ref)

# Plot
plt.figure(figsize=(6,4))
plt.plot(ga_hv["iter"], ga_hv["hv"], label="GA (weighted-sum)")
# if 'bo_hv' in locals():
#     plt.plot(bo_hv["iter"], bo_hv["hv"], label="qPMHI (BO)")
plt.xlabel("Iteration")
plt.ylabel("Hypervolume")
plt.title("HV vs Iteration (ref = [{:.1f}, {:.1f}])".format(ref[0].item(), ref[1].item()))
plt.legend()
plt.tight_layout()
plt.show()
