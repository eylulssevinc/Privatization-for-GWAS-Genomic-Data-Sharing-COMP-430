from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np


# ----------------------------
# Project root + directories
# ----------------------------

def find_project_root() -> Path:
    """
    Robust repo root detection:
    Define PROJECT_ROOT as the nearest parent directory containing
    either (a) requirements.txt or (b) .git.
    This prevents FileNotFoundError when notebooks are run from subfolders.
    """
    cwd = Path.cwd()
    if (cwd / "requirements.txt").exists() or (cwd / ".git").exists():
        return cwd

    for parent in cwd.parents:
        if (parent / "requirements.txt").exists() or (parent / ".git").exists():
            return parent

    raise RuntimeError(
        f"Could not detect repo root from {cwd}. "
        "Expected requirements.txt or .git in a parent directory."
    )


PROJECT_ROOT = find_project_root()

# Canonical data layout
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw" / "hapmap"
RAW_GENOTYPES_DIR = RAW_DIR / "genotypes"
RAW_PHASING_DIR = RAW_DIR / "phasing" / "HapMap3_r2" / "CEU" / "UNRELATED"
RAW_PHASING_META_DIR = RAW_DIR / "phasing" / "HapMap3_r2_meta"

PROC_DIR = DATA_DIR / "processed" / "hapmap"
PROC_REGIONS_DIR = PROC_DIR / "regions"
PROC_COHORTS_DIR = PROC_DIR / "cohorts"
PROC_BLOCKS_DIR = PROC_DIR / "blocks"
PROC_HAPLOTYPES_DIR = PROC_DIR / "haplotypes"

DERIVED_DIR = DATA_DIR / "derived"
DERIVED_METHOD1_DIR = DERIVED_DIR / "method1"
DERIVED_METHOD2_DIR = DERIVED_DIR / "method2"
DERIVED_METHOD3_DIR = DERIVED_DIR / "method3"


def ensure_dirs() -> None:
    """
    Safe directory creation (idempotent). Does not download or reprocess anything.
    """
    for d in [
        RAW_GENOTYPES_DIR,
        RAW_PHASING_DIR,
        RAW_PHASING_META_DIR,
        PROC_REGIONS_DIR,
        PROC_COHORTS_DIR,
        PROC_BLOCKS_DIR,
        PROC_HAPLOTYPES_DIR,
        DERIVED_METHOD1_DIR,
        DERIVED_METHOD2_DIR,
        DERIVED_METHOD3_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------
# File path helpers
# ----------------------------

def region_npz_path(region_name: str) -> Path:
    """
    region_name examples: "CEU_chr2_5Mb", "CEU_chr10_1Mb"
    """
    return PROC_REGIONS_DIR / f"{region_name}.npz"


def cohorts_json_path() -> Path:
    return PROC_COHORTS_DIR / "ceu_case_control_test_split.json"


def ceu_yri_cohorts_json_path() -> Path:
    return PROC_COHORTS_DIR / "hapmap_CEU_control_test__YRI_case.json"


def genotype_path(pop: str, chrom: int) -> Path:
    """
    pop: "CEU" | "YRI"
    chrom: chromosome number (e.g. 2, 10)
    """
    pop = pop.upper()
    if pop not in {"CEU", "YRI"}:
        raise ValueError("pop must be one of {'CEU','YRI'}")
    return RAW_GENOTYPES_DIR / f"genotypes_chr{chrom}_{pop}_r27_nr.b36_fwd.txt.gz"


def aligned_region_npz_path(pop: str, region_tag: str, other_pop: str) -> Path:
    """
    pop: "CEU" | "YRI"
    region_tag: e.g. "chr2_5Mb" or "chr10_1Mb"
    other_pop: population used for SNP intersection
    """
    pop = pop.upper()
    other_pop = other_pop.upper()
    return PROC_REGIONS_DIR / f"{pop}_{region_tag}.common_with_{other_pop}.npz"


def blocks_json_path(region_name: str) -> Path:
    return PROC_BLOCKS_DIR / f"{region_name}.blocks.json"


def control_haplotypes_path(region_name: str, phased_compatible: bool = True) -> Path:
    if phased_compatible:
        return PROC_HAPLOTYPES_DIR / f"{region_name}.control_haplotypes.phased_compatible.json"
    return PROC_HAPLOTYPES_DIR / f"{region_name}.control_haplotypes.json"


def maf_reference_path() -> Path:
    return PROC_DIR / "ceu_maf_reference.npz"


def method_dir(method: str, eps: float) -> Path:
    """
    method: "method1" | "method2" | "method3"
    eps folder naming: eps_0.1, eps_1, eps_10, etc.
    """
    if method not in {"method1", "method2", "method3"}:
        raise ValueError("method must be one of {'method1','method2','method3'}")

    # Keep folder names stable across OS and simple to read
    eps_str = str(eps).replace(".", "_")
    out = DERIVED_DIR / method / f"eps_{eps_str}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ----------------------------
# Loaders
# ----------------------------

def load_region_npz(path: Path) -> Dict[str, Any]:
    """
    Loads region .npz into a dict with friendly dtypes.
    Expected keys: G, sample_ids, snp_ids, positions,
    and allele labels (minor_alleles/major_alleles or counted_alleles/other_alleles), chrom (maybe)
    """
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files}

    # Normalize common keys to dtype object where appropriate
    for k in ["sample_ids", "snp_ids", "minor_alleles", "major_alleles", "counted_alleles", "other_alleles"]:
        if k in out:
            out[k] = out[k].astype(object)

    if "positions" in out:
        out["positions"] = out["positions"].astype(np.int64)

    if "G" in out:
        # Keep int8 to save memory (your pipeline uses -1 missing, 0/1/2 dosages)
        out["G"] = out["G"].astype(np.int8)

    return out


def load_cohorts(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads cohort split json.
    We expect:
      cohorts["case"]["sample_ids"], ["control"]["sample_ids"], ["test"]["sample_ids"]
    and usually:
      cohorts["case"]["indices_in_ceu_matrix"], etc (recommended for fast slicing).
    """
    if path is None:
        path = cohorts_json_path()
    with open(path, "r") as f:
        return json.load(f)


def get_group_indices(cohorts: Dict[str, Any], group: str) -> np.ndarray:
    """
    Returns indices_in_ceu_matrix for group in {"case","control","test"}.
    If indices are not present, raises with a clear error (so the user can regenerate cohorts properly).
    """
    if group not in {"case", "control", "test"}:
        raise ValueError("group must be one of {'case','control','test'}")

    key = "indices_in_ceu_matrix"
    if key not in cohorts[group]:
        raise KeyError(
            f"cohorts['{group}'] is missing '{key}'. "
            "Regenerate the cohort split with indices saved."
        )
    return np.array(cohorts[group][key], dtype=int)


# ----------------------------
# Quick checks (useful in notebooks)
# ----------------------------

def assert_exists(paths: Tuple[Path, ...], label: str = "FILE EXISTENCE CHECK") -> None:
    print(f"\n=== {label} ===")
    ok = True
    for p in paths:
        if p.exists():
            print(f"✅ {p.relative_to(PROJECT_ROOT)}")
        else:
            print(f"❌ {p.relative_to(PROJECT_ROOT)}")
            ok = False
    if not ok:
        raise FileNotFoundError("One or more required files are missing. See ❌ lines above.")


def print_layout_summary() -> None:
    """
    Prints where the project expects inputs/outputs.
    """
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("RAW_DIR:", RAW_DIR)
    print("PROC_DIR:", PROC_DIR)
    print("DERIVED_DIR:", DERIVED_DIR)
    print("Mechanisms notebooks folder (your preference):", PROJECT_ROOT / "src" / "mechanisms")
