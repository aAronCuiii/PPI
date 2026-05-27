"""
Functional annotation pipeline for PPI module detection results.

Methods: autoencoder_cosine, intensity_rbf, combined_pearson_jaccard
Strategy: top10perNodes
Targets: top 5 Leiden modules + top 5 MCODE cores per method (30 gene sets total)

Outputs (all in outputs/network/annotation/):
  selected_annotation_targets.csv
  corum_enrichment_selected.csv
  gprofiler_enrichment_selected.csv
  annotation_summary_selected.csv
  method_annotation_comparison.csv
  leiden_vs_mcode_comparison.csv
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULES_ROOT = os.path.join(PROJECT_ROOT, "outputs", "network", "modules")
CORUM_PATH   = os.path.join(PROJECT_ROOT, "data", "raw", "CORUM_data.txt")
OUT_DIR      = os.path.join(PROJECT_ROOT, "outputs", "network", "annotation")
os.makedirs(OUT_DIR, exist_ok=True)

METHODS   = ["autoencoder_cosine", "intensity_rbf", "combined_pearson_jaccard"]
STRATEGY  = "top10perNodes"
TOP_N     = 5
FDR_CUTOFF = 0.05

# ---------------------------------------------------------------------------
# 1. Load CORUM complexes (human, gene symbols)
# ---------------------------------------------------------------------------
print("Loading CORUM complexes...")
corum_raw = pd.read_csv(CORUM_PATH, sep="\t")
corum_raw = corum_raw[corum_raw["Organism"] == "Human"].reset_index(drop=True)

corum_db = {}
for _, row in corum_raw.iterrows():
    genes_str = str(row["subunits.Gene.name."]).strip()
    if not genes_str or genes_str == "nan":
        continue
    genes = set(g.strip() for g in genes_str.split(";") if g.strip())
    if len(genes) >= 2:
        corum_db[row["ComplexID"]] = {
            "name": row["ComplexName"],
            "genes": genes,
            "size": len(genes),
        }

print(f"  Loaded {len(corum_db)} CORUM complexes (human, >=2 subunits)")

# ---------------------------------------------------------------------------
# 2. Select annotation targets
# ---------------------------------------------------------------------------
print("Selecting annotation targets...")

records = []
for method in METHODS:
    base = os.path.join(MODULES_ROOT, method, STRATEGY)

    # Background: all proteins in the graph
    assignments = pd.read_csv(os.path.join(base, "module_assignments.csv"))
    background  = set(assignments["protein"].astype(str).str.strip())

    # Top 5 Leiden modules by size
    mod_summary = pd.read_csv(os.path.join(base, "module_summary.csv"))
    top_leiden  = mod_summary.sort_values("size", ascending=False).head(TOP_N)

    for rank, (_, row) in enumerate(top_leiden.iterrows(), 1):
        module_proteins = set(
            assignments[assignments["module_id"] == row["module_id"]]["protein"]
            .astype(str).str.strip()
        )
        records.append({
            "method":        method,
            "strategy":      STRATEGY,
            "cluster_type":  "leiden",
            "cluster_id":    row["module_id"],
            "rank":          rank,
            "size":          row["size"],
            "key_proteins":  row["key_proteins"],
            "proteins":      ";".join(sorted(module_proteins)),
            "n_background":  len(background),
            "background":    ";".join(sorted(background)),
        })

    # Top 5 MCODE cores by mcode_score
    cores   = pd.read_csv(os.path.join(base, "mcode_cores.csv"))
    members = pd.read_csv(os.path.join(base, "mcode_core_members.csv"))
    top_mc  = cores.sort_values("mcode_score", ascending=False).head(TOP_N)

    for rank, (_, row) in enumerate(top_mc.iterrows(), 1):
        core_proteins = set(
            members[members["core_id"] == row["core_id"]]["protein"]
            .astype(str).str.strip()
        )
        records.append({
            "method":        method,
            "strategy":      STRATEGY,
            "cluster_type":  "mcode",
            "cluster_id":    row["core_id"],
            "rank":          rank,
            "size":          row["size"],
            "key_proteins":  row["key_proteins"],
            "proteins":      ";".join(sorted(core_proteins)),
            "n_background":  len(background),
            "background":    ";".join(sorted(background)),
        })

targets_df = pd.DataFrame(records)
targets_df.to_csv(os.path.join(OUT_DIR, "selected_annotation_targets.csv"), index=False)
print(f"  {len(targets_df)} annotation targets saved")

# ---------------------------------------------------------------------------
# Helper: parse protein list from target row
# ---------------------------------------------------------------------------
def get_proteins(row):
    return [p.strip() for p in str(row["proteins"]).split(";") if p.strip()]

def get_background(row):
    return set(p.strip() for p in str(row["background"]).split(";") if p.strip())

# ---------------------------------------------------------------------------
# 3. CORUM hypergeometric enrichment
# ---------------------------------------------------------------------------
print("Running CORUM hypergeometric enrichment...")

corum_rows = []
for _, tgt in targets_df.iterrows():
    query_genes = set(get_proteins(tgt))
    bg_genes    = get_background(tgt)
    M = len(bg_genes)

    for cid, cinfo in corum_db.items():
        complex_in_bg = cinfo["genes"] & bg_genes
        n = len(complex_in_bg)
        if n < 2:
            continue

        overlap = query_genes & complex_in_bg
        k = len(overlap)
        N = len(query_genes)

        if k == 0:
            continue

        pval = hypergeom.sf(k - 1, M, n, N)

        corum_rows.append({
            "method":        tgt["method"],
            "strategy":      tgt["strategy"],
            "cluster_type":  tgt["cluster_type"],
            "cluster_id":    tgt["cluster_id"],
            "rank":          tgt["rank"],
            "complex_id":    cid,
            "complex_name":  cinfo["name"],
            "complex_size":  cinfo["size"],
            "complex_in_bg": n,
            "query_size":    N,
            "overlap":       k,
            "pval":          pval,
            "overlap_genes": ";".join(sorted(overlap)),
        })

corum_df = pd.DataFrame(corum_rows)
if len(corum_df) > 0:
    # BH FDR correction within each cluster
    fdr_vals = []
    for _, grp in corum_df.groupby(["method", "cluster_type", "cluster_id"]):
        _, fdr, _, _ = multipletests(grp["pval"].values, method="fdr_bh")
        fdr_vals.extend(fdr.tolist())
    corum_df["fdr"] = fdr_vals
    corum_df = corum_df.sort_values(["method", "cluster_type", "cluster_id", "pval"])
    corum_df.to_csv(os.path.join(OUT_DIR, "corum_enrichment_selected.csv"), index=False)
    sig = (corum_df["fdr"] < FDR_CUTOFF).sum()
    print(f"  {len(corum_df)} CORUM tests, {sig} significant (FDR<{FDR_CUTOFF})")
else:
    corum_df.to_csv(os.path.join(OUT_DIR, "corum_enrichment_selected.csv"), index=False)
    print("  No CORUM overlaps found")

# ---------------------------------------------------------------------------
# 4. g:Profiler enrichment
# ---------------------------------------------------------------------------
print("Running g:Profiler enrichment...")

try:
    from gprofiler import GProfiler
    gp = GProfiler(return_dataframe=True)
    gp_available = True
except ImportError:
    print("  WARNING: gprofiler-official not installed. Skipping g:Profiler.")
    gp_available = False

gp_rows = []
if gp_available:
    SOURCES = ["GO:BP", "GO:CC", "GO:MF", "REAC", "KEGG", "WP", "CORUM"]
    for idx, tgt in targets_df.iterrows():
        query_genes = get_proteins(tgt)
        bg_genes    = list(get_background(tgt))
        label = f"{tgt['method']} / {tgt['cluster_type']} {tgt['cluster_id']}"
        print(f"  g:Profiler [{idx+1}/{len(targets_df)}]: {label} ({len(query_genes)} proteins)")

        try:
            result = gp.profile(
                organism="hsapiens",
                query=query_genes,
                background=bg_genes,
                sources=SOURCES,
                significance_threshold_method="fdr",
                user_threshold=FDR_CUTOFF,
                no_evidences=False,
            )
            if len(result) > 0:
                result = result.copy()
                result["method"]       = tgt["method"]
                result["strategy"]     = tgt["strategy"]
                result["cluster_type"] = tgt["cluster_type"]
                result["cluster_id"]   = tgt["cluster_id"]
                result["rank"]         = tgt["rank"]
                gp_rows.append(result)
        except Exception as e:
            print(f"    ERROR: {e}")

if gp_rows:
    gp_df = pd.concat(gp_rows, ignore_index=True)
    # Keep relevant columns
    keep_cols = ["method", "strategy", "cluster_type", "cluster_id", "rank",
                 "source", "native", "name", "p_value", "significant",
                 "term_size", "query_size", "intersection_size", "recall",
                 "precision", "intersections"]
    keep_cols = [c for c in keep_cols if c in gp_df.columns]
    gp_df = gp_df[keep_cols].sort_values(
        ["method", "cluster_type", "cluster_id", "p_value"]
    )
    gp_df.to_csv(os.path.join(OUT_DIR, "gprofiler_enrichment_selected.csv"), index=False)
    print(f"  {len(gp_df)} g:Profiler terms (FDR<{FDR_CUTOFF})")
else:
    pd.DataFrame().to_csv(os.path.join(OUT_DIR, "gprofiler_enrichment_selected.csv"), index=False)
    print("  No significant g:Profiler terms")
    gp_df = pd.DataFrame()

# ---------------------------------------------------------------------------
# 5. Annotation summary table
# ---------------------------------------------------------------------------
print("Building annotation summary table...")

summary_rows = []
for _, tgt in targets_df.iterrows():
    row_dict = {
        "method":        tgt["method"],
        "strategy":      tgt["strategy"],
        "cluster_type":  tgt["cluster_type"],
        "cluster_id":    tgt["cluster_id"],
        "rank":          tgt["rank"],
        "size":          tgt["size"],
        "key_proteins":  tgt["key_proteins"],
    }

    # CORUM hits
    c_hits = corum_df[
        (corum_df["method"]       == tgt["method"]) &
        (corum_df["cluster_type"] == tgt["cluster_type"]) &
        (corum_df["cluster_id"]   == tgt["cluster_id"]) &
        (corum_df["fdr"]          < FDR_CUTOFF)
    ] if len(corum_df) > 0 else pd.DataFrame()

    row_dict["n_corum_hits"]      = len(c_hits)
    row_dict["top_corum_complex"] = c_hits["complex_name"].iloc[0] if len(c_hits) > 0 else ""
    row_dict["top_corum_pval"]    = c_hits["pval"].iloc[0]         if len(c_hits) > 0 else np.nan
    row_dict["top_corum_fdr"]     = c_hits["fdr"].iloc[0]          if len(c_hits) > 0 else np.nan
    row_dict["top_corum_overlap"] = c_hits["overlap"].iloc[0]      if len(c_hits) > 0 else 0

    # g:Profiler hits per source
    if len(gp_df) > 0:
        gp_hits = gp_df[
            (gp_df["method"]       == tgt["method"]) &
            (gp_df["cluster_type"] == tgt["cluster_type"]) &
            (gp_df["cluster_id"]   == tgt["cluster_id"])
        ]
        for src in ["GO:BP", "GO:CC", "GO:MF", "REAC", "KEGG", "WP", "CORUM"]:
            src_hits = gp_hits[gp_hits["source"] == src] if "source" in gp_hits.columns else pd.DataFrame()
            src_key  = src.lower().replace(":", "_")
            row_dict[f"n_{src_key}"]   = len(src_hits)
            row_dict[f"top_{src_key}"] = src_hits["name"].iloc[0] if len(src_hits) > 0 else ""
    else:
        for src in ["GO:BP", "GO:CC", "GO:MF", "REAC", "KEGG", "WP", "CORUM"]:
            src_key = src.lower().replace(":", "_")
            row_dict[f"n_{src_key}"]   = 0
            row_dict[f"top_{src_key}"] = ""

    summary_rows.append(row_dict)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "annotation_summary_selected.csv"), index=False)
print(f"  Annotation summary saved ({len(summary_df)} rows)")

# ---------------------------------------------------------------------------
# 6. Method comparison table
# ---------------------------------------------------------------------------
print("Building method comparison table...")

method_rows = []
for method in METHODS:
    sub = summary_df[summary_df["method"] == method]
    leiden_sub = sub[sub["cluster_type"] == "leiden"]
    mcode_sub  = sub[sub["cluster_type"] == "mcode"]

    def safe_mean(s):
        return round(s.mean(), 2) if len(s) > 0 else np.nan

    def safe_sum(s):
        return int(s.sum()) if len(s) > 0 else 0

    method_rows.append({
        "method":                    method,
        "leiden_n_corum_total":      safe_sum(leiden_sub["n_corum_hits"]),
        "leiden_pct_with_corum":     round(100 * (leiden_sub["n_corum_hits"] > 0).mean(), 1) if len(leiden_sub) > 0 else 0,
        "leiden_mean_gobp":          safe_mean(leiden_sub["n_go_bp"]),
        "leiden_mean_reac":          safe_mean(leiden_sub["n_reac"]),
        "leiden_mean_kegg":          safe_mean(leiden_sub["n_kegg"]),
        "mcode_n_corum_total":       safe_sum(mcode_sub["n_corum_hits"]),
        "mcode_pct_with_corum":      round(100 * (mcode_sub["n_corum_hits"] > 0).mean(), 1) if len(mcode_sub) > 0 else 0,
        "mcode_mean_gobp":           safe_mean(mcode_sub["n_go_bp"]),
        "mcode_mean_reac":           safe_mean(mcode_sub["n_reac"]),
        "mcode_mean_kegg":           safe_mean(mcode_sub["n_kegg"]),
        "total_sig_terms":           safe_sum(sub[[c for c in sub.columns if c.startswith("n_") and c not in ["n_corum_hits"]]].sum(axis=1)),
    })

method_cmp = pd.DataFrame(method_rows)
method_cmp.to_csv(os.path.join(OUT_DIR, "method_annotation_comparison.csv"), index=False)
print(f"  Method comparison saved ({len(method_cmp)} rows)")

# ---------------------------------------------------------------------------
# 7. Leiden vs MCODE comparison table
# ---------------------------------------------------------------------------
print("Building Leiden vs MCODE comparison table...")

compare_rows = []
for method in METHODS:
    for cluster_type in ["leiden", "mcode"]:
        sub = summary_df[
            (summary_df["method"]       == method) &
            (summary_df["cluster_type"] == cluster_type)
        ]
        if len(sub) == 0:
            continue

        n_cols = [c for c in sub.columns if c.startswith("n_") and c != "n_corum_hits"]
        total_terms = sub[n_cols].sum().sum() if len(n_cols) > 0 else 0

        compare_rows.append({
            "method":              method,
            "cluster_type":        cluster_type,
            "n_clusters":          len(sub),
            "mean_size":           round(sub["size"].mean(), 1),
            "n_corum_total":       int(sub["n_corum_hits"].sum()),
            "pct_with_any_corum":  round(100 * (sub["n_corum_hits"] > 0).mean(), 1),
            "total_gp_terms":      int(total_terms),
            "mean_gp_per_cluster": round(total_terms / len(sub), 1) if len(sub) > 0 else 0,
            "mean_gobp":           round(sub["n_go_bp"].mean(), 1) if "n_go_bp" in sub.columns else 0,
            "mean_reac":           round(sub["n_reac"].mean(), 1) if "n_reac" in sub.columns else 0,
        })

lm_cmp = pd.DataFrame(compare_rows)
lm_cmp.to_csv(os.path.join(OUT_DIR, "leiden_vs_mcode_comparison.csv"), index=False)
print(f"  Leiden vs MCODE comparison saved ({len(lm_cmp)} rows)")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n=== Annotation pipeline complete ===")
print(f"Output directory: {OUT_DIR}")
for fname in [
    "corum_enrichment_selected.csv",
    "gprofiler_enrichment_selected.csv",
    "annotation_summary_selected.csv",
    "method_annotation_comparison.csv",
    "leiden_vs_mcode_comparison.csv",
]:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        print(f"  {fname}: {len(df)} rows")
    else:
        print(f"  {fname}: MISSING")
