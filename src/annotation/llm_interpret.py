"""
LLM interpretation of g:Profiler enrichment results via OpenAI API.

Usage:
  python src/annotation/llm_interpret.py

Reads:  outputs/network/annotation/gprofiler_enrichment_selected.csv
        outputs/network/annotation/corum_enrichment_selected.csv
        outputs/network/annotation/selected_annotation_targets.csv
Writes: outputs/network/annotation/llm_interpretations_openai.csv
"""

import os
import ast
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys from project root .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANN_DIR      = os.path.join(PROJECT_ROOT, "outputs", "network", "annotation")

MODEL        = "gpt-4o"
TEMPERATURE  = 0.2
TOP_N_TERMS  = 5   # top terms per source included in prompt
FDR_CUTOFF   = 0.05
SLEEP_S      = 1.0 # seconds between API calls

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_intersections(value) -> list[str]:
    """
    Parse g:Profiler intersection fields from CSV without executing code.
    """
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [
            item.strip()
            for item in text.replace(";", ",").split(",")
            if item.strip()
        ]

    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]

    parsed_text = str(parsed).strip()
    return [parsed_text] if parsed_text else []

# ---------------------------------------------------------------------------
# 1.  Build per-cluster input payload
# ---------------------------------------------------------------------------

def build_cluster_input(
    row_target: pd.Series,
    gp_df: pd.DataFrame,
    corum_df: pd.DataFrame,
    top_n: int = TOP_N_TERMS,
) -> dict:
    """
    Compress one cluster's enrichment results into a structured dict
    suitable for inclusion in a prompt. Keeps the top_n terms per source
    (by p-value) and the top CORUM complex hits.
    """
    method       = row_target["method"]
    cluster_type = row_target["cluster_type"]
    cluster_id   = row_target["cluster_id"]

    # Filter g:Profiler results for this cluster
    mask = (
        (gp_df["method"]       == method) &
        (gp_df["cluster_type"] == cluster_type) &
        (gp_df["cluster_id"]   == cluster_id)
    )
    gp_cluster = gp_df[mask].copy()

    # Top N terms per source
    sources_out = {}
    for src, grp in gp_cluster.groupby("source"):
        top = (
            grp.sort_values("p_value")
               .head(top_n)[["name", "p_value", "term_size", "intersection_size", "intersections"]]
               .copy()
        )
        top["p_value"] = top["p_value"].apply(lambda x: float(f"{x:.2e}"))
        # intersections is stored as a string repr of a list in CSV exports.
        top["hit_genes"] = top["intersections"].apply(parse_intersections)
        top = top.drop(columns=["intersections"])
        sources_out[src] = top.to_dict(orient="records")

    # Top CORUM hits (FDR < cutoff)
    if len(corum_df) > 0:
        cmask = (
            (corum_df["method"]       == method) &
            (corum_df["cluster_type"] == cluster_type) &
            (corum_df["cluster_id"]   == cluster_id) &
            (corum_df["fdr"]          < FDR_CUTOFF)
        )
        c_hits = (
            corum_df[cmask]
            .sort_values("pval")
            .head(top_n)[["complex_name", "overlap", "pval", "fdr", "overlap_genes"]]
            .copy()
        )
        c_hits["pval"] = c_hits["pval"].apply(lambda x: float(f"{x:.2e}"))
        c_hits["fdr"]  = c_hits["fdr"].apply(lambda x:  float(f"{x:.2e}"))
        corum_hits = c_hits.to_dict(orient="records")
    else:
        corum_hits = []

    return {
        "method":        method,
        "strategy":      row_target["strategy"],
        "cluster_type":  cluster_type,
        "cluster_id":    cluster_id,
        "rank":          int(row_target["rank"]),
        "size":          int(row_target["size"]),
        "key_proteins":  row_target["key_proteins"].split(";"),
        "n_background":  int(row_target["n_background"]),
        "enrichment": {
            "gprofiler": sources_out,
            "corum":     corum_hits,
        },
    }


# ---------------------------------------------------------------------------
# 2.  Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a computational proteomics expert specializing in protein-protein \
interaction (PPI) network analysis. You receive enrichment results for a \
cluster of co-expressed proteins from a patient-level proteomics dataset and \
produce a concise, information-dense biological interpretation.

Your interpretation must:
1. State the most likely biological identity of the cluster (complex, pathway, \
   or functional module) in ≤1 sentence.
2. List the key evidence (top CORUM complex match, top GO:CC term, top pathway).
3. Identify any biologically surprising or contradictory signals (e.g. multiple \
   unrelated pathways, contamination from a generic term like "metabolic process").
4. Assign a confidence label: High / Moderate / Low, with one-line justification.
5. Suggest one specific downstream hypothesis or validation experiment relevant \
   to a PPI network study.

Respond in the following JSON schema — no extra text:
{
  "biological_identity": "<string>",
  "key_evidence": ["<string>", ...],
  "surprising_signals": "<string or null>",
  "confidence": "High | Moderate | Low",
  "confidence_reason": "<string>",
  "hypothesis": "<string>"
}"""


def build_user_message(payload: dict) -> str:
    """Render the cluster input payload as a structured user message."""
    lines = []
    lines.append(f"## Cluster: {payload['method']} / {payload['cluster_type'].upper()} {payload['cluster_id']}")
    lines.append(f"- **Rank:** {payload['rank']}  |  **Size:** {payload['size']} proteins  |  **Background:** {payload['n_background']} proteins")
    lines.append(f"- **Key proteins:** {', '.join(payload['key_proteins'][:10])}")
    lines.append("")

    enr = payload["enrichment"]

    # CORUM
    if enr["corum"]:
        lines.append("### CORUM complex matches (hypergeometric, BH FDR<0.05)")
        for h in enr["corum"]:
            genes = h['overlap_genes'] if isinstance(h['overlap_genes'], list) else h['overlap_genes'].split(";")
            lines.append(
                f"- **{h['complex_name']}** | overlap={h['overlap']} | "
                f"p={h['pval']:.2e} | FDR={h['fdr']:.2e} | "
                f"genes: {', '.join(genes[:8])}"
            )
        lines.append("")

    # g:Profiler per source
    source_order = ["GO:CC", "GO:BP", "GO:MF", "REAC", "KEGG", "WP"]
    for src in source_order:
        terms = enr["gprofiler"].get(src, [])
        if not terms:
            continue
        lines.append(f"### {src} (top {len(terms)} terms)")
        for t in terms:
            genes = t.get("hit_genes", [])
            gene_str = ", ".join(genes[:6]) if isinstance(genes, list) else str(genes)[:60]
            lines.append(
                f"- **{t['name']}** | p={t['p_value']:.2e} | "
                f"term_size={t['term_size']} | hits={t['intersection_size']} | "
                f"genes: {gene_str}"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3.  OpenAI API call
# ---------------------------------------------------------------------------

def call_openai(client: OpenAI, user_msg: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


# ---------------------------------------------------------------------------
# 4.  Main loop
# ---------------------------------------------------------------------------

def main():
    client = OpenAI()  # reads OPENAI_API_KEY from env

    gp_df    = pd.read_csv(os.path.join(ANN_DIR, "gprofiler_enrichment_selected.csv"))
    corum_df = pd.read_csv(os.path.join(ANN_DIR, "corum_enrichment_selected.csv"))
    tgt_df   = pd.read_csv(os.path.join(ANN_DIR, "selected_annotation_targets.csv"))

    results = []
    for idx, row in tgt_df.iterrows():
        label = f"{row['method']} / {row['cluster_type']} {row['cluster_id']}"
        print(f"[{idx+1}/{len(tgt_df)}] {label}")

        payload  = build_cluster_input(row, gp_df, corum_df)
        user_msg = build_user_message(payload)

        # --- dry-run: print first message and exit ---
        if idx == 0:
            print("\n--- EXAMPLE USER MESSAGE (first cluster) ---")
            print(user_msg)
            print("--- END ---\n")

        try:
            result = call_openai(client, user_msg)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "biological_identity": "ERROR",
                "key_evidence": [],
                "surprising_signals": str(e),
                "confidence": "Low",
                "confidence_reason": "API error",
                "hypothesis": "",
            }

        results.append({
            "method":               row["method"],
            "strategy":             row["strategy"],
            "cluster_type":         row["cluster_type"],
            "cluster_id":           row["cluster_id"],
            "rank":                 row["rank"],
            "size":                 row["size"],
            "key_proteins":         row["key_proteins"],
            "biological_identity":  result.get("biological_identity", ""),
            "key_evidence":         " | ".join(result.get("key_evidence", [])),
            "surprising_signals":   result.get("surprising_signals", ""),
            "confidence":           result.get("confidence", ""),
            "confidence_reason":    result.get("confidence_reason", ""),
            "hypothesis":           result.get("hypothesis", ""),
        })

        time.sleep(SLEEP_S)

    out_path = os.path.join(ANN_DIR, "llm_interpretations_openai.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} interpretations → {out_path}")


if __name__ == "__main__":
    main()
