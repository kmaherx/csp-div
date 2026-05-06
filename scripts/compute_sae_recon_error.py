"""SAE reconstruction error per (seed, ckpt) — an "on-manifold" score.

Reads shifts.pt (which holds per-row shift vectors and a single mean_vanilla
baseline), reconstructs mean_csp = mean_vanilla + shift, encodes/decodes
through the configured SAE, and writes a per-row error JSON that the 3D
plot can use as a continuous colormap.

Layer caveat
------------
The configured SAE (`andyrdt/llama-3.1-8b-instruct-resid_post_layer_15`) is
trained at L15 resid_post. Our shifts.pt was extracted at L16 (post-block).
That's roughly one transformer block of slippage between training and
application — reconstruction error is therefore inflated relative to a
strictly-matched run, but the *relative ordering* across (seed, ckpt) pairs
should still be informative. Use --activations-pt to swap in a separately
extracted L15 shifts.pt later if needed.

Usage
-----
  python scripts/compute_sae_recon_error.py
  python scripts/compute_sae_recon_error.py \\
      --shifts-paths results/llama_act/shifts.pt \\
      --out          results/llama_act/sae_recon_error.json
"""
import argparse
import json
import os

import torch

from csp_div import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_sae(release, sae_id, device):
    from sae_lens import SAE
    print(f"Loading SAE: release={release!r}, sae_id={sae_id!r}")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id)
    if isinstance(sae, tuple):
        sae = sae[0]  # newer sae_lens returns (sae, cfg, sparsity)
    sae = sae.to(device)
    sae.eval()
    return sae


def reconstruction_error(sae, x):
    """L2 reconstruction error for a batch of activations.

    Returns (abs_err, rel_err) tensors of shape (batch,).
      abs_err = ||x - SAE(x)||_2
      rel_err = ||x - SAE(x)||_2 / ||x||_2
    """
    with torch.no_grad():
        feats = sae.encode(x)
        x_hat = sae.decode(feats)
    diff = (x - x_hat).float()
    abs_err = diff.norm(dim=-1)
    x_norm = x.float().norm(dim=-1).clamp_min(1e-8)
    rel_err = abs_err / x_norm
    return abs_err, rel_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shifts-paths", nargs="+",
                        default=["results/llama/shifts.pt"],
                        help="One or more shifts.pt files. Each row's shift "
                             "is added to that file's mean_vanilla to recover "
                             "mean_csp before SAE encoding.")
    parser.add_argument("--sae-release", default=config.SAE_RELEASE)
    parser.add_argument("--sae-id",      default=config.SAE_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", default=None,
                        help="Output JSON path. Default: sibling of the first "
                             "shifts.pt as 'sae_recon_error.json'.")
    args = parser.parse_args()

    if args.sae_release is None or args.sae_id is None:
        raise SystemExit("config.SAE_RELEASE / SAE_ID are unset and no "
                         "--sae-release / --sae-id were provided.")

    available = [p if os.path.isabs(p) else os.path.join(ROOT, p)
                 for p in args.shifts_paths]
    available = [p for p in available if os.path.isfile(p)]
    if not available:
        raise SystemExit(f"No shifts.pt files found in {args.shifts_paths}")

    sae = load_sae(args.sae_release, args.sae_id, args.device)
    print(f"  SAE input dim: {sae.cfg.d_in}  hook: {sae.cfg.hook_name}")

    rows_out = []
    for shifts_path in available:
        print(f"\nProcessing {shifts_path}")
        d = torch.load(shifts_path, map_location="cpu", weights_only=True)
        layer = d.get("layer")
        rows = d["rows"]
        mean_vanilla = d["mean_vanilla"].float().to(args.device)  # (hidden,)
        if mean_vanilla.shape[0] != sae.cfg.d_in:
            raise SystemExit(
                f"shape mismatch: shifts.pt hidden={mean_vanilla.shape[0]} "
                f"vs SAE d_in={sae.cfg.d_in}"
            )

        # Stack mean_csp = mean_vanilla + shift for all rows.
        shifts = torch.stack([r["shift"].float() for r in rows]).to(args.device)
        mean_csp = mean_vanilla.unsqueeze(0) + shifts  # (n_rows, hidden)

        # Batch through SAE
        all_abs, all_rel = [], []
        for i in range(0, mean_csp.shape[0], args.batch_size):
            x = mean_csp[i:i + args.batch_size]
            abs_err, rel_err = reconstruction_error(sae, x)
            all_abs.append(abs_err.cpu())
            all_rel.append(rel_err.cpu())
        all_abs = torch.cat(all_abs).tolist()
        all_rel = torch.cat(all_rel).tolist()

        for r, ae, re in zip(rows, all_abs, all_rel):
            rows_out.append({
                "shifts_path": os.path.relpath(shifts_path, ROOT),
                "group": r["group"],
                "ckpt": r["ckpt"],
                "step": r["step"],
                "kl": r["kl"],
                "activation_layer": layer,
                "abs_err": ae,
                "rel_err": re,
            })

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.dirname(available[0]), "sae_recon_error.json")
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    summary = {
        "method": "sae_reconstruction_error",
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "sae_d_in": sae.cfg.d_in,
        "sae_hook_name": sae.cfg.hook_name,
        "activation_layer_in_shifts": rows_out[0]["activation_layer"]
            if rows_out else None,
        "layer_mismatch_note": (
            "SAE trained on L15 resid_post; shifts.pt extracted at L16 "
            "post-block. Reconstruction error is inflated by ~one "
            "transformer block of basis drift; relative ordering across "
            "rows is still informative."
        ),
        "n_rows": len(rows_out),
        "abs_err_stats": {
            "min": min(r["abs_err"] for r in rows_out),
            "max": max(r["abs_err"] for r in rows_out),
            "mean": sum(r["abs_err"] for r in rows_out) / len(rows_out),
        },
        "rel_err_stats": {
            "min": min(r["rel_err"] for r in rows_out),
            "max": max(r["rel_err"] for r in rows_out),
            "mean": sum(r["rel_err"] for r in rows_out) / len(rows_out),
        },
        "rows": rows_out,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"  abs_err: min={summary['abs_err_stats']['min']:.3f}  "
          f"mean={summary['abs_err_stats']['mean']:.3f}  "
          f"max={summary['abs_err_stats']['max']:.3f}")
    print(f"  rel_err: min={summary['rel_err_stats']['min']:.3f}  "
          f"mean={summary['rel_err_stats']['mean']:.3f}  "
          f"max={summary['rel_err_stats']['max']:.3f}")


if __name__ == "__main__":
    main()
