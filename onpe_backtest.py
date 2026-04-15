from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import onpe_proyeccion as op


@dataclass
class EvalRow:
    snapshot_index: int
    snapshot_at: str
    actas_pct: float
    mae_topn: float
    max_err_topn: float
    top2_hit: bool
    top2_exact: bool


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backtest del modelo ONPE usando snapshots guardados")
    ap.add_argument("--state-file", type=str, default=str(Path(__file__).with_name("onpe_state_province.json")))
    ap.add_argument("--target-index", type=int, default=-1, help="snapshot objetivo contra el que medir; por defecto el ultimo")
    ap.add_argument("--start-index", type=int, default=0, help="primer snapshot a evaluar")
    ap.add_argument("--draws", type=int, default=1200)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--swing", type=str, default="0.03", help="uno o varios swings separados por coma")
    ap.add_argument("--late-bias-strength", type=str, default="0.65", help="uno o varios valores separados por coma")
    ap.add_argument("--late-bias-window", type=int, default=op.LATE_BIAS_DEFAULT_WINDOW)
    ap.add_argument("--late-bias-recency", type=str, default=str(op.LATE_BIAS_DEFAULT_RECENCY), help="uno o varios valores separados por coma")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-n", type=int, default=6, help="cuantos candidatos relevantes medir en el error")
    ap.add_argument("--min-actas-pct", type=float, default=0.0, help="ignora snapshots por debajo de este avance")
    return ap.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def load_snapshots(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    snapshots = data.get("snapshots", [])
    if not isinstance(snapshots, list) or not snapshots:
        raise RuntimeError(f"No hay snapshots en {path}")
    return snapshots


def regions_from_snapshot(snapshot: dict) -> list[op.Region]:
    rows = []
    for row in snapshot.get("regions", {}).values():
        votes_map = row.get("votos_por_candidato", {})
        cands = list(votes_map.keys())
        votos = np.array([int(votes_map[c]) for c in cands], dtype=np.int64)
        rows.append(
            op.Region(
                nombre=row.get("nombre", "?"),
                ubigeo=int(row.get("ubigeo", 0)),
                ubigeo_region=int(row.get("ubigeo_region", 0)),
                ambito=int(row.get("ambito", 1)),
                geo_level=row.get("geo_level", "region"),
                actas_contabilizadas=int(row.get("actas_contabilizadas", 0)),
                actas_totales=int(row.get("actas_totales", 0)),
                pct_actas=float(row.get("pct_actas", 0.0)),
                padron=int(row.get("padron", 0)),
                candidatos=cands,
                votos=votos,
                votos_validos=int(row.get("votos_validos", int(votos.sum()))),
                votos_emitidos=int(row.get("votos_emitidos", 0)),
                participacion_ciudadana=float(row.get("participacion_ciudadana", 0.0)),
                ref_validos_por_acta=float(row.get("ref_validos_por_acta", 0.0)),
                ref_emitidos_por_acta=float(row.get("ref_emitidos_por_acta", 0.0)),
                ref_valid_ratio=float(row.get("ref_valid_ratio", 0.0)),
            )
        )
    op._inyectar_referencias_turnout(rows)
    op._inyectar_referencias_validos_por_acta(rows)
    return rows


def candidate_pct_map(snapshot: dict) -> dict[str, float]:
    total = int(snapshot.get("votos_validos", 0))
    votes: dict[str, int] = {}
    for row in snapshot.get("regions", {}).values():
        for cand, value in row.get("votos_por_candidato", {}).items():
            votes[cand] = votes.get(cand, 0) + int(value)
    return {cand: op.pct(value, total) for cand, value in votes.items()}


def top_candidates(actual_map: dict[str, float], top_n: int) -> list[str]:
    return [
        cand
        for cand, _ in sorted(
            ((cand, pct) for cand, pct in actual_map.items() if "BLANCO" not in cand.upper() and "NULO" not in cand.upper()),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
    ]


def project_snapshot(
    snapshots: list[dict],
    eval_index: int,
    draws: int,
    alpha: float,
    swing: float,
    strength: float,
    window: int,
    recency: float,
    seed: int,
) -> dict[str, float]:
    current = snapshots[eval_index]
    regiones = regions_from_snapshot(current)
    late_bias_model = None
    if strength > 0:
        late_bias_model = op.build_late_bias_model(
            regiones=regiones,
            history_snapshots=snapshots[:eval_index],
            current_snapshot=current,
            window=max(window, 2),
            strength=op.clip01(strength),
            recency_decay=op.clip01(recency),
        )
    universe, _, _, pct_draws = op.proyectar(
        regiones,
        draws=draws,
        alpha_prior=alpha,
        swing=swing,
        late_bias_model=late_bias_model,
        rng=np.random.default_rng(seed + eval_index),
    )
    med = np.median(pct_draws, axis=0)
    return {cand: float(med[i]) for i, cand in enumerate(universe)}


def evaluate_combo(
    snapshots: list[dict],
    target_index: int,
    start_index: int,
    min_actas_pct: float,
    draws: int,
    alpha: float,
    swing: float,
    strength: float,
    window: int,
    recency: float,
    seed: int,
    top_n: int,
) -> tuple[list[EvalRow], dict]:
    target_snapshot = snapshots[target_index]
    target_map = candidate_pct_map(target_snapshot)
    relevant = top_candidates(target_map, top_n)
    target_top2 = relevant[:2]

    rows: list[EvalRow] = []
    for idx in range(max(start_index, 0), target_index):
        snap = snapshots[idx]
        actas_pct = op.pct(int(snap.get("actas_contadas", 0)), int(snap.get("actas_totales", 0)))
        if actas_pct < min_actas_pct:
            continue
        proj_map = project_snapshot(snapshots, idx, draws, alpha, swing, strength, window, recency, seed)
        errors = [abs(proj_map.get(cand, 0.0) - target_map.get(cand, 0.0)) for cand in relevant]
        proj_top2 = [
            cand
            for cand, _ in sorted(
                ((cand, proj_map.get(cand, 0.0)) for cand in proj_map if "BLANCO" not in cand.upper() and "NULO" not in cand.upper()),
                key=lambda item: item[1],
                reverse=True,
            )[:2]
        ]
        rows.append(
            EvalRow(
                snapshot_index=idx,
                snapshot_at=snap.get("captured_at", "?"),
                actas_pct=actas_pct,
                mae_topn=float(np.mean(errors)) if errors else 0.0,
                max_err_topn=float(np.max(errors)) if errors else 0.0,
                top2_hit=set(proj_top2) == set(target_top2),
                top2_exact=proj_top2 == target_top2,
            )
        )

    summary = {
        "swing": swing,
        "late_bias_strength": strength,
        "late_bias_recency": recency,
        "rows": rows,
        "mean_mae_topn": float(np.mean([row.mae_topn for row in rows])) if rows else float("inf"),
        "mean_max_err_topn": float(np.mean([row.max_err_topn for row in rows])) if rows else float("inf"),
        "top2_hit_rate": float(np.mean([1.0 if row.top2_hit else 0.0 for row in rows])) if rows else 0.0,
        "top2_exact_rate": float(np.mean([1.0 if row.top2_exact else 0.0 for row in rows])) if rows else 0.0,
        "target_snapshot_at": target_snapshot.get("captured_at"),
        "target_actas_pct": op.pct(int(target_snapshot.get("actas_contadas", 0)), int(target_snapshot.get("actas_totales", 0))),
        "target_top": [(cand, target_map[cand]) for cand in relevant[:4]],
    }
    return rows, summary


def print_combo_result(summary: dict) -> None:
    print(
        f"swing={summary['swing']:.3f}  "
        f"late_bias_strength={summary['late_bias_strength']:.2f}  "
        f"late_bias_recency={summary['late_bias_recency']:.2f}  "
        f"mae_topn={summary['mean_mae_topn']:.3f}pp  "
        f"max_err={summary['mean_max_err_topn']:.3f}pp  "
        f"top2_hit={summary['top2_hit_rate']*100:.1f}%  "
        f"top2_exact={summary['top2_exact_rate']*100:.1f}%"
    )


def print_best_run(rows: list[EvalRow], summary: dict) -> None:
    print()
    print("Mejor combinacion encontrada")
    print("-" * 96)
    print_combo_result(summary)
    print(f"Target: {summary['target_snapshot_at']}  ({summary['target_actas_pct']:.2f}% actas)")
    print("Top target:", ", ".join(f"{cand} {pct:.2f}%" for cand, pct in summary["target_top"]))
    print()
    print(f"{'Idx':>4} {'Actas':>8} {'MAE topN':>10} {'Max err':>10} {'Top2 set':>9} {'Top2 orden':>11}  Snapshot")
    print("-" * 96)
    for row in rows:
        print(
            f"{row.snapshot_index:>4} {row.actas_pct:>7.2f}% {row.mae_topn:>9.3f} {row.max_err_topn:>9.3f} "
            f"{('si' if row.top2_hit else 'no'):>9} {('si' if row.top2_exact else 'no'):>11}  {row.snapshot_at}"
        )


def main() -> None:
    args = parse_args()
    snapshots = load_snapshots(Path(args.state_file))
    target_index = args.target_index if args.target_index >= 0 else len(snapshots) + args.target_index
    if target_index <= 0 or target_index >= len(snapshots):
        raise RuntimeError("target-index invalido para la cantidad de snapshots disponible")

    best_summary = None
    best_rows: list[EvalRow] = []
    for swing in parse_float_list(args.swing):
        for strength in parse_float_list(args.late_bias_strength):
            for recency in parse_float_list(args.late_bias_recency):
                rows, summary = evaluate_combo(
                    snapshots=snapshots,
                    target_index=target_index,
                    start_index=args.start_index,
                    min_actas_pct=args.min_actas_pct,
                    draws=args.draws,
                    alpha=args.alpha,
                    swing=swing,
                    strength=strength,
                    window=args.late_bias_window,
                    recency=recency,
                    seed=args.seed,
                    top_n=max(args.top_n, 2),
                )
                print_combo_result(summary)
                if best_summary is None or summary["mean_mae_topn"] < best_summary["mean_mae_topn"]:
                    best_summary = summary
                    best_rows = rows

    if best_summary is None:
        raise RuntimeError("No se pudo evaluar ninguna combinacion")
    print_best_run(best_rows, best_summary)


if __name__ == "__main__":
    main()
