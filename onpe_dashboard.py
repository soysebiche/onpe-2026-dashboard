from __future__ import annotations

import argparse
import html
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import onpe_proyeccion as op

DEPARTMENT_NAMES = {
    0: "EXTRANJERO",
    10000: "AMAZONAS",
    20000: "ANCASH",
    30000: "APURIMAC",
    40000: "AREQUIPA",
    50000: "AYACUCHO",
    60000: "CAJAMARCA",
    70000: "CALLAO",
    80000: "CUSCO",
    90000: "HUANCAVELICA",
    100000: "HUANUCO",
    110000: "ICA",
    120000: "JUNIN",
    130000: "LA LIBERTAD",
    140000: "LIMA",
    150000: "LAMBAYEQUE",
    160000: "LORETO",
    170000: "MADRE DE DIOS",
    180000: "MOQUEGUA",
    190000: "PASCO",
    200000: "PIURA",
    210000: "PUNO",
    220000: "SAN MARTIN",
    230000: "TACNA",
    240000: "TUMBES",
    250000: "UCAYALI",
}

BLOCK_ORDER = ("regiones", "lima", "extranjero")
BLOCK_LABELS = {
    "regiones": "Regiones sin Lima",
    "lima": "Lima",
    "extranjero": "Extranjero",
}
TRACK_COLORS = {
    "JUNTOS POR EL PERÚ": "#1f7a45",
    "RENOVACIÓN POPULAR": "#1e40af",
    "PARTIDO DEL BUEN GOBIERNO": "#d97706",
    "FUERZA POPULAR": "#6b7280",
}
TRACK_SHORT_LABELS = {
    "JUNTOS POR EL PERÚ": "JPP",
    "RENOVACIÓN POPULAR": "RP",
    "PARTIDO DEL BUEN GOBIERNO": "PBG",
    "FUERZA POPULAR": "FP",
}

# Cartogram grid: (col, row) → ubigeo department code
# 6 cols × 8 rows, geographic arrangement N→S, W→E
DEPT_GRID: dict[tuple[int, int], int] = {
    (0, 0): 240000, (1, 0): 200000, (2, 0): 150000,                   (5, 0): 160000,
                    (1, 1):  60000, (2, 1):  10000,
    (0, 2): 130000,                 (2, 2): 220000,
                    (1, 3):  20000, (2, 3): 100000, (3, 3): 250000,
    (0, 4):  70000, (1, 4): 140000, (2, 4): 190000, (3, 4): 120000,
                    (1, 5): 110000, (2, 5):  90000, (3, 5):  50000, (4, 5):  80000, (5, 5): 170000,
                                    (2, 6):  40000, (3, 6):  30000, (4, 6): 210000,
                                    (2, 7): 180000, (3, 7): 230000,
}
DEPT_SHORT: dict[int, str] = {
    240000: "TUM", 200000: "PIU", 150000: "LAM", 160000: "LOR",
     60000: "CAJ",  10000: "AMA", 130000: "LAL", 220000: "SMA",
     20000: "ANC", 100000: "HUA", 250000: "UCA",
     70000: "CAL", 140000: "LIM", 190000: "PAS", 120000: "JUN",
    110000: "ICA",  90000: "HVC",  50000: "AYA",  80000: "CUS", 170000: "MAD",
     40000: "ARE",  30000: "APU", 210000: "PUN",
    180000: "MOQ", 230000: "TAC",
}
CARTO_COLS = max(c for c, _ in DEPT_GRID) + 1
CARTO_ROWS = max(r for _, r in DEPT_GRID) + 1
DRILLDOWN_LEFT = "RENOVACIÓN POPULAR"
DRILLDOWN_RIGHT = "JUNTOS POR EL PERÚ"
DRILLDOWN_LABELS = {
    DRILLDOWN_LEFT: "RLA",
    DRILLDOWN_RIGHT: "Sánchez",
}
LIMA_PROVINCE_UBIGEO = 140100
PAPER = "#F5F2EB"
INK = "#1C1A20"
ACCENT = "#BF3030"
SOFT = "#EDE8DF"
CARD = "#FFFFFF"
FOREIGN_GEO_NAME_CACHE = Path(__file__).with_name("output") / "foreign_geo_names.json"
FOREIGN_HISTORY_URL = (
    "https://www.web.onpe.gob.pe/modElecciones/elecciones/elecciones2016/"
    "PRPCP2016/participacion-ciudadana-Extranjero-{code}-Pie.html"
)
FOREIGN_CONTINENT_NAMES = {
    910000: "África",
    920000: "América",
    930000: "Asia",
    940000: "Europa",
    950000: "Oceanía",
}

DASHBOARD_TZ = ZoneInfo("America/Chicago")


@dataclass
class DepartmentProjection:
    code: int
    name: str
    pct_actas: float
    emitidos_proj: float
    candidate_votes: dict[str, float]
    candidate_pct: dict[str, float]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=12000)
    ap.add_argument("--api-base", type=str, default=op.API)
    ap.add_argument("--referer", type=str, default=op.DEFAULT_REFERER)
    ap.add_argument("--election-id", type=int, default=op.ID_ELECCION)
    ap.add_argument("--geo-level", choices=["region", "province", "district"], default="district")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--swing", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trend-window", type=int, default=4)
    ap.add_argument("--history-size", type=int, default=96)
    ap.add_argument("--late-bias-window", type=int, default=op.LATE_BIAS_DEFAULT_WINDOW)
    ap.add_argument("--late-bias-strength", type=float, default=op.LATE_BIAS_DEFAULT_STRENGTH)
    ap.add_argument("--late-bias-recency", type=float, default=op.LATE_BIAS_DEFAULT_RECENCY)
    ap.add_argument("--no-late-bias", action="store_true")
    ap.add_argument("--track", type=str, default=",".join(op.TRACK_DEFAULT))
    ap.add_argument("--state-file", type=str, default=None)
    ap.add_argument("--html-out", type=str, default="dashboard.html")
    ap.add_argument("--json-out", type=str, default="dashboard_data.json")
    return ap.parse_args()


def fmt_int(value: float) -> str:
    return f"{int(round(value)):,}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}%"


def iso_to_local_label(value: str | None) -> str:
    if not value:
        return "sin dato"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(DASHBOARD_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def epoch_millis_to_local_label(value: int | float | None) -> str:
    if not value:
        return "sin dato"
    dt = datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).astimezone(DASHBOARD_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def fetch_onpe_updated_at_millis() -> int | None:
    candidates = []
    for params in (
        {"tipoFiltro": "ambito_geografico", "idAmbitoGeografico": 1},
        {"tipoFiltro": "ambito_geografico", "idAmbitoGeografico": 2},
    ):
        try:
            totals = op.fetch("resumen-general/totales", **params) or {}
        except Exception:
            totals = op.fetch_totales(**params) or {}
        value = totals.get("fechaActualizacion")
        if value:
            candidates.append(int(value))
    return max(candidates) if candidates else None


def fetch_national_vote_totals() -> dict[str, int]:
    """Fetch vote totals per candidate from the national presidential endpoint.
    This is more up-to-date than aggregating district-level data."""
    try:
        data = op.fetch(
            "eleccion-presidencial/participantes-ubicacion-geografica-nombre",
            tipoFiltro="ambito_geografico",
            idAmbitoGeografico=1,
        )
        candidatos, votos = op._extraer_candidatos(data)
        return {cand: int(v) for cand, v in zip(candidatos, votos)}
    except Exception:
        return {}


def fetch_department_names() -> dict[int, str]:
    names = {0: "EXTRANJERO"}
    try:
        rows = op.fetch("ubigeos/departamentos", idAmbitoGeografico=1)
    except Exception:
        return {**DEPARTMENT_NAMES, **names}
    for row in rows:
        try:
            names[int(row["ubigeo"])] = str(row["nombre"])
        except (KeyError, TypeError, ValueError):
            continue
    return {**DEPARTMENT_NAMES, **names}


def ensure_snapshot(
    state_path: Path,
    regiones: list[op.Region],
    history_size: int,
) -> tuple[dict, dict, bool]:
    state = op.load_state(state_path)
    current = op.snapshot_from_regiones(regiones)
    snapshots = state.get("snapshots", [])
    changed = False
    if not snapshots:
        op.append_snapshot(state, current, history_size)
        op.save_state(state_path, state)
        changed = True
    else:
        prev = snapshots[-1]
        if op.snapshot_changed(prev, current):
            op.append_snapshot(state, current, history_size)
            op.save_state(state_path, state)
            changed = True
        else:
            current = prev
    return state, current, changed


def resolve_tracked(regiones: list[op.Region], raw_track: str) -> list[str]:
    tracked = op.resolve_candidates(regiones, op.parse_track_patterns(raw_track))
    if tracked:
        return tracked
    fallback = []
    seen = set()
    for region in regiones:
        for cand in region.candidatos:
            if cand not in seen:
                fallback.append(cand)
                seen.add(cand)
            if len(fallback) >= 3:
                return fallback
    return fallback


def region_expected_votes(
    region: op.Region,
    universe: list[str],
    alpha_prior: float,
    late_bias_model: dict | None,
) -> np.ndarray:
    votes = np.zeros(len(universe), dtype=np.float64)
    idx = {cand: i for i, cand in enumerate(universe)}
    for cand, value in zip(region.candidatos, region.votos):
        if cand in idx:
            votes[idx[cand]] = float(value)

    remaining = region.votos_faltantes_estimados()
    total_r = float(votes.sum())
    if total_r > 0:
        base_share = (votes + alpha_prior) / (total_r + alpha_prior * len(universe))
    else:
        base_share = np.full(len(universe), 1.0 / len(universe))
    expected_share = op.expected_share_with_late_bias(region, universe, base_share, late_bias_model)
    return votes + expected_share * remaining


def department_projections(
    regiones: list[op.Region],
    universe: list[str],
    tracked: list[str],
    alpha_prior: float,
    late_bias_model: dict | None,
    department_name_map: dict[int, str] | None = None,
) -> tuple[list[DepartmentProjection], dict[str, dict[str, float]]]:
    idx = {cand: i for i, cand in enumerate(universe)}
    department_name_map = department_name_map or DEPARTMENT_NAMES
    buckets: dict[int, dict[str, object]] = {}

    for region in regiones:
        code = 0 if region.ambito == 2 else region.ubigeo_region
        bucket = buckets.setdefault(
            code,
            {
                "name": department_name_map.get(code, DEPARTMENT_NAMES.get(code, f"UBIGEO_{code}")),
                "actas_done": 0,
                "actas_total": 0,
                "emitidos_proj": 0.0,
                "candidate_votes": {cand: 0.0 for cand in tracked},
            },
        )
        expected_votes = region_expected_votes(region, universe, alpha_prior, late_bias_model)
        emitidos_proj = float(expected_votes.sum())
        bucket["actas_done"] += region.actas_contabilizadas
        bucket["actas_total"] += region.actas_totales
        bucket["emitidos_proj"] += emitidos_proj
        for cand in tracked:
            bucket["candidate_votes"][cand] += expected_votes[idx[cand]]

    rows: list[DepartmentProjection] = []
    for code, bucket in buckets.items():
        emitidos_proj = float(bucket["emitidos_proj"])
        candidate_votes = {
            cand: float(value) for cand, value in bucket["candidate_votes"].items()
        }
        candidate_pct = {
            cand: (100.0 * value / emitidos_proj if emitidos_proj else 0.0)
            for cand, value in candidate_votes.items()
        }
        rows.append(
            DepartmentProjection(
                code=code,
                name=str(bucket["name"]),
                pct_actas=op.pct(bucket["actas_done"], bucket["actas_total"]),
                emitidos_proj=emitidos_proj,
                candidate_votes=candidate_votes,
                candidate_pct=candidate_pct,
            )
        )

    rows.sort(key=lambda row: (row.code == 0, row.name))

    blocks = {
        "regiones": {"emitidos_proj": 0.0, "candidate_votes": {cand: 0.0 for cand in tracked}},
        "lima": {"emitidos_proj": 0.0, "candidate_votes": {cand: 0.0 for cand in tracked}},
        "extranjero": {"emitidos_proj": 0.0, "candidate_votes": {cand: 0.0 for cand in tracked}},
        "total": {"emitidos_proj": 0.0, "candidate_votes": {cand: 0.0 for cand in tracked}},
    }
    for row in rows:
        if row.code == 0:
            block = "extranjero"
        elif row.code == 140000:
            block = "lima"
        else:
            block = "regiones"
        for key in (block, "total"):
            blocks[key]["emitidos_proj"] += row.emitidos_proj
            for cand in tracked:
                blocks[key]["candidate_votes"][cand] += row.candidate_votes[cand]
    return rows, blocks


def current_probs(universe: list[str], contado: np.ndarray, pct_draws: np.ndarray) -> list[dict]:
    total_counted = int(contado.sum())
    actual_pct = contado / total_counted * 100.0 if total_counted else np.zeros_like(contado, dtype=float)
    med = np.median(pct_draws, axis=0)
    lo = np.percentile(pct_draws, 2.5, axis=0)
    hi = np.percentile(pct_draws, 97.5, axis=0)

    excluding = {i for i, cand in enumerate(universe) if "BLANCO" in cand.upper() or "NULO" in cand.upper()}
    valid_draws = pct_draws.copy()
    if excluding:
        valid_draws[:, list(excluding)] = -1
    top2 = np.argsort(-valid_draws, axis=1)[:, :2]
    counts = np.zeros(len(universe), dtype=np.int64)
    for row in top2:
        counts[row] += 1
    top2_prob = counts / pct_draws.shape[0]

    rows = []
    for i, cand in enumerate(universe):
        rows.append(
            {
                "candidate": cand,
                "actual_pct": float(actual_pct[i]),
                "projected_pct": float(med[i]),
                "ic_lo": float(lo[i]),
                "ic_hi": float(hi[i]),
                "top2_prob": float(top2_prob[i]),
            }
        )
    rows.sort(key=lambda row: row["projected_pct"], reverse=True)
    return rows


def latest_delta_blocks(state: dict, trend_window: int, tracked: list[str]) -> dict[str, object]:
    snapshots = state.get("snapshots", [])
    if len(snapshots) < 2:
        return {
            "latest": None,
            "trend": None,
            "has_history": False,
            "tracked": tracked,
        }

    latest = op.diff_snapshots(snapshots[-2], snapshots[-1])
    base_index = max(len(snapshots) - min(trend_window, len(snapshots) - 1) - 1, 0)
    trend = op.diff_snapshots(snapshots[base_index], snapshots[-1])

    def serialize(delta: dict) -> dict:
        block_rows = []
        for block in BLOCK_ORDER:
            row = delta["blocks"][block]
            cand_rows = []
            for cand in tracked:
                stats = row["candidates"].get(cand, {"curr": 0, "new": 0})
                cand_rows.append(
                    {
                        "candidate": cand,
                        "curr_pct": op.pct(stats["curr"], row["curr_validos"]),
                        "new_pct": op.pct(stats["new"], row["new_validos"]),
                        "new_votes": int(stats["new"]),
                    }
                )
            block_rows.append(
                {
                    "block": block,
                    "label": BLOCK_LABELS[block],
                    "new_validos": int(row["new_validos"]),
                    "curr_validos": int(row["curr_validos"]),
                    "candidates": cand_rows,
                }
            )
        return {
            "base_at": delta["base_at"],
            "curr_at": delta["curr_at"],
            "blocks": block_rows,
        }

    return {
        "latest": serialize(latest),
        "trend": serialize(trend),
        "has_history": True,
        "tracked": tracked,
    }


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


def projection_point_from_blocks(
    dept_blocks: dict[str, dict[str, float]],
    tracked: list[str],
) -> dict[str, dict[str, float]]:
    total_emitidos = float(dept_blocks["total"]["emitidos_proj"])
    out: dict[str, dict[str, float]] = {}
    for cand in tracked:
        votes = float(dept_blocks["total"]["candidate_votes"][cand])
        out[cand] = {
            "projected_pct": (100.0 * votes / total_emitidos) if total_emitidos else 0.0,
            "projected_votes": round(votes),
        }
    return out


def snapshot_projection_point(
    snapshots: list[dict],
    index: int,
    tracked: list[str],
    alpha_prior: float,
    late_bias_window: int,
    late_bias_strength: float,
    late_bias_recency: float,
) -> dict[str, object]:
    snapshot = snapshots[index]
    regiones = regions_from_snapshot(snapshot)
    universe = []
    for region in regiones:
        for candidate in region.candidatos:
            if candidate not in universe:
                universe.append(candidate)

    late_bias_model = None
    if late_bias_strength > 0 and index > 0:
        late_bias_model = op.build_late_bias_model(
            regiones=regiones,
            history_snapshots=snapshots[:index],
            current_snapshot=snapshot,
            window=max(late_bias_window, 2),
            strength=op.clip01(late_bias_strength),
            recency_decay=op.clip01(late_bias_recency),
        )

    _, dept_blocks = department_projections(
        regiones=regiones,
        universe=universe,
        tracked=tracked,
        alpha_prior=alpha_prior,
        late_bias_model=late_bias_model,
    )
    return {
        "snapshot_at": snapshot.get("captured_at"),
        "actas_pct": op.pct(
            int(snapshot.get("actas_contadas", 0)),
            int(snapshot.get("actas_totales", 0)),
        ),
        "candidates": projection_point_from_blocks(dept_blocks, tracked),
    }


def build_projection_shift(
    state: dict,
    tracked: list[str],
    current_projection_point: dict[str, dict[str, float]],
    alpha_prior: float,
    trend_window: int,
    late_bias_window: int,
    late_bias_strength: float,
    late_bias_recency: float,
) -> dict[str, object]:
    snapshots = state.get("snapshots", [])
    if len(snapshots) < 2:
        return {
            "latest": None,
            "trend": None,
            "has_history": False,
            "tracked": tracked,
        }

    prev_index = len(snapshots) - 2
    trend_index = max(len(snapshots) - min(trend_window, len(snapshots) - 1) - 1, 0)
    previous_point = snapshot_projection_point(
        snapshots=snapshots,
        index=prev_index,
        tracked=tracked,
        alpha_prior=alpha_prior,
        late_bias_window=late_bias_window,
        late_bias_strength=late_bias_strength,
        late_bias_recency=late_bias_recency,
    )
    trend_point = snapshot_projection_point(
        snapshots=snapshots,
        index=trend_index,
        tracked=tracked,
        alpha_prior=alpha_prior,
        late_bias_window=late_bias_window,
        late_bias_strength=late_bias_strength,
        late_bias_recency=late_bias_recency,
    )

    def serialize(base_point: dict[str, object], label: str) -> dict[str, object]:
        rows = []
        for cand in tracked:
            current = current_projection_point.get(cand, {"projected_pct": 0.0, "projected_votes": 0})
            base = base_point["candidates"].get(cand, {"projected_pct": 0.0, "projected_votes": 0})
            delta_pct = float(current["projected_pct"] - base["projected_pct"])
            delta_votes = int(current["projected_votes"] - base["projected_votes"])
            direction = "sin cambio"
            if delta_pct > 0.005:
                direction = "sube"
            elif delta_pct < -0.005:
                direction = "baja"
            rows.append(
                {
                    "candidate": cand,
                    "current_pct": float(current["projected_pct"]),
                    "base_pct": float(base["projected_pct"]),
                    "delta_pct": delta_pct,
                    "current_votes": int(current["projected_votes"]),
                    "base_votes": int(base["projected_votes"]),
                    "delta_votes": delta_votes,
                    "direction": direction,
                }
            )
        rows.sort(key=lambda row: row["current_pct"], reverse=True)
        return {
            "label": label,
            "base_at": base_point["snapshot_at"],
            "base_actas_pct": base_point["actas_pct"],
            "rows": rows,
        }

    return {
        "latest": serialize(previous_point, "Contra el snapshot previo"),
        "trend": serialize(trend_point, "Contra la base de tendencia"),
        "has_history": True,
        "tracked": tracked,
    }


def remaining_blocks(regiones: list[op.Region]) -> list[dict[str, float]]:
    rows = []
    for block in BLOCK_ORDER:
        remaining = sum(
            region.votos_faltantes_estimados()
            for region in regiones
            if op.block_key(region.ambito, region.ubigeo_region) == block
        )
        counted = sum(
            region.votos_validos
            for region in regiones
            if op.block_key(region.ambito, region.ubigeo_region) == block
        )
        total = counted + remaining
        rows.append(
            {
                "block": block,
                "label": BLOCK_LABELS[block],
                "remaining_votes": int(remaining),
                "counted_votes": int(counted),
                "remaining_pct": op.pct(remaining, total),
            }
        )
    return rows


def fetch_name_map(
    path: str,
    query_key: str,
    query_values: list[int],
    value_key: str = "nombre",
) -> dict[int, str]:
    names: dict[int, str] = {}
    for query_value in query_values:
        try:
            rows = op.fetch(path, idAmbitoGeografico=1, **{query_key: query_value})
        except Exception:
            continue
        for row in rows:
            try:
                names[int(row["ubigeo"])] = str(row[value_key])
            except (KeyError, TypeError, ValueError):
                continue
    return names


def _load_foreign_name_cache() -> dict[str, dict[str, str]]:
    if not FOREIGN_GEO_NAME_CACHE.exists():
        return {"countries": {}, "continents": {}}
    try:
        payload = json.loads(FOREIGN_GEO_NAME_CACHE.read_text())
    except Exception:
        return {"countries": {}, "continents": {}}
    return {
        "countries": {str(k): str(v) for k, v in (payload.get("countries") or {}).items()},
        "continents": {str(k): str(v) for k, v in (payload.get("continents") or {}).items()},
    }


def _save_foreign_name_cache(cache: dict[str, dict[str, str]]) -> None:
    FOREIGN_GEO_NAME_CACHE.parent.mkdir(parents=True, exist_ok=True)
    FOREIGN_GEO_NAME_CACHE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True)
    )


def _fetch_historical_foreign_name(code: int, kind: str) -> str | None:
    pattern = r"Pa[ií]s:\s*([^<\r\n]+)" if kind == "country" else r"Continente:\s*([^<\r\n]+)"
    url = FOREIGN_HISTORY_URL.format(code=code)
    for attempt in range(3):
        try:
            response = op.requests.get(
                url,
                headers={"User-Agent": op.UA, "Accept": "text/html,application/xhtml+xml"},
                timeout=20,
            )
            response.raise_for_status()
            match = re.search(pattern, response.text, re.IGNORECASE)
            if match:
                return html.unescape(match.group(1)).strip()
        except Exception:
            if attempt == 2:
                break
            time.sleep(0.4 * (attempt + 1))
    return None


def resolve_foreign_names(country_codes: list[int], continent_codes: list[int]) -> tuple[dict[int, str], dict[int, str]]:
    cache = _load_foreign_name_cache()
    countries = {int(k): v for k, v in cache["countries"].items()}
    continents = {int(k): v for k, v in cache["continents"].items()}
    changed = False

    for continent_code in continent_codes:
        if continent_code in continents:
            continue
        continents[continent_code] = (
            _fetch_historical_foreign_name(continent_code, "continent")
            or FOREIGN_CONTINENT_NAMES.get(continent_code, f"CONT_{continent_code}")
        )
        changed = True

    for country_code in country_codes:
        if country_code in countries:
            continue
        countries[country_code] = (
            _fetch_historical_foreign_name(country_code, "country")
            or f"PAÍS {country_code}"
        )
        changed = True

    if changed:
        _save_foreign_name_cache(
            {
                "countries": {str(k): v for k, v in countries.items()},
                "continents": {str(k): v for k, v in continents.items()},
            }
        )

    return countries, continents


def _estimate_foreign_remaining_votes(
    counted_actas: int,
    total_actas: int,
    total_valid: float,
    foreign_region: op.Region | None,
) -> float:
    remaining_actas = max(total_actas - counted_actas, 0)
    if remaining_actas <= 0:
        return 0.0

    foreign_valid_ratio = 0.0
    foreign_emitidos_por_acta = 0.0
    if foreign_region is not None:
        if foreign_region.votos_emitidos > 0 and foreign_region.votos_validos > 0:
            foreign_valid_ratio = foreign_region.votos_validos / foreign_region.votos_emitidos
        else:
            foreign_valid_ratio = float(foreign_region.ref_valid_ratio or 0.0)
        if foreign_region.actas_contabilizadas > 0 and foreign_region.votos_emitidos > 0:
            foreign_emitidos_por_acta = foreign_region.votos_emitidos / foreign_region.actas_contabilizadas
        else:
            foreign_emitidos_por_acta = float(foreign_region.ref_emitidos_por_acta or 0.0)

    local_emitidos_por_acta = 0.0
    if counted_actas > 0 and total_valid > 0:
        local_validos_por_acta = total_valid / counted_actas
        local_emitidos_por_acta = (
            local_validos_por_acta / foreign_valid_ratio
            if foreign_valid_ratio > 0
            else local_validos_por_acta
        )

    if local_emitidos_por_acta > 0 and foreign_emitidos_por_acta > 0:
        weight = min(counted_actas / 5.0, 1.0)
        emitidos_por_acta = (
            weight * local_emitidos_por_acta
            + (1.0 - weight) * foreign_emitidos_por_acta
        )
    else:
        emitidos_por_acta = local_emitidos_por_acta or foreign_emitidos_por_acta

    return max(0.0, remaining_actas * emitidos_por_acta)


def fetch_foreign_country_rows(
    foreign_region: op.Region | None,
    left_candidate: str,
    right_candidate: str,
) -> list[dict[str, float | int | str]]:
    continent_rows = op.fetch(
        "resumen-general/mapa-calor",
        tipoFiltro="ambito_geografico",
        idAmbitoGeografico=2,
    )
    continent_codes = sorted(
        {int(row.get("ubigeoNivel01") or 0) for row in continent_rows if int(row.get("ambitoGeografico", 0)) == 2}
    )

    raw_rows: dict[int, dict[str, int | float]] = {}
    for continent_code in continent_codes:
        rows = op.fetch(
            "resumen-general/mapa-calor",
            tipoFiltro="ubigeo_nivel_01",
            ubigeoNivel1=continent_code,
            idAmbitoGeografico=2,
        )
        for row in rows:
            if int(row.get("ubigeoNivel01") or 0) != continent_code:
                continue
            country_code = int(row.get("ubigeoNivel02") or 0)
            counted_actas = int(row.get("actasContabilizadas", 0))
            pct_actas = float(row.get("porcentajeActasContabilizadas", 0.0))
            total_actas = int(round(counted_actas / (pct_actas / 100.0))) if pct_actas > 0 else counted_actas
            raw_rows[country_code] = {
                "continent_code": continent_code,
                "country_code": country_code,
                "counted_actas": counted_actas,
                "pct_actas": pct_actas,
                "total_actas": total_actas,
            }

    def _worker(row: dict[str, int | float]) -> dict[str, float | int | str]:
        continent_code = int(row["continent_code"])
        country_code = int(row["country_code"])
        counted_actas = int(row["counted_actas"])
        total_actas = int(row["total_actas"])
        pct_actas = float(row["pct_actas"])
        left_votes = 0.0
        right_votes = 0.0
        total_valid = 0.0
        if counted_actas > 0:
            data = op.fetch(
                "eleccion-presidencial/participantes-ubicacion-geografica-nombre",
                tipoFiltro="ubigeo_nivel_02",
                ubigeoNivel1=continent_code,
                ubigeoNivel2=country_code,
                idAmbitoGeografico=2,
            )
            candidatos, votos = op._extraer_candidatos(data)
            vote_map = {cand: float(value) for cand, value in zip(candidatos, votos)}
            left_votes = float(vote_map.get(left_candidate, 0.0))
            right_votes = float(vote_map.get(right_candidate, 0.0))
            total_valid = float(sum(vote_map.values()))

        payload = territory_compare_row(
            name=f"PAÍS {country_code}",
            code=country_code,
            pct_actas=pct_actas,
            remaining_votes=_estimate_foreign_remaining_votes(
                counted_actas=counted_actas,
                total_actas=total_actas,
                total_valid=total_valid,
                foreign_region=foreign_region,
            ),
            left_votes=left_votes,
            right_votes=right_votes,
            total_valid=total_valid,
        )
        payload["continent_code"] = continent_code
        payload["counted_actas"] = counted_actas
        payload["total_actas"] = total_actas
        return payload

    rows: list[dict[str, float | int | str]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_worker, row) for row in raw_rows.values()]
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(
        key=lambda row: (float(row["remaining_votes"]), float(row["left_pct"] + row["right_pct"])),
        reverse=True,
    )
    top_rows = rows[:12]
    country_names, continent_names = resolve_foreign_names(
        [int(row["code"]) for row in top_rows],
        sorted({int(row["continent_code"]) for row in top_rows}),
    )
    for row in top_rows:
        row["name"] = country_names.get(int(row["code"]), str(row["name"]))
        row["continent"] = continent_names.get(
            int(row["continent_code"]),
            FOREIGN_CONTINENT_NAMES.get(int(row["continent_code"]), "Extranjero"),
        )
    return top_rows


def territory_compare_row(
    *,
    name: str,
    code: int,
    pct_actas: float,
    remaining_votes: float,
    left_votes: float,
    right_votes: float,
    total_valid: float,
) -> dict[str, float | int | str]:
    left_pct = 100.0 * left_votes / total_valid if total_valid else 0.0
    right_pct = 100.0 * right_votes / total_valid if total_valid else 0.0
    return {
        "name": name,
        "code": code,
        "pct_actas": pct_actas,
        "remaining_votes": int(round(remaining_votes)),
        "left_pct": left_pct,
        "right_pct": right_pct,
        "gap_pp": right_pct - left_pct,
    }


def build_territory_drilldown(
    province_units: list[op.Region],
    district_units: list[op.Region],
    department_name_map: dict[int, str] | None = None,
    left_candidate: str = DRILLDOWN_LEFT,
    right_candidate: str = DRILLDOWN_RIGHT,
) -> dict[str, object]:
    department_name_map = department_name_map or DEPARTMENT_NAMES
    province_region_codes = sorted({r.ubigeo_region for r in province_units if r.ambito == 1})
    province_name_map = fetch_name_map(
        "ubigeos/provincias",
        "idUbigeoDepartamento",
        province_region_codes,
    )
    district_name_map = fetch_name_map(
        "ubigeos/distritos",
        "idUbigeoProvincia",
        [LIMA_PROVINCE_UBIGEO],
    )

    region_agg: dict[int, dict[str, float | int | str]] = {}
    provinces_by_region: dict[int, list[dict[str, float | int | str]]] = {}

    for region in province_units:
        if region.ambito != 1:
            continue
        candidate_votes = region.votos_por_candidato()
        left_votes = float(candidate_votes.get(left_candidate, 0))
        right_votes = float(candidate_votes.get(right_candidate, 0))
        region_code = region.ubigeo_region
        province_row = territory_compare_row(
            name=province_name_map.get(region.ubigeo, region.nombre),
            code=region.ubigeo,
            pct_actas=region.pct_actas,
            remaining_votes=region.votos_faltantes_estimados(),
            left_votes=left_votes,
            right_votes=right_votes,
            total_valid=float(region.votos_validos),
        )
        provinces_by_region.setdefault(region_code, []).append(province_row)

        bucket = region_agg.setdefault(
            region_code,
            {
                "name": department_name_map.get(region_code, DEPARTMENT_NAMES.get(region_code, f"UBIGEO_{region_code}")),
                "actas_done": 0,
                "actas_total": 0,
                "remaining_votes": 0.0,
                "left_votes": 0.0,
                "right_votes": 0.0,
                "total_valid": 0.0,
            },
        )
        bucket["actas_done"] += region.actas_contabilizadas
        bucket["actas_total"] += region.actas_totales
        bucket["remaining_votes"] += region.votos_faltantes_estimados()
        bucket["left_votes"] += left_votes
        bucket["right_votes"] += right_votes
        bucket["total_valid"] += float(region.votos_validos)

    region_rows = []
    for region_code, bucket in region_agg.items():
        region_rows.append(
            territory_compare_row(
                name=str(bucket["name"]),
                code=region_code,
                pct_actas=op.pct(float(bucket["actas_done"]), float(bucket["actas_total"])),
                remaining_votes=float(bucket["remaining_votes"]),
                left_votes=float(bucket["left_votes"]),
                right_votes=float(bucket["right_votes"]),
                total_valid=float(bucket["total_valid"]),
            )
        )
    region_rows.sort(key=lambda row: row["remaining_votes"], reverse=True)

    for rows in provinces_by_region.values():
        rows.sort(key=lambda row: row["remaining_votes"], reverse=True)

    lima_district_rows = []
    for region in district_units:
        if region.ambito != 1 or region.ubigeo_region != op.LIMA_UBIGEO:
            continue
        candidate_votes = region.votos_por_candidato()
        lima_district_rows.append(
            territory_compare_row(
                name=district_name_map.get(region.ubigeo, region.nombre),
                code=region.ubigeo,
                pct_actas=region.pct_actas,
                remaining_votes=region.votos_faltantes_estimados(),
                left_votes=float(candidate_votes.get(left_candidate, 0)),
                right_votes=float(candidate_votes.get(right_candidate, 0)),
                total_valid=float(region.votos_validos),
            )
        )
    lima_district_rows.sort(key=lambda row: row["remaining_votes"], reverse=True)

    foreign_region = next((region for region in province_units if region.ambito == 2), None)
    foreign_rows = fetch_foreign_country_rows(
        foreign_region=foreign_region,
        left_candidate=left_candidate,
        right_candidate=right_candidate,
    )

    top_regions = region_rows[:10]
    top_region_details = []
    for row in top_regions[:6]:
        top_region_details.append(
            {
                "name": row["name"],
                "remaining_votes": row["remaining_votes"],
                "left_pct": row["left_pct"],
                "right_pct": row["right_pct"],
                "gap_pp": row["gap_pp"],
                "provinces": provinces_by_region.get(int(row["code"]), [])[:6],
            }
        )

    return {
        "left_candidate": left_candidate,
        "right_candidate": right_candidate,
        "left_label": DRILLDOWN_LABELS.get(left_candidate, TRACK_SHORT_LABELS.get(left_candidate, left_candidate)),
        "right_label": DRILLDOWN_LABELS.get(right_candidate, TRACK_SHORT_LABELS.get(right_candidate, right_candidate)),
        "regions_top_remaining": top_regions,
        "region_details": top_region_details,
        "lima_districts_top_remaining": lima_district_rows[:15],
        "foreign": foreign_rows,
    }


def snapshot_candidate_share(snapshot: dict, candidate: str) -> float:
    total_validos = int(snapshot.get("votos_validos", 0))
    if total_validos <= 0:
        return 0.0
    candidate_votes = 0
    for region in snapshot.get("regions", {}).values():
        candidate_votes += int(region.get("votos_por_candidato", {}).get(candidate, 0))
    return 100.0 * candidate_votes / total_validos


def compress_snapshots(snapshots: list[dict], max_points: int = 18) -> list[dict]:
    if len(snapshots) <= max_points:
        return snapshots
    indexes = sorted(set(np.linspace(0, len(snapshots) - 1, max_points, dtype=int).tolist()))
    return [snapshots[idx] for idx in indexes]


def nice_step(span: float, target_ticks: int = 6) -> float:
    if span <= 0:
        return 1.0
    raw = span / max(target_ticks, 1)
    power = 10 ** math.floor(math.log10(raw))
    scaled = raw / power
    if scaled <= 1:
        factor = 1
    elif scaled <= 2:
        factor = 2
    elif scaled <= 5:
        factor = 5
    else:
        factor = 10
    return factor * power


def build_trajectory_data(
    state: dict,
    focus_candidates: list[str],
    tracked_projection: list[dict],
) -> dict:
    snapshots = compress_snapshots(state.get("snapshots", []))
    projection_map = {row["candidate"]: row["projected_pct"] for row in tracked_projection}
    series = []
    all_x = []
    all_y = []

    for candidate in focus_candidates:
        actual_points = []
        for snapshot in snapshots:
            x = op.pct(
                int(snapshot.get("actas_contadas", 0)),
                int(snapshot.get("actas_totales", 0)),
            )
            y = snapshot_candidate_share(snapshot, candidate)
            actual_points.append(
                {
                    "x": x,
                    "y": y,
                    "label": iso_to_local_label(snapshot.get("captured_at")),
                }
            )
            all_x.append(x)
            all_y.append(y)
        projected_y = projection_map.get(candidate, actual_points[-1]["y"] if actual_points else 0.0)
        series.append(
            {
                "candidate": candidate,
                "color": TRACK_COLORS.get(candidate, "#675842"),
                "actual": actual_points,
                "projected": {"x": 100.0, "y": projected_y},
            }
        )
        all_y.append(projected_y)

    if not all_x:
        return {"series": [], "x_ticks": [], "y_ticks": [], "bounds": None}

    x_min = max(75.0, math.floor(min(all_x) / 5.0) * 5.0)
    x_max = 100.0
    y_min_raw = min(all_y)
    y_max_raw = max(all_y)
    y_padding = max(0.25, (y_max_raw - y_min_raw) * 0.14)
    y_min = math.floor((y_min_raw - y_padding) * 2.0) / 2.0
    y_max = math.ceil((y_max_raw + y_padding) * 2.0) / 2.0

    x_ticks = []
    tick = x_min
    while tick <= x_max + 1e-9:
        x_ticks.append(round(tick, 2))
        tick += 5.0

    y_ticks = []
    y_step = nice_step(y_max - y_min, target_ticks=6)
    tick = math.floor(y_min / y_step) * y_step
    while tick <= y_max + 1e-9:
        if tick >= y_min - 1e-9:
            y_ticks.append(round(tick, 2))
        tick += y_step

    return {
        "series": series,
        "x_ticks": x_ticks,
        "y_ticks": y_ticks,
        "bounds": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
    }


def payload_to_jsonable(
    projection_rows: list[dict],
    tracked: list[str],
    dept_rows: list[DepartmentProjection],
    dept_blocks: dict[str, dict[str, float]],
    territory_drilldown: dict[str, object],
    current_snapshot: dict,
    onpe_updated_at_millis: int | None,
    state: dict,
    delta_info: dict[str, object],
    projection_shift: dict[str, object],
    remaining_info: list[dict[str, float]],
    late_bias_model: dict | None,
    changed: bool,
    regiones: list | None = None,
    national_vote_totals: dict[str, int] | None = None,
) -> dict:
    # Priority: national endpoint totals > fresh regiones > stale snapshot
    if national_vote_totals:
        current_votes_by_candidate: dict[str, int] = dict(national_vote_totals)
    elif regiones:
        current_votes_by_candidate = {}
        for region in regiones:
            for cand, votes in zip(region.candidatos, region.votos):
                current_votes_by_candidate[cand] = current_votes_by_candidate.get(cand, 0) + int(votes)
    else:
        current_votes_by_candidate = {}
        for region in current_snapshot.get("regions", {}).values():
            for candidate, votes in region.get("votos_por_candidato", {}).items():
                current_votes_by_candidate[candidate] = current_votes_by_candidate.get(candidate, 0) + int(votes)

    # Fresh total valid votes from regiones (more accurate than stale snapshot)
    total_validos_fresh = (
        sum(r.votos_validos for r in regiones)
        if regiones else int(current_snapshot.get("votos_validos", 0))
    )

    def current_comparison_row(left: str, right: str) -> dict | None:
        if left not in current_votes_by_candidate or right not in current_votes_by_candidate:
            return None
        total_validos = total_validos_fresh
        left_votes = current_votes_by_candidate[left]
        right_votes = current_votes_by_candidate[right]
        return {
            "left": left,
            "right": right,
            "left_votes": left_votes,
            "right_votes": right_votes,
            "vote_diff": left_votes - right_votes,
            "left_pct": op.pct(left_votes, total_validos),
            "right_pct": op.pct(right_votes, total_validos),
            "pp_diff": op.pct(left_votes, total_validos) - op.pct(right_votes, total_validos),
        }

    def block_candidate_pct(block: str, candidate: str) -> float:
        emitidos = dept_blocks[block]["emitidos_proj"]
        votes = dept_blocks[block]["candidate_votes"][candidate]
        return 100.0 * votes / emitidos if emitidos else 0.0

    tracked_projection = []
    for cand in tracked:
        proj = next((row for row in projection_rows if row["candidate"] == cand), None)
        if not proj:
            continue
        tracked_projection.append(
            {
                **proj,
                "projected_votes": round(dept_blocks["total"]["candidate_votes"][cand]),
            }
        )
    tracked_projection.sort(key=lambda row: row["projected_pct"], reverse=True)
    focus_candidates = tracked[:3] if len(tracked) > 3 else tracked[:]
    trajectory = build_trajectory_data(state, focus_candidates, tracked_projection)

    department_rows = []
    for row in dept_rows:
        item = {
            "code": row.code,
            "name": row.name,
            "pct_actas": row.pct_actas,
            "emitidos_proj": round(row.emitidos_proj),
            "candidates": [],
        }
        for cand in tracked:
            item["candidates"].append(
                {
                    "candidate": cand,
                    "votes": round(row.candidate_votes[cand]),
                    "pct": row.candidate_pct[cand],
                }
            )
        department_rows.append(item)

    comparison = []
    for i, left in enumerate(tracked_projection):
        for right in tracked_projection[i + 1 :]:
            comparison.append(
                {
                    "left": left["candidate"],
                    "right": right["candidate"],
                    "left_votes": left["projected_votes"],
                    "right_votes": right["projected_votes"],
                    "vote_diff": left["projected_votes"] - right["projected_votes"],
                    "left_pct": left["projected_pct"],
                    "right_pct": right["projected_pct"],
                    "pp_diff": left["projected_pct"] - right["projected_pct"],
                }
            )

    actual_comparison = [
        row
        for row in (
            current_comparison_row("RENOVACIÓN POPULAR", "PARTIDO DEL BUEN GOBIERNO"),
            current_comparison_row("RENOVACIÓN POPULAR", "JUNTOS POR EL PERÚ"),
        )
        if row is not None
    ]

    return {
        "generated_at": datetime.now(DASHBOARD_TZ).isoformat(),
        "snapshot_at": current_snapshot.get("captured_at"),
        "onpe_updated_at": onpe_updated_at_millis,
        "actas_contadas": int(current_snapshot.get("actas_contadas", 0)),
        "actas_totales": int(current_snapshot.get("actas_totales", 0)),
        "actas_pct": op.pct(
            int(current_snapshot.get("actas_contadas", 0)),
            int(current_snapshot.get("actas_totales", 0)),
        ),
        "votos_validos": int(current_snapshot.get("votos_validos", 0)),
        "onpe_changed": changed,
        "tracked": tracked,
        "focus_candidates": focus_candidates,
        "tracked_projection": tracked_projection,
        "trajectory": trajectory,
        "comparison": comparison,
        "actual_comparison": actual_comparison,
        "projection_rows": projection_rows,
        "department_rows": department_rows,
        "territory_drilldown": territory_drilldown,
        "projection_shift": projection_shift,
        "block_summary": [
            {
                "block": key,
                "label": BLOCK_LABELS[key],
                "emitidos_proj": round(dept_blocks[key]["emitidos_proj"]),
                "candidates": [
                    {
                        "candidate": cand,
                        "votes": round(dept_blocks[key]["candidate_votes"][cand]),
                        "pct": block_candidate_pct(key, cand),
                    }
                    for cand in tracked
                ],
            }
            for key in BLOCK_ORDER
        ],
        "remaining_blocks": remaining_info,
        "delta_info": delta_info,
        "late_bias": None
        if not late_bias_model
        else {
            "snapshots_used": late_bias_model["snapshots_used"],
            "base_at": late_bias_model["base_at"],
            "curr_at": late_bias_model["curr_at"],
            "strength": late_bias_model["strength"],
            "recency_decay": late_bias_model["recency_decay"],
        },
    }


def render_candidate_card(candidate: dict, position: int) -> str:
    color = TRACK_COLORS.get(candidate["candidate"], "#675842")
    top2 = candidate.get("top2_prob", 0.0)
    top2_pct = top2 * 100.0
    short = TRACK_SHORT_LABELS.get(candidate["candidate"], candidate["candidate"][:3])
    return f"""
    <article class="candidate-card" style="--c-color:{color};">
      <div class="candidate-card__top">
        <span class="candidate-card__rank">#{position} · {escape(short)}</span>
      </div>
      <h3 class="candidate-card__name">{escape(candidate["candidate"])}</h3>
      <div class="candidate-card__pcts">
        <div class="candidate-card__pct-block">
          <span class="candidate-card__big">{fmt_pct(candidate["actual_pct"])}</span>
          <span class="candidate-card__small">actual</span>
        </div>
        <div class="candidate-card__pct-block">
          <span class="candidate-card__big candidate-card__big--proj">{fmt_pct(candidate["projected_pct"])}</span>
          <span class="candidate-card__small">proyectado</span>
        </div>
      </div>
      <div class="candidate-card__bar">
        <span style="width:{min(candidate['projected_pct'] * 4.5, 100):.1f}%"></span>
      </div>
      <div class="candidate-card__ic">
        IC 95%: {fmt_pct(candidate['ic_lo'])} — {fmt_pct(candidate['ic_hi'])}
      </div>
      <div class="candidate-card__top2">
        <div class="candidate-card__top2-label">Prob. 2da vuelta</div>
        <div class="candidate-card__top2-bar">
          <span style="width:{top2_pct:.1f}%"></span>
        </div>
        <div class="candidate-card__top2-pct">{top2_pct:.1f}%</div>
      </div>
    </article>
    """


def render_cartogram(payload: dict) -> str:
    dept_rows = payload.get("department_rows", [])
    tracked = payload.get("tracked", [])
    # Build leader map: code -> (candidate, pct, pct_actas)
    leader_map: dict[int, tuple[str, float, float]] = {}
    for row in dept_rows:
        code = int(row["code"])
        if code == 0:
            continue
        best_cand, best_pct = "", -1.0
        for c in row.get("candidates", []):
            if c["pct"] > best_pct:
                best_pct = c["pct"]
                best_cand = c["candidate"]
        leader_map[code] = (best_cand, best_pct, float(row.get("pct_actas", 0.0)))

    reverse_grid: dict[tuple[int, int], int] = DEPT_GRID

    tiles: list[str] = []
    for row_idx in range(CARTO_ROWS):
        for col in range(CARTO_COLS):
            code = reverse_grid.get((col, row_idx))
            if code is None:
                tiles.append('<div class="carto-cell"></div>')
                continue
            name = DEPARTMENT_NAMES.get(code, f"DEP{code}")
            short = DEPT_SHORT.get(code, name[:3])
            info = leader_map.get(code)
            if info and info[0]:
                cand, pct, pct_actas = info
                color = TRACK_COLORS.get(cand, "#94a3b8")
                slabel = TRACK_SHORT_LABELS.get(cand, cand[:3])
                tooltip = f"{name}: {slabel} {pct:.1f}% ({pct_actas:.0f}% actas)"
                tiles.append(
                    f'<div class="carto-tile" style="--tile-color:{color};" title="{escape(tooltip)}">'
                    f'<span class="carto-tile__abbr">{escape(short)}</span>'
                    f'<span class="carto-tile__pct">{pct:.1f}%</span>'
                    f'<span class="carto-tile__party">{escape(slabel)}</span>'
                    f'<div class="carto-tile__actas"><span style="width:{pct_actas:.0f}%"></span></div>'
                    f'</div>'
                )
            else:
                tiles.append(
                    f'<div class="carto-tile carto-tile--nd" title="{escape(name)}: sin datos">'
                    f'<span class="carto-tile__abbr">{escape(short)}</span>'
                    f'<span class="carto-tile__pct">—</span>'
                    f'</div>'
                )

    legend_items = "".join(
        f'<div class="carto-legend__item">'
        f'<span class="carto-legend__dot" style="background:{TRACK_COLORS.get(c, "#888")};"></span>'
        f'<span>{escape(TRACK_SHORT_LABELS.get(c, c))}</span>'
        f'<span class="carto-legend__full">{escape(c)}</span>'
        f'</div>'
        for c in tracked
    )

    tiles_html = "".join(tiles)
    return f"""
    <section>
      <div class="section-title">
        <span class="eyebrow">Mapa político</span>
        <h2>Candidato líder por departamento</h2>
        <p class="section-note">Proyección al 100%. Barra inferior = porcentaje de actas contadas. Hover para detalle.</p>
      </div>
      <div class="carto-body">
        <div class="carto-grid" style="--carto-cols:{CARTO_COLS}; --carto-rows:{CARTO_ROWS};">
          {tiles_html}
        </div>
        <div class="carto-sidebar">
          <div class="carto-legend">
            {legend_items}
          </div>
          <p class="carto-legend__note">Barra inferior = % actas procesadas</p>
        </div>
      </div>
    </section>
    """


def render_battle_gauge(payload: dict) -> str:
    left_cand = DRILLDOWN_LEFT
    right_cand = DRILLDOWN_RIGHT
    left_color = TRACK_COLORS.get(left_cand, "#1e40af")
    right_color = TRACK_COLORS.get(right_cand, "#1f7a45")
    left_short = TRACK_SHORT_LABELS.get(left_cand, left_cand[:3])
    right_short = TRACK_SHORT_LABELS.get(right_cand, right_cand[:3])

    # Find from actual_comparison (no projection)
    actual = next(
        (r for r in payload.get("actual_comparison", [])
         if r["left"] == left_cand and r["right"] == right_cand),
        None,
    )
    proj = next(
        (r for r in payload.get("comparison", [])
         if r["left"] == left_cand and r["right"] == right_cand),
        None,
    )
    if not actual and not proj:
        return ""

    src = actual or proj
    left_pct = float(src["left_pct"])
    right_pct = float(src["right_pct"])
    left_votes = int(src["left_votes"])
    right_votes = int(src["right_votes"])
    is_proj = actual is None

    # Always positive — leader determined by which side has more votes
    diff_votes = abs(left_votes - right_votes)
    diff_pp = abs(left_pct - right_pct)
    label = "proyectado al 100%" if is_proj else "corte actual"
    if left_votes >= right_votes:
        leader, leader_color = left_short, left_color
    else:
        leader, leader_color = right_short, right_color

    total = left_pct + right_pct or 1.0
    left_bar = 100.0 * left_pct / total
    right_bar = 100.0 * right_pct / total

    return f"""
    <section>
      <div class="section-title">
        <span class="eyebrow">Duelo clave</span>
        <h2>2° lugar: {escape(left_short)} vs {escape(right_short)}</h2>
        <p class="section-note">Diferencia al {escape(label)}. Solo entre estos dos candidatos.</p>
      </div>
      <div class="battle-gauge">
        <div class="battle-gauge__header">
          <div class="battle-gauge__cand" style="text-align:left;">
            <div class="battle-gauge__cand-name" style="color:{left_color};">{escape(left_short)}</div>
            <div class="battle-gauge__cand-pct" style="color:{left_color};">{fmt_pct(left_pct)}</div>
            <div class="battle-gauge__cand-votes" style="color:{left_color};">{fmt_int(left_votes)} votos</div>
            <div class="battle-gauge__cand-full">{escape(left_cand)}</div>
          </div>
          <div class="battle-gauge__vs">vs</div>
          <div class="battle-gauge__cand" style="text-align:right;">
            <div class="battle-gauge__cand-name" style="color:{right_color};">{escape(right_short)}</div>
            <div class="battle-gauge__cand-pct" style="color:{right_color};">{fmt_pct(right_pct)}</div>
            <div class="battle-gauge__cand-votes" style="color:{right_color};">{fmt_int(right_votes)} votos</div>
            <div class="battle-gauge__cand-full">{escape(right_cand)}</div>
          </div>
        </div>
        <div class="battle-gauge__bar">
          <div class="battle-gauge__bar-left" style="width:{left_bar:.2f}%; background:{left_color};"></div>
          <div class="battle-gauge__bar-right" style="width:{right_bar:.2f}%; background:{right_color};"></div>
        </div>
        <div class="battle-gauge__labels">
          <span>{fmt_pct(left_pct)}</span>
          <span>{fmt_pct(right_pct)}</span>
        </div>
        <div class="battle-gauge__diff">
          <span style="color:{leader_color}; font-weight:700;">{escape(leader)}</span>
          lidera por <strong>{fmt_int(diff_votes)}</strong> votos
          (<strong>{diff_pp:.2f} pp</strong>)
        </div>
      </div>
    </section>
    """


def render_actual_comparison(payload: dict) -> str:
    rows = []
    for row in payload.get("actual_comparison", []):
        if not (
            row["left"] == "RENOVACIÓN POPULAR"
            and row["right"] == "JUNTOS POR EL PERÚ"
        ):
            continue
        sign_votes = "+" if row["vote_diff"] >= 0 else ""
        sign_pp = "+" if row["pp_diff"] >= 0 else ""
        rows.append(
            f"""
            <tr>
              <td>{escape(row['left'])}</td>
              <td>{escape(row['right'])}</td>
              <td>{fmt_int(row['left_votes'])}</td>
              <td>{fmt_int(row['right_votes'])}</td>
              <td>{sign_votes}{fmt_int(row['vote_diff'])}</td>
              <td>{sign_pp}{row['pp_diff']:.2f} pp</td>
            </tr>
            """
        )
    if not rows:
        return '<p class="empty-state">No se encontraron ambos candidatos en el corte actual.</p>'
    return f"""
      <div class="projection-shell">
        <table>
          <thead>
            <tr>
              <th>Candidato A</th>
              <th>Candidato B</th>
              <th>Votos A</th>
              <th>Votos B</th>
              <th>Diferencia real</th>
              <th>Diferencia real en puntos</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    """


def render_projected_comparison(payload: dict) -> str:
    rows = []
    for row in payload.get("comparison", []):
        if not (
            row["left"] == "RENOVACIÓN POPULAR"
            and row["right"] == "JUNTOS POR EL PERÚ"
        ):
            continue
        sign_votes = "+" if row["vote_diff"] >= 0 else ""
        sign_pp = "+" if row["pp_diff"] >= 0 else ""
        rows.append(
            f"""
            <tr>
              <td>{escape(row['left'])}</td>
              <td>{escape(row['right'])}</td>
              <td>{fmt_int(row['left_votes'])}</td>
              <td>{fmt_int(row['right_votes'])}</td>
              <td>{sign_votes}{fmt_int(row['vote_diff'])}</td>
              <td>{sign_pp}{row['pp_diff']:.2f} pp</td>
            </tr>
            """
        )
    if not rows:
        return '<p class="empty-state">No hay comparaciones proyectadas disponibles.</p>'
    return f"""
      <div class="projection-shell">
        <table>
          <thead>
            <tr>
              <th>Candidato A</th>
              <th>Candidato B</th>
              <th>Votos A proj.</th>
              <th>Votos B proj.</th>
              <th>Diferencia simulada</th>
              <th>Diferencia simulada en puntos</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    """


def render_remaining_blocks(payload: dict) -> str:
    max_remaining = max((row["remaining_votes"] for row in payload["remaining_blocks"]), default=1)
    cards = []
    for row in payload["remaining_blocks"]:
        width = 100.0 * row["remaining_votes"] / max_remaining if max_remaining else 0.0
        cards.append(
            f"""
            <article class="remaining-card">
              <div class="remaining-card__header">
                <h3 class="remaining-card__title">{escape(row['label'])}</h3>
                <span class="remaining-card__count">{fmt_int(row['remaining_votes'])} votos</span>
              </div>
              <div class="remaining-bar"><span style="width:{width:.2f}%"></span></div>
              <p class="remaining-card__note">Falta ~{fmt_pct(row['remaining_pct'])} del bloque.</p>
            </article>
            """
        )
    return "".join(cards)


def render_block_summary(payload: dict) -> str:
    blocks = []
    for block in payload["block_summary"]:
        bars = []
        for candidate in block["candidates"]:
            color = TRACK_COLORS.get(candidate["candidate"], "#675842")
            bars.append(
                f"""
                <div class="block-bar">
                  <div class="block-bar__head">
                    <span class="block-bar__name">{escape(candidate['candidate'])}</span>
                    <span class="block-bar__pct">{fmt_pct(candidate['pct'])}</span>
                  </div>
                  <div class="block-bar__track"><span style="width:{candidate['pct']:.2f}%; background:{color};"></span></div>
                </div>
                """
            )
        blocks.append(
            f"""
            <article class="block-card">
              <div class="block-card__header">
                <h3 class="block-card__title">{escape(block['label'])}</h3>
                <span class="block-card__total">{fmt_int(block['emitidos_proj'])} proj.</span>
              </div>
              {''.join(bars)}
            </article>
            """
        )
    return "".join(blocks)


def svg_polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def render_trajectory_chart(payload: dict) -> str:
    trajectory = payload.get("trajectory", {})
    series = trajectory.get("series", [])
    bounds = trajectory.get("bounds")
    if not series or not bounds:
        return ""
    legend_items = []
    endpoint_cards = []
    for line in series:
        short_label = TRACK_SHORT_LABELS.get(line["candidate"], line["candidate"])
        legend_items.append(
            f'''
            <div class="traj-legend__item">
              <span class="traj-legend__swatch" style="background:{line["color"]};"></span>
              <span>{escape(short_label)}</span>
              <span style="font-size:0.72rem;color:var(--ink-3);font-weight:400;">{escape(line["candidate"])}</span>
            </div>
            '''
        )
        endpoint_cards.append(
            f"""
            <article class="traj-endpoint">
              <div class="traj-endpoint__head">
                <span class="traj-endpoint__dot" style="background:{line['color']};"></span>
                <span class="traj-endpoint__code">{escape(short_label)}</span>
              </div>
              <span class="traj-endpoint__pct">{line['projected']['y']:.2f}%</span>
              <span class="traj-endpoint__name">{escape(line['candidate'])}</span>
            </article>
            """
        )

    latest_pct = payload["actas_pct"]
    trajectory_json = json.dumps(trajectory, ensure_ascii=False).replace("</", "<\\/")

    return f"""
    <section>
      <div class="section-title">
        <p class="eyebrow">Trayectoria</p>
        <h2>Evolución observada y estimación al 100%</h2>
        <p class="section-note">Línea continua = cortes observados. Tramo punteado = proyección. Punto final = estimación al cierre.</p>
      </div>
      <div class="traj-card">
        <div class="traj-header">
          <div class="traj-legend">
            {''.join(legend_items)}
            <div class="traj-legend__hint">Sólido = observado · Punteado = proyección al 100%</div>
          </div>
          <div class="traj-badge">
            <span class="eyebrow" style="text-align:right;">ONPE</span>
            <span class="traj-badge__pct">{fmt_pct(latest_pct, 3)}</span>
            <div class="traj-badge__label">actas procesadas</div>
          </div>
        </div>
        <div class="traj-shell">
          <div class="traj-canvas-wrap">
            <canvas id="trajectoryChart" aria-label="Trayectoria real y proyectada de candidatos"></canvas>
          </div>
          <aside class="traj-endpoints">
            {''.join(endpoint_cards)}
          </aside>
        </div>
        <script id="trajectory-data" type="application/json">{trajectory_json}</script>
      </div>
    </section>
    """


def render_delta_section(delta: dict | None, title: str) -> str:
    if not delta:
        return f"""
        <section class="delta-section">
          <div class="section-title">
            <p class="eyebrow">Incrementales</p>
            <h2>{escape(title)}</h2>
          </div>
          <p class="empty-state">Aún no hay al menos dos snapshots guardados para calcular esta comparación.</p>
        </section>
        """

    block_cards = []
    for block in delta["blocks"]:
        candidate_rows = []
        for candidate in block["candidates"]:
            color = TRACK_COLORS.get(candidate["candidate"], "#675842")
            candidate_rows.append(
                f"""
                <div class="delta-line">
                  <div class="delta-line__head">
                    <span class="delta-line__name">{escape(candidate['candidate'])}</span>
                    <span class="delta-line__pct">{fmt_pct(candidate['new_pct'])}</span>
                  </div>
                  <div class="delta-bar"><span style="width:{candidate['new_pct']:.2f}%; background:{color};"></span></div>
                  <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:var(--ink-3);margin-top:3px;">
                    <span>Acum. {fmt_pct(candidate['curr_pct'])}</span>
                    <span>{fmt_int(candidate['new_votes'])} nuevos</span>
                  </div>
                </div>
                """
            )
        block_cards.append(
            f"""
            <article class="delta-card">
              <div class="delta-card__header">
                <h3 class="delta-card__title">{escape(block['label'])}</h3>
                <span class="delta-card__count">{fmt_int(block['new_validos'])} nuevos válidos</span>
              </div>
              {''.join(candidate_rows)}
            </article>
            """
        )

    return f"""
    <section class="delta-section">
      <div class="section-title">
        <p class="eyebrow">Incrementales</p>
        <h2>{escape(title)}</h2>
        <p>{escape(iso_to_local_label(delta['base_at']))} → {escape(iso_to_local_label(delta['curr_at']))}</p>
      </div>
      <div class="delta-grid">
        {''.join(block_cards)}
      </div>
    </section>
    """


def render_projection_shift_section(shift: dict | None) -> str:
    if not shift:
        return f"""
        <section class="delta-section">
          <div class="section-title">
            <p class="eyebrow">Movimiento de proyección</p>
            <h2>Cómo cambia la estimación final</h2>
          </div>
          <p class="empty-state">Aún no hay suficiente historial para medir cómo se movió la proyección entre cortes.</p>
        </section>
        """

    cards = []
    for row in shift["rows"]:
        color = TRACK_COLORS.get(row["candidate"], "#675842")
        delta_pct = row["delta_pct"]
        delta_votes = row["delta_votes"]
        pct_sign = "+" if delta_pct >= 0 else ""
        votes_sign = "+" if delta_votes >= 0 else ""
        cards.append(
            f"""
            <article class="shift-card" style="--c-color:{color};">
              <div class="shift-card__header">
                <h3 class="shift-card__name">{escape(row['candidate'])}</h3>
                <span class="shift-pill">{escape(row['direction'])}</span>
              </div>
              <span class="shift-pct">{fmt_pct(row['current_pct'])}</span>
              <span class="shift-delta">{pct_sign}{delta_pct:.2f} pp vs base · {votes_sign}{fmt_int(delta_votes)} votos proj.</span>
              <div class="candidate-card__bar" style="margin-top:8px;">
                <span style="width:{row['current_pct']:.2f}%;"></span>
              </div>
            </article>
            """
        )

    return f"""
    <section class="stack">
      <div class="section-title">
        <p class="eyebrow">Movimiento de proyección</p>
        <h2>{escape(shift['label'])}</h2>
        <p>
          {escape(iso_to_local_label(shift['base_at']))} · {fmt_pct(shift['base_actas_pct'], 3)} de actas.
          Esto muestra cuánto se movió la estimación final al entrar nueva data.
        </p>
      </div>
      <div class="cards-grid shift-grid">
        {''.join(cards)}
      </div>
    </section>
    """


def render_department_table(payload: dict) -> str:
    headers = "".join(f"<th>{escape(cand)}</th><th>Votos</th>" for cand in payload["tracked"])
    rows = []
    for row in payload["department_rows"]:
        candidate_cells = []
        for candidate in row["candidates"]:
            candidate_cells.append(f"<td>{fmt_pct(candidate['pct'])}</td><td>{fmt_int(candidate['votes'])}</td>")
        rows.append(
            f"""
            <tr>
              <td>{escape(row['name'])}</td>
              <td>{fmt_pct(row['pct_actas'])}</td>
              <td>{fmt_int(row['emitidos_proj'])}</td>
              {''.join(candidate_cells)}
            </tr>
            """
        )
    return f"""
    <section>
      <div class="section-title">
        <p class="eyebrow">Territorio</p>
        <h2>Proyección por departamento</h2>
        <p class="section-note">Click en columna para ordenar. Votos proyectados al 100% de actas.</p>
      </div>
      <div class="table-wrap">
        <div class="table-inner">
          <table class="sortable">
            <thead>
              <tr>
                <th>Departamento</th>
                <th>% actas</th>
                <th>Emitidos proj.</th>
                {headers}
              </tr>
            </thead>
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
        </div>
      </div>
    </section>
    """


def render_territory_drilldown(payload: dict) -> str:
    drilldown = payload.get("territory_drilldown") or {}
    regions = drilldown.get("regions_top_remaining") or []
    if not regions:
        return ""

    left_label = escape(str(drilldown.get("left_label", "RLA")))
    right_label = escape(str(drilldown.get("right_label", "Sánchez")))

    region_rows = []
    for row in regions:
        gap_sign = "+" if row["gap_pp"] >= 0 else ""
        region_rows.append(
            f"""
            <tr>
              <td>{escape(str(row['name']))}</td>
              <td>{fmt_pct(row['pct_actas'], 3)}</td>
              <td>{fmt_int(row['remaining_votes'])}</td>
              <td>{fmt_pct(row['left_pct'])}</td>
              <td>{fmt_pct(row['right_pct'])}</td>
              <td>{gap_sign}{row['gap_pp']:.2f} pp</td>
            </tr>
            """
        )

    detail_cards = []
    for region in drilldown.get("region_details", []):
        province_rows = []
        for province in region["provinces"]:
            gap_sign = "+" if province["gap_pp"] >= 0 else ""
            province_rows.append(
                f"""
                <tr>
                  <td>{escape(str(province['name']))}</td>
                  <td>{fmt_int(province['remaining_votes'])}</td>
                  <td>{fmt_pct(province['left_pct'])}</td>
                  <td>{fmt_pct(province['right_pct'])}</td>
                  <td>{gap_sign}{province['gap_pp']:.2f} pp</td>
                </tr>
                """
            )
        detail_cards.append(
            f"""
            <article class="focus-card">
              <div class="focus-card__header">
                <div>
                  <p class="eyebrow">Región crítica</p>
                  <h3>{escape(str(region['name']))}</h3>
                </div>
                <span>{fmt_int(region['remaining_votes'])} faltantes</span>
              </div>
              <div class="focus-card__chips">
                <span class="focus-chip">{left_label} {fmt_pct(region['left_pct'])}</span>
                <span class="focus-chip">{right_label} {fmt_pct(region['right_pct'])}</span>
                <span class="focus-chip focus-chip--gap">Brecha {region['gap_pp']:+.2f} pp</span>
              </div>
              <div class="focus-table-shell">
                <div class="table-inner">
                  <table class="micro-table sortable">
                    <thead>
                      <tr>
                        <th>Provincia</th>
                        <th>Faltan</th>
                        <th>{left_label}</th>
                        <th>{right_label}</th>
                        <th>{right_label}-{left_label}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {''.join(province_rows)}
                    </tbody>
                  </table>
                </div>
              </div>
            </article>
            """
        )

    lima_rows = []
    for row in drilldown.get("lima_districts_top_remaining", []):
        gap_sign = "+" if row["gap_pp"] >= 0 else ""
        lima_rows.append(
            f"""
            <tr>
              <td>{escape(str(row['name']))}</td>
              <td>{fmt_pct(row['pct_actas'], 3)}</td>
              <td>{fmt_int(row['remaining_votes'])}</td>
              <td>{fmt_pct(row['left_pct'])}</td>
              <td>{fmt_pct(row['right_pct'])}</td>
              <td>{gap_sign}{row['gap_pp']:.2f} pp</td>
            </tr>
            """
        )

    foreign_rows = []
    for row in drilldown.get("foreign", []):
        gap_sign = "+" if row["gap_pp"] >= 0 else ""
        foreign_rows.append(
            f"""
            <tr>
              <td>{escape(str(row.get('continent', '')))}</td>
              <td>{escape(str(row['name']))}</td>
              <td>{fmt_pct(row['pct_actas'], 3)}</td>
              <td>{fmt_int(row['remaining_votes'])}</td>
              <td>{fmt_pct(row['left_pct'])}</td>
              <td>{fmt_pct(row['right_pct'])}</td>
              <td>{gap_sign}{row['gap_pp']:.2f} pp</td>
            </tr>
            """
        )

    return f"""
    <section>
      <div class="section-title">
        <p class="eyebrow">Territorio fino</p>
        <h2>Dónde está el remanente que define RLA vs Sánchez</h2>
        <p class="section-note">Regiones y provincias con más votos pendientes, más un zoom de Lima por distrito. La brecha se muestra como {right_label} menos {left_label}. Click en columna para ordenar.</p>
      </div>
      <div class="table-wrap">
        <div class="table-inner">
          <table class="sortable">
            <thead>
              <tr>
                <th>Región</th>
                <th>% actas</th>
                <th>Faltan</th>
                <th>{left_label}</th>
                <th>{right_label}</th>
                <th>{right_label}-{left_label}</th>
              </tr>
            </thead>
            <tbody>
              {''.join(region_rows)}
            </tbody>
          </table>
        </div>
      </div>
      <div class="focus-grid">
        {''.join(detail_cards)}
      </div>
      <div class="focus-lower-grid">
        <div class="table-wrap">
          <div class="focus-subtitle">
            <p class="eyebrow">Lima</p>
            <h3>Distritos con más votos faltantes</h3>
          </div>
          <div class="table-inner">
            <table class="sortable">
              <thead>
                <tr>
                  <th>Distrito</th>
                  <th>% actas</th>
                  <th>Faltan</th>
                  <th>{left_label}</th>
                  <th>{right_label}</th>
                  <th>{right_label}-{left_label}</th>
                </tr>
              </thead>
              <tbody>
                {''.join(lima_rows)}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div class="focus-foreign-stack">
        <div class="table-wrap">
          <div class="focus-subtitle">
            <div>
              <p class="eyebrow">Extranjero</p>
              <h3>Países con más votos faltantes</h3>
            </div>
          </div>
          <div class="table-inner">
            <table class="sortable">
              <thead>
                <tr>
                  <th>Continente</th>
                  <th>País</th>
                  <th>% actas</th>
                  <th>Faltan</th>
                  <th>{left_label}</th>
                  <th>{right_label}</th>
                  <th>{right_label}-{left_label}</th>
                </tr>
              </thead>
              <tbody>
                {''.join(foreign_rows)}
              </tbody>
            </table>
          </div>
          <p class="table-note">El faltante por país se estima con actas pendientes y votos por acta del propio país, suavizado con el promedio del exterior.</p>
        </div>
      </div>
    </section>
    """


def render_html(payload: dict) -> str:
    cards = "".join(
        render_candidate_card(candidate, position + 1)
        for position, candidate in enumerate(payload["tracked_projection"])
    )
    projection_rows = []
    for row in payload["projection_rows"][:10]:
        top2 = f"{row['top2_prob'] * 100:.1f}%"
        projection_rows.append(
            f"""
            <tr>
              <td>{escape(row['candidate'])}</td>
              <td class="num">{fmt_pct(row['actual_pct'])}</td>
              <td class="num">{fmt_pct(row['projected_pct'])}</td>
              <td class="num">{fmt_pct(row['ic_lo'])} — {fmt_pct(row['ic_hi'])}</td>
              <td class="num">{top2}</td>
            </tr>
            """
        )

    trajectory_chart = render_trajectory_chart(payload)
    actas_pct = float(payload.get("actas_pct", 0.0))

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="300">
  <title>Elecciones 2026 — Primera Vuelta</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@400;500;600;700&family=Nunito+Sans:ital,wght@0,300;0,400;0,600;0,700;0,800;1,400&display=swap');
    :root {{
      --bg: {PAPER};
      --surface: {CARD};
      --ink: {INK};
      --ink-2: #4D4A58;
      --ink-3: #908D9C;
      --border: #E2DDD5;
      --border-2: #C5C0B8;
      --red: {ACCENT};
      --shadow-sm: 0 1px 3px rgba(0,0,0,0.07), 0 4px 12px rgba(0,0,0,0.05);
      --shadow-md: 0 2px 8px rgba(0,0,0,0.07), 0 12px 28px rgba(0,0,0,0.06);
      --r-sm: 8px; --r-md: 14px; --r-lg: 22px; --r-xl: 32px;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--ink);
      font-family: "Nunito Sans", "Segoe UI", sans-serif;
      font-size: 15px;
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{
      width: min(1280px, calc(100vw - 40px));
      margin: 28px auto 80px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }}
    /* ── HERO ──────────────────────────────────── */
    .hero {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-xl);
      box-shadow: var(--shadow-md);
      padding: 36px 40px;
      position: relative;
      overflow: hidden;
    }}
    .hero::before {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        radial-gradient(ellipse 55% 45% at 95% 5%, rgba(191,48,48,0.06), transparent),
        radial-gradient(ellipse 45% 35% at 5% 95%, rgba(30,63,164,0.06), transparent);
      pointer-events: none;
    }}
    .hero__eyebrow {{
      font-size: 10px;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--red);
      font-weight: 700;
      font-family: "JetBrains Mono", monospace;
      margin-bottom: 10px;
    }}
    .hero__title {{
      font-family: "DM Serif Display", Georgia, serif;
      font-size: clamp(2.4rem, 4.5vw, 4.2rem);
      font-weight: 400;
      line-height: 1.06;
      letter-spacing: -0.015em;
      color: var(--ink);
      margin-bottom: 10px;
    }}
    .hero__sub {{
      color: var(--ink-2);
      font-size: 1rem;
      max-width: 64ch;
      margin-bottom: 24px;
      line-height: 1.6;
    }}
    .hero__metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 22px;
    }}
    .hero__metric {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--r-md);
      padding: 11px 16px;
      min-width: 130px;
    }}
    .hero__metric-label {{
      font-size: 9px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--ink-3);
      font-family: "JetBrains Mono", monospace;
      font-weight: 600;
      margin-bottom: 4px;
    }}
    .hero__metric-value {{
      font-family: "JetBrains Mono", monospace;
      font-weight: 700;
      font-size: 0.95rem;
      color: var(--ink);
      line-height: 1.2;
      word-break: break-all;
    }}
    .hero__progress-label {{
      display: flex;
      justify-content: space-between;
      font-size: 0.78rem;
      color: var(--ink-2);
      margin-bottom: 5px;
      font-family: "JetBrains Mono", monospace;
      font-weight: 600;
    }}
    .hero__progress-bar {{
      height: 10px;
      background: var(--border);
      border-radius: 99px;
      overflow: hidden;
    }}
    .hero__progress-bar span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, #1e3fa4 0%, #4f72e0 100%);
      border-radius: 99px;
    }}
    /* ── SECTION TITLE ─────────────────────────── */
    .section-title {{
      margin-bottom: 16px;
    }}
    .eyebrow {{
      font-size: 9px;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--ink-3);
      font-family: "JetBrains Mono", monospace;
      font-weight: 700;
      display: block;
      margin-bottom: 5px;
    }}
    h2 {{
      font-family: "DM Serif Display", Georgia, serif;
      font-size: 1.55rem;
      font-weight: 400;
      color: var(--ink);
      line-height: 1.18;
      margin-bottom: 4px;
    }}
    h3 {{
      font-size: 0.875rem;
      font-weight: 700;
      color: var(--ink);
    }}
    .section-note {{
      font-size: 0.8rem;
      color: var(--ink-3);
      margin-top: 2px;
      line-height: 1.5;
    }}
    /* ── CANDIDATE CARDS ───────────────────────── */
    .candidates-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .candidate-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-lg);
      box-shadow: var(--shadow-sm);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      position: relative;
      overflow: hidden;
    }}
    .candidate-card::before {{
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: var(--c-color);
    }}
    .candidate-card__top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .candidate-card__rank {{
      font-family: "JetBrains Mono", monospace;
      font-size: 9px;
      letter-spacing: 0.14em;
      color: var(--ink-3);
      font-weight: 700;
    }}
    .candidate-card__name {{
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--ink);
      line-height: 1.3;
    }}
    .candidate-card__pcts {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }}
    .candidate-card__big {{
      font-family: "JetBrains Mono", monospace;
      font-size: 1.55rem;
      font-weight: 700;
      color: var(--ink);
      line-height: 1;
      display: block;
    }}
    .candidate-card__big--proj {{
      color: var(--c-color);
    }}
    .candidate-card__small {{
      font-size: 0.65rem;
      color: var(--ink-3);
      display: block;
      margin-top: 3px;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      font-family: "JetBrains Mono", monospace;
    }}
    .candidate-card__bar {{
      height: 5px;
      border-radius: 99px;
      background: var(--border);
      overflow: hidden;
    }}
    .candidate-card__bar span {{
      display: block;
      height: 100%;
      background: var(--c-color);
      opacity: 0.65;
      border-radius: 99px;
    }}
    .candidate-card__ic {{
      font-size: 0.68rem;
      color: var(--ink-3);
      font-family: "JetBrains Mono", monospace;
    }}
    .candidate-card__top2 {{ margin-top: auto; }}
    .candidate-card__top2-label {{
      font-size: 0.65rem;
      color: var(--ink-3);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      font-family: "JetBrains Mono", monospace;
      margin-bottom: 4px;
    }}
    .candidate-card__top2-bar {{
      height: 5px;
      border-radius: 99px;
      background: var(--border);
      overflow: hidden;
    }}
    .candidate-card__top2-bar span {{
      display: block;
      height: 100%;
      background: var(--c-color);
      border-radius: 99px;
    }}
    .candidate-card__top2-pct {{
      font-family: "JetBrains Mono", monospace;
      font-weight: 700;
      font-size: 0.78rem;
      color: var(--c-color);
      margin-top: 4px;
    }}
    /* ── BATTLE GAUGE ──────────────────────────── */
    .battle-gauge {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-lg);
      box-shadow: var(--shadow-sm);
      padding: 28px 32px;
    }}
    .battle-gauge__header {{
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      gap: 16px;
      align-items: end;
      margin-bottom: 20px;
    }}
    .battle-gauge__cand-name {{
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 4px;
      font-family: "JetBrains Mono", monospace;
    }}
    .battle-gauge__cand-pct {{
      font-family: "JetBrains Mono", monospace;
      font-weight: 700;
      font-size: 2.4rem;
      line-height: 1;
    }}
    .battle-gauge__cand-votes {{
      font-family: "JetBrains Mono", monospace;
      font-size: 0.78rem;
      font-weight: 600;
      opacity: 0.75;
      margin-top: 2px;
    }}
    .battle-gauge__cand-full {{
      font-size: 0.72rem;
      color: var(--ink-3);
      margin-top: 3px;
    }}
    .battle-gauge__vs {{
      font-family: "DM Serif Display", serif;
      font-size: 1.4rem;
      color: var(--ink-3);
      text-align: center;
      padding-bottom: 4px;
    }}
    .battle-gauge__bar {{
      height: 14px;
      border-radius: 99px;
      overflow: hidden;
      display: flex;
      background: var(--border);
    }}
    .battle-gauge__bar-left, .battle-gauge__bar-right {{
      height: 100%;
    }}
    .battle-gauge__labels {{
      display: flex;
      justify-content: space-between;
      margin-top: 6px;
      font-size: 0.72rem;
      color: var(--ink-3);
      font-family: "JetBrains Mono", monospace;
    }}
    .battle-gauge__diff {{
      text-align: center;
      margin-top: 14px;
      font-size: 0.875rem;
      color: var(--ink-2);
    }}
    .battle-gauge__diff strong {{
      font-family: "JetBrains Mono", monospace;
    }}
    /* ── CARTOGRAM ────────────────────────────── */
    .carto-body {{
      display: flex;
      gap: 32px;
      align-items: flex-start;
      flex-wrap: wrap;
    }}
    .carto-grid {{
      display: grid;
      grid-template-columns: repeat({CARTO_COLS}, 72px);
      grid-template-rows: repeat({CARTO_ROWS}, 72px);
      gap: 5px;
      flex-shrink: 0;
    }}
    .carto-cell {{ /* empty slot */ }}
    .carto-tile {{
      border-radius: var(--r-sm);
      border: 2px solid var(--tile-color, var(--border));
      background: color-mix(in srgb, var(--tile-color, #888) 12%, white);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 4px 3px 6px;
      cursor: default;
      transition: transform 0.12s ease, box-shadow 0.12s ease;
      gap: 1px;
      overflow: hidden;
      position: relative;
    }}
    .carto-tile:hover {{
      transform: scale(1.12);
      box-shadow: 0 6px 20px rgba(0,0,0,0.20);
      z-index: 10;
    }}
    .carto-tile--nd {{
      background: #EEEAE3;
      border-color: #D5D0C8;
    }}
    .carto-tile__abbr {{
      font-family: "JetBrains Mono", monospace;
      font-size: 0.6rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      color: color-mix(in srgb, var(--tile-color, #888) 80%, #111);
      line-height: 1;
    }}
    .carto-tile__pct {{
      font-family: "JetBrains Mono", monospace;
      font-size: 0.75rem;
      font-weight: 700;
      color: color-mix(in srgb, var(--tile-color, #888) 80%, #000);
      line-height: 1;
    }}
    .carto-tile__party {{
      font-size: 0.52rem;
      font-weight: 700;
      color: color-mix(in srgb, var(--tile-color, #888) 65%, #555);
      text-align: center;
    }}
    .carto-tile__actas {{
      position: absolute;
      bottom: 0; left: 0; right: 0;
      height: 4px;
      background: rgba(0,0,0,0.10);
    }}
    .carto-tile__actas span {{
      display: block;
      height: 100%;
      background: var(--tile-color, #888);
      opacity: 0.55;
    }}
    .carto-sidebar {{
      padding-top: 4px;
    }}
    .carto-legend {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .carto-legend__item {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--ink-2);
    }}
    .carto-legend__dot {{
      width: 12px;
      height: 12px;
      border-radius: 3px;
      flex-shrink: 0;
    }}
    .carto-legend__full {{
      font-size: 0.72rem;
      color: var(--ink-3);
      font-weight: 400;
    }}
    .carto-legend__note {{
      font-size: 0.75rem;
      color: var(--ink-3);
      line-height: 1.45;
      max-width: 22ch;
    }}
    /* ── BLOCKS ───────────────────────────────── */
    .blocks-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .block-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-lg);
      box-shadow: var(--shadow-sm);
      padding: 20px;
    }}
    .block-card__header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }}
    .block-card__title {{ font-size: 0.875rem; font-weight: 700; }}
    .block-card__total {{
      font-family: "JetBrains Mono", monospace;
      font-size: 0.75rem;
      color: var(--ink-3);
    }}
    .block-bar {{ margin-bottom: 10px; }}
    .block-bar__head {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
      font-size: 0.78rem;
    }}
    .block-bar__name {{ font-weight: 600; color: var(--ink-2); }}
    .block-bar__pct {{
      font-family: "JetBrains Mono", monospace;
      font-weight: 700;
    }}
    .block-bar__track {{
      height: 7px;
      border-radius: 99px;
      background: var(--border);
      overflow: hidden;
    }}
    .block-bar__track span {{
      display: block;
      height: 100%;
      border-radius: 99px;
    }}
    /* ── TABLES ───────────────────────────────── */
    .table-wrap {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-lg);
      box-shadow: var(--shadow-sm);
      overflow: hidden;
    }}
    .table-inner {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 520px;
      font-size: 0.845rem;
    }}
    table.sortable thead th {{ cursor: pointer; user-select: none; }}
    table.sortable thead th:hover {{ background: var(--bg); color: var(--ink); }}
    thead th {{
      padding: 11px 14px;
      text-align: left;
      font-size: 0.68rem;
      font-family: "JetBrains Mono", monospace;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--ink-3);
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      background: var(--surface);
      transition: color 0.1s;
    }}
    thead th.th-asc::after {{ content: " ↑"; color: var(--ink); }}
    thead th.th-desc::after {{ content: " ↓"; color: var(--ink); }}
    tbody td {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      color: var(--ink);
      white-space: nowrap;
      vertical-align: middle;
    }}
    tbody tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover {{ background: color-mix(in srgb, var(--bg) 70%, white); }}
    td.num {{
      font-family: "JetBrains Mono", monospace;
      font-size: 0.8rem;
      text-align: right;
      color: var(--ink-2);
    }}
    .table-note {{
      padding: 10px 14px;
      font-size: 0.75rem;
      color: var(--ink-3);
      border-top: 1px solid var(--border);
    }}
    /* ── TRAJECTORY ───────────────────────────── */
    .traj-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-lg);
      box-shadow: var(--shadow-sm);
      padding: 22px;
    }}
    .traj-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }}
    .traj-legend {{ display: flex; flex-wrap: wrap; gap: 8px 14px; }}
    .traj-legend__item {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.8rem;
      font-weight: 600;
    }}
    .traj-legend__swatch {{ width: 18px; height: 3px; border-radius: 99px; }}
    .traj-legend__hint {{ font-size: 0.75rem; color: var(--ink-3); }}
    .traj-badge {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--r-md);
      padding: 12px 16px;
      text-align: right;
      min-width: 150px;
    }}
    .traj-badge__pct {{
      font-family: "JetBrains Mono", monospace;
      font-weight: 700;
      font-size: 1.9rem;
      color: var(--ink);
      line-height: 1;
      display: block;
    }}
    .traj-badge__label {{
      font-size: 0.68rem;
      color: var(--ink-3);
      margin-top: 4px;
      font-family: "JetBrains Mono", monospace;
    }}
    .traj-shell {{
      display: grid;
      grid-template-columns: 1fr 190px;
      gap: 14px;
      align-items: stretch;
    }}
    .traj-canvas-wrap {{ min-height: 360px; }}
    canvas#trajectoryChart {{ width: 100% !important; height: 360px; display: block; }}
    .traj-endpoints {{ display: flex; flex-direction: column; gap: 8px; }}
    .traj-endpoint {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--r-md);
      padding: 11px 13px;
    }}
    .traj-endpoint__head {{
      display: flex;
      align-items: center;
      gap: 7px;
      margin-bottom: 6px;
    }}
    .traj-endpoint__dot {{
      width: 9px; height: 9px; border-radius: 50%;
    }}
    .traj-endpoint__code {{
      font-family: "JetBrains Mono", monospace;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.08em;
    }}
    .traj-endpoint__pct {{
      font-family: "JetBrains Mono", monospace;
      font-weight: 700;
      font-size: 1.2rem;
      display: block;
      line-height: 1;
    }}
    .traj-endpoint__name {{
      font-size: 0.68rem;
      color: var(--ink-3);
      margin-top: 3px;
      line-height: 1.3;
    }}
    /* ── DELTA / INCREMENTALS ─────────────────── */
    .delta-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .delta-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-lg);
      box-shadow: var(--shadow-sm);
      padding: 18px;
    }}
    .delta-card__header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }}
    .delta-card__title {{ font-size: 0.875rem; font-weight: 700; }}
    .delta-card__count {{
      font-family: "JetBrains Mono", monospace;
      font-size: 0.75rem;
      color: var(--ink-3);
    }}
    .delta-line {{ margin-bottom: 9px; }}
    .delta-line__head {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
      font-size: 0.78rem;
    }}
    .delta-line__name {{ font-weight: 600; color: var(--ink-2); }}
    .delta-line__pct {{ font-family: "JetBrains Mono", monospace; font-weight: 700; }}
    .delta-bar {{ height: 6px; border-radius: 99px; background: var(--border); overflow: hidden; }}
    .delta-bar span {{ display: block; height: 100%; border-radius: 99px; }}
    /* ── SHIFT ────────────────────────────────── */
    .shift-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
    }}
    .shift-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-lg);
      box-shadow: var(--shadow-sm);
      padding: 18px;
    }}
    .shift-card__header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .shift-card__name {{ font-size: 0.78rem; font-weight: 700; max-width: 16ch; line-height: 1.3; }}
    .shift-pill {{
      font-family: "JetBrains Mono", monospace;
      font-size: 0.64rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding: 4px 8px;
      border-radius: 6px;
      background: color-mix(in srgb, var(--c-color, #888) 14%, white);
      color: var(--c-color, #888);
      white-space: nowrap;
    }}
    .shift-pct {{
      font-family: "JetBrains Mono", monospace;
      font-weight: 700;
      font-size: 1.35rem;
      display: block;
      margin-bottom: 3px;
    }}
    .shift-delta {{
      font-family: "JetBrains Mono", monospace;
      font-size: 0.75rem;
      color: var(--ink-3);
    }}
    /* ── REMAINING ────────────────────────────── */
    .remaining-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .remaining-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-lg);
      box-shadow: var(--shadow-sm);
      padding: 18px;
    }}
    .remaining-card__header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 8px;
    }}
    .remaining-card__title {{ font-size: 0.875rem; font-weight: 700; }}
    .remaining-card__count {{
      font-family: "JetBrains Mono", monospace;
      font-size: 0.82rem;
      font-weight: 700;
    }}
    .remaining-bar {{
      height: 8px;
      border-radius: 99px;
      background: var(--border);
      overflow: hidden;
      margin-bottom: 8px;
    }}
    .remaining-bar span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, #1e3fa4, #4f72e0);
      border-radius: 99px;
    }}
    .remaining-card__note {{ font-size: 0.75rem; color: var(--ink-3); }}
    /* ── FOCUS CARDS ─────────────────────────── */
    .focus-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .focus-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-lg);
      box-shadow: var(--shadow-sm);
      padding: 18px;
    }}
    .focus-card__header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 10px;
    }}
    .focus-card__header span {{
      font-family: "JetBrains Mono", monospace;
      font-size: 0.78rem;
      color: var(--ink-3);
    }}
    .focus-subtitle {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin: 6px 6px 12px;
    }}
    .focus-card__chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 12px;
    }}
    .focus-chip {{
      padding: 5px 10px;
      border-radius: 99px;
      background: var(--bg);
      border: 1px solid var(--border);
      font-size: 0.72rem;
      font-weight: 700;
      font-family: "JetBrains Mono", monospace;
    }}
    .focus-chip--gap {{
      background: rgba(191,48,48,0.07);
      border-color: rgba(191,48,48,0.2);
      color: var(--red);
    }}
    .focus-table-shell, .focus-lower-grid, .focus-foreign-stack {{ overflow: auto; }}
    .micro-table th, .micro-table td {{ font-size: 0.76rem; padding: 7px 10px; }}
    .table-shell__note {{
      padding: 8px 12px;
      font-size: 0.75rem;
      color: var(--ink-3);
    }}
    /* ── PROJECTION SHELL ─────────────────────── */
    .projection-shell {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-lg);
      box-shadow: var(--shadow-sm);
      overflow: hidden;
    }}
    /* ── MISC ─────────────────────────────────── */
    .empty-state {{
      padding: 20px;
      color: var(--ink-3);
      font-size: 0.875rem;
    }}
    .footer {{
      border-top: 1px solid var(--border);
      padding-top: 18px;
      color: var(--ink-3);
      font-size: 0.8rem;
      line-height: 1.55;
    }}
    /* ── RESPONSIVE ───────────────────────────── */
    @media (max-width: 900px) {{
      .candidates-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .blocks-grid, .delta-grid, .remaining-grid {{ grid-template-columns: 1fr; }}
      .traj-shell {{ grid-template-columns: 1fr; }}
      .traj-endpoints {{ flex-direction: row; flex-wrap: wrap; }}
      .focus-grid {{ grid-template-columns: 1fr; }}
      .carto-grid {{
        grid-template-columns: repeat({CARTO_COLS}, 52px);
        grid-template-rows: repeat({CARTO_ROWS}, 52px);
      }}
      .hero {{ padding: 24px 20px; }}
      .battle-gauge {{ padding: 20px; }}
    }}
    @media (max-width: 600px) {{
      .candidates-grid {{ grid-template-columns: 1fr; }}
      .carto-body {{ flex-direction: column; }}
      .carto-grid {{
        grid-template-columns: repeat({CARTO_COLS}, 44px);
        grid-template-rows: repeat({CARTO_ROWS}, 44px);
      }}
      .battle-gauge__cand-pct {{ font-size: 1.8rem; }}
    }}
  </style>
</head>
<body>
  <main class="page">

    <!-- ── HERO ────────────────────────────────── -->
    <section class="hero">
      <div class="hero__eyebrow">Elecciones Generales 2026 · Primera Vuelta</div>
      <h1 class="hero__title">Estimación de resultados</h1>
      <p class="hero__sub">Conteo en vivo con proyecciones bayesianas al cierre. Actualización automática cada 5 minutos.</p>
      <div class="hero__metrics">
        <div class="hero__metric">
          <div class="hero__metric-label">Último corte</div>
          <div class="hero__metric-value">{escape(iso_to_local_label(payload['snapshot_at']))}</div>
        </div>
        <div class="hero__metric">
          <div class="hero__metric-label">Actualizado ONPE</div>
          <div class="hero__metric-value">{escape(epoch_millis_to_local_label(payload.get('onpe_updated_at')))}</div>
        </div>
        <div class="hero__metric">
          <div class="hero__metric-label">Actas procesadas</div>
          <div class="hero__metric-value">{fmt_int(payload['actas_contadas'])} / {fmt_int(payload['actas_totales'])}</div>
        </div>
        <div class="hero__metric">
          <div class="hero__metric-label">Votos válidos</div>
          <div class="hero__metric-value">{fmt_int(payload['votos_validos'])}</div>
        </div>
      </div>
      <div>
        <div class="hero__progress-label">
          <span>Avance nacional</span>
          <span>{fmt_pct(actas_pct, 2)}</span>
        </div>
        <div class="hero__progress-bar">
          <span style="width:{actas_pct:.2f}%"></span>
        </div>
      </div>
    </section>

    <!-- ── CANDIDATES ──────────────────────────── -->
    <section>
      <div class="section-title">
        <span class="eyebrow">Proyección nacional</span>
        <h2>Candidatos en la zona crítica</h2>
        <p class="section-note">Proyección bayesiana Dirichlet-Multinomial al 100% de actas. IC = intervalo de confianza 95%.</p>
      </div>
      <div class="candidates-grid">
        {cards}
      </div>
    </section>

    <!-- ── TRAJECTORY ──────────────────────────── -->
    {trajectory_chart}

    <!-- ── BATTLE GAUGE ────────────────────────── -->
    {render_battle_gauge(payload)}

    <!-- ── MAP ─────────────────────────────────── -->
    {render_cartogram(payload)}

    <!-- ── BLOCKS ──────────────────────────────── -->
    <section>
      <div class="section-title">
        <span class="eyebrow">Bloques territoriales</span>
        <h2>Proyección por bloque geográfico</h2>
      </div>
      <div class="blocks-grid">
        {render_block_summary(payload)}
      </div>
    </section>

    <!-- ── REMAINING ───────────────────────────── -->
    <section>
      <div class="section-title">
        <span class="eyebrow">Votos pendientes</span>
        <h2>Bolsones por contar</h2>
      </div>
      <div class="remaining-grid">
        {render_remaining_blocks(payload)}
      </div>
    </section>

    <!-- ── TERRITORY DRILLDOWN ─────────────────── -->
    {render_territory_drilldown(payload)}

    <!-- ── SHIFT ───────────────────────────────── -->
    {render_projection_shift_section(payload['projection_shift']['latest'])}
    {render_projection_shift_section(payload['projection_shift']['trend'])}

    <!-- ── DELTA ───────────────────────────────── -->
    {render_delta_section(payload['delta_info']['latest'], 'Última tanda de actas vs. snapshot previo')}
    {render_delta_section(payload['delta_info']['trend'], 'Tendencia acumulada de los últimos cortes')}

    <!-- ── RANKING TABLE ───────────────────────── -->
    <section>
      <div class="section-title">
        <span class="eyebrow">Ranking general</span>
        <h2>Proyección nacional — todos los candidatos</h2>
        <p class="section-note">Click en columna para ordenar.</p>
      </div>
      <div class="table-wrap">
        <div class="table-inner">
          <table class="sortable">
            <thead>
              <tr>
                <th>Candidato</th>
                <th>Actual</th>
                <th>Proyección</th>
                <th>IC 95%</th>
                <th>Prob. 2da vuelta</th>
              </tr>
            </thead>
            <tbody>
              {''.join(projection_rows)}
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- ── DEPARTMENT TABLE ────────────────────── -->
    {render_department_table(payload)}

    <!-- ── FOOTER ──────────────────────────────── -->
    <footer class="footer">
      Estimación referencial — Elecciones Generales 2026, primera vuelta.<br>
      Generado el {escape(iso_to_local_label(payload['generated_at']))} · Modelo bayesiano Dirichlet-Multinomial por distrito.
    </footer>

  </main>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <script>
    /* ── Sortable tables ── */
    document.querySelectorAll("table.sortable").forEach(table => {{
      const ths = table.querySelectorAll("thead th");
      ths.forEach((th, idx) => {{
        th.addEventListener("click", () => {{
          const asc = th.dataset.dir !== "asc";
          ths.forEach(h => {{ h.classList.remove("th-asc", "th-desc"); delete h.dataset.dir; }});
          th.dataset.dir = asc ? "asc" : "desc";
          th.classList.add(asc ? "th-asc" : "th-desc");
          const tbody = table.querySelector("tbody");
          const rows = [...tbody.querySelectorAll("tr")];
          rows.sort((a, b) => {{
            const aT = (a.cells[idx]?.textContent || "").trim();
            const bT = (b.cells[idx]?.textContent || "").trim();
            const aN = parseFloat(aT.replace(/[^\d.,-]/g, "").replace(",", ""));
            const bN = parseFloat(bT.replace(/[^\d.,-]/g, "").replace(",", ""));
            if (!isNaN(aN) && !isNaN(bN)) return asc ? aN - bN : bN - aN;
            return asc ? aT.localeCompare(bT, "es") : bT.localeCompare(aT, "es");
          }});
          rows.forEach(r => tbody.appendChild(r));
        }});
      }});
    }});

    /* ── Trajectory chart ── */
    (() => {{
      const el = document.getElementById("trajectory-data");
      const canvas = document.getElementById("trajectoryChart");
      if (!el || !canvas || !window.Chart) return;

      const trajectory = JSON.parse(el.textContent);
      const labels = {json.dumps(TRACK_SHORT_LABELS, ensure_ascii=False)};
      const datasets = [];

      for (const line of trajectory.series || []) {{
        const actual = (line.actual || []).map(p => ({{ x: p.x, y: p.y, label: p.label || "" }}));
        if (!actual.length) continue;
        const latest = actual[actual.length - 1];
        const short = labels[line.candidate] || line.candidate;
        datasets.push({{
          label: short,
          data: actual,
          borderColor: line.color,
          backgroundColor: line.color,
          pointBackgroundColor: line.color,
          pointBorderColor: "rgba(245,242,235,0.9)",
          pointBorderWidth: 1.5,
          pointRadius: 3.5,
          pointHoverRadius: 5,
          borderWidth: 2.5,
          tension: 0.3
        }});
        datasets.push({{
          label: `${{short}} cierre`,
          data: [
            {{ x: latest.x, y: latest.y, label: latest.label || "" }},
            {{ x: line.projected.x, y: line.projected.y, label: "Estimación al 100%" }}
          ],
          borderColor: line.color,
          backgroundColor: line.color,
          pointBackgroundColor: [line.color, line.color],
          pointBorderColor: ["transparent", "rgba(28,26,32,0.3)"],
          pointBorderWidth: [0, 2],
          pointRadius: [0, 6],
          pointHoverRadius: [0, 6],
          borderWidth: 1.8,
          tension: 0,
          borderDash: [5, 7]
        }});
      }}

      const xMax = Math.max(100.8, Number(trajectory.bounds?.x_max || 100) + 0.8);
      new Chart(canvas.getContext("2d"), {{
        type: "line",
        data: {{ datasets }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          interaction: {{ mode: "nearest", intersect: false }},
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              backgroundColor: "rgba(28,26,32,0.94)",
              titleFont: {{ family: "JetBrains Mono, monospace", size: 10 }},
              bodyFont: {{ family: "Nunito Sans, sans-serif", size: 12 }},
              callbacks: {{
                title: items => items?.[0]?.raw?.label || "Punto",
                label: item => `${{item.dataset.label}}: ${{item.parsed.y.toFixed(2)}}%`
              }}
            }}
          }},
          scales: {{
            x: {{
              type: "linear",
              min: trajectory.bounds?.x_min,
              max: xMax,
              grid: {{ color: "rgba(28,26,32,0.07)", drawBorder: false }},
              border: {{ display: false }},
              ticks: {{ color: "rgba(28,26,32,0.45)", stepSize: 5, font: {{ family: "JetBrains Mono, monospace", size: 10 }}, callback: v => `${{v}}%` }}
            }},
            y: {{
              min: trajectory.bounds?.y_min,
              max: trajectory.bounds?.y_max,
              grid: {{ color: "rgba(28,26,32,0.07)", drawBorder: false }},
              border: {{ display: false }},
              ticks: {{ color: "rgba(28,26,32,0.45)", font: {{ family: "JetBrains Mono, monospace", size: 10 }}, callback: v => `${{Number(v).toFixed(2)}}%` }}
            }}
          }},
          elements: {{ line: {{ capBezierPoints: true }} }}
        }}
      }});
    }})();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    op.configure_api(api_base=args.api_base, election_id=args.election_id, referer=args.referer)
    state_path = Path(args.state_file) if args.state_file else op.default_state_path_for_level(args.geo_level)
    html_out = Path(args.html_out).resolve()
    json_out = Path(args.json_out).resolve()
    onpe_updated_at_millis = fetch_onpe_updated_at_millis()
    national_vote_totals = fetch_national_vote_totals()
    department_name_map = fetch_department_names()

    regiones = op.cargar_unidades(args.geo_level, max_workers=max(args.workers, 1))
    province_units = regiones if args.geo_level == "province" else op.cargar_unidades("province", max_workers=max(args.workers, 1))
    district_units = regiones if args.geo_level == "district" else op.cargar_unidades("district", max_workers=max(args.workers, 1))
    tracked = resolve_tracked(regiones, args.track)

    state, current_snapshot, changed = ensure_snapshot(
        state_path=state_path,
        regiones=regiones,
        history_size=max(args.history_size, 2),
    )

    late_bias_model = None
    if not args.no_late_bias:
        late_bias_model = op.build_late_bias_model(
            regiones=regiones,
            history_snapshots=state.get("snapshots", []),
            current_snapshot=current_snapshot,
            window=max(args.late_bias_window, 2),
            strength=op.clip01(args.late_bias_strength),
            recency_decay=op.clip01(args.late_bias_recency),
        )

    rng = np.random.default_rng(args.seed)
    universe, counted, projected, pct_draws = op.proyectar(
        regiones,
        draws=args.draws,
        alpha_prior=args.alpha,
        swing=args.swing,
        late_bias_model=late_bias_model,
        rng=rng,
    )

    projection_rows = current_probs(universe, counted, pct_draws)
    dept_rows, dept_blocks = department_projections(
        regiones=regiones,
        universe=universe,
        tracked=tracked,
        alpha_prior=args.alpha,
        late_bias_model=late_bias_model,
        department_name_map=department_name_map,
    )
    delta_info = latest_delta_blocks(state, args.trend_window, tracked)
    current_projection_point = projection_point_from_blocks(dept_blocks, tracked)
    projection_shift = build_projection_shift(
        state=state,
        tracked=tracked,
        current_projection_point=current_projection_point,
        alpha_prior=args.alpha,
        trend_window=args.trend_window,
        late_bias_window=args.late_bias_window,
        late_bias_strength=0.0 if args.no_late_bias else args.late_bias_strength,
        late_bias_recency=args.late_bias_recency,
    )
    remaining_info = remaining_blocks(regiones)
    territory_drilldown = build_territory_drilldown(
        province_units=province_units,
        district_units=district_units,
        department_name_map=department_name_map,
    )

    payload = payload_to_jsonable(
        projection_rows=projection_rows,
        tracked=tracked,
        dept_rows=dept_rows,
        dept_blocks=dept_blocks,
        territory_drilldown=territory_drilldown,
        current_snapshot=current_snapshot,
        onpe_updated_at_millis=onpe_updated_at_millis,
        state=state,
        delta_info=delta_info,
        projection_shift=projection_shift,
        remaining_info=remaining_info,
        late_bias_model=late_bias_model,
        changed=changed,
        regiones=regiones,
        national_vote_totals=national_vote_totals,
    )

    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    html_out.write_text(render_html(payload))

    print(f"Dashboard HTML actualizado en {html_out}")
    print(f"Datos JSON actualizados en {json_out}")


if __name__ == "__main__":
    main()
