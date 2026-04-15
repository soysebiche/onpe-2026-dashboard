from __future__ import annotations

import argparse
import html
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path

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
DRILLDOWN_LEFT = "RENOVACIÓN POPULAR"
DRILLDOWN_RIGHT = "JUNTOS POR EL PERÚ"
DRILLDOWN_LABELS = {
    DRILLDOWN_LEFT: "RLA",
    DRILLDOWN_RIGHT: "Sánchez",
}
LIMA_PROVINCE_UBIGEO = 140100
PAPER = "#f8fafc"
INK = "#0f172a"
ACCENT = "#1e40af"
SOFT = "#dbe7ff"
CARD = "#ffffff"
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
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def epoch_millis_to_local_label(value: int | float | None) -> str:
    if not value:
        return "sin dato"
    dt = datetime.fromtimestamp(float(value) / 1000.0).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def fetch_onpe_updated_at_millis() -> int | None:
    candidates = []
    for params in (
        {"tipoFiltro": "ambito_geografico", "idAmbitoGeografico": 1},
        {"tipoFiltro": "ambito_geografico", "idAmbitoGeografico": 2},
    ):
        totals = op.fetch_totales(**params) or {}
        value = totals.get("fechaActualizacion")
        if value:
            candidates.append(int(value))
    return max(candidates) if candidates else None


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
) -> dict:
    current_votes_by_candidate: dict[str, int] = {}
    for region in current_snapshot.get("regions", {}).values():
        for candidate, votes in region.get("votos_por_candidato", {}).items():
            current_votes_by_candidate[candidate] = current_votes_by_candidate.get(candidate, 0) + int(votes)

    def current_comparison_row(left: str, right: str) -> dict | None:
        if left not in current_votes_by_candidate or right not in current_votes_by_candidate:
            return None
        total_validos = int(current_snapshot.get("votos_validos", 0))
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
        "generated_at": datetime.now().astimezone().isoformat(),
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
    return f"""
    <article class="candidate-card">
      <div class="candidate-card__rank">#{position}</div>
      <div>
        <p class="eyebrow">Proyección nacional</p>
        <h3>{escape(candidate["candidate"])}</h3>
      </div>
      <div class="candidate-card__numbers">
        <div>
          <span class="value">{fmt_pct(candidate["projected_pct"])}</span>
          <span class="label">mediana</span>
        </div>
        <div>
          <span class="value">{fmt_int(candidate["projected_votes"])}</span>
          <span class="label">votos proj.</span>
        </div>
      </div>
      <div class="candidate-card__range">
        <div class="candidate-card__bar">
          <span style="width:{candidate['projected_pct']:.2f}%; background:{color};"></span>
        </div>
        <div class="candidate-card__meta">
          <span>Actual {fmt_pct(candidate['actual_pct'])}</span>
          <span>IC 95% {fmt_pct(candidate['ic_lo'])} - {fmt_pct(candidate['ic_hi'])}</span>
        </div>
      </div>
    </article>
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
                <h3>{escape(row['label'])}</h3>
                <span>{fmt_int(row['remaining_votes'])} votos</span>
              </div>
              <div class="remaining-card__bar"><span style="width:{width:.2f}%"></span></div>
              <p>Falta por contar aproximadamente {fmt_pct(row['remaining_pct'])} del bloque.</p>
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
                <div class="battle-line">
                  <div class="battle-line__head">
                    <span>{escape(candidate['candidate'])}</span>
                    <span>{fmt_pct(candidate['pct'])}</span>
                  </div>
                  <div class="battle-line__bar"><span style="width:{candidate['pct']:.2f}%; background:{color};"></span></div>
                </div>
                """
            )
        blocks.append(
            f"""
            <article class="battle-card">
              <div class="battle-card__header">
                <h3>{escape(block['label'])}</h3>
                <span>{fmt_int(block['emitidos_proj'])} emitidos proj.</span>
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
            <div class="trajectory-legend__item">
              <span class="trajectory-legend__swatch" style="--swatch:{line["color"]};"></span>
              <span class="trajectory-legend__code">{escape(short_label)}</span>
              <span class="trajectory-legend__name">{escape(line["candidate"])}</span>
            </div>
            '''
        )
        endpoint_cards.append(
            f"""
            <article class="trajectory-endpoint">
              <div class="trajectory-endpoint__head">
                <span class="trajectory-endpoint__dot" style="--swatch:{line['color']};"></span>
                <span class="trajectory-endpoint__code">{escape(short_label)}</span>
              </div>
              <strong>{line['projected']['y']:.2f}%</strong>
              <small>{escape(line['candidate'])}</small>
            </article>
            """
        )

    latest_pct = payload["actas_pct"]
    trajectory_json = json.dumps(trajectory, ensure_ascii=False).replace("</", "<\\/")
    latest_box = f"""
    <div class="trajectory-card__badge">
      <span class="eyebrow">ONPE</span>
      <strong>{fmt_pct(latest_pct, 3)}</strong>
      <small>actas procesadas</small>
    </div>
    """

    return f"""
    <section class="stack">
      <div class="section-title">
        <p class="eyebrow">Trayectoria</p>
        <h2>Evolución observada y estimación al 100%</h2>
        <p>Línea continua = cortes observados. Línea punteada = tendencia. Punto final = estimación al cierre.</p>
      </div>
      <div class="trajectory-card">
        <div class="trajectory-card__header">
          <div class="trajectory-legend">
            {''.join(legend_items)}
            <div class="trajectory-legend__hint">Sólido = observado · Tramo punteado = proyección al 100%</div>
          </div>
          {latest_box}
        </div>
        <div class="trajectory-chart-shell">
          <div class="trajectory-canvas-shell">
            <canvas id="trajectoryChart" class="trajectory-canvas" aria-label="Trayectoria real y proyectada de candidatos"></canvas>
          </div>
          <aside class="trajectory-endpoints">
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
                    <span>{escape(candidate['candidate'])}</span>
                    <strong>{fmt_pct(candidate['new_pct'])}</strong>
                  </div>
                  <div class="delta-line__bar"><span style="width:{candidate['new_pct']:.2f}%; background:{color};"></span></div>
                  <div class="delta-line__meta">
                    <span>Acum. {fmt_pct(candidate['curr_pct'])}</span>
                    <span>{fmt_int(candidate['new_votes'])} votos nuevos</span>
                  </div>
                </div>
                """
            )
        block_cards.append(
            f"""
            <article class="delta-card">
              <div class="delta-card__header">
                <h3>{escape(block['label'])}</h3>
                <span>{fmt_int(block['new_validos'])} nuevos válidos</span>
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
        tone = "stable"
        if row["direction"] == "sube":
            tone = "up"
        elif row["direction"] == "baja":
            tone = "down"
        cards.append(
            f"""
            <article class="shift-card shift-card--{tone}">
              <div class="shift-card__header">
                <div>
                  <p class="eyebrow">Proyección final</p>
                  <h3>{escape(row['candidate'])}</h3>
                </div>
                <span class="shift-card__pill" style="--shift-color:{color};">{escape(row['direction'])}</span>
              </div>
              <div class="shift-card__numbers">
                <div>
                  <span class="value">{fmt_pct(row['current_pct'])}</span>
                  <span class="label">estimación actual</span>
                </div>
                <div>
                  <span class="value">{pct_sign}{delta_pct:.2f} pp</span>
                  <span class="label">vs base</span>
                </div>
              </div>
              <div class="candidate-card__bar">
                <span style="width:{row['current_pct']:.2f}%; background:{color};"></span>
              </div>
              <div class="shift-card__meta">
                <span>Base {fmt_pct(row['base_pct'])}</span>
                <span>{votes_sign}{fmt_int(delta_votes)} votos proj.</span>
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
    <section class="table-section">
      <div class="section-title">
        <p class="eyebrow">Territorio</p>
        <h2>Proyección por departamento</h2>
        <p>Valores en porcentaje de emitidos y votos absolutos proyectados al 100%.</p>
      </div>
      <div class="table-shell">
        <table>
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
                <table class="micro-table">
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
    <section class="stack">
      <div class="section-title">
        <p class="eyebrow">Territorio fino</p>
        <h2>Dónde está el remanente que define RLA vs Sánchez</h2>
        <p>Regiones y provincias con más votos pendientes, más un zoom de Lima por distrito. La brecha se muestra como {right_label} menos {left_label}.</p>
      </div>
      <div class="table-shell">
        <table>
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
      <div class="focus-grid">
        {''.join(detail_cards)}
      </div>
      <div class="focus-lower-grid">
        <div class="table-shell">
          <div class="focus-subtitle">
            <p class="eyebrow">Lima</p>
            <h3>Distritos con más votos faltantes</h3>
          </div>
          <table>
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
        <div class="table-shell">
          <div class="focus-subtitle">
            <div>
              <p class="eyebrow">Extranjero</p>
              <h3>Países con más votos faltantes</h3>
            </div>
          </div>
          <table>
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
          <p class="table-shell__note">El faltante por país se estima con actas pendientes y votos por acta del propio país, suavizado con el promedio del exterior.</p>
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
    for row in payload["projection_rows"][:8]:
        top2 = f"{row['top2_prob'] * 100:.1f}%"
        projection_rows.append(
            f"""
            <tr>
              <td>{escape(row['candidate'])}</td>
              <td>{fmt_pct(row['actual_pct'])}</td>
              <td>{fmt_pct(row['projected_pct'])}</td>
              <td>{fmt_pct(row['ic_lo'])} - {fmt_pct(row['ic_hi'])}</td>
              <td>{top2}</td>
            </tr>
            """
        )

    status_badge = f"{fmt_int(payload['votos_validos'])} votos válidos observados"
    trajectory_chart = render_trajectory_chart(payload)

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="180">
  <title>Dashboard local ONPE</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;600;700&family=Fira+Sans:wght@400;500;600;700;800&display=swap');
    :root {{
      --paper: {PAPER};
      --ink: {INK};
      --accent: {ACCENT};
      --soft: {SOFT};
      --card: {CARD};
      --muted: #475569;
      --line: rgba(15, 23, 42, 0.08);
      --blue-strong: #1d4ed8;
      --blue-soft: #dbeafe;
      --amber: #f59e0b;
      --shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
      --radius: 24px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(30, 64, 175, 0.12), transparent 20%),
        radial-gradient(circle at top right, rgba(245, 158, 11, 0.12), transparent 18%),
        linear-gradient(180deg, #f8fafc 0%, #eff6ff 48%, #f8fafc 100%);
      font-family: "Fira Sans", "Segoe UI", sans-serif;
    }}
    .page {{
      width: min(1220px, calc(100vw - 32px));
      margin: 32px auto 64px;
    }}
    .hero {{
      position: relative;
      overflow: hidden;
      padding: 36px;
      background:
        linear-gradient(135deg, rgba(255,255,255,0.96), rgba(239,246,255,0.96)),
        linear-gradient(180deg, rgba(255,255,255,0.86), rgba(255,255,255,0.86));
      border: 1px solid rgba(30, 64, 175, 0.10);
      border-radius: 34px;
      box-shadow: var(--shadow);
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -40px -40px auto;
      width: 240px;
      height: 240px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(30, 64, 175, 0.14), transparent 68%);
      pointer-events: none;
    }}
    .eyebrow {{
      margin: 0 0 8px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 11px;
      color: rgba(15, 23, 42, 0.56);
      font-family: "Fira Code", monospace;
    }}
    h1, h2, h3 {{
      margin: 0;
      font-family: "Fira Sans", "Segoe UI", sans-serif;
      font-weight: 800;
      line-height: 0.98;
    }}
    h1 {{
      font-size: clamp(2.6rem, 4vw, 4.8rem);
      max-width: none;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      max-width: none;
      line-height: 1.52;
      color: var(--muted);
      font-size: 1.02rem;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 22px;
      align-items: start;
    }}
    .meta-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 14px;
      margin-top: 6px;
    }}
    .meta-card {{
      min-width: 0;
      padding: 16px 16px 15px;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.94));
      border: 1px solid rgba(30, 64, 175, 0.10);
    }}
    .meta-card strong {{
      display: block;
      margin-top: 6px;
      font-size: clamp(0.98rem, 1.5vw, 1.18rem);
      font-family: "Fira Code", monospace;
      line-height: 1.14;
      letter-spacing: -0.02em;
      overflow-wrap: anywhere;
      word-break: break-word;
      hyphens: auto;
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 11px 16px;
      border-radius: 999px;
      background: rgba(30, 64, 175, 0.08);
      color: var(--blue-strong);
      font-weight: 600;
      border: 1px solid rgba(30, 64, 175, 0.10);
    }}
    .section-title {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 18px;
    }}
    .section-title p:last-child {{
      color: var(--muted);
      max-width: 72ch;
      margin: 0;
    }}
    .cards-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 18px;
      margin-top: 26px;
    }}
    .candidate-card, .battle-card, .remaining-card, .delta-card, .table-shell, .projection-shell, .trajectory-card {{
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      border-radius: var(--radius);
    }}
    .trajectory-card {{
      padding: 18px 18px 8px;
      overflow: hidden;
    }}
    .trajectory-card__header {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .trajectory-card__badge {{
      min-width: 180px;
      padding: 18px 20px;
      border-radius: 22px;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(239,246,255,0.94));
      border: 1px solid rgba(30, 64, 175, 0.10);
      text-align: right;
    }}
    .trajectory-card__badge strong {{
      display: block;
      font-size: 2rem;
      line-height: 1;
      margin: 6px 0 4px;
      font-family: "Fira Code", monospace;
    }}
    .trajectory-card__badge small {{
      color: var(--muted);
      font-size: 0.82rem;
    }}
    .trajectory-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      align-items: center;
    }}
    .trajectory-legend__item {{
      display: inline-flex;
      align-items: center;
      gap: 9px;
      font-weight: 600;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(248, 250, 252, 0.88);
      border: 1px solid rgba(148, 163, 184, 0.18);
    }}
    .trajectory-legend__code {{
      font-family: "Fira Code", monospace;
      font-size: 0.84rem;
      letter-spacing: 0.06em;
      color: rgba(15, 23, 42, 0.92);
    }}
    .trajectory-legend__name {{
      color: rgba(15, 23, 42, 0.62);
      font-size: 0.88rem;
      font-weight: 500;
    }}
    .trajectory-legend__swatch {{
      width: 24px;
      height: 10px;
      border-radius: 999px;
      background: var(--swatch);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.4);
    }}
    .trajectory-legend__hint {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .trajectory-chart-shell {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 220px;
      gap: 18px;
      align-items: stretch;
    }}
    .trajectory-canvas-shell {{
      position: relative;
      min-width: 0;
      min-height: 420px;
      padding: 10px 14px 6px 2px;
      border-radius: 24px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.88), rgba(248,250,252,0.72));
      border: 1px solid rgba(148, 163, 184, 0.14);
    }}
    .trajectory-canvas {{
      width: 100%;
      height: 420px;
      display: block;
    }}
    .trajectory-endpoints {{
      display: grid;
      align-content: start;
      gap: 12px;
      padding-top: 8px;
    }}
    .trajectory-endpoint {{
      padding: 14px 16px;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.94));
      border: 1px solid rgba(148, 163, 184, 0.16);
      box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
    }}
    .trajectory-endpoint__head {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .trajectory-endpoint__dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--swatch);
      box-shadow: 0 0 0 4px color-mix(in srgb, var(--swatch) 18%, white);
    }}
    .trajectory-endpoint__code {{
      font-family: "Fira Code", monospace;
      font-size: 0.84rem;
      letter-spacing: 0.06em;
      font-weight: 700;
    }}
    .trajectory-endpoint strong {{
      display: block;
      font-size: 1.35rem;
      line-height: 1;
      font-family: "Fira Code", monospace;
    }}
    .trajectory-endpoint small {{
      display: block;
      margin-top: 6px;
      color: var(--muted);
      line-height: 1.35;
      font-size: 0.82rem;
    }}
    .candidate-card {{
      position: relative;
      padding: 22px;
      overflow: hidden;
    }}
    .candidate-card__rank {{
      position: absolute;
      top: 16px;
      right: 16px;
      font-size: 12px;
      letter-spacing: 0.12em;
      color: rgba(15, 23, 42, 0.52);
      font-family: "Fira Code", monospace;
    }}
    .candidate-card__numbers {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin: 20px 0 18px;
    }}
    .value {{
      display: block;
      font-size: 1.45rem;
      font-weight: 700;
      font-family: "Fira Code", monospace;
    }}
    .label {{
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.86rem;
    }}
    .candidate-card__bar,
    .battle-line__bar,
    .remaining-card__bar,
    .delta-line__bar {{
      height: 12px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.08);
      overflow: hidden;
    }}
    .candidate-card__bar span,
    .battle-line__bar span,
    .remaining-card__bar span,
    .delta-line__bar span {{
      display: block;
      height: 100%;
      border-radius: inherit;
    }}
    .candidate-card__meta,
    .delta-line__meta,
    .battle-line__head,
    .remaining-card__header,
    .battle-card__header,
    .delta-card__header {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: baseline;
    }}
    .candidate-card__meta {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .stack,
    .delta-section,
    .table-section {{
      margin-top: 32px;
      display: grid;
      gap: 18px;
    }}
    .focus-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .focus-lower-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      align-items: start;
    }}
    .focus-foreign-stack {{
      display: grid;
      gap: 18px;
    }}
    .focus-card {{
      padding: 20px;
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      border-radius: var(--radius);
    }}
    .focus-card--foreign {{
      min-height: 100%;
    }}
    .focus-card__header,
    .focus-subtitle {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
    }}
    .focus-subtitle {{
      margin: 8px 8px 14px;
    }}
    .focus-card__header span {{
      color: var(--muted);
      font-size: 0.92rem;
      font-family: "Fira Code", monospace;
    }}
    .focus-card__chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 16px 0 18px;
    }}
    .focus-chip {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(219, 234, 254, 0.38);
      color: rgba(15, 23, 42, 0.88);
      border: 1px solid rgba(30, 64, 175, 0.08);
      font-weight: 600;
      font-size: 0.86rem;
    }}
    .focus-chip--gap {{
      background: rgba(241, 245, 249, 0.96);
    }}
    .focus-table-shell {{
      overflow: auto;
    }}
    .micro-table {{
      min-width: 100%;
    }}
    .micro-table th,
    .micro-table td {{
      padding: 10px 10px;
      font-size: 0.9rem;
    }}
    .table-shell__note {{
      margin: 10px 14px 14px;
      color: var(--muted);
      line-height: 1.5;
      font-size: 0.9rem;
    }}
    .focus-card__note {{
      margin: 6px 0 0;
      color: var(--muted);
      line-height: 1.55;
    }}
    .battle-grid, .remaining-grid, .delta-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
    }}
    .battle-card, .remaining-card, .delta-card {{
      padding: 20px;
    }}
    .shift-grid {{
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }}
    .shift-card {{
      padding: 22px;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.94));
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .shift-card--up {{
      box-shadow: 0 24px 60px rgba(31, 122, 69, 0.10);
    }}
    .shift-card--down {{
      box-shadow: 0 24px 60px rgba(217, 119, 6, 0.12);
    }}
    .shift-card__header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 18px;
    }}
    .shift-card__pill {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: color-mix(in srgb, var(--shift-color) 12%, white);
      color: var(--shift-color);
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-family: "Fira Code", monospace;
    }}
    .shift-card__numbers {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .shift-card__meta {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .battle-line, .delta-line {{
      margin-top: 14px;
    }}
    .battle-line__head, .delta-line__head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .remaining-card p {{
      margin: 12px 0 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .projection-shell, .table-shell {{
      padding: 10px;
      overflow: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 860px;
    }}
    th, td {{
      padding: 13px 14px;
      border-bottom: 1px solid rgba(31, 27, 23, 0.08);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-size: 0.86rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-family: "Fira Code", monospace;
    }}
    tbody tr:hover {{
      background: rgba(219, 234, 254, 0.32);
    }}
    .empty-state {{
      padding: 20px 22px;
      border-radius: 20px;
      background: rgba(255,255,255,0.74);
      color: var(--muted);
    }}
    .footer-note {{
      margin-top: 26px;
      color: var(--muted);
      line-height: 1.55;
      font-size: 0.92rem;
    }}
    @media (max-width: 980px) {{
      .cards-grid,
      .focus-grid,
      .battle-grid,
      .remaining-grid,
      .delta-grid,
      .meta-strip {{
        grid-template-columns: 1fr;
      }}
      .focus-lower-grid {{
        grid-template-columns: 1fr;
      }}
      .hero-grid {{
        grid-template-columns: 1fr;
      }}
      .trajectory-card__header {{
        flex-direction: column;
      }}
      .trajectory-chart-shell {{
        grid-template-columns: 1fr;
      }}
      .trajectory-endpoints {{
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      }}
      .page {{
        width: min(100vw - 20px, 1220px);
        margin-top: 12px;
      }}
      .hero {{
        padding: 24px;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <p class="eyebrow">Elecciones Generales 2026</p>
          <h1>Estimación de resultados, primera vuelta.</h1>
          <p>
            Lectura nacional y territorial del último corte disponible, con seguimiento de trayectoria y porcentaje proyectado al cierre.
          </p>
          <div class="status-pill">{escape(status_badge)}</div>
        </div>
        <div class="meta-strip">
          <div class="meta-card">
            <span class="eyebrow">Último corte</span>
            <strong>{escape(iso_to_local_label(payload['snapshot_at']))}</strong>
          </div>
          <div class="meta-card">
            <span class="eyebrow">Actualizado ONPE</span>
            <strong>{escape(epoch_millis_to_local_label(payload.get('onpe_updated_at')))}</strong>
          </div>
          <div class="meta-card">
            <span class="eyebrow">Actas procesadas</span>
            <strong>{fmt_int(payload['actas_contadas'])} / {fmt_int(payload['actas_totales'])}</strong>
          </div>
          <div class="meta-card">
            <span class="eyebrow">Avance nacional</span>
            <strong>{fmt_pct(payload['actas_pct'], 3)}</strong>
          </div>
          <div class="meta-card">
            <span class="eyebrow">Válidos observados</span>
            <strong>{fmt_int(payload['votos_validos'])}</strong>
          </div>
        </div>
      </div>
    </section>

    {trajectory_chart}

    <section class="stack">
      <div class="section-title">
        <p class="eyebrow">Foto nacional</p>
        <h2>Quiénes están peleando la zona crítica</h2>
      </div>
      <div class="cards-grid">
        {cards}
      </div>
    </section>

    {render_projection_shift_section(payload['projection_shift']['latest'])}
    {render_projection_shift_section(payload['projection_shift']['trend'])}

    <section class="stack">
      <div class="section-title">
        <p class="eyebrow">Lo que falta</p>
        <h2>Bolsones pendientes por bloque</h2>
      </div>
      <div class="remaining-grid">
        {render_remaining_blocks(payload)}
      </div>
    </section>

    <section class="stack">
      <div class="section-title">
        <p class="eyebrow">Mapa político</p>
        <h2>Proyección por bloque territorial</h2>
      </div>
      <div class="battle-grid">
        {render_block_summary(payload)}
      </div>
    </section>

    {render_territory_drilldown(payload)}

    <section class="stack">
      <div class="section-title">
        <p class="eyebrow">Diferencia real</p>
        <h2>Brechas al corte actual</h2>
        <p>Diferencias observadas hasta este momento, sin proyectar al 100%.</p>
      </div>
      {render_actual_comparison(payload)}
    </section>

    <section class="stack">
      <div class="section-title">
        <p class="eyebrow">Brecha simulada</p>
        <h2>Brechas al 100% proyectado</h2>
        <p>Diferencias estimadas al cierre entre cada par, tanto en votos proyectados como en puntos porcentuales.</p>
      </div>
      {render_projected_comparison(payload)}
    </section>

    {render_delta_section(payload['delta_info']['latest'], 'Última tanda contra el snapshot previo')}
    {render_delta_section(payload['delta_info']['trend'], 'Tendencia acumulada de los últimos cortes')}

    <section class="stack">
      <div class="section-title">
        <p class="eyebrow">Ranking general</p>
        <h2>Top de proyección nacional</h2>
      </div>
      <div class="projection-shell">
        <table>
          <thead>
            <tr>
              <th>Candidato</th>
              <th>Actual</th>
              <th>Proyección</th>
              <th>IC 95%</th>
              <th>Prob. top-2</th>
            </tr>
          </thead>
          <tbody>
            {''.join(projection_rows)}
          </tbody>
        </table>
      </div>
    </section>

    {render_department_table(payload)}

    <p class="footer-note">
      Estimación referencial de resultados de Elecciones Generales 2026, primera vuelta. Actualizado localmente el {escape(iso_to_local_label(payload['generated_at']))}.
    </p>
</main>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <script>
    (() => {{
      const el = document.getElementById("trajectory-data");
      const canvas = document.getElementById("trajectoryChart");
      if (!el || !canvas || !window.Chart) return;

      const trajectory = JSON.parse(el.textContent);
      const labels = {json.dumps(TRACK_SHORT_LABELS, ensure_ascii=False)};
      const datasets = [];

      for (const line of trajectory.series || []) {{
        const actual = (line.actual || []).map((point) => ({{
          x: point.x,
          y: point.y,
          label: point.label || ""
        }}));
        if (!actual.length) continue;

        const latest = actual[actual.length - 1];
        const short = labels[line.candidate] || line.candidate;

        datasets.push({{
          label: short,
          data: actual,
          borderColor: line.color,
          backgroundColor: line.color,
          pointBackgroundColor: line.color,
          pointBorderColor: "rgba(255,255,255,0.92)",
          pointBorderWidth: 1.8,
          pointRadius: 4,
          pointHoverRadius: 5,
          borderWidth: 3,
          tension: 0.28
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
          pointBorderColor: ["transparent", "rgba(15,23,42,0.28)"],
          pointBorderWidth: [0, 2],
          pointRadius: [0, 6],
          pointHoverRadius: [0, 6],
          borderWidth: 2,
          tension: 0,
          borderDash: [6, 8]
        }});
      }}

      const xMax = Math.max(100.8, Number(trajectory.bounds.x_max || 100) + 0.8);

      new Chart(canvas.getContext("2d"), {{
        type: "line",
        data: {{ datasets }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          layout: {{
            padding: {{
              right: 12
            }}
          }},
          interaction: {{
            mode: "nearest",
            intersect: false
          }},
          plugins: {{
            legend: {{
              display: false
            }},
            tooltip: {{
              backgroundColor: "rgba(15,23,42,0.92)",
              titleFont: {{ family: "Fira Code, monospace", size: 11 }},
              bodyFont: {{ family: "Fira Sans, sans-serif", size: 12 }},
              callbacks: {{
                title: (items) => {{
                  const raw = items?.[0]?.raw;
                  return raw?.label || "Punto";
                }},
                label: (item) => `${{item.dataset.label}}: ${{item.parsed.y.toFixed(2)}}%`
              }}
            }}
          }},
          scales: {{
            x: {{
              type: "linear",
              min: trajectory.bounds.x_min,
              max: xMax,
              grid: {{
                color: "rgba(15, 23, 42, 0.08)",
                drawBorder: false
              }},
              border: {{
                display: false
              }},
              ticks: {{
                color: "rgba(15, 23, 42, 0.52)",
                stepSize: 5,
                font: {{
                  family: "Fira Code, monospace",
                  size: 11
                }},
                callback: (value) => `${{value}}%`
              }}
            }},
            y: {{
              min: trajectory.bounds.y_min,
              max: trajectory.bounds.y_max,
              grid: {{
                color: "rgba(15, 23, 42, 0.08)",
                drawBorder: false
              }},
              border: {{
                display: false
              }},
              ticks: {{
                color: "rgba(15, 23, 42, 0.52)",
                font: {{
                  family: "Fira Code, monospace",
                  size: 11
                }},
                callback: (value) => `${{Number(value).toFixed(2)}}%`
              }}
            }}
          }},
          elements: {{
            line: {{
              capBezierPoints: true
            }}
          }}
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
    )

    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    html_out.write_text(render_html(payload))

    print(f"Dashboard HTML actualizado en {html_out}")
    print(f"Datos JSON actualizados en {json_out}")


if __name__ == "__main__":
    main()
