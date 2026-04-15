"""
Analisis de ausentismo e instalacion tardia de mesas en Lima Metropolitana.

Objetivo:
- medir cuantas mesas se instalaron tarde por distrito;
- estimar como cambia el ausentismo segun la hora de instalacion;
- aproximar cuantos votos pudieron perder los candidatos por ese retraso.

Supuestos operativos:
- "Lima" = provincia de Lima Metropolitana (ubigeo provincia 140100).
- Se trabaja con la eleccion presidencial (idEleccion=10).
- La hora de instalacion se extrae del PDF publico `AIPRE{mesa}_STAE.pdf`.
- La perdida de votos por candidato se prorratea con la participacion
  distrital de votos validos observados.

Salida:
- output/lima_instalacion_mesas.csv
- output/lima_instalacion_distritos.csv
- output/lima_instalacion_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import requests


API = "https://resultadoelectoral.onpe.gob.pe/presentacion-backend"
ID_ELECCION = 10
ID_AMBITO_PERU = 1
LIMA_DEPARTAMENTO = 140000
LIMA_PROVINCIA = 140100
TRACKED_CANDIDATES = (
    "RENOVACIÓN POPULAR",
    "PARTIDO DEL BUEN GOBIERNO",
    "JUNTOS POR EL PERÚ",
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Referer": "https://resultadoelectoral.onpe.gob.pe/main/actas",
    "Origin": "https://resultadoelectoral.onpe.gob.pe",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-PE,es;q=0.9,en;q=0.8",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "X-Requested-With": "XMLHttpRequest",
}

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
CACHE_PATH = OUTPUT_DIR / "lima_instalacion_cache.json"
MESA_CODES_PATH = OUTPUT_DIR / "lima_instalacion_mesa_codes.json"
MESAS_CSV = OUTPUT_DIR / "lima_instalacion_mesas.csv"
DISTRITOS_CSV = OUTPUT_DIR / "lima_instalacion_distritos.csv"
SUMMARY_JSON = OUTPUT_DIR / "lima_instalacion_summary.json"

INSTALL_RE = re.compile(
    r"ACTA DE INSTALACI[ÓO]N\s+([0-9]{1,2}:[0-9]{2}\s*[ap]\.\s*m\.)",
    re.IGNORECASE | re.MULTILINE,
)
TIME_RE = re.compile(r"([0-9]{1,2}):([0-9]{2})\s*([ap])\.\s*m\.", re.IGNORECASE)


@dataclass
class District:
    ubigeo: int
    name: str


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def api_get(session: requests.Session, path: str, timeout: int = 45, **params) -> object:
    url = f"{API}/{path.lstrip('/')}"
    last_err = None
    for attempt in range(1, 5):
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            if not payload.get("success", False):
                raise RuntimeError(f"API {path} devolvio error: {payload}")
            return payload.get("data")
        except Exception as err:  # noqa: BLE001
            last_err = err
            if attempt == 4:
                raise
            time.sleep(0.5 * attempt)
    raise last_err if last_err else RuntimeError(f"No se pudo consultar {path}")


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text())


def save_cache(cache: dict[str, dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def load_mesa_codes() -> dict[str, list[str]]:
    if not MESA_CODES_PATH.exists():
        return {}
    return json.loads(MESA_CODES_PATH.read_text())


def save_mesa_codes(data: dict[str, list[str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MESA_CODES_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def get_lima_districts(session: requests.Session) -> list[District]:
    rows = api_get(
        session,
        "ubigeos/distritos",
        idEleccion=ID_ELECCION,
        idAmbitoGeografico=ID_AMBITO_PERU,
        idUbigeoProvincia=LIMA_PROVINCIA,
    )
    return [District(ubigeo=int(row["ubigeo"]), name=row["nombre"]) for row in rows]


def list_presidential_mesas_for_district(
    session: requests.Session,
    district: District,
    page_size: int = 500,
) -> list[str]:
    # ONPE devuelve `content=[]` en varios distritos cuando `tamanio` es demasiado
    # grande, aun cuando `totalRegistros` es positivo. Ajustamos de forma adaptativa.
    page_size = max(20, min(int(page_size), 500))
    if page_size >= 500:
        candidate_sizes = [500, 200, 100, 50, 20]
    else:
        candidate_sizes = sorted({page_size, 200, 100, 50, 20}, reverse=True)

    selected_size = None
    first_page = None
    for size in candidate_sizes:
        data = api_get(
            session,
            "actas",
            timeout=20,
            pagina=1,
            tamanio=size,
            idAmbitoGeografico=ID_AMBITO_PERU,
            idUbigeo=district.ubigeo,
        )
        total_registros = int(data.get("totalRegistros", 0) or 0)
        if total_registros == 0 or data.get("content"):
            selected_size = size
            first_page = data
            break
    if selected_size is None or first_page is None:
        raise RuntimeError(f"No se pudo listar actas para {district.name} ({district.ubigeo})")

    page = 1
    out: list[str] = []
    seen: set[str] = set()
    while True:
        data = first_page if page == 1 else api_get(
            session,
            "actas",
            timeout=20,
            pagina=page,
            tamanio=selected_size,
            idAmbitoGeografico=ID_AMBITO_PERU,
            idUbigeo=district.ubigeo,
        )
        for row in data["content"]:
            if int(row.get("idEleccion") or 0) != ID_ELECCION:
                continue
            code = str(row.get("codigoMesa") or "").zfill(6)
            if code and code not in seen:
                seen.add(code)
                out.append(code)
        if page >= int(data.get("totalPaginas", 1)):
            break
        page += 1
    return out


def get_candidate_shares_for_district(session: requests.Session, district_ubigeo: int) -> dict[str, float]:
    rows = api_get(
        session,
        "eleccion-presidencial/participantes-ubicacion-geografica-nombre",
        tipoFiltro="ubigeo_nivel_03",
        ubigeoNivel1=LIMA_DEPARTAMENTO,
        ubigeoNivel2=LIMA_PROVINCIA,
        ubigeoNivel3=district_ubigeo,
        idAmbitoGeografico=ID_AMBITO_PERU,
        idEleccion=ID_ELECCION,
    )
    total_valid = sum(float(row.get("totalVotosValidos") or 0) for row in rows)
    shares: dict[str, float] = {}
    for row in rows:
        name = str(row.get("nombreAgrupacionPolitica") or "").strip()
        shares[name] = (float(row.get("totalVotosValidos") or 0) / total_valid) if total_valid else 0.0
    return shares


def collect_district_mesas(
    district: District,
    page_size: int,
    force_refresh: bool,
    cached_mesas: list[str] | None,
) -> tuple[int, str, list[str]]:
    if cached_mesas and not force_refresh:
        return district.ubigeo, district.name, cached_mesas
    session = build_session()
    mesas = list_presidential_mesas_for_district(session, district, page_size=page_size)
    return district.ubigeo, district.name, mesas


def parse_time_label(text: str) -> str | None:
    match = INSTALL_RE.search(text)
    if not match:
        return None
    return " ".join(match.group(1).split())


def time_label_to_hour(label: str | None) -> float | None:
    if not label:
        return None
    match = TIME_RE.search(label)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    meridian = match.group(3).lower()
    if meridian == "p" and hour != 12:
        hour += 12
    if meridian == "a" and hour == 12:
        hour = 0
    return hour + (minute / 60.0)


def fetch_installation_pdf_text(mesa_code: str) -> str | None:
    url = f"https://actas-stae.onpe.gob.pe/AIPRE{mesa_code}_STAE.pdf"
    cmd = [
        "bash",
        "-lc",
        f"curl -L -sS {url!s} | pdftotext - - -enc UTF-8",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    return text or None


def fetch_mesa_snapshot(session: requests.Session, mesa_code: str) -> dict | None:
    rows = api_get(session, "actas/buscar/mesa", codigoMesa=mesa_code)
    for row in rows:
        if int(row.get("idEleccion") or 0) == ID_ELECCION:
            return row
    return None


def hydrate_mesa(
    mesa_code: str,
    district_names: dict[int, str],
    force: bool = False,
    cached: dict | None = None,
) -> dict | None:
    if cached and not force:
        has_turnout = cached.get("totalElectoresHabiles") is not None
        has_install = cached.get("installTimeLabel") is not None
        if has_turnout and has_install:
            return cached

    session = build_session()
    mesa = fetch_mesa_snapshot(session, mesa_code)
    if not mesa:
        return None

    district_ubigeo = int(mesa.get("idUbigeo") or 0)
    total_electores = mesa.get("totalElectoresHabiles")
    total_asistentes = mesa.get("totalAsistentes")
    install_text = fetch_installation_pdf_text(mesa_code)
    install_label = parse_time_label(install_text or "")
    install_hour = time_label_to_hour(install_label)

    out = {
        "mesaCode": mesa_code,
        "districtUbigeo": district_ubigeo,
        "districtName": district_names.get(district_ubigeo, ""),
        "nombreLocalVotacion": mesa.get("nombreLocalVotacion"),
        "codigoLocalVotacion": mesa.get("codigoLocalVotacion"),
        "codigoEstadoActa": mesa.get("codigoEstadoActa"),
        "descripcionEstadoActa": mesa.get("descripcionEstadoActa"),
        "totalElectoresHabiles": int(total_electores) if total_electores is not None else None,
        "totalAsistentes": int(total_asistentes) if total_asistentes is not None else None,
        "porcentajeParticipacionCiudadana": float(mesa.get("porcentajeParticipacionCiudadana") or 0.0)
        if mesa.get("porcentajeParticipacionCiudadana") is not None
        else None,
        "installTimeLabel": install_label,
        "installHour": install_hour,
        "delayHours": (install_hour - 7.0) if install_hour is not None else None,
        "fetchedAt": datetime.now().isoformat(),
    }
    if out["totalElectoresHabiles"] and out["totalAsistentes"] is not None:
        out["ausentismoPct"] = pct(
            out["totalElectoresHabiles"] - out["totalAsistentes"],
            out["totalElectoresHabiles"],
        )
    else:
        out["ausentismoPct"] = None
    return out


def fit_delay_model(rows: list[dict]) -> dict[str, float]:
    valid = [
        row for row in rows
        if row.get("installHour") is not None
        and row.get("totalElectoresHabiles")
        and row.get("ausentismoPct") is not None
    ]
    if len(valid) < 3:
        return {"intercept": 0.0, "slope_pp_per_hour": 0.0, "n": len(valid), "weighted_r2": 0.0}

    x = np.array([float(row["installHour"]) for row in valid], dtype=float)
    y = np.array([float(row["ausentismoPct"]) for row in valid], dtype=float)
    w = np.array([float(row["totalElectoresHabiles"]) for row in valid], dtype=float)
    coeff = np.polyfit(x, y, 1, w=np.sqrt(w))
    slope, intercept = float(coeff[0]), float(coeff[1])

    y_hat = intercept + slope * x
    y_bar = float(np.average(y, weights=w))
    ss_res = float(np.sum(w * (y - y_hat) ** 2))
    ss_tot = float(np.sum(w * (y - y_bar) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "intercept": intercept,
        "slope_pp_per_hour": slope,
        "n": len(valid),
        "weighted_r2": r2,
    }


def summarize_hour_buckets(rows: list[dict]) -> dict[str, dict[str, float]]:
    buckets = {
        "07_08": [],
        "08_10": [],
        "10_12": [],
        "12_plus": [],
    }
    for row in rows:
        hour = row.get("installHour")
        if hour is None:
            continue
        if 7.0 <= hour < 8.0:
            buckets["07_08"].append(row)
        elif 8.0 <= hour < 10.0:
            buckets["08_10"].append(row)
        elif 10.0 <= hour < 12.0:
            buckets["10_12"].append(row)
        elif hour >= 12.0:
            buckets["12_plus"].append(row)

    out: dict[str, dict[str, float]] = {}
    for key, items in buckets.items():
        electores = sum(int(item.get("totalElectoresHabiles") or 0) for item in items)
        ausentismo_num = sum(
            max(int(item.get("totalElectoresHabiles") or 0) - int(item.get("totalAsistentes") or 0), 0)
            for item in items
            if item.get("totalElectoresHabiles") is not None and item.get("totalAsistentes") is not None
        )
        out[key] = {
            "mesas": len(items),
            "electores": electores,
            "ausentismo_pct_weighted": pct(ausentismo_num, electores),
        }
    return out


def district_summary(
    rows: list[dict],
    candidate_shares: dict[int, dict[str, float]],
    slope_pp_per_hour: float,
) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["districtUbigeo"]), []).append(row)

    out: list[dict] = []
    slope = max(slope_pp_per_hour, 0.0)
    for district_ubigeo, items in grouped.items():
        known = [item for item in items if item.get("installHour") is not None]
        electores = sum(int(item.get("totalElectoresHabiles") or 0) for item in known)
        asistentes = sum(int(item.get("totalAsistentes") or 0) for item in known if item.get("totalAsistentes") is not None)
        extra_ausentes = 0.0
        delays = []
        after_8 = 0
        after_10 = 0
        after_12 = 0
        for item in known:
            delay = max(float(item.get("delayHours") or 0.0), 0.0)
            delays.append(delay)
            if delay >= 1.0:
                after_8 += 1
            if delay >= 3.0:
                after_10 += 1
            if delay >= 5.0:
                after_12 += 1
            extra_ausentes += (int(item.get("totalElectoresHabiles") or 0) * slope * delay) / 100.0
        shares = candidate_shares.get(district_ubigeo, {})
        row = {
            "districtUbigeo": district_ubigeo,
            "districtName": items[0]["districtName"],
            "mesasConHora": len(known),
            "mesasDespues8": after_8,
            "mesasDespues10": after_10,
            "mesasDespues12": after_12,
            "horaMediana": float(np.median([item["installHour"] for item in known])) if known else None,
            "delayPromedioHoras": float(np.mean(delays)) if delays else None,
            "electoresObservados": electores,
            "asistentesObservados": asistentes,
            "ausentismoPctObservado": pct(max(electores - asistentes, 0), electores),
            "ausentesExtraEstimados": extra_ausentes,
        }
        for candidate in TRACKED_CANDIDATES:
            row[f"share::{candidate}"] = shares.get(candidate, 0.0)
            row[f"lostVotes::{candidate}"] = extra_ausentes * shares.get(candidate, 0.0)
        out.append(row)
    return sorted(out, key=lambda row: row["ausentesExtraEstimados"], reverse=True)


def total_candidate_losses(rows: list[dict]) -> dict[str, float]:
    out = {candidate: 0.0 for candidate in TRACKED_CANDIDATES}
    for row in rows:
        for candidate in TRACKED_CANDIDATES:
            out[candidate] += float(row.get(f"lostVotes::{candidate}") or 0.0)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    districts = get_lima_districts(session)
    district_names = {district.ubigeo: district.name for district in districts}
    print(f"→ Distritos Lima Metropolitana: {len(districts)}", file=sys.stderr)

    mesa_codes_cache = load_mesa_codes()
    mesas_by_district: dict[int, list[str]] = {}
    all_mesas: list[str] = []
    districts_for_listing = districts
    if args.max_mesas:
        # En modo piloto no hace falta recorrer toda Lima; tomamos distritos
        # hasta superar el universo pedido.
        districts_for_listing = []
        preview_total = 0
        for district in districts:
            cached = mesa_codes_cache.get(str(district.ubigeo)) or []
            districts_for_listing.append(district)
            preview_total += len(cached)
            if preview_total >= args.max_mesas and preview_total > 0:
                break

    with ThreadPoolExecutor(max_workers=args.district_workers) as executor:
        futures = {
            executor.submit(
                collect_district_mesas,
                district,
                args.page_size,
                args.force_refresh,
                mesa_codes_cache.get(str(district.ubigeo)),
            ): district
            for district in districts_for_listing
        }
        for future in as_completed(futures):
            district = futures[future]
            district_ubigeo, district_name, mesas = future.result()
            mesa_codes_cache[str(district_ubigeo)] = mesas
            mesas_by_district[district_ubigeo] = mesas
            all_mesas.extend(mesas)
            save_mesa_codes(mesa_codes_cache)
            print(f"  . {district_name:<22} {len(mesas):>5} mesas presidenciales", file=sys.stderr)
    save_mesa_codes(mesa_codes_cache)

    all_mesas = sorted(set(all_mesas))
    if args.max_mesas:
        all_mesas = all_mesas[: args.max_mesas]
    print(f"→ Mesas presidenciales en universo: {len(all_mesas):,}", file=sys.stderr)

    candidate_shares = {
        district.ubigeo: get_candidate_shares_for_district(session, district.ubigeo)
        for district in districts
    }

    cache = load_cache()
    rows: list[dict] = []
    updated = 0
    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for mesa_code in all_mesas:
            futures[executor.submit(
                hydrate_mesa,
                mesa_code,
                district_names,
                args.force_refresh,
                cache.get(mesa_code),
            )] = mesa_code
        done = 0
        total = len(futures)
        for future in as_completed(futures):
            done += 1
            mesa_code = futures[future]
            row = future.result()
            if row:
                if cache.get(mesa_code) != row:
                    updated += 1
                cache[mesa_code] = row
                rows.append(row)
            if done == 1 or done % 250 == 0 or done == total:
                print(f"  . mesas procesadas {done:,}/{total:,}", file=sys.stderr)

    save_cache(cache)
    rows.sort(key=lambda row: row["mesaCode"])
    write_csv(MESAS_CSV, rows)

    model = fit_delay_model(rows)
    bucket_summary = summarize_hour_buckets(rows)
    district_rows = district_summary(rows, candidate_shares, model["slope_pp_per_hour"])
    write_csv(DISTRITOS_CSV, district_rows)
    candidate_losses = total_candidate_losses(district_rows)

    mesas_con_hora = [row for row in rows if row.get("installHour") is not None]
    summary = {
        "generatedAt": datetime.now().isoformat(),
        "scope": "Lima Metropolitana",
        "districtCount": len(districts),
        "mesaCount": len(rows),
        "mesaCountWithInstallTime": len(mesas_con_hora),
        "cacheUpdated": updated,
        "regression": model,
        "hourBuckets": bucket_summary,
        "lateMesaCounts": {
            "after_08": sum(1 for row in mesas_con_hora if float(row["installHour"]) >= 8.0),
            "after_10": sum(1 for row in mesas_con_hora if float(row["installHour"]) >= 10.0),
            "after_12": sum(1 for row in mesas_con_hora if float(row["installHour"]) >= 12.0),
        },
        "totalExtraAusentesEstimated": float(sum(row["ausentesExtraEstimados"] for row in district_rows)),
        "candidateLossesEstimated": candidate_losses,
        "topDistrictsByExtraAusentes": district_rows[:15],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Analisis de instalacion tardia de mesas en Lima Metropolitana")
    ap.add_argument("--workers", type=int, default=12, help="workers paralelos para mesas")
    ap.add_argument("--district-workers", type=int, default=8, help="workers paralelos para barrido distrital")
    ap.add_argument("--page-size", type=int, default=500, help="tamaño de pagina para listado de actas por distrito")
    ap.add_argument("--max-mesas", type=int, default=0, help="limitar mesas para pruebas")
    ap.add_argument("--force-refresh", action="store_true", help="ignorar cache local")
    args = ap.parse_args()

    summary = run(args)
    slope = summary["regression"]["slope_pp_per_hour"]
    print()
    print("Analisis Lima Metropolitana - instalacion vs ausentismo")
    print("-" * 72)
    print(f"Mesas con hora extraida: {summary['mesaCountWithInstallTime']:,}/{summary['mesaCount']:,}")
    print(f"Pendiente estimada: {slope:+.3f} pp de ausentismo por hora")
    print(f"Mesas instaladas despues de 8:00:  {summary['lateMesaCounts']['after_08']:,}")
    print(f"Mesas instaladas despues de 10:00: {summary['lateMesaCounts']['after_10']:,}")
    print(f"Mesas instaladas despues de 12:00: {summary['lateMesaCounts']['after_12']:,}")
    print(f"Ausentes extra estimados: {summary['totalExtraAusentesEstimated']:,.0f}")
    for candidate, votes in summary["candidateLossesEstimated"].items():
        print(f"  {candidate:<30} {votes:>10,.0f}")


if __name__ == "__main__":
    main()
