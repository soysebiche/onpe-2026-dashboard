"""
Proyeccion bayesiana e incremental de la eleccion presidencial peruana 2026.

Modelo base:
- Dirichlet-Multinomial por circunscripcion.
- Estima votos faltantes por region y simula el total nacional.

Capa incremental:
- Guarda snapshots sucesivos de ONPE.
- Calcula que composicion traen los votos nuevos entre cortes.
- Ajusta una proyeccion heuristica del faltante usando la tendencia reciente
  por bloques: regiones sin Lima, Lima y extranjero.

Uso:
    python3 onpe_proyeccion.py
    python3 onpe_proyeccion.py --draws 20000
    python3 onpe_proyeccion.py --geo-level province
    python3 onpe_proyeccion.py --track-state
    python3 onpe_proyeccion.py --track-state --trend-window 4 --trend-weight 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

API = "https://resultadoelectoral.onpe.gob.pe/presentacion-backend"
PROXY_API = "https://onpe-proxy.sebbs21.workers.dev"
DEFAULT_REFERER = "https://resultadoelectoral.onpe.gob.pe/main/resumen"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Referer": DEFAULT_REFERER,
    "Origin": "https://resultadoelectoral.onpe.gob.pe",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-PE,es;q=0.9,en;q=0.8",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "X-Requested-With": "XMLHttpRequest",
}
ID_ELECCION = 10  # presidencial EG2026
LIMA_UBIGEO = 140000
DEFAULT_STATE_FILE = Path(__file__).with_name("onpe_state.json")
FINE_GEO_FILTER = "ubigeo_nivel_03"
DEFAULT_GEO_LEVEL = "province"
DEFAULT_MAX_WORKERS = 12
TRACK_DEFAULT = [
    "JUNTOS POR EL PERU",
    "RENOVACION POPULAR",
    "PARTIDO DEL BUEN GOBIERNO",
    "FUERZA POPULAR",
]
LATE_BIAS_DEFAULT_STRENGTH = 0.72
LATE_BIAS_DEFAULT_WINDOW = 4
LATE_BIAS_DEFAULT_RECENCY = 0.82
LATE_MOMENTUM_BLEND = 0.65
LOCAL_VPA_SMALL_UNIT_SHARE = 0.45
LOCAL_VPA_MIN_UNITS = 4

# Padron electoral por circunscripcion (JNE-INFOgob Reporte N.5, EG2026).
# Clave = nombre normalizado (sin tildes, mayusculas).
PADRON = {
    "AMAZONAS": 345_245,
    "ANCASH": 971_385,
    "APURIMAC": 361_338,
    "AREQUIPA": 1_226_525,
    "AYACUCHO": 520_238,
    "CAJAMARCA": 1_198_773,
    "CALLAO": 858_968,
    "CUSCO": 1_133_754,
    "HUANCAVELICA": 339_448,
    "HUANUCO": 656_517,
    "ICA": 713_997,
    "JUNIN": 1_062_500,
    "LA LIBERTAD": 1_552_691,
    "LAMBAYEQUE": 1_051_350,
    "LIMA": 7_822_555 + 828_473,  # Metro + Provincias
    "LORETO": 775_923,
    "MADRE DE DIOS": 147_577,
    "MOQUEGUA": 164_628,
    "PASCO": 223_695,
    "PIURA": 1_534_085,
    "PUNO": 963_489,
    "SAN MARTIN": 723_605,
    "TACNA": 302_615,
    "TUMBES": 181_317,
    "UCAYALI": 453_928,
    "EXTRANJERO": 1_210_813,
}


def norm(s: str) -> str:
    """Mayusculas sin tildes."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.upper().strip()


def pct(v: float, total: float) -> float:
    return 100.0 * v / total if total else 0.0


def clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def region_key(ambito: int, ubigeo: int) -> str:
    return f"{ambito}:{ubigeo}"


def infer_region_code(ambito: int, ubigeo: int) -> int:
    if ambito == 2:
        return 0
    if ubigeo <= 0:
        return 0
    if ubigeo % 10000 == 0:
        return ubigeo
    return (ubigeo // 10000) * 10000


def block_key(ambito: int, ubigeo_region: int) -> str:
    if ambito == 2:
        return "extranjero"
    if ubigeo_region == LIMA_UBIGEO:
        return "lima"
    return "regiones"


def block_label(key: str) -> str:
    return {
        "regiones": "Regiones sin Lima",
        "lima": "Lima",
        "extranjero": "Extranjero",
        "total": "Total nacional",
    }[key]


# --------- Fetch -----------------------------------------------------------

_session = requests.Session()
_session.headers.update(HEADERS)
FETCH_TIMEOUT = 30
FETCH_RETRIES = 4


def configure_api(api_base: str | None = None, election_id: int | None = None, referer: str | None = None) -> None:
    global API, ID_ELECCION
    if api_base:
        API = api_base.rstrip("/")
    if election_id is not None:
        ID_ELECCION = int(election_id)
    effective_referer = referer or _session.headers.get("Referer") or DEFAULT_REFERER
    _session.headers["Referer"] = effective_referer


def _fetch_from(base_url: str, path: str, params: dict) -> requests.Response:
    """Intenta fetching desde base_url. Lanza excepcion si falla o devuelve HTML."""
    url = f"{base_url}/{path.lstrip('/')}"
    last_err = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            r = _session.get(url, params=params, timeout=FETCH_TIMEOUT)
            break
        except requests.exceptions.ReadTimeout as err:
            last_err = err
            if attempt == FETCH_RETRIES:
                raise
            time.sleep(0.8 * attempt)
        except requests.exceptions.RequestException as err:
            last_err = err
            if attempt == FETCH_RETRIES:
                raise
            time.sleep(0.5 * attempt)
    else:
        raise last_err if last_err else RuntimeError(f"No se pudo consultar {url}")
    r.raise_for_status()
    if "application/json" not in r.headers.get("content-type", ""):
        raise RuntimeError(f"ONPE devolvio HTML para {url} (posible bloqueo de IP).")
    return r


def fetch(path: str, **params) -> dict:
    params.setdefault("idEleccion", ID_ELECCION)
    # Intenta directo; si devuelve HTML (IP bloqueada), usa el proxy de Cloudflare
    try:
        r = _fetch_from(API, path, params)
    except RuntimeError as direct_err:
        if "HTML" in str(direct_err):
            r = _fetch_from(PROXY_API, path, params)
        else:
            raise
    j = r.json()
    if not j.get("success", False):
        raise RuntimeError(f"ONPE error {r.url}: {j}")
    return j["data"]


def fetch_totales(**params) -> dict | None:
    params.setdefault("idEleccion", ID_ELECCION)
    for base_url in (API, PROXY_API):
        url = f"{base_url}/resumen-general/totales"
        for attempt in range(1, FETCH_RETRIES + 1):
            try:
                r = _session.get(url, params=params, timeout=FETCH_TIMEOUT)
            except requests.exceptions.RequestException:
                if attempt == FETCH_RETRIES:
                    break
                time.sleep(0.5 * attempt)
                continue
            if r.status_code == 204:
                return None
            r.raise_for_status()
            if "application/json" not in r.headers.get("content-type", ""):
                break
            j = r.json()
            if not j.get("success", False):
                break
            return j.get("data")
    return None


# --------- Modelado --------------------------------------------------------


@dataclass
class Region:
    nombre: str
    ubigeo: int
    ubigeo_region: int
    ambito: int  # 1=Peru, 2=Extranjero
    geo_level: str
    actas_contabilizadas: int
    actas_totales: int
    pct_actas: float
    padron: int
    candidatos: list[str] = field(default_factory=list)
    votos: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    votos_validos: int = 0
    votos_emitidos: int = 0
    participacion_ciudadana: float = 0.0
    ref_validos_por_acta: float = 0.0
    ref_emitidos_por_acta: float = 0.0
    ref_valid_ratio: float = 0.0

    @property
    def actas_faltantes(self) -> int:
        return max(self.actas_totales - self.actas_contabilizadas, 0)

    @property
    def valid_ratio_obs(self) -> float:
        if self.votos_emitidos > 0:
            return clip01(self.votos_validos / self.votos_emitidos)
        if self.ref_valid_ratio > 0:
            return self.ref_valid_ratio
        return 0.0

    @property
    def emitidos_por_acta_obs(self) -> float:
        if self.actas_contabilizadas <= 0:
            return 0.0
        if self.votos_emitidos > 0:
            return self.votos_emitidos / self.actas_contabilizadas
        ratio = self.valid_ratio_obs
        if self.votos_validos > 0 and ratio > 0:
            return (self.votos_validos / ratio) / self.actas_contabilizadas
        return 0.0

    def votos_faltantes_estimados(self) -> int:
        """
        Estima votos válidos faltantes modelando primero votos emitidos y luego
        la fracción válida. Si falta granularidad de turnout, cae a referencias
        locales/empíricas por acta y, al final, al padrón como último recurso.
        """
        if self.actas_totales == 0:
            return 0
        if self.actas_faltantes <= 0:
            return 0

        ref_valid_ratio = self.ref_valid_ratio if self.ref_valid_ratio > 0 else self.valid_ratio_obs
        ref_emitidos_per_acta = self.ref_emitidos_por_acta

        if self.actas_contabilizadas > 0 and self.votos_validos > 0:
            emitidos_local = self.emitidos_por_acta_obs
            valid_ratio_local = self.valid_ratio_obs
            progress = self.pct_actas / 100.0
            local_weight = clip01(0.30 + 0.60 * progress)

            emitidos_per_acta = emitidos_local
            if ref_emitidos_per_acta > 0 and emitidos_local > 0:
                emitidos_per_acta = local_weight * emitidos_local + (1.0 - local_weight) * ref_emitidos_per_acta
            elif ref_emitidos_per_acta > 0:
                emitidos_per_acta = ref_emitidos_per_acta

            valid_ratio = valid_ratio_local
            if ref_valid_ratio > 0 and valid_ratio_local > 0:
                valid_ratio = local_weight * valid_ratio_local + (1.0 - local_weight) * ref_valid_ratio
            elif ref_valid_ratio > 0:
                valid_ratio = ref_valid_ratio

            if emitidos_per_acta > 0 and valid_ratio > 0:
                return int(round(emitidos_per_acta * self.actas_faltantes * valid_ratio))

        if ref_emitidos_per_acta > 0 and ref_valid_ratio > 0:
            return int(round(ref_emitidos_per_acta * self.actas_faltantes * ref_valid_ratio))
        if self.ref_validos_por_acta > 0:
            return int(round(self.ref_validos_por_acta * self.actas_faltantes))
        if self.padron > 0:
            participacion = 0.55
            return int(round(self.padron * participacion * (self.actas_faltantes / self.actas_totales)))
        return 0

    def votos_por_candidato(self) -> dict[str, int]:
        return {c: int(v) for c, v in zip(self.candidatos, self.votos)}

    def to_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "ubigeo": self.ubigeo,
            "ubigeo_region": self.ubigeo_region,
            "ambito": self.ambito,
            "geo_level": self.geo_level,
            "actas_contabilizadas": self.actas_contabilizadas,
            "actas_totales": self.actas_totales,
            "pct_actas": self.pct_actas,
            "padron": self.padron,
            "votos_validos": self.votos_validos,
            "votos_emitidos": self.votos_emitidos,
            "participacion_ciudadana": self.participacion_ciudadana,
            "ref_validos_por_acta": self.ref_validos_por_acta,
            "ref_emitidos_por_acta": self.ref_emitidos_por_acta,
            "ref_valid_ratio": self.ref_valid_ratio,
            "votos_por_candidato": self.votos_por_candidato(),
        }


def _inyectar_referencias_validos_por_acta(regiones: list[Region]) -> None:
    """
    Estima una referencia local de votos válidos por acta para unidades sin data.

    En vez de usar solo el promedio nacional/ambito, toma primero una mediana
    "chica" dentro de la misma region madre. El tamaño de la unidad se aproxima
    con `actas_totales`: las unidades con menos actas suelen parecerse más a las
    zonas tardías/rurales que terminan entrando al final.
    """
    scoped: dict[tuple[int, int], list[tuple[int, float]]] = {}
    ambito_vpas: dict[int, list[float]] = {}
    global_vpas: list[float] = []

    for r in regiones:
        if r.actas_contabilizadas <= 0 or r.votos_validos <= 0:
            continue
        vpa = r.votos_validos / r.actas_contabilizadas
        scope = (r.ambito, 0 if r.ambito == 2 else r.ubigeo_region)
        scoped.setdefault(scope, []).append((max(r.actas_totales, r.actas_contabilizadas, 1), vpa))
        ambito_vpas.setdefault(r.ambito, []).append(vpa)
        global_vpas.append(vpa)

    ref_global = float(np.median(global_vpas)) if global_vpas else 0.0
    ambito_ref = {
        ambito: float(np.median(values))
        for ambito, values in ambito_vpas.items()
        if values
    }

    scope_ref: dict[tuple[int, int], float] = {}
    for scope, rows in scoped.items():
        rows_sorted = sorted(rows, key=lambda item: item[0])
        cut = max(1, int(round(len(rows_sorted) * LOCAL_VPA_SMALL_UNIT_SHARE)))
        if len(rows_sorted) >= LOCAL_VPA_MIN_UNITS:
            chosen = [vpa for _, vpa in rows_sorted[:cut]]
        else:
            chosen = [vpa for _, vpa in rows_sorted]
        scope_ref[scope] = float(np.median(chosen))

    for r in regiones:
        scope = (r.ambito, 0 if r.ambito == 2 else r.ubigeo_region)
        r.ref_validos_por_acta = scope_ref.get(scope, ambito_ref.get(r.ambito, ref_global))


def _inyectar_referencias_turnout(
    regiones: list[Region],
    parent_refs: dict[tuple[int, int], dict[str, float]] | None = None,
) -> None:
    parent_refs = parent_refs or {}
    scoped_emitidos: dict[tuple[int, int], list[float]] = {}
    scoped_valid_ratio: dict[tuple[int, int], list[float]] = {}
    ambito_emitidos: dict[int, list[float]] = {}
    ambito_valid_ratio: dict[int, list[float]] = {}

    for r in regiones:
        if r.actas_contabilizadas <= 0:
            continue
        emitidos_per_acta = r.emitidos_por_acta_obs
        valid_ratio = r.valid_ratio_obs
        scope = (r.ambito, 0 if r.ambito == 2 else r.ubigeo_region)
        if emitidos_per_acta > 0:
            scoped_emitidos.setdefault(scope, []).append(emitidos_per_acta)
            ambito_emitidos.setdefault(r.ambito, []).append(emitidos_per_acta)
        if valid_ratio > 0:
            scoped_valid_ratio.setdefault(scope, []).append(valid_ratio)
            ambito_valid_ratio.setdefault(r.ambito, []).append(valid_ratio)

    ambito_emit_ref = {k: float(np.median(v)) for k, v in ambito_emitidos.items() if v}
    ambito_ratio_ref = {k: float(np.median(v)) for k, v in ambito_valid_ratio.items() if v}
    global_emit_ref = float(np.median([x for v in ambito_emitidos.values() for x in v])) if ambito_emitidos else 0.0
    global_ratio_ref = float(np.median([x for v in ambito_valid_ratio.values() for x in v])) if ambito_valid_ratio else 0.90

    for r in regiones:
        scope = (r.ambito, 0 if r.ambito == 2 else r.ubigeo_region)
        parent = parent_refs.get((r.ambito, r.ubigeo), parent_refs.get(scope, {}))
        emit_ref = None
        ratio_ref = None
        if scoped_emitidos.get(scope):
            emit_ref = float(np.median(scoped_emitidos[scope]))
        if scoped_valid_ratio.get(scope):
            ratio_ref = float(np.median(scoped_valid_ratio[scope]))
        r.ref_emitidos_por_acta = (
            emit_ref
            or parent.get("emitidos_por_acta", 0.0)
            or ambito_emit_ref.get(r.ambito, global_emit_ref)
        )
        r.ref_valid_ratio = clip01(
            ratio_ref
            or parent.get("valid_ratio", 0.0)
            or ambito_ratio_ref.get(r.ambito, global_ratio_ref)
        )


def _infer_total_actas(contab: int, pct_actas: float) -> int:
    return int(round(contab / (pct_actas / 100))) if pct_actas > 0 else contab


def _fetch_region_acta_totals() -> dict[int, dict[str, int]]:
    rows = fetch("resumen-general/mapa-calor", tipoFiltro="ambito_geografico")
    out: dict[int, dict[str, int]] = {}
    for row in rows:
        if int(row.get("ambitoGeografico", 1)) != 1:
            continue
        ubigeo = int(row.get("ubigeoNivel01") or 0)
        contab = int(row.get("actasContabilizadas", 0))
        p_actas = float(row.get("porcentajeActasContabilizadas", 0.0))
        out[ubigeo] = {
            "actas_contabilizadas": contab,
            "actas_totales": _infer_total_actas(contab, p_actas),
        }
    return out


def _make_turnout_ref(totals: dict | None) -> dict[str, float]:
    if not totals:
        return {}
    total_actas = int(totals.get("totalActas", 0) or 0)
    total_emitidos = int(totals.get("totalVotosEmitidos", 0) or 0)
    total_validos = int(totals.get("totalVotosValidos", 0) or 0)
    emitidos_por_acta = total_emitidos / total_actas if total_actas > 0 else 0.0
    valid_ratio = clip01(total_validos / total_emitidos) if total_emitidos > 0 else 0.0
    participacion = float(totals.get("participacionCiudadana", 0.0) or 0.0)
    return {
        "emitidos_por_acta": emitidos_por_acta,
        "valid_ratio": valid_ratio,
        "participacion_ciudadana": participacion,
        "votos_emitidos": total_emitidos,
        "votos_validos": total_validos,
    }


def _fetch_region_turnout_refs() -> dict[tuple[int, int], dict[str, float]]:
    refs: dict[tuple[int, int], dict[str, float]] = {}
    rows = fetch("resumen-general/mapa-calor", tipoFiltro="ambito_geografico")
    for row in rows:
        ambito = int(row.get("ambitoGeografico", 1) or 1)
        if ambito != 1:
            continue
        ubigeo = int(row.get("ubigeoNivel01") or 0)
        totals = fetch_totales(tipoFiltro="ubigeo_nivel_01", idUbigeoDepartamento=ubigeo, idAmbitoGeografico=1)
        refs[(1, ubigeo)] = _make_turnout_ref(totals)
    totals_ext = fetch_totales(tipoFiltro="ambito_geografico", idAmbitoGeografico=2)
    refs[(2, 0)] = _make_turnout_ref(totals_ext)
    return refs


def _align_group_totals_to_regions(
    grouped: dict[tuple[int, int, int | None], dict],
    region_totals: dict[int, dict[str, int]],
) -> None:
    """
    Alinea los totales de actas del nivel fino con el total regional reportado
    por ONPE. Esto evita que provincias/distritos con 0% de actas queden con
    tamanio cero por falta de un denominador propio.
    """
    keys_by_region: dict[int, list[tuple[int, int, int | None]]] = {}
    for key in grouped:
        keys_by_region.setdefault(key[0], []).append(key)

    for ubigeo_region, keys in keys_by_region.items():
        parent = region_totals.get(ubigeo_region)
        if not parent:
            continue
        target_total = int(parent.get("actas_totales", 0))
        if target_total <= 0:
            continue

        total_subunits = sum(max(int(grouped[k].get("subunits", 0)), 1) for k in keys)
        known_total = sum(max(int(grouped[k].get("actas_totales", 0)), 0) for k in keys)
        known_subunits = sum(
            max(int(grouped[k].get("subunits", 0)), 1)
            for k in keys
            if int(grouped[k].get("actas_totales", 0)) > 0
        )
        avg_per_subunit = (
            known_total / known_subunits
            if known_subunits > 0
            else target_total / max(total_subunits, 1)
        )

        weights: list[float] = []
        min_sum = 0
        for key in keys:
            slot = grouped[key]
            contab = int(slot.get("actas_contabilizadas", 0))
            subunits = max(int(slot.get("subunits", 0)), 1)
            base_total = int(slot.get("actas_totales", 0))
            if base_total <= 0:
                base = avg_per_subunit * subunits
            else:
                base = float(base_total)
            base = max(base, float(contab))
            weights.append(base)
            min_sum += contab

        if target_total <= min_sum:
            for key in keys:
                grouped[key]["actas_totales"] = int(grouped[key].get("actas_contabilizadas", 0))
            continue

        extra_target = target_total - min_sum
        weight_sum = sum(weights)
        if weight_sum <= 0:
            weight_sum = float(len(keys))
            weights = [1.0] * len(keys)

        extras = [extra_target * w / weight_sum for w in weights]
        floors = [int(x) for x in extras]
        remainder = extra_target - sum(floors)
        order = sorted(
            range(len(keys)),
            key=lambda i: (extras[i] - floors[i], weights[i]),
            reverse=True,
        )
        for idx in order[:remainder]:
            floors[idx] += 1

        for i, key in enumerate(keys):
            grouped[key]["actas_totales"] = int(grouped[key].get("actas_contabilizadas", 0)) + floors[i]


def _cargar_extranjero(turnout_ref: dict[str, float] | None = None) -> Region:
    mc = fetch("resumen-general/mapa-calor", tipoFiltro="ambito_geografico")
    items = list(mc)
    if not any(it.get("ambitoGeografico") == 2 for it in items):
        items += fetch("resumen-general/mapa-calor", tipoFiltro="ambito_geografico", idAmbitoGeografico=2)
    contab = 0
    totales = 0
    for it in items:
        if int(it["ambitoGeografico"]) != 2:
            continue
        c = int(it.get("actasContabilizadas", 0))
        p = float(it.get("porcentajeActasContabilizadas", 0.0))
        contab += c
        totales += _infer_total_actas(c, p)

    print("  . extranjero (consolidado)...", file=sys.stderr, end=" ")
    data = fetch(
        "eleccion-presidencial/participantes-ubicacion-geografica-nombre",
        tipoFiltro="ambito_geografico",
        idAmbitoGeografico=2,
    )
    cands, votos = _extraer_candidatos(data)
    pct_ext = pct(contab, totales)
    print(f"{pct_ext:.1f}%", file=sys.stderr)
    return Region(
        nombre="EXTRANJERO",
        ubigeo=0,
        ubigeo_region=0,
        ambito=2,
        geo_level="foreign",
        actas_contabilizadas=contab,
        actas_totales=totales,
        pct_actas=pct_ext,
        padron=PADRON["EXTRANJERO"],
        candidatos=cands,
        votos=votos,
        votos_validos=int(votos.sum()),
        votos_emitidos=int((turnout_ref or {}).get("votos_emitidos", 0)),
        participacion_ciudadana=float((turnout_ref or {}).get("participacion_ciudadana", 0.0)),
        ref_emitidos_por_acta=float((turnout_ref or {}).get("emitidos_por_acta", 0.0)),
        ref_valid_ratio=float((turnout_ref or {}).get("valid_ratio", 0.0)),
    )


def cargar_regiones() -> list[Region]:
    print("-> mapa-calor (Peru)...", file=sys.stderr)
    mc_peru = fetch("resumen-general/mapa-calor", tipoFiltro="ambito_geografico")
    print("-> mapa-calor (Extranjero ya viene en la misma respuesta)", file=sys.stderr)

    items = list(mc_peru)
    if not any(it.get("ambitoGeografico") == 2 for it in items):
        items += fetch("resumen-general/mapa-calor", tipoFiltro="ambito_geografico", idAmbitoGeografico=2)

    turnout_refs = _fetch_region_turnout_refs()
    regiones: list[Region] = []
    extranjero_acc = {"contab": 0, "totales": 0}

    for it in items:
        ambito = int(it["ambitoGeografico"])
        ubigeo = int(it["ubigeoNivel01"])
        contab = int(it.get("actasContabilizadas", 0))
        p_actas = float(it.get("porcentajeActasContabilizadas", 0.0))
        totales = int(round(contab / (p_actas / 100))) if p_actas > 0 else contab
        if ambito == 2:
            extranjero_acc["contab"] += contab
            extranjero_acc["totales"] += totales
            continue

        print(f"  . ubigeo {ubigeo}...", file=sys.stderr, end=" ")
        time.sleep(0.1)
        data = fetch(
            "eleccion-presidencial/participantes-ubicacion-geografica-nombre",
            tipoFiltro="ubigeo_nivel_01",
            ubigeoNivel1=ubigeo,
            idAmbitoGeografico=1,
        )
        nombre = f"DEPT_{ubigeo:06d}"
        cands, votos = _extraer_candidatos(data)
        regiones.append(
            Region(
                nombre=nombre,
                ubigeo=ubigeo,
                ubigeo_region=ubigeo,
                ambito=1,
                geo_level="region",
                actas_contabilizadas=contab,
                actas_totales=totales,
                pct_actas=p_actas,
                padron=0,
                candidatos=cands,
                votos=votos,
                votos_validos=int(votos.sum()),
                votos_emitidos=int(turnout_refs.get((1, ubigeo), {}).get("votos_emitidos", 0)),
                participacion_ciudadana=float(turnout_refs.get((1, ubigeo), {}).get("participacion_ciudadana", 0.0)),
                ref_emitidos_por_acta=float(turnout_refs.get((1, ubigeo), {}).get("emitidos_por_acta", 0.0)),
                ref_valid_ratio=float(turnout_refs.get((1, ubigeo), {}).get("valid_ratio", 0.0)),
            )
        )
        print(f"{nombre} {p_actas:.1f}% ({len(cands)} cands, {int(votos.sum()):,} votos)", file=sys.stderr)

    regiones.append(_cargar_extranjero(turnout_ref=turnout_refs.get((2, 0), {})))

    _inyectar_referencias_turnout(regiones, turnout_refs)
    _inyectar_referencias_validos_por_acta(regiones)
    return regiones


def cargar_unidades(geo_level: str, max_workers: int = DEFAULT_MAX_WORKERS) -> list[Region]:
    if geo_level == "region":
        return cargar_regiones()

    fine_rows = fetch_fine_coverage()
    region_totals = _fetch_region_acta_totals()
    regiones: list[Region] = []
    grouped: dict[tuple[int, int, int | None], dict] = {}

    for row in fine_rows:
        if int(row.get("ambitoGeografico", 1)) != 1:
            continue
        ub1 = int(row.get("ubigeoNivel01") or 0)
        ub2 = int(row.get("ubigeoNivel02") or 0)
        ub3 = row.get("ubigeoNivel03")
        ub3 = int(ub3) if ub3 is not None else None
        contab = int(row.get("actasContabilizadas", 0))
        p_actas = float(row.get("porcentajeActasContabilizadas", 0.0))
        totales = _infer_total_actas(contab, p_actas)
        if geo_level == "province":
            key = (ub1, ub2, None)
            name = f"PROV_{ub2:06d}"
        elif geo_level == "district":
            if ub3 is None:
                continue
            key = (ub1, ub2, ub3)
            name = f"DIST_{ub3:06d}"
        else:
            raise ValueError(f"Nivel geografico no soportado: {geo_level}")

        slot = grouped.setdefault(
            key,
            {
                "nombre": name,
                "ubigeo_region": ub1,
                "ubigeo": ub2 if geo_level == "province" else ub3,
                "actas_contabilizadas": 0,
                "actas_totales": 0,
                "subunits": 0,
            },
        )
        slot["actas_contabilizadas"] += contab
        slot["actas_totales"] += totales
        slot["subunits"] += 1

    _align_group_totals_to_regions(grouped, region_totals)

    turnout_refs = _fetch_region_turnout_refs()
    province_turnout_refs: dict[tuple[int, int], dict[str, float]] = {}
    if geo_level == "district":
        province_keys = sorted({(key[0], key[1]) for key in grouped})
        print(f"-> turnout por provincia ({max_workers} workers)...", file=sys.stderr)

        def _province_turnout_worker(key: tuple[int, int]) -> tuple[tuple[int, int], dict[str, float]]:
            ub1, ub2 = key
            totals = fetch_totales(
                tipoFiltro="ubigeo_nivel_01",
                idUbigeoDepartamento=ub1,
                idUbigeoProvincia=ub2,
                idAmbitoGeografico=1,
            )
            return key, _make_turnout_ref(totals)

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {executor.submit(_province_turnout_worker, key): key for key in province_keys}
            done = 0
            total = len(futures)
            for future in as_completed(futures):
                done += 1
                if done == 1 or done % 50 == 0 or done == total:
                    print(f"  . turnout {done}/{total}", file=sys.stderr)
                key, ref = future.result()
                province_turnout_refs[key] = ref

    print(f"-> carga por {geo_level} ({max_workers} workers)...", file=sys.stderr)
    grouped_items = sorted(grouped.items())

    def _worker(item: tuple[tuple[int, int, int | None], dict]) -> Region:
        (ub1, ub2, ub3), meta = item
        unit_turnout_ref: dict[str, float] = {}
        if int(meta["actas_contabilizadas"]) <= 0:
            cands, votos = [], np.zeros(0, dtype=np.int64)
        else:
            params = {
                "tipoFiltro": "ubigeo_nivel_02" if geo_level == "province" else "ubigeo_nivel_03",
                "ubigeoNivel1": ub1,
                "ubigeoNivel2": ub2,
                "idAmbitoGeografico": 1,
            }
            if geo_level == "district":
                params["ubigeoNivel3"] = ub3
            data = fetch("eleccion-presidencial/participantes-ubicacion-geografica-nombre", **params)
            cands, votos = _extraer_candidatos(data)
        if geo_level == "province":
            unit_turnout_ref = _make_turnout_ref(
                fetch_totales(
                    tipoFiltro="ubigeo_nivel_01",
                    idUbigeoDepartamento=ub1,
                    idUbigeoProvincia=ub2,
                    idAmbitoGeografico=1,
                )
            )
        elif geo_level == "district":
            unit_turnout_ref = province_turnout_refs.get((ub1, ub2), {})
        p_actas = pct(meta["actas_contabilizadas"], meta["actas_totales"])
        return Region(
            nombre=meta["nombre"],
            ubigeo=int(meta["ubigeo"]),
            ubigeo_region=int(meta["ubigeo_region"]),
            ambito=1,
            geo_level=geo_level,
            actas_contabilizadas=int(meta["actas_contabilizadas"]),
            actas_totales=int(meta["actas_totales"]),
            pct_actas=p_actas,
            padron=0,
            candidatos=cands,
            votos=votos,
            votos_validos=int(votos.sum()),
            votos_emitidos=int(unit_turnout_ref.get("votos_emitidos", 0)) if geo_level == "province" else 0,
            participacion_ciudadana=float(unit_turnout_ref.get("participacion_ciudadana", 0.0)),
            ref_emitidos_por_acta=float(unit_turnout_ref.get("emitidos_por_acta", 0.0)),
            ref_valid_ratio=float(unit_turnout_ref.get("valid_ratio", 0.0)),
        )

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {executor.submit(_worker, item): item for item in grouped_items}
        done = 0
        total = len(futures)
        for future in as_completed(futures):
            done += 1
            if done == 1 or done % 100 == 0 or done == total:
                print(f"  . {done}/{total}", file=sys.stderr)
            regiones.append(future.result())

    regiones.sort(key=lambda r: (r.ambito, r.ubigeo_region, r.ubigeo))

    regiones.append(_cargar_extranjero(turnout_ref=turnout_refs.get((2, 0), {})))
    combined_turnout_refs = dict(turnout_refs)
    if geo_level == "district":
        combined_turnout_refs.update({(1, key[1]): value for key, value in province_turnout_refs.items() if value})
    _inyectar_referencias_turnout(regiones, combined_turnout_refs)
    _inyectar_referencias_validos_por_acta(regiones)
    return regiones


def _extraer_candidatos(data) -> tuple[list[str], np.ndarray]:
    """Devuelve lista de nombres de partido y array de votos."""
    if isinstance(data, dict):
        for k in ("participantes", "listaParticipantes", "data"):
            if k in data and isinstance(data[k], list):
                lst = data[k]
                break
        else:
            lst = next((v for v in data.values() if isinstance(v, list)), [])
    else:
        lst = data

    nombres, votos = [], []
    for c in lst:
        nom = c.get("nombreAgrupacionPolitica") or c.get("nombreCandidato") or "?"
        v = int(c.get("totalVotosValidos", 0))
        nombres.append(nom)
        votos.append(v)
    return nombres, np.array(votos, dtype=np.int64)


# --------- Geografia fina --------------------------------------------------


def fetch_fine_coverage(filter_name: str = FINE_GEO_FILTER) -> list[dict]:
    """
    Trae cobertura de actas a nivel fino.

    ONPE expone la cobertura fina via `mapa-calor`; luego esa estructura se
    cruza con la ruta de votos por candidato para armar unidades mas chicas
    que la region. Tambien la reutilizamos para medir cuan concentrado queda
    el faltante dentro de cada region.
    """
    data = fetch("resumen-general/mapa-calor", tipoFiltro=filter_name)
    return data if isinstance(data, list) else []


def summarize_fine_coverage(regiones: list[Region], fine_rows: list[dict]) -> list[dict]:
    refs: dict[int, dict] = {}
    for r in regiones:
        if r.ambito != 1:
            continue
        refs.setdefault(
            r.ubigeo_region,
            {"nombre": f"DEPT_{r.ubigeo_region:06d}", "ref_validos_por_acta": r.ref_validos_por_acta},
        )
    grouped: dict[int, list[tuple[int, float]]] = {}

    for row in fine_rows:
        if int(row.get("ambitoGeografico", 1)) != 1:
            continue
        ubigeo1 = int(row.get("ubigeoNivel01") or 0)
        parent = refs.get(ubigeo1)
        if not parent:
            continue
        contab = int(row.get("actasContabilizadas", 0))
        p_actas = float(row.get("porcentajeActasContabilizadas", 0.0))
        totales = int(round(contab / (p_actas / 100))) if p_actas > 0 else contab
        falt = max(totales - contab, 0)
        rem_valid = float(parent["ref_validos_por_acta"]) * falt
        grouped.setdefault(ubigeo1, []).append((falt, rem_valid))

    summary = []
    for ubigeo, rows in grouped.items():
        total_falt = sum(f for f, _ in rows)
        total_rem_valid = sum(v for _, v in rows)
        if total_falt <= 0:
            continue
        falt_sorted = sorted((f for f, _ in rows), reverse=True)
        top10_share = sum(falt_sorted[:10]) / total_falt if total_falt else 0.0
        hhi = sum((f / total_falt) ** 2 for f in falt_sorted if f > 0)
        zero_units = sum(1 for f, _ in rows if f > 0)
        # 0 = faltante muy disperso; 1 = faltante muy concentrado.
        concentration_score = clip01(0.6 * top10_share + 0.4 * min(hhi * len(rows), 1.0))
        summary.append(
            {
                "region": refs[ubigeo]["nombre"],
                "ubigeo": ubigeo,
                "units": len(rows),
                "units_with_remaining": zero_units,
                "remaining_actas": total_falt,
                "remaining_valid_est": total_rem_valid,
                "top10_share": top10_share,
                "hhi": hhi,
                "concentration_score": concentration_score,
            }
        )

    return sorted(summary, key=lambda x: (-x["concentration_score"], -x["remaining_valid_est"]))


def imprimir_fine_coverage_summary(summary: list[dict], topn: int = 12) -> None:
    print()
    print("Radar de faltante fino (concentracion del faltante dentro de cada region)")
    print("-" * 108)
    print(
        f"{'Region':<20} {'Actas falt.':>12} {'Validos est.':>14} {'Subunids':>10} "
        f"{'Activas':>8} {'Top10%':>8} {'Score':>8}"
    )
    print("-" * 108)
    for row in summary[:topn]:
        print(
            f"{row['region'][:20]:<20} "
            f"{row['remaining_actas']:>12,} "
            f"{row['remaining_valid_est']:>14,.0f} "
            f"{row['units']:>10} "
            f"{row['units_with_remaining']:>8} "
            f"{row['top10_share']*100:>7.1f}% "
            f"{row['concentration_score']:>7.2f}"
        )
    print()
    print("Interpretacion: score alto = el faltante de esa region esta mas concentrado en pocas subunidades.")
    print("Eso no cambia el porcentaje esperado por candidato, pero si reduce nuestra confianza en extrapolar linealmente la region completa.")


# --------- Snapshots e incrementales --------------------------------------


def snapshot_from_regiones(regiones: list[Region]) -> dict:
    actas_contadas = sum(r.actas_contabilizadas for r in regiones)
    actas_totales = sum(r.actas_totales for r in regiones)
    votos_validos = sum(r.votos_validos for r in regiones)
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "actas_contadas": actas_contadas,
        "actas_totales": actas_totales,
        "votos_validos": votos_validos,
        "regions": {
            region_key(r.ambito, r.ubigeo): {
                "nombre": r.nombre,
                "ambito": r.ambito,
                "ubigeo": r.ubigeo,
                "ubigeo_region": r.ubigeo_region,
                "geo_level": r.geo_level,
                "actas_contabilizadas": r.actas_contabilizadas,
                "actas_totales": r.actas_totales,
                "pct_actas": r.pct_actas,
                "votos_validos": r.votos_validos,
                "votos_emitidos": r.votos_emitidos,
                "participacion_ciudadana": r.participacion_ciudadana,
                "ref_validos_por_acta": r.ref_validos_por_acta,
                "ref_emitidos_por_acta": r.ref_emitidos_por_acta,
                "ref_valid_ratio": r.ref_valid_ratio,
                "votos_por_candidato": r.votos_por_candidato(),
            }
            for r in regiones
        },
    }


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"snapshots": []}
    try:
        with path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"snapshots": []}
    snapshots = data.get("snapshots", [])
    return {"snapshots": snapshots if isinstance(snapshots, list) else []}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def append_snapshot(state: dict, snapshot: dict, keep: int) -> None:
    state.setdefault("snapshots", []).append(snapshot)
    state["snapshots"] = state["snapshots"][-keep:]


def snapshot_changed(prev: dict, curr: dict) -> bool:
    if curr.get("votos_validos", 0) != prev.get("votos_validos", 0):
        return True
    if curr.get("actas_contadas", 0) != prev.get("actas_contadas", 0):
        return True
    return curr.get("regions", {}) != prev.get("regions", {})


def diff_snapshots(base: dict, curr: dict) -> dict:
    out = {
        "base_at": base.get("captured_at"),
        "curr_at": curr.get("captured_at"),
        "blocks": {
            key: {
                "prev_validos": 0,
                "curr_validos": 0,
                "new_validos": 0,
                "candidates": {},
            }
            for key in ("regiones", "lima", "extranjero", "total")
        },
    }

    all_keys = set(base.get("regions", {})) | set(curr.get("regions", {}))
    for key in all_keys:
        prev_r = base.get("regions", {}).get(key, {})
        curr_r = curr.get("regions", {}).get(key, {})
        ambito = curr_r.get("ambito", prev_r.get("ambito", 1))
        ubigeo_region = curr_r.get("ubigeo_region", prev_r.get("ubigeo_region", curr_r.get("ubigeo", prev_r.get("ubigeo", 0))))
        b_key = block_key(ambito, int(ubigeo_region or 0))

        prev_validos = int(prev_r.get("votos_validos", 0))
        curr_validos = int(curr_r.get("votos_validos", 0))
        new_validos = max(curr_validos - prev_validos, 0)

        for k in (b_key, "total"):
            out["blocks"][k]["prev_validos"] += prev_validos
            out["blocks"][k]["curr_validos"] += curr_validos
            out["blocks"][k]["new_validos"] += new_validos

        prev_votes = prev_r.get("votos_por_candidato", {})
        curr_votes = curr_r.get("votos_por_candidato", {})
        for cand in set(prev_votes) | set(curr_votes):
            prev_v = int(prev_votes.get(cand, 0))
            curr_v = int(curr_votes.get(cand, 0))
            new_v = max(curr_v - prev_v, 0)
            for k in (b_key, "total"):
                slot = out["blocks"][k]["candidates"].setdefault(cand, {"prev": 0, "curr": 0, "new": 0})
                slot["prev"] += prev_v
                slot["curr"] += curr_v
                slot["new"] += new_v

    return out


def _empty_late_scope() -> dict:
    return {
        "prev_validos": 0.0,
        "new_validos": 0.0,
        "candidate_prev": {},
        "candidate_new": {},
    }


def _accumulate_scope(
    scope: dict,
    prev_validos: int,
    new_validos: int,
    prev_votes: dict,
    curr_votes: dict,
    weight: float,
) -> None:
    scope["prev_validos"] += prev_validos * weight
    scope["new_validos"] += new_validos * weight
    for cand in set(prev_votes) | set(curr_votes):
        prev_v = int(prev_votes.get(cand, 0))
        curr_v = int(curr_votes.get(cand, 0))
        new_v = max(curr_v - prev_v, 0)
        scope["candidate_prev"][cand] = scope["candidate_prev"].get(cand, 0.0) + prev_v * weight
        scope["candidate_new"][cand] = scope["candidate_new"].get(cand, 0.0) + new_v * weight


def _stabilized_ratio(prev_votes: float, prev_validos: float, new_votes: float, new_validos: float, min_validos: float) -> tuple[float, float]:
    if prev_validos <= 0 or new_validos <= 0:
        return 1.0, 0.0
    prev_share = prev_votes / prev_validos if prev_validos else 0.0
    new_share = new_votes / new_validos if new_validos else 0.0
    if prev_share <= 0 or new_share <= 0:
        return 1.0, 0.0
    raw = np.clip(new_share / max(prev_share, 1e-4), 0.35, 2.8)
    confidence = 1.0 - np.exp(-new_validos / max(min_validos, 1.0))
    ratio = float(np.exp(np.log(raw) * confidence))
    return float(np.clip(ratio, 0.55, 1.85)), clip01(confidence)


def _finalize_scope_ratios(scopes: dict, min_validos: float) -> tuple[dict, dict]:
    ratios: dict = {}
    confidence: dict = {}
    for key, scope in scopes.items():
        scope_ratios = {}
        scope_conf = 0.0
        for cand in set(scope["candidate_prev"]) | set(scope["candidate_new"]):
            ratio, conf = _stabilized_ratio(
                scope["candidate_prev"].get(cand, 0),
                scope["prev_validos"],
                scope["candidate_new"].get(cand, 0),
                scope["new_validos"],
                min_validos,
            )
            scope_ratios[cand] = ratio
            scope_conf = max(scope_conf, conf)
        ratios[key] = scope_ratios
        confidence[key] = scope_conf
    return ratios, confidence


def _finalize_scope_shares(scopes: dict, min_validos: float) -> tuple[dict, dict]:
    shares: dict = {}
    confidence: dict = {}
    for key, scope in scopes.items():
        new_validos = float(scope.get("new_validos", 0.0))
        if new_validos <= 0:
            continue
        cand_shares = {
            cand: max(float(votes) / new_validos, 0.0)
            for cand, votes in scope.get("candidate_new", {}).items()
            if float(votes) > 0
        }
        total_share = float(sum(cand_shares.values()))
        if total_share <= 0:
            continue
        shares[key] = {cand: value / total_share for cand, value in cand_shares.items()}
        confidence[key] = clip01(1.0 - np.exp(-new_validos / max(min_validos, 1.0)))
    return shares, confidence


def build_late_bias_model(
    regiones: list[Region],
    history_snapshots: list[dict],
    current_snapshot: dict,
    window: int,
    strength: float,
    recency_decay: float,
) -> dict | None:
    """
    Aprende un sesgo tardio no lineal desde cortes sucesivos:
    - bloque: regiones sin Lima / Lima / extranjero
    - departamento: shrinkage hacia el bloque cuando la muestra es chica
    - progreso de actas: el efecto cae de forma no lineal cuando la unidad ya esta muy avanzada
    """
    if strength <= 0:
        return None

    snapshots = list(history_snapshots or [])
    if not snapshots or snapshot_changed(snapshots[-1], current_snapshot):
        snapshots.append(current_snapshot)
    if len(snapshots) < 2:
        return None

    selected = snapshots[-max(window, 2) :]
    block_scopes = {key: _empty_late_scope() for key in ("regiones", "lima", "extranjero")}
    region_scopes: dict[int, dict] = {}

    pairs = list(zip(selected[:-1], selected[1:]))
    for pair_idx, (base, curr) in enumerate(pairs):
        weight = float(recency_decay ** (len(pairs) - pair_idx - 1))
        all_keys = set(base.get("regions", {})) | set(curr.get("regions", {}))
        for key in all_keys:
            prev_r = base.get("regions", {}).get(key, {})
            curr_r = curr.get("regions", {}).get(key, {})
            ambito = int(curr_r.get("ambito", prev_r.get("ambito", 1)) or 1)
            ubigeo = int(curr_r.get("ubigeo", prev_r.get("ubigeo", 0)) or 0)
            ubigeo_region = int(
                curr_r.get(
                    "ubigeo_region",
                    prev_r.get("ubigeo_region", infer_region_code(ambito, ubigeo)),
                )
                or 0
            )
            b_key = block_key(ambito, ubigeo_region)

            prev_validos = int(prev_r.get("votos_validos", 0))
            curr_validos = int(curr_r.get("votos_validos", 0))
            new_validos = max(curr_validos - prev_validos, 0)
            prev_votes = prev_r.get("votos_por_candidato", {})
            curr_votes = curr_r.get("votos_por_candidato", {})

            _accumulate_scope(block_scopes[b_key], prev_validos, new_validos, prev_votes, curr_votes, weight)
            if ambito == 1:
                scope = region_scopes.setdefault(ubigeo_region, _empty_late_scope())
                _accumulate_scope(scope, prev_validos, new_validos, prev_votes, curr_votes, weight)

    block_ratios, block_conf = _finalize_scope_ratios(block_scopes, min_validos=10_000.0)
    region_raw_ratios, region_conf_raw = _finalize_scope_ratios(region_scopes, min_validos=4_000.0)
    block_recent_share, block_recent_conf = _finalize_scope_shares(block_scopes, min_validos=18_000.0)
    region_recent_raw, region_recent_conf_raw = _finalize_scope_shares(region_scopes, min_validos=7_500.0)

    region_ratios: dict[int, dict[str, float]] = {}
    region_conf: dict[int, float] = {}
    region_recent_share: dict[int, dict[str, float]] = {}
    region_recent_conf: dict[int, float] = {}
    for ubigeo_region, ratios in region_raw_ratios.items():
        b_key = block_key(1, ubigeo_region)
        local_conf = 0.55 * region_conf_raw.get(ubigeo_region, 0.0)
        blended = {}
        for cand in set(block_ratios.get(b_key, {})) | set(ratios):
            b_ratio = block_ratios.get(b_key, {}).get(cand, 1.0)
            r_ratio = ratios.get(cand, b_ratio)
            blended[cand] = float(np.exp((1 - local_conf) * np.log(b_ratio) + local_conf * np.log(r_ratio)))
        region_ratios[ubigeo_region] = blended
        region_conf[ubigeo_region] = local_conf

    for ubigeo_region, shares in region_recent_raw.items():
        b_key = block_key(1, ubigeo_region)
        local_conf = 0.60 * region_recent_conf_raw.get(ubigeo_region, 0.0)
        blended = {}
        for cand in set(block_recent_share.get(b_key, {})) | set(shares):
            b_share = block_recent_share.get(b_key, {}).get(cand, 0.0)
            r_share = shares.get(cand, b_share)
            blended[cand] = max((1 - local_conf) * b_share + local_conf * r_share, 0.0)
        total_share = float(sum(blended.values()))
        if total_share > 0:
            region_recent_share[ubigeo_region] = {
                cand: value / total_share for cand, value in blended.items()
            }
            region_recent_conf[ubigeo_region] = local_conf

    return {
        "strength": clip01(strength),
        "snapshots_used": len(selected),
        "base_at": selected[0].get("captured_at"),
        "curr_at": selected[-1].get("captured_at"),
        "recency_decay": recency_decay,
        "block_ratios": block_ratios,
        "block_conf": block_conf,
        "block_recent_share": block_recent_share,
        "block_recent_conf": block_recent_conf,
        "region_ratios": region_ratios,
        "region_conf": region_conf,
        "region_recent_share": region_recent_share,
        "region_recent_conf": region_recent_conf,
    }


def resolve_candidates(regiones: list[Region], patterns: list[str]) -> list[str]:
    universe = []
    for r in regiones:
        for cand in r.candidatos:
            if cand not in universe:
                universe.append(cand)

    resolved = []
    seen = set()
    for pattern in patterns:
        p = norm(pattern)
        match = next((cand for cand in universe if p in norm(cand)), None)
        if match and match not in seen:
            resolved.append(match)
            seen.add(match)
    return resolved


def candidate_votes(region: Region, candidate: str) -> int:
    for cand, votos in zip(region.candidatos, region.votos):
        if cand == candidate:
            return int(votos)
    return 0


def current_block_summary(regiones: list[Region], tracked: list[str]) -> dict:
    blocks = {
        key: {
            "counted_validos": 0,
            "remaining_validos": 0,
            "final_validos": 0,
            "candidate_counted": {cand: 0.0 for cand in tracked},
            "candidate_base_proj": {cand: 0.0 for cand in tracked},
        }
        for key in ("regiones", "lima", "extranjero", "total")
    }

    for r in regiones:
        b_key = block_key(r.ambito, r.ubigeo_region)
        remaining = r.votos_faltantes_estimados()
        final_total = r.votos_validos + remaining
        for key in (b_key, "total"):
            blocks[key]["counted_validos"] += r.votos_validos
            blocks[key]["remaining_validos"] += remaining
            blocks[key]["final_validos"] += final_total
        for cand in tracked:
            counted = candidate_votes(r, cand)
            share = counted / r.votos_validos if r.votos_validos else 0.0
            base_proj = counted + share * remaining
            for key in (b_key, "total"):
                blocks[key]["candidate_counted"][cand] += counted
                blocks[key]["candidate_base_proj"][cand] += base_proj
    return blocks


def imprimir_incremental(delta: dict, tracked: list[str], title: str) -> None:
    print()
    print(title)
    print(f"  Base: {delta['base_at']}")
    print(f"  Corte: {delta['curr_at']}")
    print("-" * 94)
    print(f"{'Bloque':<20} {'Nuevos validos':>14} {'Candidato':<36} {'Acum %':>8} {'Nuevo %':>8} {'Delta':>8}")
    print("-" * 94)
    for b_key in ("regiones", "lima", "extranjero", "total"):
        block = delta["blocks"][b_key]
        curr_total = block["curr_validos"]
        new_total = block["new_validos"]
        if new_total <= 0:
            print(f"{block_label(b_key):<20} {new_total:>14,} {'(sin votos nuevos)':<36} {'-':>8} {'-':>8} {'-':>8}")
            continue
        first = True
        for cand in tracked:
            stats = block["candidates"].get(cand, {"curr": 0, "new": 0})
            curr_share = pct(stats["curr"], curr_total)
            new_share = pct(stats["new"], new_total)
            delta_pp = new_share - curr_share
            new_total_label = f"{new_total:,}" if first else ""
            print(
                f"{block_label(b_key) if first else '':<20} "
                f"{new_total_label:>14} "
                f"{cand[:36]:<36} "
                f"{curr_share:>7.2f}% "
                f"{new_share:>7.2f}% "
                f"{delta_pp:>+7.2f}"
            )
            first = False


def imprimir_proyeccion_tendencia(
    regiones: list[Region],
    tracked: list[str],
    recent_delta: dict,
    trend_weight: float,
) -> None:
    blocks = current_block_summary(regiones, tracked)
    print()
    print(f"Proyeccion ajustada por tendencia reciente (peso incremental = {trend_weight:.2f})")
    print("-" * 104)
    for cand in tracked:
        print()
        print(cand)
        print(
            f"{'Bloque':<20} {'Emitidos proj.':>14} {'Base %':>8} {'Reciente %':>10} "
            f"{'Ajustada %':>10} {'Base votos':>12} {'Ajustados':>12}"
        )
        print("-" * 104)
        for b_key in ("regiones", "lima", "extranjero", "total"):
            block = blocks[b_key]
            counted = block["candidate_counted"][cand]
            counted_total = block["counted_validos"]
            remaining = block["remaining_validos"]
            final_total = block["final_validos"]
            base_votes = block["candidate_base_proj"][cand]
            base_share = counted / counted_total if counted_total else 0.0
            delta_block = recent_delta["blocks"][b_key]
            recent_new_total = delta_block["new_validos"]
            recent_votes = delta_block["candidates"].get(cand, {"new": 0})["new"]
            recent_share = recent_votes / recent_new_total if recent_new_total else base_share
            adjusted_share = clip01(base_share + trend_weight * (recent_share - base_share))
            adjusted_votes = counted + adjusted_share * remaining
            print(
                f"{block_label(b_key):<20} {final_total:>14,.0f} "
                f"{pct(base_votes, final_total):>7.2f}% "
                f"{100.0 * recent_share:>9.2f}% "
                f"{pct(adjusted_votes, final_total):>9.2f}% "
                f"{base_votes:>12,.0f} {adjusted_votes:>12,.0f}"
            )


def manejar_track_state(
    regiones: list[Region],
    tracked: list[str],
    state_file: Path,
    trend_window: int,
    trend_weight: float,
    history_size: int,
) -> None:
    state = load_state(state_file)
    current = snapshot_from_regiones(regiones)
    snapshots = state.get("snapshots", [])
    prev = snapshots[-1] if snapshots else None

    if prev is None:
        append_snapshot(state, current, history_size)
        save_state(state_file, state)
        print()
        print(f"Estado inicial guardado en {state_file}")
        print("Vuelve a correr con --track-state cuando ONPE tenga un nuevo corte para ver incrementales.")
        return

    if not snapshot_changed(prev, current):
        print()
        print(f"Sin cambios reales desde el ultimo corte guardado ({prev.get('captured_at')}).")
        return

    latest_delta = diff_snapshots(prev, current)
    if len(snapshots) >= trend_window:
        trend_base = snapshots[-trend_window]
    else:
        trend_base = snapshots[0]
    recent_delta = diff_snapshots(trend_base, current)

    imprimir_incremental(latest_delta, tracked, "Votos nuevos desde el ultimo corte")
    if trend_base is not prev:
        imprimir_incremental(
            recent_delta,
            tracked,
            f"Tendencia acumulada en los ultimos {min(trend_window, len(snapshots))} cortes",
        )
    imprimir_proyeccion_tendencia(regiones, tracked, recent_delta, trend_weight)

    append_snapshot(state, current, history_size)
    save_state(state_file, state)
    print()
    print(f"Estado actualizado en {state_file}")


# --------- Modelo bayesiano ------------------------------------------------


def _late_bias_gate(region: Region) -> float:
    progress = region.pct_actas / 100.0
    remaining = region.votos_faltantes_estimados()
    total_est = region.votos_validos + remaining
    remaining_share = remaining / total_est if total_est > 0 else 0.0
    progress_gate = 1.0 - sigmoid((progress - 0.82) / 0.10)
    remaining_gate = min(max(remaining_share, 0.0) ** 0.40, 1.0)
    return clip01(progress_gate * remaining_gate)


def _late_momentum_gate(region: Region) -> float:
    progress = region.pct_actas / 100.0
    remaining = region.votos_faltantes_estimados()
    total_est = region.votos_validos + remaining
    remaining_share = remaining / total_est if total_est > 0 else 0.0
    progress_gate = 1.0 - sigmoid((progress - 0.90) / 0.06)
    remaining_gate = min(max(remaining_share, 0.0) ** 0.28, 1.0)
    return clip01(progress_gate * remaining_gate)


def expected_share_with_late_bias(
    region: Region,
    universe: list[str],
    base_share: np.ndarray,
    late_bias_model: dict | None,
) -> np.ndarray:
    if not late_bias_model:
        return base_share

    b_key = block_key(region.ambito, region.ubigeo_region)
    block_conf = late_bias_model.get("block_conf", {}).get(b_key, 0.0)
    region_conf = late_bias_model.get("region_conf", {}).get(region.ubigeo_region, block_conf)
    conf_gate = clip01(0.65 * block_conf + 0.35 * region_conf)
    gate = _late_bias_gate(region) * late_bias_model.get("strength", 0.0) * conf_gate
    if gate <= 0:
        return base_share

    block_ratios = late_bias_model.get("block_ratios", {}).get(b_key, {})
    region_ratios = late_bias_model.get("region_ratios", {}).get(region.ubigeo_region, {})

    weights = np.ones_like(base_share, dtype=np.float64)
    for i, cand in enumerate(universe):
        ratio = region_ratios.get(cand, block_ratios.get(cand, 1.0))
        weights[i] = ratio

    biased = base_share * weights
    biased_sum = float(biased.sum())
    if biased_sum <= 0:
        return base_share
    biased /= biased_sum
    mixed = (1.0 - gate) * base_share + gate * biased
    mixed /= mixed.sum()

    recent_block_share = late_bias_model.get("block_recent_share", {}).get(b_key, {})
    recent_region_share = late_bias_model.get("region_recent_share", {}).get(region.ubigeo_region, {})
    recent_block_conf = late_bias_model.get("block_recent_conf", {}).get(b_key, 0.0)
    recent_region_conf = late_bias_model.get("region_recent_conf", {}).get(region.ubigeo_region, recent_block_conf)
    recent_conf = clip01(max(recent_block_conf, 0.70 * recent_region_conf))
    momentum_gate = _late_momentum_gate(region) * late_bias_model.get("strength", 0.0) * recent_conf
    if momentum_gate <= 0:
        return mixed

    momentum = mixed.copy()
    for i, cand in enumerate(universe):
        recent_share = recent_region_share.get(cand, recent_block_share.get(cand, mixed[i]))
        momentum[i] = max(mixed[i] + LATE_MOMENTUM_BLEND * (recent_share - mixed[i]), 0.0)
    momentum_sum = float(momentum.sum())
    if momentum_sum <= 0:
        return mixed
    momentum /= momentum_sum
    mixed = (1.0 - momentum_gate) * mixed + momentum_gate * momentum
    mixed /= mixed.sum()
    return mixed


def proyectar(
    regiones: list[Region],
    draws: int = 10_000,
    alpha_prior: float = 1.0,
    swing: float = 0.03,
    late_bias_model: dict | None = None,
    rng=None,
):
    """
    Para cada region sampleamos draws veces:
        p ~ Dirichlet(alpha_prior + votos_obs * concentracion)
        v_falt ~ Multinomial(votos_faltantes_estimados, p)

    `swing` introduce overdispersion: el vector p para los votos faltantes no se
    asume identico al de los contados. Es una banda heuristica, no calibrada.
    """
    rng = rng or np.random.default_rng(42)
    concentracion = 1.0 / max(swing, 1e-6) ** 2 if swing > 0 else float("inf")

    universo = []
    for r in regiones:
        for c in r.candidatos:
            if c not in universo:
                universo.append(c)
    k = len(universo)
    idx = {c: i for i, c in enumerate(universo)}

    contado_total = np.zeros(k, dtype=np.int64)
    sims = np.zeros((draws, k), dtype=np.int64)

    for r in regiones:
        votos_r = np.zeros(k, dtype=np.int64)
        for c, v in zip(r.candidatos, r.votos):
            votos_r[idx[c]] = v
        contado_total += votos_r

        v_falt = r.votos_faltantes_estimados()
        if v_falt <= 0:
            continue

        total_r = float(votos_r.sum())
        base_share = (votos_r.astype(np.float64) + alpha_prior) / (total_r + alpha_prior * k) if total_r > 0 else np.full(k, 1.0 / k)
        expected_share = expected_share_with_late_bias(r, universo, base_share, late_bias_model)
        if total_r > 0 and np.isfinite(concentracion):
            alpha = alpha_prior + expected_share * concentracion
        else:
            alpha = alpha_prior + expected_share * max(concentracion if np.isfinite(concentracion) else k, k)
        p_draws = rng.dirichlet(alpha, size=draws)
        for i in range(draws):
            sims[i] += rng.multinomial(v_falt, p_draws[i])

    proyectado = sims + contado_total
    porcentajes = proyectado / proyectado.sum(axis=1, keepdims=True) * 100.0
    return universo, contado_total, proyectado, porcentajes


# --------- Reporte ---------------------------------------------------------


def imprimir_resumen(unidades, universo, contado, proy, pct_draws):
    total_actas = sum(r.actas_totales for r in unidades)
    actas_contadas = sum(r.actas_contabilizadas for r in unidades)
    pct_global = pct(actas_contadas, total_actas)
    domestic_levels = {r.geo_level for r in unidades if r.ambito == 1}
    unit_label = next(iter(domestic_levels)) if domestic_levels else "region"

    print()
    print("=" * 78)
    print("  Proyeccion eleccion presidencial Peru 2026 - 1ra vuelta")
    print(f"  Actas procesadas: {actas_contadas:,}/{total_actas:,} ({pct_global:.2f}%)")
    print(f"  Unidades analizadas ({unit_label}): {len(unidades)}  .  Draws: {pct_draws.shape[0]:,}")
    print("=" * 78)

    actual_pct = contado / contado.sum() * 100.0
    med = np.median(pct_draws, axis=0)
    lo = np.percentile(pct_draws, 2.5, axis=0)
    hi = np.percentile(pct_draws, 97.5, axis=0)
    orden = np.argsort(-med)

    print(f"\n{'Candidato/Partido':<38} {'Actual %':>9} {'Proy %':>9} {'IC 95%':>18}")
    print("-" * 78)
    for i in orden:
        ic = f"[{lo[i]:5.2f}, {hi[i]:5.2f}]"
        print(f"{universo[i][:38]:<38} {actual_pct[i]:>8.2f}% {med[i]:>8.2f}% {ic:>18}")

    excluir = {i for i, n in enumerate(universo) if "BLANCO" in n.upper() or "NULO" in n.upper()}
    pct_validos = pct_draws.copy()
    if excluir:
        pct_validos[:, list(excluir)] = -1
    top2 = np.argsort(-pct_validos, axis=1)[:, :2]
    counts = np.zeros(pct_draws.shape[1], dtype=np.int64)
    for row in top2:
        counts[row] += 1
    probs = counts / pct_draws.shape[0]
    print("\nProbabilidad de quedar en el top-2 (excl. blancos/nulos):")
    for i in np.argsort(-probs):
        if probs[i] < 0.005 or i in excluir:
            continue
        print(f"  {universo[i][:46]:<46} {probs[i]*100:6.1f}%")


def imprimir_tabla_regiones(regiones):
    unit_label = next((r.geo_level for r in regiones if r.ambito == 1), "region")
    print(f"\nDetalle por {unit_label} (% actas, votos validos contados, votos faltantes estimados):")
    print("-" * 78)
    print(f"{'Unidad':<20} {'%Actas':>8} {'Contados':>12} {'Faltantes est.':>16} {'Ref/Padron':>12}")
    for r in sorted(regiones, key=lambda x: (-x.votos_faltantes_estimados(), x.nombre)):
        ref = f"{r.ref_validos_por_acta:.1f}/a" if r.padron == 0 else f"{r.padron:,}"
        print(
            f"{r.nombre[:20]:<20} {r.pct_actas:>7.2f}% "
            f"{r.votos_validos:>12,} {r.votos_faltantes_estimados():>16,} {ref:>12}"
        )


def guardar_csv(path, regiones, universo, pct_draws):
    import csv

    med = np.median(pct_draws, axis=0)
    lo = np.percentile(pct_draws, 2.5, axis=0)
    hi = np.percentile(pct_draws, 97.5, axis=0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidato", "proy_pct_mediana", "ic95_lo", "ic95_hi"])
        for i, c in enumerate(universo):
            w.writerow([c, f"{med[i]:.4f}", f"{lo[i]:.4f}", f"{hi[i]:.4f}"])
        w.writerow([])
        w.writerow(["region", "ubigeo", "pct_actas", "votos_validos", "votos_faltantes_est", "ref_validos_por_acta", "padron"])
        for r in regiones:
            w.writerow(
                [
                    r.nombre,
                    r.ubigeo,
                    f"{r.pct_actas:.3f}",
                    r.votos_validos,
                    r.votos_faltantes_estimados(),
                    f"{r.ref_validos_por_acta:.4f}",
                    r.padron,
                ]
            )
    print(f"\n-> CSV guardado en {path}")


# --------- Main ------------------------------------------------------------


def parse_track_patterns(raw: str) -> list[str]:
    parts = [norm(p) for p in raw.split(",")]
    return [p for p in parts if p]


def default_state_path_for_level(geo_level: str) -> Path:
    suffix = "" if geo_level == "region" else f"_{geo_level}"
    return DEFAULT_STATE_FILE.with_name(f"{DEFAULT_STATE_FILE.stem}{suffix}{DEFAULT_STATE_FILE.suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10_000)
    ap.add_argument("--api-base", type=str, default=API, help="base del backend ONPE")
    ap.add_argument("--referer", type=str, default=DEFAULT_REFERER, help="referer HTTP para consultas ONPE")
    ap.add_argument("--election-id", type=int, default=ID_ELECCION, help="idEleccion ONPE")
    ap.add_argument(
        "--geo-level",
        choices=["region", "province", "district"],
        default=DEFAULT_GEO_LEVEL,
        help="nivel geografico para el modelo base; recomendado: province",
    )
    ap.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help="workers para fetch paralelo en provincia/distrito")
    ap.add_argument("--alpha", type=float, default=1.0, help="prior Dirichlet (1.0 = uniforme debil)")
    ap.add_argument(
        "--swing",
        type=float,
        default=0.03,
        help="overdispersion heuristica por candidato/region. 0=ingenuo, 0.03=aprox +/-3pp",
    )
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-raw", action="store_true", help="vuelca el JSON de cada region")
    ap.add_argument("--fine-coverage", action="store_true", help="muestra radar de concentracion del faltante con geografia mas fina")
    ap.add_argument("--fine-top", type=int, default=12, help="cuantas regiones listar en el radar de geografia fina")
    ap.add_argument("--track-state", action="store_true", help="compara contra el ultimo corte guardado y actualiza estado")
    ap.add_argument("--state-file", type=str, default=None)
    ap.add_argument("--history-size", type=int, default=96, help="cuantos snapshots guardar")
    ap.add_argument("--trend-window", type=int, default=4, help="cuantos cortes usar para suavizar incrementales")
    ap.add_argument("--late-bias-window", type=int, default=LATE_BIAS_DEFAULT_WINDOW, help="cuantos snapshots usar para aprender sesgo tardio no lineal")
    ap.add_argument("--late-bias-strength", type=float, default=LATE_BIAS_DEFAULT_STRENGTH, help="intensidad del sesgo tardio no lineal sobre el faltante (0..1)")
    ap.add_argument("--late-bias-recency", type=float, default=LATE_BIAS_DEFAULT_RECENCY, help="cuanto pesan mas los cortes recientes en el sesgo tardio (0..1)")
    ap.add_argument("--no-late-bias", action="store_true", help="desactiva el sesgo tardio no lineal y vuelve al modelo base")
    ap.add_argument(
        "--trend-weight",
        type=float,
        default=0.50,
        help="peso heuristico de la tendencia reciente sobre el faltante (0..1)",
    )
    ap.add_argument(
        "--track",
        type=str,
        default=",".join(TRACK_DEFAULT),
        help="candidatos a monitorear por substring, separados por coma",
    )
    args = ap.parse_args()
    configure_api(api_base=args.api_base, election_id=args.election_id, referer=args.referer)
    state_file = Path(args.state_file) if args.state_file else default_state_path_for_level(args.geo_level)

    regiones = cargar_unidades(args.geo_level, max_workers=max(args.workers, 1))
    if args.dump_raw:
        print(json.dumps([r.to_dict() for r in regiones], ensure_ascii=False, indent=2))
        return

    state = load_state(state_file)
    current_snapshot = snapshot_from_regiones(regiones)
    late_bias_model = None
    if not args.no_late_bias:
        late_bias_model = build_late_bias_model(
            regiones=regiones,
            history_snapshots=state.get("snapshots", []),
            current_snapshot=current_snapshot,
            window=max(args.late_bias_window, 2),
            strength=clip01(args.late_bias_strength),
            recency_decay=clip01(args.late_bias_recency),
        )

    rng = np.random.default_rng(args.seed)
    universo, contado, proy, pct_draws = proyectar(
        regiones,
        draws=args.draws,
        alpha_prior=args.alpha,
        swing=args.swing,
        late_bias_model=late_bias_model,
        rng=rng,
    )
    imprimir_resumen(regiones, universo, contado, proy, pct_draws)
    if late_bias_model:
        print()
        print(
            "Sesgo tardio no lineal activo: "
            f"{late_bias_model['snapshots_used']} snapshots, "
            f"base {late_bias_model['base_at']} -> {late_bias_model['curr_at']}, "
            f"intensidad {late_bias_model['strength']:.2f}, "
            f"recencia {late_bias_model['recency_decay']:.2f}"
        )
    imprimir_tabla_regiones(regiones)
    if args.fine_coverage:
        fine_rows = fetch_fine_coverage()
        fine_summary = summarize_fine_coverage(regiones, fine_rows)
        imprimir_fine_coverage_summary(fine_summary, topn=max(args.fine_top, 1))
    if args.csv:
        guardar_csv(args.csv, regiones, universo, pct_draws)

    if args.track_state:
        tracked = resolve_candidates(regiones, parse_track_patterns(args.track))
        if not tracked:
            print("\nNo se pudieron resolver candidatos para monitoreo incremental.", file=sys.stderr)
            return
        manejar_track_state(
            regiones=regiones,
            tracked=tracked,
            state_file=state_file,
            trend_window=max(args.trend_window, 1),
            trend_weight=clip01(args.trend_weight),
            history_size=max(args.history_size, 2),
        )


if __name__ == "__main__":
    main()
