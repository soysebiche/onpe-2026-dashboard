"""
Proyección bayesiana v3 — nivel DISTRITO (2102 distritos + extranjero).

Mejoras sobre v2:
  · Granularidad a nivel distrito (2102 vs 196 provincias).
  · votos_por_acta: para distritos sin actas, usa mediana del departamento
    calculada sobre distritos chicos (rural proxy), no el promedio global.
    Esto replica el enfoque del repo jlrolando/Peru_elecciones2026 de excluir
    capitales del baseline.
  · Una sola llamada a mapa-calor trae todos los distritos (no hay loop).
  · ~2100 requests paralelos para votos (10 workers, ~30s total).

Uso:
    python3 onpe_proyeccion_v3.py
    python3 onpe_proyeccion_v3.py --draws 20000 --swing 0.03
    python3 onpe_proyeccion_v3.py --regiones
    python3 onpe_proyeccion_v3.py --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

# ── Config ──────────────────────────────────────────────────────────────────
API = "https://resultadoelectoral.onpe.gob.pe/presentacion-backend"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Referer": "https://resultadoelectoral.onpe.gob.pe/main/resumen",
    "Origin": "https://resultadoelectoral.onpe.gob.pe",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-PE,es;q=0.9",
}
ID_ELECCION = 10

RENIEC_CSV_PATHS = [
    Path(__file__).parent / "geodir-ubigeo-reniec.csv",
    Path("/tmp/ubigeo_reniec.csv"),
]

# Umbral de población para "distrito rural" (proxy para excluir capitales del baseline)
UMBRAL_RURAL = 15_000

# ── HTTP ─────────────────────────────────────────────────────────────────────
_session = requests.Session()
_session.headers.update(HEADERS)


def fetch(path: str, **params) -> object:
    params.setdefault("idEleccion", ID_ELECCION)
    url = f"{API}/{path.lstrip('/')}"
    for attempt in range(4):
        try:
            r = _session.get(url, params=params, timeout=30)
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if "application/json" not in ct:
                raise RuntimeError(f"HTML response for {url}")
            j = r.json()
            if not j.get("success", False):
                return j.get("data", [])
            return j["data"]
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(1.5 ** attempt)


# ── RENIEC ───────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").upper().strip()


def cargar_reniec() -> tuple[dict[int, str], dict[int, str], dict[int, str], dict[int, int]]:
    """
    dist_nombre:  ubigeoNivel3 → nombre distrito
    prov_nombre:  ubigeoNivel2 → nombre provincia
    dept_nombre:  ubigeoNivel1 → nombre departamento
    dist_pob:     ubigeoNivel3 → población
    """
    path = next((p for p in RENIEC_CSV_PATHS if p.exists()), None)
    if path is None:
        print("⚠ CSV RENIEC no encontrado — usando códigos numéricos", file=sys.stderr)
        return {}, {}, {}, {}

    dist_nombre: dict[int, str] = {}
    prov_nombre: dict[int, str] = {}
    dept_nombre: dict[int, str] = {}
    dist_pob: dict[int, int] = {}

    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw = row.get("Ubigeo", "").strip().zfill(6)
            if len(raw) < 6:
                continue
            d1 = int(raw[:2]) * 10_000
            d2 = int(raw[:4]) * 100
            d3 = int(raw[:6])          # RENIEC 6-digit == ONPE ubigeoNivel3 (int)
            dept_nombre.setdefault(d1, _norm(row.get("Departamento", "")))
            prov_nombre.setdefault(d2, _norm(row.get("Provincia", "")))
            dist_nombre[d3] = _norm(row.get("Distrito", ""))
            pob_str = row.get("Poblacion", "0").replace(",", "").strip()
            try:
                dist_pob[d3] = int(pob_str)
            except ValueError:
                dist_pob[d3] = 0

    return dist_nombre, prov_nombre, dept_nombre, dist_pob


# ── Data class ───────────────────────────────────────────────────────────────

@dataclass
class Distrito:
    ubigeo3: int          # ONPE ubigeoNivel3
    ubigeo2: int          # ONPE ubigeoNivel2 (provincia)
    ubigeo1: int          # ONPE ubigeoNivel1 (departamento)
    nombre: str
    prov: str
    dept: str
    ambito: int
    actas_cont: int
    actas_tot: int
    pct_actas: float
    poblacion: int
    candidatos: list[str] = field(default_factory=list)
    votos: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def votos_validos(self) -> int:
        return int(self.votos.sum())

    @property
    def votos_por_acta_obs(self) -> float:
        return self.votos_validos / self.actas_cont if self.actas_cont > 0 else 0.0

    @property
    def actas_faltantes(self) -> int:
        return max(self.actas_tot - self.actas_cont, 0)


# ── Carga de datos ───────────────────────────────────────────────────────────

def cargar_distritos(verbose: bool = True, max_workers: int = 12) -> list[Distrito]:
    dist_nombre, prov_nombre, dept_nombre, dist_pob = cargar_reniec()

    if verbose:
        print("→ Descargando mapa-calor nivel distrito (una sola llamada)…", file=sys.stderr)

    mc = fetch("resumen-general/mapa-calor", tipoFiltro="ubigeo_nivel_02")
    items_peru = [
        it for it in mc
        if it.get("ambitoGeografico") == 1 and it.get("ubigeoNivel03") is not None
    ]
    if verbose:
        print(f"   {len(items_peru)} distritos encontrados", file=sys.stderr)

    # ── Fetch paralelo de votos por distrito ──────────────────────────────
    if verbose:
        print(f"→ Fetching votos por distrito ({max_workers} workers)…", file=sys.stderr)

    resultados: dict[int, tuple[list[str], np.ndarray]] = {}

    def _worker(it):
        u1 = int(it["ubigeoNivel01"])
        u2 = int(it["ubigeoNivel02"])
        u3 = int(it["ubigeoNivel03"])
        # Solo fetch si hay actas contabilizadas; sin actas los votos serán 0
        if int(it.get("actasContabilizadas", 0)) == 0:
            return u3, ([], np.zeros(0, dtype=np.int64))
        data = fetch(
            "eleccion-presidencial/participantes-ubicacion-geografica-nombre",
            tipoFiltro="ubigeo_nivel_03",
            ubigeoNivel1=u1,
            ubigeoNivel2=u2,
            ubigeoNivel3=u3,
            idAmbitoGeografico=1,
        )
        cands, votos = _parse_candidatos(data)
        return u3, (cands, votos)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_worker, it): it for it in items_peru}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if verbose and done % 200 == 0:
                print(f"   {done}/{len(items_peru)}…", file=sys.stderr)
            try:
                u3, cv = fut.result()
                resultados[u3] = cv
            except Exception as e:
                it = futures[fut]
                print(f"\n   ⚠ error dist {it.get('ubigeoNivel03')}: {e}", file=sys.stderr)

    # ── Construir objetos Distrito ────────────────────────────────────────
    distritos: list[Distrito] = []
    for it in items_peru:
        u1 = int(it["ubigeoNivel01"])
        u2 = int(it["ubigeoNivel02"])
        u3 = int(it["ubigeoNivel03"])
        contab = int(it.get("actasContabilizadas", 0))
        pct = float(it.get("porcentajeActasContabilizadas", 0.0))
        tot = int(round(contab / (pct / 100))) if pct > 0 else 0
        cands, votos = resultados.get(u3, ([], np.zeros(0, dtype=np.int64)))
        distritos.append(Distrito(
            ubigeo3=u3, ubigeo2=u2, ubigeo1=u1,
            nombre=dist_nombre.get(u3, f"DIST_{u3}"),
            prov=prov_nombre.get(u2, f"PROV_{u2}"),
            dept=dept_nombre.get(u1, f"DEPT_{u1}"),
            ambito=1, actas_cont=contab, actas_tot=tot,
            pct_actas=pct, poblacion=dist_pob.get(u3, 0),
            candidatos=cands, votos=votos,
        ))

    # ── Extranjero ────────────────────────────────────────────────────────
    if verbose:
        print("→ Fetching extranjero…", file=sys.stderr)
    ext_items = [it for it in mc if it.get("ambitoGeografico") == 2]
    ext_contab = sum(int(it.get("actasContabilizadas", 0)) for it in ext_items)
    ext_tot = sum(
        int(round(int(it["actasContabilizadas"]) / (it["porcentajeActasContabilizadas"] / 100)))
        for it in ext_items if it.get("porcentajeActasContabilizadas", 0) > 0
    )
    ext_pct = 100.0 * ext_contab / ext_tot if ext_tot else 0.0
    ext_data = fetch(
        "eleccion-presidencial/participantes-ubicacion-geografica-nombre",
        tipoFiltro="ambito_geografico", idAmbitoGeografico=2,
    )
    ext_cands, ext_votos = _parse_candidatos(ext_data)
    distritos.append(Distrito(
        ubigeo3=900000, ubigeo2=900000, ubigeo1=900000,
        nombre="EXTRANJERO", prov="EXTRANJERO", dept="EXTRANJERO",
        ambito=2, actas_cont=ext_contab, actas_tot=ext_tot,
        pct_actas=ext_pct, poblacion=1_210_813,
        candidatos=ext_cands, votos=ext_votos,
    ))

    if verbose:
        total = sum(d.actas_tot for d in distritos)
        cont = sum(d.actas_cont for d in distritos)
        n_sin = sum(1 for d in distritos if d.actas_cont == 0)
        print(
            f"→ Listo: {len(distritos)} distritos · "
            f"{cont:,}/{total:,} actas ({100*cont/total:.2f}%) · "
            f"{n_sin} distritos sin actas",
            file=sys.stderr,
        )
    return distritos


def _parse_candidatos(data) -> tuple[list[str], np.ndarray]:
    lst = data if isinstance(data, list) else []
    cands, votos = [], []
    for c in lst:
        nom = c.get("nombreAgrupacionPolitica") or c.get("nombreCandidato") or "?"
        cands.append(nom)
        votos.append(int(c.get("totalVotosValidos", 0)))
    return cands, np.array(votos, dtype=np.int64)


# ── votos_por_acta baseline rural ───────────────────────────────────────────

def calcular_vpa_rural(distritos: list[Distrito]) -> dict[int, float]:
    """
    Para cada departamento, calcula la mediana de votos_por_acta de los distritos
    "rurales" (población < UMBRAL_RURAL y actas_cont > 0).
    Usado como estimador de votos_faltantes para distritos sin datos.
    Excluye capitales (distritos grandes) del baseline — réplica del enfoque jlrolando.
    """
    dept_vpas: dict[int, list[float]] = defaultdict(list)
    for d in distritos:
        if d.ambito == 2 or d.actas_cont == 0:
            continue
        es_rural = d.poblacion < UMBRAL_RURAL or d.poblacion == 0
        if es_rural:
            dept_vpas[d.ubigeo1].append(d.votos_por_acta_obs)

    dept_vpa_mediana: dict[int, float] = {}
    global_vpas = []
    for u1, vpas in dept_vpas.items():
        m = float(np.median(vpas))
        dept_vpa_mediana[u1] = m
        global_vpas.extend(vpas)

    fallback = float(np.median(global_vpas)) if global_vpas else 250.0
    return dept_vpa_mediana, fallback


# ── Modelo bayesiano ─────────────────────────────────────────────────────────

def proyectar(
    distritos: list[Distrito],
    draws: int = 10_000,
    alpha_prior: float = 1.0,
    swing: float = 0.03,
    rng=None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(42)
    concentracion = 1.0 / swing**2 if swing > 0 else float("inf")

    # Universo de candidatos
    universo: list[str] = []
    for d in distritos:
        for c in d.candidatos:
            if c not in universo:
                universo.append(c)
    K = len(universo)
    idx = {c: i for i, c in enumerate(universo)}

    # Prior departamental (votos agregados rurales por dept)
    dept_agg: dict[int, np.ndarray] = {}
    for d in distritos:
        if d.votos_validos > 0 and d.ambito == 1 and d.poblacion < UMBRAL_RURAL:
            v = np.zeros(K, dtype=np.int64)
            for c, vv in zip(d.candidatos, d.votos):
                v[idx[c]] = vv
            u1 = d.ubigeo1
            dept_agg[u1] = dept_agg.get(u1, np.zeros(K, dtype=np.int64)) + v

    national_agg = sum(dept_agg.values(), np.zeros(K, dtype=np.int64)) if dept_agg else np.zeros(K, dtype=np.int64)

    # votos_por_acta rural por departamento
    dept_vpa, vpa_fallback = calcular_vpa_rural(distritos)

    contado_total = np.zeros(K, dtype=np.int64)
    sims = np.zeros((draws, K), dtype=np.int64)

    for d in distritos:
        votos_r = np.zeros(K, dtype=np.int64)
        for c, v in zip(d.candidatos, d.votos):
            votos_r[idx[c]] = v
        contado_total += votos_r

        # Estimar votos faltantes
        if d.actas_cont > 0 and d.votos_validos > 0:
            # Distrito con datos: usa su propia densidad
            v_falt = int(d.votos_validos * d.actas_faltantes / d.actas_cont)
        elif d.actas_faltantes > 0:
            # Sin datos: usa mediana rural del departamento
            vpa = dept_vpa.get(d.ubigeo1, vpa_fallback)
            v_falt = int(vpa * d.actas_faltantes)
        else:
            v_falt = 0

        if v_falt <= 0:
            continue

        # Alpha para Dirichlet posterior
        total_r = float(votos_r.sum())
        if total_r > 0 and np.isfinite(concentracion):
            alpha = alpha_prior + (votos_r / total_r) * concentracion
        else:
            # Sin datos propios: prior departamental rural → prior nacional
            ref = dept_agg.get(d.ubigeo1, national_agg)
            ref_sum = float(ref.sum())
            if ref_sum > 0:
                alpha = alpha_prior + (ref / ref_sum) * concentracion
            else:
                alpha = np.full(K, alpha_prior)

        p_draws = rng.dirichlet(alpha, size=draws)
        for i in range(draws):
            sims[i] += rng.multinomial(v_falt, p_draws[i])

    proyectado = sims + contado_total
    pct = proyectado / proyectado.sum(axis=1, keepdims=True) * 100.0
    return universo, contado_total, proyectado, pct


# ── Output ────────────────────────────────────────────────────────────────────

def _excluir_idx(universo):
    return {i for i, n in enumerate(universo) if "BLANCO" in n.upper() or "NULO" in n.upper()}


def imprimir_resumen(distritos, universo, contado, proy, pct):
    total_ac = sum(d.actas_tot for d in distritos)
    cont_ac = sum(d.actas_cont for d in distritos)
    pct_g = 100.0 * cont_ac / total_ac if total_ac else 0.0
    n_sin = sum(1 for d in distritos if d.actas_cont == 0 and d.actas_tot > 0)

    print()
    print("=" * 84)
    print("  Proyección bayesiana — Presidencial Perú 2026, 1ra vuelta  [v3 distrito]")
    print(f"  Actas: {cont_ac:,}/{total_ac:,} ({pct_g:.2f}%)  ·  "
          f"Distritos: {len(distritos)}  ·  Sin actas: {n_sin}  ·  Draws: {pct.shape[0]:,}")
    print("=" * 84)

    actual = contado / contado.sum() * 100.0
    med = np.median(pct, axis=0)
    lo = np.percentile(pct, 2.5, axis=0)
    hi = np.percentile(pct, 97.5, axis=0)
    excl = _excluir_idx(universo)
    orden = np.argsort(-med)

    print(f"\n{'Candidato/Partido':<44} {'Actual':>8} {'Proy.':>8}  {'IC 95%':>18}")
    print("-" * 84)
    for i in orden:
        marca = " *" if i in excl else "  "
        ic = f"[{lo[i]:5.2f}, {hi[i]:5.2f}]"
        print(f"{universo[i][:44]:<44} {actual[i]:>7.2f}% {med[i]:>7.2f}%  {ic:>18}{marca}")
    print("  * blancos/nulos excluidos del cálculo de pase a 2da vuelta")

    # Top-2
    pct_v = pct.copy()
    pct_v[:, list(excl)] = -1
    top2 = np.argsort(-pct_v, axis=1)[:, :2]
    counts = np.zeros(pct.shape[1], dtype=np.int64)
    for row in top2:
        counts[row] += 1
    probs = counts / pct.shape[0]

    print("\nProbabilidad de pase a 2da vuelta:")
    for i in np.argsort(-probs):
        if probs[i] < 0.005 or i in excl:
            continue
        bar = "█" * int(probs[i] * 40)
        print(f"  {universo[i][:46]:<46} {probs[i]*100:5.1f}%  {bar}")


def imprimir_por_departamento(distritos, universo):
    excl = _excluir_idx(universo)
    validos = [i for i in range(len(universo)) if i not in excl]

    dept_votos: dict[int, np.ndarray] = {}
    dept_nombre: dict[int, str] = {}
    dept_actas: dict[int, tuple[int, int]] = {}

    for d in distritos:
        if d.ambito != 1:
            continue
        K = len(universo)
        if d.ubigeo1 not in dept_votos:
            dept_votos[d.ubigeo1] = np.zeros(K, dtype=np.int64)
            dept_nombre[d.ubigeo1] = d.dept
            dept_actas[d.ubigeo1] = (0, 0)
        for c, v in zip(d.candidatos, d.votos):
            if c in universo:
                dept_votos[d.ubigeo1][universo.index(c)] += v
        ac, at = dept_actas[d.ubigeo1]
        dept_actas[d.ubigeo1] = (ac + d.actas_cont, at + d.actas_tot)

    print("\n\nResumen por departamento (% sobre votos emitidos contados):")
    print("-" * 84)
    for u1 in sorted(dept_nombre):
        vv = dept_votos[u1]
        tot = vv.sum()
        if tot == 0:
            continue
        pcts = vv[validos] / tot * 100.0
        top1_vi = np.argmax(pcts)
        top2_vi = np.argsort(-pcts)[1]
        top1_i = validos[top1_vi]
        top2_i = validos[top2_vi]
        ac, at = dept_actas[u1]
        pct_d = 100.0 * ac / at if at else 0.0
        print(
            f"  {dept_nombre[u1][:18]:<18}  {pct_d:5.1f}% actas  "
            f"1° {universo[top1_i][:20]:<20} {pcts[top1_vi]:5.1f}%  "
            f"2° {universo[top2_i][:20]:<20} {pcts[top2_vi]:5.1f}%"
        )


def guardar_csv(path, distritos, universo, pct):
    med = np.median(pct, axis=0)
    lo = np.percentile(pct, 2.5, axis=0)
    hi = np.percentile(pct, 97.5, axis=0)
    contado = np.zeros(len(universo), dtype=np.int64)
    for d in distritos:
        for c, v in zip(d.candidatos, d.votos):
            if c in universo:
                contado[universo.index(c)] += v
    total = contado.sum()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidato", "actual_pct", "proy_mediana", "ic95_lo", "ic95_hi"])
        for i, c in enumerate(universo):
            w.writerow([c, f"{contado[i]/total*100:.4f}",
                        f"{med[i]:.4f}", f"{lo[i]:.4f}", f"{hi[i]:.4f}"])
        w.writerow([])
        w.writerow(["distrito","provincia","departamento","ubigeo3",
                    "pct_actas","votos_validos","actas_cont","actas_tot","poblacion"])
        for d in distritos:
            w.writerow([d.nombre, d.prov, d.dept, d.ubigeo3,
                        f"{d.pct_actas:.2f}", d.votos_validos,
                        d.actas_cont, d.actas_tot, d.poblacion])
    print(f"\n→ CSV guardado: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Proyección electoral Perú 2026 — nivel distrito v3")
    ap.add_argument("--draws", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--swing", type=float, default=0.03,
                    help="overdispersión por candidato/distrito. 0=ingenuo, 0.03=±3pp")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--regiones", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    distritos = cargar_distritos(max_workers=args.workers)
    rng = np.random.default_rng(args.seed)
    universo, contado, proy, pct = proyectar(
        distritos, draws=args.draws, alpha_prior=args.alpha,
        swing=args.swing, rng=rng,
    )
    imprimir_resumen(distritos, universo, contado, proy, pct)
    if args.regiones:
        imprimir_por_departamento(distritos, universo)
    if args.csv:
        guardar_csv(args.csv, distritos, universo, pct)


if __name__ == "__main__":
    main()
