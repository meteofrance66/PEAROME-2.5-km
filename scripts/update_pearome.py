#!/usr/bin/env python3
"""PEAROME (AROME ensemble, Météo-France) : probabilités de dépassement de seuils.

Deux sorties : les cartes (grilles recadrées sur une fenêtre autour du 66, products/) et, depuis le 24 nov. 2026, les
valeurs à chaque commune de France (departements/XX.json, sans carte). Un seul appel par produit et par échéance :
la grille de la France entière, dont on recadre la fenêtre des cartes (vérifié contre un appel direct de la fenêtre).

L'API WCS PE-AROME 0,025° n'expose pas les membres un par un mais des produits de probabilité
(rafales >= x km/h, pluie sur 1/6/24 h >= x mm...). On récupère, pour le dernier run, chaque produit
retenu à un pas de 3 h, recadré sur une fenêtre autour du 66, et on publie un JSON compact
(valeurs 0-100 en uint8 base64). Clé : variable d'environnement METEOFRANCE_PAQUET_API_KEY
(jamais en dur, jamais exposée côté site).
"""
import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import requests

BASE = "https://public-api.meteofrance.fr/public/pearome/1.0/wcs/MF-NWP-HIGHRES-PEAROME-0025-FRANCE-WCS/"
UA = "Mozilla/5.0 (compatible; AlertesMeteo-PEAROME/1.0)"
LAT0, LAT1, LON0, LON1 = 41.9, 43.3, 1.4, 3.5
# France métropolitaine et Corse (bornes sur la grille de 0,025°) : un appel ramène toute la France, la fenêtre des cartes en est recadrée.
FR_LAT0, FR_LAT1, FR_LON0, FR_LON1 = 41.0, 51.5, -5.5, 10.0
CATALOGUE = Path(__file__).resolve().parent.parent / "config" / "communes-france.json"
HORS_GRILLE = 255
STEP_H = 3
# Grandeurs horaires (rafale de l'heure, pluie de l'heure) : publiées au pas de 3 h, mais en prenant pour chaque pixel la
# probabilité la plus forte des 3 heures de la tranche. Avec la seule heure ronde, un pic tombé entre deux (rafales de
# 72-78 km/h prévues par AROME vers 14 h UTC le 21 oct. 2026, PEAROME lu à 12 h et 15 h) n'apparaissait pas.
FAMILLES_HORAIRES = {"rafales", "pluie1"}
# Change quand le calcul d'un produit change : les produits gardés en cache par un passage précédent sont alors refaits.
CACHE_VERSION = "v4"
# Environ 30 appels/min : le quota du portail est partagé avec d'autres pipelines.
MIN_INTERVAL = 2.0

# (id court, préfixe de couverture, libellé, famille, seuil)
PRODUCTS = [
    # Orages : probabilité de réflectivité radar simulée élevée (averses orageuses), de grêle et de supercellules.
    ("rfx40", "N_PROBA_RFX_40DBZ__GROUND_OR_WATER_SURFACE", "Orages / fortes averses (réflectivité ≥ 40 dBZ)", "orages", 40),
    ("rfx45", "N_PROBA_RFX_45DBZ__GROUND_OR_WATER_SURFACE", "Orages forts (réflectivité ≥ 45 dBZ)", "orages", 45),
    ("grele8", "N_PROBA_D_GRELE_8__GROUND_OR_WATER_SURFACE", "Grêle (≥ 8 kg/m²)", "grele", 8),
    ("scp1", "N_PROBA_D_SCP_1__GROUND_OR_WATER_SURFACE", "Supercellules (indice SCP > 1)", "supercellules", 1),
    # Rafales : seuils de Météo-France en NŒUDS (« Wind gust more than 50kt probability », DescribeCoverage du
    # 21 oct. 2026), convertis en km/h (1 nœud = 1,852 km/h) ; ils étaient affichés à tort comme des km/h.
    ("raf30", "N_PROBA_RAF_30__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "Rafales ≥ 56 km/h (30 nœuds)", "rafales", 56),
    ("raf40", "N_PROBA_RAF_40__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "Rafales ≥ 74 km/h (40 nœuds)", "rafales", 74),
    ("raf50", "N_PROBA_RAF_50__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "Rafales ≥ 93 km/h (50 nœuds)", "rafales", 93),
    ("raf60", "N_PROBA_RAF_60__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "Rafales ≥ 111 km/h (60 nœuds)", "rafales", 111),
    ("raf70", "N_PROBA_RAF_70__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND", "Rafales ≥ 130 km/h (70 nœuds)", "rafales", 130),
    ("p24_20", "N_PROBA_PRECI24_20__GROUND_OR_WATER_SURFACE", "Pluie 24 h ≥ 20 mm", "pluie24", 20),
    ("p24_50", "N_PROBA_PRECI24_50__GROUND_OR_WATER_SURFACE", "Pluie 24 h ≥ 50 mm", "pluie24", 50),
    ("p24_80", "N_PROBA_PRECI24_80__GROUND_OR_WATER_SURFACE", "Pluie 24 h ≥ 80 mm", "pluie24", 80),
    ("p24_120", "N_PROBA_PRECI24_120__GROUND_OR_WATER_SURFACE", "Pluie 24 h ≥ 120 mm", "pluie24", 120),
    ("p24_200", "N_PROBA_PRECI24_200__GROUND_OR_WATER_SURFACE", "Pluie 24 h ≥ 200 mm", "pluie24", 200),
    ("p06_20", "N_PROBA_PRECI06_20__GROUND_OR_WATER_SURFACE", "Pluie 6 h ≥ 20 mm", "pluie6", 20),
    ("p06_60", "N_PROBA_PRECI06_60__GROUND_OR_WATER_SURFACE", "Pluie 6 h ≥ 60 mm", "pluie6", 60),
    ("p06_100", "N_PROBA_PRECI06_100__GROUND_OR_WATER_SURFACE", "Pluie 6 h ≥ 100 mm", "pluie6", 100),
    ("p01_20", "N_PROBA_PRECI01_20__GROUND_OR_WATER_SURFACE", "Pluie 1 h ≥ 20 mm", "pluie1", 20),
]

_last_call = 0.0


class Quota(Exception):
    pass


def call(session: requests.Session, query: str, key: str) -> requests.Response:
    """Un appel WCS espacé de MIN_INTERVAL, avec reprise sur 429 (quota partagé)."""
    global _last_call
    for attempt in range(60):
        wait = _last_call + MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        response = session.get(BASE + query, headers={"apikey": key, "User-Agent": UA}, timeout=(15, 120))
        if response.status_code == 429:
            # Fenêtre d'une minute : on attend la fin de la minute annoncée ("after ... 08:49:00").
            seconds = 20
            match = re.search(r"after \S+ (\d{2}):(\d{2}):(\d{2})", response.text)
            if match:
                now = datetime.now(timezone.utc)
                target = now.replace(hour=int(match[1]), minute=int(match[2]), second=int(match[3]), microsecond=0)
                seconds = max(5, min(70, (target - now).total_seconds() + 3))
            # Quota partagé avec le pipeline AROME (même clé) : après dix refus de suite, on laisse passer
            # une minute entière entre deux essais, jusqu'à ~50 min au total.
            if attempt >= 10:
                seconds = max(seconds, 60)
            print(f"Quota Météo-France atteint, attente {seconds:.0f} s ({attempt + 1}/60).", flush=True)
            time.sleep(seconds)
            continue
        return response
    raise Quota("Quota Météo-France épuisé après 60 tentatives")


def latest_runs(session: requests.Session, key: str) -> dict[str, str]:
    """Dernier run (ex. 2026-10-04T03.00.00Z) de chaque produit retenu."""
    response = call(session, "GetCapabilities?service=WCS&version=2.0.1&language=eng", key)
    response.raise_for_status()
    # Le XML du catalogue n'est pas toujours bien formé (&, etc.) : extraction par expression régulière.
    ids = re.findall(r"<wcs:CoverageId>([^<]+)</wcs:CoverageId>", response.text)
    runs: dict[str, str] = {}
    for _short, prefix, *_ in PRODUCTS:
        stamps = sorted(i.split("___")[1] for i in ids if i.startswith(prefix + "___"))
        if stamps:
            runs[prefix] = stamps[-1]
    return runs


def describe_times(session: requests.Session, key: str, coverage: str) -> list[datetime]:
    """Instants disponibles d'une couverture (pas horaire entre beginPosition et endPosition)."""
    response = call(session, f"DescribeCoverage?service=WCS&version=2.0.1&coverageid={coverage}", key)
    response.raise_for_status()
    text = response.text
    begin = re.search(r"<gml:beginPosition[^>]*>([^<]+)<", text)
    end = re.search(r"<gml:endPosition[^>]*>([^<]+)<", text)
    if not begin or not end:
        raise ValueError("Période de la couverture introuvable")
    t0 = datetime.fromisoformat(begin[1].replace("Z", "+00:00"))
    t1 = datetime.fromisoformat(end[1].replace("Z", "+00:00"))
    out, t = [], t0
    while t <= t1:
        out.append(t)
        t += timedelta(hours=1)
    return out


class Geo(NamedTuple):
    """Grille régulière en latitude/longitude : ligne 0 au nord, colonne 0 à l'ouest."""

    lat_n: float
    lon_w: float
    dlat: float
    dlon: float
    nj: int
    ni: int


def decode(content: bytes, avec_geometrie: bool = False):
    from eccodes import codes_get, codes_get_values, codes_new_from_message, codes_release

    handle = codes_new_from_message(content)
    try:
        ni, nj = codes_get(handle, "Ni"), codes_get(handle, "Nj")
        values = np.asarray(codes_get_values(handle), dtype=float).reshape(nj, ni)
        if codes_get(handle, "jScansPositively"):
            values = values[::-1]
        if not avec_geometrie:
            return values
        if codes_get(handle, "iScansNegatively"):
            raise ValueError("grille balayée d'est en ouest : non gérée")
        la1, la2 = codes_get(handle, "latitudeOfFirstGridPointInDegrees"), codes_get(handle, "latitudeOfLastGridPointInDegrees")
        lo1, lo2 = codes_get(handle, "longitudeOfFirstGridPointInDegrees"), codes_get(handle, "longitudeOfLastGridPointInDegrees")
        lo1, lo2 = (x - 360 if x > 180 else x for x in (lo1, lo2))
        lat_n, lat_s = max(la1, la2), min(la1, la2)
        return values, Geo(lat_n, lo1, (lat_n - lat_s) / (nj - 1), (lo2 - lo1) / (ni - 1), nj, ni)
    finally:
        codes_release(handle)


def recadrer(grille: np.ndarray, geo: Geo) -> np.ndarray:
    """Fenêtre des cartes (LAT0-LAT1, LON0-LON1) découpée dans la grille de la France."""
    r0, r1 = round((geo.lat_n - LAT1) / geo.dlat), round((geo.lat_n - LAT0) / geo.dlat) + 1
    c0, c1 = round((LON0 - geo.lon_w) / geo.dlon), round((LON1 - geo.lon_w) / geo.dlon) + 1
    if r0 < 0 or c0 < 0 or r1 > grille.shape[0] or c1 > grille.shape[1]:
        raise ValueError("fenêtre des cartes hors de la grille de la France")
    return grille[r0:r1, c0:c1]


def charger_communes(chemin: Path = CATALOGUE) -> list:
    """Communes du catalogue : [code INSEE, nom, département, codes postaux, population, latitude, longitude]."""
    return json.loads(chemin.read_text(encoding="utf-8"))["communes"]


def positions_communes(communes: list, geo: Geo):
    """Ligne et colonne du pixel le plus proche de chaque commune, et indicateur « dans la grille »."""
    lat = np.array([float(c[5]) for c in communes])
    lon = np.array([float(c[6]) for c in communes])
    lignes = np.rint((geo.lat_n - lat) / geo.dlat).astype(int)
    colonnes = np.rint((lon - geo.lon_w) / geo.dlon).astype(int)
    dedans = (lignes >= 0) & (lignes < geo.nj) & (colonnes >= 0) & (colonnes < geo.ni)
    return lignes, colonnes, dedans


def echantillonner(grille: np.ndarray, positions) -> np.ndarray:
    """Probabilité (0-100, entier) au pixel de chaque commune ; HORS_GRILLE si la commune est hors de la grille."""
    lignes, colonnes, dedans = positions
    sortie = np.full(len(lignes), HORS_GRILLE, dtype=np.uint8)
    sortie[dedans] = np.rint(grille[lignes[dedans], colonnes[dedans]]).astype(np.uint8)
    return sortie


class Etat:
    """Mode « France » : actif tant que l'appel de la France et le recadrage de la fenêtre sont cohérents."""

    def __init__(self, actif: bool, communes: list | None = None):
        self.actif = actif and communes is not None
        self.communes = communes
        self.verifie = False
        self.positions = None
        self.geo: Geo | None = None


def already_published(url: str, run_iso: str) -> bool:
    """Run déjà publié en entier (une publication partielle, coupée par le quota, est complétée au passage suivant)."""
    try:
        r = requests.get(url, timeout=(10, 30), headers={"User-Agent": UA})
        if r.status_code != 200:
            return False
        index = r.json()
        return index.get("run_time") == run_iso and index.get("complete", True) is True
    except (requests.RequestException, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build/pearome")
    parser.add_argument("--current-index-url", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-dir", default=".cache/pearome")
    parser.add_argument("--sans-france", action="store_true", help="ne publier que les cartes (pas les valeurs aux communes)")
    args = parser.parse_args()
    key = os.environ.get("METEOFRANCE_PAQUET_API_KEY", "")
    if not key:
        print("METEOFRANCE_PAQUET_API_KEY manquante", file=sys.stderr)
        return 1
    session = requests.Session()
    runs = latest_runs(session, key)
    if not runs:
        print("Aucun produit PEAROME dans le catalogue", file=sys.stderr)
        return 1
    missing = [p[0] for p in PRODUCTS if p[1] not in runs]
    if missing:
        print(f"Produits absents du catalogue : {missing}", file=sys.stderr)
    run_stamp = max(set(runs.values()), key=lambda s: sum(1 for v in runs.values() if v == s))
    run_iso = run_stamp.replace(".", ":")
    if not args.force and args.current_index_url and already_published(args.current_index_url, run_iso):
        print(f"Run {run_iso} déjà publié.")
        return 0
    run_dt = datetime.fromisoformat(run_iso.replace("Z", "+00:00"))
    out = Path(args.output_dir)
    (out / "products").mkdir(parents=True, exist_ok=True)

    # Produits déjà récupérés pour CE run lors d'un passage précédent (coupé par le quota) : réutilisés tels quels.
    cache = Path(args.cache_dir) / run_stamp
    cache.mkdir(parents=True, exist_ok=True)
    for old in Path(args.cache_dir).iterdir():  # runs précédents : inutiles
        if old.is_dir() and old.name != run_stamp:
            for f in old.iterdir():
                f.unlink()
            old.rmdir()

    etat = Etat(not args.sans_france, None if args.sans_france else charger_communes())
    france_par_produit: dict[str, dict] = {}
    products_meta = []
    shape: tuple[int, int] | None = None
    expected = [p for p in PRODUCTS if runs.get(p[1]) == run_stamp]
    partial = False
    for short, prefix, label, family, threshold in expected:
        cached = cache / f"{short}.{CACHE_VERSION}.json"
        if cached.is_file():
            try:
                saved = json.loads(cached.read_text(encoding="utf-8"))
                (out / "products" / f"{short}.json").write_text(json.dumps(saved["product"], separators=(",", ":")), encoding="utf-8")
                products_meta.append(saved["meta"])
                shape = shape or tuple(saved["shape"])
                if saved.get("france"):
                    france_par_produit[short] = {"meta": saved["meta"], "steps": saved["france"]}
                print(f"{short}: repris du cache", flush=True)
                continue
            except (ValueError, KeyError, OSError):
                cached.unlink(missing_ok=True)
        try:
            result = fetch_product(session, key, run_stamp, run_dt, short, prefix, label, family, threshold, shape, etat)
        except Quota as exc:
            # Quota épuisé : on publie ce qui a été obtenu, le passage suivant complétera grâce au cache.
            print(f"::warning::{exc} - publication partielle ({len(products_meta)}/{len(expected)} produits), suite au prochain passage.", flush=True)
            partial = True
            break
        if result is None:
            continue
        product, meta, shape, france = result
        (out / "products" / f"{short}.json").write_text(json.dumps(product, separators=(",", ":")), encoding="utf-8")
        cached.write_text(json.dumps({"product": product, "meta": meta, "shape": list(shape), "france": france}, separators=(",", ":")), encoding="utf-8")
        products_meta.append(meta)
        if france:
            france_par_produit[short] = {"meta": meta, "steps": france}
    if not products_meta or shape is None:
        if partial:
            raise Quota("Quota Météo-France épuisé avant le premier produit")
        print("Aucun produit récupéré", file=sys.stderr)
        return 1
    communes_ecrites = ecrire_departements(out, run_iso, france_par_produit, etat.communes) if etat.communes and france_par_produit else None
    return write_index(out, run_iso, shape, products_meta, complete=not partial, communes_ecrites=communes_ecrites)


def fetch_grid(session, key, coverage, short, t):
    """Grille (probabilités 0-100) d'un instant ; None si l'échéance n'a pas de donnée."""
    stamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")
    response = call(
        session,
        f"GetCoverage?service=WCS&version=2.0.1&coverageid={coverage}&format=application%2Fwmo-grib"
        f"&subset=time({stamp})&subset=lat({LAT0},{LAT1})&subset=long({LON0},{LON1})",
        key,
    )
    if response.status_code != 200 or response.content[:4] != b"GRIB":
        # 404 : échéance sans donnée pour ce produit (les cumuls n'existent qu'à partir de leur durée).
        if response.status_code != 404:
            print(f"{short} {stamp}: HTTP {response.status_code}, ignoré", flush=True)
        return None
    return np.clip(np.nan_to_num(decode(response.content), nan=0.0), 0, 100)


def fetch_instant(session, key, coverage, short, t, etat: Etat):
    """(grille de la fenêtre des cartes, grille de la France ou None) à un instant ; (None, None) sans donnée.

    Mode France : un appel sur la France entière, fenêtre recadrée ; au premier appel, la fenêtre est aussi demandée
    directement pour vérifier que le recadrage donne les mêmes valeurs (sinon retour à l'appel de la fenêtre seule).
    """
    if not etat.actif:
        return fetch_grid(session, key, coverage, short, t), None
    stamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")
    response = call(
        session,
        f"GetCoverage?service=WCS&version=2.0.1&coverageid={coverage}&format=application%2Fwmo-grib"
        f"&subset=time({stamp})&subset=lat({FR_LAT0},{FR_LAT1})&subset=long({FR_LON0},{FR_LON1})",
        key,
    )
    if response.status_code == 404:
        return None, None
    if response.status_code != 200 or response.content[:4] != b"GRIB":
        print(f"{short} {stamp}: appel de la France refusé (HTTP {response.status_code}), retour à la fenêtre seule", flush=True)
        etat.actif = False
        return fetch_grid(session, key, coverage, short, t), None
    try:
        brute, geo = decode(response.content, avec_geometrie=True)
        france = np.clip(np.nan_to_num(brute, nan=0.0), 0, 100)
        fenetre = recadrer(france, geo)
    except Exception as exc:  # grille inattendue : on n'y touche pas, les cartes restent calculées comme avant
        print(f"{short} {stamp}: grille de la France illisible ({exc}), retour à la fenêtre seule", flush=True)
        etat.actif = False
        return fetch_grid(session, key, coverage, short, t), None
    if not etat.verifie:
        directe = fetch_grid(session, key, coverage, short, t)
        if directe is None or directe.shape != fenetre.shape or float(np.abs(directe - fenetre).max()) > 0.6:
            ecart = None if directe is None or directe.shape != fenetre.shape else float(np.abs(directe - fenetre).max())
            print(f"Recadrage non conforme (formes {None if directe is None else directe.shape} / {fenetre.shape}, écart {ecart}) : fenêtre seule.", flush=True)
            etat.actif = False
            return directe, None
        etat.verifie = True
        etat.geo = geo
        etat.positions = positions_communes(etat.communes, geo)
        print(f"Recadrage vérifié ({fenetre.shape[1]}x{fenetre.shape[0]}), grille France {geo.ni}x{geo.nj}.", flush=True)
        return directe, france
    if geo != etat.geo:
        print(f"{short} {stamp}: géométrie de la grille différente, retour à la fenêtre seule", flush=True)
        etat.actif = False
        return fenetre, None
    return fenetre, france


def fetch_product(session, key, run_stamp, run_dt, short, prefix, label, family, threshold, shape, etat: Etat):
    """Toutes les échéances d'un produit ; None si aucune n'est disponible."""
    coverage = f"{prefix}___{run_stamp}"
    times = describe_times(session, key, coverage)
    disponibles = set(times)
    steps = [t for t in times if t > run_dt and int((t - run_dt).total_seconds() // 3600) % STEP_H == 0]
    horaire = family in FAMILLES_HORAIRES
    entries = []
    france_entrees = []
    for t in steps:
        stamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")
        # Grandeur horaire : les 3 heures de la tranche (t-2 h, t-1 h, t), maximum pixel par pixel.
        instants = [t - timedelta(hours=h) for h in range(STEP_H - 1, -1, -1)] if horaire else [t]
        grilles = []
        grilles_france = []
        for u in instants:
            if u <= run_dt or u not in disponibles:
                continue
            g, gf = fetch_instant(session, key, coverage, short, u, etat)
            if g is None:
                continue
            shape = shape or g.shape
            if g.shape == shape:
                grilles.append(g)
                if gf is not None:
                    grilles_france.append(gf)
        if not grilles:
            continue
        grid = np.max(np.stack(grilles), axis=0)
        if etat.actif and len(grilles_france) == len(grilles):
            france_entrees.append(
                {
                    "time": stamp,
                    "lead": int((t - run_dt).total_seconds() // 3600),
                    "data": base64.b64encode(echantillonner(np.max(np.stack(grilles_france), axis=0), etat.positions).tobytes()).decode("ascii"),
                }
            )
        entries.append(
            {
                "time": stamp,
                "lead": int((t - run_dt).total_seconds() // 3600),
                "max": round(float(grid.max()), 1),
                "mean": round(float(grid.mean()), 1),
                "data": base64.b64encode(np.rint(grid).astype(np.uint8).tobytes()).decode("ascii"),
            }
        )
    print(f"{short}: {len(entries)} échéances", flush=True)
    if not entries:
        return None
    # window_hours : chaque échéance donne la probabilité la plus forte des N heures qui la précèdent (1 = l'heure ronde).
    window = STEP_H if horaire else 1
    product = {"id": short, "label": label, "family": family, "threshold": threshold, "window_hours": window, "steps": entries}
    meta = {
        "id": short,
        "label": label,
        "family": family,
        "threshold": threshold,
        "window_hours": window,
        "file": f"products/{short}.json",
        "max": max(e["max"] for e in entries),
        "steps": len(entries),
    }
    # Valeurs aux communes : seulement si toutes les échéances du produit ont pu être lues sur la grille de la France.
    france = france_entrees if etat.actif and len(france_entrees) == len(entries) else None
    return product, meta, shape, france


def ecrire_departements(out: Path, run_iso: str, france_par_produit: dict, communes: list) -> dict:
    """departements/XX.json : probabilités aux communes du département, une chaîne base64 (1 octet par commune) par échéance."""
    (out / "departements").mkdir(parents=True, exist_ok=True)
    par_dep: dict[str, list[int]] = {}
    for i, c in enumerate(communes):
        par_dep.setdefault(str(c[2]), []).append(i)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ecrites = {}
    for code, indices in sorted(par_dep.items()):
        idx = np.array(indices)
        produits = []
        for short, item in france_par_produit.items():
            meta = item["meta"]
            etapes = []
            for e in item["steps"]:
                tout = np.frombuffer(base64.b64decode(e["data"]), dtype=np.uint8)
                etapes.append({"time": e["time"], "lead": e["lead"], "data": base64.b64encode(tout[idx].tobytes()).decode("ascii")})
            produits.append(
                {
                    "id": short,
                    "label": meta["label"],
                    "family": meta["family"],
                    "threshold": meta["threshold"],
                    "window_hours": meta["window_hours"],
                    "steps": etapes,
                }
            )
        payload = {
            "schema_version": 1,
            "department": code,
            "run_time": run_iso,
            "generated_at": generated,
            "unit": "% de probabilité de dépasser le seuil",
            "no_data": HORS_GRILLE,
            "step_hours": STEP_H,
            "columns": ["code_insee", "name", "postal_codes", "population", "latitude", "longitude"],
            "communes": [[communes[i][0], communes[i][1], communes[i][3], communes[i][4], communes[i][5], communes[i][6]] for i in indices],
            "products": produits,
        }
        (out / "departements" / f"{code}.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        ecrites[code] = len(indices)
    print(f"Valeurs aux communes : {sum(ecrites.values())} communes, {len(ecrites)} départements, {len(france_par_produit)} produits.", flush=True)
    return ecrites


def write_index(out: Path, run_iso: str, shape, products_meta, complete: bool, communes_ecrites: dict | None = None) -> int:
    index = {
        "schema_version": 1,
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {"name": "PEAROME (AROME ensemble)", "provider": "Météo-France", "resolution_km": 2.5},
        "run_time": run_iso,
        "unit": "% de probabilité de dépasser le seuil",
        "bounds": [[LAT0, LON0], [LAT1, LON1]],
        "height": shape[0],
        "width": shape[1],
        "step_hours": STEP_H,
        # False : publication coupée par le quota, complétée au passage suivant (voir already_published).
        "complete": complete,
        "products": products_meta,
    }
    if communes_ecrites:
        # Valeurs aux communes de toute la France, sans carte : departements/XX.json.
        index["coverage"] = {"label": "France métropolitaine et Corse", "communes": sum(communes_ecrites.values()), "departments": len(communes_ecrites)}
        index["departments"] = {code: {"communes": n} for code, n in communes_ecrites.items()}
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"PEAROME {run_iso} : {len(products_meta)} produits{'' if complete else ' (partiel)'}, grille {shape[1]}x{shape[0]}.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Quota as exc:
        # Quota partagé avec les autres workflows : on garde les données publiées et on réessaiera au prochain cycle.
        print(f"::warning::{exc} - données précédentes conservées, nouvel essai au prochain cycle.")
        sys.exit(0)
