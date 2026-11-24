import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import update_pearome as pe  # noqa: E402

# Grille de la France telle que demandée à l'API : 41,0-51,5 N, 5,5 O-10,0 E, pas de 0,025°.
GEO = pe.Geo(lat_n=51.5, lon_w=-5.5, dlat=0.025, dlon=0.025, nj=421, ni=621)


def grille_france():
    """Chaque pixel porte une valeur qui dépend de sa position : toute erreur d'un pixel se voit."""
    lignes, colonnes = np.indices((GEO.nj, GEO.ni))
    return ((lignes * 7 + colonnes * 3) % 101).astype(float)


class Recadrage(unittest.TestCase):
    def test_la_fenetre_des_cartes_fait_57_sur_85_comme_avant(self):
        self.assertEqual(pe.recadrer(grille_france(), GEO).shape, (57, 85))

    def test_les_coins_de_la_fenetre_sont_ceux_de_la_france(self):
        g = grille_france()
        f = pe.recadrer(g, GEO)
        # coin nord-ouest de la fenêtre : 43,3 N, 1,4 E
        ligne = round((51.5 - 43.3) / 0.025)
        colonne = round((1.4 + 5.5) / 0.025)
        self.assertEqual(f[0, 0], g[ligne, colonne])
        self.assertEqual(f[-1, -1], g[ligne + 56, colonne + 84])

    def test_fenetre_hors_de_la_grille_refusee(self):
        petite = pe.Geo(lat_n=43.0, lon_w=2.0, dlat=0.025, dlon=0.025, nj=40, ni=40)
        with self.assertRaises(ValueError):
            pe.recadrer(np.zeros((40, 40)), petite)


class Communes(unittest.TestCase):
    COMMUNES = [
        ["66136", "Perpignan", "66", ["66000"], 121616, 42.699, 2.9045],
        ["75056", "Paris", "75", ["75001"], 2100000, 48.8566, 2.3522],
        ["2A004", "Ajaccio", "2A", ["20000"], 70000, 41.9192, 8.7386],
        ["97101", "Hors grille", "97", ["97100"], 1000, 16.2, -61.5],
    ]

    def test_le_pixel_le_plus_proche_de_chaque_commune(self):
        g = grille_france()
        pos = pe.positions_communes(self.COMMUNES, GEO)
        v = pe.echantillonner(g, pos)
        ligne = round((51.5 - 42.699) / 0.025)
        colonne = round((2.9045 + 5.5) / 0.025)
        self.assertEqual(v[0], int(round(g[ligne, colonne])))
        self.assertEqual(v[3], pe.HORS_GRILLE)  # commune hors de la grille
        self.assertTrue(all(x != pe.HORS_GRILLE for x in v[:3]))

    def test_le_catalogue_du_depot_couvre_96_departements(self):
        communes = pe.charger_communes()
        self.assertGreater(len(communes), 34000)
        self.assertEqual(len({str(c[2]) for c in communes}), 96)
        # toutes les communes de métropole et de Corse tombent dans la grille demandée à l'API
        _, _, dedans = pe.positions_communes(communes, GEO)
        self.assertTrue(dedans.all(), f"{int((~dedans).sum())} communes hors grille")


class Ecriture(unittest.TestCase):
    def test_un_fichier_par_departement_avec_les_valeurs_de_ses_communes(self):
        communes = Communes.COMMUNES[:3]
        valeurs = np.array([10, 20, 30], dtype=np.uint8)
        france = {"rfx40": {"meta": {"label": "Orages", "family": "orages", "threshold": 40, "window_hours": 1},
                            "steps": [{"time": "2026-11-24T12:00:00Z", "lead": 3, "data": base64.b64encode(valeurs.tobytes()).decode()}]}}
        with tempfile.TemporaryDirectory() as d:
            ecrites = pe.ecrire_departements(Path(d), "2026-11-24T09:00:00Z", france, communes)
            self.assertEqual(ecrites, {"2A": 1, "66": 1, "75": 1})
            f = json.loads((Path(d) / "departements" / "66.json").read_text(encoding="utf-8"))
        self.assertEqual(f["department"], "66")
        self.assertEqual([c[1] for c in f["communes"]], ["Perpignan"])
        etape = f["products"][0]["steps"][0]
        self.assertEqual(list(base64.b64decode(etape["data"])), [10])
        self.assertEqual(etape["lead"], 3)


if __name__ == "__main__":
    unittest.main()
