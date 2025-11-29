# 🏠 Warsaw Real Estate Clustering

Analiza klasteryzacji cen nieruchomości w Warszawie z wykorzystaniem **prawdziwych granic dzielnic** z OpenStreetMap.

## ✨ Kluczowe funkcjonalności

- **Prawdziwe granice 18 dzielnic Warszawy** - pobierane z OpenStreetMap (algorytm Manual Stitching)
- **Generowanie danych syntetycznych WEWNĄTRZ prawdziwych polygonów** - nie w przybliżonych okręgach
- **K-means z automatyczną optymalizacją k** (silhouette + elbow method)
- **DBSCAN** dla porównania (opcjonalnie)
- **Interaktywna mapa Folium** z warstwami:
  - Prawdziwe granice dzielnic (toggleable)
  - Punkty nieruchomości per klaster
  - Cluster boundaries (convex hull)
  - Heatmapa cen
  - Light/Dark theme
- **4 segmenty rynkowe**: Premium, Upper, Standard, Economy

## 🚀 Szybki start

```bash
# 1. Rozpakuj projekt
unzip warsaw_real_estate_clustering.zip
cd warsaw_real_estate_clustering

# 2. Zainstaluj zależności
pip install -r requirements.txt

# 3. Uruchom analizę
python main.py -v
```

## 📋 Opcje uruchomienia

```bash
python main.py -v                     # Verbose mode
python main.py -d                     # Debug mode (szczegółowe logi)
python main.py --k 5                  # Wymusz 5 klastrów
python main.py --extended-features    # Użyj więcej cech (area, year, floor)
python main.py --compare-dbscan       # Porównaj z DBSCAN
python main.py --force-refresh        # Pobierz granice dzielnic ponownie z OSM
python main.py --n-samples 1000       # Wygeneruj 1000 nieruchomości
```

## 📁 Struktura projektu

```
warsaw_real_estate_clustering/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Centralna konfiguracja
│   └── logging_config.py    # Setup loggera
│
├── src/
│   ├── boundaries/           # ← Prawdziwe granice dzielnic z OSM
│   │   ├── __init__.py
│   │   └── district_fetcher.py
│   │
│   ├── data_collection/
│   │   ├── __init__.py
│   │   └── synthetic_generator.py  # ← Generuje punkty W polygonach
│   │
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── cleaner.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── clustering.py     # K-means, DBSCAN
│   │   ├── optimizer.py      # Automatyczny wybór k
│   │   └── statistics.py
│   │
│   └── visualization/
│       ├── __init__.py
│       └── map_creator.py    # ← Mapa z prawdziwymi granicami
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── cache/
│   └── boundaries/           # Cache granic dzielnic
│
├── outputs/
│   ├── maps/                 # Mapy HTML
│   ├── plots/
│   └── reports/
│
├── main.py                   # Entry point
├── requirements.txt
└── README.md
```

## 🎯 Wyniki

Po uruchomieniu znajdziesz:

| Plik | Opis |
|------|------|
| `outputs/maps/cluster_map.html` | Interaktywna mapa Folium |
| `data/processed/properties_clustered.csv` | Dane z przypisanymi klastrami |
| `data/boundaries/districts_cache.geojson` | Cache granic dzielnic |

## 🗺️ Mapa interaktywna

Mapa zawiera następujące warstwy (każda toggleable):
- 📍 **District Boundaries** - prawdziwe granice 18 dzielnic
- 📷 **Cluster boundaries** - per klaster
- 🏠 **Properties** - punkty nieruchomości
- 🔥 **Price Heatmap** - gradient cen
- 📍 **Cluster Centers** - centroidy klastrów

## ⚙️ Konfiguracja

Główna konfiguracja w `config/settings.py`:

```python
# Zakres k do testowania
k_range = (3, 10)

# Cechy do klasteryzacji
feature_columns = ["price_per_sqm", "area_sqm", "distance_from_center_km"]

# Rozszerzone cechy
extended_features = ["price_per_sqm", "area_sqm", "distance_from_center_km", 
                     "year_built", "floor"]
```

## 🔧 Technologie

- **Python 3.9+**
- **Shapely** + **GeoPandas** - geometria
- **Scikit-learn** - klasteryzacja
- **Folium** - mapy interaktywne
- **Overpass API** - dane OpenStreetMap

## 📊 Segmenty rynkowe

| Segment | Udział | Cena (PLN/m²) | Dzielnice |
|---------|--------|---------------|-----------|
| Premium | 15% | 22,000-35,000 | Śródmieście, Mokotów, Żoliborz, Wilanów |
| Upper | 25% | 16,000-22,000 | Mokotów, Ochota, Wola, Bielany |
| Standard | 35% | 11,000-16,000 | Wola, Praga-Południe, Ursynów, Bemowo |
| Economy | 25% | 7,000-12,000 | Białołęka, Targówek, Wawer, Rembertów |

## ⚠️ Ważne uwagi

1. **Algorytm stitch_line_segments()** - NIE MODYFIKOWAĆ, działa idealnie
2. **Cache granic** - pobierane raz, zapisywane w `data/boundaries/`
3. **EPSG:32634** - używane do obliczeń powierzchni (UTM 34N dla Polski)
4. **Punkty WEWNĄTRZ polygonów** - walidowane przez `polygon.contains(point)`

## 📝 Licencja

Projekt edukacyjny - Zaawansowana Eksploracja Danych

---

*Wygenerowano: 2025*
