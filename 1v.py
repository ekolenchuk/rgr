#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geant4 Log Parser and Visualizer
Модифицированная версия с поддержкой расчета дозовых карт
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geant4 Log Parser and Visualizer
Модифицированная версия с поддержкой расчета дозовых карт
"""

import os
import sys
import re
import tempfile
import hashlib
import colorsys
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import make_interp_spline, splprep, splev, UnivariateSpline
from sklearn.mixture import GaussianMixture
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import colors
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogLocator, AutoMinorLocator, MaxNLocator, NullFormatter, FuncFormatter
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

try:
    import hdbscan
    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False

from cache_manager import CacheManager

matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

sns.set_style("whitegrid")
sns.set_context("notebook", rc={"lines.linewidth": 2.0})

ENERGY_REGEX = re.compile(
    r'(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*'
    r'(?P<unit>MeV|keV|eV|meV|m\s*eV|e\s*V)',
    re.IGNORECASE
)

THREAD_ANYWHERE_RE = re.compile(r'\b(G4WT\d+)\b')
THREAD_PREFIX_RE = re.compile(r'^(G4WT\d+)\s*\>\s*')

TRACK_STORING_RE = re.compile(r'\(([^,]+),\s*trackID=(\d+),\s*parentID=(\d+)\)')
TRACK_INFO_RE = re.compile(
    r'\*\s*G4Track Information:\s*Particle\s*=\s*([^,]+),\s*Track ID\s*=\s*(\d+),\s*Parent ID\s*=\s*(\d+)',
    re.IGNORECASE
)

SECONDARY_LINE_RE = re.compile(
    r'([a-zA-Z0-9+\-]+)\s*:\s*energy\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-z]+)',
    re.IGNORECASE
)

STEP_LINE_RE = re.compile(
    r'^\s*(?P<step>\d+)\s+'
    r'(?P<x>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?P<xu>fm|pm|nm|um|mm|cm|m|Ang)\s+'
    r'(?P<y>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?P<yu>fm|pm|nm|um|mm|cm|m|Ang)\s+'
    r'(?P<z>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?P<zu>fm|pm|nm|um|mm|cm|m|Ang)\s+'
    r'(?P<kval>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?P<kunit>MeV|keV|eV|meV|m\s*eV|e\s*V)\s+'
    r'(?P<dval>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?P<dunit>MeV|keV|eV|meV|m\s*eV|e\s*V)\s+'
    r'(?P<slval>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?P<slunit>fm|pm|nm|um|mm|cm|m|Ang)\s+'
    r'(?P<tlval>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?P<tlunit>fm|pm|nm|um|mm|cm|m|Ang)\s+'
    r'(?P<vol>\S+)\s+(?P<proc>\S+)\s*$',
    re.IGNORECASE
)

EVENT_START_RE = re.compile(r'Now start processing an event', re.IGNORECASE)
PROCESS_ONE_EVENT_RE = re.compile(r'G4EventManager::ProcessOneEvent', re.IGNORECASE)
VERTICES_PASSED_RE = re.compile(r'vertices passed from G4Event', re.IGNORECASE)


# ==================== НОВЫЙ КЛАСС: Layer для представления слоя материала ====================

class Layer:
    """Класс для хранения информации о слое материала."""

    def __init__(self, name, material, density_g_cm3, z_min, z_max, unit="мм"):
        self.name = name  # Имя объема из Geant4
        self.material = material  # Название материала
        self.density_g_cm3 = density_g_cm3  # Плотность в г/см³
        self.density_g_mm3 = density_g_cm3 * 1e-3  # Перевод в г/мм³ для расчетов
        self.z_min_mm = self._to_mm(z_min, unit)
        self.z_max_mm = self._to_mm(z_max, unit)
        self.thickness_mm = self.z_max_mm - self.z_min_mm
        self.volume_mm3 = None  # Будет вычислен позже с учетом X, Y размеров
        self.total_dose_gray = 0.0  # Суммарная доза в слое
        self.total_energy_mev = 0.0  # Суммарное энерговыделение в слое

    def _to_mm(self, value, unit):
        """Конвертация в миллиметры."""
        conversions = {
            "нм": 1e-6, "мкм": 1e-3, "мм": 1.0, "см": 10.0, "м": 1000.0,
            "nm": 1e-6, "um": 1e-3, "mm": 1.0, "cm": 10.0, "m": 1000.0,
            "Ang": 1e-7, "fm": 1e-12, "pm": 1e-9
        }
        return value * conversions.get(unit.lower(), 1.0)

    def contains(self, z_mm):
        """Проверяет, принадлежит ли координата Z этому слою."""
        return self.z_min_mm <= z_mm <= self.z_max_mm

    def get_mass_mg(self, area_mm2=None):
        """Возвращает массу слоя в мг (для заданной площади или всего слоя)."""
        if area_mm2:
            volume = area_mm2 * self.thickness_mm
        else:
            volume = self.volume_mm3 if self.volume_mm3 else 0
        # масса (мг) = объем (мм³) * плотность (г/мм³) * 1000 (перевод г в мг)
        return volume * self.density_g_mm3 * 1000

    def add_energy_deposition(self, dE_mev):
        """Добавляет энерговыделение к слою."""
        self.total_energy_mev += dE_mev
        # Обновляем дозу (приближенно, используя полный объем слоя)
        if self.volume_mm3 and self.volume_mm3 > 0:
            mass_g = self.volume_mm3 * self.density_g_mm3
            if mass_g > 0:
                self.total_dose_gray += dE_mev / mass_g / 6.242e12

    def get_average_dose_gray(self):
        """Возвращает среднюю дозу в слое."""
        if self.volume_mm3 and self.volume_mm3 > 0:
            mass_g = self.volume_mm3 * self.density_g_mm3
            if mass_g > 0:
                return self.total_energy_mev / mass_g / 6.242e12
        return 0.0

    def get_dose_stats(self):
        """Возвращает статистику по дозе в слое."""
        avg_dose = self.get_average_dose_gray()
        return {
            'name': self.name,
            'material': self.material,
            'thickness_mm': self.thickness_mm,
            'density_g_cm3': self.density_g_cm3,
            'volume_mm3': self.volume_mm3,
            'total_energy_mev': self.total_energy_mev,
            'avg_dose_gray': avg_dose,
            'avg_dose_rad': avg_dose * 100
        }

    def __repr__(self):
        return f"Layer({self.name}, {self.material}, {self.thickness_mm:.2f}мм, доза={self.get_average_dose_gray():.2e} Гр)"


# ==================== PARSER ====================

class Parser:
    def __init__(self):
        self._track_prev_energy = None
        self.thread_event_id = None
        self.current_particle = None
        self.current_track_id = None
        self.current_parent_id = None
        self.summary_data = {}
        self.particle_count = 1
        self.primary_particle_type = None
        self.cache_mgr = None
        self.material_dimensions = MaterialDimensions()
        self.filter_secondary_first_step_flag = False
        self.exclude_transport = True

        # Словарь для отслеживания track_id -> (particle, parent_id)
        self.track_info = {}
        # Словарь для отслеживания текущего трека в каждом потоке
        self.thread_current_track = {}
        self.primary_track_ids = set()

        self.debug_counters = {
            "step_lines_seen": 0,
            "step_parsed": 0,
            "skip_no_current_track": 0,
            "skip_unknown_particle": 0,
            "skip_track_id_0": 0,
            "skip_event_inactive": 0,
            "skip_parse_failed": 0,
        }
        self.require_event_active = True

        # ===== НОВЫЕ ПОЛЯ: для работы со слоями и дозой =====
        self.layers = []  # Список слоев
        self.global_xmin = -5.0  # значения по умолчанию
        self.global_xmax = 5.0
        self.global_ymin = -5.0
        self.global_ymax = 5.0
        self.global_zmin = -5.0
        self.global_zmax = 5.0
        self.volume_to_layer = {}  # Словарь для быстрого поиска слоя по имени volume
        self.material_densities = {}  # Словарь {имя_материала: плотность}
        # ===== КОНЕЦ НОВЫХ ПОЛЕЙ =====

    @staticmethod
    def mev_formatter(x, pos):
        """Универсальный форматтер для оси в MeV"""
        if abs(x) < 1e-4:
            return f"{x:.2f} MeV"
        elif abs(x) >= 1.0:
            return f"{x:.3f} MeV"
        else:
            return f"{x:.4f} MeV"

    @staticmethod
    def kev_formatter(x, pos):
        """Форматтер для keV (когда weights уже умножены на 1000)"""
        return f"{x:.3f} keV"

    def load_csv_file(self, file_path):
        try:
            print(f"[INFO] Загрузка CSV файла: {file_path}")
            dtype_map = {
                "thread": "string",
                "volume": "string",
                "process": "string",
                "particle": "string",
                "step_number": "Int32",
                "track_id": "Int32",
                "parent_id": "Int32",
                "event_id": "Int32",
                "x_mm": "float32",
                "y_mm": "float32",
                "z_mm": "float32",
                "kinetic_energy_mev": "float32",
                "energy_loss_mev": "float32",
                "process_energy_loss_mev": "float32",
                "step_length_mm": "float32",
                "track_length_mm": "float32",
            }
            try:
                df = pd.read_csv(
                    file_path,
                    encoding="utf-8",
                    dtype=dtype_map,
                    engine="pyarrow",
                )
            except Exception:
                df = pd.read_csv(
                    file_path,
                    encoding="utf-8",
                    dtype=dtype_map,
                    low_memory=False,
                )
            print(f"[INFO] Загружено {len(df)} строк, {len(df.columns)} колонок")
            print(f"[INFO] Колонки: {list(df.columns)}")

            required_columns = [
                "thread", "step_number", "x_mm", "y_mm", "z_mm",
                "kinetic_energy_mev", "energy_loss_mev", "step_length_mm",
                "track_length_mm", "volume", "process", "track_id",
                "parent_id", "particle"
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                for col in missing_columns:
                    if col == "is_primary":
                        df[col] = (df["parent_id"] == 0) if "parent_id" in df.columns else False
                    elif col == "is_secondary":
                        df[col] = (df["parent_id"] != 0) if "parent_id" in df.columns else True
                    elif col == "generation":
                        df[col] = 0
                    else:
                        df[col] = 0

            numeric_columns = [
                "x_mm", "y_mm", "z_mm", "kinetic_energy_mev",
                "energy_loss_mev", "step_length_mm", "track_length_mm",
                "process_energy_loss_mev"
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

            if "generation" not in df.columns and "parent_id" in df.columns and "track_id" in df.columns:
                parent_df = df[["track_id", "parent_id"]].dropna().drop_duplicates("track_id")
                self._csv_parent_map = dict(zip(parent_df["track_id"].astype(int).tolist(),
                                                parent_df["parent_id"].astype(int).tolist()))
                self._csv_generation_cache = {}
                df["generation"] = df["track_id"].astype(int).map(lambda tid: self.get_generation_csv(tid, df))

            if "is_primary" not in df.columns and "parent_id" in df.columns:
                df["is_primary"] = df["parent_id"] == 0
            if "is_secondary" not in df.columns and "parent_id" in df.columns:
                df["is_secondary"] = df["parent_id"] != 0
            if "is_first_step" not in df.columns and "step_number" in df.columns and "track_id" in df.columns:
                df["is_first_step"] = df.groupby("track_id")["step_number"].transform("min") == df["step_number"]

            print(f"[INFO] CSV загружен успешно")
            print(f"[INFO] Статистика:")
            print(f" Всего записей: {len(df)}")
            print(f" Уникальных частиц: {df['particle'].nunique() if 'particle' in df.columns else 'N/A'}")
            print(f" Типы частиц: {df['particle'].unique()[:10] if 'particle' in df.columns else 'N/A'}")
            print(f" Первичных: {df['is_primary'].sum() if 'is_primary' in df.columns else 'N/A'}")
            print(f" Вторичных: {df['is_secondary'].sum() if 'is_secondary' in df.columns else 'N/A'}")
            return df

        except Exception as e:
            print(f"[ERROR] Ошибка загрузки CSV: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_dataframe(self, df, filepath, format_type="csv"):
        try:
            fmt = format_type.lower()
            if fmt == "csv":
                df.to_csv(filepath, index=False, encoding="utf-8-sig")
            elif fmt in ["xlsx", "excel"]:
                df.to_excel(filepath, index=False, engine="openpyxl")
            elif fmt == "dat":
                df.to_csv(filepath, sep="\t", index=False, header=True, encoding="utf-8")
            else:
                return False
            return True
        except Exception as e:
            print(f"Ошибка при сохранении файла {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_figure(self, fig, filepath, fmt='png', dpi=300):
        """
        Сохраняет matplotlib Figure в файл
        Поддерживаемые форматы: png, jpg, pdf, svg
        """
        try:
            supported = {'png', 'jpg', 'jpeg', 'pdf', 'svg'}
            if fmt.lower() not in supported:
                print(f"Формат {fmt} не поддерживается. Используется png.")
                fmt = 'png'
            fig.savefig(
                filepath,
                dpi=dpi,
                bbox_inches='tight',
                format=fmt.lower(),
                transparent=False
            )
            return True
        except Exception as e:
            print(f"Ошибка сохранения графика: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_generation_csv(self, track_id, df):
        try:
            tid = int(track_id)
        except Exception:
            return 0
        cache = getattr(self, "_csv_generation_cache", None)
        parent_map = getattr(self, "_csv_parent_map", None)
        if cache is not None and tid in cache:
            return cache[tid]
        if parent_map is None:
            parent_rows = df[df["track_id"] == tid]
            if parent_rows.empty:
                return 0
            try:
                parent_id = int(parent_rows["parent_id"].iloc[0])
            except Exception:
                return 0
        else:
            parent_id = int(parent_map.get(tid, 0))
        if parent_id == 0:
            if cache is not None:
                cache[tid] = 0
            return 0
        seen = set()
        gen = 0
        cur = tid
        while True:
            if cur in seen:
                break
            seen.add(cur)
            pid = int(parent_map.get(cur, 0)) if parent_map is not None else 0
            if pid == 0:
                break
            gen += 1
            cur = pid
            if gen > 100000:
                break
        if cache is not None:
            cache[tid] = gen
        return gen

    def parse_energy_to_MeV(self, text):
        m = ENERGY_REGEX.search(text)
        if not m:
            return None
        value = float(m.group("value"))
        unit_raw = m.group("unit").replace(" ", "")
        if unit_raw == "MeV":
            return value
        u = unit_raw.lower()
        if u == "mev":  # milli-eV
            return value * 1e-9
        if u == "ev":
            return value * 1e-6
        if u == "kev":
            return value * 1e-3
        return None

    # ===== НОВЫЙ МЕТОД: Парсинг геометрии из лога =====
    def _parse_geometry_from_log(self, lines):
        """Извлекает информацию о геометрии и материалах из лога."""
        # Паттерны для поиска (адаптируйте под ваш формат лога)
        volume_pattern = re.compile(r"Volume:\s+(\w+).*Material:\s+(\w+).*Density:\s+([\d.]+)\s+g/cm3", re.IGNORECASE)
        material_pattern = re.compile(r"Material:\s+(\w+).*Density:\s+([\d.]+)\s+g/cm3", re.IGNORECASE)
        size_pattern = re.compile(r"Size\s+([XYZ]):\s+([\d.]+)\s+(\w+)", re.IGNORECASE)
        world_size_pattern = re.compile(r"World\s+size:\s+([\d.]+)\s+(\w+)", re.IGNORECASE)
        position_pattern = re.compile(r"Position:\s*\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)\s*(\w+)", re.IGNORECASE)

        current_vol = None
        current_mat = None
        current_density = None
        temp_layers = {}

        print("[GEOMETRY] Начинаю парсинг геометрии...")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Поиск информации о материале и плотности
            mat_match = material_pattern.search(line)
            if mat_match:
                material = mat_match.group(1)
                density = float(mat_match.group(2))
                self.material_densities[material] = density
                print(f"[GEOMETRY] Найден материал: {material}, плотность: {density} г/см³")

            # Поиск информации об объеме
            vol_match = volume_pattern.search(line)
            if vol_match:
                current_vol = vol_match.group(1)
                current_mat = vol_match.group(2)
                current_density = float(vol_match.group(3))
                temp_layers[current_vol] = {
                    "material": current_mat,
                    "density": current_density,
                    "z_min": None,
                    "z_max": None,
                    "x_min": None,
                    "x_max": None,
                    "y_min": None,
                    "y_max": None
                }
                print(f"[GEOMETRY] Найден объем: {current_vol}, материал: {current_mat}")

            # Поиск размеров мира
            world_match = world_size_pattern.search(line)
            if world_match:
                size = float(world_match.group(1))
                unit = world_match.group(2)
                half_size = self._convert_units(size / 2, unit, "length")
                self.global_xmin = -half_size
                self.global_xmax = half_size
                self.global_ymin = -half_size
                self.global_ymax = half_size
                self.global_zmin = -half_size
                self.global_zmax = half_size
                print(f"[GEOMETRY] Мир: размер {size} {unit}, границы: ±{half_size:.2f} мм")

            # Поиск размеров по осям
            size_match = size_pattern.search(line)
            if size_match and current_vol and current_vol in temp_layers:
                axis = size_match.group(1)
                value = float(size_match.group(2))
                unit = size_match.group(3)
                half_value = self._convert_units(value / 2, unit, "length")

                if axis == 'X':
                    temp_layers[current_vol]["x_min"] = -half_value
                    temp_layers[current_vol]["x_max"] = half_value
                elif axis == 'Y':
                    temp_layers[current_vol]["y_min"] = -half_value
                    temp_layers[current_vol]["y_max"] = half_value
                elif axis == 'Z':
                    temp_layers[current_vol]["z_min"] = -half_value
                    temp_layers[current_vol]["z_max"] = half_value

            # Поиск позиции объема (для нецентрированных объемов)
            pos_match = position_pattern.search(line)
            if pos_match and current_vol and current_vol in temp_layers:
                x = float(pos_match.group(1))
                y = float(pos_match.group(2))
                z = float(pos_match.group(3))
                unit = pos_match.group(4)

                x_mm = self._convert_units(x, unit, "length")
                y_mm = self._convert_units(y, unit, "length")
                z_mm = self._convert_units(z, unit, "length")

                # Корректируем границы с учетом позиции
                if temp_layers[current_vol]["x_min"] is not None:
                    half_x = (temp_layers[current_vol]["x_max"] - temp_layers[current_vol]["x_min"]) / 2
                    temp_layers[current_vol]["x_min"] = x_mm - half_x
                    temp_layers[current_vol]["x_max"] = x_mm + half_x

                if temp_layers[current_vol]["y_min"] is not None:
                    half_y = (temp_layers[current_vol]["y_max"] - temp_layers[current_vol]["y_min"]) / 2
                    temp_layers[current_vol]["y_min"] = y_mm - half_y
                    temp_layers[current_vol]["y_max"] = y_mm + half_y

                if temp_layers[current_vol]["z_min"] is not None:
                    half_z = (temp_layers[current_vol]["z_max"] - temp_layers[current_vol]["z_min"]) / 2
                    temp_layers[current_vol]["z_min"] = z_mm - half_z
                    temp_layers[current_vol]["z_max"] = z_mm + half_z

        # Создаем объекты Layer из собранной информации
        self.layers = []
        for vol_name, data in temp_layers.items():
            # Проверяем, что есть границы по Z
            if data["z_min"] is not None and data["z_max"] is not None:
                # Если нет границ по X/Y, используем глобальные
                if data["x_min"] is None:
                    data["x_min"] = self.global_xmin
                    data["x_max"] = self.global_xmax
                if data["y_min"] is None:
                    data["y_min"] = self.global_ymin
                    data["y_max"] = self.global_ymax

                layer = Layer(
                    name=vol_name,
                    material=data["material"],
                    density_g_cm3=data["density"],
                    z_min=data["z_min"],
                    z_max=data["z_max"],
                    unit="мм"
                )

                # Вычисляем объем слоя
                x_size = data["x_max"] - data["x_min"]
                y_size = data["y_max"] - data["y_min"]
                layer.volume_mm3 = x_size * y_size * layer.thickness_mm

                self.layers.append(layer)
                self.volume_to_layer[vol_name] = layer

                print(f"[GEOMETRY] Создан слой: {vol_name}, материал={data['material']}, "
                      f"Z=[{data['z_min']:.2f}, {data['z_max']:.2f}] мм, "
                      f"объем={layer.volume_mm3:.2f} мм³")

        print(f"[GEOMETRY] Всего найдено слоев: {len(self.layers)}")

        # Если слои не найдены, создаем один слой из глобальных границ
        if not self.layers and self.material_densities:
            # Берем первую попавшуюся плотность
            first_mat, first_dens = next(iter(self.material_densities.items()))
            default_layer = Layer(
                name="Default",
                material=first_mat,
                density_g_cm3=first_dens,
                z_min=self.global_zmin,
                z_max=self.global_zmax,
                unit="мм"
            )
            x_size = self.global_xmax - self.global_xmin
            y_size = self.global_ymax - self.global_ymin
            default_layer.volume_mm3 = x_size * y_size * default_layer.thickness_mm
            self.layers.append(default_layer)
            self.volume_to_layer["Default"] = default_layer
            print(f"[GEOMETRY] Создан слой по умолчанию")

    # ===== КОНЕЦ НОВОГО МЕТОДА =====

    def parse_log_file(self, file_path):
        steps_data = []
        step_counter = 0
        self.primary_track_ids.clear()
        self.track_info.clear()
        self.thread_current_track.clear()
        self.thread_event_id = {}
        self.thread_event_active = {}
        self._track_prev_energy = {}

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            i = 0
            lines = list(f)  # Сохраняем все строки в список

            # ===== НОВЫЙ КОД: Парсим геометрию из сохраненных строк =====
            self._parse_geometry_from_log(lines)
            # ===== КОНЕЦ НОВОГО КОДА =====

            for raw in lines:
                line = raw.rstrip("\n")
                s = line.strip()

                if s.startswith("G4WT") and (
                        VERTICES_PASSED_RE.search(s) or EVENT_START_RE.search(s) or PROCESS_ONE_EVENT_RE.search(s)):
                    thread = self._get_thread_from_line(s)
                    # инкремент события для конкретного потока
                    self.thread_event_id[thread] = self.thread_event_id.get(thread, -1) + 1
                    self.thread_event_active[thread] = True
                    # если уже есть текущий трек в потоке --- обновим его event_id
                    if thread in self.thread_current_track:
                        self.thread_current_track[thread]["event_id"] = self.thread_event_id[thread]
                    # сброс prev energy для этого потока (чтобы real_energy_loss не подтекал между событиями)
                    drop = [k for k in self._track_prev_energy.keys() if k[0] == thread]
                    for k in drop:
                        self._track_prev_energy.pop(k, None)
                    continue

                if "### Storing a track" in s and "trackID" in s:
                    try:
                        m = re.search(r"\(([^,]+),trackID=(\d+),parentID=(\d+)\)", s)
                        if m:
                            particle = m.group(1).strip().lower()
                            track_id = int(m.group(2))
                            parent_id = int(m.group(3))
                            if particle == "e-":
                                particle = "electron"
                            elif particle == "e+":
                                particle = "positron"
                            elif particle == "gamma":
                                particle = "gamma"
                            self.track_info[track_id] = {
                                "particle": particle,
                                "parent_id": parent_id,
                                "track_id": track_id
                            }
                            if parent_id == 0:
                                self.primary_track_ids.add(track_id)
                    except Exception:
                        pass
                    continue

                if s.startswith("G4WT") and "Track (trackID" in s and "parentID" in s:
                    thread = self._get_thread_from_line(s)
                    m = re.search(r"trackID\s*(\d+)\s*,\s*parentID\s*(\d+)", s)
                    if m:
                        track_id = int(m.group(1))
                        parent_id = int(m.group(2))
                        particle = "unknown"
                        if track_id in self.track_info:
                            particle = self.track_info[track_id].get("particle", "unknown")
                        event_id = self.thread_event_id.get(thread, 0)
                        self.thread_current_track[thread] = {
                            "particle": particle,
                            "track_id": track_id,
                            "parent_id": parent_id,
                            "event_id": event_id
                        }
                    continue

                if "G4Track Information:" in s:
                    try:
                        tm = re.match(r"^(G4WT\d+)\s*\>", s)
                        thread = tm.group(1) if tm else "Unknown"
                        particle_match = re.search(r"Particle\s*=\s*([^,]+)", s, re.IGNORECASE)
                        track_match = re.search(r"Track ID\s*=\s*(\d+)", s, re.IGNORECASE)
                        parent_match = re.search(r"Parent ID\s*=\s*(\d+)", s, re.IGNORECASE)
                        if particle_match and track_match:
                            particle = particle_match.group(1).strip().lower()
                            track_id = int(track_match.group(1))
                            parent_id = 0
                            if parent_match:
                                parent_id = int(parent_match.group(1))
                            elif track_id in self.track_info:
                                parent_id = int(self.track_info[track_id].get("parent_id", 0))
                            if particle == "e-":
                                particle = "electron"
                            elif particle == "e+":
                                particle = "positron"
                            elif particle == "gamma":
                                particle = "gamma"
                            event_id = self.thread_event_id.get(thread, 0)
                            self.thread_current_track[thread] = {
                                "particle": particle,
                                "track_id": track_id,
                                "parent_id": parent_id,
                                "event_id": event_id
                            }
                    except Exception:
                        pass
                    continue

                if ":------------------------- List of secondaries -------------------------" in s:
                    current_thread = self._get_thread_from_line(s)
                    current_info = self.thread_current_track.get(current_thread, {})
                    parent_track_id = int(current_info.get("track_id", 0))
                    creating_process = current_info.get("process", "unknown")
                    while True:
                        try:
                            nxt = next(lines)
                        except StopIteration:
                            break
                        sec = nxt.rstrip("\n")
                        sec_s = sec.strip()
                        if sec_s.startswith("G4WT"):
                            th = self._get_thread_from_line(sec_s)
                            if th != current_thread:
                                break
                        if ":------------------------------------------------------------------" in sec_s:
                            continue
                        m = re.search(
                            r"([a-zA-Z0-9+\-]+)\s*:\s*energy\s*=\s*([\d.eE+-]+)\s*(\w+)",
                            sec_s
                        )
                        if m:
                            raw_particle = m.group(1).lower().strip()
                            particle_map = {
                                "e-": "electron", "e+": "positron", "gamma": "gamma",
                                "proton": "proton", "neutron": "neutron", "alpha": "alpha", "he4": "alpha"
                            }
                            particle = particle_map.get(raw_particle, raw_particle)
                            energy_mev = self.parse_energy_to_MeV(sec_s)
                            new_track_id = max(self.track_info.keys(), default=0) + 1
                            self.track_info[new_track_id] = {
                                "particle": particle,
                                "parent_id": parent_track_id,
                                "track_id": new_track_id,
                                "initial_energy_mev": energy_mev,
                                "created_by": creating_process
                            }
                            continue
                        if not sec_s:
                            continue
                        if sec_s.startswith(":----") or sec_s.startswith(":---"):
                            break
                    continue

                if s.startswith("G4WT") and self._is_step_data_line(s):
                    self.debug_counters["step_lines_seen"] += 1
                    thread = self._get_thread_from_line(s)
                    current_info = self.thread_current_track.get(thread)
                    if not current_info:
                        self.debug_counters["skip_no_current_track"] += 1
                        continue
                    if current_info.get("particle") == "unknown":
                        self.debug_counters["skip_unknown_particle"] += 1
                        continue
                    if int(current_info.get("track_id", 0)) == 0:
                        self.debug_counters["skip_track_id_0"] += 1
                        continue
                    step_counter += 1
                    step_data = self._parse_step_line(s, current_info, thread)
                    if step_data:
                        steps_data.append(step_data)
                        self.debug_counters["step_parsed"] += 1
                    else:
                        self.debug_counters["skip_parse_failed"] += 1
                    continue

        if steps_data:
            df = pd.DataFrame(steps_data)
            if "track_id" in df.columns and "step_number" in df.columns:
                df["is_first_step"] = df.groupby("track_id")["step_number"].transform("min") == df["step_number"]
            primary_track_ids = list(self.primary_track_ids)
            df["is_primary"] = df["track_id"].isin(primary_track_ids)
            df["is_secondary"] = ~df["is_primary"]
            df["generation"] = df["track_id"].apply(lambda tid: self.get_generation(tid, self.track_info))
            return df

        print("Парсинг завершен. Не найдено записей.")
        print("[PARSER DEBUG] Counters:", self.debug_counters)
        return pd.DataFrame()

    def _extract_track_id_from_storing(self, line):
        """Извлекает track_id из строки создания трека"""
        match = re.search(r'trackID=(\d+)', line)
        return int(match.group(1)) if match else None

    def get_generation(self, track_id, track_info):
        """
        Итеративно вычисляет поколение (generation) трека.
        Защищено от циклов и очень длинных цепочек.
        """
        generation = 0
        current = track_id
        seen = set()  # отслеживаем посещенные track_id для обнаружения цикла
        while current != 0:
            if current in seen:
                print(f"[WARNING] Обнаружен ЦИКЛ в цепочке для track_id={track_id}")
                return -1  # или любое отрицательное значение как сигнал ошибки
            seen.add(current)
            parent_info = track_info.get(current, {})
            parent_id = parent_info.get('parent_id', 0)
            if parent_id == 0:
                break  # дошли до первичной
            current = parent_id
            generation += 1
            # Защита от аномально длинных цепочек (маловероятно, но на всякий)
            if generation > 10000:
                print(f"[WARNING] Очень длинная цепочка (>10000) для track_id={track_id}")
                return generation
        return generation

    def filter_secondary_first_step(self, df):
        """Фильтровать вторичные частицы по первому шагу"""
        if not hasattr(self, 'filter_secondary_first_step_flag') or not self.filter_secondary_first_step_flag:
            return df
        if 'is_first_step' not in df.columns:
            df['is_first_step'] = df.groupby('track_id')['step_number'].transform('min') == df['step_number']
        secondary_mask = df['is_secondary']
        first_step_mask = df['is_first_step']
        return df[df['is_primary'] | (secondary_mask & first_step_mask)]

    # ===== ИЗМЕНЕННЫЙ МЕТОД: _parse_step_line с расчетом дозы =====
    def _parse_step_line(self, line, current_info, thread):
        try:
            clean_line = re.sub(r"^G4WT\d+\s*\>\s*", "", line).strip()
            m = re.match(r"(\d+)", clean_line)
            if not m:
                return None
            step_number = int(m.group(1))

            length_matches = re.findall(
                r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(fm|pm|nm|um|mm|cm|m|Ang)",
                clean_line
            )
            if len(length_matches) < 5:
                return None

            x, xu = float(length_matches[0][0]), length_matches[0][1]
            y, yu = float(length_matches[1][0]), length_matches[1][1]
            z, zu = float(length_matches[2][0]), length_matches[2][1]

            x = self._convert_units(x, xu, "length")
            y = self._convert_units(y, yu, "length")
            z = self._convert_units(z, zu, "length")

            stepLeng, stepLeng_unit = float(length_matches[-2][0]), length_matches[-2][1]
            trakLeng, trakLeng_unit = float(length_matches[-1][0]), length_matches[-1][1]

            stepLeng = self._convert_units(stepLeng, stepLeng_unit, "length")
            trakLeng = self._convert_units(trakLeng, trakLeng_unit, "length")

            energy_matches = list(ENERGY_REGEX.finditer(clean_line))
            if len(energy_matches) < 2:
                return None

            kineE = self.parse_energy_to_MeV(energy_matches[0].group(0))
            dEStep = self.parse_energy_to_MeV(energy_matches[1].group(0))

            if kineE is None or dEStep is None:
                return None

            if getattr(self, "require_event_active", True):
                if not self.thread_event_active.get(thread, False):
                    self.debug_counters["skip_event_inactive"] += 1
                    return None

            event_id = self.thread_event_id.get(thread, -1)
            if event_id < 0:
                return None

            track_id = int(current_info.get("track_id", 0))
            track_key = (thread, event_id, track_id)

            prev_energy = self._track_prev_energy.get(track_key)
            edep = max(float(dEStep), 0.0)
            real_energy_loss = max(prev_energy - kineE, 0.0) if prev_energy is not None else 0.0

            self._track_prev_energy[track_key] = kineE

            tail = clean_line.split()
            if len(tail) < 3:
                return None

            volume = tail[-2]
            process = tail[-1]

            if thread in self.thread_current_track:
                self.thread_current_track[thread]["process"] = process

            # Формируем базовый словарь с данными
            step_data = {
                "thread": thread,
                "event_id": event_id,
                "step_number": step_number,
                "x_mm": x, "y_mm": y, "z_mm": z,
                "kinetic_energy_mev": kineE,
                "energy_loss_mev": real_energy_loss,
                "process_energy_loss_mev": edep,
                "step_length_mm": stepLeng,
                "track_length_mm": trakLeng,
                "volume": volume,
                "process": process,
                "track_id": track_id,
                "parent_id": int(current_info.get("parent_id", 0)),
                "particle": current_info.get("particle", "unknown"),
            }

            # ===== НОВЫЙ КОД: Расчет дозы =====
            step_data['dose_gray'] = 0.0
            step_data['dose_rad'] = 0.0
            step_data['layer_name'] = 'unknown'

            if volume in self.volume_to_layer:
                layer = self.volume_to_layer[volume]
                step_data['layer_name'] = layer.name
                rho_g_mm3 = layer.density_g_mm3
                dE_mev = step_data['process_energy_loss_mev']

                # Добавляем энерговыделение к слою для статистики
                layer.add_energy_deposition(dE_mev)

                # Оценка массы области взаимодействия
                # Используем step_length как характерный размер
                step_length_mm = step_data.get('step_length_mm', 0.1)
                if step_length_mm <= 0:
                    step_length_mm = 0.1

                # Предполагаем сферическую область взаимодействия
                # Объем сферы = 4/3 * π * r³
                radius_mm = step_length_mm / 2
                volume_mm3 = (4 / 3) * 3.14159 * (radius_mm ** 3)

                # Масса в граммах: объем(мм³) * плотность(г/мм³)
                mass_g = volume_mm3 * rho_g_mm3

                if mass_g > 0:
                    # 1 Гр = 1 Дж/кг = 6.242e12 МэВ/г
                    # Доза в Гр = dE(МэВ) / масса(г) / 6.242e12
                    dose_gray = dE_mev / mass_g / 6.242e12
                    step_data['dose_gray'] = dose_gray
                    step_data['dose_rad'] = dose_gray * 100  # 1 Гр = 100 рад
            # ===== КОНЕЦ НОВОГО КОДА =====

            return step_data

        except Exception:
            return None

    # ===== КОНЕЦ ИЗМЕНЕННОГО МЕТОДА =====

    def analyze_interaction_chains(self, df):
        """Анализ цепочек взаимодействий"""
        # print("\n=== ЦЕПОЧКИ ВЗАИМОДЕЙСТВИЙ ===")
        # Группируем по parent_id для построения деревьев
        for parent_id in sorted(df['parent_id'].unique()):
            children = df[df['parent_id'] == parent_id]
            if not children.empty:
                parent_row = df[df['track_id'] == parent_id]
                if not parent_row.empty:
                    parent_particle = parent_row['particle'].iloc[0]
                    parent_energy = parent_row['kinetic_energy_mev'].iloc[0]
                    # print(f"\nРодитель: {parent_particle} (id={parent_id}, E={parent_energy:.3f} МэВ)")
                    # for _, child in children.iterrows():
                    #     print(
                    #         f" → {child['particle']} (id={child['track_id']}, E={child['kinetic_energy_mev']:.3f} МэВ)")

    def check_physics_consistency(self, df):
        """Проверка физической согласованности данных"""
        # print("\n=== ПРОВЕРКА ФИЗИЧЕСКОЙ СОГЛАСОВАННОСТИ ===")
        primaries = df[df['is_primary']]
        secondaries = df[df['is_secondary']]
        # if primaries.empty or secondaries.empty:
        #     print("Недостаточно данных для проверки")
        #     return
        # 1. Энергия вторичных не должна превышать энергию первичных
        max_primary_energy = primaries['kinetic_energy_mev'].max()
        high_energy_secondaries = secondaries[
            secondaries['kinetic_energy_mev'] > max_primary_energy * 0.9
            ]
        # if len(high_energy_secondaries) > 0:
        #     print(f"⚠️ ВНИМАНИЕ: Найдены {len(high_energy_secondaries)} вторичных частиц")
        #     print(f" с энергией > 90% от максимальной первичной ({max_primary_energy:.3f} МэВ)")
        #     print(" Возможные причины:")
        #     print(" - Ошибка классификации")
        #     print(" - Ошибка в данных Geant4")
        #     print(" - Особые физические процессы (деление ядра и т.д.)")
        # 2. Общая энергия должна сохраняться (очень грубая проверка)
        total_primary_energy = primaries['kinetic_energy_mev'].sum()
        total_secondary_energy = secondaries['kinetic_energy_mev'].sum()
        # print(f"\nЭнергетический баланс:")
        # print(f"Суммарная энергия первичных: {total_primary_energy:.3f} МэВ")
        # print(f"Суммарная энергия вторичных: {total_secondary_energy:.3f} МэВ")
        # print(f"Соотношение: {total_secondary_energy / total_primary_energy:.3%}")
        # 3. Типичные вторичные частицы
        expected_secondaries = ['electron', 'positron', 'gamma', 'proton', 'neutron']
        unexpected_primaries = []
        for particle in primaries['particle'].unique():
            if particle in expected_secondaries:
                unexpected_primaries.append(particle)
        # if unexpected_primaries:
        #     print(f"\n⚠️ ВНИМАНИЕ: Следующие типы частиц обычно являются вторичными,")
        #     print(f" но помечены как первичные: {unexpected_primaries}")

    def _debug_primary_identification(self, df):
        """Отладка определения первичных частиц"""
        # print("\n=== ОТЛАДКА ОПРЕДЕЛЕНИЯ ПЕРВИЧНЫХ ЧАСТИЦ ===")
        #
        # # Смотрим распределение track_id и parent_id
        # print("\nУникальные track_id:", sorted(df['track_id'].unique())[:20])
        # print("Количество уникальных track_id:", len(df['track_id'].unique()))
        print("\nРаспределение track_id:")
        track_counts = df['track_id'].value_counts().sort_index()
        for track_id, count in track_counts.head(10).items():
            print(f" track_id={track_id}: {count} записей")
        print("\nРаспределение parent_id:")
        parent_counts = df['parent_id'].value_counts().sort_index()
        for parent_id, count in parent_counts.head(10).items():
            print(f" parent_id={parent_id}: {count} записей")

        # Проверяем частицы с track_id=1
        track_1_df = df[df['track_id'] == 1]
        print(f"\nЗаписей с track_id=1: {len(track_1_df)}")
        if not track_1_df.empty:
            print("Распределение parent_id для track_id=1:")
            print(track_1_df['parent_id'].value_counts())

        # Проверяем, все ли track_id=1 имеют parent_id=0
        # if (track_1_df['parent_id'] == 0).all():
        #     print("✓ Все частицы с track_id=1 имеют parent_id=0 (ожидаемо для первичных)")
        # else:
        #     print("⚠️ НЕКОТОРЫЕ частицы с track_id=1 имеют parent_id != 0!")

        # Проверяем количество уникальных комбинаций (thread, track_id)
        if 'thread' in df.columns:
            unique_tracks = df.groupby(['thread', 'track_id']).size()
            print(f"\nУникальных комбинаций (thread, track_id): {len(unique_tracks)}")

    def analyze_tracks_correctly(self, df):
        """Анализ по трекам, а не по шагам"""
        # print("\n=== ПРАВИЛЬНЫЙ АНАЛИЗ ПО ТРЕКАМ ===")
        # Группируем по уникальным трекам (thread, track_id)
        tracks = df.groupby(['thread', 'track_id', 'parent_id']).agg({
            'particle': 'first',
            'kinetic_energy_mev': 'mean',
            'step_number': 'count'
        }).reset_index()
        tracks.columns = ['thread', 'track_id', 'parent_id', 'particle',
                          'avg_energy', 'n_steps']

        # Анализ цепочек рождения
        # print("\n--- ЦЕПОЧКИ РОЖДЕНИЯ ---")
        # Строим граф рождения частиц
        birth_chains = {}
        for _, row in tracks.iterrows():
            if row['parent_id'] > 0:
                birth_chains.setdefault(row['parent_id'], []).append(row['track_id'])

        # Истинные первичные: parent_id=0 И (track_id=1 ИЛИ track_id=2)
        # Потому что в логе видно, что в событии 2 первичные частицы
        true_primaries = tracks[tracks['parent_id'] == 0]
        suspicious_primaries = tracks[
            (tracks['parent_id'] == 0) &
            (tracks['track_id'] > 2)
            ]

        # print(f"Истинные первичные (track_id=1,2): {len(true_primaries)}")
        # print(f"Подозрительные 'первичные' (track_id>2): {len(suspicious_primaries)}")
        # if len(suspicious_primaries) > 0:
        #     print("\nПодозрительные 'первичные' частицы:")
        #     print(suspicious_primaries[['thread', 'track_id', 'particle', 'avg_energy']].head(10))

        # Пересчитываем статистику с исправленной классификацией
        df['is_true_primary'] = df['track_id'].isin(self.primary_track_ids)
        df['is_true_secondary'] = ~df['is_true_primary']

        # print(f"\n--- ИСПРАВЛЕННАЯ СТАТИСТИКА ---")
        # print(f"Истинно первичных частиц: {df['is_true_primary'].sum()}")
        # print(f"Истинно вторичных частиц: {df['is_true_secondary'].sum()}")

        # Анализ энергий
        primaries_energy = df[df['is_true_primary']]['kinetic_energy_mev']
        secondaries_energy = df[df['is_true_secondary']]['kinetic_energy_mev']

        print(f"\nСредняя энергия истинно первичных: {primaries_energy.mean():.3f} МэВ")
        print(f"Средняя энергия истинно вторичных: {secondaries_energy.mean():.3f} МэВ")

        # Анализ типов частиц
        print(f"\nТипы истинно первичных частиц:")
        print(df[df['is_true_primary']]['particle'].value_counts())
        print(f"\nТипы истинно вторичных частиц:")
        print(df[df['is_true_secondary']]['particle'].value_counts())

        return tracks, true_primaries, suspicious_primaries

    def analyze_suspicious_tracks(self, df):
        """Анализ подозрительных треков"""
        # print("\n=== АНАЛИЗ ПОДОЗРИТЕЛЬНЫХ ТРЕКОВ ===")
        # Ищем треки с parent_id=0 но track_id>2
        suspicious = df[
            (df['parent_id'] == 0) &
            (df['track_id'] > 2)
            ]
        # if len(suspicious) == 0:
        #     print("Нет подозрительных треков - отлично!")
        #     return
        # print(f"Найдено {len(suspicious)} подозрительных треков")
        # Группируем по thread и track_id
        suspicious_tracks = suspicious.groupby(['thread', 'track_id', 'particle']).agg({
            'kinetic_energy_mev': ['mean', 'min', 'max'],
            'step_number': 'count'
        }).reset_index()
        suspicious_tracks.columns = ['thread', 'track_id', 'particle',
                                     'avg_energy', 'min_energy', 'max_energy', 'n_steps']
        # print("\nТоп-10 подозрительных треков:")
        # print(suspicious_tracks.sort_values('avg_energy', ascending=False).head(10))
        # Проверяем, может ли это быть:
        # 1. Ошибка парсинга (неправильно определили parent_id)
        # 2. Особые случаи в Geant4 (например, ядерные реакции)
        # 3. Проблемы с многопоточностью
        # Анализируем энергии - если энергия ~10 МэВ, это похоже на первичную
        # high_energy_suspicious = suspicious_tracks[suspicious_tracks['avg_energy'] > 5]
        # print(f"\nПодозрительных треков с энергией >5 МэВ: {len(high_energy_suspicious)}")
        # if len(high_energy_suspicious) > 0:
        #     print("Это могут быть:")
        #     print("1. Дополнительные первичные частицы (если в событии >2 первичных)")
        #     print("2. Ошибка парсинга parent_id")
        #     print("3. Особые процессы в Geant4")

    def _get_particle_color(self, particle_name, is_primary, track_id=None, particle_count=1):
        particle_lower = particle_name.lower()
        color_map = {
            'e-': ('red', '#FF4500', 'electron'),
            'electron': ('red', '#FF4500', 'electron'),
            'e+': ('blue', '#0000FF', 'positron'),
            'positron': ('blue', '#0000FF', 'positron'),
            'proton': ('royalblue', '#4169E1', 'proton'),
            'neutron': ('#DAA520', '#B8860B', 'neutron'),
            'gamma': ('#228B22', '#006400', 'γ'),
            'alpha': ('#FF8C00', '#CD853F', 'α'),
            'he4': ('#FF8C00', '#CD853F', 'α'),
            'muon': ('#8B4513', '#A52A2A', 'μ'),
            'pion+': ('#9932CC', '#BA55D3', 'π+'),
            'pion-': ('#8B008B', '#9370DB', 'π-'),
            'pion0': ('#C71585', '#DB7093', 'π0'),
        }
        for key, (primary_color, secondary_color, label) in color_map.items():
            if key in particle_lower:
                return (primary_color, label) if is_primary else (secondary_color, label)
        if self._is_ion_particle(particle_name):
            if not is_primary:
                hash_val = int(hashlib.md5(particle_name.encode()).hexdigest()[:8], 16)
                hue = hash_val % 360 / 360.0
                r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.8)
                color = (r, g, b)
            else:
                normalized = min(track_id / max(particle_count, 10), 1.0) if track_id else 0.3
                gray_value = 0.1 + 0.8 * normalized
                gray_value = max(0.1, min(0.9, gray_value))
                color = (gray_value, gray_value, gray_value)
            return color, particle_name
        hash_val = int(hashlib.md5(particle_name.encode()).hexdigest()[:8], 16)
        hue = hash_val % 360 / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.7)
        if not is_primary:
            r = min(1.0, r + 0.15)
            g = min(1.0, g + 0.15)
            b = min(1.0, b + 0.15)
        return (r, g, b), particle_name

    def _is_ion_particle(self, particle_name):
        elementary_particles = ['e-', 'e+', 'gamma', 'proton', 'neutron', 'muon',
                                'mu-', 'mu+', 'pion', 'pi+', 'pi-', 'pi0', 'kaon',
                                'alpha', 'positron', 'electron', 'photon']
        particle_lower = particle_name.lower()
        for elem in elementary_particles:
            if elem in particle_lower:
                return False
        element_patterns = [r'^[A-Z][a-z]?\d*', r'ion', r'[A-Z][a-z]?[+-]?\d*$']
        for pattern in element_patterns:
            if re.match(pattern, particle_name):
                return True
        if re.search(r'\d+', particle_name) and not particle_name.isdigit():
            if re.search(r'[A-Za-z]', particle_name):
                return True
        return False

    def _parse_particle_count(self, line):
        try:
            # Общий паттерн: "число частица of" или "The run is: число"
            patterns = [
                r'The run is:\s*(\d+)\s*',  # The run is: 1000
                r'(\d+)\s+(\w[\w+-]*)\s+of',  # 1000 gamma of, 1000 e- of, 1000 proton of
                r'Primary\s+particles\s*:\s*(\d+)',  # Primary particles: 1000
                r'Number\s+of\s+primaries\s*:\s*(\d+)'  # Number of primaries: 1000
            ]
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Если паттерн имеет группу для названия частицы (вторую группу)
                    if len(match.groups()) >= 2:
                        particle_name = match.group(2).lower()
                        # Определяем тип первичной частицы
                        if particle_name in ['gamma', 'γ']:
                            self.primary_particle_type = 'gamma'
                        elif particle_name in ['e-', 'electron']:
                            self.primary_particle_type = 'electron'
                        elif particle_name in ['e+', 'positron']:
                            self.primary_particle_type = 'positron'
                        elif particle_name in ['proton']:
                            self.primary_particle_type = 'proton'
                        elif particle_name in ['neutron']:
                            self.primary_particle_type = 'neutron'
                        # Добавьте другие типы по мере необходимости
                    self.particle_count = int(match.group(1))
                    return
        except Exception as e:
            print(f"Ошибка парсинга количества частиц: {e}")

    # внутри класса Parser
    def export_all_typical_plots(self, df, output_dir, prefix="plot", dpi=200, formats=None):
        """
        Экспортирует набор типичных графиков в указанную папку.
        Параметры:
        df - DataFrame с данными
        output_dir - путь к папке
        prefix - префикс имени файлов (например "primary", "secondary")
        dpi - разрешение
        formats - список форматов, например ['png', 'pdf']
        """
        if formats is None:
            formats = ['png']  # по умолчанию только png
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        exported = []
        plots_to_generate = [
            ("energy", lambda: self._visualize_energy_distributions(df, prefix, use_cache=False)),
            ("energy_loss", lambda: self._visualize_energy_loss_distribution(df, prefix, use_cache=False)),
            ("dE", lambda: self._visualize_dE_distribution(df, prefix, use_cache=False)),
            ("processes", lambda: self._visualize_additional_plots(df, prefix, use_cache=False)),
            ("heatmap", lambda: self._visualize_heatmap(df, prefix, use_cache=False)),
            # можно добавить 3D, но он тяжелый и часто не нужен в массовом экспорте
        ]
        for plot_name, plot_func in plots_to_generate:
            try:
                fig = plot_func()
                if fig is None:
                    continue
                base_name = f"{prefix}_{plot_name}"
                for fmt in formats:
                    filename = os.path.join(output_dir, f"{base_name}.{fmt}")
                    success = self.save_figure(fig, filename, fmt=fmt, dpi=dpi)
                    if success:
                        exported.append(filename)
                # закрываем фигуру, чтобы не засорять память
                plt.close(fig)
            except Exception as e:
                print(f"Ошибка при создании {plot_name} для {prefix}: {e}")
        return exported

    def export_summary_report(self, df, file_path):
        try:
            summary = self.generate_text_summary(df)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            return True
        except Exception as e:
            print(f"Ошибка экспорта сводки: {e}")
            return False

    def _is_energy_summary_line(self, line):
        energy_patterns = [
            r'Energy deposit\s*:\s*([\d.]+)\s*(\w+)',
            r'Total energy deposit\s*:\s*([\d.]+)\s*(\w+)',
            r'Energy deposition\s*:\s*([\d.]+)\s*(\w+)'
        ]
        for pattern in energy_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        return False

    def generate_text_summary(self, df):
        analysis_result = self._analyze_and_compare(df)
        summary = "СВОДКА АНАЛИЗА\n"
        summary += "=" * 50 + "\n"
        summary += analysis_result + "\n\n"
        summary += "ОСНОВНАЯ СТАТИСТИКА:\n"
        summary += f"Всего записей: {len(df)}\n"
        if not df.empty:
            summary += f"Типы частиц: {', '.join(df['particle'].unique())}\n"
            summary += f"Общие потери энергии: {df['energy_loss_mev'].sum():.6f} МэВ\n\n"
        summary += "СТАТИСТИКА ПО ПЕРВИЧНЫМ ЧАСТИЦАМ:\n"
        primary_df = df[df['is_primary']]
        if not primary_df.empty:
            primary_stats = primary_df.groupby('particle').agg({
                'kinetic_energy_mev': ['count', 'min', 'max', 'mean'],
                'energy_loss_mev': 'sum'
            }).round(6)
            for particle in primary_stats.index:
                stats = primary_stats.loc[particle]
                summary += f"{particle}:\n"
                summary += f" Количество: {stats[('kinetic_energy_mev', 'count')]}\n"
                summary += (f" Энергия: {stats[('kinetic_energy_mev', 'min')]:.3f} - "
                            f"{stats[('kinetic_energy_mev', 'max')]:.3f} МэВ\n")
                summary += f" Средняя энергия: {stats[('kinetic_energy_mev', 'mean')]:.3f} МэВ\n"
                summary += f" Суммарные потери: {stats[('energy_loss_mev', 'sum')]:.6f} МэВ\n"
        else:
            summary += "Нет данных по первичным частицам.\n"
        summary += "\nСТАТИСТИКА ПО ВТОРИЧНЫМ ЧАСТИЦАМ:\n"
        secondary_df = df[~df['is_primary']]
        if not secondary_df.empty:
            secondary_stats = secondary_df.groupby('particle').agg({
                'kinetic_energy_mev': ['count', 'min', 'max', 'mean'],
                'energy_loss_mev': 'sum'
            }).round(6)
            for particle in secondary_stats.index:
                stats = secondary_stats.loc[particle]
                summary += f"{particle}:\n"
                summary += f" Количество: {stats[('kinetic_energy_mev', 'count')]}\n"
                summary += (f" Энергия: {stats[('kinetic_energy_mev', 'min')]:.3f} - "
                            f"{stats[('kinetic_energy_mev', 'max')]:.3f} МэВ\n")
                summary += f" Средняя энергия: {stats[('kinetic_energy_mev', 'mean')]:.3f} МэВ\n"
                summary += f" Суммарные потери: {stats[('energy_loss_mev', 'sum')]:.6f} МэВ\n"
        else:
            summary += "Нет данных по вторичным частицам.\n"
        if "process_calls" in self.summary_data:
            summary += self._compare_process_frequency(df)
        return summary

    def _visualize_energy_distributions(
            self,
            df,
            particle_type="частиц",
            show_kde=True,
            show_stats=True,
            min_y_log=0.8,
            use_cache=True,
            normalization='none',
            selected_particles=None,
            xlim=None,
            ylim=None
    ):
        if df.empty:
            return self._create_empty_plot(f'Нет данных для {particle_type} частиц')
        energy_df = df[df['kinetic_energy_mev'] >= 0].copy()
        if energy_df.empty:
            return self._create_empty_plot(f'Нет данных по кинетической энергии > 0 для {particle_type} частиц')
        if selected_particles and len(selected_particles) > 0:
            energy_df = energy_df[energy_df['particle'].isin(selected_particles)]

        # ─── Кэширование ──────────────────────────────────────────────────────────
        if use_cache and self.cache_mgr:
            data_hash = hashlib.md5(
                pd.util.hash_pandas_object(energy_df[['particle', 'kinetic_energy_mev', 'track_id']]).values.tobytes()
            ).hexdigest()
            cache_key = {
                "plot_type": "energy_distribution",
                "particle_type": particle_type,
                "show_kde": show_kde,
                "show_stats": show_stats,
                "min_y_log": min_y_log,
                "normalization": normalization,
                "selected_particles": tuple(selected_particles) if selected_particles else None,
                "data_hash": data_hash[:16]
            }
            cached_fig = self.cache_mgr.load_figure("energy_dist", cache_key)
            if cached_fig is not None:
                print(f"[CACHE HIT] Загружен из кэша: {particle_type} {data_hash[:8]}...")
                return cached_fig

        # ─── Фильтрация и сортировка частиц ───────────────────────────────────────
        counts = energy_df['particle'].value_counts()
        min_steps = 50
        candidates = counts[counts >= min_steps]
        if len(candidates) > 0:
            particles = candidates.sort_values(ascending=False).index.tolist()
        else:
            particles = counts.head(8).index.tolist()
        if not particles:
            return self._create_empty_plot(f'Нет значимых {particle_type} частиц')
        energy_df = energy_df[energy_df['particle'].isin(particles)]

        # ─── Подготовка фигуры ────────────────────────────────────────────────────
        fig = Figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        is_primary = particle_type == "первичных"
        particle_colors = {}
        particle_legend_names = {}
        for particle in particles:
            color, legend_name = self._get_particle_color(particle, is_primary)
            particle_colors[particle] = color
            particle_legend_names[particle] = legend_name

        # ─── Определяем ylabel и use_log_scale ОДИН РАЗ перед циклом ──────────────
        if normalization == 'none':
            ylabel = 'Число шагов'
            use_log_scale = True
        elif normalization == 'particles':
            ylabel = 'Шагов на частицу'
            use_log_scale = False
        elif normalization == 'steps':
            ylabel = 'Доля шагов'
            use_log_scale = False
        elif normalization == 'density':
            ylabel = 'Плотность вероятности (1/МэВ)'
            use_log_scale = False
        else:
            raise ValueError(f"Неизвестный режим нормировки: {normalization}")

        # ─── Построение гистограмм и KDE ──────────────────────────────────────────
        for particle in particles:
            particle_data = energy_df[energy_df['particle'] == particle]
            if len(particle_data) == 0:
                continue
            energies = particle_data['kinetic_energy_mev']
            filtered = energies
            if len(filtered) == 0:
                continue

            # ─── Определяем параметры гистограммы ────────────────────────────────
            if normalization == 'none':
                weights = None
                density = False
            elif normalization == 'particles':
                n_particles = particle_data['track_id'].nunique()
                weights = np.ones_like(filtered) / n_particles if n_particles > 0 else None
                density = False
            elif normalization == 'steps':
                n_steps = len(particle_data)
                weights = np.ones_like(filtered) / n_steps if n_steps > 0 else None
                density = False
            elif normalization == 'density':
                weights = None
                density = True

            # ─── Построение гистограммы ─────────────────────────────────────────
            n, bins, patches = ax.hist(
                filtered,
                bins=100,
                weights=weights,
                density=density,
                color=particle_colors[particle],
                alpha=0.45,
                label=particle_legend_names[particle],
                edgecolor='black',
                linewidth=0.4
            )

            # ─── Построение KDE ─────────────────────────────────────────────────
            if show_kde:
                kde_input = self._prep_kde_input_1d(filtered)
                if kde_input is not None:
                    try:
                        bw = self._safe_bw_method(kde_input, base=0.3)
                        kde = gaussian_kde(kde_input, bw_method=bw)
                        x = np.linspace(kde_input.min(), kde_input.max(), 300)
                        kde_vals = kde(x)
                        if normalization == 'none':
                            # масштабируем KDE под "counts", как у тебя
                            bin_width = (filtered.max() - filtered.min()) / 100
                            if not np.isfinite(bin_width) or bin_width <= 0:
                                bin_width = (kde_input.max() - kde_input.min()) / 100
                            scaled = kde_vals * len(filtered) * bin_width
                        elif normalization == 'density':
                            # KDE уже плотность (1/MeV)
                            scaled = kde_vals
                        else:
                            # нормировки particles/steps
                            if normalization == 'particles':
                                denom = particle_data['track_id'].nunique()
                            else:  # 'steps'
                                denom = len(particle_data)
                            scale_factor = (1.0 / denom) if denom > 0 else 1.0
                            scaled = kde_vals * scale_factor
                        ax.plot(x, scaled,
                                color=particle_colors[particle],
                                lw=1.6, alpha=0.9, ls='-')
                    except Exception as e:
                        # тихо: не спамим, можно оставить один print при DEBUG
                        # print(f"KDE error for {particle}: {e}")
                        pass

        # ─── Оформление осей ПОСЛЕ построения всех гистограмм ───────────────────
        if use_log_scale:
            ax.set_yscale('log')
            ax.set_ylim(bottom=min_y_log)
        else:
            ax.set_yscale('linear')
            ax.autoscale(axis='y')
        ax.set_title(f'Распределение кинетической энергии {particle_type} частиц',
                     fontweight='bold', fontsize=14, pad=16)
        ax.set_xlabel('Кинетическая энергия (МэВ)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(False)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major',
                       length=7, width=1.1, direction='in',
                       labelsize=10,
                       top=True, right=True, bottom=True, left=True)
        ax.tick_params(axis='both', which='minor',
                       length=4, width=0.9, direction='in',
                       top=True, right=True, bottom=True, left=True)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # ─── Легенда ──────────────────────────────────────────────────────────────
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            label_to_handle = dict(zip(labels, handles))
            ordered_handles = []
            ordered_labels = []
            for p in particles:
                leg_name = particle_legend_names.get(p, p)
                if leg_name in label_to_handle:
                    ordered_handles.append(label_to_handle[leg_name])
                    ordered_labels.append(leg_name)
            n = len(ordered_labels)
            ncol = 1 if n <= 5 else 2 if n <= 12 else 3
            # Определяем положение легенды в зависимости от количества элементов
            if n <= 5:
                legend_loc = 'upper right'
            elif n <= 10:
                legend_loc = 'upper left'
            else:
                legend_loc = 'center left'
            legend = ax.legend(
                ordered_handles,
                ordered_labels,
                title='Тип частицы',
                loc=legend_loc,
                framealpha=0.35,
                fancybox=True,
                fontsize=9.5,
                title_fontsize=11,
                ncol=ncol if legend_loc in ['center left', 'upper left'] else 1,
                borderpad=0.6
            )

        # ─── Статистика --- располагаем в противоположном углу ──────────────────
        stats_artist = None
        if show_stats:
            stats_text = []
            total_steps_all = len(energy_df)  # Все шаги в выборке
            for particle in particles:
                sub = energy_df[energy_df['particle'] == particle]
                if len(sub) == 0:
                    continue
                n_steps = len(sub)
                mean_e = sub['kinetic_energy_mev'].mean()
                if normalization == 'none':
                    # Без нормировки: просто количество шагов
                    stats_text.append(f"{particle}: {n_steps} шагов")
                    stats_text.append(f" μ = {mean_e:.3f} МэВ")
                elif normalization == 'particles':
                    # Нормировано на частицу: нужны только относительные значения
                    stats_text.append(f"{particle}: μ = {mean_e:.3f} МэВ")
                elif normalization == 'steps':
                    # Нормировано на шаги: показываем долю
                    percentage = (n_steps / total_steps_all * 100) if total_steps_all > 0 else 0
                    stats_text.append(f"{particle}: {percentage:.1f}% шагов")
                    stats_text.append(f" μ = {mean_e:.3f} МэВ")
                else:  # 'density'
                    # Плотность: статистика распределения
                    std_e = sub['kinetic_energy_mev'].std()
                    stats_text.append(f"{particle}: μ = {mean_e:.3f} МэВ")
                    stats_text.append(f" σ = {std_e:.3f} МэВ")
            n_rare = len(counts) - len(particles)
            if n_rare > 0:
                rare_steps = counts[~counts.index.isin(particles)].sum()
                rare_pct = rare_steps / len(energy_df) * 100 if len(energy_df) > 0 else 0
                stats_text.append(f"... ещё {n_rare} частиц")
            if stats_text:
                stats_cols = 1 if len(stats_text) <= 8 else 2 if len(stats_text) <= 20 else 3
                stats_block = self._format_stats_columns(stats_text, ncol=stats_cols)
                # Определяем положение статистики в зависимости от положения легенды
                if legend_loc == 'upper right':
                    stats_x, stats_y, stats_ha, stats_va = 0.02, 0.98, 'left', 'top'  # верхний левый
                elif legend_loc == 'upper left':
                    stats_x, stats_y, stats_ha, stats_va = 0.98, 0.98, 'right', 'top'  # верхний правый
                elif legend_loc == 'center left':
                    stats_x, stats_y, stats_ha, stats_va = 0.98, 0.98, 'right', 'top'  # верхний правый
                else:
                    stats_x, stats_y, stats_ha, stats_va = 0.02, 0.98, 'left', 'top'  # верхний левый по умолчанию
                # Если статистики много, сдвигаем ниже
                if len(stats_text) > 8:
                    stats_y = 0.02
                    stats_va = 'bottom'
                stats_artist = ax.text(
                    stats_x, stats_y,
                    stats_block,
                    transform=ax.transAxes,
                    fontsize=8.2,
                    va=stats_va, ha=stats_ha,
                    bbox=dict(boxstyle='square', pad=0.45, fc='white', alpha=0.25, edgecolor='gray', linewidth=0.5)
                )
        else:
            # Если нет легенды, статистику размещаем в правом верхнем углу
            if show_stats:
                stats_text = []
                for particle in particles:
                    sub = energy_df[energy_df['particle'] == particle]
                    if len(sub) == 0:
                        continue
                    n_tracks = sub['track_id'].nunique()
                    n_steps = len(sub)
                    mean_e = sub['kinetic_energy_mev'].mean()
                    if normalization == 'none':
                        stats_text.append(f"{particle}: треков = {n_tracks}, шагов = {n_steps}, μ = {mean_e:.3f} МэВ")
                    elif normalization == 'particles':
                        stats_text.append(f"{particle}: {n_tracks} треков, {n_steps} шагов")
                    elif normalization == 'steps':
                        stats_text.append(f"{particle}: {n_steps} шагов")
                    else:  # 'density'
                        stats_text.append(f"{particle}: треков = {n_tracks}, μ = {mean_e:.3f} МэВ")
                if stats_text:
                    stats_cols = 1 if len(stats_text) <= 8 else 2 if len(stats_text) <= 20 else 3
                    stats_block = self._format_stats_columns(stats_text, ncol=stats_cols)
                    stats_artist = ax.text(
                        0.98, 0.98,
                        stats_block,
                        transform=ax.transAxes,
                        fontsize=8.2,
                        va='top', ha='right',
                        bbox=dict(boxstyle='square', pad=0.45, fc='white', alpha=0.25, edgecolor='gray', linewidth=0.5)
                    )
        # Важно! Сохраняем объект статистики в фигуру
        fig._stats_artist = stats_artist

        # Регулировка отступов
        if handles and legend_loc in ['upper left', 'center left']:
            fig.subplots_adjust(left=0.15, right=0.85)
        elif stats_artist and stats_artist.get_ha() == 'right':
            fig.subplots_adjust(left=0.1, right=0.75)
        else:
            fig.subplots_adjust(left=0.1, right=0.85)
        fig.tight_layout()

        if use_cache and self.cache_mgr:
            self.cache_mgr.save_figure(fig, "energy_dist", cache_key)
            print(f"[CACHE SAVE] Сохранён в кэш: {particle_type} {data_hash[:8]}...")
        return fig

    def _format_stats_columns(self, lines, ncol=2):
        if len(lines) <= 10:
            return '\n'.join(lines)
        cols = [lines[i::ncol] for i in range(ncol)]
        max_len = max(len(c) for c in cols)
        for c in cols:
            while len(c) < max_len:
                c.append("")
        rows = zip(*cols)
        return "\n".join(" ".join(row) for row in rows)

    def _prep_kde_input_1d(self, arr, max_points=250_000):
        """
        Только для KDE: убираем NaN/Inf, проверяем дисперсию,
        при необходимости делаем downsample, ничего не меняя в исходной статистике.
        Возвращает np.ndarray или None если KDE делать нельзя.
        """
        import numpy as np
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 20:
            return None
        # если почти все значения одинаковые -> KDE бессмысленна и падает
        # используем range вместо std, чтобы ловить "ступеньки"
        if (x.max() - x.min()) <= 0:
            return None
        # ограничим количество точек (KDE O(N*M), сильно тормозит на больших)
        if x.size > max_points:
            # равномерная подвыборка, чтобы форма не ломалась
            idx = np.random.choice(x.size, size=max_points, replace=False)
            x = x[idx]
        return x

    def _safe_bw_method(self, x, base=0.3):
        """
        Стабильный bw_method для gaussian_kde.
        Если распределение очень узкое --- расширяем bw, чтобы не было численных проблем.
        """
        import numpy as np
        # относительная "ширина"
        span = x.max() - x.min()
        if span <= 0:
            return base
        # если span очень маленький, увеличим bandwidth
        # (иначе будет вырожденность)
        if span < 1e-9:
            return max(base, 1.0)
        return base

    def _create_empty_plot(self, message):
        """Создает пустой график с сообщением"""
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, message,
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    def _visualize_energy_loss_distribution(
            self,
            df,
            particle_type="частиц",
            show_kde=True,
            show_stats=True,
            min_y_log=0.8,
            use_cache=True,
            normalization='none',
            selected_particles=None,
            xlim=None,
            ylim=None
    ):
        if df.empty:
            return self._create_empty_plot(f'Нет данных для {particle_type} частиц')
        loss_df = df[df['energy_loss_mev'] >= 0].copy()
        if loss_df.empty:
            return self._create_empty_plot(f'Нет данных по потерям энергии > 0 для {particle_type} частиц')
        if selected_particles and len(selected_particles) > 0:
            loss_df = loss_df[loss_df['particle'].isin(selected_particles)]
        # Убираем NaN и бесконечности ПЕРЕД кэшированием
        loss_df = loss_df[np.isfinite(loss_df['energy_loss_mev'])]
        loss_df = loss_df[loss_df['energy_loss_mev'] >= 0]
        loss_df_plot = loss_df[loss_df['energy_loss_mev'] > 0]
        if loss_df.empty:
            return self._create_empty_plot(f'Нет валидных данных для {particle_type} частиц')

        # ─── Кэширование ──────────────────────────────────────────────────────────
        if use_cache and self.cache_mgr:
            data_hash = hashlib.md5(
                pd.util.hash_pandas_object(loss_df[['particle', 'energy_loss_mev']]).values.tobytes()
            ).hexdigest()
            cache_key = {
                "plot_type": "energy_loss",
                "particle_type": particle_type,
                "show_kde": show_kde,
                "show_stats": show_stats,
                "min_y_log": min_y_log,
                "normalization": normalization,
                "selected_particles": tuple(selected_particles) if selected_particles else None,
                "data_hash": data_hash[:16]
            }
            cached_fig = self.cache_mgr.load_figure("energy_loss", cache_key)
            if cached_fig is not None:
                print(f"[CACHE HIT] Energy_loss: {particle_type} {data_hash[:8]}...")
                return cached_fig

        # ─── Фильтрация только значимых частиц ───────────────────────────────────
        counts = loss_df['particle'].value_counts()
        counts_plot = loss_df_plot['particle'].value_counts()
        min_steps = 50
        important_particles = counts[counts >= min_steps]
        important_particles_plot = counts_plot[counts_plot >= min_steps]
        if len(important_particles) == 0:
            important_particles = counts.head(8)
        if len(important_particles_plot) == 0:
            important_particles_plot = counts_plot.head(8)
        particles = important_particles.sort_values(ascending=False).index.tolist()
        loss_df = loss_df[loss_df['particle'].isin(particles)]
        particles_plot = important_particles_plot.sort_values(ascending=False).index.tolist()
        loss_df_plot = loss_df_plot[loss_df_plot['particle'].isin(particles_plot)]

        fig = Figure(figsize=(9, 6.5))
        ax = fig.add_subplot(111)
        is_primary = particle_type == "первичных"
        particle_colors = {}
        particle_legend_names = {}
        for particle in particles:
            color, legend_name = self._get_particle_color(particle, is_primary)
            particle_colors[particle] = color
            particle_legend_names[particle] = legend_name

        # Фильтрация выбросов
        # q_low, q_high = loss_df['energy_loss_mev'].quantile([0.001, 0.999])
        # filtered_loss = loss_df[(loss_df['energy_loss_mev'] >= q_low) & (loss_df['energy_loss_mev'] <= q_high)]
        filtered_loss = loss_df
        q_low, q_high = loss_df_plot['energy_loss_mev'].quantile([0.001, 0.999])
        filtered_loss_plot = loss_df_plot[(loss_df_plot['energy_loss_mev'] >= q_low) &
                                          (loss_df_plot['energy_loss_mev'] <= q_high)]

        # ─── Определяем ylabel и режим построения ─────────────────────────────
        if normalization == 'none':
            ylabel = 'Число шагов'
            use_log_y = True  # Для потерь энергии ВСЕГДА используем логарифм по Y!
        elif normalization == 'particles':
            ylabel = 'Шагов на частицу'
            use_log_y = False
        elif normalization == 'steps':
            ylabel = 'Доля шагов'
            use_log_y = False
        elif normalization == 'density':
            ylabel = 'Плотность вероятности (1/МэВ)'
            use_log_y = False
        else:
            ylabel = 'Число шагов'
            use_log_y = True

        # ─── Построение гистограмм с правильными настройками ─────────────────
        for particle in particles:
            data = filtered_loss_plot[filtered_loss_plot['particle'] == particle]
            if len(data) == 0:
                continue
            # ─── Определяем stat и weights в зависимости от режима ───────────────
            if normalization == 'density':
                stat = 'density'
                weights = None
            else:
                stat = 'count'
                weights = None
            if normalization == 'particles':
                n_particles = data['track_id'].nunique()
                if n_particles > 0:
                    weights = np.ones(len(data)) / n_particles
            elif normalization == 'steps':
                n_steps = len(data)
                if n_steps > 0:
                    weights = np.ones(len(data)) / n_steps

            # Определяем, нужно ли логарифмическое масштабирование по Y
            # Для 'none' режима используем логарифмическую ось Y, для остальных - нет
            if normalization == 'none':
                log_scale_y = True
            else:
                log_scale_y = False

            if weights is not None:
                plot_data = pd.DataFrame({
                    'energy_loss_mev': data['energy_loss_mev'].values,
                    'weights': weights
                })
                sns.histplot(
                    data=plot_data,
                    x='energy_loss_mev',
                    weights='weights',
                    bins=100,
                    kde=show_kde,
                    color=particle_colors[particle],
                    alpha=0.5,
                    label=particle_legend_names[particle],
                    log_scale=(True, log_scale_y),
                    ax=ax,
                    stat=stat
                )
            else:
                sns.histplot(
                    data=data['energy_loss_mev'],
                    bins=60,
                    kde=show_kde,
                    color=particle_colors[particle],
                    alpha=0.5,
                    label=particle_legend_names[particle],
                    log_scale=(True, log_scale_y),
                    ax=ax,
                    stat=stat
                )

        # ─── Настройка осей ─────────────────────────────────────────────────
        # Для режима 'none' устанавливаем логарифмическую ось Y
        if normalization == 'none':
            ax.set_yscale('log')
            if min_y_log:
                ax.set_ylim(bottom=min_y_log)
        # Настраиваем метки оси X для логарифмической шкалы
        ax.set_xlabel('Потеря энергии на шаг (МэВ)', fontsize=12)
        # Форматируем метки оси X для логарифмической шкалы
        from matplotlib.ticker import LogFormatterSciNotation
        # Используем форматтер для логарифмической оси
        formatter = LogFormatterSciNotation(base=10.0, labelOnlyBase=False)
        ax.xaxis.set_major_formatter(formatter)
        # Устанавливаем удобные деления
        ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1))
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Распределение потерь энергии на шаге ({particle_type} частиц)',
                     fontweight='bold', fontsize=14, pad=15)
        ax.grid(False)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major',
                       length=7, width=1.1, direction='in',
                       labelsize=10,
                       top=True, right=True, bottom=True, left=True)
        ax.tick_params(axis='both', which='minor',
                       length=4, width=0.9, direction='in',
                       top=True, right=True, bottom=True, left=True)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # ─── Легенда ──────────────────────────────────────────────────────────────
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            label_to_handle = dict(zip(labels, handles))
            ordered_handles = [label_to_handle[particle_legend_names.get(p, p)] for p in particles
                               if particle_legend_names.get(p, p) in label_to_handle]
            ordered_labels = [particle_legend_names.get(p, p) for p in particles
                              if particle_legend_names.get(p, p) in label_to_handle]
            n = len(ordered_labels)
            # Определяем положение легенды в зависимости от количества элементов
            if n <= 5:
                legend_loc = 'upper right'
                ncol = 1
            elif n <= 10:
                legend_loc = 'upper left'
                ncol = 2
            else:
                legend_loc = 'center left'
                ncol = 2
            legend = ax.legend(
                ordered_handles, ordered_labels,
                title='Тип частицы',
                loc=legend_loc,
                framealpha=0.35,
                fancybox=True,
                fontsize=9.5,
                title_fontsize=11,
                ncol=ncol,
                borderpad=0.6
            )

        # ─── Статистика --- располагаем в противоположном углу ──────────────────
        stats_artist = None
        if show_stats:
            stats_text = []
            total_steps_all = len(filtered_loss)
            for particle in particles:
                data = filtered_loss[filtered_loss['particle'] == particle]
                if len(data) == 0:
                    continue
                n_steps = len(data)
                mean_loss = data['energy_loss_mev'].mean()
                if normalization == 'none':
                    stats_text.append(f"{particle}: {n_steps} шагов")
                    stats_text.append(f" μ = {mean_loss:.3f} МэВ/шаг")
                elif normalization == 'particles':
                    stats_text.append(f"{particle}: μ = {mean_loss:.3f} МэВ/шаг")
                elif normalization == 'steps':
                    percentage = (n_steps / total_steps_all * 100) if total_steps_all > 0 else 0
                    stats_text.append(f"{particle}: {percentage:.1f}% шагов")
                    stats_text.append(f" μ = {mean_loss:.3f} МэВ/шаг")
                else:  # 'density'
                    std_loss = data['energy_loss_mev'].std()
                    stats_text.append(f"{particle}: μ = {mean_loss:.3f} МэВ/шаг")
                    stats_text.append(f" σ = {std_loss:.3f} МэВ/шаг")
            n_rare = len(counts) - len(particles)
            if n_rare > 0:
                rare_steps = counts[~counts.index.isin(particles)].sum()
                rare_pct = rare_steps / len(loss_df) * 100
                stats_text.append(f"... ещё {n_rare} редких частиц")
            if stats_text:
                stats_cols = 1 if len(stats_text) <= 8 else 2 if len(stats_text) <= 20 else 3
                stats_block = self._format_stats_columns(stats_text, ncol=stats_cols)
                # Определяем положение статистики в зависимости от положения легенды
                if legend_loc == 'upper right':
                    stats_x, stats_y, stats_ha, stats_va = 0.02, 0.98, 'left', 'top'  # верхний левый
                elif legend_loc == 'upper left':
                    stats_x, stats_y, stats_ha, stats_va = 0.98, 0.98, 'right', 'top'  # верхний правый
                elif legend_loc == 'center left':
                    stats_x, stats_y, stats_ha, stats_va = 0.98, 0.98, 'right', 'top'  # верхний правый
                else:
                    stats_x, stats_y, stats_ha, stats_va = 0.02, 0.98, 'left', 'top'  # верхний левый
                # Если статистики много, сдвигаем ниже
                if len(stats_text) > 8:
                    stats_y = 0.02
                    stats_va = 'bottom'
                stats_artist = ax.text(
                    stats_x, stats_y,
                    stats_block,
                    transform=ax.transAxes,
                    fontsize=8.8,
                    va=stats_va, ha=stats_ha,
                    bbox=dict(boxstyle='square', pad=0.5, fc='white', alpha=0.22, edgecolor='gray', linewidth=0.5)
                )
                fig._stats_artist = stats_artist
        else:
            # Если нет легенды, статистику размещаем в правом верхнем углу
            if show_stats:
                stats_text = []
                total_steps_all = len(filtered_loss)
                for particle in particles:
                    data = filtered_loss[filtered_loss['particle'] == particle]
                    if len(data) == 0:
                        continue
                    n_steps = len(data)
                    mean_loss = data['energy_loss_mev'].mean()
                    std_loss = data['energy_loss_mev'].std()
                    if normalization == 'none':
                        stats_text.append(f"{particle}: {n_steps} шагов")
                        stats_text.append(f" μ = {mean_loss:.3f} МэВ/шаг")
                    elif normalization == 'particles':
                        stats_text.append(f"{particle}: μ = {mean_loss:.3f} МэВ/шаг")
                    elif normalization == 'steps':
                        percentage = (n_steps / total_steps_all * 100) if total_steps_all > 0 else 0
                        stats_text.append(f"{particle}: {percentage:.1f}% шагов")
                        stats_text.append(f" μ = {mean_loss:.3f} МэВ/шаг")
                    else:  # 'density'
                        stats_text.append(f"{particle}: μ = {mean_loss:.3f} МэВ/шаг")
                        stats_text.append(f" σ = {std_loss:.3f} МэВ/шаг")
                if stats_text:
                    stats_cols = 1 if len(stats_text) <= 8 else 2 if len(stats_text) <= 20 else 3
                    stats_block = self._format_stats_columns(stats_text, ncol=stats_cols)
                    stats_artist = ax.text(
                        0.98, 0.98,
                        stats_block,
                        transform=ax.transAxes,
                        fontsize=8.8,
                        va='top', ha='right',
                        bbox=dict(boxstyle='square', pad=0.5, fc='white', alpha=0.22, edgecolor='gray', linewidth=0.5)
                    )
                fig._stats_artist = stats_artist

        # Регулировка отступов
        if handles and legend_loc in ['upper left', 'center left']:
            fig.subplots_adjust(left=0.15, right=0.85)
        else:
            fig.subplots_adjust(left=0.1, right=0.85)
        fig.tight_layout()

        if use_cache and self.cache_mgr:
            self.cache_mgr.save_figure(fig, "energy_loss", cache_key)
            print(f"[CACHE SAVE] Energy_loss сохранён в кэш: {particle_type} {data_hash[:8]}...")
        return fig

    def visualize_interaction_chains(self, df):
        """Визуализация цепочек взаимодействий"""
        fig = Figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        # Построение графа взаимодействий
        import networkx as nx
        G = nx.DiGraph()
        # Добавляем узлы и ребра
        for _, row in df.iterrows():
            G.add_node(row['track_id'],
                       particle=row['particle'],
                       energy=row['kinetic_energy_mev'])
            if row['parent_id'] > 0:
                G.add_edge(row['parent_id'], row['track_id'],
                           process=row.get('process', 'unknown'))
        # Визуализация графа
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax, with_labels=True, node_size=500,
                node_color='lightblue', font_size=10)
        ax.set_title("Цепочки взаимодействий частиц")
        return fig

    def visualize_correct_energy_distributions(self, df):  # Добавляем self
        """Визуализация распределений по ТРЕКАМ, а не по шагам"""
        # if not hasattr(self, 'tracks_info'):
        #     print("Сначала выполните analyze_tracks_correctly!")
        #     return
        fig = Figure(figsize=(12, 8))
        # 1. Распределение средних энергий по трекам
        ax1 = fig.add_subplot(2, 2, 1)
        primary_avg_energies = self.primary_tracks['avg_energy']
        secondary_avg_energies = self.secondary_tracks['avg_energy']
        if len(primary_avg_energies) > 0:
            ax1.hist(primary_avg_energies, bins=30, alpha=0.5, label='Первичные треки',
                     color='blue', density=True)
        if len(secondary_avg_energies) > 0:
            ax1.hist(secondary_avg_energies, bins=30, alpha=0.5, label='Вторичные треки',
                     color='red', density=True)
        ax1.set_xlabel('Средняя энергия трека (МэВ)')
        ax1.set_ylabel('Нормализованная частота')
        ax1.set_title('Распределение средних энергий по трекам')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Количество шагов в треках
        ax2 = fig.add_subplot(2, 2, 2)
        primary_steps = self.primary_tracks['n_steps']
        secondary_steps = self.secondary_tracks['n_steps']
        if len(primary_steps) > 0:
            ax2.hist(primary_steps, bins=30, alpha=0.5, label='Первичные треки',
                     color='blue', density=True, log=True)
        if len(secondary_steps) > 0:
            ax2.hist(secondary_steps, bins=30, alpha=0.5, label='Вторичные треки',
                     color='red', density=True, log=True)
        ax2.set_xlabel('Количество шагов в треке')
        ax2.set_ylabel('Нормализованная частота (лог)')
        ax2.set_title('Распределение количества шагов по трекам')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Распределение типов частиц
        ax3 = fig.add_subplot(2, 2, 3)
        all_particles = pd.concat([
            self.primary_tracks['particle'],
            self.secondary_tracks['particle']
        ])
        particle_counts = all_particles.value_counts().head(10)
        ax3.bar(range(len(particle_counts)), particle_counts.values)
        ax3.set_xticks(range(len(particle_counts)))
        ax3.set_xticklabels(particle_counts.index, rotation=45, ha='right')
        ax3.set_ylabel('Количество треков')
        ax3.set_title('Топ-10 типов частиц по трекам')
        ax3.grid(True, alpha=0.3)

        # 4. Энергия vs parent_id
        ax4 = fig.add_subplot(2, 2, 4)
        # Берем только первые 500 треков для читаемости
        sample_tracks = self.tracks_info.head(500)
        colors = ['blue' if p == 0 else 'red' for p in sample_tracks['parent_id']]
        ax4.scatter(sample_tracks['track_id'], sample_tracks['avg_energy'],
                    c=colors, alpha=0.5, s=20)
        ax4.set_xlabel('Track ID')
        ax4.set_ylabel('Средняя энергия (МэВ)')
        ax4.set_title('Энергия vs Track ID (синие=первичные)')
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def _visualize_additional_plots(self, df, particle_type="частиц"):
        if self.exclude_transport:
            # exclude = ['OutOfWorld', 'Transportation', 'initStep', 'NoProcess', 'CoulombScat']
            exclude = ['OutOfWorld', 'Transportation', 'initStep', 'NoProcess']
            df = df[~df['process'].isin(exclude)]
        if df.empty:
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Нет данных для {particle_type} частиц',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        # non_physical_processes = ['OutOfWorld', 'Transportation', 'initStep', 'NoProcess', 'CoulombScat']
        non_physical_processes = ['OutOfWorld', 'Transportation', 'initStep', 'NoProcess']
        process_counts = df['process'].value_counts()
        filtered_processes = {k: v for k, v in process_counts.items()
                              if k not in non_physical_processes}
        total = sum(filtered_processes.values())
        freqs = {k: v / total for k, v in filtered_processes.items()}
        sorted_processes = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True))
        # sorted_processes = dict(sorted(filtered_processes.items(),
        # key=lambda x: x[1], reverse=True))

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        processes = list(sorted_processes.keys())
        counts = list(sorted_processes.values())
        if len(processes) > 0:
            x_pos = range(len(processes))
            bars = ax.bar(x_pos, counts, color=sns.color_palette("viridis", len(processes)),
                          edgecolor='black', alpha=0.7)
            ax.set_title(f'Частота физических процессов ({particle_type} частиц)',
                         fontweight='bold', fontsize=14)
            ax.set_xlabel('Процесс', fontsize=12)
            ax.set_ylabel('Частота процессов', fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(processes, rotation=45, ha='right', fontsize=11)
            ax.grid(False)
            max_count = max(counts) if counts else 0
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                if height > max_count * 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2, height + max_count * 0.01,
                            f'{count:.2f}', ha='center', va='bottom',
                            fontweight='bold', fontsize=11)
            fig.tight_layout()
            return fig
        else:
            ax.text(0.5, 0.5, f'Нет физических процессов для {particle_type} частиц',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

    from sklearn.mixture import GaussianMixture

    def _adaptive_density_field(self, x, y, xx, yy, weights=None, normalize_for_density=False):
        """
        Быстрая версия поля плотности/энерговыделения.
        - counts: адаптивная kNN плотность
        - dE: кластеризация HDBSCAN/DBSCAN по топ-энергиям + суммирование, лёгкое сглаживание
        - Оптимизация: downsample, cKDTree, уменьшение сетки, векторизация
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter
        from scipy.spatial import cKDTree
        from sklearn.neighbors import NearestNeighbors
        from hdbscan import HDBSCAN

        x = np.asarray(x, float)
        y = np.asarray(y, float)
        n = len(x)
        if weights is None:
            weights = np.ones_like(x)
        weights = np.asarray(weights, float)

        if n < 10:
            # Мало данных --- простая сумма на сетке
            grid = np.column_stack([xx.ravel(), yy.ravel()])
            tree = cKDTree(np.column_stack([x, y]))
            r = max(np.std(x), np.std(y)) * 0.1 if n > 1 else 1e-6
            neighbors = tree.query_ball_point(grid, r=r)
            Z = np.array([np.sum(weights[ind]) if len(ind) > 0 else 0 for ind in neighbors])
            return Z.reshape(xx.shape)

        # --- downsampling больших массивов для ускорения ---
        max_points = 50000
        if n > max_points:
            idx = np.random.choice(n, max_points, replace=False)
            x_sample = x[idx]
            y_sample = y[idx]
            w_sample = weights[idx]
        else:
            x_sample, y_sample, w_sample = x, y, weights

        points = np.column_stack([x_sample, y_sample])
        grid = np.column_stack([xx.ravel(), yy.ravel()])

        if not normalize_for_density:
            # --- dE режим: как counts, только энергия вместо количества ---
            # k подбираем похоже на counts
            k = max(15, min(100, int(np.sqrt(len(points)) * 1.5)))
            nn = NearestNeighbors(n_neighbors=k).fit(points)
            dists, indices = nn.kneighbors(grid)
            rk = dists[:, -1]
            rk[rk == 0] = 1e-12
            # сумма энергии в окрестности (MeV или keV)
            energy_sums = np.sum(w_sample[indices], axis=1)
            # плотность энергии (MeV/mm^2)
            Z = energy_sums / (np.pi * rk ** 2 + 1e-12)
            Z = Z.reshape(xx.shape)
            # сглаживание (как "разрешение"), можно 0.6..1.2
            Z = gaussian_filter(Z, sigma=1.0)
            return Z
        else:
            # --- counts режим с адаптивной плотностью ---
            k = max(15, min(100, int(np.sqrt(len(points)) * 1.5)))
            nn = NearestNeighbors(n_neighbors=k).fit(points)
            dists, indices = nn.kneighbors(grid)
            rk = dists[:, -1]
            weighted_sums = np.sum(w_sample[indices], axis=1)
            Z = weighted_sums / (np.pi * rk ** 2 + 1e-12)
            Z = Z.reshape(xx.shape)
            # Лог нормализация для визуализации
            zmax = Z.max()
            if zmax > 0:
                Z /= zmax  # Нормализация без логарифмирования

            Z = gaussian_filter(Z, sigma=1.0)
            return Z

    def _visualize_heatmap(self, df, particle_type="частиц", heatmap_mode="counts",
                           energy_column="process_energy_loss_mev",
                           unit="MeV", use_cache=True):
        if df.empty or not all(col in df.columns for col in ['x_mm', 'y_mm', 'z_mm']):
            return self._create_empty_plot(f'Нет координатных данных для {particle_type} частиц')

        # Кэширование
        if use_cache and self.cache_mgr:
            cache_key = {
                "plot_type": "heatmap",
                "particle_type": particle_type,
                "heatmap_mode": heatmap_mode,
                "unit": unit,
                "energy_column": energy_column,
                "data_hash": hashlib.md5(
                    pd.util.hash_pandas_object(df[['x_mm', 'y_mm', 'z_mm', energy_column] if heatmap_mode == "dE" else [
                        'x_mm', 'y_mm', 'z_mm']]).values.tobytes()
                ).hexdigest(),
                "material_limits": str(self.material_dimensions.get_limits())
            }
            cached_fig = self.cache_mgr.load_figure("heatmap", cache_key)
            if cached_fig is not None:
                print(f"[CACHE HIT] Heatmap загружен: {particle_type} {cache_key['data_hash'][:8]}...")
                return cached_fig

        df_clean = df.copy()
        limits = self.material_dimensions.get_limits()

        fig = Figure(figsize=(13.5, 4.8), dpi=120)
        projections = [
            ('x_mm', 'y_mm', 'X / Y', 'X (мм)', 'Y (мм)'),
            ('x_mm', 'z_mm', 'X / Z', 'X (мм)', 'Z (мм)'),
            ('y_mm', 'z_mm', 'Y / Z', 'Y (мм)', 'Z (мм)')
        ]

        plot_data = []
        for x_col, y_col, title, xlabel, ylabel in projections:
            x = df_clean[x_col].values
            y = df_clean[y_col].values

            if len(x) < 3 or len(y) < 3:
                continue

            x_min, x_max = limits[x_col[0]]
            y_min, y_max = limits[y_col[0]]

            dx = (x_max - x_min) * 0.025
            dy = (y_max - y_min) * 0.025

            # xi = np.linspace(x_min - dx, x_max + dx, 140)
            # yi = np.linspace(y_min - dy, y_max + dy, 140)
            # xx, yy = np.meshgrid(xi, yi)
            n_bins = 140
            x_edges = np.linspace(x_min - dx, x_max + dx, n_bins + 1)
            y_edges = np.linspace(y_min - dy, y_max + dy, n_bins + 1)
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            xx, yy = np.meshgrid(x_centers, y_centers)

            cmap = None
            if heatmap_mode == "counts":
                weights = np.ones(len(df_clean))
                cbar_label = "Плотность шагов"
                cmap = 'magma'
                zi = self._adaptive_density_field(x, y, xx, yy, weights=weights, normalize_for_density=True)
            elif heatmap_mode == "dE":
                if energy_column not in df_clean.columns:
                    continue
                cmap = 'hot'
                weights = df_clean[energy_column].values
                if unit.lower() == "kev":
                    weights = weights * 1e3
                    cbar_label = "Плотность энерговыделения (keV / mm²)"
                else:
                    cbar_label = "Плотность энерговыделения (MeV / mm²)"
                # ✅ НОВОЕ: строим поле через твой адаптивный метод (кластеры + овалы)
                zi = self._adaptive_density_field(x, y, xx, yy, weights=weights, normalize_for_density=False)

            if not np.isfinite(zi).all():
                print("WARN: zi has NaN/inf")
            if zi is None:
                continue

            plot_data.append({
                'xi': x_centers, 'yi': y_centers, 'zi': zi,
                'x_edges': x_edges, 'y_edges': y_edges,
                'title': title, 'xlabel': xlabel, 'ylabel': ylabel,
                'xlim': (x_min, x_max), 'ylim': (y_min, y_max),
                'cbar_label': cbar_label,
                'heatmap_mode': heatmap_mode
            })

        if not plot_data:
            return self._create_empty_plot('Не удалось построить проекции')

        n_plots = len(plot_data)
        for i, data in enumerate(plot_data):
            ax = fig.add_subplot(1, n_plots, i + 1)
            zi = data['zi']

            contour = ax.contourf(data['xi'], data['yi'], zi, levels=50, cmap=cmap)
            ax.contour(data['xi'], data['yi'], zi, levels=35, colors='black', linewidths=0.3, alpha=0.35)

            ax.set_title(data['title'], fontsize=10.5, pad=6)
            ax.set_xlabel(data['xlabel'], fontsize=9.5)
            ax.set_ylabel(data['ylabel'], fontsize=9.5)
            ax.tick_params(labelsize=8.5)
            ax.set_aspect('equal')
            ax.grid(False)
            ax.set_xlim(data['xlim'])
            ax.set_ylim(data['ylim'])

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            cbar = fig.colorbar(contour, cax=cax, orientation='vertical')
            cbar.set_label(data['cbar_label'], fontsize=8.5)
            cbar.ax.tick_params(labelsize=7.5)

        # Заголовок
        if heatmap_mode == "counts":
            title_text = f'Пространственное распределение {particle_type} частиц'
        else:
            total_dE = df_clean[energy_column].sum()
            if unit.lower() == "kev":
                total_dE *= 1e3
            title_text = f'Суммарное энерговыделение {particle_type} частиц (ΣdE = {total_dE:.3f} {unit})'

        fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.96)
        fig.subplots_adjust(left=0.07, right=0.93, top=0.90, bottom=0.15, wspace=0.42)
        fig._no_tight_layout = True

        if use_cache and self.cache_mgr:
            self.cache_mgr.save_figure(fig, "heatmap", cache_key)
            print(f"[CACHE SAVE] Heatmap сохранён: {particle_type} {cache_key['data_hash'][:8]}...")
        return fig

    def _energy_field_gaussian_clusters(self, points, weights, grid, labels, min_pts=8, reg=1e-6):
        """
        points: (M,2)
        weights: (M,)
        grid: (G,2)
        labels: (M,) cluster labels, -1 = noise
        return: Z (G,) energy density, integrates ~ sum(weights of clustered points)
        """
        import numpy as np
        Z = np.zeros(len(grid), dtype=float)
        bandwidth = 1.1
        good_clusters = [c for c in np.unique(labels) if c >= 0]
        if not good_clusters:
            return Z

        gx = grid[:, 0][:, None]  # (G,1)
        gy = grid[:, 1][:, None]

        for c in good_clusters:
            mask = labels == c
            if mask.sum() < min_pts:
                continue
            X = points[mask]
            w = weights[mask]
            W = w.sum()
            if W <= 0:
                continue

            # weighted mean
            mu = np.average(X, axis=0, weights=w)  # (2,)
            # weighted covariance (2x2)
            Xm = X - mu
            cov = (w[:, None] * Xm).T @ Xm / (W + 1e-12)
            cov *= (bandwidth ** 2)
            # regularize
            cov[0, 0] += reg
            cov[1, 1] += reg
            # inverse + det
            det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
            if det <= 0:
                continue
            inv = (1.0 / det) * np.array([[cov[1, 1], -cov[0, 1]],
                                          [-cov[1, 0], cov[0, 0]]], dtype=float)

            dx = grid[:, 0] - mu[0]
            dy = grid[:, 1] - mu[1]
            q = inv[0, 0] * dx * dx + (inv[0, 1] + inv[1, 0]) * dx * dy + inv[1, 1] * dy * dy
            norm = 1.0 / (2.0 * np.pi * np.sqrt(det) + 1e-24)  # gaussian pdf normalization
            tail_cut = 4.0  # в сигмах; 3-5 обычно норм
            q_max = tail_cut ** 2  # потому что q ~ r^2 в сигмах
            e = np.exp(-0.5 * q)
            e[q > q_max] = 0.0  # <-- обрезаем дальние хвосты
            Z += W * norm * e
        return Z

    from scipy.interpolate import splprep, splev

    def smooth_track(self, x, y, z, n_points=150, smooth_factor=None):
        """
        Параметрическое 3D сглаживание.

        Parameters
        ----------
        x, y, z : arrays
            Координаты трека
        n_points : int
            Количество точек после сглаживания
        smooth_factor : float or None
            Параметр сглаживания. None → автоматический, small N → меньше сглаживания
        """
        # Если мало точек, делаем минимальное сглаживание
        if smooth_factor is None:
            smooth_factor = max(0.0, len(x) * 0.05)
        try:
            tck, u = splprep([x, y, z], s=smooth_factor, k=min(3, len(x) - 1))
            u_fine = np.linspace(0, 1, n_points)
            x_smooth, y_smooth, z_smooth = splev(u_fine, tck)
            return x_smooth, y_smooth, z_smooth
        except Exception:
            # fallback --- оригинальные точки
            return x, y, z

    # ===== НОВЫЙ МЕТОД: Разрыв трека на непрерывные сегменты =====
    def _split_track_into_segments(self, steps, x, y, z=None):
        """
        Разбивает трек на непрерывные сегменты.

        Args:
            steps: массив номеров шагов
            x, y, z: массивы координат
            z: опционально, для 3D

        Returns:
            list: список кортежей (start_idx, end_idx) для каждого сегмента
        """
        if steps.size <= 1:
            return [(0, steps.size)]

        # Разрыв по номерам шагов (если пропущен номер - значит разрыв)
        step_breaks = np.where(np.diff(steps) > 1)[0] + 1

        # Разрыв по пространству (слишком большой скачок)
        dx = np.diff(x)
        dy = np.diff(y)

        if z is not None:
            dz = np.diff(z)
            dr = np.sqrt(dx * dx + dy * dy + dz * dz)
        else:
            dr = np.sqrt(dx * dx + dy * dy)

        # Медианное расстояние между точками
        valid_dr = dr[np.isfinite(dr)]
        if valid_dr.size > 0:
            med_dr = np.median(valid_dr)
            # Порог для разрыва (в 10 раз больше медианы, но не менее 0.5 мм)
            threshold = max(0.5, 10.0 * med_dr)
        else:
            threshold = 1.0

        space_breaks = np.where(dr > threshold)[0] + 1

        # Объединяем все разрывы
        all_breaks = np.unique(np.concatenate([step_breaks, space_breaks]))
        all_breaks = all_breaks[(all_breaks > 0) & (all_breaks < steps.size)]

        if all_breaks.size == 0:
            return [(0, steps.size)]

        # Формируем сегменты
        segments = []
        start = 0
        for b in all_breaks:
            if b > start:
                segments.append((start, b))
            start = b
        if start < steps.size:
            segments.append((start, steps.size))

        return segments

    # ===== КОНЕЦ НОВОГО МЕТОДА =====

    # ===== НОВЫЙ МЕТОД: Расчет дозовой карты для проекции =====
    def _calculate_dose_map(self, df, proj_x_col, proj_y_col, grid_size=140, unit="Gy"):
        """
        Вычисляет карту дозы для заданной проекции.

        Args:
            df: DataFrame с данными
            proj_x_col: колонка для оси X проекции ('x_mm', 'y_mm')
            proj_y_col: колонка для оси Y проекции ('y_mm', 'z_mm')
            grid_size: размер сетки
            unit: единицы измерения ('Gy' или 'rad')

        Returns:
            tuple: (dose_map, x_centers, y_centers)
        """
        if df.empty:
            return None, None, None

        # Определяем границы
        x_min = df[proj_x_col].min()
        x_max = df[proj_x_col].max()
        y_min = df[proj_y_col].min()
        y_max = df[proj_y_col].max()

        # Добавляем небольшой отступ
        x_pad = (x_max - x_min) * 0.02
        y_pad = (y_max - y_min) * 0.02
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        # Создаем сетку
        x_edges = np.linspace(x_min, x_max, grid_size + 1)
        y_edges = np.linspace(y_min, y_max, grid_size + 1)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        # Выбираем колонку с дозой
        dose_column = 'dose_rad' if unit.lower() == 'rad' else 'dose_gray'

        # Создаем 2D гистограмму с весами = доза
        H, _, _ = np.histogram2d(
            df[proj_x_col].values,
            df[proj_y_col].values,
            bins=[x_edges, y_edges],
            weights=df[dose_column].values
        )

        # Транспонируем для правильной ориентации
        dose_map = H.T

        # Применяем логарифмическое масштабирование (добавляем маленькое число чтобы избежать log(0))
        # dose_map = np.log10(dose_map + 1e-15)

        return dose_map, x_centers, y_centers

    # ===== КОНЕЦ НОВОГО МЕТОДА =====

    # ===== НОВЫЙ МЕТОД: Визуализация дозовых карт =====
    def _visualize_dose_map(self, df, particle_type="частиц", unit="Gy", per_layer=False, use_cache=True):
        """
        Визуализация дозовых карт в трех проекциях.

        Args:
            df: DataFrame с данными
            particle_type: тип частиц для заголовка
            unit: единицы измерения ('Gy' или 'rad')
            per_layer: если True, показывает отдельные карты для каждого слоя
            use_cache: использовать кэширование
        """
        if df.empty:
            return self._create_empty_plot(f'Нет данных для {particle_type} частиц')

        # Проверка наличия колонок с дозой
        if 'dose_gray' not in df.columns:
            return self._create_empty_plot('Нет данных о дозе. Возможно, не удалось определить слои материала.')

        # Кэширование
        if use_cache and self.cache_mgr:
            cache_key = {
                "plot_type": "dose_map",
                "particle_type": particle_type,
                "unit": unit,
                "per_layer": per_layer,
                "data_hash": hashlib.md5(
                    pd.util.hash_pandas_object(df[['x_mm', 'y_mm', 'z_mm', 'dose_gray']]).values.tobytes()
                ).hexdigest()[:16],
            }
            cached_fig = self.cache_mgr.load_figure("dose_map", cache_key)
            if cached_fig:
                print(f"[CACHE HIT] Dose map: {particle_type}")
                return cached_fig

        unit_display = "Грей" if unit == "Gy" else "рад"

        if per_layer and len(self.layers) > 1:
            # Для многослойной структуры - отдельные карты для каждого слоя
            active_layers = [l for l in self.layers if not df[df['z_mm'].between(l.z_min_mm, l.z_max_mm)].empty]
            n_layers = len(active_layers)

            if n_layers == 0:
                return self._create_empty_plot('Нет данных для отображения по слоям')

            fig = Figure(figsize=(14, 4 * n_layers))

            plot_idx = 1
            for layer_idx, layer in enumerate(active_layers):
                layer_df = df[df['z_mm'].between(layer.z_min_mm, layer.z_max_mm)]
                if layer_df.empty:
                    continue

                # Три проекции для каждого слоя
                projections = [
                    ('x_mm', 'y_mm', f'{layer.name}: X/Y проекция'),
                    ('x_mm', 'z_mm', f'{layer.name}: X/Z проекция'),
                    ('y_mm', 'z_mm', f'{layer.name}: Y/Z проекция')
                ]

                for proj_x, proj_y, title in projections:
                    ax = fig.add_subplot(n_layers, 3, plot_idx)
                    plot_idx += 1

                    # Вычисляем карту дозы
                    dose_map, x_centers, y_centers = self._calculate_dose_map(
                        layer_df, proj_x, proj_y, unit=unit
                    )

                    if dose_map is None:
                        ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
                        continue

                    # Визуализация
                    im = ax.contourf(x_centers, y_centers, dose_map, levels=50, cmap='hot')
                    ax.contour(x_centers, y_centers, dose_map, levels=20, colors='black',
                               linewidths=0.3, alpha=0.3)

                    # Добавляем информацию о слое
                    dose_info = f"ρ={layer.density_g_cm3:.2f} г/см³, h={layer.thickness_mm:.1f} мм"
                    ax.text(0.02, 0.98, dose_info, transform=ax.transAxes, fontsize=8,
                            va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))

                    ax.set_title(title, fontsize=10)
                    ax.set_xlabel(f'{proj_x[0].upper()} (мм)')
                    ax.set_ylabel(f'{proj_y[0].upper()} (мм)')
                    ax.set_aspect('equal')

                    # Цветовая шкала
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = fig.colorbar(im, cax=cax)
                    cbar.set_label(f'log10(Доза, {unit_display})')

            fig.suptitle(f'Дозовые карты по слоям ({particle_type} частицы)',
                         fontsize=14, fontweight='bold')

        else:
            # Общая карта для всех слоев вместе
            fig = Figure(figsize=(13.5, 4.8), dpi=120)

            projections = [
                ('x_mm', 'y_mm', 'X / Y проекция', 'X (мм)', 'Y (мм)'),
                ('x_mm', 'z_mm', 'X / Z проекция', 'X (мм)', 'Z (мм)'),
                ('y_mm', 'z_mm', 'Y / Z проекция', 'Y (мм)', 'Z (мм)')
            ]

            for i, (proj_x, proj_y, title, xlabel, ylabel) in enumerate(projections):
                ax = fig.add_subplot(1, 3, i + 1)

                # Вычисляем карту дозы
                dose_map, x_centers, y_centers = self._calculate_dose_map(
                    df, proj_x, proj_y, unit=unit
                )

                if dose_map is None:
                    ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
                    continue

                # Визуализация
                im = ax.contourf(x_centers, y_centers, dose_map, levels=50, cmap='hot')
                ax.contour(x_centers, y_centers, dose_map, levels=20, colors='black',
                           linewidths=0.3, alpha=0.3)

                ax.set_title(title, fontsize=10.5)
                ax.set_xlabel(xlabel, fontsize=9.5)
                ax.set_ylabel(ylabel, fontsize=9.5)
                ax.set_aspect('equal')

                # Цветовая шкала
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label(f'Доза, {unit_display}')


            # Добавляем информацию о слоях
            if self.layers:
                layer_info = "Слои: " + ", ".join([f"{l.name}({l.material})" for l in self.layers])
                fig.text(0.5, 0.92, layer_info, ha='center', fontsize=9,
                         bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.7))

            fig.suptitle(f'Дозовые карты (суммарно, {particle_type} частицы)',
                         fontsize=14, fontweight='bold', y=0.98)

        fig.tight_layout()

        if use_cache and self.cache_mgr:
            self.cache_mgr.save_figure(fig, "dose_map", cache_key)
            print(f"[CACHE SAVE] Dose map: {particle_type}")

        return fig

    # ===== КОНЕЦ НОВОГО МЕТОДА =====

    def _visualize_2d_trajectory_projections(
            self,
            df,
            particle_type="частиц",
            selected_particles=None,
            max_tracks_to_show=80,
            max_points_per_track=3000,
            use_cache=True
    ):
        import numpy as np
        import pandas as pd
        import hashlib
        from matplotlib.figure import Figure

        if df is None or df.empty:
            return self._create_empty_plot(f'Нет данных для 2D проекций траекторий {particle_type} частиц')

        need_cols = {"x_mm", "y_mm", "z_mm", "track_id", "step_number"}
        if not need_cols.issubset(df.columns):
            return self._create_empty_plot(f'Нет данных для 2D проекций траекторий {particle_type} частиц')

        if use_cache and getattr(self, "cache_mgr", None):
            cache_key = {
                "plot_type": "2d_traj_projections",
                "particle_type": particle_type,
                "selected_particles": tuple(selected_particles) if selected_particles else None,
                "data_hash": hashlib.md5(
                    pd.util.hash_pandas_object(
                        df[["track_id", "particle", "step_number", "x_mm", "y_mm", "z_mm"]]
                        if "particle" in df.columns
                        else df[["track_id", "step_number", "x_mm", "y_mm", "z_mm"]]
                    ).values.tobytes()
                ).hexdigest()[:16],
                "max_tracks_to_show": int(max_tracks_to_show),
                "max_points_per_track": int(max_points_per_track),
            }
            cached_fig = self.cache_mgr.load_figure("2d_traj_projections", cache_key)
            if cached_fig is not None:
                return cached_fig

        df_filtered = df.copy()

        if selected_particles and "particle" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["particle"].isin(selected_particles)]
        if df_filtered.empty:
            return self._create_empty_plot(f'Нет данных для выбранных типов частиц: {selected_particles}')

        is_primary_mode = (particle_type == "первичных")
        if "is_primary" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["is_primary"]] if is_primary_mode else df_filtered[
                ~df_filtered["is_primary"]]
        if df_filtered.empty:
            return self._create_empty_plot(f'Нет данных для {particle_type} частиц')

        track_key_cols = [c for c in ("thread", "event_id", "track_id") if c in df_filtered.columns]
        if not track_key_cols:
            track_key_cols = ["track_id"]

        track_sizes = df_filtered.groupby(track_key_cols).size().sort_values(ascending=False)
        if track_sizes.empty:
            return self._create_empty_plot(f'Нет треков для отображения ({particle_type})')

        top_keys = track_sizes.index[:max_tracks_to_show]

        if len(track_key_cols) == 1:
            df_draw = df_filtered[df_filtered[track_key_cols[0]].isin(top_keys)]
        else:
            df_draw = df_filtered.set_index(track_key_cols).loc[top_keys].reset_index()

        if df_draw.empty:
            return self._create_empty_plot(f'Нет треков для отображения ({particle_type})')

        try:
            limits = self.material_dimensions.get_limits()
        except Exception:
            limits = None

        fig = Figure(figsize=(13.5, 4.8), dpi=120)
        projections = [
            ("x_mm", "y_mm", "X / Y", "X (мм)", "Y (мм)"),
            ("x_mm", "z_mm", "X / Z", "X (мм)", "Z (мм)"),
            ("y_mm", "z_mm", "Y / Z", "Y (мм)", "Z (мм)"),
        ]

        axes = []
        for i, (_, _, title, xl, yl) in enumerate(projections):
            ax = fig.add_subplot(1, 3, i + 1)
            ax.set_title(title, fontsize=10.5, pad=6)
            ax.set_xlabel(xl, fontsize=9.5)
            ax.set_ylabel(yl, fontsize=9.5)
            ax.tick_params(labelsize=8.5)
            ax.set_aspect("equal")
            ax.grid(False)
            axes.append(ax)

        n_tracks = int(len(top_keys))
        built_segments = 0
        drawn_points = 0

        for _, tdf in df_draw.groupby(track_key_cols, sort=False):
            tdf = tdf.sort_values("step_number", kind="mergesort")

            steps = tdf["step_number"].to_numpy()
            if steps.size == 0:
                continue

            if "particle" in tdf.columns:
                particle = str(tdf["particle"].iloc[0])
            else:
                particle = "unknown"

            tid = int(tdf["track_id"].iloc[0]) if "track_id" in tdf.columns else 0
            color, _ = self._get_particle_color(particle, is_primary_mode, tid, n_tracks)

            if steps.size == 1:
                x = tdf["x_mm"].to_numpy()
                y = tdf["y_mm"].to_numpy()
                z = tdf["z_mm"].to_numpy()

                axes[0].scatter(x[0], y[0], color=color, s=18, alpha=0.95, zorder=6, edgecolors="none")
                axes[1].scatter(x[0], z[0], color=color, s=18, alpha=0.95, zorder=6, edgecolors="none")
                axes[2].scatter(y[0], z[0], color=color, s=18, alpha=0.95, zorder=6, edgecolors="none")
                drawn_points += 1
                continue

            # ===== ИЗМЕНЕННЫЙ КОД: Используем новый метод разрыва =====
            x = tdf["x_mm"].to_numpy()
            y = tdf["y_mm"].to_numpy()
            z = tdf["z_mm"].to_numpy()

            # Разбиваем трек на сегменты
            segments = self._split_track_into_segments(steps, x, y, z)

            for start_idx, end_idx in segments:
                n_points = end_idx - start_idx
                if n_points == 0:
                    continue
                elif n_points == 1:
                    # Одна точка
                    axes[0].scatter(x[start_idx], y[start_idx], color=color, s=18, alpha=0.95,
                                    zorder=6, edgecolors="none")
                    axes[1].scatter(x[start_idx], z[start_idx], color=color, s=18, alpha=0.95,
                                    zorder=6, edgecolors="none")
                    axes[2].scatter(y[start_idx], z[start_idx], color=color, s=18, alpha=0.95,
                                    zorder=6, edgecolors="none")
                    drawn_points += 1
                else:
                    # Сегмент из нескольких точек
                    x_seg = x[start_idx:end_idx]
                    y_seg = y[start_idx:end_idx]
                    z_seg = z[start_idx:end_idx]

                    if len(x_seg) > max_points_per_track:
                        step = max(1, len(x_seg) // max_points_per_track)
                        x_seg = x_seg[::step]
                        y_seg = y_seg[::step]
                        z_seg = z_seg[::step]

                    # Рисуем линию
                    axes[0].plot(x_seg, y_seg, color=color, alpha=0.7, linewidth=1.0, zorder=5)
                    axes[1].plot(x_seg, z_seg, color=color, alpha=0.7, linewidth=1.0, zorder=5)
                    axes[2].plot(y_seg, z_seg, color=color, alpha=0.7, linewidth=1.0, zorder=5)

                    # Отмечаем начало (кружок) и конец (квадрат)
                    axes[0].scatter(x_seg[0], y_seg[0], color=color, s=18, alpha=0.95,
                                    zorder=6, edgecolors="none")
                    axes[0].scatter(x_seg[-1], y_seg[-1], color=color, s=18, alpha=0.95,
                                    zorder=6, marker="s", edgecolors="none")

                    axes[1].scatter(x_seg[0], z_seg[0], color=color, s=18, alpha=0.95,
                                    zorder=6, edgecolors="none")
                    axes[1].scatter(x_seg[-1], z_seg[-1], color=color, s=18, alpha=0.95,
                                    zorder=6, marker="s", edgecolors="none")

                    axes[2].scatter(y_seg[0], z_seg[0], color=color, s=18, alpha=0.95,
                                    zorder=6, edgecolors="none")
                    axes[2].scatter(y_seg[-1], z_seg[-1], color=color, s=18, alpha=0.95,
                                    zorder=6, marker="s", edgecolors="none")

                    built_segments += 1
                    drawn_points += int(len(x_seg))
            # ===== КОНЕЦ ИЗМЕНЕННОГО КОДА =====

        if built_segments == 0 and drawn_points == 0:
            return self._create_empty_plot(f'Не удалось построить траектории {particle_type} частиц')

        def _bounds(a):
            a = np.asarray(a, float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return (-1.0, 1.0)
            if a.size < 3:
                lo = float(np.min(a))
                hi = float(np.max(a))
                if lo == hi:
                    m = 1.0
                    return (lo - m, hi + m)
                m = (hi - lo) * 0.05
                return (lo - m, hi + m)
            lo = float(np.percentile(a, 2))
            hi = float(np.percentile(a, 98))
            if lo == hi:
                m = 1.0
                return (lo - m, hi + m)
            m = (hi - lo) * 0.05
            return (lo - m, hi + m)

        def _apply_limits(ax, x_key, y_key, xdata, ydata):
            if limits is not None:
                try:
                    ax.set_xlim(*limits[x_key])
                    ax.set_ylim(*limits[y_key])
                    return
                except Exception:
                    pass
            ax.set_xlim(*_bounds(xdata))
            ax.set_ylim(*_bounds(ydata))

        _apply_limits(axes[0], "x", "y", df_draw["x_mm"].to_numpy(), df_draw["y_mm"].to_numpy())
        _apply_limits(axes[1], "x", "z", df_draw["x_mm"].to_numpy(), df_draw["z_mm"].to_numpy())
        _apply_limits(axes[2], "y", "z", df_draw["y_mm"].to_numpy(), df_draw["z_mm"].to_numpy())

        fig.suptitle(
            f'Проекции траекторий {particle_type} частиц (сегменты: {built_segments}, точек: {drawn_points})',
            fontsize=13.5,
            fontweight="bold",
            y=0.96
        )

        fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, wspace=0.30)
        fig._no_tight_layout = True

        if use_cache and getattr(self, "cache_mgr", None):
            self.cache_mgr.save_figure(fig, "2d_traj_projections", cache_key)

        return fig

    def _visualize_3d_trajectories(self, df, particle_type="частиц", selected_particles=None, use_cache=True):
        if df.empty or not all(c in df.columns for c in ("x_mm", "y_mm", "z_mm")):
            return self._create_empty_plot(f"Нет данных для построения 3D траекторий {particle_type} частиц")

        if use_cache and self.cache_mgr:
            cols = [c for c in
                    ("thread", "event_id", "track_id", "parent_id", "particle", "is_primary", "step_number", "x_mm",
                     "y_mm", "z_mm", "kinetic_energy_mev") if c in df.columns]
            cache_key = {
                "plot_type": "3d_trajectories_v2",
                "particle_type": particle_type,
                "selected_particles": tuple(selected_particles) if selected_particles else None,
                "data_hash":
                    hashlib.md5(pd.util.hash_pandas_object(df[cols]).values.tobytes()).hexdigest()[:16],
            }
            cached_fig = self.cache_mgr.load_figure("3d_trajectories", cache_key)
            if cached_fig is not None:
                return cached_fig

        d = df.copy()
        if selected_particles and "particle" in d.columns:
            d = d[d["particle"].isin(selected_particles)]
        if d.empty:
            return self._create_empty_plot(f"Нет данных для выбранных типов частиц: {selected_particles}")

        is_primary_mode = (particle_type == "первичных")
        if "is_primary" in d.columns:
            d = d[d["is_primary"]] if is_primary_mode else d[~d["is_primary"]]
        if d.empty:
            return self._create_empty_plot(f"Нет данных для {particle_type} частиц")

        if "step_number" not in d.columns:
            d = d.assign(step_number=np.arange(len(d), dtype=np.int64))

        key_cols = [c for c in ("thread", "event_id", "track_id", "parent_id") if c in d.columns]
        if not key_cols:
            key_cols = ["track_id"]

        sizes = d.groupby(key_cols, sort=False).size().sort_values(ascending=False)
        max_tracks_to_show = 30
        top_keys = sizes.index[:max_tracks_to_show]

        if len(key_cols) == 1:
            draw = d[d[key_cols[0]].isin(top_keys)]
        else:
            draw = d.set_index(key_cols).loc[top_keys].reset_index()

        if draw.empty:
            return self._create_empty_plot(f"Нет треков для отображения ({particle_type})")

        fig = Figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")

        built_segments = 0
        total_points = 0
        particle_stats = {}

        n_tracks = int(len(top_keys))

        for key, tdf in draw.groupby(key_cols, sort=False):
            tdf = tdf.sort_values("step_number", kind="mergesort")

            x0 = tdf["x_mm"].to_numpy(dtype=float)
            y0 = tdf["y_mm"].to_numpy(dtype=float)
            z0 = tdf["z_mm"].to_numpy(dtype=float)
            s0 = tdf["step_number"].to_numpy(dtype=np.int64)

            m = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(z0) & np.isfinite(s0)
            x0, y0, z0, s0 = x0[m], y0[m], z0[m], s0[m]

            if x0.size == 0:
                continue

            particle = tdf["particle"].iloc[0] if "particle" in tdf.columns and len(tdf) else "unknown"
            tid = int(tdf["track_id"].iloc[0]) if "track_id" in tdf.columns and len(tdf) else 0
            color, _ = self._get_particle_color(particle, is_primary_mode, tid, n_tracks)

            if particle not in particle_stats:
                particle_stats[particle] = {"tracks": 0, "points": 0, "energy_sum": 0.0, "energy_count": 0}
            particle_stats[particle]["tracks"] += 1
            particle_stats[particle]["points"] += int(x0.size)

            if "kinetic_energy_mev" in tdf.columns:
                ke = tdf["kinetic_energy_mev"].to_numpy(dtype=float)
                ke = ke[np.isfinite(ke)]
                if ke.size:
                    particle_stats[particle]["energy_sum"] += float(np.mean(ke))
                    particle_stats[particle]["energy_count"] += 1

            # ===== ИЗМЕНЕННЫЙ КОД: Используем новый метод разрыва =====
            segments = self._split_track_into_segments(s0, x0, y0, z0)

            for a, b in segments:
                n = b - a
                if n <= 0:
                    continue
                x = x0[a:b]
                y = y0[a:b]
                z = z0[a:b]

                if n == 1:
                    ax.scatter(x[0], y[0], z[0], color=color, s=22, alpha=0.9, zorder=10, edgecolors="none")
                    built_segments += 1
                    total_points += 1
                    continue

                ax.plot(x, y, z, color=color, alpha=0.75, linewidth=1.3, zorder=5)
                ax.scatter(x[0], y[0], z[0], color=color, s=26, alpha=0.9, zorder=10, edgecolors="none")
                ax.scatter(x[-1], y[-1], z[-1], color=color, s=26, alpha=0.9, zorder=10, marker="s", edgecolors="none")

                built_segments += 1
                total_points += int(n)
            # ===== КОНЕЦ ИЗМЕНЕННОГО КОДА =====

        if built_segments == 0:
            return self._create_empty_plot(f"Не удалось построить траектории {particle_type} частиц")

        ax.set_title(f"3D траектории {particle_type} частиц", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("X (мм)", fontsize=11, labelpad=10)
        ax.set_ylabel("Y (мм)", fontsize=11, labelpad=10)
        ax.set_zlabel("Z (мм)", fontsize=11, labelpad=10)
        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(-5.0, 5.0)
        ax.set_zlim(-5.0, 5.0)
        ax.set_box_aspect((1, 1, 1))
        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_alpha(0.03)
        ax.yaxis.pane.set_alpha(0.03)
        ax.zaxis.pane.set_alpha(0.03)
        ax.view_init(elev=20, azim=135)

        if len(particle_stats) <= 8:
            from matplotlib.lines import Line2D
            handles = []
            for p, st in sorted(particle_stats.items(), key=lambda kv: kv[1]["tracks"], reverse=True):
                c, name = self._get_particle_color(p, is_primary_mode, 0, 1)
                label = name
                if st["energy_count"] > 0:
                    label += f", Eср={st['energy_sum'] / st['energy_count']:.2f} МэВ"
                handles.append(Line2D([0], [0], color=c, linewidth=1.5, label=label))
            if handles:
                leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.98),
                                fontsize=9, title_fontsize=10, framealpha=0.3, fancybox=False,
                                edgecolor=(0, 0, 0, 0.3), facecolor="white", borderpad=0.8,
                                labelspacing=0.7, handlelength=2.0, borderaxespad=0.5)
                leg.get_frame().set_linewidth(0.8)
                leg.get_frame().set_boxstyle("square", pad=0.3)

        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

        if use_cache and self.cache_mgr:
            self.cache_mgr.save_figure(fig, "3d_trajectories", cache_key)

        return fig

    def _visualize_dE_distribution(
            self,
            df,
            particle_type="частиц",
            show_kde=True,
            show_stats=True,
            min_y_log=0.8,
            use_cache=True,
            normalization='none',
            selected_particles=None,
            xlim=None,
            ylim=None
    ):
        if df.empty or 'process_energy_loss_mev' not in df.columns:
            return self._create_empty_plot(f'Нет данных dE для {particle_type} частиц')

        df_filtered = df.copy()
        df_filtered = df_filtered[df_filtered['process_energy_loss_mev'] >= 0].copy()
        df_filtered['dE'] = df_filtered['process_energy_loss_mev']
        df_filtered_kde = df_filtered[df_filtered['dE'] > 0].copy()
        df_filtered_kde['dE'] = df_filtered_kde['process_energy_loss_mev']

        if selected_particles and len(selected_particles) > 0:
            df_filtered = df_filtered[df_filtered['particle'].isin(selected_particles)].copy()
            df_filtered_kde = df_filtered_kde[df_filtered_kde['particle'].isin(selected_particles)].copy()

        if df_filtered.empty:
            return self._create_empty_plot(f'Нет данных dE >= 0 для {particle_type} частиц')
        if df_filtered_kde.empty:
            return self._create_empty_plot(f'Нет данных dE > 0 для {particle_type} частиц')

        # Кэширование
        if use_cache and hasattr(self, 'cache_mgr') and self.cache_mgr:
            data_hash = hashlib.md5(
                pd.util.hash_pandas_object(df_filtered[['particle', 'dE', 'track_id']]).values.tobytes()
            ).hexdigest()
            cache_key = {
                "plot_type": "dE",
                "particle_type": particle_type,
                "show_kde": show_kde,
                "show_stats": show_stats,
                "min_y_log": min_y_log,
                "normalization": normalization,
                "selected_particles": tuple(selected_particles) if selected_particles else None,
                "data_hash": data_hash[:16]
            }
            cached_fig = self.cache_mgr.load_figure("dE", cache_key)
            if cached_fig is not None:
                print(f"[CACHE HIT] dE: {particle_type} {data_hash[:8]}...")
                return cached_fig

        fig = Figure(figsize=(9, 6.5))
        ax = fig.add_subplot(111)

        # ─── ФИЛЬТРАЦИЯ ВЫБРОСОВ (аналогично другим методам) ───────────────
        # q_low, q_high = df_filtered['dE'].quantile([0.001, 0.999])
        # filtered = df_filtered[(df_filtered['dE'] >= q_low) & (df_filtered['dE'] <= q_high)]
        filtered = df_filtered
        q_low, q_high = df_filtered_kde['dE'].quantile([0.001, 0.999])
        filtered_kde = df_filtered_kde[(df_filtered_kde['dE'] >= q_low) & (df_filtered_kde['dE'] <= q_high)]

        if filtered.empty:
            print("[WARNING] Filtered dE KDE data is empty after outlier removal. Using original.")
            filtered = df_filtered
        if filtered_kde.empty:
            print("[WARNING] Filtered dE KDE data is empty after outlier removal. Using original.")
            filtered_kde = df_filtered_kde

        # ─── ОПРЕДЕЛЕНИЕ ПАРАМЕТРОВ ПОСТРОЕНИЯ ─────────────────────────────
        if normalization == 'none':
            ylabel = 'Число шагов'
            use_log_y = True  # Для dE без нормировки используем логарифм по Y
            stat = 'count'
        elif normalization == 'particles':
            ylabel = 'Шагов на частицу'
            use_log_y = False
            stat = 'count'
        elif normalization == 'steps':
            ylabel = 'Доля шагов'
            use_log_y = False
            stat = 'count'
        elif normalization == 'density':
            ylabel = 'Плотность вероятности (1/МэВ)'
            use_log_y = False
            stat = 'density'
        else:
            raise ValueError(f"Неизвестный режим нормировки: {normalization}")

        dE_values = filtered['dE'].values

        # Определяем веса в зависимости от режима нормировки
        weights = None
        if normalization == 'particles':
            n_particles = filtered['track_id'].nunique() if 'track_id' in filtered.columns else 0
            if n_particles > 0:
                weights = np.ones(len(dE_values)) / n_particles
        elif normalization == 'steps':
            n_steps = len(filtered)
            if n_steps > 0:
                weights = np.ones(len(dE_values)) / n_steps
        elif normalization == 'density':
            pass

        # Создаем DataFrame для seaborn с весами, если они есть
        if weights is not None:
            plot_data = pd.DataFrame({
                'dE': dE_values,
                'weights': weights
            })
            sns.histplot(
                data=plot_data,
                x='dE',
                weights='weights',
                bins=100,
                kde=False,
                color='crimson',
                alpha=0.5,
                label='Все частицы',
                log_scale=(True, use_log_y),
                ax=ax,
                stat=stat
            )
        else:
            sns.histplot(
                data=dE_values,
                bins=100,
                kde=False,
                color='crimson',
                alpha=0.5,
                label='Все частицы',
                log_scale=(True, use_log_y),
                ax=ax,
                stat=stat
            )

        if show_kde:
            # --- ВАЖНО: density оставляем как было (твой "пиздато" вариант) ---
            if normalization == 'density':
                dE_kde = filtered_kde['dE'].values
                if len(dE_kde) > 1:
                    sns.kdeplot(
                        x=dE_kde,
                        ax=ax,
                        color='crimson',
                        linewidth=1.8,
                        log_scale=True,
                        cut=0
                    )
            else:
                # --- Для остальных режимов: KDE в log10(dE) + гладкое масштабирование под counts/weights ---
                dE_kde = filtered_kde['dE'].values
                dE_kde = dE_kde[np.isfinite(dE_kde) & (dE_kde > 0)]
                if len(dE_kde) > 1:
                    from scipy.stats import gaussian_kde
                    z = np.log10(dE_kde)
                    kde_z = gaussian_kde(z)
                    z_grid = np.linspace(z.min(), z.max(), 512)
                    x_grid = 10 ** z_grid
                    # pdf по z -> pdf по x: f_x(x) = f_z(log10 x) / (x ln 10)
                    y_density = kde_z(z_grid) / (x_grid * np.log(10.0))  # 1/MeV
                    # сумма весов == "масса" гистограммы в stat='count'
                    sum_w = float(np.sum(weights)) if weights is not None else float(len(dE_values))
                    # гладкая ширина лог-бина: Δx ≈ x * (r - 1)
                    # r определяем из диапазона и числа бинов (у тебя bins=100)
                    x_min = max(np.min(dE_values[dE_values > 0]), np.min(dE_kde))
                    x_max = max(np.max(dE_values), np.max(dE_kde))
                    nbins = 100
                    r = (x_max / x_min) ** (1.0 / nbins)
                    dx = x_grid * (r - 1.0)
                    # перевод density -> expected "count per bin" (или per-particle / fraction через weights)
                    y_plot = y_density * sum_w * dx
                    ax.plot(x_grid, y_plot, color='crimson', linewidth=1.8)

        # ─── НАСТРОЙКА ОСЕЙ И ПРИМЕНЕНИЕ ЛИМИТОВ ───────────────────────────
        if normalization == 'none':
            ax.set_yscale('log')
            if min_y_log:
                ax.set_ylim(bottom=min_y_log)
        else:
            ax.set_yscale('linear')

        # Применяем заданные лимиты
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        from matplotlib.ticker import LogFormatterSciNotation
        formatter = LogFormatterSciNotation(base=10.0, labelOnlyBase=False)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1))

        ax.set_title(f'Распределение dE за шаг {particle_type} частиц',
                     fontweight='bold', fontsize=14, pad=16)
        ax.set_xlabel('dE за шаг (МэВ)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(False)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major',
                       length=7, width=1.1, direction='in',
                       labelsize=10, top=True, right=True, bottom=True, left=True)
        ax.tick_params(axis='both', which='minor',
                       length=4, width=0.9, direction='in',
                       top=True, right=True, bottom=True, left=True)

        # ─── ЛЕГЕНДА И СТАТИСТИКА (как в других методах) ──────────────────
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            label_to_handle = dict(zip(labels, handles))
            # Одна серия -> порядок просто как есть (обычно 1 элемент)
            ordered_labels = list(label_to_handle.keys())
            ordered_handles = [label_to_handle[lbl] for lbl in ordered_labels]
            n = len(ordered_labels)
            if n <= 5:
                legend_loc = 'upper right'
                ncol = 1
            elif n <= 10:
                legend_loc = 'upper left'
                ncol = 2
            else:
                legend_loc = 'center left'
                ncol = 2
            legend = ax.legend(
                ordered_handles,
                ordered_labels,
                title='Тип частицы',
                loc=legend_loc,
                framealpha=0.35,
                fancybox=True,
                fontsize=9.5,
                title_fontsize=11,
                ncol=ncol,
                borderpad=0.6
            )

        # Статистика
        stats_artist = None
        if show_stats:
            stats_text = []
            total_steps_all = len(df_filtered)
            valid_data = df_filtered['dE'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_data) > 0:
                n_steps = len(valid_data)
                mean_de = valid_data.mean()
                std_de = valid_data.std()
                total_de = valid_data.sum()
            else:
                n_steps = 0
                mean_de = np.nan
                std_de = np.nan
                total_de = np.nan

            # один блок вместо цикла по particles
            if normalization == 'none':
                stats_text.append(f"Все частицы: {n_steps} шагов")
                stats_text.append(f" μ = {mean_de:.3f} МэВ")
                stats_text.append(f" Σ dE = {total_de:.3f} МэВ")
            elif normalization == 'particles':
                stats_text.append(f"Все частицы: μ = {mean_de:.3f} МэВ")
                stats_text.append(f" Σ dE = {total_de:.3f} МэВ")
            elif normalization == 'steps':
                # при одной группе это всегда 100%
                stats_text.append(f"Все частицы: 100.0% шагов")
                stats_text.append(f" μ = {mean_de:.3f} МэВ")
                stats_text.append(f" Σ dE = {total_de:.3f} МэВ")
            else:  # 'density'
                stats_text.append(f"Все частицы: μ = {mean_de:.3f} МэВ")
                stats_text.append(f" Σ dE = {total_de:.3f} МэВ")
                stats_text.append(f" σ = {std_de:.3f} МэВ")

            if stats_text:
                stats_cols = 1 if len(stats_text) <= 8 else 2 if len(stats_text) <= 20 else 3
                stats_block = self._format_stats_columns(stats_text, ncol=stats_cols)

                # Определяем положение статистики в зависимости от положения легенды
                if legend_loc == 'upper right':
                    stats_x, stats_y, stats_ha, stats_va = 0.02, 0.98, 'left', 'top'
                elif legend_loc == 'upper left':
                    stats_x, stats_y, stats_ha, stats_va = 0.98, 0.98, 'right', 'top'
                elif legend_loc == 'center left':
                    stats_x, stats_y, stats_ha, stats_va = 0.98, 0.98, 'right', 'top'
                else:
                    stats_x, stats_y, stats_ha, stats_va = 0.02, 0.98, 'left', 'top'

                # Если статистики много, сдвигаем ниже
                if len(stats_text) > 8:
                    stats_y = 0.02
                    stats_va = 'bottom'

                stats_artist = ax.text(
                    stats_x, stats_y,
                    stats_block,
                    transform=ax.transAxes,
                    fontsize=8.8,
                    va=stats_va, ha=stats_ha,
                    bbox=dict(boxstyle='square', pad=0.5, fc='white', alpha=0.22)
                )
                fig._stats_artist = stats_artist
        else:
            if show_stats:
                stats_text = []
                total_steps_all = len(df_filtered)
                valid_data = df_filtered['dE'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_data) > 0:
                    n_steps = len(valid_data)
                    mean_de = valid_data.mean()
                    std_de = valid_data.std()
                    total_de = valid_data.sum()
                else:
                    n_steps = 0
                    mean_de = np.nan
                    std_de = np.nan
                    total_de = np.nan

                if normalization == 'none':
                    stats_text.append(f"Все частицы: {n_steps} шагов")
                    stats_text.append(f" μ = {mean_de:.3f} МэВ")
                    stats_text.append(f" Σ dE = {total_de:.3f} МэВ")
                elif normalization == 'particles':
                    stats_text.append(f"Все частицы: μ = {mean_de:.3f} МэВ")
                    stats_text.append(f" Σ dE = {total_de:.3f} МэВ")
                elif normalization == 'steps':
                    stats_text.append(f"Все частицы: 100.0% шагов")
                    stats_text.append(f" μ = {mean_de:.3f} МэВ")
                    stats_text.append(f" Σ dE = {total_de:.3f} МэВ")
                else:  # 'density'
                    stats_text.append(f"Все частицы: μ = {mean_de:.3f} МэВ")
                    stats_text.append(f" Σ dE = {total_de:.3f} МэВ")
                    stats_text.append(f" σ = {std_de:.3f} МэВ")

                if stats_text:
                    stats_cols = 1 if len(stats_text) <= 8 else 2 if len(stats_text) <= 20 else 3
                    stats_block = self._format_stats_columns(stats_text, ncol=stats_cols)
                    stats_artist = ax.text(
                        0.98, 0.98,
                        stats_block,
                        transform=ax.transAxes,
                        fontsize=8.8,
                        va='top', ha='right',
                        bbox=dict(boxstyle='square', pad=0.5, fc='white', alpha=0.22)
                    )
                    fig._stats_artist = stats_artist

        # Регулировка отступов
        if handles and legend_loc in ['upper left', 'center left']:
            fig.subplots_adjust(left=0.15, right=0.85)
        else:
            fig.subplots_adjust(left=0.1, right=0.85)
        fig.tight_layout()

        # Кэширование
        if use_cache and hasattr(self, 'cache_mgr') and self.cache_mgr:
            self.cache_mgr.save_figure(fig, "dE", cache_key)
            print(f"[CACHE SAVE] dE: {particle_type} {data_hash[:8]}...")

        return fig

    def _visualize_process_energy_distribution(
            self,
            df,
            particle_type="частиц",
            process_name=None,
            energy_column='process_energy_loss_mev',
            show_kde=True,
            show_stats=True,
            min_y_log=0.8,
            use_cache=True,
            normalization='none',
            xlim=None,
            ylim=None
    ):
        """Распределение dE для конкретного процесса или всех процессов"""
        # Исключаем транспортные процессы
        exclude = ['OutOfWorld', 'Transportation', 'initStep', 'NoProcess']
        df_filtered = df[~df['process'].isin(exclude)].copy()
        if df_filtered.empty:
            return self._create_empty_plot(f'Нет данных для {particle_type} частиц')

        # Если указан конкретный процесс, фильтруем
        if process_name:
            df_filtered = df_filtered[df_filtered['process'] == process_name]
            if df_filtered.empty:
                return self._create_empty_plot(f'Нет данных для процесса "{process_name}" у {particle_type} частиц')

        # Проверяем наличие колонки энергии
        if energy_column not in df_filtered.columns:
            return self._create_empty_plot(f'Колонка {energy_column} не найдена в данных')

        # Отфильтровываем отрицательные значения и создаем копию для гистограммы
        df_hist = df_filtered[df_filtered[energy_column] >= 0].copy()
        if df_hist.empty:
            return self._create_empty_plot(f'Нет положительных значений {energy_column} для {particle_type} частиц')

        # Создаем отдельный датафрейм для KDE только с положительными значениями (>0)
        df_kde = df_hist[df_hist[energy_column] > 0].copy()

        # Кэширование
        if use_cache and self.cache_mgr:
            cache_key = {
                "plot_type": "process_energy_dist",
                "particle_type": particle_type,
                "process_name": process_name,
                "energy_column": energy_column,
                "show_kde": show_kde,
                "show_stats": show_stats,
                "normalization": normalization,
                "min_y_log": min_y_log,
                "data_hash": hashlib.md5(
                    pd.util.hash_pandas_object(df_filtered[['process', energy_column]]).values.tobytes()).hexdigest()
            }
            cached_fig = self.cache_mgr.load_figure("process_energy_dist", cache_key)
            if cached_fig is not None:
                print(
                    f"[CACHE HIT] Process energy distribution: {particle_type} {process_name if process_name else 'all'}")
                return cached_fig

        # СОЗДАЕМ ФИГУРУ ВНЕ ЗАВИСИМОСТИ ОТ process_name
        if process_name:
            fig = Figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            energies = df_hist[energy_column].to_numpy()
            # разделяем нули и >0
            energies_pos = energies[energies > 0]
            n_zero = int(np.sum(energies == 0))

            # log-x используем только если есть >0
            use_log_x = energies_pos.size > 0

            # bins
            if use_log_x:
                xmin = float(energies_pos.min())
                xmax = float(energies_pos.max())
                if xmin == xmax:
                    xmax = xmin * 1.01
                bins = np.logspace(np.log10(xmin), np.log10(xmax), 101)
            else:
                bins = 100

            if len(energies) == 0:
                return self._create_empty_plot('Нет данных для гистограммы')

            if use_log_x:
                energies_hist = energies[energies > 0]
            else:
                energies_hist = energies[energies >= 0]

            if energies_hist.size == 0:
                return self._create_empty_plot('Нет данных для гистограммы')

            # Датафрейм специально под seaborn (важно для weights!)
            df_plot = pd.DataFrame({energy_column: energies_hist})

            # --- Фильтруем выбросы для KDE (только положительные значения)
            filtered_kde = None
            if not df_kde.empty:
                kde_energies = df_kde[energy_column].values
                if len(kde_energies) > 1:
                    q_low_kde, q_high_kde = np.quantile(kde_energies, [0.001, 0.999])
                    filtered_kde = kde_energies[(kde_energies >= q_low_kde) & (kde_energies <= q_high_kde)]
                    if len(filtered_kde) == 0:
                        filtered_kde = kde_energies

            weights = None
            density_flag = False
            use_log_y = False

            if normalization == 'none':
                ylabel = 'Число событий'
                use_log_y = True
            elif normalization == 'density':
                ylabel = 'Плотность вероятности (1/МэВ)'
                density_flag = True
                use_log_y = False
            elif normalization == 'steps':
                ylabel = 'Доля событий'
                use_log_y = True
                weights = np.ones_like(energies_hist, dtype=float) / float(len(energies_hist))
            elif normalization == 'particles':
                ylabel = 'Событий на частицу'
                use_log_y = True
                if 'track_id' in df_hist.columns:
                    n_particles = int(df_hist['track_id'].nunique())
                    weights = (np.ones_like(energies_hist, dtype=float) / float(
                        n_particles)) if n_particles > 0 else None
                else:
                    ylabel = 'Число событий'
                    use_log_y = True

            bin_color = 'crimson'

            # --- ГИСТОГРАММА
            ax.hist(
                energies_hist,
                bins=bins,
                weights=weights,
                density=density_flag,
                histtype='bar',
                rwidth=0.95,
                color=bin_color,
                alpha=0.35,
                edgecolor=bin_color,
                linewidth=1.2
            )

            if use_log_x:
                ax.set_xscale('log')
            if use_log_y:
                ax.set_yscale('log')
                ax.set_ylim(bottom=1e-2)

            kde_drawn = False
            if show_kde and (filtered_kde is not None) and (len(filtered_kde) > 1):
                if hasattr(filtered_kde, "columns") and ("dE" in filtered_kde.columns):
                    dE_kde = filtered_kde["dE"].to_numpy()
                else:
                    dE_kde = np.asarray(filtered_kde)
                dE_kde = dE_kde[np.isfinite(dE_kde) & (dE_kde > 0)]

                if len(dE_kde) > 3:
                    from scipy.stats import gaussian_kde
                    from scipy.interpolate import UnivariateSpline

                    bin_edges = None
                    rects = []
                    for p in getattr(ax, "patches", []):
                        if hasattr(p, "get_x") and hasattr(p, "get_width") and hasattr(p, "get_height"):
                            w = float(p.get_width())
                            h = float(p.get_height())
                            if np.isfinite(w) and (w > 0) and np.isfinite(h) and (h > 0):
                                rects.append(p)

                    if rects:
                        lefts = np.array([float(r.get_x()) for r in rects], dtype=float)
                        widths = np.array([float(r.get_width()) for r in rects], dtype=float)
                        okr = np.isfinite(lefts) & np.isfinite(widths) & (widths > 0)
                        lefts = lefts[okr]
                        widths = widths[okr]
                        if lefts.size > 1:
                            order = np.argsort(lefts)
                            lefts = lefts[order]
                            widths = widths[order]
                            rights = lefts + widths
                            edges = np.concatenate([lefts, [rights[-1]]])
                            edges = edges[np.isfinite(edges)]
                            edges = np.unique(edges)
                            if edges.size > 2:
                                bin_edges = edges

                    if bin_edges is not None:
                        bin_edges = np.asarray(bin_edges, dtype=float)
                        bin_edges = bin_edges[np.isfinite(bin_edges)]
                        bin_edges = bin_edges[bin_edges > 0]
                        bin_edges = np.unique(bin_edges)

                    z = np.log10(dE_kde)
                    z = z[np.isfinite(z)]

                    if len(z) > 3:
                        z_lo, z_hi = np.quantile(z, [0.001, 0.999])
                        z_use = z[(z >= z_lo) & (z <= z_hi)]
                        if len(z_use) < 4:
                            z_use = z

                        zmin, zmax = float(z_use.min()), float(z_use.max())
                        span = zmax - zmin

                        z_reflect_lo = 2.0 * zmin - z_use
                        z_reflect_hi = 2.0 * zmax - z_use
                        z_aug = np.concatenate([z_use, z_reflect_lo, z_reflect_hi])

                        kde_z = gaussian_kde(z_aug)
                        kde_z.set_bandwidth(bw_method=kde_z.factor * 1.05)

                        if normalization == "density":
                            pad = 0.08 * span if span > 0 else 0.05
                            z_grid = np.linspace(zmin - pad, zmax + pad, 900)
                            x_grid = 10 ** z_grid
                            y_density = kde_z(z_grid) / (x_grid * np.log(10.0))

                            ok = np.isfinite(x_grid) & np.isfinite(y_density) & (x_grid > 0) & (y_density > 0)
                            x_plot = x_grid[ok]
                            y_plot = y_density[ok]

                            if len(x_plot) > 10:
                                ax.plot(x_plot, y_plot, color="crimson", linewidth=1.8, label="KDE")
                                kde_drawn = True
                        else:
                            if (bin_edges is not None) and (len(bin_edges) > 2):
                                # режем края под рабочий диапазон KDE
                                z_edges = np.log10(bin_edges)
                                z_edges = z_edges[np.isfinite(z_edges)]
                                z_edges = z_edges[(z_edges >= zmin) & (z_edges <= zmax)]

                                if z_edges.size > 2:
                                    z_edges = np.unique(np.concatenate(
                                        [[max(zmin, z_edges.min())], z_edges, [min(zmax, z_edges.max())]]))
                                    z_edges = z_edges[np.isfinite(z_edges)]
                                    z_edges = np.unique(z_edges)
                                    z_edges = z_edges[np.argsort(z_edges)]

                                    if z_edges.size > 2:
                                        masses = np.array(
                                            [kde_z.integrate_box_1d(z_edges[i], z_edges[i + 1]) for i in
                                             range(len(z_edges) - 1)],
                                            dtype=float,
                                        )
                                        masses = np.where(np.isfinite(masses) & (masses > 0), masses, np.nan)

                                        # центры бинов в x
                                        x_edges = 10 ** z_edges
                                        x_centers = np.sqrt(x_edges[:-1] * x_edges[1:]) if use_log_x else 0.5 * (
                                                x_edges[:-1] + x_edges[1:])

                                        y_bin = None
                                        if normalization == "none":
                                            y_bin = masses * float(len(energies_hist))
                                        elif normalization == "steps":
                                            y_bin = masses
                                        elif normalization == "particles":
                                            y_bin = None
                                            if hasattr(df_hist, "columns") and ("track_id" in df_hist.columns):
                                                n_particles = float(df_hist["track_id"].nunique())
                                                if n_particles > 0:
                                                    y_bin = masses * (float(len(energies_hist)) / n_particles)

                                        if y_bin is not None:
                                            ok = np.isfinite(x_centers) & np.isfinite(y_bin) & (x_centers > 0) & (
                                                    y_bin > 0)
                                            xc = x_centers[ok]
                                            yc = y_bin[ok]

                                            if xc.size > 6:
                                                if use_log_x:
                                                    X = np.log10(xc)
                                                    Y = np.log10(yc)
                                                    order = np.argsort(X)
                                                    X = X[order]
                                                    Y = Y[order]

                                                    # сглаживание: s можно подкрутить (меньше -> ближе к точкам)
                                                    spl = UnivariateSpline(X, Y, s=0.5 * len(X))
                                                    Xg = np.linspace(X.min(), X.max(), 900)
                                                    x_plot = 10 ** Xg
                                                    y_plot = 10 ** spl(Xg)
                                                else:
                                                    X = xc
                                                    Y = np.log10(yc)
                                                    order = np.argsort(X)
                                                    X = X[order]
                                                    Y = Y[order]

                                                    spl = UnivariateSpline(X, Y, s=0.5 * len(X))
                                                    x_plot = np.linspace(X.min(), X.max(), 900)
                                                    y_plot = 10 ** spl(x_plot)

                                                ok2 = np.isfinite(x_plot) & np.isfinite(y_plot) & (x_plot > 0) & (
                                                        y_plot > 0)
                                                x_plot = x_plot[ok2]
                                                y_plot = y_plot[ok2]

                                                if x_plot.size > 10:
                                                    ax.plot(x_plot, y_plot, color="crimson", linewidth=1.8, label="KDE")
                                                    kde_drawn = True

            if kde_drawn:
                ax.legend(loc="upper right", framealpha=0.3)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            from matplotlib.ticker import LogFormatterSciNotation
            formatter = LogFormatterSciNotation(base=10.0, labelOnlyBase=False)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1))

            ax.set_title(f'Распределение dE за шаг {particle_type} частиц',
                         fontweight='bold', fontsize=14, pad=16)
            ax.set_xlabel('dE за шаг (МэВ)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major',
                           length=7, width=1.1, direction='in',
                           labelsize=10, top=True, right=True, bottom=True, left=True)
            ax.tick_params(axis='both', which='minor',
                           length=4, width=0.9, direction='in',
                           top=True, right=True, bottom=True, left=True)

            title = f'Распределение dE для процесса "{process_name}"'
            ax.set_title(title, fontweight='bold', fontsize=14, pad=16)
            ax.set_xlabel(f'dE (МэВ)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(False)

            try:
                show_stats = bool(show_stats.get())
            except AttributeError:
                show_stats = bool(show_stats)

            try:
                show_kde = bool(show_kde.get())
            except AttributeError:
                show_kde = bool(show_kde)

            # ─── ЛЕГЕНДА И СТАТИСТИКА ─────────────────────────────────────────
            fig._stats_artist = None  # важно: сброс, чтобы show_plot не хватал старое
            handles, labels = ax.get_legend_handles_labels()

            # 1) Легенда (если есть что показывать)
            legend_loc = 'upper right'
            ncol = 1
            if handles:
                # сохраняем порядок, но убираем дубликаты
                label_to_handle = {}
                for h, lbl in zip(handles, labels):
                    if lbl and lbl not in label_to_handle:
                        label_to_handle[lbl] = h

                ordered_labels = list(label_to_handle.keys())
                ordered_handles = [label_to_handle[lbl] for lbl in ordered_labels]
                n = len(ordered_labels)

                if n <= 5:
                    legend_loc, ncol = 'upper right', 1
                elif n <= 10:
                    legend_loc, ncol = 'upper left', 2
                else:
                    legend_loc, ncol = 'center left', 2

                ax.legend(
                    ordered_handles,
                    ordered_labels,
                    title='Тип частицы',
                    loc=legend_loc,
                    framealpha=0.35,
                    fancybox=True,
                    fontsize=9.5,
                    title_fontsize=11,
                    ncol=ncol,
                    borderpad=0.6
                )

            if show_stats:
                stats_text = []
                stats_text.append(f"Процесс: {process_name}")
                stats_text.append(f"Всего событий: {len(df_hist)}")

                # данные по энергии (только валидные)
                vals = df_hist[energy_column].replace([np.inf, -np.inf], np.nan).dropna()
                if len(vals) > 0:
                    total_e = float(vals.sum())
                    mean_e = float(vals.mean())
                    std_e = float(vals.std())
                    med_e = float(np.median(vals.values))
                    max_e = float(vals.max())
                else:
                    total_e = mean_e = std_e = med_e = max_e = float('nan')

                stats_text.append(f"Σ dE: {total_e:.3f} МэВ")
                stats_text.append(f"μ: {mean_e:.3f} МэВ")
                stats_text.append(f"медиана: {med_e:.3f} МэВ")
                stats_text.append(f"σ: {std_e:.3f} МэВ")
                stats_text.append(f"max: {max_e:.3f} МэВ")

                if 'track_id' in df_hist.columns:
                    n_tracks = int(df_hist['track_id'].nunique())
                    stats_text.append(f"Уникальных частиц: {n_tracks}")
                    if n_tracks > 0 and np.isfinite(total_e):
                        stats_text.append(f"dE/частицу: {total_e / n_tracks:.3f} МэВ")

                # форматируем в колонки, если есть твой хелпер
                stats_cols = 1 if len(stats_text) <= 8 else 2 if len(stats_text) <= 20 else 3
                if hasattr(self, "_format_stats_columns"):
                    stats_block = self._format_stats_columns(stats_text, ncol=stats_cols)
                else:
                    stats_block = "\n".join(stats_text)

                PAD_X = 0.02
                PAD_Y = 0.98  # именно ВВЕРХ

                # ставим в противоположный угол по X, а по Y всегда вверх
                if legend_loc in ('upper right', 'right'):
                    stats_x, stats_y, stats_ha, stats_va = PAD_X, PAD_Y, 'left', 'top'  # вверх-влево
                elif legend_loc in ('upper left', 'center left', 'left'):
                    stats_x, stats_y, stats_ha, stats_va = 0.98, PAD_Y, 'right', 'top'  # вверх-вправо
                else:
                    stats_x, stats_y, stats_ha, stats_va = PAD_X, PAD_Y, 'left', 'top'

                # если текста много --- НЕ роняем вниз, а чуть опускаем от верхней границы
                if len(stats_text) > 8:
                    stats_y = 0.95
                if len(stats_text) > 12:
                    stats_y = 0.92

                stats_artist = ax.text(
                    stats_x, stats_y,
                    stats_block,
                    transform=ax.transAxes,
                    fontsize=8.8,
                    va=stats_va, ha=stats_ha,
                    bbox=dict(boxstyle='square', pad=0.5, fc='white', alpha=0.22)
                )
                fig._stats_artist = stats_artist

            # 3) Отступы (с учётом левой легенды)
            if handles and legend_loc in ('upper left', 'center left'):
                fig.subplots_adjust(left=0.15, right=0.85)
            else:
                fig.subplots_adjust(left=0.10, right=0.85)

        else:  # режим без указания конкретного процесса
            fig = Figure(figsize=(14, 10))

            # Используем df_hist для статистики
            if df_hist.empty:
                return self._create_empty_plot(f'Нет данных по {energy_column}')

            # Статистика по процессам
            process_stats = (
                df_hist
                .groupby('process')[energy_column]
                .agg(['count', 'sum', 'mean', 'std', 'median', 'max'])
                .round(6)
            )
            process_stats = process_stats.sort_values('sum', ascending=False)

            top_processes = process_stats.head(10).index.tolist()
            if not top_processes:
                return self._create_empty_plot(f'Нет процессов с данными по {energy_column}')

            # Создаем 4 подграфика
            axes = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 2, 3),
                fig.add_subplot(2, 2, 4),
            ]

            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_processes)))

            # 1) суммарная энергия
            ax1 = axes[0]
            total_energy = process_stats.loc[top_processes, 'sum'].values
            bars1 = ax1.bar(range(len(top_processes)), total_energy, color=colors, alpha=0.7)
            ax1.set_xticks(range(len(top_processes)))
            ax1.set_xticklabels(top_processes, rotation=45, ha='right', fontsize=9)
            ax1.set_ylabel('Суммарное энерговыделение (МэВ)')
            ax1.set_title('Суммарное энерговыделение по процессам')
            ax1.grid(False)

            # Подписи значений
            for bar, val in zip(bars1, total_energy):
                if val > max(total_energy) * 0.05:
                    ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                             f'{val:.1e}', ha='center', va='bottom', fontsize=8)

            # 2) среднее энерговыделение
            ax2 = axes[1]
            mean_energy = process_stats.loc[top_processes, 'mean'].values
            bars2 = ax2.bar(range(len(top_processes)), mean_energy, color=colors, alpha=0.7)
            ax2.set_xticks(range(len(top_processes)))
            ax2.set_xticklabels(top_processes, rotation=45, ha='right', fontsize=9)
            ax2.set_ylabel('Среднее энерговыделение (МэВ)')
            ax2.set_title('Среднее энерговыделение на событие')
            ax2.grid(False)

            # 3) количество событий
            ax3 = axes[2]
            counts = process_stats.loc[top_processes, 'count'].values
            bars3 = ax3.bar(range(len(top_processes)), counts, color=colors, alpha=0.7)
            ax3.set_xticks(range(len(top_processes)))
            ax3.set_xticklabels(top_processes, rotation=45, ha='right', fontsize=9)
            ax3.set_ylabel('Количество событий')
            ax3.set_title('Количество событий по процессам')
            ax3.grid(False)

            # 4) boxplot (топ-5 процессов) с логарифмической шкалой
            ax4 = axes[3]
            boxplot_data = []
            boxplot_labels = []
            for p in top_processes[:5]:
                vals = df_hist[df_hist['process'] == p][energy_column].values
                # Берем только положительные значения для boxplot
                pos_vals = vals[vals > 0]
                if len(pos_vals) >= 5:
                    boxplot_data.append(pos_vals)
                    boxplot_labels.append(p)

            if boxplot_data:
                bp = ax4.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors[:len(boxplot_data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax4.set_ylabel(f'{energy_column} (МэВ)')
                ax4.set_title('Распределение энерговыделения (топ-5 процессов)')
                ax4.set_yscale('log')
                ax4.grid(False)
                ax4.tick_params(axis='x', rotation=45)

            fig.suptitle(
                f'Статистика {energy_column} по процессам ({particle_type} частицы)',
                fontsize=14, fontweight='bold', y=0.98
            )
            fig.tight_layout()

        # Кэширование
        if use_cache and self.cache_mgr:
            self.cache_mgr.save_figure(fig, "process_energy_dist", cache_key)
            print(f"[CACHE SAVE] Process energy distribution: {particle_type} {cache_key['data_hash'][:8]}...")

        return fig

    def _visualize_process_heatmap(self, df, particle_type="частиц", selected_process=None,
                                   heatmap_mode="counts", unit="MeV",
                                   energy_column="process_energy_loss_mev",
                                   use_cache=True):
        """Тепловая карта для выбранного процесса или всех процессов"""
        exclude_processes = ['OutOfWorld', 'Transportation', 'initStep', 'NoProcess']
        df_filtered = df[~df['process'].isin(exclude_processes)]
        if df_filtered.empty:
            return self._create_empty_plot(f'Нет данных о процессах для {particle_type} частиц')

        if selected_process is None:
            return self._create_process_selection_ui(df_filtered, particle_type)

        # Если указан конкретный процесс
        df_filtered = df_filtered[df_filtered['process'] == selected_process]
        if df_filtered.empty:
            return self._create_empty_plot(f'Нет данных для процесса "{selected_process}" у {particle_type} частиц')

        # Кэширование
        if use_cache and self.cache_mgr:
            cache_key = {
                "plot_type": "process_heatmap",
                "particle_type": particle_type,
                "selected_process": selected_process,
                "heatmap_mode": heatmap_mode,
                "unit": unit,
                "energy_column": energy_column,
                "data_hash":
                    hashlib.md5(pd.util.hash_pandas_object(df_filtered).values.tobytes()).hexdigest()
            }
            cached_fig = self.cache_mgr.load_figure("process_heatmap", cache_key)
            if cached_fig is not None:
                print(
                    f"[CACHE HIT] process_heatmap загружен: {particle_type} {selected_process} {cache_key['data_hash'][:8]}...")
                return cached_fig

        return self._build_single_process_heatmap(df_filtered, selected_process, particle_type,
                                                  heatmap_mode=heatmap_mode,
                                                  energy_column=energy_column,
                                                  unit=unit,
                                                  use_cache=use_cache)

    def _visualize_energy_deposition_heatmap(self, df, particle_type="частиц",
                                             value_column='process_energy_loss_mev',
                                             use_cache=True):
        """Тепловая карта с энерговыделением"""
        if df.empty or not all(col in df.columns for col in ['x_mm', 'y_mm', 'z_mm', value_column]):
            return self._create_empty_plot(f'Нет данных для {particle_type} частиц')

        # Кэширование
        if use_cache and self.cache_mgr:
            cache_key = {
                "plot_type": "energy_deposition_heatmap",
                "particle_type": particle_type,
                "value_column": value_column,
                "data_hash": hashlib.md5(
                    pd.util.hash_pandas_object(df[['x_mm', 'y_mm', 'z_mm', value_column]]).values.tobytes()
                ).hexdigest(),
                "material_limits": str(self.material_dimensions.get_limits())
            }
            cached_fig = self.cache_mgr.load_figure("energy_deposition_heatmap", cache_key)
            if cached_fig is not None:
                print(f"[CACHE HIT] Energy deposition heatmap: {particle_type}")
                return cached_fig

        df_clean = df.copy()
        limits = self.material_dimensions.get_limits()

        # Фильтруем бесконечности и NaN
        df_clean = df_clean[np.isfinite(df_clean[value_column])]
        df_clean = df_clean[df_clean[value_column] >= 0]  # Только положительные значения

        if df_clean.empty:
            return self._create_empty_plot(f'Нет валидных данных энерговыделения для {particle_type} частиц')

        fig = Figure(figsize=(13.5, 4.8), dpi=120)

        projections = [
            ('x_mm', 'y_mm', 'X / Y', 'X (мм)', 'Y (мм)'),
            ('x_mm', 'z_mm', 'X / Z', 'X (мм)', 'Z (мм)'),
            ('y_mm', 'z_mm', 'Y / Z', 'Y (мм)', 'Z (мм)')
        ]

        for i, (x_col, y_col, title, xlabel, ylabel) in enumerate(projections):
            ax = fig.add_subplot(1, 3, i + 1)
            x = df_clean[x_col].values
            y = df_clean[y_col].values
            values = df_clean[value_column].values

            if len(x) < 3:
                ax.text(0.5, 0.5, 'мало данных', ha='center', va='center', fontsize=10)
                continue

            # Сетка
            x_min, x_max = limits[x_col[0]]
            y_min, y_max = limits[y_col[0]]

            grid_size = 140
            xi = np.linspace(x_min, x_max, grid_size)
            yi = np.linspace(y_min, y_max, grid_size)

            # Суммируем энергию в ячейках
            zi, xedges, yedges = np.histogram2d(x, y, bins=[xi, yi], weights=values)

            # Логарифмическое масштабирование для лучшей визуализации
            zi_log = np.log10(zi + 1e-10)  # Добавляем маленькое значение для избежания log(0)

            im = ax.pcolormesh(xedges, yedges, zi_log.T, cmap='hot', shading='auto')
            ax.set_title(title, fontsize=10.5, pad=6)
            ax.set_xlabel(xlabel, fontsize=9.5)
            ax.set_ylabel(ylabel, fontsize=9.5)
            ax.set_aspect('equal')
            ax.grid(False)

            # Цветовая шкала
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.set_label(f'dE, МэВ)', fontsize=8.5)

        fig.suptitle(f'Карта энерговыделения {particle_type} частиц ({value_column})\n',
                     fontsize=14, fontweight='bold', y=0.96)

        if use_cache and self.cache_mgr:
            self.cache_mgr.save_figure(fig, "energy_deposition_heatmap", cache_key)
            print(f"[CACHE SAVE] Energy deposition heatmap: {particle_type} {cache_key['data_hash'][:8]}...")

        return fig

    def _create_process_selection_ui(self, df, particle_type="частиц"):
        """Создает интерфейс для выбора процесса"""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Получаем список процессов с количеством событий
        process_counts = df['process'].value_counts()

        # Создаем текстовое представление
        text = f"Выберите процесс для {particle_type} частиц:\n\n"
        for i, (process, count) in enumerate(process_counts.items(), 1):
            percentage = (count / len(df)) * 100
            text += f"{i:2}. {process:<25} - {count:>6} событий ({percentage:.1f}%)\n"
        text += "\nИспользуйте кнопку 'Тепловая карта процессов' и выпадающий список в интерфейсе"

        ax.text(0.5, 0.5, text,
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title(f'Доступные процессы для {particle_type} частиц\n'
                     f'Всего процессов: {len(process_counts)}, Всего событий: {len(df)}',
                     fontsize=12, pad=20)
        fig.tight_layout()
        return fig

    def _build_single_process_heatmap(self, df, process_name, particle_type="частиц",
                                      heatmap_mode="counts",
                                      energy_column="process_energy_loss_mev",
                                      unit="MeV",
                                      use_cache=True):
        """Построение тепловой карты для одного процесса"""
        if len(df) < 2:
            return self._create_empty_plot(f'Недостаточно данных для процесса "{process_name}"')

        if use_cache and self.cache_mgr:
            cache_key = {
                "plot_type": "single_process_heatmap",
                "particle_type": particle_type,
                "process_name": process_name,
                "heatmap_mode": heatmap_mode,
                "unit": unit,
                "energy_column": energy_column,
                "data_hash": hashlib.md5(
                    pd.util.hash_pandas_object(
                        df[['x_mm', 'y_mm', 'z_mm', energy_column]] if heatmap_mode == "dE" else df[
                            ['x_mm', 'y_mm', 'z_mm']]
                    ).values.tobytes()
                ).hexdigest()[:16],
                "limits": str(self.material_dimensions.get_limits())
            }
            cached_fig = self.cache_mgr.load_figure("single_process_heatmap", cache_key)
            if cached_fig is not None:
                print(f"[CACHE HIT] single_process_heatmap {process_name} {heatmap_mode} {unit}")
                return cached_fig

        # Получаем пределы
        limits = self.material_dimensions.get_limits()

        # Фигура
        fig = Figure(figsize=(13, 5), dpi=120)

        # Верхняя панель для информации
        ax_info = fig.add_axes([0.1, 0.95, 0.8, 0.05])
        ax_info.axis('off')

        # Получаем все процессы для переключения
        all_processes = sorted(df['process'].unique())
        current_idx = list(all_processes).index(process_name) if process_name in all_processes else 0

        # Информация о процессе
        n_events = len(df)
        info_text = (f'Процесс: {process_name} | '
                     f'Событий: {n_events:,} | '
                     f'{particle_type} частицы | '
                     f'Режим: {heatmap_mode}')
        if heatmap_mode == "dE":
            info_text += f" | Единицы: {unit}"
        ax_info.text(0.5, 0.5, info_text,
                     ha='center', va='center',
                     fontsize=11, fontweight='bold')

        # Проекции
        projections = [
            ('x_mm', 'y_mm', 'X/Y', 'X (мм)', 'Y (мм)'),
            ('x_mm', 'z_mm', 'X/Z', 'X (мм)', 'Z (мм)'),
            ('y_mm', 'z_mm', 'Y/Z', 'Y (мм)', 'Z (мм)')
        ]

        for i, (x_col, y_col, proj_title, xlabel, ylabel) in enumerate(projections):
            ax = fig.add_subplot(1, 3, i + 1)
            x = df[x_col].values
            y = df[y_col].values

            if len(x) < 3 or len(y) < 3:
                ax.text(0.5, 0.5, 'мало данных', ha='center', va='center', fontsize=8)
                continue

            x_min, x_max = limits[x_col[0]]
            y_min, y_max = limits[y_col[0]]

            # Сетка
            grid_size = 140
            xi = np.linspace(x_min, x_max, grid_size)
            yi = np.linspace(y_min, y_max, grid_size)
            xx, yy = np.meshgrid(xi, yi)

            try:
                if heatmap_mode == "counts":
                    weights = np.ones(len(x))
                    cbar_label = "Плотность шагов"
                    cmap = 'magma'
                elif heatmap_mode == "dE":
                    if energy_column not in df.columns:
                        ax.text(0.5, 0.5, 'нет данных dE', ha='center', va='center', fontsize=8)
                        continue
                    weights = df[energy_column].values
                    if unit.lower() == "kev":
                        weights = weights * 1e3
                        cbar_label = "Плотность энерговыделения (keV / mm²)"
                    else:
                        cbar_label = "Плотность энерговыделения (MeV / mm²)"
                    cmap = 'hot'

                if heatmap_mode == "counts":
                    zi = self._adaptive_density_field(x, y, xx, yy, weights=weights, normalize_for_density=True)
                else:
                    # dE: так же как counts, только веса = dE и normalize_for_density=False
                    zi = self._adaptive_density_field(x, y, xx, yy, weights=weights, normalize_for_density=False)

                if zi is None:
                    ax.text(0.5, 0.5, 'мало данных', ha='center', va='center', fontsize=8)
                    continue

                # Построение контурной карты
                contour = ax.contourf(xi, yi, zi, levels=50, cmap=cmap)
                ax.contour(xi, yi, zi, levels=35, colors='black', linewidths=0.3, alpha=0.3)

                ax.set_title(f'{proj_title}', fontsize=10, pad=6)
                ax.set_xlabel(xlabel, fontsize=9)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(False)
                ax.set_aspect('equal')

                # Цветовая шкала
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="4%", pad=0.1)
                cbar = fig.colorbar(contour, cax=cax, orientation='vertical')
                cbar.set_label(cbar_label, fontsize=8)
                cbar.ax.tick_params(labelsize=7)

            except Exception as e:
                print(f"\nОшибка построения {proj_title}: {e}")
                import traceback
                traceback.print_exc()
                ax.text(0.5, 0.5, 'ошибка', ha='center', va='center', fontsize=8)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        # Отступы
        fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.3)

        # Сохраняем информацию для навигации
        fig._process_info = {
            'current_process': process_name,
            'all_processes': all_processes,
            'particle_type': particle_type,
            'heatmap_mode': heatmap_mode,
            'unit': unit,
            'data_frame': df.copy()
        }
        fig._no_tight_layout = True

        if use_cache and self.cache_mgr:
            self.cache_mgr.save_figure(fig, "single_process_heatmap", cache_key)
            print(f"[CACHE SAVE] single_process_heatmap {process_name} {heatmap_mode} {unit}")

        return fig

    def _visualize_process_energy_heatmap(
            self,
            df,
            particle_type="частиц",
            process_name=None,
            energy_column='process_energy_loss_mev',
            use_cache=True
    ):
        """Тепловая карта пространственного распределения энерговыделения для процесса"""
        if df.empty or not all(col in df.columns for col in ['x_mm', 'y_mm', 'z_mm', energy_column]):
            return self._create_empty_plot(f'Нет координатных данных или {energy_column} для {particle_type} частиц')

        # Исключаем транспортные процессы
        exclude = ['OutOfWorld', 'Transportation', 'initStep', 'NoProcess']
        df_filtered = df[~df['process'].isin(exclude)].copy()

        if process_name:
            df_filtered = df_filtered[df_filtered['process'] == process_name]
            if df_filtered.empty:
                return self._create_empty_plot(f'Нет данных для процесса "{process_name}"')

        if df_filtered.empty:
            return self._create_empty_plot(f'Нет данных для {particle_type} частиц')

        df_filtered = df_filtered[df_filtered[energy_column] >= 0].copy()
        if df_filtered.empty:
            return self._create_empty_plot(f'Нет положительных значений {energy_column}')

        # Кэширование
        if use_cache and self.cache_mgr:
            cache_key = {
                "plot_type": "process_energy_heatmap",
                "particle_type": particle_type,
                "process_name": process_name,
                "energy_column": energy_column,
                "data_hash": hashlib.md5(pd.util.hash_pandas_object(
                    df_filtered[['x_mm', 'y_mm', 'z_mm', energy_column]]).values.tobytes()).hexdigest(),
                "material_limits": str(self.material_dimensions.get_limits())
            }
            cached_fig = self.cache_mgr.load_figure("process_energy_heatmap", cache_key)
            if cached_fig is not None:
                print(f"[CACHE HIT] Process energy heatmap: {particle_type} {process_name if process_name else 'all'}")
                return cached_fig

        df_clean = df_filtered.copy()
        limits = self.material_dimensions.get_limits()

        fig = Figure(figsize=(13.5, 4.8), dpi=120)

        projections = [
            ('x_mm', 'y_mm', 'X / Y', 'X (мм)', 'Y (мм)'),
            ('x_mm', 'z_mm', 'X / Z', 'X (мм)', 'Z (мм)'),
            ('y_mm', 'z_mm', 'Y / Z', 'Y (мм)', 'Z (мм)')
        ]

        for i, (x_col, y_col, title, xlabel, ylabel) in enumerate(projections):
            ax = fig.add_subplot(1, 3, i + 1)
            x = df_clean[x_col].values
            y = df_clean[y_col].values
            energies = df_clean[energy_column].values

            if len(x) < 3:
                ax.text(0.5, 0.5, 'мало данных', ha='center', va='center', fontsize=10)
                continue

            x_min, x_max = limits[x_col[0]]
            y_min, y_max = limits[y_col[0]]

            # Создаем сетку
            grid_size = 140
            xi = np.linspace(x_min, x_max, grid_size)
            yi = np.linspace(y_min, y_max, grid_size)

            # Суммируем энергию в ячейках сетки
            zi, xedges, yedges = np.histogram2d(
                x, y,
                bins=[xi, yi],
                weights=energies,
                density=False
            )

            # Логарифмическое масштабирование для лучшей визуализации
            zi_log = np.log10(zi + 1e-10)  # + маленькое значение чтобы избежать log(0)

            # Тепловая карта
            im = ax.pcolormesh(xedges, yedges, zi_log.T, cmap='hot', shading='auto')
            ax.set_title(title, fontsize=10.5, pad=6)
            ax.set_xlabel(xlabel, fontsize=9.5)
            ax.set_ylabel(ylabel, fontsize=9.5)
            ax.set_aspect('equal')
            ax.grid(False)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # Контуры
            try:
                ax.contour(
                    xi[:-1], yi[:-1], zi_log.T,
                    levels=30,
                    colors='black',
                    linewidths=0.3,
                    alpha=0.3
                )
            except:
                pass

            # Цветовая шкала
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.set_label(f'dE, МэВ', fontsize=8.5)
            cbar.ax.tick_params(labelsize=7.5)

        # Заголовок
        if process_name:
            title_text = f'Пространственное распределение {energy_column} для процесса "{process_name}"\n({particle_type} частицы)'
        else:
            title_text = f'Пространственное распределение {energy_column} по всем процессам\n({particle_type} частицы)'

        fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.96)
        fig.subplots_adjust(left=0.07, right=0.93, top=0.90, bottom=0.15, wspace=0.42)

        # Добавляем статистику
        if process_name:
            stats_text = []
            stats_text.append(f"Процесс: {process_name}")
            stats_text.append(f"Событий: {len(df_filtered):,}")
            stats_text.append(f"Суммарная энергия: {df_filtered[energy_column].sum():.3f} МэВ")
            stats_text.append(f"Средняя энергия/событие: {df_filtered[energy_column].mean():.3f} МэВ")
            stats_block = "\n".join(stats_text)

            # Добавляем текст в левый верхний угол
            fig.text(0.02, 0.98, stats_block,
                     transform=fig.transFigure,
                     fontsize=9,
                     va='top', ha='left',
                     bbox=dict(boxstyle='square', pad=0.5, fc='white', alpha=0.3, edgecolor='gray'))

        fig._no_tight_layout = True

        if use_cache and self.cache_mgr:
            self.cache_mgr.save_figure(fig, "process_energy_heatmap", cache_key)
            print(f"[CACHE SAVE] Process energy heatmap: {particle_type} {process_name if process_name else 'all'}")

        return fig

    def _decide_clustering(self, Z, k=5):
        """
        + Decide whether clustering is statistically justified.
        + Returns: (bool, labels or None)
        """
        n = len(Z)
        if n < 50:
            return False, None

        # ---- kNN dispersion test ----
        nn = NearestNeighbors(n_neighbors=min(k, n - 1)).fit(Z)
        dists, _ = nn.kneighbors(Z)
        dk = dists[:, -1]
        med = np.median(dk)
        iqr = np.subtract(*np.percentile(dk, [75, 25]))
        dispersion = iqr / med if med > 0 else 0.0

        # homogeneous distributions → small dispersion
        likely_clustered = dispersion > 0.6

        if not likely_clustered:
            return False, None

        # ---- attempt real clustering ----
        labels = self._run_clustering(Z)
        if labels is None:
            return False, None

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_frac = np.mean(labels == -1)

        # stability conditions
        if n_clusters < 2:
            return False, None
        if noise_frac > 0.8:
            return False, None

        return True, labels

    def _run_clustering(self, Z):
        """
        + HDBSCAN if available, otherwise scale-normalized DBSCAN.
        + """
        if _HDBSCAN_AVAILABLE:
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=max(15, len(Z) // 200),
                    min_samples=None
                )
                return clusterer.fit_predict(Z)
            except Exception:
                pass

        # ---- DBSCAN fallback ----
        try:
            nn = NearestNeighbors(n_neighbors=min(6, len(Z) - 1)).fit(Z)
            dists, _ = nn.kneighbors(Z)
            dk = dists[:, -1]
            eps = np.percentile(dk, 85)
            if eps <= 0:
                return None
            min_samples = max(5, len(Z) // 300)
            db = DBSCAN(eps=eps, min_samples=min_samples)
            return db.fit_predict(Z)
        except Exception:
            return None

    def _knn_density(self, x, y, xx, yy, k=None):
        """
        kNN-based physical density field.
        """
        X = np.column_stack([x, y])
        n = len(X)
        if k is None:
            k = max(15, int(np.sqrt(n)))
        tree = NearestNeighbors(n_neighbors=k).fit(X)
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        dists, _ = tree.kneighbors(grid)
        rk = dists[:, -1]
        density = k / (np.pi * rk ** 2 + 1e-12)
        return density.reshape(xx.shape)

    def _gmm_density(self, x, y, xx, yy, max_components=6):
        """
        Gaussian mixture model PDE with BIC selection.
        """
        from sklearn.mixture import GaussianMixture
        X = np.column_stack([x, y])
        Z = StandardScaler().fit_transform(X)

        best_gmm = None
        best_bic = np.inf

        for k in range(1, max_components + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                reg_covar=1e-6
            )
            gmm.fit(Z)
            bic = gmm.bic(Z)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

        if best_gmm is None or best_gmm.n_components == 1:
            return None

        grid = np.column_stack([xx.ravel(), yy.ravel()])
        Zg = StandardScaler().fit_transform(X).transform(grid)
        logp = best_gmm.score_samples(Zg)
        return np.exp(logp).reshape(xx.shape)

    def _calculate_eps(self, points, k=5):
        """Автоматический расчет eps для DBSCAN"""
        try:
            from sklearn.neighbors import NearestNeighbors
            n_samples = len(points)
            if n_samples < 10:
                return 0.1

            # Анализ расстояний до k-го соседа
            k = min(k, n_samples - 1)
            nbrs = NearestNeighbors(n_neighbors=k).fit(points)
            distances, _ = nbrs.kneighbors(points)

            # Берем расстояние до k-го соседа и усредняем
            k_distances = distances[:, k - 1]
            eps = np.percentile(k_distances, 50)  # Медиана
            return max(0.05, min(eps, 1.0))
        except Exception:
            return 0.1  # Значение по умолчанию

    def _build_all_processes_heatmap(self, df, particle_type="частиц", limits=None,
                                     heatmap_mode="counts",
                                     energy_column="process_energy_loss_mev",
                                     unit="MeV",
                                     use_cache=True):
        """Метод для всех процессов (для совместимости)"""
        if use_cache and self.cache_mgr:
            cache_key = {
                "plot_type": "process_heatmaps",
                "particle_type": particle_type,
                "heatmap_mode": heatmap_mode,
                "unit": unit,
                "energy_column": energy_column,
                "data_hash": hashlib.md5(
                    pd.util.hash_pandas_object(df[['x_mm', 'y_mm', 'z_mm', 'process',
                                                   energy_column] if heatmap_mode == "dE" else ['x_mm', 'y_mm', 'z_mm',
                                                                                                'process']]).values.tobytes()
                ).hexdigest(),
                "limits": str(limits)
            }
            cached = self.cache_mgr.load_figure("process_heatmaps", cache_key)
            if cached is not None:
                print("[CACHE HIT] process heatmaps")
                return cached

        if limits is None:
            limits = self.material_dimensions.get_limits()

        process_counts = df['process'].value_counts()
        min_samples = 2
        valid_processes = [p for p, c in process_counts.items() if c >= min_samples][:6]

        if not valid_processes:
            return self._create_empty_plot(f'Нет процессов с достаточным количеством данных')

        n_processes = len(valid_processes)
        n_projections = 3
        fig_width = 4.0 * n_projections
        fig_height = 3.4 * n_processes

        fig = Figure(figsize=(fig_width, fig_height), dpi=110)

        projections = [
            ('x_mm', 'y_mm', 'X/Y', 'X (мм)', 'Y (мм)'),
            ('x_mm', 'z_mm', 'X/Z', 'X (мм)', 'Z (мм)'),
            ('y_mm', 'z_mm', 'Y/Z', 'Y (мм)', 'Z (мм)')
        ]

        plot_idx = 1

        for proc_idx, process in enumerate(valid_processes):
            process_df = df[df['process'] == process]
            if len(process_df) < min_samples:
                continue

            fig.text(0.02, 0.96 - proc_idx * (0.94 / n_processes),
                     process, va='center', ha='left',
                     fontsize=11, fontweight='bold', color='#222')

            df_clean = process_df.copy()

            for proj_idx, (x_col, y_col, proj_title, xlabel, ylabel) in enumerate(projections):
                ax = fig.add_subplot(n_processes, n_projections, plot_idx)
                plot_idx += 1

                x = df_clean[x_col].values
                y = df_clean[y_col].values

                if heatmap_mode == "counts":
                    weights = np.ones(len(x))
                    cbar_label = "Плотность шагов"
                    cmap = 'magma'
                elif heatmap_mode == "dE":
                    if energy_column not in df_clean.columns:
                        ax.text(0.5, 0.5, 'нет dE', ha='center', va='center', fontsize=8)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue
                    weights = df_clean[energy_column].values
                    if unit.lower() == "kev":
                        weights *= 1e3
                        cbar_label = "Плотность энерговыделения (keV / mm²)"
                    else:
                        cbar_label = "Плотность энерговыделения (MeV / mm²)"
                    cmap = 'hot'

                if len(x) < 2 or len(y) < 2:
                    ax.text(0.5, 0.5, 'мало данных', ha='center', va='center', fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                x_min, x_max = limits[x_col[0]]
                y_min, y_max = limits[y_col[0]]

                grid_size = 100  # Немного меньше для сводной карты
                xi = np.linspace(x_min, x_max, grid_size)
                yi = np.linspace(y_min, y_max, grid_size)
                xx, yy = np.meshgrid(xi, yi)

                try:
                    if heatmap_mode == "counts":
                        zi = self._adaptive_density_field(x, y, xx, yy, weights=weights, normalize_for_density=True)
                    else:
                        zi = self._adaptive_density_field(x, y, xx, yy, weights=weights, normalize_for_density=False)

                    if zi is None:
                        ax.text(0.5, 0.5, 'мало данных', ha='center', va='center', fontsize=8)
                        continue

                    # Построение контурной карты
                    im = ax.contourf(xi, yi, zi, levels=40, cmap=cmap)
                    ax.contour(xi, yi, zi, levels=20, colors='black', linewidths=0.3, alpha=0.3)

                    ax.set_title(proj_title, fontsize=9, pad=4)
                    ax.set_xlabel(xlabel, fontsize=8)
                    ax.set_ylabel(ylabel, fontsize=8)
                    ax.tick_params(labelsize=7.5)
                    ax.grid(False)

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="4%", pad=0.08)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.set_label(cbar_label, fontsize=8)
                    cbar.ax.tick_params(labelsize=6.5)

                except Exception as e:
                    print(f"Ошибка в {process} {proj_title}: {e}")
                    ax.text(0.5, 0.5, 'ошибка', ha='center', va='center', fontsize=8)

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

        # Общий заголовок
        if heatmap_mode == "counts":
            main_title = f'Распределение процессов по координатам --- {particle_type} частицы'
        else:
            main_title = f'Энерговыделение процессов --- {particle_type} частицы (режим: {unit})'

        fig.suptitle(main_title, fontsize=13, fontweight='bold', y=0.99)
        fig.subplots_adjust(
            left=0.10,
            right=0.92,
            top=0.94,
            bottom=0.12,
            hspace=0.42,
            wspace=0.35
        )

        if use_cache and self.cache_mgr:
            self.cache_mgr.save_figure(fig, "process_heatmaps", cache_key)
            print(f"[CACHE SAVE] process heatmaps: {particle_type} {cache_key['data_hash'][:8]}...")

        return fig

    def _visualize_energy_comparison(self, df):
        """Сравнение энергий первичных и вторичных частиц"""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        primary_energies = df[df['is_primary']]['kinetic_energy_mev']
        secondary_energies = df[df['is_secondary']]['kinetic_energy_mev']

        # Гистограммы
        if len(primary_energies) > 0:
            ax.hist(primary_energies, bins=50, alpha=0.5, label='Первичные',
                    color='blue', density=True)
        if len(secondary_energies) > 0:
            ax.hist(secondary_energies, bins=50, alpha=0.5, label='Вторичные',
                    color='red', density=True)

        ax.set_xlabel('Кинетическая энергия (МэВ)')
        ax.set_ylabel('Нормализованная частота')
        ax.set_title('Сравнение распределений энергий первичных и вторичных частиц')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Статистика на графике
        stats_text = []
        if len(primary_energies) > 0:
            stats_text.append(f"Первичные: n={len(primary_energies)}")
            stats_text.append(f"μ={primary_energies.mean():.2f} МэВ")
        if len(secondary_energies) > 0:
            stats_text.append(f"Вторичные: n={len(secondary_energies)}")
            stats_text.append(f"μ={secondary_energies.mean():.2f} МэВ")

        ax.text(0.02, 0.98, '\n'.join(stats_text),
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.tight_layout()
        return fig

    def _is_process_summary_line(self, line):
        return any(phrase in line for phrase in [
            'Process calls frequency',
            'Process summary',
            'Number of process calls'
        ])

    def _parse_energy_summary(self, line):
        try:
            patterns = [
                r'Energy deposit\s*:\s*([\d.]+)\s*(\w+)',
                r'Total energy deposit\s*:\s*([\d.]+)\s*(\w+)',
                r'Energy deposition\s*:\s*([\d.]+)\s*(\w+)'
            ]
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2)
                    if unit == 'MeV':
                        energy_mev = value
                    elif unit.lower() == 'kev':
                        energy_mev = value / 1000
                    elif unit.lower() == 'ev':
                        energy_mev = value / 1e6
                    elif unit == 'meV':
                        energy_mev = value / 1e9
                    elif unit.lower() == 'gev':
                        energy_mev = value * 1000
                    else:
                        energy_mev = value
                    self.summary_data['total_energy_deposit_mev'] = energy_mev
                    break
        except Exception as e:
            print(f"Ошибка парсинга итоговой энергии: {e}")

    def _parse_process_summary(self, lines, current_index):
        try:
            process_calls = {}
            i = current_index + 1
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith('---') or line.startswith('==='):
                    break
                process_matches = re.findall(r'([\w-]+)\s*=\s*(\d+)', line)
                for process_name, calls_str in process_matches:
                    calls = int(calls_str)
                    process_calls[process_name] = calls
                i += 1
            if process_calls:
                self.summary_data['process_calls'] = process_calls
            return i - 1
        except Exception as e:
            print(f"Ошибка парсинга секции процессов: {e}")
            return current_index

    def _analyze_and_compare(self, df):
        result = []
        if not df.empty:
            total_parsed_energy = df['energy_loss_mev'].sum() / self.particle_count
            result.append(f"\n1. СУММАРНЫЕ ПОТЕРИ ЭНЕРГИИ ИЗ ДАННЫХ ШАГОВ:")
            result.append(f"   Total dEStep (parsed): {total_parsed_energy:.6f} МэВ")
            result.append(f"\n2. ПОТЕРИ ЭНЕРГИИ ПО ТИПАМ ЧАСТИЦ:")
            particle_energy = df.groupby('particle')['energy_loss_mev'].sum() / self.particle_count
            for particle, energy in particle_energy.items():
                result.append(f"   {particle}: {energy:.6f} МэВ")
        else:
            result.append(f"\n1. Нет данных для анализа")

        if 'total_energy_deposit_mev' in self.summary_data:
            summary_energy = self.summary_data['total_energy_deposit_mev']
            result.append(f"\n3. СРАВНЕНИЕ С ИТОГОВОЙ СВОДКОЙ:")
            result.append(f"   Energy deposit (log summary): {summary_energy:.6f} МэВ")
            if not df.empty:
                result.append(f"   Energy deposit (parsed steps): {total_parsed_energy:.6f} МэВ")
                absolute_diff = abs(total_parsed_energy - summary_energy)
                relative_diff = (absolute_diff / summary_energy * 100) if summary_energy != 0 else 0
                result.append(f"\n4. РАСХОЖДЕНИЯ:")
                result.append(f"   Абсолютная разница: {absolute_diff:.6f} МэВ")
                result.append(f"   Относительная разница: {relative_diff:.6f} %")
        else:
            result.append(f"\nИтоговая энергия в сводке не найдена!")

        if 'process_calls' in self.summary_data and not df.empty:
            result.append(self._compare_process_frequency(df))

        return "\n".join(result)

    def _compare_process_frequency(self, df):
        result = []
        non_physical_processes = ['initStep', 'OutOfWorld']
        parsed_processes = df['process'].value_counts().to_dict()
        parsed_counts = {k: v for k, v in parsed_processes.items()
                         if k not in non_physical_processes}
        summary_processes = self.summary_data['process_calls']
        result.append("Сравнение частоты процессов:")
        result.append("=" * 60)
        result.append(f" {'Процесс':<20} {'Парсинг':<10} {'Сводка':<10} {'Разница':<10}")
        result.append("-" * 60)
        all_processes = set(parsed_counts.keys()) | set(summary_processes.keys())
        for process in sorted(all_processes):
            if process == 'Transportation':
                continue
            parsed_count = parsed_counts.get(process, 0)
            summary_count = summary_processes.get(process, 0)
            diff = parsed_count - summary_count
            result.append(f" {process:<20} {parsed_count:<10} {summary_count:<10} {diff:<10}")
        return "\n".join(result) + "\n\n"

    def _is_step_data_line(self, line):
        if "Step#" in line:
            return False
        if "G4Track Information" in line:
            return False
        if "List of secondaries" in line:
            return False
        if "*" in line and "***********************************" in line:
            return False
        if "Track (trackID" in line:
            return False
        return STEP_LINE_RE.match(re.sub(r'^G4WT\d+\s*\>\s*', '', line).strip()) is not None

    def _parse_particle_info(self, line):
        try:
            patterns = [
                r'Particle\s*=\s*([\w+-]+)',
                r'Particle:?\s*([\w+-]+)',
                r'particle\s*=\s*([\w+-]+)',
                r'Primary particle:\s*([\w+-]+)',
                r'### Storing a track \(([\w+-]+),trackID=(\d+),parentID=(\d+)\)'  # НОВЫЙ паттерн!
            ]

            particle = None
            track_id = None
            parent_id = 0

            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if 'Storing' in pattern:
                        # Извлекаем из строки создания трека
                        raw_particle = match.group(1)
                        track_id = int(match.group(2))
                        parent_id = int(match.group(3))
                    else:
                        raw_particle = match.group(1)

                    if raw_particle == 'e-':
                        particle = 'electron'
                    elif raw_particle == 'e+':
                        particle = 'positron'
                    elif raw_particle.lower() in ['gamma', 'γ']:
                        particle = 'gamma'
                    else:
                        particle = raw_particle
                    break

            # Если не нашли в строках создания, ищем track_id отдельно
            if track_id is None:
                track_patterns = [
                    r'Track\s*ID\s*=\s*(\d+)',
                    r'Track\s*ID:?\s*(\d+)',
                    r'track\s*id\s*=\s*(\d+)'
                ]
                for pattern in track_patterns:
                    track_match = re.search(pattern, line, re.IGNORECASE)
                    if track_match:
                        track_id = int(track_match.group(1))
                        break

            # Если не нашли parent_id в строках создания, ищем отдельно
            if parent_id == 0 and 'Parent ID' in line:
                parent_patterns = [
                    r'Parent\s*ID\s*=\s*(\d+)',
                    r'Parent\s*ID:?\s*(\d+)',
                    r'parent\s*id\s*=\s*(\d+)',
                    r'Parent\s*=\s*(\d+)'
                ]
                for pattern in parent_patterns:
                    parent_match = re.search(pattern, line, re.IGNORECASE)
                    if parent_match:
                        parent_id = int(parent_match.group(1))
                        break

            # ОТЛАДКА: выводим найденные значения
            # print(f"[DEBUG PARSE] line: {line[:50]}...")
            # print(f" -> particle: {particle}, track_id: {track_id}, parent_id: {parent_id}")

            return {
                'particle': particle,
                'track_id': track_id,
                'parent_id': parent_id
            }
        except Exception as e:
            print(f"Ошибка парсинга информации о частице: {e}")
            print(f"Строка: {line}")
            return None

    def analyze_classification(self, df):
        """Анализ правильности классификации первичных/вторичных частиц"""
        # print("\n=== АНАЛИЗ ПРАВИЛЬНОСТИ КЛАССИФИКАЦИИ ===")
        # 1. Проверяем, что первичные частицы имеют parent_id=0
        primaries = df[df['is_primary']]
        secondaries = df[df['is_secondary']]
        print(f"Всего первичных: {len(primaries)}")
        print(f"Всего вторичных: {len(secondaries)}")

        # 2. Проверяем parent_id
        # print(f"\nПроверка parent_id у первичных:")
        # if not primaries.empty:
        #     parent_id_counts = primaries['parent_id'].value_counts()
        #     print(parent_id_counts)
        #
        #     if 0 not in parent_id_counts:
        #         print("ВНИМАНИЕ: Первичные частицы не имеют parent_id=0!")
        #
        # print(f"\nПроверка parent_id у вторичных:")
        # if not secondaries.empty:
        #     parent_id_counts = secondaries['parent_id'].value_counts()
        #     print(parent_id_counts.head(10))

        # 3. Проверяем энергии
        print(f"\nСравнение распределений энергий:")
        if not primaries.empty and not secondaries.empty:
            # Гистограммы энергий
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.hist(primaries['kinetic_energy_mev'].values, bins=50, alpha=0.5,
                     label='Первичные', color='blue', density=True)
            plt.hist(secondaries['kinetic_energy_mev'].values, bins=50, alpha=0.5,
                     label='Вторичные', color='red', density=True)
            plt.xlabel('Энергия (МэВ)')
            plt.ylabel('Плотность вероятности')
            plt.title('Сравнение распределений энергий')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            # QQ-plot для сравнения распределений
            from scipy import stats
            if len(primaries) > 100 and len(secondaries) > 100:
                qq_x = np.percentile(primaries['kinetic_energy_mev'].values,
                                     np.linspace(0, 100, 100))
                qq_y = np.percentile(secondaries['kinetic_energy_mev'].values,
                                     np.linspace(0, 100, 100))
                plt.scatter(qq_x, qq_y, alpha=0.5)
                plt.plot([qq_x.min(), qq_x.max()], [qq_x.min(), qq_x.max()],
                         'r--', label='y=x')
                plt.xlabel('Квантили первичных')
                plt.ylabel('Квантили вторичных')
                plt.title('QQ-plot сравнения распределений')
                plt.legend()
                plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            # Статистический тест
            if len(primaries) > 30 and len(secondaries) > 30:
                from scipy.stats import ks_2samp
                stat, p_value = ks_2samp(primaries['kinetic_energy_mev'].values,
                                         secondaries['kinetic_energy_mev'].values)
                print(f"\nТест Колмогорова-Смирнова:")
                print(f"Статистика: {stat:.4f}, p-value: {p_value:.4e}")
                if p_value < 0.05:
                    print("✓ Распределения статистически значимо различаются (ожидаемо)")
                else:
                    print("⚠️ Распределения НЕ различаются (неожиданно!)")

        # 4. Анализ типов частиц
        print(f"\nТипы первичных частиц:")
        if not primaries.empty:
            print(primaries['particle'].value_counts())
        print(f"\nТипы вторичных частиц:")
        if not secondaries.empty:
            print(secondaries['particle'].value_counts())

    def analyze_primary_secondary_misclassification(self, df):
        """Анализ неправильной классификации первичных/вторичных частиц"""
        # print("\n=== АНАЛИЗ НЕПРАВИЛЬНОЙ КЛАССИФИКАЦИИ ===")
        # 1. Частицы с parent_id=0 но track_id>1
        false_primaries = df[(df['parent_id'] == 0) & (df['track_id'] > 1)]
        print(f"\nЧастицы с parent_id=0 но track_id>1 (ложно-первичные): {len(false_primaries)}")
        if not false_primaries.empty:
            print("Распределение по track_id:")
            print(false_primaries['track_id'].value_counts().sort_index().head(10))
            print("\nРаспределение по частицам:")
            print(false_primaries['particle'].value_counts())

        # 2. Истинные первичные (track_id=1, parent_id=0)
        true_primaries = df[(df['track_id'] == 1) & (df['parent_id'] == 0)]
        print(f"\nИстинные первичные (track_id=1, parent_id=0): {len(true_primaries)}")
        if not true_primaries.empty:
            print("Типы частиц:")
            print(true_primaries['particle'].value_counts())

        # 3. Вторичные частицы (parent_id > 0)
        secondaries = df[df['parent_id'] > 0]
        print(f"\nВторичные частицы (parent_id > 0): {len(secondaries)}")
        if not secondaries.empty:
            print("Распределение по parent_id:")
            print(secondaries['parent_id'].value_counts().sort_index().head(10))
            print("\nТипы частиц:")
            print(secondaries['particle'].value_counts())

        # 4. Предлагаемое исправление
        # print("\n=== ПРЕДЛАГАЕМОЕ ИСПРАВЛЕНИЕ ===")
        # print("Первичные частицы: только те, у которых track_id=1 И parent_id=0")
        # print(f"Будет {len(true_primaries)} первичных частиц вместо {df['is_primary'].sum()}")
        # print(f"Будет {len(df) - len(true_primaries)} вторичных частиц вместо {df['is_secondary'].sum()}")

    def verify_primary_secondary_separation(self, df):
        """Проверка корректности разделения первичных/вторичных частиц"""
        # print("\n=== ПРОВЕРКА РАЗДЕЛЕНИЯ ЧАСТИЦ ===")
        # 1. Статистика по parent_id
        print(f"\nРаспределение parent_id:")
        parent_stats = df['parent_id'].value_counts().sort_index()
        for parent_id, count in parent_stats.items():
            print(f" parent_id={parent_id}: {count} частиц")

        # 2. Энергии первичных vs вторичных
        print(f"\nСравнение энергий:")
        if not df.empty:
            primary_df = df[df['is_primary']]
            secondary_df = df[df['is_secondary']]
            print(f"Первичные частицы: {len(primary_df)} записей")
            if not primary_df.empty:
                print(f" Средняя энергия: {primary_df['kinetic_energy_mev'].mean():.3f} МэВ")
                print(f" Макс энергия: {primary_df['kinetic_energy_mev'].max():.3f} МэВ")
            print(f"\nВторичные частицы: {len(secondary_df)} записей")
            if not secondary_df.empty:
                print(f" Средняя энергия: {secondary_df['kinetic_energy_mev'].mean():.3f} МэВ")
                print(f" Макс энергия: {secondary_df['kinetic_energy_mev'].max():.3f} МэВ")

            # Проверка: вторичные не должны иметь энергию выше первичных
            if not primary_df.empty:
                max_primary_energy = primary_df['kinetic_energy_mev'].max()
                high_energy_secondaries = secondary_df[
                    secondary_df['kinetic_energy_mev'] > max_primary_energy * 0.5]
                if len(high_energy_secondaries) > 0:
                    print(f"\n⚠️ ВНИМАНИЕ: Найдены {len(high_energy_secondaries)} вторичных частиц")
                    print(f" с энергией > 50% от максимальной первичной ({max_primary_energy:.3f} МэВ)")
                    print(" Проверьте парсинг parent_id!")

    def _get_thread_from_line(self, line):
        m = THREAD_PREFIX_RE.match(line)
        if m:
            return m.group(1)
        m = THREAD_ANYWHERE_RE.search(line)
        if m:
            return m.group(1)
        return None

    def _convert_units(self, value, unit, quantity_type):
        original_unit = unit
        unit = unit.lower()
        if quantity_type == 'energy':
            if original_unit == 'MeV':
                return value
            elif unit in ['kev']:
                return value / 1000
            elif unit in ['ev']:
                return value / 1e6
            elif original_unit == 'meV':
                return value / 1e9
            elif unit in ['gev']:
                return value * 1000
        elif quantity_type == 'length':
            if unit in ['mm']:
                return value
            elif unit in ['cm']:
                return value * 10
            elif unit in ['m']:
                return value * 1000
            elif unit in ['um', 'µm']:
                return value / 1000
            elif unit in ['fm']:
                return value * 1e-12
            elif unit == "ang":
                return value * 1e-7  # Å → mm
            elif unit == "fm":
                return value * 1e-12  # fm → mm
            elif unit == "pm":
                return value * 1e-9  # pm → mm
            elif unit == "nm":
                return value * 1e-6  # nm → mm
        return value


# ==================== DIMENSIONS ====================

class MaterialDimensions:
    """Класс для управления размерами материала и конвертации единиц"""

    UNIT_CONVERSION = {
        'нм': 1e-6,  # нанометры в миллиметры
        'мкм': 1e-3,  # микрометры в миллиметры
        'мм': 1.0,  # миллиметры
        'см': 10.0,  # сантиметры в миллиметры
        'м': 1000.0,  # метры в миллиметры
    }

    def __init__(self):
        self.size_x = 10.0  # мм по умолчанию
        self.size_y = 10.0  # мм по умолчанию
        self.size_z = 10.0  # мм по умолчанию
        self.unit = 'мм'  # единица измерения по умолчанию

    def set_dimensions(self, size_x, size_y, size_z, unit='мм'):
        """Установить размеры материала"""
        self.unit = unit
        self.size_x = self.convert_to_mm(size_x, unit)
        self.size_y = self.convert_to_mm(size_y, unit)
        self.size_z = self.convert_to_mm(size_z, unit)

    def set_cube(self, side_length, unit='мм'):
        """Установить кубический материал"""
        self.set_dimensions(side_length, side_length, side_length, unit)

    def convert_to_mm(self, value, from_unit):
        """Конвертировать значение в миллиметры"""
        if from_unit in self.UNIT_CONVERSION:
            return value * self.UNIT_CONVERSION[from_unit]
        return value

    def convert_from_mm(self, value, to_unit):
        """Конвертировать значение из миллиметров в указанные единицы"""
        if to_unit in self.UNIT_CONVERSION:
            return value / self.UNIT_CONVERSION[to_unit]
        return value

    def get_limits(self):
        """Получить пределы координат (предполагаем, что центр в 0,0,0)"""
        return {
            'x': (-self.size_x / 2, self.size_x / 2),
            'y': (-self.size_y / 2, self.size_y / 2),
            'z': (-self.size_z / 2, self.size_z / 2)
        }

    def format_coordinate(self, value, unit=None):
        """Форматировать координату с указанием единиц"""
        if unit is None:
            unit = self.unit
        converted = self.convert_from_mm(value, unit)

        # Определяем префикс для компактного представления
        if unit == 'м':
            if abs(converted) >= 1:
                return f"{converted:.2f} м"
            elif abs(converted) >= 0.01:
                return f"{converted * 100:.1f} см"
            else:
                return f"{converted * 1000:.1f} мм"
        elif unit == 'см':
            if abs(converted) >= 100:
                return f"{converted / 100:.2f} м"
            elif abs(converted) >= 1:
                return f"{converted:.1f} см"
            else:
                return f"{converted * 10:.1f} мм"
        elif unit == 'мм':
            if abs(converted) >= 1000:
                return f"{converted / 1000:.2f} м"
            elif abs(converted) >= 100:
                return f"{converted / 100:.1f} см"
            elif abs(converted) >= 1:
                return f"{converted:.1f} мм"
            else:
                return f"{converted * 1000:.1f} мкм"
        elif unit == 'мкм':
            if abs(converted) >= 1000:
                return f"{converted / 1000:.2f} мм"
            elif abs(converted) >= 1:
                return f"{converted:.0f} мкм"
            else:
                return f"{converted * 1000:.0f} нм"
        elif unit == 'нм':
            if abs(converted) >= 1000:
                return f"{converted / 1000:.2f} мкм"
            else:
                return f"{converted:.0f} нм"
        return f"{converted:.3f} {unit}"

    def get_size_string(self):
        """Получить строковое представление размеров"""
        if abs(self.size_x - self.size_y) < 1e-6 and abs(self.size_y - self.size_z) < 1e-6:
            # Кубический материал
            return f"{self.format_coordinate(self.size_x)} куб"
        else:
            # Прямоугольный материал
            return f"{self.format_coordinate(self.size_x)} × {self.format_coordinate(self.size_y)} × {self.format_coordinate(self.size_z)}"


# ==================== ОСНОВНОЕ ОКНО ПРИЛОЖЕНИЯ ====================

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Geant4 Log Parser (Tkinter)")
        self.geometry("1400x900")
        self.parser = Parser()
        self.df = None
        self.filtered_df = None
        self.current_figure = None
        self.current_canvas = None
        self.current_toolbar = None
        self.legend_visible = tk.BooleanVar(value=True)
        self.stats_visible = tk.BooleanVar(value=True)
        self.current_legend = None
        self.current_stats = None
        self.selected_processes = []
        self.process_checkboxes = {}
        self.cache_mgr = CacheManager()
        self.parser.cache_mgr = self.cache_mgr
        self.show_kde = tk.BooleanVar(value=True)
        self.cache_enabled = tk.BooleanVar(value=True)
        self.material_dimensions = MaterialDimensions()
        self.filter_secondary_first_step = tk.BooleanVar(value=False)
        self.selected_particles_dist = []
        self.particle_vars_dist = {}
        self.normalization_mode = tk.StringVar(value='none')  # 'none', 'particles', 'steps', 'density'
        self.xlim_min = tk.StringVar()
        self.xlim_max = tk.StringVar()
        self.ylim_min = tk.StringVar()
        self.ylim_max = tk.StringVar()
        self.apply_limits_to_plot = tk.BooleanVar(value=False)
        self.data_consistency_debug = tk.BooleanVar(value=False)
        self.heatmap_mode_var = tk.StringVar(value="counts")  # counts | dE
        self.heatmap_unit_var = tk.StringVar(value="MeV")  # MeV | keV

        # ===== НОВЫЕ ПЕРЕМЕННЫЕ ДЛЯ ДОЗЫ =====
        self.dose_unit = tk.StringVar(value="Gy")  # Gy | rad
        self.dose_per_layer = tk.BooleanVar(value=False)  # показывать по слоям
        # ===== КОНЕЦ НОВЫХ ПЕРЕМЕННЫХ =====

        self._build_ui()

    # ==================== UI ====================

    def _build_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.file_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)
        self.summary_tab = ttk.Frame(self.notebook)
        self.export_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.file_tab, text="Загрузка файла")
        self.notebook.add(self.analysis_tab, text="Анализ")
        self.notebook.add(self.summary_tab, text="Сводка")
        self.notebook.add(self.export_tab, text="Экспорт")

        self._build_file_tab()
        self._build_analysis_tab()
        self._build_summary_tab()
        self._build_export_tab()

    # ==================== FILE TAB ====================

    def _build_file_tab(self):
        frame = self.file_tab

        ttk.Button(frame, text="📁 Загрузить файл (лог/CSV)", command=self.smart_load_file).pack(pady=10)
        # ttk.Button(frame, text="📊 Загрузить CSV данные",
        #              command=self.load_csv_file, width=20).pack(side="left", padx=5)
        ttk.Button(frame, text="🔍 Проверить классификацию",
                   command=self.show_classification_check_plot).pack(pady=5)

        self.file_label = ttk.Label(frame, text="Файл не загружен")
        self.file_label.pack()

        filter_box = ttk.LabelFrame(frame, text="Фильтры")
        filter_box.pack(fill="x", padx=10, pady=10)

        # Тип частицы
        row = ttk.Frame(filter_box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Тип частицы:").pack(side="left")
        self.particle_combo = ttk.Combobox(row, state="disabled")
        self.particle_combo.pack(side="left", padx=5)

        # Категория
        row = ttk.Frame(filter_box)
        row.pack(fill="x", pady=2)
        self.category = tk.StringVar(value="all")
        ttk.Radiobutton(row, text="Все", variable=self.category, value="all").pack(side="left")
        ttk.Radiobutton(row, text="Первичные", variable=self.category, value="primary").pack(side="left")
        ttk.Radiobutton(row, text="Вторичные", variable=self.category, value="secondary").pack(side="left")

        # Энергия
        row = ttk.Frame(filter_box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Энергия (MeV): от").pack(side="left")
        self.energy_min = ttk.Entry(row, width=10)
        self.energy_min.pack(side="left", padx=2)
        ttk.Label(row, text="до").pack(side="left")
        self.energy_max = ttk.Entry(row, width=10)
        self.energy_max.pack(side="left", padx=2)

        # Кнопки
        row = ttk.Frame(filter_box)
        row.pack(fill="x", pady=5)
        ttk.Button(row, text="Применить", command=self.apply_filters).pack(side="left", padx=5)
        ttk.Button(row, text="Сбросить", command=self.reset_filters).pack(side="left")

        # Статистика
        stats_box = ttk.LabelFrame(frame, text="Статистика")
        stats_box.pack(fill="both", expand=True, padx=10, pady=10)
        self.stats_text = tk.Text(stats_box, height=15)
        self.stats_text.pack(fill="both", expand=True)

        ttk.Button(frame, text="🔍 Отладка классификации",
                   command=self.debug_classification).pack(pady=5)

        # Добавляем блок для размеров материала
        size_frame = ttk.LabelFrame(self.file_tab, text="Размеры материала (куб)")
        size_frame.pack(fill="x", padx=10, pady=10)

        # Поля для ввода размеров
        ttk.Label(size_frame, text="Длина стороны (a):").grid(row=0, column=0, padx=5, pady=5)
        self.material_size_entry = ttk.Entry(size_frame, width=10)
        self.material_size_entry.insert(0, "10")  # значение по умолчанию
        self.material_size_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(size_frame, text="Единицы:").grid(row=0, column=2, padx=5, pady=5)
        self.material_units_combo = ttk.Combobox(size_frame,
                                                 values=["нм", "мкм", "мм", "см"],
                                                 state="readonly",
                                                 width=5)
        self.material_units_combo.set("мм")
        self.material_units_combo.grid(row=0, column=3, padx=5, pady=5)

        ttk.Button(size_frame, text="Применить",
                   command=self.apply_material_size).grid(row=0, column=4, padx=5, pady=5)

        # Checkbox для ограничения осей размерами материала
        self.limit_axes = tk.BooleanVar(value=False)
        ttk.Checkbutton(size_frame,
                        text="Ограничить графики размерами материала",
                        variable=self.limit_axes).grid(row=1, column=0, columnspan=5, pady=5)

    # ==================== ANALYSIS TAB ====================

    def _build_analysis_tab(self):
        main_container = ttk.Frame(self.analysis_tab)
        main_container.pack(fill="both", expand=True)

        # === ЛЕВАЯ ПАНЕЛЬ С ПРОКРУТКОЙ ===
        # Создаем Canvas и Scrollbar
        left_container = ttk.Frame(main_container)
        left_container.pack(side="left", fill="y", padx=5, pady=5)

        # Canvas для прокрутки
        canvas = tk.Canvas(left_container, width=320, height=700, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)

        # Связываем canvas и scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)

        # Упаковываем scrollbar и canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Фрейм внутри canvas (это будет наша прокручиваемая область)
        left = ttk.Frame(canvas)

        # Создаем окно в canvas для нашего фрейма
        canvas_window = canvas.create_window((0, 0), window=left, anchor="nw", width=300)

        # Функция для обновления scrollregion при изменении размера фрейма
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        left.bind("<Configure>", on_frame_configure)

        # Функция для настройки ширины окна canvas при изменении размера
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", on_canvas_configure)

        # === ПРАВАЯ ПАНЕЛЬ ДЛЯ ГРАФИКОВ ===
        right = ttk.Frame(main_container)
        right.pack(side="right", fill="both", expand=True)

        # === БЛОК ПЕРВИЧНЫХ ЧАСТИЦ ===
        ttk.Label(left, text="Первичные", font=("", 11, "bold")).pack(pady=(0, 5))

        primary_frame = ttk.Frame(left)
        primary_frame.pack(fill="x", pady=(0, 15))

        ttk.Button(primary_frame, text="Распределение энергий",
                   command=lambda: self.plot_energy(True)).pack(fill="x", pady=1)

        ttk.Button(primary_frame, text="Потери энергии",
                   command=lambda: self.plot_loss(True)).pack(fill="x", pady=1)

        ttk.Button(primary_frame, text="dE распределение",
                   command=lambda: self.plot_dE(True)).pack(fill="x", pady=1)

        ttk.Button(primary_frame, text="Частота процессов",
                   command=lambda: self.plot_process(True)).pack(fill="x", pady=1)

        ttk.Button(primary_frame, text="Тепловая карта координат",
                   command=lambda: self.plot_heatmap(True)).pack(fill="x", pady=1)

        ttk.Button(primary_frame, text="Тепловая карта процессов",
                   command=lambda: self.plot_process_heatmap_with_selection(True)).pack(fill="x", pady=1)

        ttk.Button(primary_frame, text="2D траектории",
                   command=lambda: self.plot_2d(True)).pack(fill="x", pady=1)

        ttk.Button(primary_frame, text="3D траектории",
                   command=lambda: self.plot_3d(True)).pack(fill="x", pady=1)

        ttk.Separator(left).pack(fill="x", pady=10)

        # === БЛОК ВТОРИЧНЫХ ЧАСТИЦ ===
        ttk.Label(left, text="Вторичные", font=("", 11, "bold")).pack(pady=(0, 5))

        secondary_frame = ttk.Frame(left)
        secondary_frame.pack(fill="x", pady=(0, 15))

        ttk.Button(secondary_frame, text="Распределение энергий",
                   command=lambda: self.plot_energy(False)).pack(fill="x", pady=1)

        ttk.Button(secondary_frame, text="Потери энергии",
                   command=lambda: self.plot_loss(False)).pack(fill="x", pady=1)

        ttk.Button(secondary_frame, text="dE распределение",
                   command=lambda: self.plot_dE(False)).pack(fill="x", pady=1)

        ttk.Button(secondary_frame, text="Частота процессов",
                   command=lambda: self.plot_process(False)).pack(fill="x", pady=1)

        ttk.Button(secondary_frame, text="Тепловая карта координат",
                   command=lambda: self.plot_heatmap(False)).pack(fill="x", pady=1)

        ttk.Button(secondary_frame, text="Тепловая карта процессов",
                   command=lambda: self.plot_process_heatmap_with_selection(False)).pack(fill="x", pady=1)

        ttk.Button(secondary_frame, text="2D траектории",
                   command=lambda: self.plot_2d(False)).pack(fill="x", pady=1)

        ttk.Button(secondary_frame, text="3D траектории",
                   command=lambda: self.plot_3d(False)).pack(fill="x", pady=1)

        ttk.Separator(left).pack(fill="x", pady=10)

        # === БЛОК ЭНЕРГОВЫДЕЛЕНИЯ ===
        ttk.Label(left, text="Энерговыделение",
                  font=("", 11, "bold")).pack(pady=(0, 5))

        edep_frame = ttk.Frame(left)
        edep_frame.pack(fill="x", pady=(0, 15))

        ttk.Button(edep_frame, text="Суммарное распределение энергии по процессам",
                   command=lambda: self.plot_process_energy_distribution(True)).pack(fill="x", pady=1)

        ttk.Separator(left).pack(fill="x", pady=10)

        # === БЛОК ВЫБОРА ЧАСТИЦ ДЛЯ РАСПРЕДЕЛЕНИЙ ===
        ttk.Label(left, text="Выбор частиц для распределений и 3D",
                  font=("", 11, "bold")).pack(pady=(0, 5))

        particle_select_frame = ttk.LabelFrame(left, text="Вторичные частицы")
        particle_select_frame.pack(fill="x", pady=5, padx=5)

        # Кнопки управления
        btn_frame = ttk.Frame(particle_select_frame)
        btn_frame.pack(fill="x", pady=2)

        ttk.Button(btn_frame, text="Выбрать все",
                   command=self.select_all_dist_particles).pack(side="left", padx=2)

        ttk.Button(btn_frame, text="Сбросить",
                   command=self.deselect_all_dist_particles).pack(side="left", padx=2)

        ttk.Button(btn_frame, text="Обновить список",
                   command=self.update_dist_particle_list).pack(side="left", padx=2)

        # Фрейм для чекбоксов с прокруткой
        self.particle_dist_canvas = tk.Canvas(particle_select_frame, height=150)
        scrollbar = ttk.Scrollbar(particle_select_frame, orient="vertical",
                                  command=self.particle_dist_canvas.yview)

        self.particle_dist_scroll_frame = ttk.Frame(self.particle_dist_canvas)
        self.particle_dist_scroll_frame.bind(
            "<Configure>",
            lambda e: self.particle_dist_canvas.configure(scrollregion=self.particle_dist_canvas.bbox("all"))
        )

        self.particle_dist_canvas.create_window((0, 0), window=self.particle_dist_scroll_frame, anchor="nw")
        self.particle_dist_canvas.configure(yscrollcommand=scrollbar.set)

        self.particle_dist_canvas.pack(side="left", fill="both", expand=True, padx=(5, 0))
        scrollbar.pack(side="right", fill="y")

        ttk.Separator(left).pack(fill="x", pady=10)

        # === НАСТРОЙКИ НОРМИРОВКИ ===
        ttk.Label(left, text="Режим нормировки", font=("", 11, "bold")).pack(pady=(0, 5))

        norm_frame = ttk.Frame(left)
        norm_frame.pack(fill="x", pady=5)

        # Радиокнопки для выбора режима нормировки
        self.normalization_mode = tk.StringVar(value='none')

        ttk.Radiobutton(norm_frame, text="Без нормировки (число шагов)",
                        variable=self.normalization_mode, value='none').pack(anchor="w", pady=2)

        ttk.Radiobutton(norm_frame, text="На частицу (шагов/частица)",
                        variable=self.normalization_mode, value='particles').pack(anchor="w", pady=2)

        ttk.Radiobutton(norm_frame, text="На долю шагов",
                        variable=self.normalization_mode, value='steps').pack(anchor="w", pady=2)

        ttk.Radiobutton(norm_frame, text="Плотность вероятности",
                        variable=self.normalization_mode, value='density').pack(anchor="w", pady=2)

        ttk.Separator(left).pack(fill="x", pady=10)

        # === НАСТРОЙКИ HEATMAP ===
        ttk.Label(left, text="Heatmap", font=("", 11, "bold")).pack(pady=(0, 5))

        heatmap_frame = ttk.Frame(left)
        heatmap_frame.pack(fill="x", pady=5)

        ttk.Radiobutton(
            heatmap_frame,
            text="Плотность шагов",
            variable=self.heatmap_mode_var,
            value="counts",
            command=self._update_heatmap_controls
        ).pack(anchor="w", pady=1)

        ttk.Radiobutton(
            heatmap_frame,
            text="Энерговыделение (dE)",
            variable=self.heatmap_mode_var,
            value="dE",
            command=self._update_heatmap_controls
        ).pack(anchor="w", pady=1)

        self.energy_unit_frame = ttk.Frame(left)
        self.energy_unit_frame.pack(fill="x", padx=15, pady=(0, 5))

        ttk.Label(self.energy_unit_frame, text="Единицы dE").pack(anchor="w")
        ttk.Radiobutton(
            self.energy_unit_frame,
            text="MeV",
            variable=self.heatmap_unit_var,
            value="MeV"
        ).pack(anchor="w")
        ttk.Radiobutton(
            self.energy_unit_frame,
            text="keV",
            variable=self.heatmap_unit_var,
            value="keV"
        ).pack(anchor="w")

        # ===== НОВЫЙ БЛОК: Дозовые карты =====
        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Label(left, text="Дозовые карты", font=("", 11, "bold")).pack(pady=(0, 5))

        dose_frame = ttk.Frame(left)
        dose_frame.pack(fill="x", pady=(0, 15))

        # Переключатель единиц дозы
        unit_frame = ttk.Frame(dose_frame)
        unit_frame.pack(fill="x", pady=2)
        ttk.Label(unit_frame, text="Единицы:").pack(side="left")
        self.dose_unit = tk.StringVar(value="Gy")
        ttk.Radiobutton(unit_frame, text="Грей", variable=self.dose_unit, value="Gy").pack(side="left", padx=5)
        ttk.Radiobutton(unit_frame, text="Рад", variable=self.dose_unit, value="rad").pack(side="left", padx=5)

        # Переключатель режима отображения
        layer_frame = ttk.Frame(dose_frame)
        layer_frame.pack(fill="x", pady=2)
        ttk.Label(layer_frame, text="Режим:").pack(side="left")
        self.dose_per_layer = tk.BooleanVar(value=False)
        ttk.Radiobutton(layer_frame, text="Суммарно", variable=self.dose_per_layer, value=False).pack(side="left",
                                                                                                      padx=5)
        ttk.Radiobutton(layer_frame, text="По слоям", variable=self.dose_per_layer, value=True).pack(side="left",
                                                                                                     padx=5)

        # Переключатель типа дозовой карты
        dose_type_frame = ttk.Frame(dose_frame)
        dose_type_frame.pack(fill="x", pady=2)
        ttk.Label(dose_type_frame, text="Тип дозовой карты:").pack(anchor="w")
        self.dose_type = tk.StringVar(value="all")
        ttk.Radiobutton(dose_type_frame, text="Дозовая карта (все частицы)", variable=self.dose_type, value="all",
                        command=lambda: self.plot_dose_map(primary=None, per_layer=self.dose_per_layer.get())).pack(
            anchor="w", pady=2)
        ttk.Radiobutton(dose_type_frame, text="Доза (первичные)", variable=self.dose_type, value="primary",
                        command=lambda: self.plot_dose_map(primary=True, per_layer=self.dose_per_layer.get())).pack(
            anchor="w", pady=2)
        ttk.Radiobutton(dose_type_frame, text="Доза (вторичные)", variable=self.dose_type, value="secondary",
                        command=lambda: self.plot_dose_map(primary=False, per_layer=self.dose_per_layer.get())).pack(
            anchor="w", pady=2)
        # ===== КОНЕЦ НОВОГО БЛОКА =====

        # =========================================
        ttk.Label(left, text="Лимиты осей", font=("", 11, "bold")).pack(pady=(0, 5))

        limits_frame = ttk.LabelFrame(left, text="Ограничения осей графиков")
        limits_frame.pack(fill="x", pady=5, padx=5)

        # Флажок применения лимитов
        ttk.Checkbutton(limits_frame, text="Применить лимиты к графикам",
                        variable=self.apply_limits_to_plot).pack(anchor="w", pady=2)

        # Поля для X
        x_frame = ttk.Frame(limits_frame)
        x_frame.pack(fill="x", pady=2)
        ttk.Label(x_frame, text="X min:").pack(side="left")
        ttk.Entry(x_frame, textvariable=self.xlim_min, width=10).pack(side="left", padx=2)
        ttk.Label(x_frame, text="X max:").pack(side="left")
        ttk.Entry(x_frame, textvariable=self.xlim_max, width=10).pack(side="left", padx=2)

        # Поля для Y
        y_frame = ttk.Frame(limits_frame)
        y_frame.pack(fill="x", pady=2)
        ttk.Label(y_frame, text="Y min:").pack(side="left")
        ttk.Entry(y_frame, textvariable=self.ylim_min, width=10).pack(side="left", padx=2)
        ttk.Label(y_frame, text="Y max:").pack(side="left")
        ttk.Entry(y_frame, textvariable=self.ylim_max, width=10).pack(side="left", padx=2)

        # Кнопки управления
        btn_frame = ttk.Frame(limits_frame)
        btn_frame.pack(fill="x", pady=5)

        ttk.Button(btn_frame, text="Применить к текущему",
                   command=self.apply_limits_current).pack(side="left", padx=2)

        ttk.Button(btn_frame, text="Авто",
                   command=self.reset_limits).pack(side="left", padx=2)

        # =========================================================

        ttk.Checkbutton(
            left, text="Только первый шаг для вторичных",
            variable=self.filter_secondary_first_step,
            command=self.update_secondary_filter
        ).pack(anchor="w", pady=1)

        self.exclude_transport = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Исключить транспортные процессы",
                        variable=self.exclude_transport,
                        command=self.update_exclude_transport).pack(anchor="w")

        # === НАСТРОЙКИ ===
        ttk.Label(left, text="Настройки", font=("", 11, "bold")).pack(pady=(0, 5))

        ttk.Checkbutton(
            left, text="Легенда",
            variable=self.legend_visible,
            command=self.toggle_legend
        ).pack(anchor="w", pady=1)

        ttk.Checkbutton(
            left, text="Статистика",
            variable=self.stats_visible,
            command=self.toggle_stats
        ).pack(anchor="w", pady=1)

        ttk.Checkbutton(
            left, text="Показывать KDE",
            variable=self.show_kde,
            command=self.toggle_kde
        ).pack(anchor="w", pady=1)

        ttk.Checkbutton(
            left, text="Кэшировать графики",
            variable=self.cache_enabled
        ).pack(anchor="w", pady=1)

        ttk.Checkbutton(
            left, text="Отладка согласованности данных",
            variable=self.data_consistency_debug
        ).pack(anchor="w", pady=1)

        ttk.Separator(left).pack(fill="x", pady=10)

        # === ИНСТРУМЕНТЫ ===
        ttk.Label(left, text="Инструменты", font=("", 11, "bold")).pack(pady=(0, 5))

        # Используем сетку для размещения кнопок в 2 колонки
        tools_frame = ttk.Frame(left)
        tools_frame.pack(fill="x", pady=5)

        # Кнопки в 2 колонки
        ttk.Button(tools_frame, text="🗑 Очистить график",
                   command=self.clear_plot).grid(row=0, column=0, sticky="ew", padx=2, pady=2)

        ttk.Button(tools_frame, text="🧹 Очистить кэш",
                   command=self.clear_cache).grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        ttk.Button(tools_frame, text="🔄 Перестроить все",
                   command=self.rebuild_all_plots).grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        # Настраиваем одинаковое расширение для колонок
        tools_frame.columnconfigure(0, weight=1)
        tools_frame.columnconfigure(1, weight=1)

        # Фрейм для графиков в правой панели
        self.plot_frame = ttk.Frame(right)
        self.plot_frame.pack(fill="both", expand=True)

        # Сохраняем ссылку на canvas для прокрутки мышью
        self.left_panel_canvas = canvas
        self._bind_mouse_scroll()
        self._update_heatmap_controls()

    def _bind_mouse_scroll(self):
        """Прокрутка колесом: левый список и список частиц --- каждый сам себя крутит"""

        def bind_wheel(widget, target_canvas):
            def _on_mousewheel(event):
                # Windows / Linux (X11 в некоторых случаях тоже отдаёт delta)
                if hasattr(event, "delta") and event.delta:
                    target_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                return "break"

            def _on_button4(event):
                target_canvas.yview_scroll(-1, "units")
                return "break"

            def _on_button5(event):
                target_canvas.yview_scroll(1, "units")
                return "break"

            # Включаем скролл только когда курсор внутри widget
            def _enter(_):
                widget.bind_all("<MouseWheel>", _on_mousewheel)
                widget.bind_all("<Button-4>", _on_button4)
                widget.bind_all("<Button-5>", _on_button5)

            def _leave(_):
                widget.unbind_all("<MouseWheel>")
                widget.unbind_all("<Button-4>")
                widget.unbind_all("<Button-5>")

            widget.bind("<Enter>", _enter)
            widget.bind("<Leave>", _leave)

        # 1) Левая панель
        bind_wheel(self.left_panel_canvas, self.left_panel_canvas)
        # 2) Список частиц (внутренний)
        # важно: bind на сам canvas списка частиц
        bind_wheel(self.particle_dist_canvas, self.particle_dist_canvas)

    def _update_heatmap_controls(self):
        """Включает / выключает выбор единиц dE"""
        if self.heatmap_mode_var.get() == "dE":
            self.energy_unit_frame.pack(anchor="w", padx=12, pady=(4, 0))
        else:
            self.energy_unit_frame.forget()

    def check_data_consistency(self):
        """Проверить согласованность данных во всех компонентах"""
        if self.df is None:
            return

        # print("\n" + "=" * 60)
        # print("ПРОВЕРКА СОГЛАСОВАННОСТИ ДАННЫХ")
        # print("=" * 60)

        # 1. Основные данные
        print(f"\n1. ОСНОВНЫЕ ДАННЫЕ:")
        print(f"   Всего записей: {len(self.df)}")
        print(f"   Уникальных частиц: {self.df['particle'].nunique()}")

        # 2. Фильтрованные данные
        if self.filtered_df is not None:
            print(f"\n2. ОТФИЛЬТРОВАННЫЕ ДАННЫЕ:")
            print(f"   Записей: {len(self.filtered_df)} ({len(self.filtered_df) / len(self.df) * 100:.1f}%)")

        # 3. Данные для вторичных частиц в списке выбора
        secondary_for_list = self.df[~self.df['is_primary']]
        print(f"\n3. ДАННЫЕ ДЛЯ СПИСКА ВЫБОРА ВТОРИЧНЫХ:")
        print(f"   Всего вторичных: {len(secondary_for_list)}")

        # 4. Данные для графиков вторичных
        secondary_for_plots = self.apply_filters_to_df(self.filtered_df, primary=False)
        print(f"\n4. ДАННЫЕ ДЛЯ ГРАФИКОВ ВТОРИЧНЫХ:")
        print(f"   После всех фильтров: {len(secondary_for_plots)}")

        # 5. Сравнение по частицам
        print(f"\n5. СРАВНЕНИЕ ПО ЧАСТИЦАМ:")
        particles_in_list = secondary_for_list['particle'].unique()
        particles_in_plots = secondary_for_plots['particle'].unique()
        common_particles = set(particles_in_list) & set(particles_in_plots)
        only_in_list = set(particles_in_list) - set(particles_in_plots)
        only_in_plots = set(particles_in_plots) - set(particles_in_list)

        print(f"   Общие частицы: {len(common_particles)}")
        if only_in_list:
            print(f"   Только в списке выбора: {only_in_list}")
        if only_in_plots:
            print(f"   Только в графиках: {only_in_plots}")

        # 6. Статистика по шагам для общих частиц
        print(f"\n6. СТАТИСТИКА ШАГОВ ДЛЯ ОБЩИХ ЧАСТИЦ:")
        for particle in sorted(common_particles)[:10]:  # Первые 10
            steps_in_list = len(secondary_for_list[secondary_for_list['particle'] == particle])
            steps_in_plots = len(secondary_for_plots[secondary_for_plots['particle'] == particle])
            if steps_in_list != steps_in_plots:
                ratio = steps_in_plots / steps_in_list * 100
                print(f"   {particle:15}: список={steps_in_list:6}, графики={steps_in_plots:6} ({ratio:.1f}%)")
            else:
                print(f"   {particle:15}: {steps_in_list:6} шагов (совпадает)")

        sources = {
            "Полные данные": self.df,
            "Отфильтрованные данные": self.filtered_df,
            "Первичные": self.apply_filters_to_df(self.df, primary=True),
            "Вторичные": self.apply_filters_to_df(self.df, primary=False),
        }
        for name, df in sources.items():
            if df is not None and not df.empty:
                print(f"\n{name}:")
                print(f"   Записей: {len(df):,}")
                if 'kinetic_energy_mev' in df.columns:
                    mean_energy = df['kinetic_energy_mev'].mean()
                    print(f"   Средняя энергия: {mean_energy:.6f} МэВ")
        print("\n" + "=" * 60)

    def debug_data_sources(self):
        """Показать, какие данные где используются"""
        print("\n" + "=" * 80)
        print("ОТЛАДКА ИСТОЧНИКОВ ДАННЫХ")
        print("=" * 80)

        sources = {
            "Полные данные (self.df)": self.df,
            "Отфильтрованные данные (self.filtered_df)": self.filtered_df,
            "Для графиков первичных (get_consistent_data primary)":
                self.get_consistent_data("primary", True),
            "Для графиков вторичных (get_consistent_data secondary)":
                self.get_consistent_data("secondary", True),
            "Для списка выбора частиц (get_consistent_data secondary)":
                self.get_consistent_data("secondary", True)
        }

        for name, df in sources.items():
            if df is not None:
                print(f"\n{name}:")
                print(f"   Записей: {len(df):,}")
                if 'particle' in df.columns:
                    print(f"   Частиц: {df['particle'].nunique()}")
                    print(f"   Типы: {df['particle'].unique().tolist()[:5]}")
                if 'kinetic_energy_mev' in df.columns and len(df) > 0:
                    print(f"   Энергия: {df['kinetic_energy_mev'].mean():.6f} МэВ")
        print("=" * 80)

    def apply_limits_current(self):
        """Применить текущие лимиты к активному графику"""
        if not self.current_figure:
            return

        try:
            ax = self.current_figure.axes[0]

            # Получаем значения лимитов
            xmin = float(self.xlim_min.get()) if self.xlim_min.get() else None
            xmax = float(self.xlim_max.get()) if self.xlim_max.get() else None
            ymin = float(self.ylim_min.get()) if self.ylim_min.get() else None
            ymax = float(self.ylim_max.get()) if self.ylim_max.get() else None

            # Применяем лимиты
            if xmin is not None and xmax is not None and xmin < xmax:
                ax.set_xlim(xmin, xmax)
            elif xmin is not None:
                ax.set_xlim(left=xmin)
            elif xmax is not None:
                ax.set_xlim(right=xmax)

            if ymin is not None and ymax is not None and ymin < ymax:
                ax.set_ylim(ymin, ymax)
            elif ymin is not None:
                ax.set_ylim(bottom=ymin)
            elif ymax is not None:
                ax.set_ylim(top=ymax)

            # Перерисовываем
            self.current_canvas.draw_idle()
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные значения лимитов")

    def reset_limits(self):
        """Сбросить лимиты к автоматическим"""
        self.xlim_min.set("")
        self.xlim_max.set("")
        self.ylim_min.set("")
        self.ylim_max.set("")

        if self.current_figure:
            ax = self.current_figure.axes[0]
            ax.relim()
            ax.autoscale_view()
            self.current_canvas.draw_idle()

    def plot_process_heatmap_with_selection(self, primary):
        """Построение тепловой карты для выбранного процесса - ЕДИНЫЙ ИСТОЧНИК ДАННЫХ"""
        # Получаем отфильтрованные данные
        df = self.apply_filters_to_df(self.filtered_df, primary)
        if df.empty:
            messagebox.showinfo("Нет данных",
                                f"Нет данных для {'первичных' if primary else 'вторичных'} частиц после применения фильтров")
            return

        print(
            f"[PLOT_PROCESS_HEATMAP] Построение тепловой карты процессов для {'первичных' if primary else 'вторичных'}:")
        print(f"   Всего шагов: {len(df)}")
        print(f"   Частиц: {df['particle'].nunique()}")

        # Проверяем наличие координатных данных
        coord_columns = ['x_mm', 'y_mm', 'z_mm']
        missing_coords = [col for col in coord_columns if col not in df.columns]
        if missing_coords:
            messagebox.showwarning("Отсутствуют данные",
                                   f"Отсутствуют колонки координат: {missing_coords}\n"
                                   f"Тепловая карта не может быть построена.")
            return

        # Получаем уникальные процессы (исключаем транспортные)
        # exclude = ['OutOfWorld', 'Transportation', 'initStep', 'NoProcess', 'CoulombScat']
        exclude = ['OutOfWorld', 'Transportation', 'initStep', 'NoProcess']
        available_processes = sorted([p for p in df['process'].unique() if p not in exclude])

        if not available_processes:
            messagebox.showinfo("Нет процессов",
                                f"Нет доступных физических процессов для {'первичных' if primary else 'вторичных'} частиц "
                                f"после применения фильтров")
            return

        print(f"   Доступных процессов: {len(available_processes)}")
        print(f"   Топ-10 процессов:")
        process_counts = df['process'].value_counts()
        for process, count in process_counts.head(20).items():
            percentage = count / len(df) * 100
            print(f"     {process}: {count} шагов ({percentage:.1f}%)")

        # Создаем диалог выбора процесса
        dialog = tk.Toplevel(self)
        dialog.title(f"Выбор процесса для {'первичных' if primary else 'вторичных'} частиц")
        dialog.geometry("500x350")
        dialog.transient(self)
        dialog.grab_set()

        # Центрируем диалог относительно главного окна
        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 500) // 2
        y = self.winfo_y() + (self.winfo_height() - 350) // 2
        dialog.geometry(f"500x350+{x}+{y}")

        # Метка с информацией
        info_text = f"Выберите процесс для построения тепловой карты\nВсего шагов: {len(df):,}, Процессов: {len(available_processes)}"
        ttk.Label(dialog, text=info_text,
                  font=("", 10), justify="center").pack(pady=10)

        # Выпадающий список процессов
        selected_process = tk.StringVar()
        process_combo = ttk.Combobox(dialog, textvariable=selected_process,
                                     values=available_processes, state="readonly",
                                     width=45, height=25)  # height для выпадающего списка
        process_combo.pack(pady=10, padx=20, fill="x")

        # Информация о количестве событий
        info_frame = ttk.Frame(dialog)
        info_frame.pack(fill="x", padx=20, pady=5)
        info_label = ttk.Label(info_frame, text="")
        info_label.pack()

        # Дополнительная статистика
        stats_label = ttk.Label(info_frame, text="", font=("", 8))
        stats_label.pack()

        def update_info(*args):
            """Обновляет информацию о выбранном процессе"""
            process = selected_process.get()
            if not process:
                info_label.config(text="")
                stats_label.config(text="")
                return

            # Данные для выбранного процесса
            process_df = df[df['process'] == process]
            count = len(process_df)
            total = len(df)
            percentage = (count / total * 100) if total > 0 else 0

            # Основная информация
            info_label.config(
                text=f"Процесс '{process}': {count:,} событий из {total:,} ({percentage:.1f}%)"
            )

            # Дополнительная статистика
            if count > 0:
                particle_stats = process_df['particle'].value_counts().head(20)
                stats_text = "Частицы: "
                stats_parts = []
                for particle, pcount in particle_stats.items():
                    ppercentage = (pcount / count * 100) if count > 0 else 0
                    stats_parts.append(f"{particle} ({pcount}, {ppercentage:.1f}%)")
                if stats_parts:
                    stats_label.config(text=stats_text + ", ".join(stats_parts[:3]) +
                                            ("..." if len(stats_parts) > 3 else ""))
                else:
                    stats_label.config(text="")

        selected_process.trace("w", update_info)

        # Устанавливаем первый процесс по умолчанию
        if available_processes:
            selected_process.set(available_processes[0])
            update_info()

        # Кнопки
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)

        def build_and_close():
            """Построить график и закрыть диалог"""
            process = selected_process.get()
            if not process:
                messagebox.showwarning("Выбор процесса", "Пожалуйста, выберите процесс")
                return

            process_df = df[df['process'] == process]
            if len(process_df) < 3:
                messagebox.showwarning("Мало данных",
                                       f"Для процесса '{process}' всего {len(process_df)} событий.\n"
                                       f"Недостаточно для построения тепловой карты.")
                return

            # Логируем построение
            print(f"\n[BUILD_PROCESS_HEATMAP] Построение тепловой карты для:")
            print(f"   Процесс: {process}")
            print(f"   Событий: {len(process_df)}")
            print(f"   Частицы: {process_df['particle'].unique().tolist()}")
            print(f"   Координаты: X=[{process_df['x_mm'].min():.2f}, {process_df['x_mm'].max():.2f}], "
                  f"Y=[{process_df['y_mm'].min():.2f}, {process_df['y_mm'].max():.2f}], "
                  f"Z=[{process_df['z_mm'].min():.2f}, {process_df['z_mm'].max():.2f}]")

            dialog.destroy()

            # Строим график
            try:
                fig = self.parser._visualize_process_heatmap(
                    df,
                    "первичных" if primary else "вторичных",
                    selected_process=process,
                    heatmap_mode=self.heatmap_mode_var.get(),
                    unit=self.heatmap_unit_var.get(),
                    use_cache=self.cache_enabled.get()
                )
                self.show_plot(fig)
            except Exception as e:
                print(f"[ERROR] Ошибка построения тепловой карты: {e}")
                import traceback
                traceback.print_exc()
                messagebox.showerror("Ошибка построения",
                                     f"Не удалось построить тепловую карту:\n{str(e)}")

        def show_all_processes(self):
            """Показать график для всех процессов"""
            dialog.destroy()
            print(f"\n[BUILD_ALL_PROCESSES] Построение сводной тепловой карты для всех процессов:")
            print(f"   Всего процессов: {len(available_processes)}")
            print(f"   Всего событий: {len(df)}")
            try:
                # Используем метод для всех процессов
                fig = self.parser._build_all_processes_heatmap(
                    df,
                    "первичных" if primary else "вторичных",
                    heatmap_mode=self.heatmap_mode_var.get(),
                    unit=self.heatmap_unit_var.get(),
                    use_cache=self.cache_enabled.get()
                )
                self.show_plot(fig)
            except Exception as e:
                print(f"[ERROR] Ошибка построения сводной карты: {e}")
                import traceback
                traceback.print_exc()
                messagebox.showerror("Ошибка построения",
                                     f"Не удалось построить сводную карту:\n{str(e)}")

        def cancel_and_close():
            """Закрыть диалог без построения"""
            dialog.destroy()
            print("[PROCESS_HEATMAP] Диалог закрыт без построения")

        # Кнопки действий
        action_frame = ttk.Frame(btn_frame)
        action_frame.pack(side="left", padx=10)

        ttk.Button(action_frame, text="Построить для выбранного",
                   command=build_and_close, width=20).pack(side="left", padx=2)

        ttk.Button(action_frame, text="Сводка всех процессов",
                   command=show_all_processes, width=20).pack(side="left", padx=2)

        # Кнопка отмены
        cancel_frame = ttk.Frame(btn_frame)
        cancel_frame.pack(side="right", padx=10)

        ttk.Button(cancel_frame, text="Отмена",
                   command=cancel_and_close, width=15).pack()

        # Подсказка
        ttk.Label(dialog, text="Подсказка: Для детального анализа выберите конкретный процесс,\n"
                               "для общего обзора используйте 'Сводка всех процессов'",
                  font=("", 8), foreground="gray").pack(pady=10)

        # Обработка закрытия окна
        dialog.protocol("WM_DELETE_WINDOW", cancel_and_close)

        # Фокус на диалоге
        dialog.focus_set()

        # Ждем закрытия диалога
        self.wait_window(dialog)

    def plot_process_energy_distribution(self, primary=None):
        """Построение распределения энергии для процессов (ВСЕ частицы: primary+secondary)"""
        df_primary = self.get_consistent_data(particle_type="primary", apply_filters=True)
        df_secondary = self.get_consistent_data(particle_type="secondary", apply_filters=True)
        df_all = pd.concat([df_primary, df_secondary], ignore_index=True)

        if df_all.empty:
            messagebox.showinfo("Нет данных", "Нет данных")
            return

        # Всегда показываем/строим из одного df_all
        self._show_process_energy_dialog(
            df_all=df_all,
            primary=None,  # можно оставить, но дальше мы его не используем
            plot_type="distribution"
        )

    def _show_process_energy_dialog(self, df_all, primary, plot_type):
        particles_label = "все частицы"
        particles_short = "всех"

        exclude = ['OutOfWorld', 'Transportation', 'initStep', 'NoProcess']
        df_filtered = df_all[~df_all['process'].isin(exclude)].copy()
        if df_filtered.empty:
            messagebox.showinfo("Нет процессов", f"Нет доступных физических процессов для {particles_label}")
            return

        process_counts = df_filtered['process'].value_counts()
        all_processes = process_counts.index.tolist()
        process_info = [(p, int(process_counts[p])) for p in all_processes]  # уже отсортировано value_counts()

        dialog = tk.Toplevel(self)
        dialog.title(f"Выбор процесса для анализа энергии ({particles_label})")
        dialog.geometry("600x500")
        dialog.transient(self)
        dialog.grab_set()

        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 600) // 2
        y = self.winfo_y() + (self.winfo_height() - 500) // 2
        dialog.geometry(f"600x500+{x}+{y}")

        ttk.Label(
            dialog,
            text=f"Анализ энергии процессов ({particles_label})",
            font=("", 11, "bold"),
            anchor="center"
        ).pack(fill="x", pady=10)

        ttk.Label(
            dialog,
            text="Количество событий показано по всем частицам (primary + secondary)",
            font=("", 9),
            anchor="center"
        ).pack(fill="x", pady=5)

        energy_frame = ttk.LabelFrame(dialog, text="Тип энергии")
        energy_frame.pack(fill="x", padx=20, pady=5)

        ttk.Label(
            energy_frame,
            text="dEStep (Edep) из лога",
            font=("", 9),
            anchor="center"
        ).pack(fill="x", pady=5)

        process_frame = ttk.LabelFrame(dialog, text="Выбор процесса")
        process_frame.pack(fill="both", expand=True, padx=20, pady=10)

        process_listbox = tk.Listbox(
            process_frame,
            height=10,
            selectmode=tk.SINGLE,
            font=("Consolas", 10),  # <- моноширинный, колонки не будут "ездить"
            activestyle="none"
        )

        scrollbar = ttk.Scrollbar(process_frame, orient="vertical", command=process_listbox.yview)
        process_listbox.configure(yscrollcommand=scrollbar.set)

        for process, total_count in process_info:
            display_text = "{:<28} | {:>8}".format(process, total_count)
            process_listbox.insert(tk.END, display_text)

        if process_info:
            process_listbox.selection_set(0)

        process_listbox.pack(side="left", fill="both", expand=True, padx=(5, 0))
        scrollbar.pack(side="right", fill="y")

        all_processes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dialog, text="Показать статистику по всем процессам",
                        variable=all_processes_var).pack(pady=5)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)

        def build_plot():
            selected_idx = process_listbox.curselection()
            if all_processes_var.get():
                process_name = None
            elif selected_idx:
                selected_text = process_listbox.get(selected_idx[0])
                process_name = selected_text.split("|", 1)[0].strip()
            else:
                messagebox.showwarning("Выбор", "Выберите процесс или отметьте 'Все процессы'")
                return

            dialog.destroy()

            try:
                if plot_type == "distribution":
                    fig = self.parser._visualize_process_energy_distribution(
                        df_filtered,
                        particles_short,
                        process_name=process_name,
                        energy_column='process_energy_loss_mev',
                        show_kde=self.show_kde.get(),
                        show_stats=self.stats_visible.get(),
                        use_cache=self.cache_enabled.get(),
                        normalization=self.normalization_mode.get()
                    )
                else:
                    fig = self.parser._visualize_process_energy_heatmap(
                        df_filtered,
                        particles_short,
                        process_name=process_name,
                        energy_column='process_energy_loss_mev',
                        use_cache=self.cache_enabled.get()
                    )
                self.show_plot(fig)
            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror("Ошибка", f"Не удалось построить график:\n{str(e)}")

        ttk.Button(btn_frame, text="Построить", command=build_plot, width=15).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Отмена", command=dialog.destroy, width=15).pack(side="left", padx=10)

    def update_dist_particle_list(self):
        """Обновить список частиц для выбора в распределениях"""
        if self.df is None:
            return

        for widget in self.particle_dist_scroll_frame.winfo_children():
            widget.destroy()

        self.particle_vars_dist.clear()

        # Используем ЕДИНЫЙ метод для получения данных
        data_source = self.get_consistent_data(
            particle_type="secondary",
            apply_filters=True  # Применяем ВСЕ фильтры!
        )

        if data_source.empty:
            ttk.Label(self.particle_dist_scroll_frame,
                      text="Нет вторичных частиц после фильтров").pack(pady=10)
            return

        # Получаем уникальные типы частиц из КОНСИСТЕНТНЫХ данных
        particles = sorted(data_source['particle'].unique())

        for particle in particles:
            var = tk.BooleanVar(value=True)
            # Количество шагов этой частицы в КОНСИСТЕНТНЫХ данных
            particle_steps = len(data_source[data_source['particle'] == particle])

            # Форматируем текст
            text = f"{particle}: {particle_steps} шагов"

            cb = ttk.Checkbutton(self.particle_dist_scroll_frame,
                                 text=text,
                                 variable=var,
                                 command=lambda p=particle, v=var: self.on_dist_particle_checkbox_change(p, v))
            cb.pack(anchor="w", padx=5, pady=2)

            self.particle_vars_dist[particle] = var

        self.particle_dist_canvas.update_idletasks()
        self.particle_dist_canvas.configure(scrollregion=self.particle_dist_canvas.bbox("all"))

    def select_all_dist_particles(self):
        """Выбрать все частицы для распределений"""
        for var in self.particle_vars_dist.values():
            var.set(True)

    def deselect_all_dist_particles(self):
        """Сбросить выбор частиц для распределений"""
        for var in self.particle_vars_dist.values():
            var.set(False)

    def on_dist_particle_checkbox_change(self, particle, var):
        """Обработчик изменения состояния чекбокса"""
        if var.get():
            if particle not in self.selected_particles_dist:
                self.selected_particles_dist.append(particle)
        else:
            if particle in self.selected_particles_dist:
                self.selected_particles_dist.remove(particle)

    def get_selected_dist_particles(self):
        """Получить список выбранных частиц для распределений"""
        selected = []
        for particle, var in self.particle_vars_dist.items():
            if var.get():
                selected.append(particle)
        return selected if selected else None

    def plot_2d_selected(self, primary):
        """Построить 2D проекции траекторий с выбором частиц"""
        df = self.apply_filters_to_df(self.filtered_df, primary)
        selected_particles = self.get_selected_dist_particles()
        fig = self.parser._visualize_2d_trajectory_projections(
            df,
            "первичных" if primary else "вторичных",
            selected_particles=selected_particles,
            use_cache=self.cache_enabled.get()
        )
        self.show_plot(fig)

    def plot_3d_selected(self, primary):
        """Построить 3D траектории с выбором частиц"""
        df = self.apply_filters_to_df(self.filtered_df, primary)
        selected_particles = self.get_selected_dist_particles()
        fig = self.parser._visualize_3d_trajectories(
            df,
            "первичных" if primary else "вторичных",
            selected_particles=selected_particles,  # Передаем выбранные частицы
            use_cache=self.cache_enabled.get()
        )
        self.show_plot(fig)

    # В классе MainWindow добавить/заменить этот метод (если его нет --- добавить)
    def get_consistent_data(self, particle_type="all", apply_filters=True):
        """
        ЕДИНСТВЕННЫЙ метод получения данных для всех графиков и статистики
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        # Базовые данные
        if apply_filters and self.filtered_df is not None:
            base = self.filtered_df.copy()
        else:
            base = self.df.copy()

        # Фильтр первичные / вторичные + первый шаг для вторичных
        if particle_type == "primary":
            df = base[base['is_primary']]
        elif particle_type == "secondary":
            df = base[~base['is_primary']]
            if self.filter_secondary_first_step.get():
                df = self.parser.filter_secondary_first_step(df)
        else:
            df = base

        # Применяем остальные UI-фильтры (частица, энергия, категория)
        if apply_filters:
            if self.particle_combo.get() != "Все":
                df = df[df['particle'] == self.particle_combo.get()]

            try:
                if self.energy_min.get():
                    df = df[df['kinetic_energy_mev'] >= float(self.energy_min.get())]
                if self.energy_max.get():
                    df = df[df['kinetic_energy_mev'] <= float(self.energy_max.get())]
            except ValueError:
                pass

            if self.category.get() == "primary":
                df = df[df['is_primary']]
            elif self.category.get() == "secondary":
                df = df[~df['is_primary']]

        return df

    def apply_filters_to_df(self, df, primary):
        """Единый метод фильтрации для всех графиков"""
        if df is None or df.empty:
            return pd.DataFrame()

        filtered = df.copy()

        # Фильтр по типу частиц
        if primary:
            filtered = filtered[filtered["is_primary"]]
        else:
            filtered = filtered[~filtered["is_primary"]]

        # Фильтр "только первый шаг" ТОЛЬКО если включен
        if self.filter_secondary_first_step.get():
            steps_before = len(filtered)
            filtered = self.parser.filter_secondary_first_step(filtered)
            steps_after = len(filtered)
            print(f"[FILTER] Фильтр 'первый шаг': {steps_before} → {steps_after}")

        # ВСЕГДА применяем остальные фильтры
        if hasattr(self, 'particle_combo') and self.particle_combo.get() != "Все":
            particle_filter = self.particle_combo.get()
            filtered = filtered[filtered["particle"] == particle_filter]
            print(f"[FILTER] Частица: {particle_filter}")

        if hasattr(self, 'energy_min') and self.energy_min.get():
            try:
                min_energy = float(self.energy_min.get())
                filtered = filtered[filtered["kinetic_energy_mev"] >= min_energy]
                print(f"[FILTER] Энергия min: {min_energy}")
            except ValueError:
                pass

        if hasattr(self, 'energy_max') and self.energy_max.get():
            try:
                max_energy = float(self.energy_max.get())
                filtered = filtered[filtered["kinetic_energy_mev"] <= max_energy]
                print(f"[FILTER] Энергия max: {max_energy}")
            except ValueError:
                pass

        return filtered

    def update_current_filters_display(self):
        """Обновляет отображение текущих активных фильтров"""
        filters_text = "Активные фильтры:\n"
        if hasattr(self, 'particle_combo') and self.particle_combo.get() != "Все":
            filters_text += f"• Частица: {self.particle_combo.get()}\n"
        if hasattr(self, 'category') and self.category.get() != "all":
            filters_text += f"• Категория: {self.category.get()}\n"
        if hasattr(self, 'energy_min') and self.energy_min.get():
            filters_text += f"• Энергия min: {self.energy_min.get()} МэВ\n"
        if hasattr(self, 'energy_max') and self.energy_max.get():
            filters_text += f"• Энергия max: {self.energy_max.get()} МэВ\n"
        if self.filter_secondary_first_step.get():
            filters_text += "• Только первый шаг для вторичных\n"

        # Обновляем отображение
        if hasattr(self, 'filters_display'):
            self.filters_display.config(text=filters_text)

    def apply_material_size(self):
        """Применить размеры материала"""
        try:
            size = float(self.material_size_entry.get())
            unit = self.material_units_combo.get()

            # Устанавливаем размеры куба (все стороны равны)
            self.material_dimensions.set_dimensions(size, size, size, unit)
            # Передаем в парсер
            self.parser.material_dimensions = self.material_dimensions

            messagebox.showinfo("Размеры",
                                f"Размер материала установлен: {size} {unit}\n"
                                f"Пределы: X=[{-size / 2:.1f}, {size / 2:.1f}] {unit}")
        except ValueError:
            messagebox.showerror("Ошибка", "Неверный формат числа")

    def update_secondary_filter(self):
        """Обновить фильтр для вторичных частиц"""
        self.parser.filter_secondary_first_step_flag = self.filter_secondary_first_step.get()

    def update_exclude_transport(self):
        """Обновить фильтр транспортных процессов"""
        self.parser.exclude_transport = self.exclude_transport.get()
        print(f"[UI] exclude_transport = {self.exclude_transport.get()}")

    def rebuild_all_plots(self):
        """Принудительно перестроить все графики (обход кэша)"""
        if self.df is None:
            return

        # Временно отключаем кэш
        cache_state = self.cache_enabled.get()
        self.cache_enabled.set(False)

        # Перестраиваем текущий график если есть
        if hasattr(self, 'current_figure') and self.current_figure:
            # Определяем тип текущего графика и перестраиваем
            # Это сложно, поэтому просто очистим
            self.clear_plot()

        messagebox.showinfo("Перестроение", "Кэш временно отключен. Постройте графики заново.")

    def analyze_tracks(self):
        """Анализ по трекам"""
        if self.df is None or self.df.empty:
            return

        # Выполняем анализ через parser
        tracks, primary_tracks, secondary_tracks = self.parser.analyze_tracks_correctly(self.df)

        # Сохраняем для последующего использования
        self.tracks_info = tracks
        self.primary_tracks = primary_tracks
        self.secondary_tracks = secondary_tracks

        # Показываем визуализацию
        fig = self.parser.visualize_correct_energy_distributions(self.df)
        self.show_plot(fig)

    def analyze_energy_distribution_issues(self, df):
        """Анализ проблем с распределением энергий"""
        print("\n=== АНАЛИЗ РАСПРЕДЕЛЕНИЯ ЭНЕРГИЙ ===")

        # Анализ первичных частиц
        primaries = df[df['is_primary']]
        if not primaries.empty:
            print(f"\nПервичные частицы:")
            print(f"Всего записей: {len(primaries)}")
            print(f"Уникальных энергий: {primaries['kinetic_energy_mev'].nunique()}")

            # Проверяем, есть ли кластеризация энергий
            energy_counts = primaries['kinetic_energy_mev'].value_counts().head(20)
            print("Топ-5 самых частых энергий:")
            for energy, count in energy_counts.items():
                percentage = (count / len(primaries)) * 100
                print(f"   {energy:.3f} МэВ: {count} частиц ({percentage:.1f}%)")

        # Анализ вторичных частиц
        secondaries = df[df['is_secondary']]
        if not secondaries.empty:
            print(f"\nВторичные частицы:")
            print(f"Всего записей: {len(secondaries)}")
            print(f"Уникальных энергий: {secondaries['kinetic_energy_mev'].nunique()}")

            # Распределение по десятичным логарифмам
            log_energies = np.log10(secondaries['kinetic_energy_mev'][secondaries['kinetic_energy_mev'] > 0])
            print(f"Диапазон log10(энергий): {log_energies.min():.2f} - {log_energies.max():.2f}")

    def toggle_kde(self):
        """Обновляет текущий график при изменении настройки KDE"""
        if self.current_figure:
            # Определяем тип текущего графика
            # (это сложно, лучше перестроить график)
            self.rebuild_current_plot()

    def rebuild_current_plot(self):
        """Перестраивает текущий график с новыми настройками"""
        # Нужно определить, какой график сейчас отображается
        # Пока просто очистим график
        self.clear_plot()

    def clear_cache(self):
        if hasattr(self, 'cache_mgr'):
            size_before = self.cache_mgr.get_cache_size()
            self.cache_mgr.clear_cache()
            size_after = self.cache_mgr.get_cache_size()
            messagebox.showinfo("Кэш", f"Кэш очищен\nБыло: {size_before:.2f} МБ\nСтало: {size_after:.2f} МБ")

    def _build_process_selection_ui(self, parent_frame):
        """Создает UI для выбора процессов"""
        process_frame = ttk.LabelFrame(parent_frame, text="Выбор процессов")
        process_frame.pack(fill="x", padx=5, pady=5)

        # Кнопка "Выбрать все"
        btn_frame = ttk.Frame(process_frame)
        btn_frame.pack(fill="x", pady=2)

        ttk.Button(btn_frame, text="Выбрать все",
                   command=self.select_all_processes).pack(side="left", padx=2)

        ttk.Button(btn_frame, text="Сбросить выбор",
                   command=self.deselect_all_processes).pack(side="left", padx=2)

        # Прокручиваемый фрейм для чекбоксов
        canvas = tk.Canvas(process_frame, height=150)
        scrollbar = ttk.Scrollbar(process_frame, orient="vertical", command=canvas.yview)

        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.process_checkbox_frame = scrollable_frame

    def update_process_list(self):
        """Обновляет список процессов для выбора"""
        if self.df is None:
            return

        # Очищаем старые чекбоксы
        for widget in self.process_checkbox_frame.winfo_children():
            widget.destroy()

        self.process_checkboxes = {}

        # Получаем уникальные процессы
        all_processes = sorted(self.df['process'].unique().tolist())

        # Создаем чекбоксы для каждого процесса
        for process in all_processes:
            var = tk.BooleanVar(value=process in self.selected_processes)
            cb = ttk.Checkbutton(self.process_checkbox_frame, text=process, variable=var,
                                 command=lambda p=process, v=var: self.on_process_checkbox_change(p, v))
            cb.pack(anchor="w")
            self.process_checkboxes[process] = var

    def select_all_processes(self):
        """Выбрать все процессы"""
        if self.df is not None:
            self.selected_processes = self.df['process'].unique().tolist()
            self.update_process_list()

    def deselect_all_processes(self):
        """Сбросить выбор процессов"""
        self.selected_processes = []
        self.update_process_list()

    def on_process_checkbox_change(self, process, var):
        """Обработчик изменения состояния чекбокса"""
        if var.get():
            if process not in self.selected_processes:
                self.selected_processes.append(process)
        else:
            if process in self.selected_processes:
                self.selected_processes.remove(process)

    def plot_process_heatmap(self, primary):
        df = self.apply_filters_to_df(self.filtered_df, primary)
        fig = self.parser._visualize_process_heatmap(
            df,
            "первичных" if primary else "вторичных",
            use_cache=self.cache_enabled.get()
        )
        self.show_plot(fig)

    def toggle_legend(self):
        if self.current_legend:
            self.current_legend.set_visible(self.legend_visible.get())
            self.current_canvas.draw_idle()

    # def toggle_stats(self):
    #     if self.current_stats:
    #         self.current_stats.set_visible(self.stats_visible.get())
    #         self.current_canvas.draw_idle()

    def toggle_stats(self):
        if self.current_stats is not None:
            self.current_stats.set_visible(self.stats_visible.get())
            if self.current_canvas:
                self.current_canvas.draw_idle()

    def export_table(self, format_type):
        """Экспорт таблицы данных (CSV, Excel, DAT)"""
        if self.df is None or self.df.empty:
            messagebox.showwarning("Нет данных", "Сначала загрузите файл")
            return

        extensions = {
            'csv': '.csv',
            'xlsx': '.xlsx',
            'excel': '.xlsx',
            'dat': '.dat'
        }

        ext = extensions.get(format_type.lower(), '.csv')
        file_path = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[
                (f"{format_type.upper()} файлы", f"*{ext}"),
                ("Все файлы", "*.*")
            ],
            title=f"Сохранить данные как {format_type.upper()}"
        )

        if not file_path:
            return

        # какой датафрейм экспортируем --- отфильтрованный или полный
        df_export = self.filtered_df if self.filtered_df is not None else self.df
        success = self.parser.save_dataframe(df_export, file_path, format_type.lower())

        if success:
            messagebox.showinfo("Успешно", f"Данные сохранены:\n{file_path}")
        else:
            messagebox.showerror("Ошибка", "Не удалось сохранить файл")

    def export_current_plot(self):
        """Экспорт текущего отображаемого графика"""
        if self.current_figure is None:
            messagebox.showwarning("Нет графика", "Сначала постройте график")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG изображение", "*.png"),
                ("JPEG изображение", "*.jpg *.jpeg"),
                ("PDF документ", "*.pdf"),
                ("SVG вектор", "*.svg"),
                ("Все файлы", "*.*")
            ],
            title="Сохранить график как..."
        )

        if not file_path:
            return

        # Определяем формат по расширению
        ext = os.path.splitext(file_path)[1].lstrip('.').lower()
        if ext not in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
            ext = 'png'
            file_path = file_path + '.png' if not file_path.endswith('.') else file_path + 'png'

        success = self.parser.save_figure(
            self.current_figure,
            file_path,
            fmt=ext,
            dpi=300
        )

        if success:
            messagebox.showinfo("Успешно", f"График сохранён:\n{file_path}")
        else:
            messagebox.showerror("Ошибка", "Не удалось сохранить график")

    def export_all_plots(self):
        """Экспорт всех основных графиков в выбранную папку"""
        if self.df is None or self.df.empty:
            messagebox.showwarning("Нет данных", "Сначала загрузите файл")
            return

        folder = filedialog.askdirectory(title="Выберите папку для сохранения всех графиков")
        if not folder:
            return

        # спросят, для каких частиц делать экспорт
        dialog = tk.Toplevel(self)
        dialog.title("Какие графики экспортировать?")
        dialog.geometry("380x220")
        dialog.transient(self)
        dialog.grab_set()

        tk.Label(dialog, text="Экспортировать графики для:",
                 font=("Helvetica", 11)).pack(pady=12)

        primary_var = tk.BooleanVar(value=True)
        secondary_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(dialog, text="Первичные частицы",
                        variable=primary_var).pack(anchor="w", padx=40)
        ttk.Checkbutton(dialog, text="Вторичные частицы",
                        variable=secondary_var).pack(anchor="w", padx=40)

        tk.Label(dialog, text="Форматы файлов:",
                 font=("Helvetica", 10)).pack(pady=(20, 8))

        fmt_png = tk.BooleanVar(value=True)
        fmt_pdf = tk.BooleanVar(value=False)

        f = ttk.Frame(dialog)
        f.pack()
        ttk.Checkbutton(f, text="PNG", variable=fmt_png).pack(side="left", padx=20)
        ttk.Checkbutton(f, text="PDF", variable=fmt_pdf).pack(side="left", padx=20)

        def do_export():
            formats = []
            if fmt_png.get():
                formats.append('png')
            if fmt_pdf.get():
                formats.append('pdf')
            if not formats:
                messagebox.showwarning("Формат", "Выберите хотя бы один формат")
                return

            exported_files = []

            # первичные
            if primary_var.get():
                df_prim = self.df[self.df['is_primary']]
                if not df_prim.empty:
                    files = self.parser.export_all_typical_plots(
                        df_prim, folder, prefix="primary", dpi=220, formats=formats
                    )
                    exported_files.extend(files)

            # вторичные
            if secondary_var.get():
                df_sec = self.df[~self.df['is_primary']]
                if self.filter_secondary_first_step.get():
                    df_sec = self.parser.filter_secondary_first_step(df_sec)
                if not df_sec.empty:
                    files = self.parser.export_all_typical_plots(
                        df_sec, folder, prefix="secondary", dpi=220, formats=formats
                    )
                    exported_files.extend(files)

            dialog.destroy()

            if exported_files:
                msg = f"Сохранено {len(exported_files)} файлов в папку:\n{folder}"
                messagebox.showinfo("Экспорт завершён", msg)
            else:
                messagebox.showinfo("Результат", "Не удалось создать ни одного графика")

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=25)

        ttk.Button(btn_frame, text="Экспортировать",
                   command=do_export).pack(side="left", padx=12)
        ttk.Button(btn_frame, text="Отмена",
                   command=dialog.destroy).pack(side="left", padx=12)

    def export_plot(self, fmt="png"):
        """Экспорт текущего графика"""
        if self.current_figure is None:
            messagebox.showwarning("Нет графика", "Сначала постройте график")
            return

        # Запрашиваем путь для сохранения
        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(f"{fmt.upper()} files", f"*.{fmt}"), ("All files", "*.*")]
        )

        if not file_path:
            return

        # Вызываем метод экспорта графика у парсера
        success = self.export_plot(self.current_figure, file_path, fmt)

        if success:
            messagebox.showinfo("Успех", f"График сохранен в {file_path}")
        else:
            messagebox.showerror("Ошибка", "Не удалось сохранить график")

    # ==================== SUMMARY TAB ====================

    def _build_summary_tab(self):
        ttk.Button(self.summary_tab, text="📄 Сгенерировать",
                   command=self.generate_summary).pack(pady=5)
        self.summary_text = tk.Text(self.summary_tab)
        self.summary_text.pack(fill="both", expand=True, padx=5, pady=5)

    # ==================== EXPORT TAB ====================

    def _build_export_tab(self):
        ttk.Button(self.export_tab, text="Экспорт CSV",
                   command=lambda: self.export_table("csv")).pack(pady=5)
        ttk.Button(self.export_tab, text="Экспорт Excel",
                   command=lambda: self.export_table("xlsx")).pack(pady=5)
        ttk.Button(self.export_tab, text="Экспорт DAT",
                   command=lambda: self.export_table("dat")).pack(pady=5)

        ttk.Separator(self.export_tab).pack(fill="x", pady=10)

        ttk.Button(self.export_tab,
                   text="Сохранить текущий график",
                   command=self.export_current_plot).pack(pady=5)
        ttk.Button(self.export_tab,
                   text="Экспорт всех основных графиков",
                   command=self.export_all_plots).pack(pady=5)

    # ==================== ЛОГИКА ====================

    def load_file(self, path=None):
        """Загружает лог файл Geant4"""
        # Если путь не передан, спрашиваем пользователя
        if path is None:
            path = filedialog.askopenfilename(filetypes=[("Log files", "*.log *.txt")])
            if not path:
                return

        self.df = self.parser.parse_log_file(path)

        if self.df is None or self.df.empty:
            messagebox.showerror("Ошибка", "Не удалось распарсить файл или файл пуст")
            return

        # ПРОВЕРКА: распечатаем статистику по parent_id и track_id
        print("\n=== ПРОВЕРКА СТАТИСТИКИ ===")
        print("Распределение parent_id:")
        print(self.df['parent_id'].value_counts().sort_index())
        print("\nРаспределение track_id:")
        print(self.df['track_id'].value_counts().sort_index())
        print("\nПервые 20 записей с parent_id > 0:")
        secondary = self.df[self.df['parent_id'] > 0]
        if not secondary.empty:
            print(secondary[['particle', 'track_id', 'parent_id']].head(20))

        self.filtered_df = self.df.copy()
        self.file_label.config(text=os.path.basename(path))

        self.particle_combo["values"] = ["Все"] + sorted(self.df["particle"].unique().tolist())
        self.particle_combo.current(0)
        self.particle_combo["state"] = "readonly"

        self.update_dist_particle_list()
        self.show_basic_stats()

        if hasattr(self, 'update_particle_list_3d'):
            self.update_particle_list_3d()

        # Дополнительная визуализация для проверки
        # self.show_classification_check_plot()

    def smart_load_file(self):
        """Умная загрузка файла с автоматическим определением типа"""
        filetypes = [
            ("Geant4 лог файлы", "*.log *.txt"),
            ("CSV файлы", "*.csv"),
            ("Все файлы", "*.*")
        ]

        path = filedialog.askopenfilename(filetypes=filetypes)
        if not path:
            return

        # Определяем тип файла по расширению
        ext = os.path.splitext(path)[1].lower()

        if ext in ['.log', '.txt']:
            self.load_file(path)  # Передаём путь напрямую
        elif ext == '.csv':
            self.load_csv_file(path)  # Передаём путь напрямую
        else:
            # Пробуем определить по содержимому
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                if 'G4WT' in first_line or 'G4Track' in first_line:
                    self.load_file(path)  # Передаём путь напрямую
                else:
                    # Пробуем как CSV
                    self.load_csv_file(path)  # Передаём путь напрямую

    def load_csv_file(self, path=None):
        """Загружает CSV файл с данными симуляции"""
        # Если путь не передан, спрашиваем пользователя
        if path is None:
            filetypes = [
                ("CSV файлы", "*.csv"),
                ("Все файлы", "*.*")
            ]
            path = filedialog.askopenfilename(
                title="Выберите CSV файл с данными симуляции",
                filetypes=filetypes
            )
            if not path:
                return  # Пользователь отменил выбор

        try:
            # Показываем индикатор загрузки
            self.file_label.config(text="Загрузка CSV файла...")
            self.update()

            # Загружаем CSV через парсер
            self.df = self.parser.load_csv_file(path)

            if self.df is None or self.df.empty:
                messagebox.showerror("Ошибка",
                                     "Не удалось загрузить CSV файл или файл пуст")
                self.file_label.config(text="Ошибка загрузки CSV файла")
                return

            # Устанавливаем отфильтрованные данные
            self.filtered_df = self.df.copy()

            # Обновляем информацию о файле
            self.file_label.config(
                text=f"CSV файл: {os.path.basename(path)} ({len(self.df)} записей)"
            )

            # Обновляем доступные частицы в фильтре
            self.particle_combo["values"] = ["Все"] + sorted(self.df["particle"].unique().tolist())
            self.particle_combo.current(0)
            self.particle_combo["state"] = "readonly"

            self.update_dist_particle_list()

            # Показываем базовую статистику
            self.show_basic_stats()

            # Обновляем список процессов для выбора
            if hasattr(self, 'process_checkbox_frame'):
                self.update_process_list()

            # Показываем уведомление об успешной загрузке
            messagebox.showinfo("Успех",
                                f"CSV файл успешно загружен:\n"
                                f"• Записей: {len(self.df)}\n"
                                f"• Частиц: {self.df['particle'].nunique()}\n"
                                f"• Процессов: {self.df['process'].nunique() if 'process' in self.df.columns else 'N/A'}")

            # Выводим дополнительную информацию в консоль
            print("\n" + "=" * 50)
            print("CSV ФАЙЛ УСПЕШНО ЗАГРУЖЕН")
            print("=" * 50)
            print(f"Путь: {path}")
            print(f"Размер: {os.path.getsize(path) / 1024:.1f} KB")
            print(f"Колонки: {len(self.df.columns)}")
            print("\nПервые 5 записей:")
            print(self.df.head())

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить CSV файл:\n{str(e)}")
            self.file_label.config(text="Ошибка загрузки CSV файла")
            print(f"[ERROR] Ошибка загрузки CSV: {e}")
            import traceback
            traceback.print_exc()

    def debug_classification(self):
        """Отладка классификации частиц"""
        if self.df is None:
            return

        print("\n=== ОТЛАДКА КЛАССИФИКАЦИИ ===")

        # Сгруппируем по уникальным трекам
        unique_tracks = self.df.groupby(['track_id', 'parent_id']).agg({
            'particle': 'first',
            'kinetic_energy_mev': 'mean',
            'step_number': 'count'
        }).reset_index()

        print(f"Уникальных треков: {len(unique_tracks)}")
        print("\nТоп-20 треков:")
        print(unique_tracks.sort_values('track_id').head(20))

        # Анализ цепочек
        print("\n=== ЦЕПОЧКИ РОЖДЕНИЯ ===")
        for parent_id in sorted(unique_tracks['parent_id'].unique()):
            children = unique_tracks[unique_tracks['parent_id'] == parent_id]
            if not children.empty:
                parent_info = unique_tracks[unique_tracks['track_id'] == parent_id]
                if not parent_info.empty:
                    parent_particle = parent_info['particle'].iloc[0]
                    print(f"\nРодитель {parent_id} ({parent_particle}):")
                    for _, child in children.iterrows():
                        print(
                            f" -> {child['particle']} (track={child['track_id']}, E={child['kinetic_energy_mev']:.3f} МэВ)")

    def show_classification_check_plot(self):
        """Показать график для проверки корректности классификации"""
        if self.df is None or self.df.empty:
            return

        fig = Figure(figsize=(12, 8))

        # 1. Распределение энергий первичных и вторичных
        ax1 = fig.add_subplot(2, 2, 1)
        primary_energies = self.df[self.df['is_primary']]['kinetic_energy_mev']
        secondary_energies = self.df[self.df['is_secondary']]['kinetic_energy_mev']

        if len(primary_energies) > 0:
            ax1.hist(primary_energies, bins=50, alpha=0.5, label='Первичные',
                     color='blue', density=True)
        if len(secondary_energies) > 0:
            ax1.hist(secondary_energies, bins=50, alpha=0.5, label='Вторичные',
                     color='red', density=True)

        ax1.set_xlabel('Кинетическая энергия (МэВ)')
        ax1.set_ylabel('Нормализованная частота')
        ax1.set_title('Распределение энергий')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Соотношение parent_id и track_id
        ax2 = fig.add_subplot(2, 2, 2)
        # Берем только первые 1000 записей для читаемости
        sample_df = self.df.head(1000)
        # Цвет по классификации
        colors = ['blue' if p else 'red' for p in sample_df['is_primary']]
        ax2.scatter(sample_df['track_id'], sample_df['parent_id'],
                    c=colors, alpha=0.5, s=10)
        ax2.set_xlabel('Track ID')
        ax2.set_ylabel('Parent ID')
        ax2.set_title('Track ID vs Parent ID (синие=первичные)')
        ax2.grid(True, alpha=0.3)

        # 3. Распределение типов частиц
        ax3 = fig.add_subplot(2, 2, 3)
        particle_counts = self.df['particle'].value_counts().head(10)
        ax3.bar(range(len(particle_counts)), particle_counts.values)
        ax3.set_xticks(range(len(particle_counts)))
        ax3.set_xticklabels(particle_counts.index, rotation=45, ha='right')
        ax3.set_ylabel('Количество')
        ax3.set_title('Топ-10 типов частиц')
        ax3.grid(True, alpha=0.3)

        # 4. Энергия по track_id
        ax4 = fig.add_subplot(2, 2, 4)
        # Группируем по track_id и вычисляем среднюю энергию
        track_energy = self.df.groupby('track_id')['kinetic_energy_mev'].mean().head(20)
        track_counts = self.df.groupby('track_id').size().head(20)
        x = range(len(track_energy))

        ax4.bar(x, track_energy.values, alpha=0.5, label='Средняя энергия')
        ax4.set_xticks(x)
        ax4.set_xticklabels(track_energy.index, rotation=45, ha='right')
        ax4.set_ylabel('Средняя энергия (МэВ)')
        ax4.set_xlabel('Track ID')
        ax4.set_title('Средняя энергия по Track ID')
        ax4.grid(True, alpha=0.3)

        # Добавляем количество частиц на второй оси
        ax4_twin = ax4.twinx()
        ax4_twin.plot(x, track_counts.values, 'r-', linewidth=2, label='Количество')
        ax4_twin.set_ylabel('Количество частиц', color='red')
        ax4_twin.tick_params(axis='y', labelcolor='red')

        # Объединяем легенды
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        fig.tight_layout()

        # Показываем в отдельном окне или в интерфейсе
        self.show_plot(fig)

    def apply_filters(self):
        df = self.df.copy()

        if self.particle_combo.get() != "Все":
            df = df[df["particle"] == self.particle_combo.get()]

        if self.category.get() == "primary":
            df = df[df["is_primary"]]
        elif self.category.get() == "secondary":
            df = df[~df["is_primary"]]

        if self.energy_min.get():
            df = df[df["kinetic_energy_mev"] >= float(self.energy_min.get())]
        if self.energy_max.get():
            df = df[df["kinetic_energy_mev"] <= float(self.energy_max.get())]

        self.filtered_df = df
        self.show_filtered_stats(df)
        self.update_dist_particle_list()

        if self.data_consistency_debug.get():
            self.check_data_consistency()
            self.debug_data_sources()

    def show_filtered_stats(self, df):
        """Показать статистику отфильтрованных данных"""
        if df is None or df.empty:
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert("1.0", "Нет данных после фильтрации")
            return

        # Если включён фильтр "только первый шаг вторичных" --- применяем его здесь тоже
        if self.filter_secondary_first_step.get():
            df = self.parser.filter_secondary_first_step(df.copy())  # ← добавили согласованность

        total_rows = len(self.df) if self.df is not None else 0
        filtered_rows = len(df)

        particle_stats = df.groupby('particle').agg({
            'track_id': 'nunique',
            'step_number': 'count'
        }).sort_values('step_number', ascending=False).head(10)

        text = f"""=== СТАТИСТИКА (ПОСЛЕ ФИЛЬТРОВ) ===

Всего записей: {filtered_rows:,} (из {total_rows:,} всего, {(filtered_rows / total_rows * 100 if total_rows > 0 else 0):.1f}%)

Топ-10 частиц в отфильтрованных данных:"""

        for particle, row in particle_stats.iterrows():
            tracks = row['track_id']
            steps = row['step_number']
            text += f"\n• {particle}: {steps} шагов, {tracks} треков"

        text += f"\n\nПримененные фильтры:"
        if hasattr(self, 'particle_combo') and self.particle_combo.get() != "Все":
            text += f"\n• Тип частицы: {self.particle_combo.get()}"
        if hasattr(self, 'category') and self.category.get() != "all":
            text += f"\n• Категория: {self.category.get()}"
        if hasattr(self, 'energy_min') and self.energy_min.get():
            text += f"\n• Энергия min: {self.energy_min.get()} MeV"
        if hasattr(self, 'energy_max') and self.energy_max.get():
            text += f"\n• Энергия max: {self.energy_max.get()} MeV"
        if self.filter_secondary_first_step.get():
            text += f"\n• Только первый шаг для вторичных: ВКЛ"

        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", text)

    def reset_filters(self):
        self.filtered_df = self.df.copy()
        self.energy_min.delete(0, tk.END)
        self.energy_max.delete(0, tk.END)
        self.category.set("all")
        self.particle_combo.current(0)
        self.show_basic_stats()

    # ==================== ГРАФИКИ ====================

    def show_plot(self, fig):
        self.clear_plot()

        self.current_figure = fig
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)

        # Применяем tight_layout ТОЛЬКО если это не heatmap
        if not getattr(fig, "_no_tight_layout", False):
            try:
                fig.tight_layout()
            except Exception as e:
                print(f"Ошибка tight_layout: {e}")

        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Легенда
        self.current_legend = None
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg:
                self.current_legend = leg
                leg.set_visible(self.legend_visible.get())
                break

        # Статистика
        self.current_stats = getattr(fig, '_stats_artist', None)
        if self.current_stats is not None:
            self.current_stats.set_visible(self.stats_visible.get())

        self.current_canvas.draw_idle()

        self.current_toolbar = NavigationToolbar2Tk(self.current_canvas, self.plot_frame)
        self.current_toolbar.update()

    def clear_plot(self):
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None

        if self.current_toolbar:
            self.current_toolbar.destroy()
            self.current_toolbar = None

        for w in self.plot_frame.winfo_children():
            w.destroy()

        self.current_figure = None
        self.current_legend = None
        self.current_stats = None

    def plot_energy(self, primary):
        # Используем ЕДИНЫЙ метод
        df = self.get_consistent_data(
            particle_type="primary" if primary else "secondary",
            apply_filters=True  # ВАЖНО: применяем все фильтры!
        )

        if df.empty:
            messagebox.showinfo("Нет данных",
                                f"Нет данных для {'первичных' if primary else 'вторичных'} частиц после применения фильтров")
            return

        # Логируем для отладки
        print(f"[PLOT] {'Первичные' if primary else 'Вторичные'}: {len(df)} записей")
        print(f"   Частицы в данных: {df['particle'].unique().tolist()}")

        # Строим график
        selected_particles = None if primary else self.get_selected_dist_particles()

        # Если выбранные частицы есть, фильтруем по ним
        if selected_particles:
            print(f"   Выбранные частицы: {selected_particles}")
            df = df[df['particle'].isin(selected_particles)]
            print(f"   После фильтра по выбранным: {len(df)} записей")

        fig = self.parser._visualize_energy_distributions(
            df,
            "первичных" if primary else "вторичных",
            show_kde=self.show_kde.get(),
            show_stats=True,
            use_cache=self.cache_enabled.get(),
            normalization=self.normalization_mode.get(),
            selected_particles=None,  # Уже отфильтровали выше
            xlim=(float(self.xlim_min.get()) if self.xlim_min.get() and self.apply_limits_to_plot.get() else None,
                  float(self.xlim_max.get()) if self.xlim_max.get() and self.apply_limits_to_plot.get() else None),
            ylim=(float(self.ylim_min.get()) if self.ylim_min.get() and self.apply_limits_to_plot.get() else None,
                  float(self.ylim_max.get()) if self.ylim_max.get() and self.apply_limits_to_plot.get() else None)
        )

        self.show_plot(fig)

    def plot_loss(self, primary):
        """Построение распределения потерь энергии - ЕДИНЫЙ ИСТОЧНИК ДАННЫХ"""
        df = self.get_consistent_data(
            particle_type="primary" if primary else "secondary",
            apply_filters=True
        )

        if df.empty:
            messagebox.showinfo("Нет данных",
                                f"Нет данных для {'первичных' if primary else 'вторичных'} частиц после применения фильтров")
            return

        df_loss = df[df['energy_loss_mev'] >= 0].copy()
        if df_loss.empty:
            messagebox.showwarning("Нет данных",
                                   f"Нет данных с потерями энергии >0 для {'первичных' if primary else 'вторичных'} частиц")
            return

        print(f"\n[PLOT_LOSS] {'Первичные' if primary else 'Вторичные'}:")
        print(f"   Всего шагов: {len(df)}")
        print(f"   Шагов с потерей >0: {len(df_loss)} ({len(df_loss) / len(df) * 100:.1f}%)")
        print(f"   Суммарные потери: {df_loss['energy_loss_mev'].sum():.6f} МэВ")

        # Если есть выбранные частицы для вторичных
        selected_particles = None if primary else self.get_selected_dist_particles()

        fig = self.parser._visualize_energy_loss_distribution(
            df_loss,
            "первичных" if primary else "вторичных",
            show_kde=self.show_kde.get(),
            show_stats=True,
            min_y_log=0.8,
            use_cache=self.cache_enabled.get(),
            normalization=self.normalization_mode.get(),
            selected_particles=selected_particles,
            xlim=(float(self.xlim_min.get()) if self.xlim_min.get() and self.apply_limits_to_plot.get() else None,
                  float(self.xlim_max.get()) if self.xlim_max.get() and self.apply_limits_to_plot.get() else None),
            ylim=(float(self.ylim_min.get()) if self.ylim_min.get() and self.apply_limits_to_plot.get() else None,
                  float(self.ylim_max.get()) if self.ylim_max.get() and self.apply_limits_to_plot.get() else None)
        )

        self.show_plot(fig)

    def plot_dE(self, primary):
        """Построение распределения dE - ЕДИНЫЙ ИСТОЧНИК ДАННЫХ"""
        # Берём ВСЕ частицы: primary + secondary
        df_primary = self.get_consistent_data(
            particle_type="primary",
            apply_filters=True
        )
        df_secondary = self.get_consistent_data(
            particle_type="secondary",
            apply_filters=True
        )
        df = pd.concat([df_primary, df_secondary], ignore_index=True)

        if df.empty:
            messagebox.showinfo("Нет данных", "Нет данных для всех частиц после применения фильтров")
            return

        df_de = df[df['process_energy_loss_mev'] >= 0].copy()
        if df_de.empty:
            messagebox.showwarning("Нет данных", "Нет данных с dE >= 0 для всех частиц")
            return

        print(f"\n[PLOT_DE] Все частицы:")
        print(f"   Всего шагов: {len(df)}")
        print(f"   Шагов с dE >= 0: {len(df_de)} ({len(df_de) / len(df) * 100:.1f}%)")
        print(f"   Сумма dE: {df_de['process_energy_loss_mev'].sum():.6f} МэВ")
        print(f"   Макс dE: {df_de['process_energy_loss_mev'].max():.6f} МэВ")

        # selected_particles НЕ применяем (иначе ты снова отрежешь часть данных)
        selected_particles = None

        fig = self.parser._visualize_dE_distribution(
            df_de,
            "всех",
            show_kde=self.show_kde.get(),
            show_stats=True,
            min_y_log=0.8,
            use_cache=self.cache_enabled.get(),
            normalization=self.normalization_mode.get(),
            selected_particles=selected_particles,
            xlim=(float(self.xlim_min.get()) if self.xlim_min.get() and self.apply_limits_to_plot.get() else None,
                  float(self.xlim_max.get()) if self.xlim_max.get() and self.apply_limits_to_plot.get() else None),
            ylim=(float(self.ylim_min.get()) if self.ylim_min.get() and self.apply_limits_to_plot.get() else None,
                  float(self.ylim_max.get()) if self.ylim_max.get() and self.apply_limits_to_plot.get() else None)
        )

        self.show_plot(fig)

    def plot_process(self, primary):
        """Построение частот процессов - ЕДИНЫЙ ИСТОЧНИК ДАННЫХ"""
        df = self.get_consistent_data(
            particle_type="primary" if primary else "secondary",
            apply_filters=True
        )

        if df.empty:
            messagebox.showinfo("Нет данных",
                                f"Нет данных для {'первичных' if primary else 'вторичных'} частиц после применения фильтров")
            return

        print(f"[PLOT_PROCESS] Построение графика процессов для {'первичных' if primary else 'вторичных'}:")
        print(f"   Всего шагов: {len(df)}")
        print(f"   Уникальных процессов: {df['process'].nunique()}")

        # Статистика по процессам
        process_counts = df['process'].value_counts()
        print(f"   Топ-5 процессов:")
        for process, count in process_counts.head(20).items():
            percentage = count / len(df) * 100
            print(f"     {process}: {count} шагов ({percentage:.1f}%)")

        self.log_dataset_statistics(df, "process_distribution", primary)

        fig = self.parser._visualize_additional_plots(
            df,
            "первичных" if primary else "вторичных"
        )

        self.show_plot(fig)

    def plot_heatmap(self, primary):
        """Построение тепловой карты - ЕДИНЫЙ ИСТОЧНИК ДАННЫХ"""
        df = self.get_consistent_data(
            particle_type="primary" if primary else "secondary",
            apply_filters=True
        )

        if df.empty:
            messagebox.showinfo("Нет данных",
                                f"Нет данных для {'первичных' if primary else 'вторичных'} частиц после применения фильтров")
            return

        print(f"[PLOT_HEATMAP] Построение тепловой карты для {'первичных' if primary else 'вторичных'}:")
        print(f"   Всего шагов: {len(df)}")
        print(f"   Частиц: {df['particle'].nunique()}")

        # Проверяем наличие координатных данных
        coord_columns = ['x_mm', 'y_mm', 'z_mm']
        missing_coords = [col for col in coord_columns if col not in df.columns]
        if missing_coords:
            print(f"   ВНИМАНИЕ: Отсутствуют колонки координат: {missing_coords}")

        self.log_dataset_statistics(df, "heatmap", primary)

        fig = self.parser._visualize_heatmap(
            df,
            "первичных" if primary else "вторичных",
            heatmap_mode=self.heatmap_mode_var.get(),
            unit=self.heatmap_unit_var.get(),
            use_cache=self.cache_enabled.get()
        )

        self.show_plot(fig)

    def plot_2d(self, primary):
        """Построение 2D проекций траекторий - ЕДИНЫЙ ИСТОЧНИК ДАННЫХ"""
        df = self.get_consistent_data(
            particle_type="primary" if primary else "secondary",
            apply_filters=True
        )

        if df.empty:
            messagebox.showinfo(
                "Нет данных",
                f"Нет данных для {'первичных' if primary else 'вторичных'} частиц после применения фильтров"
            )
            return

        print(f"[PLOT_2D] Построение 2D проекций для {'первичных' if primary else 'вторичных'}:")
        print(f"   Всего шагов: {len(df)}")
        print(f"   Уникальных треков: {df['track_id'].nunique()}")

        selected_particles = self.get_selected_dist_particles()
        if selected_particles:
            print(f"   Выбранные частицы: {selected_particles}")
            for particle in selected_particles:
                if particle in df['particle'].unique():
                    particle_steps = len(df[df['particle'] == particle])
                    particle_tracks = df[df['particle'] == particle]['track_id'].nunique()
                    print(f"     {particle}: {particle_steps} шагов, {particle_tracks} треков")

        self.log_dataset_statistics(df, "2d_traj_projections", primary)

        fig = self.parser._visualize_2d_trajectory_projections(
            df,
            "первичных" if primary else "вторичных",
            selected_particles=selected_particles,
            use_cache=self.cache_enabled.get()
        )

        self.show_plot(fig)

    def plot_3d(self, primary):
        """Построение 3D траекторий - ЕДИНЫЙ ИСТОЧНИК ДАННЫХ"""
        df = self.get_consistent_data(
            particle_type="primary" if primary else "secondary",
            apply_filters=True
        )

        if df.empty:
            messagebox.showinfo("Нет данных",
                                f"Нет данных для {'первичных' if primary else 'вторичных'} частиц после применения фильтров")
            return

        print(f"[PLOT_3D] Построение 3D траекторий для {'первичных' if primary else 'вторичных'}:")
        print(f"   Всего шагов: {len(df)}")
        print(f"   Уникальных треков: {df['track_id'].nunique()}")

        selected_particles = self.get_selected_dist_particles()
        if selected_particles:
            print(f"   Выбранные частицы: {selected_particles}")
            for particle in selected_particles:
                if particle in df['particle'].unique():
                    particle_steps = len(df[df['particle'] == particle])
                    particle_tracks = df[df['particle'] == particle]['track_id'].nunique()
                    print(f"     {particle}: {particle_steps} шагов, {particle_tracks} треков")

        self.log_dataset_statistics(df, "3d_trajectories", primary)

        fig = self.parser._visualize_3d_trajectories(
            df,
            "первичных" if primary else "вторичных",
            selected_particles=selected_particles,
            use_cache=self.cache_enabled.get()
        )

        self.show_plot(fig)

    # ===== НОВЫЙ МЕТОД: Построение дозовых карт =====
    def plot_dose_map(self, primary=None, per_layer=False):
        """
        Построение дозовой карты.

        Args:
            primary: True - первичные, False - вторичные, None - все
            per_layer: True - отдельно по слоям, False - суммарно
        """
        # Получаем данные
        if primary is None:
            # Все частицы
            df_primary = self.get_consistent_data(particle_type="primary", apply_filters=True)
            df_secondary = self.get_consistent_data(particle_type="secondary", apply_filters=True)
            df = pd.concat([df_primary, df_secondary], ignore_index=True)
            particle_label = "всех"
        else:
            df = self.get_consistent_data(
                particle_type="primary" if primary else "secondary",
                apply_filters=True
            )
            particle_label = "первичных" if primary else "вторичных"

        if df.empty:
            messagebox.showinfo("Нет данных", f"Нет данных для {particle_label} частиц")
            return

        # Проверяем наличие данных о дозе
        if 'dose_gray' not in df.columns:
            messagebox.showwarning("Нет данных",
                                   "Данные о дозе не рассчитаны. Возможно, лог не содержит информацию о материалах.")
            return

        # Выводим статистику по слоям
        if self.parser.layers:
            print("\n=== СТАТИСТИКА ПО СЛОЯМ ===")
            for layer in self.parser.layers:
                layer_df = df[df['z_mm'].between(layer.z_min_mm, layer.z_max_mm)]
                if not layer_df.empty:
                    total_dE = layer_df['process_energy_loss_mev'].sum()
                    total_dose = layer_df['dose_gray'].sum()
                    avg_dose = layer.get_average_dose_gray()
                    print(f"\nСлой: {layer.name}")
                    print(f"  Материал: {layer.material}")
                    print(f"  Плотность: {layer.density_g_cm3:.2f} г/см³")
                    print(f"  Толщина: {layer.thickness_mm:.2f} мм")
                    print(f"  Объем: {layer.volume_mm3:.2f} мм³")
                    print(f"  Суммарная энергия: {total_dE:.3f} МэВ")
                    print(f"  Суммарная доза: {total_dose:.3e} Гр")
                    print(f"  Средняя доза в слое: {avg_dose:.3e} Гр ({avg_dose * 100:.3e} рад)")

        # Строим карту
        fig = self.parser._visualize_dose_map(
            df,
            particle_type=particle_label,
            unit=self.dose_unit.get(),
            per_layer=per_layer,
            use_cache=self.cache_enabled.get()
        )

        self.show_plot(fig)

    # ===== КОНЕЦ НОВОГО МЕТОДА =====

    # ==================== СТАТИСТИКА ====================

    def log_dataset_statistics(self, df, plot_type, is_primary):
        """Логирует статистику датасета для отладки"""
        print(f"\n{'=' * 60}")
        print(f"СТАТИСТИКА ДЛЯ {plot_type.upper()} ({'первичные' if is_primary else 'вторичные'}):")
        print(f"{'=' * 60}")
        print(f"Всего записей: {len(df):,}")
        print(f"Уникальных частиц: {df['particle'].nunique()}")
        print(f"Уникальных треков: {df['track_id'].nunique()}")
        if 'kinetic_energy_mev' in df.columns:
            energy_stats = df['kinetic_energy_mev'].agg(['mean', 'min', 'max', 'std'])
            print(f"\nЭнергия (МэВ):")
            print(f"  Среднее: {energy_stats['mean']:.6f}")
            print(f"  Min: {energy_stats['min']:.6f}")
            print(f"  Max: {energy_stats['max']:.6f}")
            print(f"  Std: {energy_stats['std']:.6f}")
        if 'energy_loss_mev' in df.columns:
            loss_stats = df['energy_loss_mev'].agg(['sum', 'mean', 'max'])
            print(f"\nПотери энергии:")
            print(f"  Сумма: {loss_stats['sum']:.6f} МэВ")
            print(f"  Среднее: {loss_stats['mean']:.6f} МэВ/шаг")
            print(f"  Макс: {loss_stats['max']:.6f} МэВ")
        print(f"{'=' * 60}")

    def show_basic_stats(self):
        """Показать базовую статистику с указанием примененных фильтров"""
        # Используем filtered_df для статистики
        df_to_show = self.get_consistent_data("all")

        if df_to_show is None or df_to_show.empty:
            text = "Нет данных для отображения"
        else:
            # Определяем, применены ли фильтры (сравниваем с полным df)
            filters_applied = (len(df_to_show) != len(self.df)) if self.df is not None else False

            text = f"=== СТАТИСТИКА {'(ПОСЛЕ ФИЛЬТРОВ)' if filters_applied else ''} ===\n"
            text += f"Всего записей: {len(df_to_show):,}"
            if filters_applied and self.df is not None:
                text += f" (из {len(self.df):,} всего)"
            text += f"""

Типы частиц: {', '.join(df_to_show['particle'].unique()[:5])}{'...' if df_to_show['particle'].nunique() > 5 else ''}
Всего типов частиц: {df_to_show['particle'].nunique()}

Диапазон энергий: {df_to_show['kinetic_energy_mev'].min():.3e} - {df_to_show['kinetic_energy_mev'].max():.3e} MeV
Средняя энергия: {df_to_show['kinetic_energy_mev'].mean():.3e} MeV
Суммарные потери энергии: {df_to_show['energy_loss_mev'].sum():.3e} MeV

Координаты:
X: {df_to_show['x_mm'].min():.2f} - {df_to_show['x_mm'].max():.2f} мм
Y: {df_to_show['y_mm'].min():.2f} - {df_to_show['y_mm'].max():.2f} мм
Z: {df_to_show['z_mm'].min():.2f} - {df_to_show['z_mm'].max():.2f} мм"""

            if 'is_primary' in df_to_show.columns:
                text += f"""
Первичных частиц: {df_to_show['is_primary'].sum():,}
Вторичных частиц: {df_to_show['is_secondary'].sum():,}"""

            if 'process' in df_to_show.columns:
                top_processes = df_to_show['process'].value_counts().head(20)
                text += f"\n\nТоп-5 процессов:"
                for process, count in top_processes.items():
                    text += f"\n  {process}: {count:,}"

            if 'track_id' in df_to_show.columns:
                text += f"\n\nУникальных треков: {df_to_show['track_id'].nunique():,}"
            if 'thread' in df_to_show.columns:
                text += f"\nПотоков (threads): {df_to_show['thread'].nunique():,}"

            # Информация о фильтрах (оставляем как есть)
            if filters_applied:
                text += f"\n\n=== ПРИМЕНЕННЫЕ ФИЛЬТРЫ ==="
                if hasattr(self, 'particle_combo') and self.particle_combo.get() != "Все":
                    text += f"\nТип частицы: {self.particle_combo.get()}"
                if hasattr(self, 'category') and self.category.get() != "all":
                    text += f"\nКатегория: {self.category.get()}"
                if hasattr(self, 'energy_min') and self.energy_min.get():
                    text += f"\nЭнергия min: {self.energy_min.get()} MeV"
                if hasattr(self, 'energy_max') and self.energy_max.get():
                    text += f"\nЭнергия max: {self.energy_max.get()} MeV"
                if self.filter_secondary_first_step.get():
                    text += f"\n• Только первый шаг для вторичных: ВКЛ"

            # Добавляем информацию о слоях
            if self.parser.layers:
                text += f"\n\n=== ИНФОРМАЦИЯ О СЛОЯХ ==="
                for layer in self.parser.layers:
                    text += f"\n{layer.name}: {layer.material}, ρ={layer.density_g_cm3:.2f} г/см³, толщина={layer.thickness_mm:.2f} мм"
                    if layer.volume_mm3:
                        text += f", объем={layer.volume_mm3:.2f} мм³"

        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", text)

    # ==================== СВОДКА ====================

    def generate_summary(self):
        text = self.parser.generate_text_summary(self.filtered_df)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", text)


# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
