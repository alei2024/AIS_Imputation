import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


AIS_FEATURES = ["Latitude", "Longitude", "SOG", "COG"]


@dataclass
class AISPreprocessConfig:
    data_dir: str = "./data/ais_data"
    month_prefix: str = "aisdk-2022-12-"
    freq: str = "10min"  # pandas offset alias, e.g. "1min", "5min", "10min"
    eval_length: int = 144  # 24h at 10min
    stride: int = 144  # non-overlap by default
    min_observed_ratio: float = 0.2  # per-window observed ratio threshold
    # subsampling to keep runtime manageable for large AIS
    mmsi_keep_ratio: float = 1.0  # keep MMSI by deterministic hash
    max_ships: int = 0  # 0 means no limit
    max_windows_per_ship: int = 0  # 0 means no limit


def _stable_hash_keep(mmsi: int, keep_ratio: float) -> bool:
    if keep_ratio >= 1.0:
        return True
    # deterministic pseudo-random in [0,1)
    x = (mmsi * 2654435761) & 0xFFFFFFFF
    return (x / 2**32) < keep_ratio


def _sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
    # enforce numeric and range constraints
    for c in ["Latitude", "Longitude", "SOG", "COG"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # invalid geo -> NaN
    df.loc[(df["Latitude"] < -90) | (df["Latitude"] > 90), "Latitude"] = np.nan
    df.loc[(df["Longitude"] < -180) | (df["Longitude"] > 180), "Longitude"] = np.nan
    # SOG in knots, typical AIS valid range 0..102.2 (102.3 indicates not available in some encodings)
    df.loc[(df["SOG"] < 0) | (df["SOG"] > 102.2), "SOG"] = np.nan
    # COG valid 0..359.9
    df.loc[(df["COG"] < 0) | (df["COG"] >= 360), "COG"] = np.nan

    return df


def _list_month_files(cfg: AISPreprocessConfig) -> List[str]:
    files = []
    for name in os.listdir(cfg.data_dir):
        if name.startswith(cfg.month_prefix) and name.endswith(".csv"):
            files.append(os.path.join(cfg.data_dir, name))
    files = sorted(files)
    if len(files) == 0:
        raise FileNotFoundError(
            f"No AIS csv found under {cfg.data_dir} with prefix {cfg.month_prefix}"
        )
    return files


def _cache_paths(cfg: AISPreprocessConfig, missing_ratio: float, seed: int) -> Tuple[str, str]:
    safe_freq = cfg.freq.replace("/", "_")
    base = (
        f"./data/ais2022_12_freq{safe_freq}_L{cfg.eval_length}_S{cfg.stride}"
        f"_minobs{cfg.min_observed_ratio}_keep{cfg.mmsi_keep_ratio}"
        f"_maxships{cfg.max_ships}_maxwin{cfg.max_windows_per_ship}"
    )
    data_path = base + ".pk"
    # separate cache for gt_mask randomness (depends on missing_ratio+seed)
    mask_path = base + f"_miss{missing_ratio}_seed{seed}.pk"
    return data_path, mask_path


def _build_base_cache(cfg: AISPreprocessConfig) -> Dict:
    """
    Heavy preprocessing (one-time):
    - read month csvs with minimal columns
    - parse timestamp
    - sanitize feature ranges
    - group by MMSI, resample to fixed freq, and cut into windows (observed_data/mask without gt masking)
    Returns a dict with:
      - windows_data: np.ndarray [N, L, K]
      - windows_mask: np.ndarray [N, L, K]
      - mean: np.ndarray [K]
      - std: np.ndarray [K]
      - meta: dict (e.g. feature_names)
    """
    usecols = ["# Timestamp", "MMSI"] + AIS_FEATURES
    files = _list_month_files(cfg)

    # collect per MMSI in a streaming-ish way: load per file then append by group
    per_mmsi: Dict[int, List[pd.DataFrame]] = {}
    for fp in files:
        df = pd.read_csv(fp, usecols=usecols)
        df = df.rename(columns={"# Timestamp": "Timestamp"})
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
        )
        df = df.dropna(subset=["Timestamp", "MMSI"])
        df["MMSI"] = pd.to_numeric(df["MMSI"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["MMSI"])
        df["MMSI"] = df["MMSI"].astype(np.int64)

        df = _sanitize_features(df)

        # subsample MMSI deterministically to control size
        if cfg.mmsi_keep_ratio < 1.0:
            keep = df["MMSI"].apply(lambda x: _stable_hash_keep(int(x), cfg.mmsi_keep_ratio))
            df = df[keep.values]

        # append by group
        for mmsi, g in df.groupby("MMSI"):
            per_mmsi.setdefault(int(mmsi), []).append(
                g[["Timestamp"] + AIS_FEATURES].copy()
            )

    # deterministic order of ships
    mmsi_list = sorted(per_mmsi.keys())
    if cfg.max_ships and cfg.max_ships > 0:
        mmsi_list = mmsi_list[: cfg.max_ships]

    windows_data: List[np.ndarray] = []
    windows_mask: List[np.ndarray] = []

    for mmsi in mmsi_list:
        ship_df = pd.concat(per_mmsi[mmsi], axis=0, ignore_index=True)
        ship_df = ship_df.dropna(subset=["Timestamp"])
        if len(ship_df) == 0:
            continue
        ship_df = ship_df.sort_values("Timestamp")
        ship_df = ship_df.set_index("Timestamp")
        # for duplicate timestamps, keep the last record
        ship_df = ship_df.groupby(level=0).last()

        # resample to fixed freq
        ship_df = ship_df.resample(cfg.freq).last()

        data = ship_df[AIS_FEATURES].to_numpy(dtype=np.float32)
        mask = (~np.isnan(data)).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)

        T = data.shape[0]
        L = cfg.eval_length
        if T < L:
            continue

        # cut into windows
        starts = list(range(0, T - L + 1, cfg.stride))
        if cfg.max_windows_per_ship and cfg.max_windows_per_ship > 0:
            starts = starts[: cfg.max_windows_per_ship]
        for s in starts:
            w_data = data[s : s + L]
            w_mask = mask[s : s + L]
            if w_mask.mean() < cfg.min_observed_ratio:
                continue
            windows_data.append(w_data)
            windows_mask.append(w_mask)

    if len(windows_data) == 0:
        raise RuntimeError(
            "No AIS windows generated. Try decreasing eval_length, decreasing min_observed_ratio, "
            "or setting mmsi_keep_ratio=1.0."
        )

    windows_data = np.stack(windows_data, axis=0)  # [N, L, K]
    windows_mask = np.stack(windows_mask, axis=0)  # [N, L, K]

    # compute normalization using observed entries only
    flat_data = windows_data.reshape(-1, windows_data.shape[-1])
    flat_mask = windows_mask.reshape(-1, windows_mask.shape[-1])
    mean = np.zeros(flat_data.shape[-1], dtype=np.float32)
    std = np.zeros(flat_data.shape[-1], dtype=np.float32)
    for k in range(flat_data.shape[-1]):
        c = flat_data[:, k][flat_mask[:, k] == 1]
        if len(c) == 0:
            mean[k] = 0.0
            std[k] = 1.0
        else:
            mean[k] = c.mean()
            s = c.std()
            std[k] = s if s > 1e-6 else 1.0

    windows_data = ((windows_data - mean) / std) * windows_mask

    return {
        "windows_data": windows_data.astype(np.float32),
        "windows_mask": windows_mask.astype(np.float32),
        "mean": mean,
        "std": std,
        "meta": {"features": AIS_FEATURES, "freq": cfg.freq},
    }


def _make_gt_mask(
    observed_mask: np.ndarray, missing_ratio: float, seed: int
) -> np.ndarray:
    """
    Create gt_mask by randomly masking a fraction of observed positions.
    observed_mask: [N, L, K] in {0,1}
    gt_mask: [N, L, K] in {0,1}, subset of observed_mask
    """
    rng = np.random.RandomState(seed)
    gt = observed_mask.copy()
    N, L, K = gt.shape
    for i in range(N):
        obs = np.where(gt[i].reshape(-1) == 1)[0]
        if len(obs) == 0:
            continue
        m = int(round(len(obs) * missing_ratio))
        if m <= 0:
            continue
        miss = rng.choice(obs, size=m, replace=False)
        flat = gt[i].reshape(-1)
        flat[miss] = 0.0
        gt[i] = flat.reshape(L, K)
    return gt.astype(np.float32)


class AIS2022_Dataset(Dataset):
    def __init__(
        self,
        eval_length: int = 144,
        use_index_list: Optional[np.ndarray] = None,
        missing_ratio: float = 0.1,
        seed: int = 1,
        freq: str = "10min",
        stride: Optional[int] = None,
        min_observed_ratio: float = 0.2,
        mmsi_keep_ratio: float = 1.0,
        max_ships: int = 0,
        max_windows_per_ship: int = 0,
    ):
        self.eval_length = eval_length
        cfg = AISPreprocessConfig(
            freq=freq,
            eval_length=eval_length,
            stride=stride if stride is not None else eval_length,
            min_observed_ratio=min_observed_ratio,
            mmsi_keep_ratio=mmsi_keep_ratio,
            max_ships=max_ships,
            max_windows_per_ship=max_windows_per_ship,
        )
        data_path, mask_path = _cache_paths(cfg, missing_ratio, seed)

        if not os.path.isfile(data_path):
            base = _build_base_cache(cfg)
            with open(data_path, "wb") as f:
                pickle.dump(base, f)
        else:
            with open(data_path, "rb") as f:
                base = pickle.load(f)

        self.observed_values = base["windows_data"]  # [N, L, K]
        self.observed_masks = base["windows_mask"]  # [N, L, K]
        self.mean = base["mean"]
        self.std = base["std"]

        if not os.path.isfile(mask_path):
            self.gt_masks = _make_gt_mask(self.observed_masks, missing_ratio, seed)
            with open(mask_path, "wb") as f:
                pickle.dump(self.gt_masks, f)
        else:
            with open(mask_path, "rb") as f:
                self.gt_masks = pickle.load(f)

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(
    seed: int = 1,
    batch_size: int = 16,
    missing_ratio: float = 0.1,
    eval_length: int = 144,
    freq: str = "10min",
    stride: Optional[int] = None,
    min_observed_ratio: float = 0.2,
    mmsi_keep_ratio: float = 1.0,
    max_ships: int = 0,
    max_windows_per_ship: int = 0,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    dataset = AIS2022_Dataset(
        eval_length=eval_length,
        missing_ratio=missing_ratio,
        seed=seed,
        freq=freq,
        stride=stride,
        min_observed_ratio=min_observed_ratio,
        mmsi_keep_ratio=mmsi_keep_ratio,
        max_ships=max_ships,
        max_windows_per_ship=max_windows_per_ship,
    )
    n = len(dataset)
    indlist = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indlist)

    n_test = int(round(n * test_ratio))
    n_valid = int(round(n * valid_ratio))
    test_index = indlist[:n_test]
    valid_index = indlist[n_test : n_test + n_valid]
    train_index = indlist[n_test + n_valid :]

    train_dataset = AIS2022_Dataset(
        eval_length=eval_length,
        use_index_list=train_index,
        missing_ratio=missing_ratio,
        seed=seed,
        freq=freq,
        stride=stride,
        min_observed_ratio=min_observed_ratio,
        mmsi_keep_ratio=mmsi_keep_ratio,
        max_ships=max_ships,
        max_windows_per_ship=max_windows_per_ship,
    )
    valid_dataset = AIS2022_Dataset(
        eval_length=eval_length,
        use_index_list=valid_index,
        missing_ratio=missing_ratio,
        seed=seed,
        freq=freq,
        stride=stride,
        min_observed_ratio=min_observed_ratio,
        mmsi_keep_ratio=mmsi_keep_ratio,
        max_ships=max_ships,
        max_windows_per_ship=max_windows_per_ship,
    )
    test_dataset = AIS2022_Dataset(
        eval_length=eval_length,
        use_index_list=test_index,
        missing_ratio=missing_ratio,
        seed=seed,
        freq=freq,
        stride=stride,
        min_observed_ratio=min_observed_ratio,
        mmsi_keep_ratio=mmsi_keep_ratio,
        max_ships=max_ships,
        max_windows_per_ship=max_windows_per_ship,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    scaler = torch.from_numpy(dataset.std).float()
    mean_scaler = torch.from_numpy(dataset.mean).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler

