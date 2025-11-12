import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import precision_recall_fscore_support
# 1. Data preprocessing and feature construction (refer to baseline.py, and also add track geometry and rail temperature data processing)
##############################################

def find_best_threshold(model, dataloader, device, thresholds=np.linspace(0, 1, 101)):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            # Expected batch contains: x, seg_ids, reg_target, cls_target (ignore detail_vectors)
            if len(batch) >= 4:
                x, seg_ids, _, cls_target = batch[:4]
            else:
                continue
            x = x.to(device)
            seg_ids = seg_ids.to(device).long()
            if torch.is_tensor(cls_target):
                cls_target = cls_target.to(device).float()
            else:
                cls_target = torch.tensor(cls_target, dtype=torch.float32, device=device)
            # Get classification branch output
            _, pred_damage_cls, _, _, _, _, _ = model(x, seg_ids, mask_geo=False)
            probs = torch.sigmoid(pred_damage_cls).squeeze(-1).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(cls_target.squeeze(-1).cpu().numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    best_threshold = 0.5
    best_f1 = 0.0
    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, preds, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    print(f"best : {best_threshold:.2f}, F1: {best_f1:.4f}")
    return best_threshold

def parse_start_date(s):
    """
    Processing time strings, such as '2019/01/01-2019/01/17', takes the first half as the starting date
    """
    if isinstance(s, str):
        parts = s.split('-')
        return pd.to_datetime(parts[0].strip(), errors='coerce', format='%Y/%m/%d')
    return pd.NaT


def parse_segment(seg):
    """
    Processing interval string
    """
    if isinstance(seg, str):
        parts = seg.split('-')
        return parts[0].strip()
    return seg


def label_encode_types(types_series):
    """
    Label encoding for the segment type (e.g., roadbed, bridge, tunnel, etc.).
    If a null value or unknown condition is encountered, -1 is returned.
    """
    unique_types = types_series.dropna().unique().tolist()
    type2id = {t: i for i, t in enumerate(unique_types)}

    def encode_fn(t):
        if pd.isna(t):
            return -1
        return type2id.get(t, -1)

    encoded = types_series.apply(encode_fn)
    return encoded, type2id


def process_geometry_temperature(df):
    """
    Process the track geometry and temperature data:
    - Track geometry data: Extract the columns named [Left Height (mm), Right Height (mm), Left Track Direction (mm),
    Right Track Direction (mm), Track Gauge (mm), Level (mm), Triangle Pits (mm)],
    and synthesize them into a 7-dimensional vector, saving it to a new column 'geometry_vec'.
    - Track temperature data: Extract the columns [Average Temperature, Average Temperature Difference, Maximum Temperature Difference].
    Here, we select "Average Temperature" as the temperature input for the physical information layer and save it to 'avg_temp'.
    """
    geom_cols = ['Left Height(mm)', 'Right Height(mm)', 'Left Track Direction(mm)',
                 'Right Track Direction (mm)', 'Track Gauge(mm)', 'Level(mm)', 'Triangle Pits(mm)']

    for col in geom_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[geom_cols] = df[geom_cols].fillna(df[geom_cols].mean())

    df['geometry_vec'] = df[geom_cols].apply(lambda row: np.array(row.values, dtype=float), axis=1)

    if 'Average temperature' in df.columns:
        df['avg_temp'] = pd.to_numeric(df['Average temperature'], errors='coerce').fillna(df['Average temperature'].mean())
    else:
        df['avg_temp'] = np.nan

    # Process TQI data: read "TQI (mm)" and normalize it (divide by 12)
    if 'TQI(mm)' in df.columns:
        df['TQI_norm'] = pd.to_numeric(df['TQI(mm)'], errors='coerce').fillna(12) / 12.0
    else:
        df['TQI_norm'] = np.nan

    if 'rain' in df.columns:
        df['rain'] = pd.to_numeric(df['rain'], errors='coerce').fillna(0)
    else:
        df['rain'] = np.nan

    return df


def load_real_data(file_path, encoding="utf-8"):
    """
Load the actual data file, refer to baseline.py
- Processing time: Take the start time
- Processing interval: Take only the left station
- Label encoding interval and type
- Calculate the month (relative to the minimum time of the interval)
- Call process_geometry_temperature to process the track geometry and temperature data
    """
    df = pd.read_csv(file_path, encoding=encoding, sep=",")
    if len(df.columns) == 1:
        new_columns = df.columns[0].split(",")
        df = df[df.columns[0]].str.split(",", expand=True)
        df.columns = new_columns
    df.columns = df.columns.str.strip()
    df['time'] = df['time'].apply(parse_start_date)
    df.sort_values(by=['loc', 'time'], inplace=True)

    df['interval segment_start'] = df['interval segment'].apply(parse_segment)
    station_list=[] # Defined by a specific dataset
    station2id = {st: i for i, st in enumerate(station_list)}
    df['interval_id'] = df['interval_start'].map(lambda x: station2id.get(x, -1))

    df['type_id'], type2id = label_encode_types(df['type'])

    df.dropna(subset=['time'], inplace=True)

    df['Month'] = df.groupby('loc')['time'].transform(lambda x: (x - x.min()) / np.timedelta64(1, 'M'))
    # Processing track geometry and temperature data
    df = process_geometry_temperature(df)

    df['Rolling_damage'] = df.groupby('loc')['damage'].transform(lambda x: x.rolling(window=3, min_periods=1).sum())

    return df

##############################################
# 2. Construct supervised samples (refer to PIBI.py)
##############################################
def parse_damage_details_combined(row, main_time_col='time'):
    """
    Parse damage details:
- If a major damage exists (damage count > 0), parse the major damage fields (discovery time, online time, section type, straight/curved (radius), roadbed/bridge/tunnel/culvert).
- Otherwise, if a minor damage exists (minor damage count > 0), parse the minor damage fields (minor damage discovery time, minor damage online time). Leave all other fields to default.
- Return a 7-dimensional vector in the format: [flag, dt_discover, dt_online, side_id, rail_num, curve_num, base_id].
- Flag = 1 indicates a major damage, flag = 0 indicates a minor damage, and all values are 0 if there is no damage.
    """
    heavy_damage = row.get('damage', 0)
    light_damage = row.get('Minor Injury', 0)
    try:
        heavy_damage = float(heavy_damage)
    except:
        heavy_damage = 0.0
    try:
        light_damage = float(light_damage)
    except:
        light_damage = 0.0

    flag = 0
    dt_discover = 0.0
    dt_online = 0.0
    side_id = -1
    rail_num = 0.0
    curve_num = 0.0
    base_id = -1

    main_t = row[main_time_col]
    if heavy_damage > 0:
        flag = 1
        discover_t_str = row.get('Discovery Time', None)
        discover_t = pd.to_datetime(discover_t_str, errors='coerce', format='%Y/%m/%d %H:%M')
        if pd.isna(discover_t):
            dt_discover = 0.0
        else:
            dt_discover = (discover_t - main_t).total_seconds() / 86400.0
        online_t_str = row.get('Online Time', None)
        online_t = pd.to_datetime(online_t_str, errors='coerce', format='%Y/%m/%d %H:%M')
        if pd.isna(online_t):
            dt_online = 0.0
        else:
            dt_online = (online_t - main_t).total_seconds() / 86400.0
        # side = row.get('', "")
        # if side == "":
        #     side_id = 0
        # elif side == "右":
        #     side_id = 1
        # else:
        #     side_id = -1
        # match_rail = re.search(r"(\d+)", str(rail_str))
        # if match_rail:
        #     rail_num = float(match_rail.group(1))
        curve_str = row.get('Straight/Curve (Radius)', "")
        match_curve = re.search(r"(\d+)", str(curve_str))
        if match_curve:
            curve_num = float(match_curve.group(1))
        base_str = row.get('Roadbed/bridge/tunnel/culvert', "")
        base_map = {"Roadbed": 0, "bridge": 1, "tunnel": 2, "culvert": 3}
        base_id = base_map.get(base_str.strip(), -1)
    elif light_damage > 0:
        flag = 0  #
        discover_t_str = row.get('Minor injury time of discovery', None)
        discover_t = pd.to_datetime(discover_t_str, errors='coerce', format='%Y/%m/%d %H:%M')
        if pd.isna(discover_t):
            dt_discover = 0.0
        else:
            dt_discover = (discover_t - main_t).total_seconds() / 86400.0
        online_t_str = row.get('Minor Injury Time', None)
        online_t = pd.to_datetime(online_t_str, errors='coerce', format='%Y/%m/%d %H:%M')
        if pd.isna(online_t):
            dt_online = 0.0
        else:
            dt_online = (online_t - main_t).total_seconds() / 86400.0
        side_id = -1
        rail_num = 0.0
        curve_num = 0.0
        base_id = -1
    else:
        flag = 0  #

    vec = [flag, dt_discover, dt_online, side_id, rail_num, curve_num, base_id]
    return np.array(vec, dtype=float)


def build_supervised_samples_real(df, km_id, input_window=3, horizon=1, detail_dim=7):
    """
    Constructing supervised samples:
    - Target: Freight volume and damage count at a future time (classification and regression)

    - For each time window, construct the input as follows: the first 7 dimensions are track geometry data (geometry_vec),
    and the last 4 dimensions are environmental features [freight volume, rainfall, avg_temp, TQI_norm].

    Add monthly periodic features (sin, cos), resulting in a final input dimension of 7 + 4 + 2 = 13.
    - Target: Damage count at a future time (supervised regression), and construct a detailed damage information vector
    (calling parse_damage_details_combined).
    """
    # Extract environmental features: For the environmental component, extract [freight volume, rainfall, avg_temp, TQI_norm].
    env_cols = ['cargo', 'rain', 'avg_temp', 'TQI_norm', 'interval segment id', 'type id']
    num_records = len(df)
    env_arr = df[env_cols].values.astype(float)  # shape: (num_records, 4)

    geom_list = df['geometry_vec'].values
    for i, g in enumerate(geom_list):
        if np.isnan(g).any():
            print(f"Warning: geometry_vec at row {i} contains NaN: {g}")
    geom_arr = np.stack(geom_list)  # (num_records, 7)
    if np.isnan(geom_arr).any():
        print("Warning: Found NaN in geom_arr, row indices:", np.where(np.isnan(geom_arr))[0])

    if np.isnan(geom_arr).any():
        col_means = np.nanmean(geom_arr, axis=0)
        inds = np.where(np.isnan(geom_arr))
        geom_arr[inds] = np.take(col_means, inds[1])

    # Combined into total input (geometry first, then environment): (num_records, 11)
    data_arr = np.concatenate([geom_arr, env_arr], axis=1) # (num_records, 11)
    # Adding monthly cycle features (sin, cos)
    month_vals = df['Month'].values
    month_mod = np.floor(month_vals) % 12
    month_sin = np.sin(2 * np.pi * month_mod / 12).reshape(-1, 1)
    month_cos = np.cos(2 * np.pi * month_mod / 12).reshape(-1, 1)
    data_full = np.concatenate([data_arr, month_sin, month_cos], axis=1)  # shape: (num_records, 13)

    #Target: Use "Number of Damages" as the target, and "Freight Volume" as the secondary target
    # (in this example, cumulative values are not used)
    cargo_vals = df['cargo'].values
    damage_vals = df['damage'].values

    X, Y_cargo, Y_dmg_cls, Y_dmg_reg = [], [], [], []
    km_ids, months, detail_vecs = [], [], []
    for t in range(num_records - input_window - horizon + 1):
        in_start = t
        in_end = t + input_window
        out_idx = in_end + horizon - 1
        window_feat = data_full[in_start:in_end, :]  # shape: (input_window, 13)
        y_cargo_val = cargo_vals[out_idx]
        y_damage_val = damage_vals[out_idx]
        dmg_cls = 1 if y_damage_val > 0 else 0
        if y_damage_val > 0:
            row_out = df.iloc[out_idx]
            detail_vec = parse_damage_details_combined(row_out, main_time_col='time')
        else:
            detail_vec = np.zeros((detail_dim,), dtype=float)
        X.append(window_feat)
        Y_cargo.append(y_cargo_val)
        Y_dmg_cls.append(dmg_cls)
        Y_dmg_reg.append(y_damage_val)
        km_ids.append(km_id)
        m_val = df['Month'].iloc[out_idx]
        months.append(int(np.floor(m_val)))
        detail_vecs.append(detail_vec)
    return X, Y_cargo, Y_dmg_cls, Y_dmg_reg, km_ids, months, detail_vecs


def aggregate_dataset_real(df, train=True, train_end=48, input_window=3, horizon=1, detail_dim=7):
    """
    Aggregate the supervised samples of all miles
    """
    X_all, Y_cargo_all, Y_dmg_cls_all, Y_dmg_reg_all = [], [], [], []
    km_ids_all, months_all, detail_all = [], [], []
    for km_id, group in df.groupby('loc'):
        group = group.sort_values(by='time')
        if train:
            group_part = group[group['Month'] < train_end].copy()
        else:
            group_part = group[group['Month'] >= train_end].copy()
        if len(group_part) < input_window + horizon:
            continue
        X, Y_cargo, Y_dmg_cls, Y_dmg_reg, km_ids, months, detail_vecs = build_supervised_samples_real(
            group_part, km_id, input_window, horizon, detail_dim=detail_dim)
        X_all.extend(X)
        Y_cargo_all.extend(Y_cargo)
        Y_dmg_cls_all.extend(Y_dmg_cls)
        Y_dmg_reg_all.extend(Y_dmg_reg)
        km_ids_all.extend(km_ids)
        months_all.extend(months)
        detail_all.extend(detail_vecs)
    min_km = min(km_ids_all)
    km_ids_all = [km - min_km for km in km_ids_all]
    X_all = torch.tensor(X_all, dtype=torch.float)
    Y_cargo_all = torch.tensor(Y_cargo_all, dtype=torch.float).view(-1, 1)
    Y_dmg_cls_all = torch.tensor(Y_dmg_cls_all, dtype=torch.float).view(-1, 1)
    Y_dmg_reg_all = torch.tensor(Y_dmg_reg_all, dtype=torch.float).view(-1, 1)
    km_ids_all = torch.tensor(km_ids_all, dtype=torch.long)
    months_all = torch.tensor(months_all, dtype=torch.long)
    detail_all = torch.tensor(detail_all, dtype=torch.float)
    return X_all, Y_cargo_all, Y_dmg_cls_all, Y_dmg_reg_all, km_ids_all, months_all, detail_all


def normalize_data(X_train, X_test):
    """
    Normalize continuous features: Only the first four environmental features
    (freight volume, rainfall, avg_temp, TQI_norm) are normalized.
     Note: Track geometry data (first 7 dimensions) retains its original scale.
    """
    X_train_np = X_train.numpy()
    X_test_np = X_test.numpy()
    cont_train = X_train_np[:, :, 7:11].reshape(-1, 4)
    print(cont_train)
    mean = cont_train.mean(axis=0)
    std = cont_train.std(axis=0) + 1e-6
    X_train_np[:, :, 7:11] = (X_train_np[:, :, 7:11] - mean) / std
    X_test_np[:, :, 7:11] = (X_test_np[:, :, 7:11] - mean) / std
    return torch.tensor(X_train_np, dtype=torch.float), torch.tensor(X_test_np, dtype=torch.float)

def build_adjacency_matrix_real(df):
    """
    candidate
    """
    km_ids = sorted(df['loc'].unique())
    num_km = len(km_ids)
    km_to_index = {km: i for i, km in enumerate(km_ids)}
    A = np.zeros((num_km, num_km), dtype=float)
    for i in range(num_km):
        A[i, i] = 1
        if i - 1 >= 0:
            A[i, i - 1] = 1
        if i + 1 < num_km:
            A[i, i + 1] = 1
    for sec, group in df.groupby('interval segment'):
        km_list = group['loc'].unique()
        indices = [km_to_index[km] for km in km_list]
        for i in indices:
            for j in indices:
                A[i, j] = 1
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float), num_km, km_ids
