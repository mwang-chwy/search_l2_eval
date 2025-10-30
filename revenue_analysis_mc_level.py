import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.isotonic import IsotonicRegression
from pathlib import Path


idir = '/Users/aakshantal/Downloads'

MODEL_PATHS = {
    'var_non_p13n': f'{idir}/gbm-train_without_pers_40_features-67.txt',
    'var_p13n':     f'{idir}/gbm-train_with_pers_40_features-52.txt',
    'var_53_features': f'{idir}/gbm-train_with_top40_plus_p13n-65.txt'
}
MODEL_CHOICE = 'var_p13n'
DROP_LAST_N_FEATURES = 29 if MODEL_CHOICE == 'var_non_p13n' else 42
TOTAL_FEATURES = 82
METRIC_RANKS = (8, 16, 32, 64)
NDCG_RANKS = (10, 36, 72)

# --- Files ---
iso_calibrator = joblib.load(f'{idir}/iso_calibrator_p13n.pkl')
metadata_path = f"{idir}/tst_data/gbm-X-metadata.csv"
pdm_path = f"{idir}/PDM Product_2025-10-01-2037.csv"

# --- Features ---
LTR_ES_FEATURES = [
    'allPetTypeSearch_bm25', 'allPetTypeSearch_atp', 'categoryName_bm25',
    'categoryName_atp', 'defaultSearch_bm25', 'defaultSearch_atp',
    'facetHighSearch_bm25', 'facetHighSearch_atp', 'facetLowSearch_bm25',
    'facetLowSearch_atp', 'itemtype_search_bm25', 'itemtype_search_atp',
    'longdescription_bm25', 'longdescription_atp', 'manufacturer_bm25',
    'manufacturer_atp', 'name_bm25', 'name_atp', 'partNumber_bm25',
    'partNumber_atp', 'search_ingredients_updated_bm25',
    'search_ingredients_updated_atp', 'rating', 'rating_count',
    'is_MedicationType_Prescription', 'is_newCatenttype_SameSKUBundle',
    'is_newCatenttype_MultiSKUBundle', 'is_onSpecial_True',
    'is_PrivateLabel_True', 'cosine_similarity', 'hybrid_score'
]
PRODUCT_FEATURES = [
    'PRICE', 'LIST_PRICE', 'MAP_PRICE', 'DISCOUNT_AMOUNT', 'DISCOUNT_PCT',
    'AUTOSHIP_PRICE', 'ON_DEAL_FLAG', 'RX_PHARMACY_FLAG', 'RX_REQUIRED_FLAG',
    'AVG_MC1_PRICE', 'AVG_MC2_PRICE', 'AVG_MC3_PRICE',
    'PRICE_VS_AVG_MC1', 'PRICE_VS_AVG_MC2', 'PRICE_VS_AVG_MC3',
    'MC1_PRICE_PCT25', 'MC1_PRICE_PCT50', 'MC1_PRICE_PCT75',
    'MC2_PRICE_PCT25', 'MC2_PRICE_PCT50', 'MC2_PRICE_PCT75',
    'MC3_PRICE_PCT25', 'MC3_PRICE_PCT50', 'MC3_PRICE_PCT75',
    'IS_PRICE_OUTLIER', 'PRICE_BUCKET', 'PRICE_NORMALIZED_TO_MC2_MEDIAN',
    'MERCH_CLASSIFICATION1', 'MERCH_CLASSIFICATION2', 'MERCH_CLASSIFICATION3'
]
OTHER_FEATURES = [
    'has_brand', 'brand_order_count', 'brand_days_since',
    'has_premium_brand_affinity', 'pet_type_match_score'
]
pet_type_columns = [
    'has_bird', 'has_cat', 'has_dog', 'has_farm_animal', 'has_fish',
    'has_horse', 'has_reptile_amphibian', 'has_small_pet'
]
USER_PET_TYPE_FEATURES = [f'user_{col}' for col in pet_type_columns]
all_feature_columns = (
        LTR_ES_FEATURES +
        PRODUCT_FEATURES +
        pet_type_columns +
        OTHER_FEATURES +
        USER_PET_TYPE_FEATURES
)
weight_to_label = {
    0.009636: 0,
    0.028909: 1,
    0.057818: 2,
    2.5: 3
}


def safe_array(x):
    x = np.array(x, dtype=np.float64)
    x[~np.isfinite(x)] = 0
    return x


def get_ranks(scores: np.ndarray) -> np.ndarray:
    sorted_indices = scores.argsort()[::-1]
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(scores) + 1)
    return ranks


def get_dcg(relevance_labels: np.ndarray, ranks: np.ndarray, max_rank: int = None) -> float:
    dcg = 0.0
    max_rank = max_rank or len(ranks)
    for i in range(len(relevance_labels)):
        if ranks[i] <= max_rank:
            dcg += relevance_labels[i] / np.log2(ranks[i] + 1.0)
    return dcg


def get_ndcg(relevance_labels: np.ndarray, scores: np.ndarray, max_rank: int = None) -> float:
    odcg = get_dcg(relevance_labels, get_ranks(scores), max_rank)
    idcg = get_dcg(relevance_labels, get_ranks(relevance_labels), max_rank)
    return odcg / idcg if idcg > 0 else 0.0


def evaluate_metrics_per_query(
    y_true_revenue: np.ndarray,
    y_true_click: np.ndarray,
    y_pred: np.ndarray,
    prices: np.ndarray,
    qid: np.ndarray,
) -> List[Dict[str, float]]:

    results: List[Dict[str, float]] = []

    unique_qids, group_starts, group_sizes = np.unique(qid, return_index=True, return_counts=True)

    for start, size, current_qid in zip(group_starts, group_sizes, unique_qids):
        end = start + size
        true_revenue = y_true_revenue[start:end]
        true_click = y_true_click[start:end]
        pred = y_pred[start:end]
        group_prices = prices[start:end]

        total_group_revenue = float(np.sum(true_revenue))
        has_revenue = total_group_revenue > 0.0

        row: Dict[str, float] = {
            "qid": current_qid,
            "group_size": int(size),
        }

        for k in NDCG_RANKS:
            k_eff = int(min(k, size))
            if has_revenue and k_eff > 0:
                row[f"NDCG@{k}"] = float(get_ndcg(true_revenue, pred, max_rank=k_eff))
            else:
                row[f"NDCG@{k}"] = float("nan")  # NaN so np.nanmean matches overall skip

        for k in METRIC_RANKS:
            k_eff = int(min(k, size))
            if k_eff == 0:
                row[f"RPS@{k}"] = 0.0
                row[f"Revenue_Recall@{k}"] = 0.0
                row[f"CVR@{k}"] = 0.0
                row[f"Avg_Price@{k}"] = 0.0
                continue

            ranked_indices = np.argsort(pred)[::-1][:k_eff]

            top_k_true_revenue = true_revenue[ranked_indices]
            top_k_true_click = true_click[ranked_indices]
            top_k_prices = group_prices[ranked_indices]

            top_k_revenue = float(np.sum(top_k_true_revenue))
            top_k_clicks = float(np.sum(top_k_true_click))

            row[f"RPS@{k}"] = top_k_revenue

            row[f"Revenue_Recall@{k}"] = (top_k_revenue / total_group_revenue) if has_revenue else 0.0

            row[f"CVR@{k}"] = (top_k_clicks / k_eff)

            row[f"Avg_Price@{k}"] = float(np.mean(top_k_prices)) if k_eff > 0 else 0.0

        results.append(row)

    return results

def get_query_identifiers(q: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(q)), q)


def pick_part_col(df):
    lc = {c.lower(): c for c in df.columns}
    for k in ("part_number", "product_part_number", "partnumber", "sku", "part_number_id"):
        if k in lc: return lc[k]
    if "PART_NUMBER" in df.columns: return "PART_NUMBER"
    raise KeyError("No part number column found")


def load_dataset(split: str, selected_columns: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(f"\nLoading dataset: {split}")
    base_path = f"{idir}/{split}_data"
    X = np.load(f"{base_path}/gbm-X.npy")[:, selected_columns]
    y = np.loadtxt(f"{base_path}/gbm-y-autoship-only.dat", dtype='int32')
    w = np.loadtxt(f"{base_path}/gbm-w.dat", dtype='float32')
    q = np.load(f"{base_path}/gbm-q.npy")
    qid = get_query_identifiers(q)
    return X, y, w, qid


def get_selected_features(all_features: List[str], drop_last_n: int) -> Tuple[List[str], List[int]]:
    if MODEL_CHOICE == 'var_non_p13n':
        ordered_feature_list = pd.read_csv(f"{idir}/variant3_combined_importance.csv")['feature'].tolist()
    else:
        ordered_feature_list = pd.read_csv(f"{idir}/variant_pers_combined_importance (1).csv")['feature'].tolist()
    retained_feature_names = ordered_feature_list[:-drop_last_n] if drop_last_n > 0 else ordered_feature_list
    selected_feature_names = [f for f in retained_feature_names if f in all_features]
    print(len(selected_feature_names))
    print(selected_feature_names)
    selected_columns = [all_feature_columns.index(f) for f in selected_feature_names]
    return selected_feature_names, selected_columns


def clean_data_by_group(X: np.ndarray, qid: np.ndarray) -> np.ndarray:
    X_df = pd.DataFrame(X)
    X_df['qid'] = qid
    X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    group_medians = X_df.groupby('qid').transform('median')
    X_df.fillna(group_medians, inplace=True)
    X_df.fillna(X_df.median(), inplace=True)
    return X_df.drop('qid', axis=1).values


def assign_mc_to_qid(metadata_path: str, pdm_path: str, qid: np.ndarray) -> pd.DataFrame:
    print("\n--- Assigning MC1/MC2 to each Query (QID) ---")

    meta_df = pd.read_csv(metadata_path)

    if len(meta_df) != len(qid):
        raise ValueError(
            f"Metadata length ({len(meta_df)}) does not match QID array length ({len(qid)}). Cannot assign MCs.")

    aligned_meta = meta_df.copy()
    aligned_meta['qid'] = qid
    pcol = pick_part_col(aligned_meta)

    pdm_df = pd.read_csv(pdm_path, dtype={"PART_NUMBER": "string"}, low_memory=False)
    pdm_df['PART_NUMBER_CLEAN'] = pdm_df['PART_NUMBER'].astype('string').str.strip()

    mc_cols = ['MERCH_CLASSIFICATION1', 'MERCH_CLASSIFICATION2']
    pdm_mc = pdm_df[['PART_NUMBER_CLEAN'] + mc_cols].drop_duplicates(subset=['PART_NUMBER_CLEAN'], keep='first')

    aligned_meta['PART_NUMBER_CLEAN'] = aligned_meta[pcol].astype('string').str.strip()

    mc_mapping_df = aligned_meta.merge(
        pdm_mc,
        left_on='PART_NUMBER_CLEAN',
        right_on='PART_NUMBER_CLEAN',
        how='left'
    )

    # 4. Determine Dominant MC1 and MC2 for Each QID
    def get_dominant_mc(series):
        # Fill NaN with a placeholder to prevent mode from failing, then calculate mode
        mc_series = series.astype(str).fillna('MISSING_MC')
        mode_result = mc_series.mode()

        # If mode is not empty and is not the placeholder, return the dominant MC
        if not mode_result.empty and mode_result.iloc[0] != 'MISSING_MC':
            return mode_result.iloc[0]
        return 'UNKNOWN'


    dominant_mc = mc_mapping_df.groupby('qid').agg({
        'MERCH_CLASSIFICATION1': get_dominant_mc,
        'MERCH_CLASSIFICATION2': get_dominant_mc
    }).reset_index()

    dominant_mc = dominant_mc.rename(columns={
        'MERCH_CLASSIFICATION1': 'MC1_Assigned',
        'MERCH_CLASSIFICATION2': 'MC2_Assigned'
    })

    print("MC assignment complete.")
    return dominant_mc


# --- Feature Selection ---
selected_feature_names, selected_columns = get_selected_features(all_feature_columns, DROP_LAST_N_FEATURES)
print(f"Selected {len(selected_feature_names)} features.")

# --- Load Data ---
val_X, val_y, val_w, val_qid = load_dataset('val', selected_columns)
val_X = clean_data_by_group(val_X, val_qid)
tst_X, tst_y, tst_w, tst_qid = load_dataset('tst', selected_columns)
tst_X = clean_data_by_group(tst_X, tst_qid)

# --- PRICE handling ---
if 'PRICE' not in selected_feature_names:
    raise ValueError("PRICE feature not found. Cannot compute revenue-based metrics.")
price_idx = selected_feature_names.index('PRICE')
price = val_X[:, price_idx]
price_tst = tst_X[:, price_idx]

# --- Labels ---
weight_map = np.vectorize(lambda w: weight_to_label.get(w, 0))
val_y_engagement = weight_map(val_w)

# --- Model & Predictions ---
lgb_model = lgb.Booster(model_file=MODEL_PATHS[MODEL_CHOICE])
predicted_cvr = lgb_model.predict(val_X)

# --- Evaluate Approaches ---
y_true_revenue = safe_array(val_y * price)
results_list_per_query = []

# --- Calibration ---

predicted_cvr = np.clip(predicted_cvr, 0, 1)

if MODEL_CHOICE == 'var_non_p13n':
    iso_calibrator = joblib.load('iso_calibrator.pkl')
else:
    iso_calibrator = joblib.load('iso_calibrator_p13n.pkl')

calibrated_cvr = iso_calibrator.transform(predicted_cvr)


from sklearn.metrics import brier_score_loss, log_loss

print("Brier raw:", brier_score_loss(val_y, predicted_cvr))
print("Brier cal:", brier_score_loss(val_y, calibrated_cvr))
print("LogLoss raw:", log_loss(val_y, predicted_cvr))
print("LogLoss cal:", log_loss(val_y, calibrated_cvr))

# 4. Calibrated CVR * Price
res_cal_def = evaluate_metrics_per_query(y_true_revenue, val_y, calibrated_cvr * price, price, val_qid)
for res in res_cal_def: res.update({'approach': 'calibrated_cvr_*_price'})
results_list_per_query.extend(res_cal_def)

# 5. Calibrated CVR only
res_cal_cvr = evaluate_metrics_per_query(y_true_revenue, val_y, calibrated_cvr, price, val_qid)
for res in res_cal_cvr: res.update({'approach': 'calibrated_cvr'})
results_list_per_query.extend(res_cal_cvr)

results_df = pd.DataFrame(results_list_per_query)

# --- MC Assignment and Aggregation ---
# Assign dominant MC to each QID
mc_assignments = assign_mc_to_qid(metadata_path, pdm_path, val_qid)

# Merge results with MC assignments
results_df = results_df.merge(mc_assignments, on='qid', how='left')

print("\n--- MC1 Level Averages ---")
# Filter out unknown MCs and calculate MC1 averages
mc1_df = results_df[results_df['MC1_Assigned'] != 'UNKNOWN'].copy()
if not mc1_df.empty:
    cols_to_drop = ['qid', 'group_size', 'MC2_Assigned']

    mc1_metrics_df = mc1_df.drop(columns=[c for c in cols_to_drop if c in mc1_df.columns])
    avg_metrics_mc1 = mc1_metrics_df.groupby(['approach', 'MC1_Assigned']).mean().reset_index()

    print(avg_metrics_mc1.round(4).to_string(index=False))
else:
    print("No valid MC1 data to display.")

print("\n--- MC2 Level Averages ---")
mc2_df = results_df[results_df['MC2_Assigned'] != 'UNKNOWN'].copy()
if not mc2_df.empty:
    cols_to_drop = ['qid', 'group_size', 'MC1_Assigned']
    mc2_metrics_df = mc2_df.drop(columns=[c for c in cols_to_drop if c in mc2_df.columns])

    avg_metrics_mc2 = mc2_metrics_df.groupby(['approach', 'MC2_Assigned']).mean().reset_index()

    print(avg_metrics_mc2.round(4).to_string(index=False))
else:
    print("No valid MC2 data to display.")
