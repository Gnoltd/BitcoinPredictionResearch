#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import requests
from sklearn.preprocessing import MinMaxScaler

RAW_FEATURES = [
    "transactions",
    "size",
    "sentbyaddress",
    "transactionfees",
    "blocktime",
    "difficulty",
    "hashrate",
    "transactionvalue",
    "mediantransactionvalue",
    "profitability",
    "activeaddresses",
    "sentinusd",
    "top100cap",
    "fee-to-reward-ratio",
    "mediantransactionfee",
]

INDICATORS = ["SMA", "EMA", "WMA", "STD", "VAR", "ROC", "RSI", "TRIX"]
WINDOWS = [1, 7, 14]

DEFAULT_START = "2016-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_TRAIN_START = "2016-01-01"
DEFAULT_TRAIN_END = "2023-12-31"


def fetch_bitinfocharts_data(feature: str, coin: str, session: requests.Session) -> tuple[pd.DataFrame, str]:
    url = f"https://bitinfocharts.com/comparison/{coin}-{feature}.html#alltime"
    response = session.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    content_hash = hashlib.sha256(response.content).hexdigest()
    pattern = r'\[new Date\("(.*?)"\),(.*?)\]'
    matches = re.findall(pattern, response.text)
    dates = [pd.to_datetime(match[0]) for match in matches]
    values = [float(match[1]) if match[1] != "null" else np.nan for match in matches]
    df = pd.DataFrame({"Date": dates, feature: values}).set_index("Date")
    return df, content_hash


def crawl_raw_data(
    coin: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, Path, Path, str]:
    features = features or RAW_FEATURES + ["price"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    retrieval_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.DataFrame()
    hash_log: dict[str, str] = {}

    with requests.Session() as session:
        for feature in features:
            df_temp, content_hash = fetch_bitinfocharts_data(feature, coin, session)
            hash_log[feature] = content_hash
            df_raw = df_temp if df_raw.empty else df_raw.join(df_temp, how="outer")

    if "price" in df_raw.columns:
        df_raw = df_raw.rename(columns={"price": "Close"})

    raw_file = raw_dir / f"{coin}_raw_data_{run_id}.csv"
    df_raw.to_csv(raw_file)

    log_file = raw_dir / f"{coin}_retrieval_log_{run_id}.csv"
    log_row = {
        "retrieval_timestamp_utc": retrieval_timestamp,
        "coin": coin,
        "start_date": start_date,
        "end_date": end_date,
        **{f"sha256_{key}": value for key, value in hash_log.items()},
    }
    pd.DataFrame([log_row]).to_csv(log_file, index=False)
    return df_raw, raw_file, log_file, run_id


def normalize_daily_index(df_raw: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    df_full = df_raw.reindex(full_date_range)
    df_full.index.name = "Date"
    return df_full


def apply_modwt(df: pd.DataFrame) -> pd.DataFrame:
    df_denoised = df.copy()
    data_length = len(df)
    padded_length = int(np.ceil(data_length / 32.0)) * 32
    for col in df.columns:
        signal = df[col].values
        padded_signal = np.pad(signal, (0, padded_length - data_length), mode="edge")
        coeffs = pywt.swt(padded_signal, "db2", level=5)
        denoised_coeffs = [(approx, np.zeros_like(detail)) for approx, detail in coeffs]
        denoised_padded = pywt.iswt(denoised_coeffs, "db2")
        df_denoised[col] = denoised_padded[:data_length]
    return df_denoised


def preprocess_data(
    df_full: pd.DataFrame,
    train_start: str,
    train_end: str,
    processed_dir: Path,
    coin: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if "price" in df_full.columns and "Close" not in df_full.columns:
        df_full = df_full.rename(columns={"price": "Close"})

    if "Close" not in df_full.columns:
        raise ValueError("Close price column is missing from the raw dataset.")

    y_raw_close = df_full["Close"].copy()
    df_full.interpolate(method="linear", inplace=True)
    df_full.fillna(method="ffill", inplace=True)
    df_full.fillna(method="bfill", inplace=True)

    df_full = df_full.dropna(subset=["Close"])
    y_raw_close = y_raw_close.reindex(df_full.index)

    train_mask = (df_full.index >= train_start) & (df_full.index <= train_end)
    df_train_only = df_full.loc[train_mask]
    lower_bound = df_train_only["Close"].quantile(0.02)
    upper_bound = df_train_only["Close"].quantile(0.98)

    outliers = df_full[
        (df_full["Close"] < lower_bound) | (df_full["Close"] > upper_bound)
    ][["Close"]].rename(columns={"Close": "raw_close_price"})
    outliers.to_csv(processed_dir / f"{coin}_outlier_detected_log.csv")

    df_full["Close"] = df_full["Close"].clip(lower=lower_bound, upper=upper_bound)

    feature_cols = [col for col in df_full.columns if col != "Close"]
    df_features = df_full[feature_cols].copy()
    df_features_train = df_train_only[feature_cols]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(df_features_train)
    df_features_scaled = pd.DataFrame(
        scaler.transform(df_features),
        index=df_features.index,
        columns=df_features.columns,
    )

    with open(processed_dir / f"{coin}_scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)

    df_features_preprocessed = apply_modwt(df_features_scaled)
    df_full_preprocessed = df_features_preprocessed.copy()
    df_full_preprocessed["y_raw_close_price"] = y_raw_close

    df_full_preprocessed.to_csv(processed_dir / f"{coin}_full_preprocessed.csv")
    y_raw_close.to_frame(name="raw_close_price").to_csv(
        processed_dir / f"{coin}_raw_prices.csv"
    )

    return df_full_preprocessed, y_raw_close, feature_cols


def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    if window == 1:
        return pd.Series(50, index=series.index)
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).fillna(50)


def generate_technical_indicators(df: pd.DataFrame, base_columns: list[str]) -> pd.DataFrame:
    features_dfs = [df.copy()]
    for col in base_columns:
        series = df[col]
        df_temp = pd.DataFrame(index=df.index)
        for window in WINDOWS:
            if window == 1:
                df_temp[f"{col}_{window}_SMA"] = series
                df_temp[f"{col}_{window}_EMA"] = series
                df_temp[f"{col}_{window}_WMA"] = series
                df_temp[f"{col}_{window}_STD"] = 0.0
                df_temp[f"{col}_{window}_VAR"] = 0.0
                df_temp[f"{col}_{window}_ROC"] = series.pct_change() * 100
                df_temp[f"{col}_{window}_RSI"] = compute_rsi(series, window)
                df_temp[f"{col}_{window}_TRIX"] = series.pct_change() * 100
            else:
                df_temp[f"{col}_{window}_SMA"] = series.rolling(window=window).mean()
                df_temp[f"{col}_{window}_EMA"] = series.ewm(span=window, adjust=False).mean()

                weights = np.arange(1, window + 1)
                df_temp[f"{col}_{window}_WMA"] = series.rolling(window=window).apply(
                    lambda x: np.dot(x, weights) / weights.sum(),
                    raw=True,
                )

                df_temp[f"{col}_{window}_STD"] = series.rolling(window=window).std()
                df_temp[f"{col}_{window}_VAR"] = series.rolling(window=window).var()
                df_temp[f"{col}_{window}_ROC"] = series.pct_change(periods=window) * 100
                df_temp[f"{col}_{window}_RSI"] = compute_rsi(series, window)

                ema1 = series.ewm(span=window, adjust=False).mean()
                ema2 = ema1.ewm(span=window, adjust=False).mean()
                ema3 = ema2.ewm(span=window, adjust=False).mean()
                df_temp[f"{col}_{window}_TRIX"] = ema3.pct_change() * 100

        features_dfs.append(df_temp)
    return pd.concat(features_dfs, axis=1)


def engineer_features(
    df_preprocessed: pd.DataFrame,
    feature_cols: list[str],
    processed_dir: Path,
    coin: str,
) -> pd.DataFrame:
    raw_close = df_preprocessed["y_raw_close_price"].copy()
    df_engineered = generate_technical_indicators(df_preprocessed[feature_cols], feature_cols)
    df_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)

    df_engineered["Target_1d"] = raw_close.shift(-1)
    df_engineered["Target_7d"] = raw_close.shift(-7)
    df_engineered["Target_14d"] = raw_close.shift(-14)
    df_engineered.dropna(inplace=True)

    df_engineered.to_csv(processed_dir / f"{coin}_full_engineered_features.csv")
    return df_engineered


def validate_engineered_columns(df_engineered: pd.DataFrame, base_features: list[str]) -> tuple[int, list[str], list[str]]:
    expected_columns = list(base_features)
    for col in base_features:
        for window in WINDOWS:
            for indicator in INDICATORS:
                expected_columns.append(f"{col}_{window}_{indicator}")
    expected_columns.extend(["Target_1d", "Target_7d", "Target_14d"])

    expected_set = set(expected_columns)
    actual_set = set(df_engineered.columns)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    return len(expected_columns), missing, extra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Bitcoin technical indicator features.")
    parser.add_argument("--coin", default="bitcoin")
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Base directory for raw/processed outputs.",
    )
    parser.add_argument(
        "--raw-file",
        help="Optional path to an existing raw CSV to skip crawling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_file:
        df_raw = pd.read_csv(args.raw_file, index_col=0, parse_dates=True)
        run_id = "manual"
    else:
        df_raw, raw_file, log_file, run_id = crawl_raw_data(
            coin=args.coin,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=output_dir,
        )
        print(f"Raw crawl saved to: {raw_file}")
        print(f"Retrieval log saved to: {log_file}")

    df_full = normalize_daily_index(df_raw, args.start_date, args.end_date)
    df_full.to_csv(output_dir / "raw" / f"{args.coin}_full_raw_{run_id}.csv")

    df_preprocessed, _, feature_cols = preprocess_data(
        df_full=df_full,
        train_start=args.train_start,
        train_end=args.train_end,
        processed_dir=processed_dir,
        coin=args.coin,
    )

    df_engineered = engineer_features(
        df_preprocessed=df_preprocessed,
        feature_cols=feature_cols,
        processed_dir=processed_dir,
        coin=args.coin,
    )

    expected_count, missing, extra = validate_engineered_columns(df_engineered, feature_cols)
    print(f"Engineered columns: {len(df_engineered.columns)} (expected {expected_count})")
    if missing:
        print(f"Missing columns ({len(missing)}): {missing}")
    if extra:
        print(f"Extra columns ({len(extra)}): {extra}")
    if not missing and not extra:
        print("Column names match expected pipeline output.")


if __name__ == "__main__":
    main()
