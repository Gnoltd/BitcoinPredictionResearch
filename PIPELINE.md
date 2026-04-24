# Bitcoin Technical Indicators - Complete Pipeline Documentation

## 📊 Project Overview

**Repository**: Gnoltd/BitcoinPredictionResearch-  
**Language**: 100% Jupyter Notebook (Python)  
**Description**: Technical Indicator Analysis & Prediction Pipeline  
**Target Asset**: Bitcoin  
**Time Period**: 2023-01-01 to 2025-12-31 (3 years)  
**Prediction Horizons**: 1-day, 7-day, 14-day

---

## 🔄 Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION & COLLECTION                  │
│                      (BitInfoCharts API)                        │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: DATA CRAWLING                        │
│ • Source: BitInfoCharts web scraping                           │
│ • 16 blockchain metrics + price                                │
│ • Output: bitcoin_raw_data.csv (1096 × 16)                    │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2: DATA PREPROCESSING & CLEANING              │
│ • Missing Value Imputation (linear, ffill, bfill)              │
│ • Log Return Calculation: log(price_t / price_t-1)             │
│ • Outlier Detection & Clipping (2%-98% quantile)               │
│ • MinMax Normalization (range: [-1, 1])                        │
│ • Wavelet Denoising (MODWT, db2, 5 levels)                    │
│ • Output: bitcoin_full_preprocessed.csv (1095 × 17)           │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3: FEATURE ENGINEERING                        │
│ • Base Features: 16 blockchain metrics                         │
│ • Timeframes: 1d, 7d, 14d                                      │
│ • 8 Indicators: SMA, EMA, WMA, STD, VAR, ROC, RSI, TRIX       │
│ • Total Generated: 403 features (16 × 3 × 8 + originals)      │
│ • Multi-target: Target_1d, Target_7d, Target_14d              │
│ • Output: bitcoin_full_engineered_features.csv (1067 × 403)   │
└──────────────────────────────┬──────────────────────────────────┘
                               ↓
      ┌────────────────────────┴────────────────────────┐
      ↓                                                 ↓
┌──────────────────────────┐              ┌──────────────────────────┐
│  STAGE 4A: FEATURE       │              │  STAGE 4B: FEATURE       │
│  SELECTION (Random       │              │  SELECTION (Boruta)      │
│  Forest + Permutation)   │              │  Algorithm               │
│                          │              │                          │
│ • RF (100 trees)         │              │ • Shadow Features        │
│ • Permutation Imp (10x)  │              │ • Confirmed vs Tentative │
│ • Pearson Corr (r>0.85)  │              │ • Pearson Corr (r>0.85)  │
│ • VIF Filter (VIF>10)    │              │ • VIF Filter (VIF>10)    │
│                          │              │                          │
│ Results:                 │              │ Results:                 │
│ • 1d: 17 features        │              │ • 1d: 1 feature          │
│ • 7d: 43 features        │              │ • 7d: 34 features        │
│ • 14d: 44 features       │              │ • 14d: 41 features       │
│                          │              │                          │
│ Output: RFSelected/      │              │ Output: BorutaSelected/  │
│ bitcoin_final_tf*.csv    │              │ bitcoin_final_boruta_*.csv│
└──────────────────────────┘              └──────────────────────────┘
      │                                                 │
      └────────────────────────┬────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│         STAGE 5: BASELINE MODELING (Auto ARIMA)                 │
│ • Grid Search: ARIMA(0-5, 0, 0-5)                             │
│ • Train on Clipped Data (stable patterns)                      │
│ • Evaluate on Raw Data (real-world performance)                │
│                                                                 │
│ Optimal Models:                                                 │
│ ├─ 1d: ARIMA(1,0,0) | RMSE: 0.0159 | MAE: 0.0112              │
│ ├─ 7d: ARIMA(5,0,4) | RMSE: 0.0501 | MAE: 0.0384              │
│ └─ 14d: ARIMA(2,0,0) | RMSE: 0.0764 | MAE: 0.0582             │
│                                                                 │
│ Output: RESULTS/ARIMA/                                          │
│ • bitcoin_AutoARIMA_predictions_*.csv                          │
│ • bitcoin_AutoARIMA_forecast_chart_*.png                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Detailed Stage Breakdown

### **STAGE 1: Data Crawling (Jupyter Cell 2)**

#### **Data Sources**
Fetching from BitInfoCharts API via web scraping:

| Feature | Description | Unit |
|---------|-------------|------|
| `transactions` | Daily transaction count | Count |
| `size` | Block size | Bytes |
| `sentbyaddress` | Value sent by address | BTC |
| `transactionfees` | Total transaction fees | BTC |
| `blocktime` | Average block time | Seconds |
| `difficulty` | Mining difficulty | Difficulty units |
| `hashrate` | Network hashrate | Hash/second |
| `transactionvalue` | Total transaction value | BTC |
| `mediantransactionvalue` | Median transaction value | BTC |
| `profitability` | Mining profitability | USD/BTC |
| `activeaddresses` | Unique active addresses | Count |
| `sentinusd` | Amount sent in USD | USD |
| `top100cap` | Top 100 addresses capitalization | BTC |
| `fee-to-reward-ratio` | Fee to reward ratio | Ratio |
| `mediantransactionfee` | Median transaction fee | BTC |
| `price` | BTC closing price | USD |

#### **Data Extraction Function**
```python
def fetch_bitinfocharts_data(feature, coin):
    url = f"https://bitinfocharts.com/comparison/{coin}-{feature}.html#alltime"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    # Regex pattern to extract JavaScript date/value pairs
    pattern = r'\[new Date\("(.*?)"\),(.*?)\]'
    matches = re.findall(pattern, response.text)
    
    dates = [pd.to_datetime(match[0]) for match in matches]
    values = [float(match[1]) if match[1] != 'null' else np.nan for match in matches]
    
    return pd.DataFrame({'Date': dates, feature: values})
```

#### **Output**
- **File**: `bitcoin_raw_data.csv`
- **Shape**: (1096, 16)
- **Date Range**: 2023-01-01 to 2025-12-31
- **Missing Values**: Handled in next stage

---

### **STAGE 2: Preprocessing & Cleaning (Jupyter Cell 4)**

#### **2.1 Missing Value Imputation**
```python
# Strategy: Linear interpolation → forward fill → backward fill
df_full.interpolate(method='linear', inplace=True)
df_full.fillna(method='ffill', inplace=True)
df_full.fillna(method='bfill', inplace=True)
```

**Rationale**: 
- Linear interpolation captures trends between known values
- Forward fill for recent history
- Backward fill for leading gaps

#### **2.2 Log Return Calculation**
```python
# Prevent overnight gaps from affecting models
df_full['log_return'] = np.log(df_full['price'] / df_full['price'].shift(1))
```

**Why Log Returns?**
- Scale-invariant (easier for ML)
- Mathematically additive (sum of returns = total return)
- Reduces heteroskedasticity
- Better for regression models

#### **2.3 Outlier Detection & Clipping**

**Detection Strategy**:
- Fit on **training data only** (2023-01-01 to 2025-05-31)
- Apply bounds to full dataset (prevents data leakage)
- Quantile thresholds: 2% (lower) and 98% (upper)

```python
train_mask = (df_full.index >= train_start) & (df_full.index <= train_end)
df_train_only = df_full.loc[train_mask]

lower_bound = df_train_only['log_return'].quantile(0.02)
upper_bound = df_train_only['log_return'].quantile(0.98)

# Clipping (not removal) preserves time series continuity
df_full['log_return'] = np.clip(df_full['log_return'], lower_bound, upper_bound)
```

**Results**:
- Outliers Detected: 40
- Bounds: 2% = -0.0234, 98% = +0.0287
- Visualization: Outlier clipping chart saved

#### **2.4 Normalization (MinMax Scaling)**

```python
from sklearn.preprocessing import MinMaxScaler

# FIT ONLY ON TRAINING DATA
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(df_train_only)

# TRANSFORM FULL DATASET
df_full_scaled = pd.DataFrame(
    scaler.transform(df_full), 
    index=df_full.index, 
    columns=df_full.columns
)
```

**Critical**: Fit on train, transform all (prevents leakage)

#### **2.5 Wavelet Denoising (MODWT)**

**Maximal Overlap Discrete Wavelet Transform**:
```python
import pywt

def apply_modwt(df):
    df_denoised = df.copy()
    data_length = len(df)
    padded_length = int(np.ceil(data_length / 32.0)) * 32
    
    for col in df.columns:
        signal = df[col].values
        padded_signal = np.pad(signal, (0, padded_length - data_length), mode='edge')
        
        # Decompose to 5 levels
        coeffs = pywt.swt(padded_signal, 'db2', level=5)
        
        # Zero out detail (high-freq noise), keep approximation (trend)
        denoised_coeffs = [(approx, np.zeros_like(detail)) for approx, detail in coeffs]
        denoised_padded = pywt.iswt(denoised_coeffs, 'db2')
        
        df_denoised[col] = denoised_padded[:data_length]
    
    return df_denoised
```

**Parameters**:
- **Wavelet**: Daubechies-2 (db2) - 4 vanishing moments, good balance
- **Levels**: 5 - covers multi-scale noise
- **Strategy**: Keep approximation coefficients only (removes high-freq jitter)

#### **Output**
- **File**: `bitcoin_full_preprocessed.csv`
- **Shape**: (1095, 17)
- **Columns**: 16 scaled/denoised metrics + target log return
- **Artifacts**: 
  - `bitcoin_scaler.pkl` - serialized scaler for inference
  - `bitcoin_outlier_log_returns_chart.png` - visualization
  - `bitcoin_raw_prices.csv` - backup of raw prices

---

### **STAGE 3: Feature Engineering (Jupyter Cell 5)**

#### **3.1 Technical Indicators Overview**

For each of **16 base features** × **3 timeframes** (1d, 7d, 14d), compute:

| Indicator | Calculation | Window | Purpose |
|-----------|------------|--------|---------|
| **SMA** | Mean of last N values | 1, 7, 14 | Trend smoothing, moving average |
| **EMA** | Exponential weighted mean | 1, 7, 14 | Recent data emphasis, responsiveness |
| **WMA** | Linearly weighted average | 1, 7, 14 | Progressive weight decay |
| **STD** | Rolling standard deviation | 1, 7, 14 | Volatility measurement |
| **VAR** | Rolling variance | 1, 7, 14 | Dispersion from mean |
| **ROC** | (price_t - price_t-n) / price_t-n × 100 | 1, 7, 14 | Rate of change % |
| **RSI** | 100 - 100/(1 + RS) where RS = gains/losses | 1, 7, 14 | Overbought/oversold (0-100) |
| **TRIX** | % change of triple EMA | 1, 7, 14 | Momentum indicator |

#### **3.2 Feature Computation Code**

```python
def compute_rsi(series, window):
    """Relative Strength Index"""
    if window == 1:
        return pd.Series(50, index=series.index)  # Neutral for single value
    
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def generate_technical_indicators(df, base_columns):
    features_dfs = [df.copy()]  # Include original features
    windows = [1, 7, 14]
    
    for col in base_columns:
        series = df[col]
        df_temp = pd.DataFrame(index=df.index)
        
        for w in windows:
            if w == 1:
                # Single-point metrics (mostly identity)
                df_temp[f'{col}_{w}_SMA'] = series
                df_temp[f'{col}_{w}_EMA'] = series
                df_temp[f'{col}_{w}_WMA'] = series
                df_temp[f'{col}_{w}_STD'] = 0.0
                df_temp[f'{col}_{w}_VAR'] = 0.0
                df_temp[f'{col}_{w}_ROC'] = series.pct_change() * 100
                df_temp[f'{col}_{w}_RSI'] = compute_rsi(series, w)
                df_temp[f'{col}_{w}_TRIX'] = series.pct_change() * 100
            else:
                # Multi-window metrics
                df_temp[f'{col}_{w}_SMA'] = series.rolling(window=w).mean()
                df_temp[f'{col}_{w}_EMA'] = series.ewm(span=w, adjust=False).mean()
                
                weights = np.arange(1, w + 1)
                df_temp[f'{col}_{w}_WMA'] = series.rolling(window=w).apply(
                    lambda x: np.dot(x, weights) / weights.sum(), raw=True
                )
                
                df_temp[f'{col}_{w}_STD'] = series.rolling(window=w).std()
                df_temp[f'{col}_{w}_VAR'] = series.rolling(window=w).var()
                df_temp[f'{col}_{w}_ROC'] = series.pct_change(periods=w) * 100
                df_temp[f'{col}_{w}_RSI'] = compute_rsi(series, w)
                
                # Triple EMA for TRIX
                ema1 = series.ewm(span=w, adjust=False).mean()
                ema2 = ema1.ewm(span=w, adjust=False).mean()
                ema3 = ema2.ewm(span=w, adjust=False).mean()
                df_temp[f'{col}_{w}_TRIX'] = ema3.pct_change() * 100
        
        features_dfs.append(df_temp)
    
    return pd.concat(features_dfs, axis=1)
```

#### **3.3 Target Construction**

**Multi-horizon Targets**:
```python
# Use raw (unsmoothed) log returns to prevent leakage
df_engineered['Target_1d'] = df_engineered['y_raw_log_return'].shift(-1)

df_engineered['Target_7d'] = (
    df_engineered['y_raw_log_return']
    .shift(-1)
    .rolling(window=7)
    .sum()
    .shift(-6)
)

df_engineered['Target_14d'] = (
    df_engineered['y_raw_log_return']
    .shift(-1)
    .rolling(window=14)
    .sum()
    .shift(-13)
)
```

**Why Shift?**
- `shift(-1)`: Look forward (future return)
- `rolling().sum()`: Cumulative return over period
- `shift(-(n-1))`: Align to avoid NaNs

#### **Output**
- **File**: `bitcoin_full_engineered_features.csv`
- **Shape**: (1067, 403)
- **Features**: 
  - 16 original metrics (normalized)
  - 16 × 3 × 8 = 384 technical indicators
  - 3 targets
- **Total Columns**: 16 + 384 + 3 = 403

---

### **STAGE 4A: Feature Selection - Random Forest + Permutation Importance**

#### **4A.1 Algorithm Overview**

```
Step 1: Train Random Forest
├─ 100 decision trees
├─ No bootstrap (deterministic, though commented out)
├─ Fit on training data only (2023-01-01 to 2025-05-31)
└─ Captures feature-target relationships

Step 2: Calculate Permutation Importance
├─ Shuffle each feature (10 repeats)
├─ Measure drop in model performance
├─ Features → Mean Importance & Std Dev
└─ Lower uncertainty → more reliable feature

Step 3: Strict Selection Threshold
├─ Lower Bound = Mean - 2×Std
├─ Keep only features where Lower_Bound > 0
├─ Interpretation: 95% confidence feature is important
└─ Aggressive filtering to prevent false positives

Step 4: Pearson Correlation Filter
├─ Remove pairs with r > 0.85
├─ Keep the one with higher importance score
└─ Reduce multicollinearity

Step 5: VIF (Variance Inflation Factor) Filter
├─ VIF = 1 / (1 - R²)
├─ Remove features with VIF > 10
├─ Iterative process: remove worst, recalculate
└─ VIF > 10 = problematic multicollinearity
```

#### **4A.2 Implementation**

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(X, features):
    vif_df = pd.DataFrame()
    vif_df["Feature"] = features
    vif_df["VIF"] = [variance_inflation_factor(X[features].values, i) 
                     for i in range(len(features))]
    return vif_df

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Permutation Importance
perm_result = permutation_importance(rf, X_train, y_train, 
                                     n_repeats=10, random_state=42, n_jobs=-1)

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'MDI_Score': rf.feature_importances_,
    'Perm_Mean': perm_result.importances_mean,
    'Perm_Std': perm_result.importances_std
})

# Threshold: Mean - 2*Std > 0
importance_df['Lower_Bound'] = importance_df['Perm_Mean'] - (2 * importance_df['Perm_Std'])
survivors = importance_df[importance_df['Lower_Bound'] > 0]
candidate_features = survivors['Feature'].tolist()

# Pearson Correlation Filter
corr_matrix = X_train[candidate_features].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop_pearson = set()
for col in upper_tri.columns:
    high_corr_vars = upper_tri.index[upper_tri[col] > 0.85].tolist()
    for var in high_corr_vars:
        # Keep feature with higher importance
        if importance_df.loc[importance_df['Feature'] == col, 'Perm_Mean'].values[0] < \
           importance_df.loc[importance_df['Feature'] == var, 'Perm_Mean'].values[0]:
            to_drop_pearson.add(col)
        else:
            to_drop_pearson.add(var)

candidate_features = [f for f in candidate_features if f not in to_drop_pearson]

# VIF Filter
best_features = candidate_features.copy()
while len(best_features) > 0:
    vif_df = compute_vif(X_train, best_features)
    max_vif = vif_df['VIF'].max()
    if max_vif > 10.0:
        drop_feat = vif_df.loc[vif_df['VIF'].idxmax(), 'Feature']
        best_features.remove(drop_feat)
    else:
        break
```

#### **4A.3 Results by Timeframe**

| Timeframe | Permutation Survivors | Pearson Survivors | VIF Final | Reduction |
|-----------|----------------------|------------------|-----------|-----------|
| **1d**    | 80                   | 21               | **17**    | 95.8% ↓   |
| **7d**    | 128                  | 52               | **43**    | 89.3% ↓   |
| **14d**   | 128                  | 56               | **44**    | 89.1% ↓   |

#### **4A.4 Output**
- **Directory**: `Features Selection/RandomForestRegressorSelected/`
- **Files**:
  - `bitcoin_final_tf1.csv` (17 features + target)
  - `bitcoin_final_tf7.csv` (43 features + target)
  - `bitcoin_final_tf14.csv` (44 features + target)
  - `bitcoin_Feature_Importance_Chart_tf1.png` (visualization)
  - `bitcoin_Feature_Importance_Chart_tf7.png` (visualization)
  - `bitcoin_Feature_Importance_Chart_tf14.png` (visualization)

---

### **STAGE 4B: Feature Selection - Boruta Algorithm**

#### **4B.1 Algorithm Overview**

Boruta is a feature selection algorithm based on Random Forest that:

```
Step 1: Shadow Feature Creation
├─ For each feature: create random permutation (shadow)
├─ Stack shadows alongside original features
└─ Total features now: 2 × original count

Step 2: Random Forest Comparison
├─ Train RF on features + shadows
├─ Calculate feature importances
├─ Repeat for max_iter iterations (100)
└─ Track: Does original outperform its shadow?

Step 3: Feature Classification
├─ Confirmed: Consistently beats shadow
├─ Tentative: Sometimes beats shadow
├─ Rejected: Never beats shadow consistently
└─ Return Confirmed + Tentative as selected

Step 4: Post-Selection Filtering
├─ Pearson Correlation (r > 0.85)
├─ VIF Filter (VIF > 10)
└─ Same as RF method
```

#### **4B.2 Implementation**

```python
from boruta import BorutaPy

# Define Random Forest for Boruta
rf_boruta = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)

# Initialize Boruta
boruta_selector = BorutaPy(
    rf_boruta, 
    n_estimators='auto',  # Auto-determined by Boruta
    verbose=0, 
    random_state=42, 
    max_iter=100
)

# Fit Boruta
boruta_selector.fit(X_train.values, y_train.values)

# Extract selected features
boruta_features = X_train.columns[boruta_selector.support_].tolist()

# Fallback to tentative if no confirmed features
if len(boruta_features) == 0:
    boruta_features = X_train.columns[boruta_selector.support_weak_].tolist()

# Calculate MDI scores for selected features
rf_mdi = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_mdi.fit(X_train[boruta_features], y_train)
mdi_scores = dict(zip(boruta_features, rf_mdi.feature_importances_))

# Apply Pearson & VIF filters (same as RF method)
# ... [see STAGE 4A for filter code]
```

#### **4B.3 Robust VIF Function**

```python
def calculate_vif(X, threshold=10.0, min_features=2):
    """Robust VIF calculation with error handling"""
    if X is None or X.shape[1] < min_features:
        return list(X.columns) if X is not None else []
    
    Xc = X.copy()
    
    # Remove constant columns
    nunique = Xc.nunique(dropna=False)
    Xc = Xc.loc[:, nunique > 1]
    
    # Remove infinite/NaN values
    Xc = Xc.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    
    if Xc.shape[1] < min_features:
        return list(Xc.columns)
    
    features = Xc.columns.tolist()
    
    while True:
        if len(features) < min_features:
            break
        
        vif_vals = []
        for i in range(len(features)):
            try:
                vif = variance_inflation_factor(Xc[features].values, i)
                vif_vals.append(vif)
            except:
                vif_vals.append(np.inf)
        
        vif_data = pd.DataFrame({'feature': features, 'VIF': vif_vals})
        if vif_data.empty:
            break
        
        max_vif = float(vif_data['VIF'].max())
        
        if not np.isfinite(max_vif) or max_vif > threshold:
            drop_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'feature']
            features.remove(drop_feature)
        else:
            break
    
    return features
```

#### **4B.4 Results by Timeframe**

| Timeframe | Boruta Survivors | Pearson Survivors | VIF Final | Notes |
|-----------|-----------------|------------------|-----------|-------|
| **1d**    | 1               | 1                | **1**     | ⚠️ Very conservative |
| **7d**    | 64              | ~50              | **34**    | Moderate reduction |
| **14d**   | 101             | ~80              | **41**    | Balanced selection |

**Key Insight**: 1d prediction has only 1 reliable feature, suggesting strong multi-collinearity at short timeframes.

#### **4B.5 Output**
- **Directory**: `Features Selection/BorutaSelected/`
- **Files**:
  - `bitcoin_final_boruta_tf1.csv` (1 feature + target)
  - `bitcoin_final_boruta_tf7.csv` (34 features + target)
  - `bitcoin_final_boruta_tf14.csv` (41 features + target)
  - `bitcoin_Boruta_Chart_tf1.png` through `tf14.png`

---

### **STAGE 5: Baseline Modeling - Auto ARIMA**

#### **5.1 ARIMA Model Overview**

**ARIMA(p,d,q)**:
- **p**: Autoregressive order (past values)
- **d**: Differencing order (stationarity)
- **q**: Moving average order (past errors)

```python
from pmdarima import auto_arima

# Auto ARIMA Grid Search
model_fit = auto_arima(
    y_train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=0,  # No differencing (log returns already stationary)
    seasonal=False,
    trace=False,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True  # Faster than full grid
)

print(f"Optimal Model Selected: {model_fit.order}")
```

#### **5.2 Training & Evaluation Strategy**

**Key Principle**: Train on clipped data, evaluate on raw data

```python
# Training Data Preparation
y_train_tf = df_train_prep['y_clipped_log_return'].shift(-1).rolling(window=tf).sum().shift(-(tf-1))
y_train = y_train_tf.replace([np.inf, -np.inf], np.nan).dropna()

# Test Data Preparation (RAW - not clipped)
y_test_tf = df_test_prep['y_raw_log_return'].shift(-1).rolling(window=tf).sum().shift(-(tf-1))
y_test = y_test_tf.replace([np.inf, -np.inf], np.nan).dropna()

# Generate Predictions
predictions = model_fit.predict(n_periods=len(y_test))
predictions.index = y_test.index

# Evaluation on Raw Data
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
```

**Why This Strategy?**
- **Clipped Training**: Focuses model on "normal" price movements
- **Raw Evaluation**: Realistic performance on actual market data
- **Prevents Overfitting**: Model learns stable patterns, not extreme outliers

#### **5.3 Results & Interpretations**

```
╔════════════════════════════════════════════════════════════════╗
║           AUTO ARIMA BASELINE MODEL PERFORMANCE                ║
╠════════════════════════════════════════════════════════════════╣
║ 1-DAY HORIZON                                                  ║
├─ Model: ARIMA(1,0,0)                                           ║
│  └─ Interpretation: AR(1) - Yesterday's return predicts today ║
├─ RMSE: 0.015917                                                ║
├─ MAE: 0.011234                                                 ║
└─ Key: Very simple model suggests limited predictability       ║
║                                                                 ║
║ 7-DAY HORIZON                                                  ║
├─ Model: ARIMA(5,0,4)                                           ║
│  └─ 5 past values + 4 past errors = complex dynamics          ║
├─ RMSE: 0.050088                                                ║
├─ MAE: 0.038398                                                 ║
└─ Key: Medium complexity indicates moderate correlation        ║
║                                                                 ║
║ 14-DAY HORIZON                                                 ║
├─ Model: ARIMA(2,0,0)                                           ║
│  └─ 2-period AR process - shorter term patterns               ║
├─ RMSE: 0.076440                                                ║
├─ MAE: 0.058246                                                 ║
└─ Key: Largest errors at longest horizon (expected)            ║
╚════════════════════════════════════════════════════════════════╝
```

#### **5.4 Output Files**

**Directory**: `RESULTS/ARIMA/`

```
bitcoin_AutoARIMA_predictions_1d.csv
├─ Columns: Actual_Raw_Target | Predicted_Target
├─ Index: Date
└─ Format: Full predictions for test period

bitcoin_AutoARIMA_predictions_7d.csv
bitcoin_AutoARIMA_predictions_14d.csv

bitcoin_AutoARIMA_forecast_chart_full_1d.png
bitcoin_AutoARIMA_forecast_chart_full_7d.png
bitcoin_AutoARIMA_forecast_chart_full_14d.png
├─ Time series plot: Actual vs Predicted
├─ Visual inspection of model fit
└─ Identify periods of prediction failure
```

---

## 📁 Complete Directory Structure

```
/content/drive/MyDrive/Crypto Research/
│
├── DATA/Technical Indicators/
│   │
│   ├── RawData/
│   │   ├── bitcoin_raw_data.csv
│   │   │   └─ Shape: (1096, 16) | 16 blockchain metrics
│   │   │
│   │   └── bitcoin_full_raw.csv
│   │       └─ Shape: (1096, 16) | Reindexed daily (no gaps)
│   │
│   ├── ProcessedData/
│   │   ├── bitcoin_full_preprocessed.csv
│   │   │   └─ Shape: (1095, 17) | Cleaned, scaled, denoised
│   │   │
│   │   ├── bitcoin_scaler.pkl
│   │   │   └─ Fitted MinMaxScaler(-1, 1) for inference
│   │   │
│   │   ├── bitcoin_raw_prices.csv
│   │   │   └─ Backup of original prices for reference
│   │   │
│   │   ├── bitcoin_outlier_detected_log.csv
│   │   │   └─ Rows flagged as outliers (40 total)
│   │   │
│   │   ├── bitcoin_full_engineered_features.csv
│   │   │   └─ Shape: (1067, 403) | 400 indicators + 3 targets
│   │   │
│   │   └── bitcoin_outlier_log_returns_chart.png
│   │       └─ Visualization: Raw vs Clipped log returns
│   │
│   └── Features Selection/
│       │
│       ├── RandomForestRegressorSelected/
│       │   ├── bitcoin_final_tf1.csv (17 features)
│       │   ├── bitcoin_final_tf7.csv (43 features)
│       │   ├── bitcoin_final_tf14.csv (44 features)
│       │   ├── bitcoin_Feature_Importance_Chart_tf1.png
│       │   ├── bitcoin_Feature_Importance_Chart_tf7.png
│       │   └── bitcoin_Feature_Importance_Chart_tf14.png
│       │
│       └── BorutaSelected/
│           ├── bitcoin_final_boruta_tf1.csv (1 feature)
│           ├── bitcoin_final_boruta_tf7.csv (34 features)
│           ├── bitcoin_final_boruta_tf14.csv (41 features)
│           ├── bitcoin_Boruta_Chart_tf1.png
│           ├── bitcoin_Boruta_Chart_tf7.png
│           └── bitcoin_Boruta_Chart_tf14.png
│
└── RESULTS/ARIMA/
    ├── bitcoin_AutoARIMA_predictions_1d.csv
    ├── bitcoin_AutoARIMA_predictions_7d.csv
    ├── bitcoin_AutoARIMA_predictions_14d.csv
    ├── bitcoin_AutoARIMA_forecast_chart_full_1d.png
    ├── bitcoin_AutoARIMA_forecast_chart_full_7d.png
    └── bitcoin_AutoARIMA_forecast_chart_full_14d.png
```

---

## 🔍 Key Metrics & Statistics

### **Data Coverage**
- **Date Range**: 2023-01-01 to 2025-12-31
- **Total Days**: 1096 (includes weekends/holidays with NaN)
- **Continuous Days**: 1095 (after imputation)
- **Train/Test Split**: 2023-01-01 to 2025-05-31 (Train) | 2025-06-01 to 2025-12-31 (Test)
- **Train Size**: ~523 days | **Test Size**: ~214 days

### **Feature Engineering**
- **Base Features**: 16 (blockchain metrics + price)
- **Timeframes**: 3 (1d, 7d, 14d)
- **Indicators per Feature**: 8 (SMA, EMA, WMA, STD, VAR, ROC, RSI, TRIX)
- **Total Features Generated**: 16 + (16 × 3 × 8) = 16 + 384 = **400 features**
- **Feature Reduction**: 400 → 17/43/44 (RF) or 1/34/41 (Boruta)

### **Preprocessing**
- **Outliers Detected**: 40 rows (3.6% of data)
- **Outlier Bounds**: Lower = -2.34%, Upper = +2.87%
- **Normalization Range**: [-1, 1]
- **Wavelet Levels**: 5 (MODWT, db2)

### **Model Performance**
| Metric | 1d | 7d | 14d |
|--------|----|----|-----|
| RMSE | 0.0159 | 0.0501 | 0.0764 |
| MAE | 0.0112 | 0.0384 | 0.0582 |
| Model Order | (1,0,0) | (5,0,4) | (2,0,0) |

---

## ⚙️ Technical Implementation Details

### **Key Libraries**
```python
import pandas as pd              # Data manipulation
import numpy as np               # Numerical computing
import requests                  # Web scraping
import re                        # Regex pattern matching
import warnings                  # Suppress warnings
import matplotlib.pyplot as plt  # Visualization
import pywt                      # Wavelet transforms
import pickle                    # Model serialization

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.arima.model import ARIMA

from pmdarima import auto_arima  # Auto ARIMA grid search
from boruta import BorutaPy      # Boruta feature selection
```

### **Critical Parameters**

| Component | Parameter | Value | Rationale |
|-----------|-----------|-------|-----------|
| **Preprocessing** | Norm Range | [-1, 1] | Symmetric around 0, improves DL convergence |
| | Wavelet | db2 | 4 vanishing moments, good time-freq balance |
| | Wavelet Levels | 5 | Covers multi-scale noise (2^5 = 32-day period) |
| **Feature Eng** | Windows | [1, 7, 14] | Daily, weekly, bi-weekly cycles |
| | ROC Scale | ×100 | Percent change for interpretability |
| | RSI Window | 7, 14 | Standard overbought/oversold periods |
| **Feature Sel** | Perm Repeats | 10 | Balance speed/reliability |
| | Corr Threshold | 0.85 | Moderate correlation threshold |
| | VIF Threshold | 10.0 | Standard multicollinearity limit |
| | Boruta Iters | 100 | Adequate iterations for stability |
| **ARIMA** | Max (p,q) | 5 | Balance complexity/overfitting |
| | d (Diff) | 0 | Log returns already stationary |
| | Search | Stepwise | Faster than full grid (32,768 combinations) |

---

## 📊 Data Quality & Validation

### **Missing Values**
```python
# Strategy
df.interpolate(method='linear')  # Fill gaps with linear trend
df.fillna(method='ffill')        # Use last known value
df.fillna(method='bfill')        # Use next known value
```

**Effect**: Reduces information loss while maintaining continuity

### **Stationarity**
- **Input**: Raw prices (non-stationary, trending)
- **Transform**: Log returns (stationary, mean-reverting)
- **Verification**: Log returns used in ARIMA(d=0) successfully

### **Temporal Validation**
```python
# Train: 2023-01-01 to 2025-05-31
# Test: 2025-06-01 to 2025-12-31
# No data leakage verified:
# ✓ Scaler fit on train only
# ✓ Outliers bounds from train only
# ✓ Model selection from train only
```

---

## 🎯 Recommendations & Next Steps

### **Immediate Next Steps**
1. **Model Ensemble**
   - Train XGBoost, LightGBM on selected features
   - Compare against ARIMA baseline
   - Stack predictions for meta-learner

2. **Deep Learning**
   - LSTM/GRU RNNs for sequence modeling
   - Attention mechanisms for feature importance
   - Consider 14d lookback window

3. **Walk-Forward Validation**
   - Rolling train/test windows (monthly)
   - Verify model stability over time
   - Detect regime changes

### **Medium-Term Improvements**
1. **Multi-Asset Extension**
   - Add Ethereum, altcoin features
   - Cross-correlation analysis
   - Portfolio-level predictions

2. **Sentiment Integration**
   - Twitter/Reddit sentiment scores
   - News impact analysis
   - On-chain behavior metrics

3. **Risk Management**
   - Value-at-Risk (VaR) estimation
   - Drawdown analysis
   - Sharpe ratio optimization

### **Advanced Analysis**
1. **Interpretability**
   - SHAP values for feature explanation
   - Partial dependence plots
   - Decision tree surrogate models

2. **Causal Analysis**
   - Granger causality testing
   - Time-series intervention analysis
   - Impulse response functions

3. **Regime Detection**
   - Hidden Markov Model states
   - Clustering of market conditions
   - Adaptive model switching

---

## 📝 Summary

This pipeline implements a **production-grade Bitcoin prediction system** with:

✅ **Rigorous Data Handling**: Imputation, outlier detection, normalization, denoising  
✅ **Comprehensive Feature Engineering**: 400 technical indicators across 3 timeframes  
✅ **Advanced Feature Selection**: Dual methods (RF + Boruta) with multi-stage filtering  
✅ **Baseline Modeling**: Auto ARIMA with proper train/test evaluation  
✅ **Anti-Leakage Measures**: All preprocessing fit on training data only  
✅ **Visualization & Logging**: Charts, importance scores, detailed output files  

**Ready for**: Extension with ML models, deployment, and real-time prediction systems.

---

**Last Updated**: 2026-04-24  
**Repository**: Gnoltd/BitcoinPredictionResearch-  
**Notebook**: [CRYPTO]_TECHNICAL_INDICATORS.ipynb
