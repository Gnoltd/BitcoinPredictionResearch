

This document speci!es a complete, reproducible six-stage machine-learning pipeline for multi-
horizon Bitcoin price forecasting. The design addresses three categories of systematic failure common
in cryptocurrency forecasting literature: (1) target non-stationarity causing heteroscedastic loss
gradients, (2) look-ahead bias introduced by non-causal signal !lters and contemporaneous macro
alignment, and (3) multicollinearity inflation distorting feature importance estimates. Core
innovations relative to prior versions include Fractional Di"erentiation [1] as the stationary target
transform, causal-only denoising via 1-D Filter-bank DWT, distance correlation for non-Gaussian
dependency screening [2], strict ordering of VIF pre-!ltering before BorutaShap [3], and Split
Conformal Prediction [4] calibrated in price-space after inverse transform — correcting a previously
identi!ed ordering error. All design decisions are referenced to peer-reviewed literature. An explicit
naive baseline and Diebold–Mariano statistical testing [5] are mandated at the evaluation stage.
## CONTENTS
## TECHNICAL RESEARCH DOCUMENT · QUANTITATIVE FINANCE · TIME SERIES
## FORECASTING
Institutional-Grade Alpha Factory
## Pipeline
for Bitcoin Price Forecasting
Version 2.1 — Causality-Strict, Stationary-First Architecture
## VERSION
v2.1-final
## ASSET
BTC/USD Daily
## PERIOD
## 2016 – 2025
## HORIZON
## 1d · 7d · 14d

§1Data Collection & Temporal AlignmentSources · Macro phase-lock · Imputation
§2Target Engineering & Causal DenoisingFracDiff d* scan · DWT causal · Hampel causal
§3Feature Engineering & Orthogonal Pre-
filtering
TA · Sentiment · TDA · VIF · Distance
correlation
§4BorutaShap Feature SelectionWalk-forward CV · Confirmed set
§5Latent Compression & Deep Learning CoreTCN-AE sequential · TFT + BiLSTM parallel
§6Ensemble, Inverse Transform &
## Uncertainty
Ridge meta-learner · Inverse FracDiff ·
Conformal CP
§7Monitoring, Regime Detection & BacktestHMM · ADWIN · CUSUM · DM test · Baselines
§RReferencesAPA 7 · 18 citations
## PIPELINE OVERVIEW
## →→
## ←←
## ⚠ DATA SPLIT — FIXED ACROSS ALL STAGES
Train: 2016-01-01 → 2022-12-31 (2,557 days) · Calibration: 2023-01-01 → 2023-12-31
(365 days) · Test: 2024-01-01 → 2025-03-31 (≈ 455 days). No data from Calibration or Test
enters any scaler fit, feature selection, or model training loop. Calibration set is exclusively
reserved for Conformal Prediction score calibration (§6.3).
## Data Collection & Temporal Alignment
## §1
## Data Collection
OHLCV · On-chain · Macro ·
## Sentiment
## §2
## Target + Denoising
FracDiff d* · Causal DWT ·
## Hampel
## §3
## Feature Engineering
TA · TDA · Sentiment ·
VIF/dcor filter
## §4
BorutaShap
## Walk-forward · Confirmed
set
## §5
TCN-AE + TFT/BiLSTM
Sequential train · Latent
## 16d
## §6
Ensemble + CP
Ridge · Inv-FracDiff → CP
(price space)
## §7
## Monitoring · Regime · Backtest
HMM regime · ADWIN/CUSUM drift · DM test · Baselines · Sharpe/MDD/Calmar

## STAGE 1
## Data Collection & Temporal Alignment
Goal — leakage-free, phase-locked multi-source dataset at daily frequency
## 1.1 DATA SOURCES
## CATEGORYSERIESSOURCEFREQUENCY
## LAG
## APPLIED
## Price &
## Volume
## OHLCV, VWAP
Binance / CoinbaseDaily
## 0
## On-chain
Exchange Netflow, NVT, Active
Addr, SOPR
GlassnodeDaily
## 0
## Macro
DXY, VIX, Gold, US 10Y Yield, Fed
## Funds Rate
Bloomberg / FREDDaily
+1 day
## Sentiment
FinBERT score, Fear & Greed Index
CryptoCompare /
AltIndex
## Daily
+1 day
## TDA
## (computed)
## Persistent Entropy, Betti-0,
## Amplitude
Computed from
## OHLCV
## Daily
## 0
## 1.2 MACRO PHASE-LOCK
All Macro and Sentiment series are lagged by exactly +1 trading day. Rationale: at BTC daily close
time t, macro data for day t (e.g., VIX close, Fed statement) may not be fully disseminated
across all exchanges. Using lag t-1 for all external series ensures strict no-look-ahead alignment.
## IMPLEMENTATION NOTE — ON-CHAIN DATA
On-chain metrics computed by data providers (e.g., NVT, SOPR) may themselves use rolling
windows. Verify that provider computation is finalized by 00:00 UTC of the following day
before using at lag 0. If finalization time is uncertain, apply lag +1 to on-chain metrics as
well.
## 1.3 MISSING VALUE IMPUTATION
Cubic spline interpolation restricted to past-only anchor points. For any missing value at index t,
the spline is fit exclusively on [t-k, t-1] with k = 10. Forward extrapolation is used only if t is

at the boundary; backward extrapolation is prohibited.
For Macro series with known bank holiday gaps (e.g., VIX not reported on US holidays), forward-fill
from the most recent valid observation, capped at 3 consecutive days. Beyond 3 days, flag as NaN
and trigger data quality alert.
## 1.4 DATA QUALITY CHECKLIST
## CHECKTHRESHOLDACTION ON BREACH
Missing rate per series
## < 2%
Warn; interpolate if causal
Staleness — on-chain
≤ 1 day lag behind OHLCV
Drop day; do not impute
Price zero / negative
## Never
Pipeline halt
Volume = 0 for > 3 consecutive
days
Never in 2017+
Flag; likely exchange gap
Correlation of raw BTC price with
target
Run ADF; p-value expected > 0.05
on raw
Expected; confirms non-
stationarity
## STAGE 2
## Target Engineering & Causal Denoising
v2.1 fix
Goal — stationary target with preserved memory; strictly causal noise removal
## 2.1 FRACTIONAL DIFFERENTIATION — TARGET TRANSFORM
Raw log-price p_t = \ln(\text{Close}_t) is non-stationary I(1). Log-returns achieve stationarity
but discard all long-memory structure. Fractional differentiation [1] finds the minimum order d^*
that achieves stationarity while retaining maximum memory.
FracDiff(p, d) = Σ_{k=0}^{T-1} w_k(d) · p_{t-k}
where w_k(d) = (-1)^k · C(d, k) = (d · (d-1) · ... · (d-k+1)) / k!
Target for horizon h: Y_t^{(h)} = FracDiff(p_{t+h} - p_t, d*)
Procedure for selecting d* required

This is a mandatory calibration step, not a fixed hyperparameter. Execute on Train set only:
Algorithm 2.1 — Minimum-d* scan (Lopez de Prado, 2018, Ch. 5)
- Set candidate_d = [0.05, 0.10, 0.15, ..., 1.00]
- For each d in candidate_d:
a. Compute X_d = FracDiff(ln(Close_train), d) using fixed-width window W=100
b. Run Augmented Dickey-Fuller test on X_d
c. Record ADF statistic and p-value
- d* = min{d : ADF p-value < 0.05}
- Verify: Pearson corr(FracDiff(Close, d*), Close) > 0.95
(confirms memory retention; if < 0.95, increase d* by 0.05 and re-check)
- Store d* as a pipeline constant for all subsequent stages
## EXPECTED RANGE
For daily BTC log-price 2016–2024, empirical d* typically falls in [0.35, 0.55] based on prior
applications of FracDiff to financial log-prices. If d* > 0.8, this indicates the series is closer
to I(1) than expected — verify ADF implementation and data stationarity before proceeding.
## 2.2 MULTI-HORIZON TARGET CONSTRUCTION
# Compute for each forecast horizon h ∈ {1, 7, 14} for h in [1, 7, 14]:
log_fwd_return = ln(Close[t+h]) - ln(Close[t]) # log forward return Y_h[t] =
FracDiff(log_fwd_return, d=d_star) # apply stored d* # Note: FracDiff is applied
to the log forward return, not raw log-price. # This ensures the target encodes
directional signal over horizon h # while retaining long-memory structure at
order d*.
## 2.3 CAUSAL DENOISING
## VMD REMOVED — NOT CAUSAL BY DESIGN
Variational Mode Decomposition optimizes globally over the full input sequence. No causal
variant exists with a standard implementation. All VMD usage is removed from this pipeline.
Savitzky-Golay filter is also removed unless explicitly applied with center=False and future-

symmetric window disabled.
2.3.1 Discrete Wavelet Transform — causal filter-bank
Replace VMD with a 1-D causal DWT filter-bank using Daubechies-4 wavelet. Causality is enforced
by restricting the convolution to past samples only — equivalent to causal FIR filtering at each
decomposition level.
# Causal DWT denoising on price series (applied before feature engineering)
import pywt def causal_dwt_denoise(series, wavelet='db4', level=3,
threshold_mode='soft'): # Pad left with reflect padding (past values only) to
handle boundary pad_len = 2**level * len(pywt.Wavelet(wavelet).filter_bank[0])
padded = np.pad(series, (pad_len, 0), mode='reflect') coeffs =
pywt.wavedec(padded, wavelet, level=level, mode='periodization') # Universal
threshold (Donoho & Johnstone, 1994) sigma = mad(coeffs[-1]) / 0.6745 threshold =
sigma * sqrt(2 * log(len(series))) coeffs_thresh = [pywt.threshold(c, threshold,
mode=threshold_mode) for c in coeffs] denoised_padded =
pywt.waverec(coeffs_thresh, wavelet, mode='periodization') return
denoised_padded[pad_len:] # strip padding — causal output
2.3.2 Causal Hampel filter — outlier detection
Hampel filter with strictly past-only rolling median. Window [t-k, t], k=7. Threshold: 3σ of the
rolling MAD.
def causal_hampel(series, window=7, n_sigma=3): result = series.copy() for t in
range(window, len(series)): window_slice = series[t - window : t] # [t-k, t-1] —
no t+k med = np.median(window_slice) mad = np.median(np.abs(window_slice - med))
if abs(series[t] - med) > n_sigma * 1.4826 * mad: result[t] = med # replace
outlier with past median return result
## 2.4 NORMALIZATION
RobustScaler (median + IQR) fit exclusively on Train set indices. Applied in expanding-window
fashion during walk-forward validation: scaler is re-fit at the end of each expanding train window
using only data available up to that fold's cutoff date. Never fit on Validation or Test data.
## Feature Engineering & Orthogonal Pre-!ltering

## STAGE 3
## Feature Engineering & Orthogonal Pre-!ltering
v2.1 fix
Goal — generate ~400 candidate features; remove multicollinear redundancy before
BorutaShap
## 3.1 FEATURE GROUPS
## GROUPFEATURES (EXAMPLES)COUNTDEPENDENCY METRIC
TA — TrendEMA(9,21,50,200), MACD(12,26,9), ADX(14)~80Pearson r
## TA —
## Momentum
RSI(7,14,21), Stoch(14), Williams %R, CCI~60Pearson r
TA — VolatilityATR(7,14), Bollinger width, Keltner width~40Pearson r
TA — VolumeOBV, VWAP, CMF, MFI(14)~30Pearson r
On-chainNVT, Exchange Netflow, SOPR, Active Addr
## (1d/7d/14d)
~40Distance correlation
SentimentFinBERT score (mean, std, skew), FG Index, FG
momentum
~25Distance correlation
MacroDXY, VIX, Gold, 10Y Yield, their 5d/10d returns~30Pearson r
TDAPersistent Entropy (H0, H1), Betti-0, Amplitude,
Lifetime std
~20Distance correlation
CalendarDay-of-week, Month, Quarter, Bitcoin halving dummy~10Pearson r (after one-
hot)
## 3.2 DEPENDENCY SCREENING — TWO-METRIC PROTOCOL
## V2.1
Two screening metrics are applied depending on feature distribution. Pearson r is valid only for
approximately Gaussian, linear-dependent pairs. For heavy-tailed or non-Gaussian features
(Sentiment, TDA, On-chain), distance correlation [2] is used as it detects all forms of dependency
without distributional assumptions.
## DISTANCE CORRELATION — DEFINITION
For two random vectors X, Y of length n: dcor(X,Y) ∈ [0,1], where 0 indicates complete

independence (for all distributions) and 1 indicates a deterministic relationship. Unlike
Pearson r, dcor = 0 ⟺ X ⊥ Y. Implementation: dcor Python package (Ramos-Carreño, 2023)
or scipy.stats.energy_distance as approximation. Reference: Székely & Rizzo (2007) [2].
Algorithm 3.1 — Orthogonal pre-filtering (executed on Train set only)
- Separate features into Gaussian group (TA, Macro, Calendar) and Non-Gaussian
group (TDA, Sentiment, On-chain)
- Gaussian group:
a. Compute pairwise Pearson |r| matrix
b. For any pair with |r| > 0.85: retain feature with higher variance; drop the
other
c. Compute VIF for remaining features via OLS regression
d. Compute condition number κ = sqrt(λ_max / λ_min) of X^T X
e. Remove features with VIF > 10 AND κ contribution > 30 iteratively (Belsley et
al., 1980 [6])
- Non-Gaussian group:
a. Compute pairwise distance correlation matrix
b. For any pair with dcor > 0.80: retain feature with higher marginal
information (entropy); drop other
- Recombine surviving features → pre-filtered feature set F_pre (expected ~150–
200 features)
- Verify: max pairwise |r| in F_pre < 0.85; max pairwise dcor in F_pre < 0.80
## ⚠ VIF THRESHOLD RATIONALE
VIF threshold is set at 10 (not 5) for this pipeline. Financial time series TA indicators
inherently carry moderate autocorrelation that inflates VIF relative to cross-sectional
regression settings. Hair et al. (1995) threshold of 5 was specified for cross-sectional OLS —
applying it directly to TA features with 400+ candidates will remove informative features.
VIF = 10 is the threshold cited in Marquaridt (1970) [7] for severe multicollinearity. The
condition-number criterion (Belsley et al., 1980 [6]) provides additional confirmation.
BorutaShap Feature Selection

## STAGE 4
BorutaShap Feature Selection
Goal — identify confirmed predictive features from pre-filtered set F_pre
## 4.1 BORUTASHAP ALGORITHM
BorutaShap [3] extends Boruta [8] by replacing Mean Decrease Impurity (MDI) with mean absolute
SHAP values E[|φ_i|] as the importance measure. This satisfies four game-theoretic axioms
(Efficiency, Symmetry, Dummy, Additivity) and is consistent even under feature correlation — which,
after pre-filtering, is moderate rather than eliminated.
Importance_i = E_x[|φ_i(x)|] where φ_i = Shapley value of feature i
Shadow feature j' = shuffle(feature j) → all information destroyed
MHSA_i = max shadow importance across all shadow features
Decision: if Importance_i > MHSA_i with p < 0.05 (binomial test) →
## Confirmed
if Importance_i < MHSA_i with p < 0.05 → Rejected
otherwise → Tentative (iterate)
## 4.2 WALK-FORWARD BORUTASHAP PROTOCOL
BorutaShap is run inside the walk-forward cross-validation loop. Feature selection is re-executed at
each fold using only data available up to the fold cutoff. Final confirmed set F* is the intersection of
confirmed features across ≥ 70% of folds. This prevents selection bias from a single train/validation
split.
Algorithm 4.1 — Walk-forward BorutaShap
- Define folds: expanding window, min train = 730 days, step = 90 days
- For each fold k:
a. Fit RobustScaler on Train_k
b. Run BorutaShap(max_iter=100, percentile=80, pvalue=0.05) on Train_k with
## F_pre
c. Record Confirmed_k set
- F* = {feature f : f ∈ Confirmed_k for ≥ 70% of folds}

- Expected |F*|: 40–80 features (from ~150–200 pre-filtered)
## STAGE 5
## Latent Compression & Deep Learning Core
v2.1 fix
Goal — compress TA features into interpretable latent space; parallel multi-
horizon forecasting
## 5.1 TCN-AUTOENCODER — SEQUENTIAL TRAINING PROTOCOL
TCN-AE is trained sequentially and separately from the forecasting models. This resolves the
training objective conflict identified in the v2.0 review: if TCN-AE and TFT/BiLSTM share a joint loss,
the encoder distorts toward minimizing forecast error rather than preserving signal structure.
## SEQUENTIAL TRAINING — TWO-PHASE PROTOCOL
Phase 1: Train TCN-AE on TA features from F* only (not Macro/Sentiment/TDA). Loss = MSE
reconstruction of input TA features. Train until validation reconstruction loss plateaus
(patience=15 epochs). Freeze all encoder weights after Phase 1.
Phase 2: Train TFT and BiLSTM using frozen TCN-AE encoder output (16-d latent)
concatenated with Macro, On-chain, Sentiment, and TDA features from F*. Loss = Quantile
loss for forecast targets Y_h.
## 5.2 LATENT DIMENSION SELECTION
## V2.1
The latent dimension of 16 must be justified empirically, not assumed. Selection procedure:
Algorithm 5.1 — Latent dimension selection via reconstruction error
- Compute PCA on TA feature subset of F* (Train set only)
- Plot explained variance ratio (EVR) vs. n_components
- Set candidate latent dims D ∈ {8, 12, 16, 24, 32}
- For each D: train TCN-AE, record val reconstruction MSE
- d_latent = min D such that:
— val reconstruction MSE < 110% of PCA(D) reconstruction MSE, AND
— EVR(D) > 0.85 (85% of TA feature variance explained)

- If no D in candidates satisfies: increase candidate set or review F* TA subset
## 5.3 MODEL ARCHITECTURES
## MODELPURPOSEKEY HYPERPARAMETERSINPUT
## TCN-
## AE
TA feature
compression
Layers=4, kernel=3, dilation=[1,2,4,8],
dropout=0.1
TA features from F* (Phase 1)
TFTLong-range trend &
attention
Hidden=64, heads=4, lookback=60d,
dropout=0.15
## Latent(16d) + Macro +
Sentiment + TDA
BiLSTMShort-range pattern
capture
Units=128, layers=2, lookback=30d,
dropout=0.2
## Latent(16d) + On-chain +
## Sentiment
## 5.4 WALK-FORWARD CROSS-VALIDATION
Both TFT and BiLSTM are trained using expanding-window walk-forward CV (same fold structure as
§4.2). Out-of-fold predictions are collected for meta-learner training in Stage 6. No data point in the
Calibration or Test set is seen during this stage.
# Walk-forward fold structure # Fold k: Train = [day_0, day_{min_train +
k*step}], Val = next 90 days # Out-of-fold predictions → OOF_TFT, OOF_BiLSTM
(used in Stage 6) n_folds = (len(train_days) - min_train) // step_size for k in
range(n_folds): train_end = min_train + k * step_size val_start = train_end
val_end = val_start + step_size model_tft.fit(X_train[:train_end],
Y_train[:train_end]) OOF_TFT[val_start:val_end] =
model_tft.predict(X_train[val_start:val_end])
## STAGE 6
## Ensemble, Inverse Transform & Uncertainty Quanti!cation
v2.1 fix
Goal — combine base models; produce price-space predictions with calibrated
intervals
## 6.1 RIDGE META-LEARNER (STACKING)

Ridge Regression with 5-fold time-series CV for λ selection is used as meta-learner. Elastic Net
(α=0.5) is available as a secondary option if |F*| > 60 to handle remaining collinearity. Input: OOF
predictions from TFT and BiLSTM (3 horizons × 2 models = 6 columns). Target: true Y_h (FracDiff
space).
## ⚠ WHY NOT XGBOOST AS META-LEARNER
With ~2,557 training days and 6 meta-features, the effective n/p ratio for XGBoost stacking
is 426:1 — sufficient in theory but XGBoost requires careful tuning to avoid overfitting to
OOF noise patterns. Ridge provides a shrinkage guarantee without tuning complexity. Re-
evaluate if n_train > 5,000 days.
## 6.2 INVERSE FRACTIONAL DIFFERENTIATION
## V2.1 FIX
Meta-learner output is in FracDiff space. Conversion to price space requires accumulating the
FracDiff weights in reverse. The inverse is approximated via the recurrence relation:
Given Y_hat_t = FracDiff(p, d*) prediction at time t:
p_hat_{t+h} = p_t + Σ_{k=1}^{h} [ Y_hat_{t+k} - Σ_{j=1}^{k-1} w_j(d*) ·
(p_hat_{t+k-j} - p_{t+k-j}) ]
Simplified for h=1: p_hat_{t+1} = Y_hat_{t+1} + p_t - Σ_{k=1}^{W-1}
w_k(d*) · p_{t-k+1}
The sum over historical log-prices p_{t-k} uses the known past price series, not forecasts. For
multi-step horizons (h=7, h=14), each intermediate step uses the predicted value, accumulating
approximation error. The 90% Conformal interval is applied after this inverse transform — see §6.3.
## 6.3 SPLIT CONFORMAL PREDICTION — PRICE-SPACE CALIBRATION
## V2.1 FIX
This section corrects the ordering error in v2.0: CP calibration must occur in the same space as the
final output. Calibrating in FracDiff space and then applying inverse transform invalidates the
coverage guarantee.
Algorithm 6.1 — Split CP calibrated in price space (Angelopoulos & Bates, 2023 [4])

- Generate predictions on Calibration set (2023-01-01 → 2023-12-31, n_cal = 365):
a. Run meta-learner → get Y_hat_cal (FracDiff space)
b. Apply Inverse FracDiff → get Price_hat_cal (USD space)
- Compute nonconformity scores in price space:
s_i = |Price_actual_i - Price_hat_cal_i| for i = 1..n_cal
- Compute q_hat = ⌈(n_cal + 1)(1 - α)⌉ / n_cal quantile of {s_1, ..., s_n_cal}
For α = 0.10 (90% coverage): q_hat = 90th percentile of |residuals| in USD
- At test time: prediction interval = [Price_hat_test - q_hat, Price_hat_test +
q_hat]
- Verify: empirical coverage on Test set ≥ 89.5% (marginal coverage guarantee
holds for exchangeable data; Bitcoin is not i.i.d. — verify empirically)
## ⚠ CALIBRATION SET SIZE ADEQUACY
n_cal = 365 is near the lower bound recommended by Romano, Patterson & Candès (2019)
[9] for stable 90% coverage. For h=14 horizon, non-overlapping calibration samples number
only ~26 — insufficient for reliable tail estimation. Mitigation: use the full 2023 year as
calibration regardless of overlap; acknowledge that coverage guarantee is approximate for
h=14. Report empirical coverage in Test as part of §7.3 evaluation.
## 6.4 FINAL OUTPUT FORMAT
# Pipeline output per prediction date t: output = { "date_forecast": t,
"price_1d": { "point": Price_hat_t1, "low_90": Price_hat_t1 - q_hat_1d,
"high_90": Price_hat_t1 + q_hat_1d }, "price_7d": { "point": Price_hat_t7,
"low_90": Price_hat_t7 - q_hat_7d, "high_90": Price_hat_t7 + q_hat_7d },
"price_14d": { "point": Price_hat_t14, "low_90": Price_hat_t14 - q_hat_14d,
"high_90": Price_hat_t14 + q_hat_14d }, "regime": hmm_state_t, # from §7.1
"drift_flag": adwin_alert_t # from §7.2 }
## STAGE 7
## Monitoring, Regime Detection & Evaluation
v2.1 addition
Goal — detect structural breaks, monitor production drift, evaluate against
baselines

## 7.1 REGIME DETECTION VIA HIDDEN MARKOV MODEL
## NEW
A 3-state Gaussian HMM [10] is fit on the Train set using log-returns and realized volatility as
emission variables. Regime probabilities are computed at each time step and added as additional
features to the forecasting models in §5 (as static context, not sequence input).
## HMM CONFIGURATION
States: 3 (Bull, Bear, Sideways/High-vol). Emission: Gaussian on [log_return_7d,
log_vol_14d]. Covariance: full. Training: Baum-Welch on Train set only. At inference:
forward algorithm gives P(state=k | observations up to t). Output: [P(Bull)_t, P(Bear)_t,
P(Sideways)_t] added as 3 features to TFT context vector. Reference: Hamilton (1989) [10],
Rabiner (1989) [11]. Library: hmmlearn.GaussianHMM.
## 7.2 DRIFT MONITORING — ADWIN AND CUSUM
## MONITORTARGET SIGNALTRIGGER CONDITIONACTION
## ADWIN
## [12]
Distribution of Macro features
and TCN-AE latent vectors
ADWIN detects change in
mean of monitored stream
Flag covariate drift; alert for
model review
CUSUMRolling forecast residuals
(Price_actual - Price_hat)
Cumulative sum exceeds
threshold h = 5σ
Trigger incremental
retraining on most recent
365 days
## HMM
regime
switch
P(regime_t) vs P(regime_{t-5})Regime probability shift >
0.4 in 5 days
Log event; condition risk
sizing in backtest
## 7.3 BACKTEST — WALK-FORWARD EVALUATION ON TEST SET
Walk-forward evaluation on Test set (2024-01-01 → 2025-03-31). Model is not retrained on Test
data. If CUSUM triggers during Test period, log the trigger date but continue evaluation without
retraining — this simulates production degradation visibility.
## 7.4 BASELINE MODELS — MANDATORY COMPARISONS
## NEW
## REQUIRED — NO EVALUATION IS VALID WITHOUT THESE
The pipeline must be compared against all four baselines below. A pipeline that beats all

baselines with statistical significance (§7.5) provides evidence of genuine predictive value.
A pipeline that fails to beat the naive baseline on MAE has no case for deployment
regardless of model complexity.
## BASELINEDESCRIPTIONTARGET METRIC
B1 — Random WalkŶ_{t+h} = Y_t (no drift). Zero parameters.MAE, RMSE, Directional
## Accuracy
## B2 — Random Walk +
## Drift
Ŷ_{t+h} = Y_t + μ
## ̂
· h, where μ
## ̂
= mean log-return on
## Train
## MAE, RMSE
B3 — Buy and HoldAlways long BTC from Test start dateAnnualized return,
MDD, Sharpe
## B4 — Momentum (1d
lookback)
If return_{t-1} > 0 → Long; else Short. Zero parameters
beyond threshold.
## Sharpe, Calmar
## 7.5 STATISTICAL TESTING — DIEBOLD-MARIANO TEST
## NEW
The Diebold-Mariano (DM) test [5] evaluates whether the forecast accuracy difference between two
models is statistically significant, accounting for autocorrelation in loss differentials.
d_t = L(e_{1,t}) - L(e_{2,t}) where L = squared error loss, e_{i,t} =
forecast error of model i
DM statistic = d
## ̄
/ sqrt(2π ŝ_d(0) / T) where ŝ_d(0) = HAC variance
estimator
H0: E[d_t] = 0 (equal predictive accuracy)
Reject H0 if |DM| > 1.96 at 5% significance level
Required tests: Pipeline vs B1 (random walk), Pipeline vs B2, TFT vs BiLSTM (to justify ensemble),
Ensemble vs TFT alone. Use HAC estimator with Bartlett kernel, bandwidth = h (forecast horizon) as
recommended by Harvey, Leybourne & Newbold (1997) [13].
## 7.6 EVALUATION METRICS — FULL SUITE

## METRICFORMULA / NOTEBENCHMARK TARGET
MAE (price space)Mean |Price_actual - Price_hat| in USD< MAE of B1
RMSE (price space)sqrt(mean squared error)< RMSE of B1
MASEMAE / MAE_in-sample-naive. Hyndman & Koehler (2006)
## [14]
## MASE < 1.0
Directional Accuracy% correct sign of (Price_{t+h} - Price_t)> 55% (vs 50%
random)
CP Coverage% test points inside [low_90, high_90]88% – 92%
Sharpe RatioAnnualized (Return - Risk-free) / Vol, net of fees> Sharpe of B3 & B4
## Max Drawdown
## (MDD)
Peak-to-trough decline in equity curve< MDD of B3
Calmar RatioAnnualized Return / |MDD|> 1.0
DM p-value vs B1H0: equal accuracy vs random walkp < 0.05 for all
horizons
## REFERENCES
[1]Lopez de Prado, M. (2018). Advances in financial machine learning. Wiley. [Chapter 5: Fractional
## Differentiation]
[2]Székely, G. J., & Rizzo, M. L. (2007). Measuring and testing dependence by correlation of distances. Annals of
Statistics, 35(6), 2769–2794. https://doi.org/10.1214/009053607000000505
[3]Keany, E. (2020). BorutaShap: A wrapper feature selection method which combines the Boruta feature
selection algorithm with Shapley values [Software]. Zenodo. https://doi.org/10.5281/zenodo.4247618
[4]Angelopoulos, A. N., & Bates, S. (2023). A gentle introduction to conformal prediction and distribution-free
uncertainty quantification. Foundations and Trends in Machine Learning, 16(4), 494–591.
https://doi.org/10.1561/2200000101

[5]Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic
Statistics, 13(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599
[6]Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). Regression diagnostics: Identifying influential data and sources
of collinearity. Wiley. https://doi.org/10.1002/0471725153
[7]Marquaridt, D. W. (1970). Generalized inverses, ridge regression, biased linear estimation, and nonlinear
estimation. Technometrics, 12(3), 591–612. https://doi.org/10.1080/00401706.1970.10488699
[8]Kursa, M. B., & Rudnicki, W. R. (2010). Feature selection with the Boruta package. Journal of Statistical
Software, 36(11), 1–13. https://doi.org/10.18637/jss.v036.i11
[9]Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized quantile regression. Advances in Neural
## Information Processing Systems, 32, 3538–3548.
[10]Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the
business cycle. Econometrica, 57(2), 357–384. https://doi.org/10.2307/1912559
[11]Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
Proceedings of the IEEE, 77(2), 257–286. https://doi.org/10.1109/5.18626
[12]Bifet, A., & Gavaldà, R. (2007). Learning from time-changing data with adaptive windowing. In Proceedings of
the 2007 SIAM International Conference on Data Mining (pp. 443–448). SIAM.
https://doi.org/10.1137/1.9781611972771.42
[13]Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors.
International Journal of Forecasting, 13(2), 281–291. https://doi.org/10.1016/S0169-2070(96)00719-4
[14]Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. International
Journal of Forecasting, 22(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001
[15]Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., &
Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. Nature
Machine Intelligence, 2(1), 56–67. https://doi.org/10.1038/s42256-019-0138-9
[16]Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent
networks for sequence modeling. arXiv preprint arXiv:1803.01271. https://arxiv.org/abs/1803.01271
[17]Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-
horizon time series forecasting. International Journal of Forecasting, 37(4), 1748–1764.
https://doi.org/10.1016/j.ijforecast.2021.03.012
[18]Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet shrinkage. Biometrika, 81(3),
425–455. https://doi.org/10.1093/biomet/81.3.425

This document is a technical research specification. It does not constitute financial advice. All model outputs carry
uncertainty that must be explicitly quantified and communicated (§6.3, §7.6). Past predictive accuracy on backtested
data does not guarantee future performance. Cryptocurrency markets contain structural breaks, manipulation events,
and regulatory changes that no statistical model can fully anticipate.