# The ONLY preprocessing you NEED

Feature engineering for the **Kaggle** competitions. This
repository contains **only the preprocessing pipeline** ([`preprocessing.ipynb`](preprocessing.ipynb)).

The competition data usually is **synthetically generated** from a smaller labeled
`original_dataset`, so most of the engineered features try to recover artifacts
of that generation process — digit/rounding patterns, value snapping, and
resampling ratios — on top of standard categorical interactions and target
encoding.

> **Inspiration.** This preprocessing approach is inspired by the 1st-place
> write-up for Kaggle Playground Series S6E3 by **Chris Deotte** (`cdeotte` —
> [Kaggle](https://www.kaggle.com/cdeotte), [GitHub](https://github.com/cdeotte)):
> <https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm>

## Inputs & outputs

**Inputs** (in `./data/`):

| File | Description |
| --- | --- |
| `train.csv`, `test.csv` | Competition data, indexed by `id`. |
| `original_dataset.csv` | The labeled source data the competition set was generated from. Optional — set `original_data = None` to skip the features that depend on it. |

The example target column is `Irrigation_Need`. The feature space is a mix of example numeric
columns (e.g. `Soil_pH`, `Temperature_C`, `Humidity`, `Rainfall_mm`,
`Field_Area_hectare`) and example categorical columns (e.g. `Soil_Type`, `Crop_Type`,
`Crop_Growth_Stage`, `Season`, `Irrigation_Type`, `Water_Source`, `Region`).

**Outputs:** `data/train_x_preprocessed.csv` / `data/test_x_preprocessed.csv` —
the same rows with the engineered feature columns added, ready for any
downstream model.

## Design

Everything dataset-specific (data location, target column, radix interaction
specs, target-encoding settings) lives in a **single configuration cell**, so
the feature functions stay generic and the pipeline can be retargeted to another
dataset by editing only that cell. Numeric vs. categorical columns are detected
with `select_dtypes` rather than hardcoded.

Each feature function takes `train_x`/`test_x`, derives whatever maps/stats it
needs, and returns copies with new columns; a shared `apply_to_both` helper
handles the copy-and-apply boilerplate. Crucially, every function reads the
**raw** numeric/categorical column lists captured *before* any feature
engineering, so steps never re-encode each other's outputs and `train_x` /
`test_x` always end with an identical schema. That makes the ordering between
groups safe.

The pipeline is organized into three groups by what each step needs.

### Group 1 — needs only `train` / `test`

Always runs; never touches `original_data`.

- **Decimal / digit features** (`add_decimal_features`) — decomposes each numeric
  value into digit-level components to expose generator artifacts (rounding,
  quantization, preferred fractions):
  - `{col}_fraction` — fractional part.
  - `{col}_digit{k}` — positional base-10 digit decomposition (`k<0` decimals,
    `k=0` units, `k>0` tens/hundreds/...).
  - `{col}_frac100` / `{col}_mod100` — two-digit views (first two decimals; integer
    part mod 100).
  - `{col}_round` — adaptive magnitude-based rounding (precision chosen from the
    column's max magnitude), emitted as a new column so the raw value survives.
  - `{col}_res_1_2 / 1_4 / 1_5 / 1_10` — residual distance of the fraction to the
    nearest multiple of 1/2, 1/4, 1/5, 1/10 (detects "nice"-fraction snapping).
  - `{col}_is_round` — flag for fractions ≈ 0 or ≈ 1.
  - `{col}_mod10_d1` — string combinations rebuilt from the positional digits.
- **Bigram (categorical interaction) features** (`add_bigram_features`) — every
  pairwise concatenation of categorical columns (`col1__col2 = "valA_valB"`) to
  capture co-occurrence patterns single categories miss. Emitted as raw string
  columns for gradient-boosting models that consume categoricals natively
  (CatBoost / LightGBM).
- **Frequency encoding** (`frequency_encode`) — replaces each category with its
  relative frequency, computed over `train + test` combined (transductive —
  fine for a fixed test set).
- **Radix encoding** (`radix_encode`) — packs groups of related features into a
  single integer code by placing each at a different base-10 "digit position"
  (e.g. `Soil_pH * 100 + soil_type_index * 1000`), giving the model compact,
  explicit interactions. Categorical terms use a deterministic, sorted
  `category → index` map shared across train+test so codes stay aligned. The
  interactions are read from the editable `RADIX_FEATURES` config, not hardcoded.

### Group 2 — requires the labeled original dataset

Runs only when `original_data is not None`. These steps compare each competition
row against the original data it was generated from.

- **Original-data KNN lookup** (`original_data_lookup`) — scales numerics and
  one-hot encodes categoricals (fitted on the original data), builds a `cKDTree`
  over the original feature matrix, and queries the 3 nearest neighbors for each
  row. Adds `nearest_original_label_mode` (majority neighbor label — a KNN
  prediction as a feature), `distance_to_nearest_min`, and
  `distance_to_nearest_mean`. Labels come from the original dataset, not from
  `train_y`, so there is no self-leakage.
- **Snap features** (`add_snap_features`) — snaps each numeric value to the
  nearest value that actually exists in the original dataset (`{col}_snap`) and
  records the distance (`{col}_snap_diff`). Tiny diffs flag lightly-perturbed
  rows that likely carry strong signal.
- **Count-ratio encoding** (`count_ratio_encode`) — for each category, the ratio
  of its count in `train + test` to its count in the original dataset — a direct
  measure of how strongly each level was over- or under-sampled by the
  generator.

### Group 3 — target-aware (requires `train_y`)

Runs last. This is the only step that uses the target, so it deliberately breaks
the target-free contract of the rest of the pipeline: fit on train, applied to
test.

- **Ordered Target Encoding** (`ordered_target_encode` / `OrderedTE`) — replaces
  each category with the smoothed per-class probability of the target given that
  category, one column per class (`{col}_TE_cls{class}`). Leakage is controlled
  two ways:
  - **Train** uses an ordered/cumulative scheme (each row encoded only from rows
    before it). Because the result depends on row order — and preprocessing must
    keep exactly one row per `id` — the ordered encoding is **averaged over
    several shuffles** (`TE_N_SHUFFLES`) and reindexed to the original rows,
    cancelling ordering variance while preserving alignment.
  - **Test** is encoded from the full-train per-category statistics; categories
    unseen in train fall back to the global class prior.
  - `TE_SMOOTHING` controls how strongly rare categories are pulled toward the
    prior.

### Cleanup

`drop_constant_columns` removes any engineered column with a single unique value
across `train + test` (e.g. a high digit position that is always `0`). These
carry no signal and only dilute column subsampling. The names of dropped columns
are printed.

## Running

1. Place `train.csv`, `test.csv`, and (optionally) `original_dataset.csv` in
   `./data/`.
2. Open and run [`preprocessing.ipynb`](preprocessing.ipynb) top to bottom.
3. Find `train_x_preprocessed.csv` and `test_x_preprocessed.csv` in `./data/`.

**Dependencies:** `numpy`, `pandas`, `scipy`, `scikit-learn`.

To reuse the pipeline on a different dataset, edit only the configuration cell
(`DATA_DIR`, `TARGET`, `RADIX_FEATURES`, and the `TE_*` settings).
