# auto-featurs

Polars-native, schema-driven automatic feature generation for tabular data.

## Table of Contents

- [Project Overview](#project-overview)
  - [What the project does](#what-the-project-does)
  - [The problem it solves](#the-problem-it-solves)
  - [Typical use cases](#typical-use-cases)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Python version requirements](#python-version-requirements)
  - [Install from pip](#install-from-pip)
  - [Local / editable install (recommended for contributors)](#local--editable-install-recommended-for-contributors)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Feature generation vs feature selection](#feature-generation-vs-feature-selection)
  - [How the pipeline works end-to-end](#how-the-pipeline-works-end-to-end)
- [Pipeline Class](#pipeline-class)
  - [Purpose and responsibilities](#purpose-and-responsibilities)
  - [Capabilities (what you can build)](#capabilities-what-you-can-build)
  - [Important parameters and configuration options](#important-parameters-and-configuration-options)
  - [Example usage (simple)](#example-usage-simple)
  - [Example usage (slightly advanced: layering + auxiliary)](#example-usage-slightly-advanced-layering--auxiliary)
  - [How steps are chained/executed](#how-steps-are-chainedexecuted)
- [FeatureSelector Class](#featureselector-class)
  - [Purpose and design philosophy](#purpose-and-design-philosophy)
  - [Supported selection methods](#supported-selection-methods)
  - [How selection integrates into the pipeline](#how-selection-integrates-into-the-pipeline)
  - [Example usage](#example-usage)
  - [When and why to use different strategies](#when-and-why-to-use-different-strategies)
- [Advanced Usage](#advanced-usage)
  - [Custom feature generators (custom Transformers)](#custom-feature-generators-custom-transformers)
  - [Custom selectors](#custom-selectors)
  - [Integration with sklearn / pandas / numpy workflows](#integration-with-sklearn--pandas--numpy-workflows)
  - [Handling large datasets / performance tips](#handling-large-datasets--performance-tips)
- [Pros & Limitations](#pros--limitations)
  - [Strengths](#strengths)
  - [Limitations / tradeoffs](#limitations--tradeoffs)
  - [When this project may not be the right tool](#when-this-project-may-not-be-the-right-tool)
- [Examples](#examples)
  - [Example 1: Simple tabular dataset (polynomial + arithmetic)](#example-1-simple-tabular-dataset-polynomial--arithmetic)
  - [Example 2: More realistic ML workflow (time features + grouped aggregations + selection)](#example-2-more-realistic-ml-workflow-time-features--grouped-aggregations--selection)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

### What the project does
`auto-featurs` generates derived features from a tabular dataset using a **declarative, pipeline-based API**. You describe *what families of features you want* (polynomials, arithmetic interactions, rolling aggregations, lags, etc.), and the library builds a **Polars expression plan** that can be executed lazily.

### The problem it solves
Feature engineering in real-world tabular pipelines often becomes:

- hard to reproduce (lots of ad-hoc notebook code),
- slow (Python loops, repeated materializations),
- messy (no clear “what is a feature vs metadata/label?”),
- and difficult to extend without breaking things.

This project provides a **schema-aware, composable** approach that stays close to how Polars wants you to work: expressions + lazy execution.

### Typical use cases
- **ML preprocessing** for tabular models (tree models, linear models, GBMs)
- **Feature generation layers** (create features on top of previously generated features)
- **Time-based feature engineering** (rolling windows, lags, cumulative stats)
- **Reusable data pipelines** where the same feature logic must be applied consistently across train/valid/test
- **Rapid experimentation** with controlled feature “families” and selection strategies

---

## Key Features

- **Automatic feature generation**
  - Polynomials, logs, trig transforms
  - Pairwise arithmetic and comparisons
  - Aggregations (count, mean, std, z-score, etc.)
  - Time-derived features (hour/day/month) and time diffs
  - Lags, first value, mode, n-unique
  - Grouped (`over(...)`) and rolling-window features

- **Pipeline-based design**
  - Immutable pipelines: methods return a new `Pipeline` instance (no in-place surprises)
  - Layering via `with_new_layer()` so generated columns can become inputs to subsequent steps

- **Simple-to-use selector API**
    - **Schema-aware** - explicit column types and roles, which both can be easily used to select subsets for engineering:
        - `(ColumnType.NUMERIC | ColumnType.BOOLEAN) & ~ColumnRole.LABEL` to subset only numeric and boolean columns excluding label column

- **Feature selection strategies**
  - `FeatureSelector` computes selection statistics (currently correlation and t-test)
  - Selection can be used to shrink a generated feature set to a top-k or fraction

- **Extensibility & configurability**
  - Add your own `Transformer` classes (Polars expressions in, new columns out)
  - Flexible column selection using a schema and selectors (names, types, roles, combinations)

- **Reproducibility and performance considerations**
  - All transformations are expressed as **Polars expressions**
  - Uses `LazyFrame` internally; you choose when to materialize (`collect()`)
  - Optional optimization to reduce redundant feature generation (e.g. commutative duplicates)

---

## Installation

### Python version requirements
- Python **3.13+** (see `pyproject.toml`)

### Install from pip
```shell script
pip install auto-featurs
```


### Local / editable install (recommended for contributors)
```shell script
pip install -e .
```


> This project uses **uv** in development, but standard `pip` installs work fine for users.

---

## Quick Start

Minimal end-to-end example:

- Create a `Dataset` (data + schema)
- Build a `Pipeline` and generate features
- Compute a selection report with `FeatureSelector`
- Select top features and produce a final dataset

```python
import polars as pl

from auto_featurs.base.column_specification import ColumnRole, ColumnSpecification, ColumnType
from auto_featurs.base.schema import Schema
from auto_featurs.dataset.dataset import Dataset
from auto_featurs.feature_selection.feature_selector import FeatureSelector, SelectionMethod
from auto_featurs.pipeline.pipeline import Pipeline
from auto_featurs.transformers.numeric_transformers import ArithmeticOperation

# 1) Input data
df = pl.DataFrame(
    {
        "x": [0, 1, 2, 3, 4],
        "y": [10, 11, 12, 13, 14],
        "label": [0, 0, 1, 1, 1],  # boolean-ish label for selection methods
    }
)

# 2) Schema: declare types + label role
schema = Schema(
    [
        ColumnSpecification.numeric(name="x"),
        ColumnSpecification.numeric(name="y"),
        ColumnSpecification.boolean(name="label", role=ColumnRole.LABEL),
    ]
)

dataset = Dataset(df, schema=schema)

# 3) Build a pipeline (generate features)
pipeline = (
    Pipeline(dataset=dataset)
    .with_polynomial(subset=ColumnType.NUMERIC, degrees=[2])
    .with_arithmetic(
        left_subset="x",
        right_subset="y",
        operations=[ArithmeticOperation.ADD, ArithmeticOperation.SUBTRACT],
    )
)

generated_df = pipeline.collect()

# 4) Feature selection: compute stats and select a subset
selector = FeatureSelector()
generated_dataset = Dataset(
    generated_df,
    schema=dataset.schema.with_schema(  # keep original schema + add generated cols
        pipeline.collect_plan().schema
    ),
)

report = selector.get_report(
    dataset=generated_dataset,
    feature_subset=(ColumnType.NUMERIC | ColumnType.BOOLEAN) & ~ColumnRole.LABEL,
    method=SelectionMethod.CORRELATION,
)

selected = selector.select_features(report=report, top_k=5)

# 5) Final dataset (keep selected + label)
final_df = generated_df.select(selected + ["label"])
print(final_df)
```


Notes:
- Feature selection returns **column names**; you decide how to slice your frame.
- The pipeline operates on `Dataset` and produces a Polars `DataFrame` when collected.

---

## Core Concepts

### Feature generation vs feature selection
- **Feature generation**: create new columns from existing columns (e.g. `x_pow_2`, `x_add_y`, rolling counts).
- **Feature selection**: rank or filter features based on a criterion (e.g. correlation with label), then keep only the best ones.

In practice you often do:  
**generate many → select a smaller set → train model**.

### How the pipeline works end-to-end
1. You wrap your data in a `Dataset` (Polars frame + `Schema`).
2. You define a `Pipeline` by appending transformation steps (`with_polynomial`, `with_count`, …).
3. The pipeline builds Polars expressions and keeps them organized into **layers**.
4. When you call `collect_plan()` you get a new `Dataset` whose `.data` is still lazy.
5. When you call `collect()` you materialize to a `pl.DataFrame`.

---

## Pipeline Class

### Purpose and responsibilities
`Pipeline` is the central abstraction. It:

- knows the current `Dataset` schema,
- generates transformers for column combinations,
- validates that transformers receive compatible column types,
- optionally optimizes feature generation,
- applies transformations across one or more layers,
- and returns either a lazy plan (`collect_plan`) or a materialized `DataFrame` (`collect`).

### Capabilities (what you can build)
The built-in pipeline methods cover common feature families:

- **Numeric transforms**
  - polynomials (`with_polynomial`)
  - log transforms (`with_log`)
  - sin/cos (`with_goniometric`)
  - scaling (`with_scaling`)
  - arithmetic interactions (`with_arithmetic`)
- **Comparisons**
  - equality and ordering comparisons (`with_comparison`)
- **Datetime**
  - seasonal features like hour/day/month (optionally angular/periodic) (`with_seasonal`)
  - time differences (`with_time_diff`)
- **Aggregations**
  - count / cumulative count / conditional count (`with_count`)
  - lag features (`with_lagged`)
  - first value (`with_first_value`)
  - mode (`with_mode`)
  - number of unique values (`with_num_unique`)
  - arithmetic aggregations (sum/mean/std/z-score) (`with_arithmetic_aggregation`)
- **Layering**
  - `with_new_layer()` to “freeze” features and use them as inputs for the next wave

### Important parameters and configuration options
- `dataset`: a `Dataset` instance (data + schema)
- `optimization_level`: controls redundancy reduction:
  - `OptimizationLevel.NONE`: generate all combinations
  - `OptimizationLevel.SKIP_SELF`: drop combinations like `(x, x)` for multi-input transformers
  - `OptimizationLevel.DEDUPLICATE_COMMUTATIVE`: also drop symmetric duplicates for commutative ops (e.g. keep `x + y`, drop `y + x`)
- `auxiliary=True` on feature methods:
  - marks newly generated columns to be dropped at the end (useful for “intermediate” features)

### Example usage (simple)
```python
import polars as pl
from auto_featurs.base.column_specification import ColumnSpecification, ColumnType
from auto_featurs.base.schema import Schema
from auto_featurs.dataset.dataset import Dataset
from auto_featurs.pipeline.pipeline import Pipeline

df = pl.LazyFrame({"x": [0, 1, 2, 3]})

schema = Schema([ColumnSpecification.numeric(name="x")])
dataset = Dataset(df, schema=schema)

pipeline = Pipeline(dataset=dataset).with_polynomial(subset=ColumnType.NUMERIC, degrees=[2, 3])

out = pipeline.collect()
print(out)
```


### Example usage (slightly advanced: layering + auxiliary)
```python
import math
import polars as pl

from auto_featurs.base.column_specification import ColumnSpecification, ColumnType
from auto_featurs.base.schema import Schema
from auto_featurs.dataset.dataset import Dataset
from auto_featurs.pipeline.pipeline import Pipeline

df = pl.LazyFrame({"x": [0, 1, 2, 3]})
dataset = Dataset(df, schema=Schema([ColumnSpecification.numeric(name="x")]))

pipeline = (
    Pipeline(dataset=dataset)
    .with_polynomial(subset="x", degrees=[2], auxiliary=True)  # intermediate
    .with_new_layer()  # make x_pow_2 visible to next steps
    .with_log(subset=ColumnType.NUMERIC, bases=[math.e, 10])
)

out = pipeline.collect()
print(out)  # x_pow_2 will be dropped; logs remain
```


### How steps are chained/executed
- Each `with_*` method:
  1. resolves column selections using the schema,
  2. constructs transformers for all valid combinations,
  3. validates types,
  4. adds them to the **current layer**.
- `with_new_layer()`:
  - finalizes the schema additions from the current layer and starts a new empty layer.
- `collect_plan()`:
  - sequentially applies each layer’s expressions to the dataset.
- `collect()`:
  - calls `collect_plan()` and then materializes.

---

## FeatureSelector Class

### Purpose and design philosophy
`FeatureSelector` separates *computing selection statistics* from *choosing how many features to keep*. It produces a `SelectionReport` and then selects top features based on the report.

This keeps the workflow explicit:
- “How do I score features?” (`get_report`)
- “How many do I keep?” (`select_features`)

### Supported selection methods
Currently implemented:

- **Correlation** (`SelectionMethod.CORRELATION`)
  - absolute correlation between each feature and the label
  - supported label types: numeric, boolean
- **T-Test** (`SelectionMethod.T_TEST`)
  - t-stat style separation score between label groups
  - supported label types: boolean

Both methods validate feature/label types and will raise a `ValueError` for unsupported columns.

### How selection integrates into the pipeline
Selection does **not** mutate a pipeline. A typical workflow is:

1. build and collect generated features,
2. compute a selection report on the resulting dataset,
3. pick a subset of columns,
4. slice the Polars DataFrame to your final feature matrix.

### Example usage
```python
from auto_featurs.feature_selection.feature_selector import FeatureSelector, SelectionMethod
from auto_featurs.base.column_specification import ColumnRole, ColumnType

selector = FeatureSelector()

report = selector.get_report(
    dataset=dataset,  # Dataset containing generated features + label column
    feature_subset=(ColumnType.NUMERIC | ColumnType.BOOLEAN) & ~ColumnRole.LABEL,
    method=SelectionMethod.CORRELATION,
)

top_features = selector.select_features(report=report, top_k=50)
```


### When and why to use different strategies
- Use **correlation** when:
  - your label is numeric or boolean,
  - you want a quick linear signal measure,
  - you’re doing fast iteration.
- Use **t-test** when:
  - your label is boolean (binary classification),
  - you want a simple separation statistic across classes.

If you need model-based importance or mutual information, plan to implement a custom selector (see Advanced Usage).

---

## Advanced Usage

### Custom feature generators (custom Transformers)
You can implement a new feature generator by subclassing `Transformer` and returning a Polars expression.

```python
import polars as pl

from auto_featurs.base.column_specification import ColumnType
from auto_featurs.base.column_specification import ColumnTypeSelector
from auto_featurs.transformers.base import Transformer

class AbsValueTransformer(Transformer):
    def __init__(self, column: str) -> None:
        self._column = column

    def input_type(self) -> ColumnTypeSelector:
        return ColumnType.NUMERIC.as_selector()

    @classmethod
    def is_commutative(cls) -> bool:
        return True

    def _return_type(self) -> ColumnType:
        return ColumnType.NUMERIC

    def _transform(self) -> pl.Expr:
        return pl.col(self._column).abs()

    def _name(self, transform: pl.Expr) -> pl.Expr:
        return transform.alias(f"{self._column}_abs")
```


Then add it to a pipeline by constructing transformers yourself and passing them into the pipeline’s `transformers` layers (advanced) or by contributing a `with_abs(...)` helper.

### Custom selectors
A “selector” can be as simple as:
- compute a score per feature,
- sort by score,
- pick top-k.

If your scoring requires a trained model (e.g. feature importance), you can:
- compute the model on a collected frame,
- store importance scores,
- build a report-like object and reuse `select_features` logic (or implement your own).

### Integration with sklearn / pandas / numpy workflows
- `auto-featurs` is **Polars-native**, so the smooth path is:
  1. generate features with `Pipeline`,
  2. `collect()` to `pl.DataFrame`,
  3. convert to NumPy for sklearn when needed.

```python
import numpy as np

X = out.drop("label").to_numpy()
y = out["label"].to_numpy()
```


### Handling large datasets / performance tips
- Prefer `LazyFrame` inputs and delay `collect()` until the end.
- Use `optimization_level` to cut down feature explosion early.
- Use `collect_plan(cache_computation=True)` when you need to reuse the same generated dataset multiple times (e.g. multiple selection passes).
- Be selective with pairwise operations: arithmetic/comparison over many numeric columns grows as O(n²).

---

## Pros & Limitations

### Strengths
- **Fast and scalable** feature generation via Polars expressions
- **Schema-aware**: explicit typing and column roles reduce accidental misuse
- **Clear API**: simple column subsetting and layered builder design for Pipeline
- **Composable and reproducible** pipelines with clear “what was generated”
- **Layering** supports multi-stage feature construction
- **Built-in redundancy controls** via optimization levels

### Limitations / tradeoffs
- Polars-first design: if your stack is pandas-only, you’ll need conversions.
- Feature selection is currently **statistical and relatively simple** (no built-in model-based importance).
- Automatic generation can still create a lot of columns quickly; you must manage scope (subsets, layers, optimization).
- Rolling aggregations require a datetime index column and supported Polars rolling semantics.

### When this project may not be the right tool
- You need end-to-end sklearn-compatible `fit/transform` estimators with automatic handling of train/test leakage.
- Your features are primarily NLP embeddings, image features, or deep-learning representations.
- You want a GUI/no-code feature engineering tool (this is a developer library).

---

## Examples

### Example 1: Simple tabular dataset (polynomial + arithmetic)
```python
import polars as pl

from auto_featurs.base.column_specification import ColumnSpecification, ColumnType
from auto_featurs.base.schema import Schema
from auto_featurs.dataset.dataset import Dataset
from auto_featurs.pipeline.pipeline import Pipeline
from auto_featurs.transformers.numeric_transformers import ArithmeticOperation

df = pl.LazyFrame({"x": [0, 1, 2, 3], "y": [10, 11, 12, 13]})
schema = Schema([ColumnSpecification.numeric(name="x"), ColumnSpecification.numeric(name="y")])
dataset = Dataset(df, schema=schema)

pipeline = (
    Pipeline(dataset=dataset)
    .with_polynomial(subset=ColumnType.NUMERIC, degrees=[2])
    .with_arithmetic(left_subset="x", right_subset="y", operations=[ArithmeticOperation.ADD, ArithmeticOperation.SUBTRACT])
)

out = pipeline.collect()
print(out)
```


### Example 2: More realistic ML workflow (time features + grouped aggregations + selection)
```python
import polars as pl
from datetime import UTC, datetime

from auto_featurs.base.column_specification import ColumnRole, ColumnSpecification, ColumnType
from auto_featurs.base.schema import Schema
from auto_featurs.dataset.dataset import Dataset
from auto_featurs.feature_selection.feature_selector import FeatureSelector, SelectionMethod
from auto_featurs.pipeline.pipeline import Pipeline
from auto_featurs.transformers.aggregating_transformers import ArithmeticAggregations

df = pl.DataFrame(
    {
        "user_id": ["u1", "u1", "u2", "u2", "u2"],
        "ts": [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
            datetime(2024, 1, 4, tzinfo=UTC),
        ],
        "amount": [10.0, 12.0, 5.0, 7.0, 9.0],
        "label": [0, 1, 0, 0, 1],
    }
)

schema = Schema(
    [
        ColumnSpecification.nominal(name="user_id", role=ColumnRole.IDENTIFIER),
        ColumnSpecification.datetime(name="ts", role=ColumnRole.TIME_INFO),
        ColumnSpecification.numeric(name="amount"),
        ColumnSpecification.boolean(name="label", role=ColumnRole.LABEL),
    ]
)
dataset = Dataset(df, schema=schema)

pipeline = (
    Pipeline(dataset=dataset)
    .with_seasonal(subset="ts", operations=[], periodic=False)  # no-op placeholder if you want to add seasonal ops later
    .with_arithmetic_aggregation(
        subset="amount",
        aggregations=[ArithmeticAggregations.MEAN, ArithmeticAggregations.STD],
        over_columns_combinations=[["user_id"]],
    )
    .with_count(over_columns_combinations=[["user_id"]])
)

generated = pipeline.collect()

# Selection
selector = FeatureSelector()
generated_dataset = Dataset(generated, schema=dataset.schema.with_schema(pipeline.collect_plan().schema))

report = selector.get_report(
    dataset=generated_dataset,
    feature_subset=(ColumnType.NUMERIC | ColumnType.BOOLEAN) & ~ColumnRole.LABEL,
    method=SelectionMethod.CORRELATION,
)
keep = selector.select_features(report=report, frac=0.5)

final = generated.select(keep + ["label"])
print(final)
```


---

## Project Structure

A quick map of the repository:

- `src/auto_featurs/`
  - `base/` — column specifications, schema, selectors
  - `dataset/` — `Dataset` wrapper for Polars + schema
  - `pipeline/` — `Pipeline`, optimizer, validation logic
  - `transformers/` — built-in feature generators (numeric, datetime, aggregations, wrappers)
  - `feature_selection/` — `FeatureSelector`, selection reports and methods
  - `utils/` — shared helpers/constants
- `examples/` — runnable examples / notebooks
- `LICENSE` — project license
- `pyproject.toml` — packaging, dependencies, tooling (ruff/mypy/pytest)

---

## Contributing

Contributions are welcome: bug reports, feature requests, docs improvements, and PRs.

Suggested workflow:
1. Fork and clone the repo
2. Create a branch for your change
3. Add/adjust tests where relevant
4. Run formatting/linting and the test suite
5. Open a PR with a clear description and rationale

Coding standards (brief):
- Keep code **typed** (project uses strict mypy settings)
- Prefer **small, composable** transformers over “mega” transforms
- Keep names deterministic and consistent (feature columns are the interface)
- Add tests for new behavior (pytest)

---

## License

This project is licensed under the **MIT License**.