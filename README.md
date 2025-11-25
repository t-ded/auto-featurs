# auto-featurs

### Polars-native, schema-driven automatic feature generation

`auto-featurs` is a fast, composable, **Polars-native feature engineering library**.
It provides a declarative, type-safe way to generate large families of features (polynomial, arithmetic, comparison, lagged, aggregations, rolling windows, etc.) while retaining control over:

* **Schema** (column types, numeric/ordinal/nominal/datetime)
* **Feature layers** (build features on top of earlier generated ones)
* **Optimization** (skip redundant or symmetric features before they are generated)
* **Lazy execution** via Polars (`LazyFrame → DataFrame`)

The main abstraction is the **Pipeline**, which is immutable and fully composable.

---

## ✨ Features

* 🚀 **Polars expressions**, not Python loops
* 🧱 **Layered feature engineering** (generate → freeze → generate next layer)
* 🔍 **Redundancy-aware optimization** (e.g., avoid `A + B` and `B + A` duplicates)
* 🔗 **Schema-driven selection** (e.g., “all numeric columns”)
* ⏱ **Time-window & cumulative rolling aggregations**
* 🧪 Fully immutable pipeline — no in-place mutation surprises

---

# Installation

```bash
pip install auto-featurs
```

---

# Quick Start

Below is a minimal but expressive example that highlights:

1. **Basic feature generation**
2. **Layering** — using derived features as inputs for subsequent layers
3. **Optimization** — reducing redundant feature creation

```python
import polars as pl
from auto_featurs.pipeline.pipeline import Pipeline
from auto_featurs.base.column_specification import ColumnSpecification, ColumnType
from auto_featurs.transformers.numeric_transformers import ArithmeticOperation
from auto_featurs.pipeline.optimizer import OptimizationLevel
```

---

# 1. Basic Example: Polynomial + Arithmetic

Consider a simple input frame:

```python
df = pl.LazyFrame({
    "x": [0, 1, 2, 3],
    "y": [10, 11, 12, 13],
})

schema = [
    ColumnSpecification.numeric("x"),
    ColumnSpecification.numeric("y"),
]
```

We build a pipeline:

```python
pipeline = (
    Pipeline(schema=schema)
    .with_polynomial(subset=ColumnType.NUMERIC, degrees=[2])
    .with_arithmetic(
        left_subset="x",
        right_subset="y",
        operations=[ArithmeticOperation.ADD, ArithmeticOperation.SUBTRACT],
    )
)
```

Collect the final output:

```python
result = pipeline.collect(df)
print(result)
```

**Output:**

```
shape: (4, 6)
┌─────┬─────┬───────────┬──────────────┬─────────┬──────────────┐
│ x   │ y   │ x_pow_2    │ y_pow_2       │ x_add_y │ x_subtract_y │
│ --- │ --- │ ---         │ ---            │ ---      │ ---             │
│ i64 │ i64 │ i64        │ i64           │ i64      │ i64           │
├─────┼─────┼───────────┼──────────────┼─────────┼──────────────┤
│ 0   │ 10  │ 0          │ 100          │ 10       │ -10          │
│ 1   │ 11  │ 1          │ 121          │ 12       │ -10          │
│ 2   │ 12  │ 4          │ 144          │ 14       │ -10          │
│ 3   │ 13  │ 9          │ 169          │ 16       │ -10          │
└─────┴─────┴───────────┴──────────────┴─────────┴──────────────┘
```

---

# 2. Layering: Building Features from Previous Layers

Layers let you “freeze” the current schema and then use the newly-generated columns as inputs to the next layer — enabling multi-stage feature generation.

```python
pipeline = Pipeline(schema=[ColumnSpecification.numeric("x")])

pipeline = pipeline.with_polynomial(subset=ColumnType.NUMERIC, degrees=[2])
pipeline = pipeline.with_new_layer()   # ← freeze layer 1 outputs

# Now “x_pow_2” is part of the schema, so it can be used as a numeric input
pipeline = pipeline.with_polynomial(subset=ColumnType.NUMERIC, degrees=[2])

df = pl.LazyFrame({"x": [0, 1, 2, 3]})
print(pipeline.collect(df))
```

**Output:**

```
shape: (4, 3)
┌─────┬───────────┬────────────────────┐
│ x   │ x_pow_2    │ x_pow_2_pow_2       │
│ i64 │ i64        │ i64                 │
├─────┼───────────┼────────────────────┤
│ 0   │ 0          │ 0                   │
│ 1   │ 1          │ 1                   │
│ 2   │ 4          │ 16                  │
│ 3   │ 9          │ 81                  │
└─────┴───────────┴────────────────────┘
```

This mirrors the test suite behavior but is cleaner and easier to follow.

---

# 3. Optimization: Avoiding Redundant Feature Generation

The pipeline uses an `Optimizer` to avoid predictable redundancies:

* **`SKIP_SELF`** removes operations like `x + x` or `x - x` when appropriate
* **`DEDUPLICATE_COMMUTATIVE`** removes symmetric pairs like `x + y` and `y + x`

Let’s see a clear, visual comparison.

## Example Input

```python
df = pl.LazyFrame({
    "a": [0, 1, 2],
    "b": [10, 11, 12],
})
schema = [
    ColumnSpecification.numeric("a"),
    ColumnSpecification.numeric("b"),
]
```

## Pipeline (ADD + SUBTRACT)

```python
from auto_featurs.transformers.numeric_transformers import ArithmeticOperation

def build(optimization):
    return (
        Pipeline(schema=schema, optimization_level=optimization)
        .with_arithmetic(
            left_subset=ColumnType.NUMERIC,
            right_subset=ColumnType.NUMERIC,
            operations=[ArithmeticOperation.ADD, ArithmeticOperation.SUBTRACT],
        )
    )
```

---

## 🔎 OptimizationLevel.NONE (default)

```python
print(build(OptimizationLevel.NONE).collect(df))
```

**Generated columns:**

* a_add_a
* a_add_b
* b_add_a
* b_add_b
* a_subtract_a
* a_subtract_b
* b_subtract_a
* b_subtract_b

This is the “full Cartesian explosion.”

---

## 🔎 OptimizationLevel.SKIP_SELF

```python
print(build(OptimizationLevel.SKIP_SELF).collect(df))
```

Self-operations removed:

* a_add_a ❌
* b_add_b ❌
* a_subtract_a ❌
* b_subtract_b ❌

All cross-column combos remain.

---

## 🔎 OptimizationLevel.DEDUPLICATE_COMMUTATIVE

```python
print(build(OptimizationLevel.DEDUPLICATE_COMMUTATIVE).collect(df))
```

Removes everything from `SKIP_SELF` **plus** commutative duplicates:

* b_add_a ❌ because a_add_b already exists
* Subtraction stays distinct (non-commutative)

---

### Summary Table

| Optimization                | Self ops | (a,b) vs (b,a)              | Notes                                  |
| --------------------------- | -------- | --------------------------- | -------------------------------------- |
| **NONE**                    | kept     | kept                        | Full feature set                       |
| **SKIP_SELF**               | removed  | kept                        | Reduces noise; retains all cross terms |
| **DEDUPLICATE_COMMUTATIVE** | removed  | removed for commutative ops | Best for large numeric sets            |

This is exactly the behavior exercised in the test suite — now shown clearly in the README.

---

# API Overview

### Pipeline Construction

```python
Pipeline(
    schema: list[ColumnSpecification],
    transformers: Optional[list[list[Transformer]]] = None,
    optimization_level: OptimizationLevel = OptimizationLevel.NONE,
)
```

### Core Methods

| Method                                      | Description                                                |
| ------------------------------------------- | ---------------------------------------------------------- |
| `with_polynomial(subset, degrees)`          | x → x², x³, …                                              |
| `with_arithmetic(left, right, operations)`  | Add/Sub/Mul/Div combinations                               |
| `with_comparison(left, right, comparisons)` | x > y, x == y, …                                           |
| `with_lagged(subset, lags, ...)`            | Lag features with optional groupings                       |
| `with_first_value(...)`                     | First value in group or time window                        |
| `with_arithmetic_aggregation(...)`          | Count/Sum/Mean/Std over groups or windows                  |
| `with_new_layer()`                          | Freeze current layer & use generated columns in next layer |
| `collect(lazyframe)`                        | Apply across all layers and return `pl.DataFrame`          |

All methods return **new pipelines** (pipelines are immutable).

---

# Roadmap

* 🔜 Learned feature selection (correlation / mutual information / importance-based)
* 🔜 Caching & materialization strategies
* 🔜 Group-aware window expressions once supported by Polars' stable API
* 🔜 Documentation site & examples gallery

---

# Contributing

PRs, issues, and discussions are welcome!
