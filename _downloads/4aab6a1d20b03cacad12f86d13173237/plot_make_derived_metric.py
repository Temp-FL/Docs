# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
======================
Making Derived Metrics 1
======================
"""
# %%
# This notebook demonstrates the use of the :func:`fairlearn.metrics.make_derived_metric`
# function.
# Many higher-order machine learning algorithms (such as hyperparameter tuners) make use
# of scalar metrics when deciding how to proceed.
# While the :class:`fairlearn.metrics.MetricFrame` has the ability to produce such
# scalars through its aggregation functions, its API does not conform to that usually
# expected by these algorithms.
# The :func:`~fairlearn.metrics.make_derived_metric` function exists to bridge this gap.
#
# Getting the Data
# ================
#
# *This section may be skipped. It simply creates a dataset for
# illustrative purposes*
#
# We will use the well-known UCI 'Adult' dataset as the basis of this
# demonstration. This is not for a lending scenario, but we will regard
# it as one for the purposes of this example. We will use the existing
# 'race' and 'sex' columns (trimming the former to three unique values),
# and manufacture credit score bands and loan sizes from other columns.
# We start with some uncontroversial `import` statements:

import functools
import numpy as np

import sklearn.metrics as skm
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from fairlearn.metrics import MetricFrame, make_derived_metric
from fairlearn.metrics import accuracy_score_group_min

# %%
# Next, we import the data, dropping any rows which are missing data:

print("Hello world")