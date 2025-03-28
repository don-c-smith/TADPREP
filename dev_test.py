import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tadprep as tp  # Testing package-level import - I want TADPREP to mirror Pandas in its common practice
# We don't want to truncate our dataframes in our print checks
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

from tadprep.core.transforms import PlotHandler, _build_interactions_core

# import sys
# import os
# print(sys.path)
# print(os.environ.get('PYTHONPATH'))

# FIRST-PASS TESTING
# Uploading sample datafile
df_raw = pd.read_csv(r"C:\Users\gabor\Desktop\code\TADPREP\data\sample_data_sparse.csv")

# Checking import
# print(df_raw)


# Testing file_info method
# tp.file_info(df_raw, verbose=False)
# tp.file_info(df_raw, verbose=True)


# Testing reshape method
# df_reshape = tp.reshape(df_raw, verbose=False)
# df_reshape = tp.reshape(df_raw, verbose=True)
# print(df_reshape)

## Uncomment all for latest
# ## Testing PlotHandler class
# ph = PlotHandler()
# age_tuple = ph.det_plot_type(df_raw, 'age')
# race_tuple = ph.det_plot_type(df_raw, 'race')
# print(age_tuple)
# print(race_tuple)

# ph.plot(df_raw, 'age', 'hist')
# # ph.plot(df_raw, 'race', 'hist')
# ph.recall_plot('age', 'hist')
# # ph.recall_plot('race', 'hist')
# ph.compare_plots('age')
# # ph.compare_plots('race')

# ph.plot(df_raw, 'age', 'hist')
# # ph.plot(df_raw, 'race', 'hist')
# # ph.recall_plot('age', 'hist')
# # ph.recall_plot('race', 'hist')
# ph.compare_plots('age')
# # ph.compare_plots('race')

# ph.plot(df_raw, 'age', 'box')
# ph.plot(df_raw, 'age', 'box')
# ph.recall_plot('age', 'hist')
# ph.recall_plot('age', 'box')
# ph.compare_plots('age')

## Testing build_interactions method

## "Exploratory" paradigm
# df_interact = _build_interactions_core(df_raw, features_list=['age','salary'], interact_types=['+', '-', '*', '/'])
# df_interact = _build_interactions_core(df_raw, features_list=['age', 'salary'], interact_types=['^2', '^3', '^1/2', '^1/3', 'e^x'])
# df_interact = _build_interactions_core(df_raw, features_list=['age', 'salary'], interact_types=['magnitude', 'magdiff'])
# df_interact = _build_interactions_core(df_raw, features_list=['age', 'salary'], interact_types=['poly', 'prod^1/2', 'prod^1/3'])
# df_interact = _build_interactions_core(df_raw, features_list=['age', 'salary'], interact_types=['log_inter', 'exp_inter'])
# df_interact = _build_interactions_core(df_raw, features_list=['age', 'salary'], interact_types=['mean_diff', 'mean_ratio'])
# "Exploratory" Error-seeking
# df_interact = _build_interactions_core(df_raw, features_list=['age'], interact_types=['+', '-', '*', '/'])
# df_interact = _build_interactions_core(df_raw, features_list=['age'], interact_types=['^2', '^3', '^1/2', '^1/3', 'e^x'])
# df_interact = _build_interactions_core(df_raw, features_list=['age'], interact_types=['magnitude', 'magdiff'])
# df_interact = _build_interactions_core(df_raw, features_list=['age', 'salary'], interact_types=[])
# df_interact = _build_interactions_core(df_raw, features_list=['age', 'salary'])
# df_interact = _build_interactions_core(df_raw, features_list=['age'])
## "Focused" paradigm
# df_interact = _build_interactions_core(df_raw, f1='age', f2='salary', interact_types=['+', '-', '*', '/'])
# df_interact = _build_interactions_core(df_raw, f1='age', f2='salary', interact_types=['^2', '^3', '^1/2', '^1/3', 'e^x'])
# df_interact = _build_interactions_core(df_raw, f1='age', f2='salary', interact_types=['magnitude', 'magdiff'])
# df_interact = _build_interactions_core(df_raw, f1='age', f2='salary', interact_types=['poly', 'prod^1/2', 'prod^1/3'])
# df_interact = _build_interactions_core(df_raw, f1='age', f2='salary', interact_types=['log_inter', 'exp_inter'])
# df_interact = _build_interactions_core(df_raw, f1='age', f2='salary', interact_types=['mean_diff', 'mean_ratio'])
## "Focused" Error-seeking
# df_interact = _build_interactions_core(df_raw, f2='salary', interact_types=['+', '-', '*', '/'])
# df_interact = _build_interactions_core(df_raw, f1='age', interact_types=['+', '-', '*', '/'])
## Final output
# print(df_interact)
##Seems all ok!



# Testing rename_and_tag method
# df_renamed = tp.rename_and_tag(df_raw, verbose=False, tag_features=False)
# df_renamed = tp.rename_and_tag(df_raw, verbose=True, tag_features=False)
# df_renamed = tp.rename_and_tag(df_raw, verbose=False, tag_features=True)
# df_renamed = tp.rename_and_tag(df_raw, verbose=True, tag_features=True)
# print(df_renamed)


# Testing feature_stats method
# tp.feature_stats(df_raw, verbose=False, summary_stats=False)
# tp.feature_stats(df_raw, verbose=True, summary_stats=False)
# tp.feature_stats(df_raw, verbose=False, summary_stats=True)
# tp.feature_stats(df_raw, verbose=True, summary_stats=True)


# Testing impute method
# df_impute = tp.impute(df_raw, verbose=False, skip_warnings=False)
# df_impute = tp.impute(df_raw, verbose=True, skip_warnings=False)
# df_impute = tp.impute(df_raw, verbose=False, skip_warnings=True)
# df_impute = tp.impute(df_raw, verbose=True, skip_warnings=True)
# print(df_impute)


# Testing encode method
# Note that this method must be tested with and without a passed list of features to encode
# encode_feats = ['sex', 'race', 'us_citizen']

# df_encode = tp.encode(df_raw, encode_feats, verbose=True, skip_warnings=True)
# df_encode = tp.encode(df_raw, encode_feats, verbose=True, skip_warnings=False)
# df_encode = tp.encode(df_raw, encode_feats, verbose=False, skip_warnings=True)
# df_encode = tp.encode(df_raw, encode_feats, verbose=False, skip_warnings=False)

# df_encode = tp.encode(df_raw, verbose=True, skip_warnings=True)
# df_encode = tp.encode(df_raw, verbose=True, skip_warnings=False)
# df_encode = tp.encode(df_raw, verbose=False, skip_warnings=True)
# df_encode = tp.encode(df_raw, verbose=False, skip_warnings=False)
# print(df_encode)


# Testing scale method
# Note that this method must be tested with and without a passed list of features to scale
# scale_feats = ['age']

# df_scaled = tp.scale(df_raw, scale_feats, verbose=True, skip_warnings=True)
# df_scaled = tp.scale(df_raw, scale_feats, verbose=True, skip_warnings=False)
# df_scaled = tp.scale(df_raw, scale_feats, verbose=False, skip_warnings=True)
# df_scaled = tp.scale(df_raw, scale_feats, verbose=False, skip_warnings=False)

# df_scaled = tp.scale(df_raw, verbose=True, skip_warnings=True)
# df_scaled = tp.scale(df_raw, verbose=True, skip_warnings=False)
# df_scaled = tp.scale(df_raw, verbose=False, skip_warnings=True)
# df_scaled = tp.scale(df_raw, verbose=False, skip_warnings=False)
# print(df_scaled)


# Final test step is checking the full prep_df method
# Start with creating a sample dataframe of smaller scale
# df_test = pd.DataFrame({
#     'name': ['Jack', 'Joe', 'Jill', 'Jane', 'John'],
#     'age': [29, 31, 37, 34, 41],
#     'sex': ['M', 'M', 'F', 'F', np.nan],
#     'married': [0, 0, 1, 1, 1],
#     'happy': [3, 4, 5, 2, 2],
#     'region': ['Northeast', 'Central', 'Central', np.nan, np.nan]
# })

# Check dataframe content
# print(df_test)

# Define feature lists to pass at method call
# prep_encode = ['sex', 'married']
# prep_scale = ['age']

# Test without any passed feature lists
# processed_df = tp.prep_df(df_test)

# Test with one passed feature list
# processed_df = tp.prep_df(df_test, prep_encode)

# Test with two passed feature lists
# processed_df = tp.prep_df(df_test, prep_encode, prep_scale)

# Check prepared dataframe
# print(processed_df)


# SECOND-PASS TESTING (checking method augmentations with specific corner cases)

'''
Testing augmented df_info method
Need to check for detection of:
- Duplicate instances
- Low-variance features
- Infinite values
- Non-null empty strings
'''
info_df = pd.DataFrame({
    'name': ['John', 'Joe', 'Jack', 'James', 'Jake', 'John'],  # Creating one duplicate row for 'John'
    'age': [25, 57, 17, 31, np.nan, 25],  # Make sure we have a missing value
    'weight': [155, 155, 155, 155, 155, 155],  # Should trip the low-variance check
    'height': [67, 70, np.inf, 72, 65, 67],  # Should trip the infinite values check
    'hair_color': ['brown', 'black', '', 'blonde', 'bald', 'brown']  # Should trip the non-null empty string check
})

# This should report on the duplicated instance, but should not display the follow-up warning
# tp.df_info(info_df, verbose=False)

# This should display all warnings
# tp.df_info(info_df, verbose=True)


'''
Testing new subset method
Need to test for:
- True random sampling in subset
- Seeded random sampling in subset
- Stratified random sampling in subset
- Time-boundary creation in subsetting timeseries data
- Whether methodology explanations work as intended
- Whether automatic time interval identification works properly
- Summary of subset results
- Differentiation of UX in verbose vs. non-verbose modes 
'''
# Build non-timeseries dataset with values which should make subsetting validation easy
# subset_normal = pd.DataFrame(
#     {'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#      'name': ['Adam', 'Alice', 'Alex', 'Adrian', 'Ashton', 'Amanda', 'Alice', 'Ashley', 'Aaron', 'Austin'],
#      'sex': ['M', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'M', 'M'],
#      'age': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]}
# )

# df_subset_random = tp.subset(subset_normal, verbose=True)  # Verbose true random subset
# df_subset_random = tp.subset(subset_normal, verbose=False)  # Minimal-output true random subset
# print(df_subset_random)  # Print check

# df_subset_seeded = tp.subset(subset_normal, verbose=True)  # Verbose seeded random subset
# print(df_subset_seeded)

# df_subset_strat = tp.subset(subset_normal, verbose=True)  # Verbose stratified subset
# df_subset_strat = tp.subset(subset_normal, verbose=False)  # Minimal-output stratified subset
# print(df_subset_strat)

# Testing at day level
# subset_ts = pd.DataFrame(
#     {'date': [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d %H:%M') for x in range(10)],
#      'river_volume': [101513, 175474, 167819, 130656, 110106, 113368, 109843, 119749, 110872, 196164]}
# )
# # df_subset_ts = tp.subset(subset_ts, verbose=True)  # Verbose time-bound subset
# df_subset_ts = tp.subset(subset_ts, verbose=False)  # Minimal-output time-bound subset
# print(subset_ts)
# print()
# print(df_subset_ts)

# Testing at hour level
# subset_ts = pd.DataFrame(
#     {'date': [(datetime.now() - timedelta(hours=x)).strftime('%Y-%m-%d %H:%M') for x in range(10)],
#      'river_volume': [101513, 175474, 167819, 130656, 110106, 113368, 109843, 119749, 110872, 196164]}
# )
# df_subset_ts = tp.subset(subset_ts, verbose=True)  # Verbose time-bound subset
# print(subset_ts)
# print()
# print(df_subset_ts)


'''
Testing augmented rename_and_tag method
Need to test for:
- Detection of invalid python identifiers (e.g. new feature name that starts with a number)
- Detection of problematic characters (e.g. new feature name that contains a special character)
- Detection of bad anti-pattern (e.g. new feature which is all uppercase or very short)
- Detection of python keyword duplication (e.g. new feature name is the same as a python keyword)
- Whether the method catches duplicate feature names
- Whether the method catches "All selected features already tagged as ordinal"
- Whether the method catches "All selected features already tagged as targets"
- Format and appearance of change summary table
- Whether the verbose and non-verbose UXs are functioning appropriately 
'''
# df_orig = pd.DataFrame({
#     'animal': ['Elephant', 'Rhino', 'Hippo', 'Cheetah', 'Lion'],
#     'size_ord': [3, 2, 2, 1, 1],
#     'eats_people_target': ['No', 'No', 'No', 'Yes', 'Yes']
# })
# print(df_orig)  # Print check
# df_rename = tp.rename_and_tag(df_orig, verbose=True, tag_features=False)  # Test verbose with no tagging
# df_rename = tp.rename_and_tag(df_orig, verbose=False, tag_features=False)  # Test minimal output with no tagging
# df_rename = tp.rename_and_tag(df_orig, verbose=True, tag_features=True)  # Test verbose with tagging
# df_rename = tp.rename_and_tag(df_orig, verbose=False, tag_features=True)  # Test minimal output with tagging
# print(df_rename)


'''
Testing augmented impute method
Need to test for:
- Detection of false-numerical features
- Detection of low-variance features
- Detection of high outlier counts
- Detection of highly-correlated feature pairs
- Detection of timeseries data
- Detection of high missingness rate
- Constant-value imputation
- Random-sample imputation
- Visualizations (both single-feature and before-after)
- Final imputation summary table
'''
# non_ts_data = {
#     # Numerical features with different characteristics
#     'normal': [100, 102, 98, 103, 97, 101, np.nan, 100, 102, 98, 100, np.nan, 101, 97, 103],
#     'skewed': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0, 1.0, 2.0, 5.0, 15.0, 30.0, 50.0, 100.0],  # Extreme skew
#     'low_var': [50.01, 50.02, 50.01, 50.02, 50.02, 50.01, 50.02, 50.01, np.nan, 50.02, 50.01, 50.02, 50.01, 50.02,
#                 50.01],
#     'outliers': [105, 102, 98, 350, 101, 103, 99, 345, 101, np.nan, 100, 355, 102, 98, 101],
#     'corr_1': [2, 4, 6, 8, 10, 12, 14, 16, np.nan, 20, 22, 24, 26, 28, 30],
#     'corr_2': [2.1, 4.2, 6.1, 8.1, 10.2, 12.1, 14.2, 16.1, np.nan, 20.2, 22.1, 24.2, 26.1, 28.2, 30.1],
#
#     # Categorical feature
#     'cat_normal': ['A', 'B', np.nan, 'B', 'B', 'A', 'B', 'A', np.nan, 'B', 'A', 'B', np.nan, 'B', np.nan, ],
#
#     # False numerical features (actually categorical or ordinal)
#     'false_num_bin': [0, np.nan, 0, 1, 1, 0, 1, 0, 1, np.nan, 0, 1, 0, 1, 0],
#     'false_num_ord': [np.nan, 2, 3, 2, np.nan, 1, 4, 2, 3, 1, np.nan, 2, 4, 3, np.nan]
# }
#
# # Timeseries test data
# ts_data = {
#     'date': [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(15)],
#     'numeric_value': [100, np.nan, 98, 102, np.nan, 99, 101, 98, 102, np.nan, 105, 99, 101, np.nan, 103],
#     'categorical': ['A', 'B', np.nan, 'A', 'B', 'A', np.nan, 'B', 'A', 'B', np.nan, 'A', 'B', 'A', 'B']
# }

# Create test DataFrames
# df_normal = pd.DataFrame(non_ts_data)
# df_ts = pd.DataFrame(ts_data).sort_values('date').reset_index(drop=True)

# Test impute method on non-timeseries data
# df_normal_imputed = tp.impute(df_normal, verbose=True, skip_warnings=False)  # Full walkthrough
# df_normal_imputed = tp.impute(df_normal, verbose=False, skip_warnings=True)  # Super-streamlined
# print(df_normal_imputed)

# df_ts_imputed = tp.impute(df_ts, verbose=True, skip_warnings=False)  # Full walkthrough
# df_ts_imputed = tp.impute(df_ts, verbose=False, skip_warnings=True)  # Super-streamlined
# print(df_ts_imputed)


'''
Testing augmented feature_stats method
Need to test for:
- Fixed distribution pattern calculation for categorical features (verifying top N value coverage percentages)
- Empty datetime feature handling (when all values are null or can't be converted)
- Consistent numeric formatting across all statistics
  - Large integers (should show commas)
  - Decimal values (should display 4 decimal places)
  - Integer-like floats (should display as integers without decimals)
- Skewness metric testing:
  - Approximately symmetric distributions (|skew| < 0.5)
  - Moderately skewed distributions (0.5 ≤ |skew| < 1)
  - Highly skewed distributions (|skew| ≥ 1)
- Kurtosis metric testing:
  - Platykurtic distributions (kurt < -0.5)
  - Mesokurtic distributions (-0.5 ≤ kurt ≤ 0.5)
  - Leptokurtic distributions (kurt > 0.5)
- Edge cases:
  - Features with NaN values
  - Single-value features
  - Features with extremely large/small values
- Different numerical distributions:
  - Symmetric vs. skewed distributions
  - Features with zero mean (for coefficient of variation)
'''
# Create a test dataframe
# df_feature_stats = pd.DataFrame({
#     # Boolean features - both formats
#     'bool_true_false': [True, True, False, True, False, True, True, False, True, False, True, False, True, True, None,
#                         True, False, True, False, True],
#     'bool_binary': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
#
#     # Datetime feature - explicit datetime
#     'date_explicit': [
#         datetime(2023, 1, 1),
#         datetime(2023, 1, 2),
#         datetime(2023, 1, 3),
#         datetime(2023, 1, 4),
#         datetime(2023, 1, 5),
#         datetime(2023, 1, 6),
#         None,
#         datetime(2023, 1, 8),
#         datetime(2023, 1, 9),
#         datetime(2023, 1, 10),
#         datetime(2023, 1, 11),
#         datetime(2023, 1, 12),
#         datetime(2023, 1, 13),
#         datetime(2023, 1, 14),
#         datetime(2023, 1, 15),
#         datetime(2023, 1, 16),
#         datetime(2023, 1, 17),
#         datetime(2023, 1, 18),
#         datetime(2023, 1, 19),
#         datetime(2023, 1, 20)
#     ],
#
#     # Datetime as strings - to test detection logic
#     'date_strings': ['01-01-2023', '02-01-2023', '03-01-2023', '04-01-2023', '05-01-2023', '06-01-2023', None, None,
#                      '10-01-2023', '11-01-2023', None, '01-01-2024', '02-01-2024', '03-01-2024', '04-01-2024', None,
#                      '06-01-2024', '07-01-2024', '08-01-2024', '09-01-2024'],
#
#     # Categorical features
#     'cat_normal': ['A', 'B', 'C', 'A', 'B', 'D', 'A', 'E', 'B', 'A', 'C', 'D', 'E', 'B', None, 'A', 'C', 'B', 'A', 'D'],
#     'cat_empty': ['A', 'B', 'C', '', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', '', 'N', 'O', 'P', 'Q', 'R'],
#
#     # Zero-variance feature - to test detection
#     'zero_variance': ['Same'] * 20,
#
#     # Near-constant feature (exactly 95% single value)
#     'near_constant': ['Frequent'] * 19 + ['Rare'],  # 19/20 = 95%
#
#     # Duplicated feature - to test potential duplicate detection
#     'cat_normal_dup': ['A', 'B', 'C', 'A', 'B', 'D', 'A', 'E', 'B', 'A', 'C', 'D', 'E', 'B', None, 'A', 'C', 'B',
#                        'A', 'D'],
#
#     # Numerical features
#     # Large integers - for formatting with commas
#     'num_large': [1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 11000000,
#                   12000000, 13000000, 14000000, 15000000, 16000000, 17000000, 18000000, 19000000, 20000000],
#
#     # Decimal values - for 4 decimal place formatting
#     'num_decimal': [10.1234, 11.5678, 9.7531, 10.5, 12.6789, 8.25, 11.125, 9.5, 10.375, None, 13.4567, 7.8901, 12.3456,
#                     9.9999, 11.1111, 14.7890, 8.6543, 10.0001, 12.5000, 9.8765],
#
#     # Integer-like floats - should display as integers
#     'num_int_floats': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
#                        18.0, 19.0, 20.0],
#
#     # Skewness test cases
#     # Approximately symmetric (|skew| < 0.5)
#     'num_symmetric': [50, 45, 55, 48, 52, 47, 53, 49, 51, 50, 46, 54, 48, 52, 50, 47, 53, 49, 51, 50],
#
#     # Moderately skewed (0.5 ≤ |skew| < 1)
#     'num_mod_skewed': [10, 12, 11, 13, 14, 18, 12, 11, 10, 15, 13, 12, 14, 16, 11, 17, 13, 12, 14, 11],
#
#     # Highly skewed (|skew| ≥ 1)
#     'num_highly_skewed': [1, 2, 1, 3, 1, 2, 1, 10, 200, 500, 1, 2, 3, 1, 2, 1, 2, 3, 1, 100],
#
#     # Kurtosis test cases
#     # Platykurtic (kurt < -0.5)
#     'num_platykurtic': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 15, 25, 35, 45, 55, 65, 75, 85, 95, 5],
#
#     # Mesokurtic (-0.5 ≤ kurt ≤ 0.5)
#     'num_mesokurtic': [100, 95, 105, 98, 102, 97, 103, 99, 101, 100, 96, 104, 98, 102, 100, 97, 103, 99, 101, 100],
#
#     # Leptokurtic (kurt > 0.5)
#     'num_leptokurtic': [50, 50, 50, 50, 50, 51, 49, 10, 90, 50, 50, 50, 50, 49, 51, 50, 50, 50, 50, 50],
#
#     # Zero mean - for coefficient of variation handling
#     'num_zero_mean': [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#
#     # Feature with nulls
#     'num_with_nulls': [1, 2, None, 4, 5, None, 7, 8, 9, 10, None, 12, 13, 14, 15, None, 17, 18, 19, 20]})
#
# # tp.feature_stats(df_feature_stats, verbose=True)  # Testing full output
# tp.feature_stats(df_feature_stats, verbose=False)  # Testing minimal output

'''
Testing augmented scale method
Need to test for:
- Operation with and without features_to_scale list
- Preservation of original features
- Detection of false-numeric features (e.g. 0/1 'Yes'/'No' format)
- Detection of null values, infinite values, high skewness
- Before/after visualization
- Custom minmax scaler ranges
'''
# Build test dataframe
# df_scale = pd.DataFrame({'Name': ['John', 'Joe', 'Jack', 'Jake', 'Jeff', 'Jim'],
#                          'Weight': [165, 186, 199, 207, 153, 223],
#                          'Citizen': [1, 0, 0, 1, 1, 1],
#                          'Score': [78, 89, np.inf, np.nan, np.nan, 92],
#                          'Happiness': [81, 84, 76, 91, 65, 79],
#                          'Zero_var': [55, 55, 55, 55, 55, np.nan]})
# Build list of features to scale
# scale_feats = ['Weight', 'Happiness']
# df_scaled = tp.scale(df_scale, features_to_scale=None, verbose=True, skip_warnings=False, preserve_features=True)
# df_scaled = tp.scale(df_scale, features_to_scale=scale_feats, verbose=True, skip_warnings=False, preserve_features=True)
# df_scaled = tp.scale(df_scale, features_to_scale=None, verbose=False, skip_warnings=False, preserve_features=True)
# df_scaled = tp.scale(df_scale, features_to_scale=None, verbose=False, skip_warnings=False, preserve_features=False)
# df_scaled = tp.scale(df_scale, features_to_scale=None, verbose=False, skip_warnings=True, preserve_features=False)
# print(df_scaled)

'''
Testing refactored encode method
Need to test for:
- Operation with and without features_to_encode list
- Clean column name generation for encoded features
- Preservation of original features (preserve_features=True)
- Handling of all-NaN columns (should be skipped)
- Proper reindexing when using missing_strat='drop' with non-continuous indices
- Missing Value Handling:
   - 'ignore' strategy (default) - leaves NaNs in encoded columns
   - 'category' strategy - creates separate indicator column for NaNs
   - 'drop' strategy - processes only non-null values
- Detection of false-numeric features (e.g., 0/1 'Yes'/'No' format)
- Handling of high-cardinality features
- Handling columns with single value (should be skipped)
- Handling columns with rare categories
- Safely handling values that can't be converted to float
- Custom reference category selection for dummy encoding
- Custom prefix selection for encoded columns
- Before/after distribution visualizations
'''
# # Create sample test dataset
# encode_df = pd.DataFrame({
#     # Binary categorical feature with missing values (20%)
#     'bin_cat': ['Yes', 'No', 'Yes', np.nan, 'Yes', 'No', 'Yes', 'No', 'Yes', np.nan],
#     # False-numeric feature (0/1 representation of Yes/No)
#     'false_num': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
#     # Feature with special characters in name - should create clean column names
#     'special chars & symbols!': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X'],
#     # Feature with a single value (should be skipped)
#     'single_val': ['Constant'] * 10,
#     # Feature with values that can't be converted to float
#     'non_num_vals': ['Value_A', 'Value_B', 'Value_C', 'Value_A', 'Value_B', 'Value_C', 'Value_A', 'Value_B',
#                      'Value_C', 'Value_A'],
#     # True numerical feature (should not be encoding candidate)
#     'true_num': [45.2, 67.8, 32.1, 58.9, 71.3, 49.5, 62.7, 38.4, 53.6, 44.9],
#     # Ordinal feature
#     'ordinal': ['Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Medium'],
#     # Column with all NaN values (should be skipped)
#     'all_nan': [np.nan] * 10,
#     # Unicode characters to test handling of special encoding situations - the special characters should persist
#     'unicode_chars': ['café', 'résumé', 'piñata', 'café', 'résumé', 'piñata', 'café', 'résumé', 'piñata', 'café']})

# # Build list of features to encode
# encode_feats = ['bin_cat', 'ordinal']

# df_encoded = tp.encode(encode_df, features_to_encode=None, verbose=True, skip_warnings=False, preserve_features=True)
# df_encoded = tp.encode(encode_df, features_to_encode=None, verbose=True, skip_warnings=False, preserve_features=False)
# df_encoded = tp.encode(encode_df, features_to_encode=encode_feats, verbose=True, skip_warnings=False, preserve_features=False)
# df_encoded = tp.encode(encode_df, features_to_encode=encode_feats, verbose=False, skip_warnings=False, preserve_features=False)
# df_encoded = tp.encode(encode_df, features_to_encode=encode_feats, verbose=False, skip_warnings=True, preserve_features=False)
# print(df_encoded)

'''
Testing new find_outliers method
Need to test for:
- Verbose and non-verbose operation
- All three outlier detection methods
- Custom and default thresholds
- Structure of returned dictionary output
'''
# outlier_df = pd.DataFrame({'high_outlier': [5, 10, 10, 15, 15, 20, 20, 25, 25, 8000],
#                            'low_outlier': [5, 10, 10, 15, 15, 20, 20, 25, 25, -8000],
#                            'one_outlier': [1, 1, 1, 2, 2, 2, 3, 3, 4, 20],
#                            'two_outliers': [1, 1, 2, 2, 2, 3, 3, 3, 45, 65]})

# outlier_dict = tp.find_outliers(outlier_df, method='iqr', threshold=None, verbose=True)
# outlier_dict = tp.find_outliers(outlier_df, method='zscore', threshold=None, verbose=True)
# outlier_dict = tp.find_outliers(outlier_df, method='zscore', threshold=1, verbose=True)
# outlier_dict = tp.find_outliers(outlier_df, method='modified_zscore', threshold=None, verbose=True)
# outlier_dict = tp.find_outliers(outlier_df, method='modified_zscore', threshold=None, verbose=False)
# print(outlier_dict)

'''
Testing new find_corrs method
Need to test for:
- Verbose and non-verbose operation
- All three correlation detection methods
- Custom and default correlation thresholds
- Structure of returned dictionary output
'''
# # Base feature - simple linear sequence
# base = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

# # Create features with controlled correlation levels
# corr_df = pd.DataFrame({
#     'base_feature': base,
#     # Perfect positive correlation
#     'perfect_pos': base,
#     # Perfect negative correlation
#     'perfect_neg': -base + 16,
#     # Strong positive correlation
#     'strong_pos': base + np.random.normal(0, 3, len(base)),
#     # Strong negative correlation
#     'strong_neg': -base + 16 + np.random.normal(0, 3, len(base)),
#     # Moderate positive correlation
#     'mod_pos': base + np.random.normal(0, 6, len(base)),
#     # Moderate negative correlation
#     'mod_neg': -base + 16 + np.random.normal(0, 6, len(base)),
#     # Weak positive correlation
#     'weak_pos': base + np.random.normal(0, 12, len(base)),
#     # Weak negative correlation
#     'weak_neg': -base + 16 + np.random.normal(0, 12, len(base)),
#     # Uncorrelated
#     'uncorr': np.random.normal(8, 4, len(base)),
#     # Object-type feature which should be ignored
#     'object': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
# })

# # Add a column with missing values (based on perfect_pos)
# corr_df['missing_vals'] = corr_df['perfect_pos'].copy()
# # Set ~20% of values to NaN
# mask = np.random.choice([True, False], size=len(base), p=[0.2, 0.8])
# corr_df.loc[mask, 'missing_vals'] = None

# # Build results dictionary
# # corr_dict = tp.find_corrs(corr_df, method='pearson', verbose=True)
# # corr_dict = tp.find_corrs(corr_df, method='spearman', verbose=True)
# corr_dict = tp.find_corrs(corr_df, method='kendall', verbose=True)
# # corr_dict = tp.find_corrs(corr_df, method='pearson', threshold=0.5, verbose=True)
# # corr_dict = tp.find_corrs(corr_df, method='pearson', threshold=0.9, verbose=False)
# # Display results dictionary
# print(corr_dict)

'''
Testing new transform method
Need to test for:
- Verbose and non-verbose operation
- Auto-detected features and list of features
- Feature preservation
- Warning messages operating correctly
'''
# Create test dataframe
transform_df = pd.DataFrame({'name': ['John', 'Joe', 'Jack', 'Jill', 'Jeff'],
                             'sex': ['M', 'M', 'M', 'F', 'M'],
                             'age': [35, 39, 27, 32, 28],
                             'int_10': [10, 20, 30, 40, 50],
                             'binary': [0, 1, 1, 0, 1],
                             'zeros': [1, 0, 3, 0, 5],
                             'nulls': [np.nan, 5, 10, np.nan, 15]})
# Create list of features to be transformed
transform_list = ['age', 'int_10', 'binary', 'zeros', 'nulls']
# Testing auto-detect
# transform_results = tp.transform(transform_df, features_to_transform=None, verbose=True,
#                                  preserve_features=True, skip_warnings=False)
# Pre-passed list of features
# transform_results = tp.transform(transform_df, features_to_transform=transform_list, verbose=True,
#                                  preserve_features=True, skip_warnings=False)
# # Non-verbose operation
# transform_results = tp.transform(transform_df, features_to_transform=None, verbose=False,
#                                  preserve_features=True, skip_warnings=False)
# print(transform_results)
