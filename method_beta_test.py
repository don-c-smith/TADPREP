import numpy as np
import pandas as pd
import tadprep as tp
# This is the file we will use for beta testing of public-facing methods before subsequent debugging

# Load test data
df = pd.read_csv(r'C:\Users\doncs\Documents\GitHub\TADPREP\data\river_data.csv')
# Print check
# print(df)

'''
Testing the method_list method:
Prints the names and brief descriptions of all callable methods in the TADPREP library.
'''
# tp.method_list()

'''
Questions:
- Is the name of this method appropriate?
Yes
- Does it do what a reasonable person would expect it to do?
Yes
- Are we missing any major capabilities?
No
- Are there extraneous capabilities present in the method?
No
- Are all parameters/modes necessary and/or appropriate?
N/A
- What problems or needed changes were identified?
None
'''


'''
Testing the df_info method:
This method prints comprehensive information about a DataFrame's structure, contents, and potential data quality issues.
Parameters: verbose (bool, default=True) 
    - Controls whether detailed feature information and data quality checks are displayed

Returns: None
'''
# Test non-verbose mode first
# tp.df_info(df, verbose=False)

# Test verbose mode
# tp.df_info(df, verbose=True)

'''
Questions:
- Is the name of this method appropriate?
Rename to 'summarize'
- Does it do what a reasonable person would expect it to do?
Yes
- Are we missing any major capabilities? Is this all the 'info' we need?
Check for features which are all ints or floats but typed as strings
Check for any instances which are all-Null in all features
- Are there extraneous capabilities present in the method?
No
- Are all parameters/modes necessary and/or appropriate?
Remove verbose mode entirely. It's too simple/low-detail. Just have this be a zero-param method.
- What problems or needed changes were identified?
    1. Remove verbose mode entirely, both from driver function and public method
    2. Implement check for ints or floats typed as strings
    3. Implement check for all-Null instances

** Refactoring to be done by Don **
'''

'''
Testing refactored 'summary' method (was df_info):
'''
# Using normal dataset
# tp.summary(df)

# Stress-testing the data quality checks in the method
# Create a test dataframe with examples for each data quality check
# test_df = pd.DataFrame({
    # Near-constant feature (>95% single value)
#     'near_constant': ['common_value'] * 19 + ['rare_value'],  # 95% same value
#     # Feature with infinite values
#     'has_inf': [1.0, 2.0, float('inf'), 4.0, 5.0] + [6.0] * 15,
#     # Feature with empty strings (distinct from NaN)
#     'empty_strings': ['value1', '', 'value2', '', 'value3'] + ['value4'] * 15,
#     # Numeric data stored as strings
#     'num_as_string': ['100', '200', '300', '400', '500'] + ['600'] * 15,
#     # Normal numeric feature (for comparison)
#     'normal_num': [10, 20, 30, 40, 50] + [60] * 15,
#     # Normal string feature (for comparison)
#     'normal_string': ['apple', 'banana', 'cherry', 'date', 'elderberry'] + ['fig'] * 15
# })

# Add duplicate rows
# dup_row = pd.DataFrame({
#     'near_constant': ['common_value'],
#     'has_inf': [6.0],
#     'empty_strings': ['value4'],
#     'num_as_string': ['600'],
#     'normal_num': [60],
#     'normal_string': ['fig']
# })
# test_df = pd.concat([test_df, dup_row, dup_row], ignore_index=True)  # Add 2 duplicate rows

# Add a completely empty row (all NaN)
# empty_row = pd.DataFrame([{col: np.nan for col in test_df.columns}])
# test_df = pd.concat([test_df, empty_row], ignore_index=True)

# Print check for test dataframe
# print(test_df)

# Test data quality checks
# tp.summary(test_df)


'''
Testing the subset method:
This method subsets the input DataFrame according to user specification.
Parameters: verbose (bool, default=True) 
    - Controls whether detailed process information and methodological guidance is displayed

Returns: The modified DataFrame as subset by the user's specifications
'''
# Test non-verbose mode first
# df_subset = tp.subset(df, verbose=False)

# Test verbose mode
# df_subset = tp.subset(df, verbose=True)

# Print subsetted dataframe
# print(df_subset)

'''
Questions:
- Is the name of this method appropriate?
Yes
- Does it do what a reasonable person would expect it to do?
Yes
- Are we missing any major capabilities? Is this all the subset capacity we usefully need?
 No
- Are there extraneous capabilities present in the method?
No
- Are all parameters/modes necessary and/or appropriate?
Yes
- What problems or needed changes were identified?
    1. BUG: Loop is stepping from proportion entry back to feature selection in stratified sampling when invalid input
        is entered for the proportion to subset. Likely a while loop scope problem.
    2. When subsetting by date, error messages for entering an invalid date need to be more clear/informative
        (e.g. if you enter 200 for the year, we need something more informative than "nanosecond error")
    3. Investigate datetime format handling - what's available, what's missing, is there a better/more complete way?
        Should the current verbose implementation persist in non-verbose mode?
    4. The "Randomly dropped 25.0% of instances. 45 instances remain."-type message should print in non-verbose mode.
        Consider expressing as pre-sampling/post-sampling information. Should be consistent for all sampling methods.
    5. BUG: Bizarre error message printing when attempting to stratify by feature with missing values: INVESTIGATE
    6. Asking user for explanation in verbose mode is redundant, if verbose, print explanations

** Refactoring to be done by Gabor ** 
'''


'''
Testing the reshape method:
This method interactively reshapes the input DataFrame according to user specification.
Parameters: verbose (bool, default=True) 
    - Controls whether detailed process information is displayed

Returns: None
'''
# Test non-verbose mode first
# tp.reshape(df, verbose=False)

# Test verbose mode
# tp.reshape(df, verbose=True)

'''
Questions:
- Is the name of this method appropriate?
Yes
- Does it do what a reasonable person would expect it to do?
Yes
- Are we missing any major capabilities? Is this all the reshaping capacity we usefully need?
Yes
- Are there extraneous capabilities present in the method?
No
- Are all parameters/modes necessary and/or appropriate?
Yes
- What problems or needed changes were identified?
    - Clarify ability to select multiple reshape methods at user prompt
    - Add ability to select individual features to drop if list not passed
    - Counts of missing values by feature should be made more legible
    - Enumerate features to drop rows by and have user pass indices, not feature names
    - BUG: Generalized degree of population, default decimal-percent throwing traceback (final_thresh feature)
    - Move explanation out of input dependence, have it run if verbose is true
'''


'''
Testing the find_corrs method:
This method finds correlations in numerical features of a DataFrame using a specified detection method.
Args:
    df (pd.DataFrame): The DataFrame to analyze for correlated features
    method (str, optional): Correlation method to use. Options:
        - 'pearson': Standard correlation coefficient (default)
        - 'spearman': Rank correlation, robust to outliers and non-linear relationships
        - 'kendall': Another rank correlation, more robust for small samples
    threshold (float, optional): Correlation coefficient threshold (absolute value).
        Defaults to 0.8. Values should be between 0 and 1.
    verbose (bool, optional): Whether to print detailed information about correlations.
        Defaults to True.

Returns: A dictionary containing correlation information with summary statistics and detailed pair information.
'''
# df_corrs_errors = pd.DataFrame({
#     'num_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'num_2': [2, 5, 8, 9, 11, 12, 14, 15, 17, 20],
#     'cat': ['dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog']
# })

# Make sure 'invalid method' catch fires
# tp.find_corrs(df_corrs_errors, method='magic_wizard')  # Error was caught

# Make sure 'custom threshold must be between 0 and 1' catch fires
# tp.find_corrs(df_corrs_errors, threshold=1.5)  # Too-high error was caught
# tp.find_corrs(df_corrs_errors, threshold=-1.5)  # Too-low error was caught

# Make sure 'at least two numerical features must be present' catch fires
# tp.find_corrs(df_corrs_errors)  # Error was caught

# # Build useful data
# df_corrs = pd.DataFrame({
#     'linear': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'linear_double': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
#     'linear_neg': [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
#     'near_linear': [1, 8, 13, 21, 36, 40, 53, 65, 77, 89],
#     'noise': [28, -70, 576, 2856, -7798, 44, -90, 49607, 1000000, -2568637],
#     'cat': ['dog', 'fish', 'bear', 'cat', 'kangaroo', 'whale', 'leopard', 'mongoose', 'badger', 'elephant'],
#     'missing_vals': [567, 265, 476, 244, 670, None, None, None, None, None]
# })
# # Test method's normal operation
# corr_dict = tp.find_corrs(df_corrs)

# Test non-verbose operation
# corr_dict = tp.find_corrs(df_corrs, verbose=False)

# Test alternate methods
# corr_dict = tp.find_corrs(df_corrs, method='spearman')
# corr_dict = tp.find_corrs(df_corrs, method='kendall')

# Test custom threshold
# corr_dict = tp.find_corrs(df_corrs, threshold=1)

# Print correlation dictionary
# print(corr_dict)

'''
Questions:
- Is the name of this method appropriate?
Yes
- Does it do what a reasonable person would expect it to do?
Yes
- Are we missing any major capabilities? Is this all the correlation detection capacity we usefully need?
I believe so
- Are there extraneous capabilities present in the method?
No
- Are all parameters/modes necessary and/or appropriate?
Yes
- What problems or needed changes were identified?
We need a better way of handling missing values more gracefully. Right now we just have a fail-out if all arrays aren't
of the exact same length. We need to print something more explicative to the user and maybe suggest imputation or
dropping features with missing values using either the reshape or impute methods. 
'''


'''
Testing the make_plots method:
This method interactively creates and displays plots for features in a DataFrame.
Parameters
----------
df : pandas.DataFrame
    The DataFrame containing features to plot.
features_to_plot : list[str] | None, default=None
    Optional list of specific features to consider for plotting. If None, the
    function will use all features in the DataFrame.

Returns
-------
None
    This function displays plots but does not return any values.
'''
# Test plotting
# Features are date, season, volume, avg_flag, clarity, samples, traffic

# Test first with no passed feature list
# tp.make_plots(df)

# Test with passed feature list
# tp.make_plots(df, features_to_plot=['season', 'volume', 'clarity'])

'''
Questions:
- Is the name of this method appropriate?
I'd rather call it plot_features
- Does it do what a reasonable person would expect it to do?
Yes
- Are we missing any major capabilities? Is this all the plotting capacity we usefully need?
I think at a top level, yes. More complex plots should be hand-coded. This is an EDA tool.
- Are there extraneous capabilities present in the method?
I don't think so.
- Are all parameters/modes necessary and/or appropriate?
Yes. The list-to-plot is useful if you have a lot of features or some prior knowledge of what you want to look at.
- What problems or needed changes were identified?
None
'''

'''
Testing the rename_and_tag method:
This method interactively renames features and allows the user to tag them as ordinal or target features, if desired.
Parameters
----------
df : pandas.DataFrame
    The DataFrame whose features need to be renamed and/or tagged
verbose : bool, default = True
    Controls whether detailed process information is displayed
tag_features : default = False
    Controls whether activate the feature-tagging process is activated

Returns
-------
pandas.DataFrame
    The DataFrame with renamed/tagged features


'Bad' input which should be tried to check for 'catches' when renaming features are new feature names which:
- Contain spaces
- Contain special characters (e.g. @, %, &)
- Contain double underscores
- Start with an integer
- Contains a python keyword (e.g. class, True, for)

'Poor practice' input which should be tried to check for 'catches' when renaming features are new feature names which:
- Are all uppercase
- Are quite short (<=2 characters)
- Are quite long (>=30 characters)
'''

# Testing default settings first
df_renamed = tp.rename_and_tag(df, verbose=True, tag_features=False)

# Testing non-verbose operation
# df_renamed = tp.rename_and_tag(df, verbose=False, tag_features=False)

# Testing verbose feature tagging
# df_renamed = tp.rename_and_tag(df, verbose=True, tag_features=True)

# Testing non-verbose feature tagging
# df_renamed = tp.rename_and_tag(df, verbose=False, tag_features=True)

print(df_renamed)
'''
Questions:
- Is the name of this method appropriate?

- Does it do what a reasonable person would expect it to do?

- Are we missing any major capabilities? Is this all the subset capacity we usefully need?

- Are there extraneous capabilities present in the method?

- Are all parameters/modes necessary and/or appropriate?

- What problems or needed changes were identified?
'''