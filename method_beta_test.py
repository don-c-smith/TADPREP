import numpy as np
import pandas as pd
import tadprep as tp
# This is the file we will use for beta testing of public-facing methods before subsequent debugging

# Load test data
df = pd.read_csv(r'C:\Users\doncs\Documents\GitHub\TADPREP\data\river_data.csv')
# Print check
# print(df)

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
test_df = pd.DataFrame({
    # Near-constant feature (>95% single value)
    'near_constant': ['common_value'] * 19 + ['rare_value'],  # 95% same value
    # Feature with infinite values
    'has_inf': [1.0, 2.0, float('inf'), 4.0, 5.0] + [6.0] * 15,
    # Feature with empty strings (distinct from NaN)
    'empty_strings': ['value1', '', 'value2', '', 'value3'] + ['value4'] * 15,
    # Numeric data stored as strings
    'num_as_string': ['100', '200', '300', '400', '500'] + ['600'] * 15,
    # Normal numeric feature (for comparison)
    'normal_num': [10, 20, 30, 40, 50] + [60] * 15,
    # Normal string feature (for comparison)
    'normal_string': ['apple', 'banana', 'cherry', 'date', 'elderberry'] + ['fig'] * 15
})

# Add duplicate rows
dup_row = pd.DataFrame({
    'near_constant': ['common_value'],
    'has_inf': [6.0],
    'empty_strings': ['value4'],
    'num_as_string': ['600'],
    'normal_num': [60],
    'normal_string': ['fig']
})
test_df = pd.concat([test_df, dup_row, dup_row], ignore_index=True)  # Add 2 duplicate rows

# Add a completely empty row (all NaN)
empty_row = pd.DataFrame([{col: np.nan for col in test_df.columns}])
test_df = pd.concat([test_df, empty_row], ignore_index=True)

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


# *** PICK UP TEAM DEBUGGING HERE ***
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
- Does it do what a reasonable person would expect it to do?
- Are we missing any major capabilities? Is this all the reshaping capacity we usefully need?
- Are there extraneous capabilities present in the method?
- Are all parameters/modes necessary and/or appropriate?
- What problems or needed changes were identified? 
'''


'''
Testing the find_outliers method:
This method detects outliers in numerical features of a DataFrame using a specified detection method.
Parameters:
method : str, default='iqr'
    Outlier detection method to use.
    Options:
      - 'iqr': Interquartile Range (default)
      - 'zscore': Standard Z-score
      - 'modified_zscore': Modified Z-score

threshold : float, default=None
    Threshold value for outlier detection. If None, uses method-specific defaults:
      - For IQR: 1.5 × IQR
      - For Z-score: 3.0 standard deviations
      - For Modified Z-score: 3.5

verbose : bool, default=True
    Whether to print detailed information about outliers

Returns: A dictionary containing outlier information with summary and feature-specific details
'''
df_outliers = pd.DataFrame({
    'cat_feature': ['Bob', 'Bob', 'Bob', 'Bob', 'Bob', 'Bob', 'Bob', 'Bob', 'Bob', 'Bob', ],
    'no_outliers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'one_small_outlier': [1, 1, 1, 1, 1, 1, 1, 1, 1, 10],
    'one_large_outlier': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1000],
    'two_small_outliers': [1, 1, 1, 1, 1, 1, 1, 1, 10, 50],
    'two_large_outliers': [1, 1, 1, 1, 1, 1, 1, 1, 1000, 50000],
    'constant_feature': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'all_missing': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    })

# Test default mode first
# outlier_dict = tp.find_outliers(df)

# Test non-verbose mode
# outlier_dict = tp.find_outliers(df, verbose=False)

# Test other detection methodologies
# outlier_dict = tp.find_outliers(df, method='zscore')
# outlier_dict = tp.find_outliers(df, method='modified_zscore')

# Test specified threshold
# outlier_dict = tp.find_outliers(df, threshold=0.75)

# Print outlier dictionary
# print(outlier_dict)

'''
Questions:
- Is the name of this method appropriate?
- Does it do what a reasonable person would expect it to do?
- Are we missing any major capabilities? Is this all the outlier detection capacity we usefully need?
- Are there extraneous capabilities present in the method?
- Are all parameters/modes necessary and/or appropriate?
- What problems or needed changes were identified? 
'''