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
Testing the subset method:
This method subsets the input DataFrame according to user specification.
Parameters: verbose (bool, default=True) 
    - Controls whether detailed process information and methodological guidance is displayed

Returns: The modified DataFrame as subset by the user's specifications
'''
# Test non-verbose mode first
# df_subset = tp.subset(df, verbose=False)

# Test verbose mode
df_subset = tp.subset(df, verbose=True)

# Print subsetted dataframe
print(df_subset)

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

# DON: CREATE A DATASET WITH OUTLIERS!
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