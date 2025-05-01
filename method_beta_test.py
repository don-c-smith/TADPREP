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
- Does it do what a reasonable person would expect it to do?
- Are we missing any major capabilities? Is this all the 'info' we need?
- Are there extraneous capabilities present in the method?
- Are all parameters/modes necessary and/or appropriate?
- What problems or needed changes were identified? 
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
- Does it do what a reasonable person would expect it to do?
- Are we missing any major capabilities? Is this all the reshaping capacity we usefully need?
- Are there extraneous capabilities present in the method?
- Are all parameters/modes necessary and/or appropriate?
- What problems or needed changes were identified? 
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
# df_subset = tp.subset(df, verbose=True)

# Print subsetted dataframe
# print(df_subset)

'''
Questions:
- Is the name of this method appropriate?
- Does it do what a reasonable person would expect it to do?
- Are we missing any major capabilities? Is this all the subset capacity we usefully need?
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
- Are we missing any major capabilities? Is this all the subset capacity we usefully need?
- Are there extraneous capabilities present in the method?
- Are all parameters/modes necessary and/or appropriate?
- What problems or needed changes were identified? 
'''