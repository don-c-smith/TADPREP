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
tp.df_info(df, verbose=False)

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
This method Interactively reshapes the input DataFrame according to user specification.
Parameters: verbose (bool, default=True) 
    - Controls whether detailed process information is displayed
Returns: None
'''
# Test non-verbose mode first
tp.reshape(df, verbose=False)

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