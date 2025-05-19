# TADPREP
*A personal project to generate a time-saving tabular data preprocessing library for Python, along with a resilient 
end-to-end CLI-based data preparation script for less-technical users.*

### Background
The TADPREP (**T**abular **A**utomated **D**ata **PREP**rocessing) library is a unified, importable package 
intended to streamline the preprocessing of tabular data. The data preparation process that is performed on tabular 
datafiles before advanced data analysis is reasonably consistent and often leads to the reuse and/or 
reproduction of older, potentially-deprecated data preprocessing code. The TADPREP library helps to obviate that 
code reproduction and to save a busy data scientist time and effort.

While every analytical task inevitably has its own particular corner cases and unique challenges, the core steps 
involved in tabular data preparation (especially assuming a naive-case approach) are consistent in form and order.
Writing fresh code to handle data preparation for each new project is labor-intensive and time-inefficient. 

As such, the TADPREP library (in addition to providing a full end-to-end script which can walk a user through data 
preprocessing) contains a number of core data-mutation methods which, when combined as per the user's specifications, 
can greatly accelerate their data preprocessing workflow.

TADPREP leverages validated data preparation libraries, extensive error handling, and a streamlined, lightweight UX 
intended to make the tabular data preparation process as smooth and low-effort as possible.

TADPREP can be imported into any standard IDE. The end-to-end pipeline script can be run directly from the CLI.

### Using the End-to-End Pipeline Script on the CLI
The TADPREP library contains an end-to-end CLI-executable data preprocessing pipeline script. Once the TADPREPS library
is installed on your machine, this interactive script can be activated by running the following command on the CLI:
```bash
tadprep
```

Alternately, if that simple execution method fails, you may run this more explicit call on the CLI:
```bash
python -m tadprep
```

This script serves as a more fully interactive, "hand-holding" process which will walk a user through the ingestion of
a tabular file, the mutation of the data in the file, and the export of the transformed data into a new static file. 

The script accepts .csv, .tsv, and Excel-type files as input, and supports the creation of .csv, .tsv, Excel, Pickle, 
Parquet, and JSON files during the file-export process.

The end-to-end script has two primary use cases:
1. It can be used by less programming-proficient people who need some more help with the process.
2. It can be used by anyone who is engaged in a straightforward "flat to flat" data-processing task.

The script has extensive error handling, and in addition to its console output, it generates a summary runtime
log which describes the major decisions made during the process. (The log also documents any major errors that occur.)
The log file is automatically saved in the same directory as the final exported datafile.

**Notes:**
- The interactive pipeline script can only ingest locally-stored files.
- Users may identify the file to import and the export save directory either by passing an absolute filepath or by 
using a file browser window.
- In order for the script to run properly, users will need write-level permissions on their machines.
- The script includes a dependency check to ensure that all dependent libraries are installed and operable.
- The script is not readily "customizable," and is intended for use in ordinary data processing tasks. 
- Users who wish to segment or extend the core data mutation steps should import the package into an IDE and use the 
supplied methods to augment their own task-specific data-processing code.
- The script now includes a rollback capability, allowing users to revert to previous stages in the pipeline if needed.

### Using the Library Methods
The TADPREP library contains a series of callable methods, each of which represent specific subsets/segments of the full
data preparation process as conducted in the interactive CLI pipeline script. 

These methods are intended for in-IDE use such that a user may import a dataset into a Pandas dataframe by whatever 
means they deem appropriate, and then apply the data mutation methods as desired to that dataframe.

These methods are broadly parallel in form and function to various Pandas methods. Therefore, the process of using them
ought to be relatively comprehensible to anyone used to working with the Pandas library.

The methods provide a great deal of visual parsing of console output and offer explanations/warnings which are relevant 
to each step. Each method has a boolean `verbose` parameter (which is `True` by default), but can be set to `False` when 
the method is called to simplify printed output and suppress explanations.

Additionally, some methods have a further boolean `skip_warnings` parameter (which is `False` by default), but can be 
set to `True` to allow the user to skip past any mathematics- or best-practice related warnings relevant to a given 
data preprocessing step, e.g. warnings about summoning the curse of dimensionality when encoding high-cardinality 
features.

Therefore, experienced data scientists who have an excellent sense of their dataset and their intentions may prefer to
call the methods with `verbose=False` and `skip_warnings=True` to have an efficient, "quiet" user experience and to
move the data preparation process along as quickly as possible.

**EXAMPLE USAGE:**
```python
import pandas as pd
import tadprep as tp
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression

# Create database connection
engine = create_engine('postgresql://username:password@localhost:5432/mydatabase')

# Run query and store results in Pandas dataframe
df_raw = pd.read_sql_table(
    table_name='sales_data',
    con=engine,
    schema='public',
    columns=['sale_id', 
             'rep_id', 
             'product_name', 
             'quantity', 
             'sale_date', 
             'customer_satisfaction'],
    index_col='sale_id'
)

# Display general dataframe information using TADPREP
tp.summary(df_raw)

# Reshape data using TADPREP
df_reshape = tp.reshape(df_raw, verbose=False)

# Perform imputation using TADPREP
df_imputed = tp.impute(df_reshape, verbose=False, skip_warnings=True)

# Run basic regression on reshaped data using 'customer_satisfaction' as the target feature
X = df_imputed.drop(['customer_satisfaction'], axis=1)
y = df_imputed['customer_satisfaction']

model = LinearRegression()
model.fit(X, y)
```

### Full Library Method List

TADPREP provides a suite of interactive methods for data preparation, each focused on a specific aspect of the 
data preparation process:

#### Core Pipeline Method

`prep_df(df, features_to_encode=None, features_to_scale=None)`
- Runs the complete TADPREP pipeline on an existing DataFrame
- Guides users through all preparation steps interactively
- Takes optional parameters for specifying features to encode and scale
- Returns the fully prepared DataFrame
- Ideal for users who want to prepare their data in a single cohesive workflow
- This method (in essence) runs the full interactive script without the file I/O process

#### Library-Level Informative Methods
`method_list()`
- Displays the names and brief descriptions of all callable methods supported by the TADPREP library
- Methods are categorized by general functionality/process type

#### Individual Data Preparation Methods

`summary(df, verbose=True)`
- Displays comprehensive information about DataFrame structure and contents
- Shows feature counts, data types, and missingness statistics
- Identifies potential data quality issues such as:
  - Near-constant features (>95% single value)
  - Features containing infinite values
  - Features containing empty strings (i.e. distinct from NULL/NaN values)
  - Potentially-duplicated features
- Provides memory usage information when verbose=True
- Set `verbose=False` for condensed output focusing on basic statistics

`find_outliers(df, method='iqr', threshold=None, verbose=True)`
- Detects outliers in numerical features using statistical methods
- Supports multiple detection approaches:
  - *IQR-based* detection (default, using 1.5 × IQR)
  - *Z-score* method (using standard deviations from mean)
  - *Modified Z-score* method (robust to skewed distributions)
- Automatically handles method-specific threshold defaults
- Returns comprehensive dictionary with outlier information
- Provides feature-level and dataset-level outlier statistics
- Set `verbose=False` for minimal process output
- Set custom `threshold` values appropriate for your data

`find_corrs(df, method='pearson', threshold=0.8, verbose=True)`
- Identifies highly-correlated numerical feature pairs in a dataset
- Supports multiple correlation methods: Pearson (default), Spearman, and Kendall
- Automatically analyzes all numerical features in the DataFrame
- Returns a comprehensive dictionary with correlation information and summary statistics
- Set custom `threshold` parameter to adjust sensitivity (higher values = fewer correlations)
- Use `method='spearman'` for non-linear relationships or data with outliers
- Use `method='kendall'` for small sample sizes or when handling many tied ranks
- Set `verbose=False` for minimal output when running in scripts or notebooks
- Provides counts of correlations per feature to identify the most redundant variables

`reshape(df, verbose=True)`
- Handles missing value deletion and feature deletion
- Returns reshaped DataFrame
- Set `verbose=False` for minimal process output

`subset(df, verbose=True)`
- Facilitates advanced subsetting of instances in a dataset
- Supports multiple subsetting methods:
  - *Unseeded* random sampling (for true randomness)
  - *Seeded* random sampling (for reproducibility)
  - Stratified random sampling (maintains category proportions)
  - Time-based subsetting for timeseries data
- Automatically detects datetime columns and indices
- Returns subset DataFrame
- Set `verbose=False` for minimal process output

`rename_and_tag(df, verbose=True, tag_features=False)`
- Facilitates feature renaming with comprehensive validation:
  - Validates Python identifier compliance
  - Checks for problematic characters
  - Warns about potentially-problematic naming anti-patterns
  - Prevents Python keyword conflicts
- Optionally tags features as ordinal or target features when `tag_features=True`
- Tracks and summarizes all naming operations when `verbose=True`
- Returns DataFrame with new, validated feature names
- Set `verbose=False` for streamlined interaction without warnings and confirmation checks

`feature_stats(df, verbose=True, summary_stats=False)`
- Analyzes features by type (boolean, datetime, categorical, numerical)
- Calculates and displays key statistics for each feature type
  - For categorical features: shows unique values, mode, entropy
  - For numerical features: shows distribution statistics, skewness, kurtosis
  - For datetime features: shows date range and time granularity
  - For boolean features: shows true/false counts and percentages
- Set `verbose=False` for minimized output

`plot_features(df, some_parameters=value)`
- Gabor will be authoring this method, which provides appropriate plots for dataset features

`impute(df, verbose=True, skip_warnings=False)`
- Handles missing value imputation using various methods:
  - Statistical methods: mean, median, mode
  - Constant value imputation
  - Random sampling from non-null values
  - Forward/backward fill for time series data
- Automatically detects time series data and offers appropriate imputation methods
- Performs extensive data quality checks including:
  - Low variance detection
  - Outlier detection
  - Correlation analysis
  - Skewness assessment
- Provides visual distribution comparisons pre/post imputation
- Returns DataFrame with imputed values
- Set `skip_warnings=True` to bypass data quality warnings
- Set `verbose=False` for streamlined interaction

`encode(df, features_to_encode=None, verbose=True, skip_warnings=False, preserve_features=False)`
- Encodes categorical features using One-Hot or Dummy encoding
- Can be passed a list of specific features to encode via the `features_to_encode` parameter
- Auto-detects categorical features if `features_to_encode` is None
- Returns DataFrame with encoded features
- Set `skip_warnings=True` to bypass cardinality and null value warnings
- Set `preserve_features=True` to create new columns with encoded values instead of replacing original features
- Properly sanitizes column names with special characters or spaces
- Allows customization of column prefixes in verbose mode

`scale(df, features_to_scale=None, verbose=True, skip_warnings=False, preserve_features=False)`
- Scales numerical features using Standard, Robust, or MinMax scaling
- Can be passed a list of specific features to scale via the `features_to_scale` parameter
- Auto-detects numerical features if `features_to_scale` is None
- Returns DataFrame with scaled features
- Set `skip_warnings=True` to bypass distribution and outlier warnings
- Set `preserve_features=True` to create new columns with scaled values instead of replacing original features
- Supports custom range definition for MinMax scaling
- Provides interactive handling of infinite values with multiple replacement strategies
- Offers visualization comparisons of pre- and post-scaling distributions

`transform(df, features_to_transform=None, verbose=True, preserve_features=False, skip_warnings=False)`
- Applies mathematical transformations to numerical features to improve their distributions
- Supports multiple transformation methods:
  - Log transformation (natural log, log10)
  - Square root transformation
  - Box-Cox transformation
  - Yeo-Johnson transformation
  - Power transformations (squared, cubed)
  - Reciprocal transformation
- Automatically suggests transformations based on feature distribution characteristics
- Handles special cases like zeros and negative values appropriately
- Set `features_to_transform` to specify which features to transform
- Set `preserve_features=True` to keep original features alongside transformed versions
- Set `verbose=False` for minimal process output
- Set `skip_warnings=True` to bypass distribution anomaly warnings

`extract_datetime(df, datetime_features=None, verbose=True, preserve_features=False, components=None)`
- Extracts useful features from datetime columns to enable ML algorithms to capture temporal patterns
- Creates new features representing various datetime components:
  - Year, month, day, day of week, hour, minute
  - Quarter, week of year, day of year
  - Is weekend, is month start/end, is quarter start/end
- Supports cyclical encoding of periodic features using sin/cos transformations
- Automatically detects datetime columns or attempts to parse string columns as dates
- Set `datetime_features` to specify which columns to process
- Set `components` to select specific datetime components to extract
- Set `preserve_features=True` to keep original datetime columns
- Set `verbose=False` for minimal process output

`build_interactions(df, features_to_combine=None, interaction_types=None, verbose=True, preserve_features=True, 
max_features=None)`
- Creates new features by combining existing features through mathematical operations
- Enables linear models to capture non-linear relationships and feature interactions
- Supports multiple interaction types:
  - Multiplication (pairwise products)
  - Division (with safeguards against division by zero)
  - Addition and subtraction
  - Polynomial transformations (squared, cubed features)
- Intelligently names new features based on source features and operations
- Set `features_to_combine` to specify which features to consider for interactions
- Set `interaction_types` to control which types of interactions to create
- Set `max_features` to limit the number of interaction features generated
- Set `verbose=False` for minimal process output
- Set `preserve_features=False` to remove original features after interaction creation

`make_plots(df, features_to_plot=None)`
- Creates visualizations for DataFrame features with an interactive approach
- Automatically identifies feature types and suggests appropriate visualizations
- Supports single-feature plots (histograms, box plots, bar charts, pie charts)
- Supports multi-feature plots (scatter plots, line plots, heat maps)
- Handles mixed feature types intelligently (numerical, categorical, datetime)
- Provides guidance about plot types and their appropriate use cases
- Allows color differentiation (hue) for categorical variables
- Set `features_to_plot` to specify which columns to visualize
- Maximum of 3 features can be selected for a single visualization
- Users can create multiple plots in sequence during a single call to the method

#### Notes on Method Usage

- All methods include interactive prompts to guide users through the preparation process
- Methods with `verbose` parameters allow control over output detail level
- Each method can be used independently or as part of the full pipeline via `prep_df()`
- When used without re-assignment, the methods preserve the original DataFrame and return a new dataframe with 
the applied transformations. However, the user must take care that potential overwrites are handled correctly:
```python
# Prevents overwrite of original dataframe by returning new dataframe object
df_reshaped = tp.reshape(df, verbose=False)

# Overwrites original dataframe via re-assignment
df = tp.reshape(df, verbose=False)
```

### Provisos and Limitations
TADPREP is designed to be a practical, first-principles library for common tabular data preprocessing tasks. 
While it provides interactive guidance throughout the data preparation process, it intentionally implements only 
fundamental techniques for certain operations:

#### Imputation
- Supports common imputation methods:
  - Statistical: mean, median, mode
  - Constant value imputation
  - Random sampling from non-null values
  - Forward/backward fill for time series data
- Includes data quality checks and distribution visualization capabilities
- Does not implement more sophisticated imputation approaches such as:
  - Multiple imputation
  - K-Nearest Neighbors imputation
  - Regression-based imputation
  - Machine learning models for imputation

#### Encoding
- Limited to One-Hot and Dummy encoding for categorical features
- Does not implement advanced encoding techniques such as:
  - Target encoding
  - Weight of Evidence encoding
  - Feature hashing
  - Embedding techniques

#### Scaling
- Implements three common scaling methods: Standard (Z-score), Robust, and MinMax scaling
- Does not include more specialized techniques like:
  - Quantile transformation
  - Power transformations (e.g., Box-Cox, Yeo-Johnson)
  - Custom scaling methods

#### Intended Use

TADPREP is most effectively used as part of a larger data preparation workflow.
1. Use TADPREP for initial data exploration and basic preparation, such as:
   - Understanding feature distributions and relationships
   - Identifying and handling missing values
   - Basic feature renaming and organization
   - Simple transformations using supported methods

2. For more complex operations:
   - Use TADPREP to prepare and clean your data
   - Export a partially prepared dataset
   - Apply your own custom transformations using specialized libraries like scikit-learn, feature-engine, 
   or category_encoders

Using the library in a deliberate, as-needed manner allows you to leverage TADPREP's interactive guidance for basic 
tasks while maintaining the flexibility to implement more sophisticated methods as required for your specific use case.

For example, if you want to use advanced imputation techniques:
```python
# 1. Use TADPREP for initial preparation
import pandas as pd
import tadprep as tp
df_initial = tp.prep_df(df)  # Run data-cleaning pipeline, skipping imputation

# 2. Import and apply custom imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df_initial), 
                          columns=df_initial.columns)
```

### Non-Native Dependencies
- numpy *(For array operations and infinity checks)*
- pandas *(Used extensively for dataframe manipulations)*
- scikit-learn *(Supplies scalers for numerical features)*
- matplotlib *(For basic plotting)*
- seaborn *(For generating feature distribution plots)*

### Future Development Ideas/Questions
- Are we sure the "tagging" feature in `rename_and_tag` is actually useful?
- Should we build some kind of assumption-testing method, like whether linear models are appropriate, etc.?
- Are there core capabilities missing from the library? Are we failing to notice a process-need in tabular data prep?
- What procedural shape and UX should the `prep_df` method have?
- How are we going to re-conceptualize/reconfigure the CLI pipeline?
- Testing structure - what's the best programmatic way to test the methods?

### Acknowledgments
- Dr. Sean Connin at the Ritchie School of Engineering for his general encouragement and package design advice.
- Thomas Mooney at the McDonnell Genome Institute for his UX testing and code review.