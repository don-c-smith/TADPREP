# Public-Facing Method Development Planning

## Method: `df_info`
### Core Purpose:
Prints summary, top-level information about a dataframe to the console.

### Parameters:
- `df` Input Pandas dataframe.
- `verbose` (Boolean, default = True) which controls level of detail in output.

### Returns:
- None - void method. Prints info to console.

### Current State:
- If `verbose=False`:
  - Prints number of instances (rows) in df
  - Prints number of features (columns) in df
  - Prints total number of instances with *any* missing values
  - Prints % of instances with *any* missing values as a proportion of total instances
  - Prints count of duplicate instances


- If `verbose=True`, the method **also** prints:
  - A warning to assess whether the existence of duplicate instances indicates a data error has occurred
  - A list of near-constant features (i.e. those with >= 95% matched values)
  - A list of features containing any infinite values (i.e. np.inf values)
  - A list of object-type features containing *empty* strings (i.e. distinct from NULL/NaN values)
  - Names and datatypes of all features, along with memory use, etc. *(This is a call to Pandas' .info() method)*
  - A line of dashes on either side of the .info() output for visual separation

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- None as of current state

### Method History
 - Alpha build by Don Smith
 - Beta build by Don Smith (Current State)


## Method: `find_outliers`

### Core Purpose:
Provides functionality for detecting outliers in numerical features using various statistical detection methods, 
with comprehensive information about outliers at both the feature and dataset level.

### Parameters:
- `df` Input Pandas dataframe.
- `method` (String, default = 'iqr') which specifies the outlier detection method to use.
- `threshold` (Float, default = None) which controls the sensitivity of outlier detection, with method-specific 
defaults.
- `verbose` (Boolean, default = True) which controls level of detail in output.

### Returns:
- Dictionary containing detailed outlier information, including summary statistics and feature-specific results.

### Current State:
- Core Functionality (Always Run):
  - Identifies all numerical features for outlier analysis
  - Supports three detection methods:
    - IQR-based detection (default)
    - Z-score (standard deviations from mean)
    - Modified Z-score (robust to non-normal distributions)
  - Uses appropriate method-specific default thresholds
  - Handles edge cases such as:
    - Features with no variance
    - Features with all null values
    - Features with zero standard deviation or MAD
  - Tracks outliers on both feature and instance level
  - Calculates comprehensive statistics including:
    - Total outlier count
    - Affected rows count and percentage
    - Features with outliers
    - Feature-specific outlier metrics
  - Provides thresholds used for detection


- If `verbose=False`:
  - Performs outlier detection silently
  - Returns results dictionary without printing
  - No progress information or statistics displayed


- If `verbose=True`, the method **also** provides:
  - Information about the detection method used
  - Threshold values and their meaning
  - Feature-by-feature outlier analysis
  - Warnings about skipped features and why
  - Examples of extreme outlier values (both high and low)
  - Detailed summary statistics
  - Percentage of rows affected by outliers
  - Feature-specific outlier counts and percentages

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- Possible additional functionality:
  - Option to return a DataFrame with outliers masked or flagged
  - Support for automated outlier removal or replacement
  - Visual representations of outliers (boxplots, histograms)
  - Multivariate outlier detection methods
  - Local Outlier Factor (LOF) method support

### Method History:
- Alpha build by Don Smith (Current State)

## Method: `find_corrs`
### Core Purpose:
Identifies and returns information about highly-correlated feature pairs in a dataframe.

### Parameters:
- `df` Input Pandas dataframe.
- `method` (String, default = 'pearson') Correlation method to use. Options: 'pearson', 'spearman', 'kendall'.
- `threshold` (Float, default = 0.8) Correlation coefficient threshold (absolute value) above which features are 
considered highly correlated.
- `verbose` (Boolean, default = True) Controls whether detailed correlation information is printed to console.

### Returns:
- Dictionary containing correlation information with the following structure:
  - `summary`: Dictionary with correlation statistics
    - `method`: Correlation method used
    - `num_correlated_pairs`: Total count of highly correlated pairs
    - `max_correlation`: Maximum correlation coefficient found
    - `avg_correlation`: Average correlation among high pairs
    - `features_involved`: List of features involved in high correlations
  - `correlation_pairs`: List of dictionaries, each containing:
    - `feature1`: Name of first feature in pair
    - `feature2`: Name of second feature in pair
    - `correlation`: Actual correlation coefficient (can be negative)
    - `abs_correlation`: Absolute value of correlation coefficient

### Current State:
- Automatically identifies numerical features in the provided dataframe
- Calculates correlations between all numerical feature pairs using specified method
- Identifies feature pairs with correlation coefficients exceeding the threshold
- Returns comprehensive dictionary with correlation information
- When `verbose=True`:
  - Prints analysis information including feature count and threshold used
  - Prints each highly-correlated feature pair with its correlation sign and value
  - Displays maximum and average correlation values found
  - Lists all features involved in high correlations and their occurrence count
  - Uses visual separators for readability

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- Add visualization option to generate a filtered correlation heatmap
- Add capability to analyze relationship types (linear vs. non-linear)

### Method History
- Alpha build by Don Smith (Current State)


## Method: `reshape`
### Core Purpose:
Perform row- or column-dependent instance (row) removals, as well as feature (column) removals.

### Parameters:
- `df` Input Pandas dataframe.
- `features_to_reshape` (list[str]) 
- `verbose` (Boolean, default = True) Controls level of detail in output.

### Returns:
- Modified dataframe containing only the subset of instances not removed by user.

### Current State:
- Docstrings updated for all but drop_columns child func.

- ***UX-focused elements factored out of current build
  - A bit hardline on this for now, avoiding additional user input beyond confirmation of operations.
- Core Functionality:
  - Provides user 3 options for reshaping by removal:
    - 1. Row-dependent row removal
      - Provides user with a default missingness threshold
      - Sums, by row, total missing features
      - Encodes, by row, whether those rows' missingness is at/above threshold
    - 2. Column-dependent row removal
      - Displays row missingness by features provided
      - Removes rows with missingness in 'features_to_reshape'
    - 3. Column removal
      - This might not need to exist outside of UX
      - Would likely just be a pandas wrapper

- "Verbosity" differentiation not fully implemented
  - I'm leaning toward this as a UX-side feature of the package
  - Full method functionality is a bit dense with all of the "possible" info the user is getting or not.

### Observed Bugs/Problems:
- testing of current iteration incomplete
  - bugs not detected so far

### Ideas for Development:
- Data types considered "missing" could be subject to user interpretation
  - Consider implementing options for these
- "Architecture" of package as related to this method needs clarifying w/ Don
  - What functionality should go in `transforms.py` vs `tadprep_interactive.py`

### Method History:
- Alpha build by Don Smith
- Beta build by Gabor Horvath (Current State)


## Class: `PlotHandler`
### Core Purpose:
Produce, store, and compare relevant, straightforward visualizations on a per-feature basis as guided by user.

### Methods:
- `.plot(self, df, col_name, plot_type):`
  -  Generates and stores a Seaborn plot for a specified pandas DataFrame column with plot type determined by user input.
- `.det_plot_type(self, df, col_name):`
  - Determines an appropriate plot type for a set of data (pd.Series) based on the pandas dtype of the user's DataFrame column.
- `.recall_plot(self, col_name, plot_type):`
  - Fetches data for most recently-created plot of provided type and redraws plot with it.
- `compare_plots(self, col_name):`
  - Creates a plt.subplots() figure of appropriate dimension to display all plots of all types for a specified DataFrame column

### Returns:
- None - is Class

### Current State:
- Alpha build
- Colorblind color palette active
- Data "snapshot" storage system implemented
  - Relies on storing a pd.Series of the data used for a given viz when that viz is created
  - Plot "recall/redraw" and basic comparative viz functionality implemented for histplots
    - plt.subplots() implementation with control-flow for proper axes object positioning and labeling

### Observed Bugs/Problems:
- `__init__.py` and `tadprep` import functionality has issues recognizing the `PlotHandler` class
  - Have to specify the following to enable instantiation of PlotHandler() class objects for testing:
    - 'from tadprep.core.tansforms import PlotHandler'
    - Unsure if this is expected behavior or if is issue with file structure

- `compare_plots` method failing to populate first column (histograms) in 2D case


### Ideas for Development:
- Testing will help indicate whether we should refactor `.det_plot_type()` and `_rename_and_tag_core` for more effective plot type determination.

- As color-blind friendly as we are capable of!
- Static viz only, stores and recalls "snapshots" of data for plotting

### Method History:
- Alpha build by Gabor Horvath

## Method: `build_interactions`
### Core Purpose:
Creates new features by combining existing ones through mathematical operations, enabling linear models to capture non-linear relationships between variables.

### Parameters:
- `df` Input Pandas dataframe.
- `features_to_combine` (list[str] | None, default=None) Optional list of features to consider for interactions.
- `interaction_types` (list[str] | None, default=None) Optional list of interaction types to create 
(e.g., 'multiply', 'divide', 'add', 'subtract', 'polynomial').
- `verbose` (Boolean, default=True) Controls level of detail/guidance in output.
- `preserve_features` (Boolean, default=True) Controls whether original features are preserved alongside interactions.
- `max_features` (int | None, default=None) Optional maximum number of interaction features to create.

### Returns:
- Modified dataframe containing newly-created interaction features and (possibly not) original features-to-interact.

### Current State:
- Basic mathematical operations implemented
- "Complex" mathematical operations implemented
- Data validation not implemented
- User-directed division order not implemented
- In-method feature selection functionality not implemented
- Verbosity not implemented
- Warnings for feature explosion and cancellation not implemented

### Observed Bugs/Problems:
- To-be-tested

### Ideas for development:
- Determine possibilities for additional interaction types beyond "polynomial"
- Possible "two paradigms" for how this method could be applied
  1. Broad-stroke "exploratory" interaction creation
    - Meant to create larger number of interactions for attempting novel data analysis
  2. Low-count "targeted" interaction creation
    - Meant for creation of specific interaction terms based on domain foreknowledge

### Method History:
- Pre-alpha build by Gabor Horvath

## Method: `subset`
### Core Purpose:
Provides interactive functionality for subsetting data through random sampling (with or without seed), 
stratified sampling, or time-based instance selection for timeseries data.

### Parameters:
- `df` Input Pandas dataframe.
- `verbose` (Boolean, default = True) which controls level of detail/guidance in output.

### Returns:
- Modified dataframe containing only the subset of instances specified by user.

### Current State:
- Core Functionality (Always Run):
  - Detects categorical features for stratified sampling
  - Identifies datetime features and indices for time-based subsetting
  - Automatically converts string datetime columns to datetime type
  - Supports multiple subsetting methods:
    - True random sampling (unseeded)
    - Reproducible random sampling (seeded)
    - Stratified random sampling (if categorical features are present)
    - Time-based subsetting (if datetime elements are present)
  - Validates all user inputs and subset parameters
  - Ensures minimum instance counts are maintained
  - Preserves data integrity during subsetting


- If `verbose=False`:
  - Shows only available subsetting methods
  - Displays minimal progress information
  - Presents only essential user prompts
  - Shows basic confirmation of successful operations
  - Reports final instance count


- If `verbose=True`, the method **also** provides:
  - Detailed explanation of available subsetting methods
  - Comprehensive feature type identification
  - For time series data:
    - Data frequency analysis
    - Time span information
    - Example timestamps
    - Guidance on date format inputs
  - For stratified sampling:
    - Detailed explanation of the process
    - Category distribution information
    - Guidance on maintaining representativeness
  - Detailed summary of subsetting results including:
    - Original instance count
    - Subset instance count
    - Subsetting method used
    - Additional method-specific details:
      - Time boundaries (for time-based subsetting)
      - Category proportions (for stratified sampling)

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- Possible additional subset/sampling methodologies:
  - Option for weighted random sampling
  - Option for bootstrap sampling

### Method History:
- Alpha build by Don Smith (Current State)


## Method: `rename_and_tag`
### Core Purpose:
Provides interactive functionality for renaming features and optionally tagging them as ordinal or target features 
by appending appropriate suffixes ('_ord' or '_target').

### Parameters:
- `df` Input Pandas dataframe.
- `verbose` (Boolean, default = True) which controls level of detail/guidance in output.
- `tag_features` (Boolean, default = False) which controls whether the feature-tagging process runs.

### Returns:
- Modified dataframe with renamed and/or tagged features.

### Current State:
- Core Functionality (Always Run):
  - Displays enumerated list of current features
  - Validates all feature names and indices
  - Ensures new feature names don't conflict with existing names
  - Provides cancel-out options at each stage
  - Maintains data integrity during renaming
  - Performs extensive feature name validation:
    - Checks for valid Python identifiers
    - Identifies problematic characters
    - Helps avoid undesired anti-patterns
    - Validates against Python keywords
    - Prevents duplicate feature names
  - Tracks all renaming operations for reporting
  - Provides operation cancellation options at each stage


- Parameter `tag_features` Controls Tagging Process Activation:
  - When `tag_features=False` (Default):
    - Only feature renaming functionality is available
    - Feature tagging stages are completely skipped
    - Process completes after renaming stage


  - When `tag_features=True`:
    - Enables complete feature tagging workflow
    - Adds ordinal feature tagging stage ('_ord' suffix)
    - Adds target feature tagging stage ('_target' suffix)
    - Validates that features aren't already tagged
    - Prevents duplicate tagging of features
    - All stages (rename, ordinal tag, target tag) can be skipped individually


- If `verbose=False`:
  - Shows only basic feature list
  - Displays minimal progress information
  - Presents only essential user prompts
  - Shows basic confirmation of successful operations


- If `verbose=True` (Default), the method **also** provides:
  - Clear stage demarcation with visual separators
  - Process initiation notifications
  - Detailed confirmation of each rename operation
  - Warning messages for problematic feature names
  - Preview and confirmation of each name change
  - For feature tagging:
    - Explanatory messages about tagging process
    - Clear separation between ordinal and target tagging stages
  - Comprehensive summary of all changes made including:
    - Detailed summary table of all operations
    - Changes grouped by operation type
    - Feature renames with before/after values
    - List of tagged features by tag type
    - Clear notation of process completion

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- **Foundational Question:** Is the tagging functionality useful? Is it out of scope for a method that's mostly about 
feature renaming? What useful data science purpose does it serve?

### Method History:
- Alpha build by Don Smith
- Beta build by Don Smith (Current State)


## Method: `feature_stats`

### Core Purpose:
Displays feature-level statistics and information for each feature in a dataframe, categorizing features by type 
and providing appropriate descriptive statistics based on feature type.

### Parameters:
- `df` Input Pandas dataframe.
- `verbose` (Boolean, default = True) which controls level of detail in output.

### Returns:
- None. This is a void method that prints information to the console.

### Current State:
- Core Functionality (Always Run):
  - Separates features into categories by type:
    - Boolean features (logical values or 0/1 integers)
    - Datetime features
    - Categorical features
    - Numerical features
  - For all features:
    - Shows missingness information (count and percentage)
  - For boolean features:
    - Shows true/false value counts and percentages
  - For datetime features:
    - Shows date range information
  - For categorical features:
    - Shows unique value counts
    - Shows mode values
    - Shows category distributions
  - For numerical features:
    - Shows mean
    - Shows range (min/max)
    - Shows basic descriptive statistics
  - Performs data quality checks:
    - Zero-variance features
    - Near-constant features (>95% single value)
    - Potential duplicate features


- If `verbose=False`:
  - Shows only feature names and types
  - Shows basic statistics without formatting
  - Shows minimal explanatory text
  - Presents condensed output


- If `verbose=True`, the method **also** provides:
  - Detailed feature type categorization and distribution
  - Data quality alerts for potential issues
  - Extended statistics:
    - For categorical: entropy values, top frequency ratios, distribution patterns
    - For numerical: quartile information, skewness, kurtosis, coefficient of variation
  - Formatted output with visual separators
  - Contextual interpretation of statistical measures

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- None as of current state

### Method History:
- Alpha build by Don Smith
- Beta build by Don Smith (Current State)


## Method: `plot_features`

### Core Purpose:
Displays feature-level plots of appropriate types/formats.

Gabor is writing the Alpha for this. He has some neat OOP-related ideas on how to do it.

## Method: `impute`
### Core Purpose:
Works interactively with user to impute missing values in features using common imputation methods.

### Parameters:
- `df` Input Pandas dataframe.
- `verbose` (Boolean, default = True) which controls level of detail/guidance in output.
- `skip_warnings` (Boolean, default = False) which controls whether to skip data quality and missingness warnings.

### Returns:
- Modified dataframe with imputed values as specified by user.

### Current State:
- Core Functionality (Always Run):
  - Detects and validates datetime features to identify time series data
  - Checks for "false numeric" features (e.g., 1/0 representations of Yes/No)
  - Performs data quality checks including:
    - Near-zero variance features
    - Features with high outlier counts
    - Highly correlated feature pairs
    - Features with extreme skewness
  - Supports multiple imputation methods:
    - Statistical methods (mean, median, mode)
    - Constant value imputation
    - Random sampling from non-null values
    - Forward/backward fill (for time series data)
  - Tracks imputation actions for summary reporting


- If `verbose=False`:
  - Shows minimal feature type classification
  - Displays only essential user prompts
  - Presents basic imputation choices
  - Shows only critical warnings
  - Provides basic confirmation of successful operations


- If `verbose=True`, additionally provides:
  - Detailed initial and final feature classifications
  - Comprehensive missingness statistics
  - Detailed explanations of imputation methods
  - Pre-imputation feature distributions and statistics
  - Visual distribution plots for numerical features
  - Step-by-step guidance through the imputation process
  - Post-imputation distribution comparisons
  - Comprehensive imputation summary including:
    - Feature names
    - Number of values imputed
    - Methods used
    - Imputation values or approaches


- If `skip_warnings=False`, additionally checks and warns about:
  - Features with high missingness rates (>10%)
  - Data quality issues including:
    - Near-zero variance
    - High outlier counts
    - High correlations between features
    - Extreme skewness
  - Provides detailed guidance on handling problematic features
  - Allows user to 'proceed with caution' after each warning

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- None as of current state

### Method History:
- Alpha build by Don Smith
- Beta build by Don Smith (Current State)


## Method: `encode`
### Core Purpose:
Works interactively with user to encode categorical features using standard encoding methods.

### Parameters:
- `df` Input Pandas dataframe.
- `features_to_encode` (list[str] | None, default=None) Optional list of features to encode.
- `verbose` (Boolean, default = True) Controls level of detail/guidance in output.
- `skip_warnings` (Boolean, default = False) Controls whether to skip data quality warnings.
- `preserve_features` (Boolean, default = False) Controls whether original features are kept alongside encoded ones.

### Returns:
- Modified dataframe with encoded categorical features as specified by user.

### Current State:
- Core Functionality (Always Run):
 - Validates input parameters and feature existence
 - Identifies categorical features when none specified
 - Detects numeric features that might be categorical (e.g., 1/0 values)
 - Supports two encoding methods:
   - One-Hot Encoding (creates column for each category)
   - Dummy Encoding (creates n-1 columns)
 - Sanitizes column names to ensure valid Python identifiers
 - Properly handles special characters in feature names
 - Concatenates all encoded features into the dataframe
 - Tracks encoding operations for summary reporting
 - Allows preserving original features alongside encoded ones


- If `verbose=False`:
 - Shows minimal feature guidance
 - Displays only essential user prompts
 - Presents basic encoding choices
 - Shows only critical warnings
 - Provides basic confirmation of successful operations


- If `verbose=True`, additionally provides:
 - Comprehensive feature category information
 - Detailed explanations of encoding methods
 - Value distributions for categorical features
 - Visual distribution plots for features
 - Custom prefix options for encoded columns
 - Step-by-step guidance through encoding decisions
 - Comprehensive encoding summary including:
   - Feature names
   - Encoding methods applied


- If `skip_warnings=False`, additionally checks and warns about:
 - Features containing null values
 - Features with high cardinality (>20 unique values)
 - Features with low-frequency categories (<10 instances)
 - Provides detailed guidance on handling problematic features
 - Allows user to 'proceed with caution' after each warning

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- None as of current state

### Method History:
- Alpha build by Don Smith
- Beta build by Don Smith (Current state)


## Method: `scale`
### Core Purpose:
Provides interactive functionality for scaling numerical features in a dataset using standard statistical methods, 
enabling proper normalization of data for machine learning algorithms that are sensitive to feature magnitudes.

### Parameters:
- `df` Input Pandas dataframe.
- `features_to_scale` (List[str] | None, default = None) Optional specific features to scale. If None, method 
identifies numerical features interactively.
- `verbose` (Boolean, default = True) Controls level of detail/guidance in output.
- `skip_warnings` (Boolean, default = False) Controls whether data quality warnings (null values, outliers, skewness) 
are displayed.
- `preserve_features` (Boolean, default = False) Controls whether original features are preserved. When True, creates 
new columns with scaled values instead of replacing originals.

### Returns:
- Modified dataframe with scaled numerical features. If `preserve_features=True`, original features are retained and 
new columns with scaled values are added.

### Current State:
- Core Functionality (Always Run):
  - Identifies numerical features that are appropriate for scaling
  - Validates all selected features exist in the dataframe
  - Distinguishes between true numerical features and categorical features encoded as numbers
  - Prevents scaling of constant features (no variance)
  - Offers three scaling methods appropriate for different data characteristics:
    - Standard Scaler (Z-score normalization)
    - Robust Scaler (based on median and IQR)
    - MinMax Scaler (supports custom range specification)
  - Handles infinite values with multiple replacement strategies
  - Provides feature-by-feature scaling selection
  - Maintains data integrity during scaling operations
  - Supports individual feature skipping within the process flow
  - Tracks all scaling operations for final reporting
  - Returns modified dataframe with scaled features
  - Offers side-by-side visualization of pre- and post-scaling distributions
  - Supports preserving original features by creating new scaled columns


- Parameter `features_to_scale` Controls Feature Selection:
  - When `features_to_scale=None` (Default):
    - Automatically identifies numerical features in the dataset
    - Checks for potentially miscategorized numerical features (e.g., numeric encoding of categories)
    - Interactively determines which features to include in scaling
    - Allows users to exclude pseudo-categorical features

  - When `features_to_scale=[list of features]`:
    - Scales only the specific features in the provided list
    - Validates all requested features exist in the dataframe
    - Skip automatic feature type detection


- Parameter `preserve_features` Controls Output Columns:
  - When `preserve_features=False` (Default):
    - Replaces original feature values with scaled values
    - Original data is not retained in the returned dataframe

  - When `preserve_features=True`:
    - Creates new columns with naming pattern '{original_column}_scaled' 
    - If column name conflicts exist, adds numeric suffixes (e.g., '{original_column}_scaled_1')
    - Preserves original data while adding scaled versions
    - Both original and scaled versions available in returned dataframe


- If `skip_warnings=False` (Default):
  - Checks for null values in features before scaling
  - Identifies infinite values that may disrupt scaling
  - Detects extreme skewness that might require transformation before scaling
  - Requires user confirmation to proceed when issues are found


- If `skip_warnings=True`:
  - Bypasses data quality checks for nulls, infinites, and skewness
  - Proceeds directly with scaling operations without warnings
  - May be preferred for experienced users confident in their data quality


- If `verbose=False`:
  - Shows only basic feature list
  - Displays minimal progress information
  - Presents only essential user prompts
  - Shows basic confirmation of successful operations


- If `verbose=True` (Default), the method **also** provides:
  - Clear stage demarcation with visual separators
  - Process initiation notifications
  - Detailed explanations of scaling methods and their appropriate use cases
  - Displays pre-scaling statistics and distributions for each feature
  - Offers visualization of feature distributions
  - Educational content about scaling methods and when each is appropriate
  - Reminders about not scaling target features
  - Comprehensive final summary of all scaling operations performed
  - Groups scaled features by scaling method used

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- None as of current state

### Method History:
- Alpha build by Don Smith
- Beta build by Don Smith (Current state)


# Method: `transform`

### Core Purpose:
Applies mathematical transformations to numerical features to improve their distributions for modeling, with a 
focus on normalization and linearization.

### Parameters:
- `df` Input Pandas dataframe.
- `features_to_transform` (list[str] | None, default=None) Optional list of features to transform.
- `verbose` (Boolean, default=True) Controls level of detail/guidance in output.
- `preserve_features` (Boolean, default=False) Controls whether original features are preserved.
- `skip_warnings` (Boolean, default=False) Controls whether to skip distribution and outlier warnings.

### Returns:
- Modified dataframe with transformed features. If `preserve_features=True`, original features are retained.

### Implementation Plan:
- **Input Validation**
  - Verify input is a Pandas DataFrame
  - Validate existence of specified features if provided
  - Ensure DataFrame is not empty

- **Feature Identification**
  - If `features_to_transform` is None:
    - Identify numerical features in DataFrame
    - Filter out Boolean/binary features (0/1 values)
    - Allow user to select which features to transform

- **Feature Analysis**
  - For each feature to transform:
    - Calculate descriptive statistics (mean, median, min, max)
    - Measure skewness and kurtosis
    - Check for infinities and nulls
    - If `verbose=True`, show distribution plots
    - Suggest appropriate transformations based on statistics

- **Transformation Options**
  - Implement the following transformations:
    - Log transformation (natural log, log10)
    - Square root transformation
    - Box-Cox transformation
    - Yeo-Johnson transformation
    - Power transformations (squared, cubed)
    - Reciprocal transformation

- **Transformation Process**
  - For each selected feature:
    - Present transformation options based on data characteristics
    - Handle special cases (zeros for log transforms, negative values)
    - Apply selected transformation
    - If `verbose=True`, show before/after plots
    - Track transformation details for reporting

- **Output Handling**
  - If `preserve_features=True`:
    - Create new columns with naming pattern '{feature}_transformed'
  - If `preserve_features=False`:
    - Replace original features with transformed versions
  - Return updated DataFrame

- **Error Handling**
  - Handle invalid transformations (e.g., log of negative values)
  - Provide informative error messages
  - Offer fallback options when primary transformation fails
  - Allow skipping problematic features

- **Reporting**
  - If `verbose=True`:
    - Summarize transformations applied
    - Report improvement in normality metrics
    - Show before/after summary statistics

### Expected Behavior:
- Core Functionality (Always Run):
  - Analyzes numerical features for distribution characteristics
  - Provides appropriate transformation options based on data properties
  - Supports multiple transformation methods for different distribution types
  - Handles edge cases like zeros and negative values
  - Properly names and organizes transformed features
  - Maintains data integrity during transformation

- If `verbose=False`:
  - Shows minimal feature guidance
  - Displays only essential user prompts
  - Presents basic transformation choices
  - Shows only critical warnings
  - Provides basic confirmation of successful operations

- If `verbose=True`:
  - Shows detailed feature distribution analysis
  - Provides visualization of before/after distributions
  - Explains reasoning behind suggested transformations
  - Displays comprehensive normality metrics
  - Offers educational content about transformation methods

- If `skip_warnings=False`:
  - Provides warnings about skewed distributions
  - Alerts about potential issues with transformations
  - Suggests alternative approaches for problematic features

### Method History:
- Alpha build by Don Smith (Current State)


# Method: `extract_datetime`

### Core Purpose:
Extracts useful features from datetime columns in a dataframe, converting temporal information into features that 
machine learning models can use more effectively.

### Parameters:
- `df` Input Pandas dataframe.
- `datetime_features` (list[str] | None, default=None) Optional list of datetime features to process.
- `verbose` (Boolean, default=True) Controls level of detail/guidance in output.
- `preserve_features` (Boolean, default=False) Controls whether original datetime features are preserved.
- `components` (list[str] | None, default=None) Optional list of specific datetime components to extract.

### Returns:
- Modified dataframe with extracted datetime features. If `preserve_features=True`, original datetime columns are 
retained.

### Implementation Plan:
- **Input Validation**
 - Verify input is a Pandas DataFrame
 - Validate existence of specified datetime features if provided
 - Ensure DataFrame is not empty

- **Feature Identification**
 - If `datetime_features` is None:
   - Identify columns with datetime dtype
   - Attempt to parse string columns as datetime
   - Allow user to select which columns to process

- **Component Identification**
 - If `components` is None:
   - Offer standard components: year, month, day, dayofweek, hour, minute, quarter, weekofyear, dayofyear
   - For time series data, offer additional components: is_weekend, is_month_start, is_month_end, is_quarter_start, 
 is_quarter_end
   - Allow user to select which components to extract

- **DateTime Extraction Process**
 - For each selected datetime feature:
   - Ensure column is in datetime format (convert if needed)
   - Extract selected components into new columns
   - Handle timezone information if present
   - Apply appropriate naming conventions for new columns
   - Encode cyclical features (like month, day of week) using sin/cos transformation if requested

- **Output Handling**
 - Create new columns with naming pattern '{original_column}_{component}'
 - If `preserve_features=False`:
   - Remove original datetime columns
 - Return updated DataFrame

- **Time Series Detection**
 - Detect if data appears to be time series based on:
   - Regular intervals between datetime values
   - Sorted datetime values
   - If identified as time series, suggest additional useful features
   - Offer lagged features or rolling statistics if appropriate

- **Error Handling**
 - Handle invalid datetime formats
 - Provide informative error messages
 - Offer parsing options for ambiguous formats (MM/DD vs DD/MM)
 - Allow skipping problematic features

- **Reporting**
 - If `verbose=True`:
   - Summarize extracted components
   - Show examples of original and extracted values
   - Provide guidance on using extracted features in models

### Expected Behavior:
- Core Functionality (Always Run):
 - Identifies datetime columns (explicit datetime types and string-based dates)
 - Extracts selected components into separate columns
 - Maintains data integrity during extraction
 - Applies appropriate naming conventions
 - Creates all specified datetime-derived features
 - Returns modified dataframe with new features

- If `verbose=False`:
 - Shows minimal feature guidance
 - Displays only essential user prompts
 - Provides basic confirmation of successful operations
 - Minimizes explanatory text

- If `verbose=True`:
 - Explains datetime feature extraction concepts
 - Provides component selection guidance
 - Shows examples of extracted values
 - Offers visualization of time-based patterns if appropriate
 - Explains cyclical encoding concepts for periodic features

- Parameter `preserve_features`:
 - When `True`, keeps original datetime columns alongside extracted features
 - When `False`, removes original datetime columns after extraction

### Method History:
- Alpha build by Don Smith (Current State)


# Method: `build_interactions`

### Core Purpose:
Creates new features by combining existing features through mathematical operations, enabling linear models to 
capture non-linear relationships and interactions between variables.

### Parameters:
- `df` Input Pandas dataframe.
- `features_to_combine` (list[str] | None, default=None) Optional list of features to consider for interactions.
- `interaction_types` (list[str] | None, default=None) Optional list of interaction types to create 
(e.g., 'multiply', 'divide', 'add', 'subtract', 'polynomial').
- `verbose` (Boolean, default=True) Controls level of detail/guidance in output.
- `preserve_features` (Boolean, default=True) Controls whether original features are preserved alongside interactions.
- `max_features` (int | None, default=None) Optional maximum number of interaction features to create.

### Returns:
- Modified dataframe with original features and newly created interaction features.

### Implementation Plan:
- **Input Validation**
  - Verify input is a Pandas DataFrame
  - Validate existence of specified features if provided
  - Ensure DataFrame is not empty
  - Validate interaction_types if provided


- **Feature Identification**
 - If `features_to_combine` is None:
   - Identify numerical features in DataFrame
   - Allow user to select which features to consider for interactions
   - Provide option to include categorical features (after encoding)


- **Interaction Type Selection**
 - If `interaction_types` is None:
   - Offer standard interaction types: multiply, divide, add, subtract, polynomial
   - Allow user to select which types to create
   - If `verbose=True`, explain each interaction type's purpose


- **Interaction Creation Process**
 - For multiplication interactions:
   - Create pairwise products between selected features
   - Name new features as '{feature1}_x_{feature2}'
 
 - For division interactions (with safeguards):
   - Create pairwise divisions between selected features
   - Handle division by zero with appropriate methods
   - Name new features as '{feature1}_div_{feature2}'
 
 - For addition interactions:
   - Create pairwise sums between selected features
   - Name new features as '{feature1}_plus_{feature2}'
 
 - For subtraction interactions:
   - Create pairwise differences between selected features
   - Name new features as '{feature1}_minus_{feature2}'
 
 - For polynomial features:
   - Create squared and cubed versions of individual features
   - Name new features as '{feature}_squared', '{feature}_cubed'


- **Feature Selection and Limitation**
  - If `max_features` is set:
    - Prioritize interactions based on correlation with target (if available)
    - Use variance or other metrics to select most relevant interactions
    - Limit total number of interaction features created


- **Output Handling**
  - Add created interaction features to dataframe
  - If `preserve_features=False`:
    - Remove original features used in interactions
  - Return updated DataFrame


- **Error Handling**
  - Handle division by zero or near-zero values
  - Manage potential feature explosion with many interaction combinations
  - Provide warnings about large feature space expansion
  - Allow cancellation if too many features would be created


- **Reporting**
 - If `verbose=True`:
   - Summarize created interaction features
   - Show examples of original and interaction values
   - Provide guidance on using interaction features in models
   - Warn about potential multicollinearity issues

### Expected Behavior:
- Core Functionality (Always Run):
  - Creates specified interaction features between selected columns
  - Applies appropriate naming conventions
  - Maintains data integrity during feature creation
  - Handles edge cases (division by zero, etc.)
  - Returns modified dataframe with new features


- If `verbose=False`:
  - Shows minimal feature guidance
  - Displays only essential user prompts
  - Provides basic confirmation of successful operations
  - Minimizes explanatory text


- If `verbose=True`:
  - Explains interaction feature concepts
  - Shows examples of created features
  - Provides guidance on feature selection
  - Warns about potential pitfalls (multicollinearity, feature explosion)
  - Offers educational content about when interactions are beneficial


- Parameter `preserve_features`:
  - When `True` (default), keeps original features alongside interactions
  - When `False`, removes original features after interaction creation

### Method History:
- Proposed by Don Smith (Current State)