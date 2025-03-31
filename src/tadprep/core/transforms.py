import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy import stats


def _df_info_core(df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Core function to print general top-level information about the full, unaltered datafile for the user.

    Args:
        df (pd.DataFrame): A Pandas dataframe containing the full, unaltered dataset.
        verbose (bool): Whether to print more detailed information about the file. Defaults to True.

    Returns:
        None. This is a void function.
    """
    # Print number of instances in file
    print(f'The dataset has {df.shape[0]} instances.')  # [0] is rows

    # Print number of features in file
    print(f'The dataset has {df.shape[1]} features.')  # [1] is columns

    # Handle instances with missing values
    row_nan_cnt = df.isnull().any(axis=1).sum()  # Compute count of instances with any missing values
    row_nan_rate = (row_nan_cnt / len(df) * 100).round(2)  # Compute rate
    print(f'{row_nan_cnt} instance(s) ({row_nan_rate}%) contain at least one missing value.')

    # Handle duplicate instances
    # NOTE: This check runs regardless of verbose status because it can help catch a data integrity error
    row_dup_cnt = df.duplicated().sum()  # Compute count of duplicate instances
    if row_dup_cnt > 0:  # If any such instances exist
        row_dup_rate = ((row_dup_cnt / len(df)) * 100)  # Compute duplication rate
        print(f'\nThe dataset contains {row_dup_cnt} duplicate instance(s). ({row_dup_rate:.2f}% of all instances)')
        if verbose:
            print('WARNING: If duplicate instances should not exist in your data, this may indicate an error in how '
                  'your data were collected or imported.')

    # Subsequent data checks only run if verbose is True
    if verbose:
        # Create a list of near-constant features, if any exist
        near_constant_cols = [(column,
                               value_counts.index[0],  # Index 0 is the most frequent value in the column
                               value_counts.iloc[0] * 100)
                              # For each column
                              for column in df.columns
                              # Check if most frequent value in column exceeds 95% of data present
                              if (value_counts := df[column].value_counts(normalize=True)).iloc[0] > 0.95]  # Walrus!

        if near_constant_cols:
            print(f'\nALERT: {len(near_constant_cols)} feature(s) has very low variance (>95% single-value):')
            for column, value, rate in near_constant_cols:
                print(f'- {column} (Dominant value: {value}, Dominance rate: {rate:.2f}%)')

        # Create a list of features containing infinite values, if any exist
        inf_cols = [(column, np.isinf(df[column]).sum())  # Column name and count of inf values
                    for column in df.select_dtypes(include=['int64', 'float64']).columns  # For each numeric column
                    if np.isinf(df[column]).sum() > 0]  # Append if the column contains any inf values

        if inf_cols:
            print(f'\nALERT: {len(inf_cols)} feature(s) contains infinite values:')
            for column, inf_count in inf_cols:
                print(f'- {column} contains {inf_count} infinite value(s)')

        # Create a list of features containing empty strings (i.e. distinct from NULL/NaN), if any exist
        empty_string_cols = [(column, (df[column] == '').sum())  # Column name and count of empty strings
                             for column in df.select_dtypes(include=['object']).columns  # For each string column
                             if (df[column] == '').sum() > 0]  # If column contains any empty strings

        if empty_string_cols:
            print(f'\nALERT: {len(empty_string_cols)} feature(s) contains non-NULL empty strings '
                  f'(e.g. values of "", distinct from NULL/NaN):')
            for column, empty_count in empty_string_cols:
                print(f'- {column} contains {empty_count} non-NULL empty string(s)')

        # Finally, print names and datatypes of features in file
        print('\nNAMES AND DATATYPES OF FEATURES:')
        print('-' * 50)  # Visual separator
        print(df.info(verbose=True, memory_usage=True, show_counts=True))
        print('-' * 50)  # Visual separator


def _find_outliers_core(df: pd.DataFrame, method: str = 'iqr', threshold: float = None,
                        verbose: bool = True) -> dict:
    """
    Core function to detect outliers in dataframe features using a specified detection method.

    This function analyzes numerical features in the dataframe and identifies outliers using the specified detection
    method. It supports three common approaches for outlier detection: IQR-based detection, Z-score method, and
    Modified Z-score method.

    Args:
        df (pd.DataFrame): The DataFrame to analyze for outliers
        method (str, optional): Outlier detection method to use.
            Options:
              - 'iqr': Interquartile Range (default)
              - 'zscore': Standard Z-score
              - 'modified_zscore': Modified Z-score
        threshold (float, optional): Threshold value for outlier detection. If None, uses method-specific defaults:
            - For IQR: 1.5 × IQR
            - For Z-score: 3.0 standard deviations
            - For Modified Z-score: 3.5
        verbose (bool, default=True): Whether to print detailed information about outliers
    """
    def numpy_to_python(obj):
        """This helper function converts Numpy numeric types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [numpy_to_python(item) for item in obj]
        return obj

    # Validate selected method
    valid_methods = ['iqr', 'zscore', 'modified_zscore']
    if method not in valid_methods:
        raise ValueError(f'Invalid outlier detection method: "{method}". '
                         f'Valid options are: {", ".join(valid_methods)}')

    # Set appropriate default thresholds based on method
    if threshold is None:
        if method == 'iqr':
            threshold = 1.5
        elif method == 'zscore':
            threshold = 3.0
        elif method == 'modified_zscore':
            threshold = 3.5

    # Identify all numerical features
    num_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]

    # Handle any case where no numerical features are present
    if not num_cols:
        if verbose:
            print('No numerical features found in dataframe. Outlier detection requires numerical data.')
        return {
            'summary': {
                'total_outliers': 0,
                'affected_rows_count': 0,
                'affected_rows_percent': 0.0,
                'features_with_outliers': []
            },
            'feature_results': {}
        }

    if verbose:
        print('-' * 50)
        print(f'Analyzing {len(num_cols)} numerical features for outliers...')
        print(f'Detection method: {method}')

        # Print threshold information based on the selected method
        if method == 'iqr':
            print(f'IQR threshold multiplier: {threshold}')
            print('\nIQR METHOD CONTEXT:')
            print('- Best for: Data with unknown distribution, resistant to extreme outliers')
            print('- Limitations: Less effective with small datasets, may miss subtle outliers')
            print('- Use case: General-purpose outlier detection for most datasets')

        elif method == 'zscore':
            print(f'Z-score threshold: {threshold} standard deviations')
            print('\nZ-SCORE METHOD CONTEXT:')
            print('- Best for: Normally distributed data, larger datasets')
            print('- Limitations: Sensitive to extreme outliers, skewed distributions')
            print('- Use case: When data approximates normal distribution and outliers are moderate')

        elif method == 'modified_zscore':
            print(f'Modified Z-score threshold: {threshold}')
            print('\nMODIFIED Z-SCORE METHOD CONTEXT:')
            print('- Best for: Datasets with extreme outliers, skewed distributions')
            print('- Limitations: More complex to interpret than standard Z-score')
            print('- Use case: Small datasets or when extreme outliers are present')
        print('-' * 50)

    # Initialize results dictionary
    results = {
        'summary': {
            'total_outliers': 0,
            'affected_rows_count': 0,
            'affected_rows_percent': 0.0,
            'features_with_outliers': []
        },
        'feature_results': {}
    }

    # Initialize array to track all instances with outliers
    outlier_rows = np.zeros(len(df), dtype=bool)

    # Process each numerical feature
    for column in num_cols:
        # Skip columns with all NaN values
        if df[column].isna().all():
            if verbose:
                print(f'Skipping "{column}": All values are NaN/Missing.')
            continue

        # Skip columns with only one unique value
        if df[column].nunique() <= 1:
            if verbose:
                print(f'Skipping "{column}": No variance present in feature.')
            continue

        if verbose:
            print(f'\nAnalyzing feature: "{column}"')

        # Get data without NaN values
        data = df[column].dropna()

        if method == 'iqr':
            # IQR-based outlier detection
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1

            # Define bounds using threshold (default 1.5)
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            method_desc = 'IQR-based detection'

        elif method == 'zscore':
            # Z-score based outlier detection
            mean = data.mean()
            std = data.std()

            # Skip if standard deviation is zero
            if std == 0:
                if verbose:
                    print(f'Skipping "{column}": Standard deviation is zero.')
                continue

            # Define bounds using Z-score threshold
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std

            method_desc = 'Z-score based detection'

        elif method == 'modified_zscore':
            # Modified Z-score based outlier detection
            median = data.median()
            # Median Absolute Deviation
            mad = np.median(np.abs(data - median))

            # Skip if MAD is zero
            if mad == 0:
                if verbose:
                    print(f'Skipping "{column}": Median Absolute Deviation is zero.')
                continue

            # Constant 0.6745 is used to make MAD comparable to standard deviation for normal distributions
            lower_bound = median - threshold * (mad / 0.6745)
            upper_bound = median + threshold * (mad / 0.6745)

            method_desc = 'Modified Z-score based detection'

        # Identify outliers
        outliers = df[
            ((df[column] < lower_bound) | (df[column] > upper_bound)) & ~df[column].isna()]

        # Store results
        if not outliers.empty:
            outlier_count = len(outliers)
            outlier_percent = (outlier_count / len(data)) * 100

            # Update global summary counts
            results['summary']['total_outliers'] += outlier_count

            # Update outlier rows tracking
            outlier_rows = outlier_rows | (
                    ((df[column] < lower_bound) | (df[column] > upper_bound)) & ~df[column].isna()).values

            # Add feature to list of features with outliers
            results['summary']['features_with_outliers'].append(column)

            # Store feature-specific results
            results['feature_results'][column] = {
                'method': method,
                'method_description': method_desc,
                'outlier_count': outlier_count,
                'outlier_percent': outlier_percent,
                'outlier_indices': outliers.index.tolist(),
                'thresholds': {
                    'lower': lower_bound,
                    'upper': upper_bound
                }
            }

            if verbose:
                print(f'Method: {method_desc}')
                print(f'Found {outlier_count} outliers ({outlier_percent:.2f}% of non-null values)')
                print(f'Thresholds: Lower = {lower_bound:.4f}, Upper = {upper_bound:.4f}')

                # Show extreme outliers (up to 3 min and max)
                if outlier_count > 0:
                    print('Sample of outlier values:')
                    extreme_low = outliers[outliers[column] < lower_bound][column].nsmallest(3)
                    extreme_high = outliers[outliers[column] > upper_bound][column].nlargest(3)

                    if not extreme_low.empty:
                        print('Low outliers:', extreme_low.tolist())
                    if not extreme_high.empty:
                        print('High outliers:', extreme_high.tolist())

        elif verbose:
            print(f'No outliers detected using {method_desc}')

    # Update summary with affected rows information
    affected_rows_count = np.sum(outlier_rows)
    affected_rows_percent = (affected_rows_count / len(df)) * 100
    results['summary']['affected_rows_count'] = int(affected_rows_count)
    results['summary']['affected_rows_percent'] = affected_rows_percent

    # Print summary if in verbose mode
    if verbose:
        print('-' * 50)
        print('OUTLIER DETECTION SUMMARY:')
        print('-' * 50)
        print(f'Total outliers detected: {results["summary"]["total_outliers"]}')
        print(f'Rows containing outliers: {affected_rows_count} ({affected_rows_percent:.2f}% of all rows)')

        if results['summary']['features_with_outliers']:
            print(f'Features containing outliers: {len(results["summary"]["features_with_outliers"])}')
            for feature in results['summary']['features_with_outliers']:
                result = results['feature_results'][feature]
                print(f'- {feature}: {result["outlier_count"]} outliers ({result["outlier_percent"]:.2f}%)')

        else:
            print('No outliers detected in any features.')

    # Apply Numpy-to-Python conversion to the results dictionary
    results = numpy_to_python(results)

    # Return dictionary summarizing results of outlier analysis
    return results


def _find_corrs_core(df: pd.DataFrame, method: str = 'pearson', threshold: float = 0.8, verbose: bool = True) -> dict:
    """
    Core function to detect highly-correlated features in a dataframe.

    This function analyzes numerical features in the dataframe and identifies pairs with
    correlation coefficients exceeding the specified threshold. High correlations may indicate
    redundant features that could be simplified or removed to improve model performance.

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

    Returns:
        dict: A dictionary containing correlation information with the following structure:
            {
                'summary': {
                    'method': str,                # Correlation method used
                    'num_correlated_pairs': int,  # Total number of highly correlated pairs
                    'max_correlation': float,     # Maximum correlation found
                    'avg_correlation': float,     # Average correlation among high pairs
                    'features_involved': list,    # List of features involved in high correlations
                },
                'correlation_pairs': [
                    {
                        'feature1': str,          # Name of first feature
                        'feature2': str,          # Name of second feature
                        'correlation': float,     # Correlation coefficient
                        'abs_correlation': float  # Absolute correlation value
                    },
                    ...
                ]
            }
    """
    def numpy_to_python(obj):
        """This helper function converts Numpy numeric types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [numpy_to_python(item) for item in obj]
        return obj

    # Validate correlation method
    valid_methods = ['pearson', 'spearman', 'kendall']
    if method not in valid_methods:
        raise ValueError(f'Invalid correlation method: "{method}". '
                         f'Valid options are: {", ".join(valid_methods)}')

    # Set appropriate default thresholds based on method if None is passed as value for threshold parameter
    if threshold is None:
        if method == 'pearson':
            threshold = 0.8
        elif method == 'spearman':
            threshold = 0.6
        elif method == 'kendall':
            threshold = 0.5

    # Validate threshold
    if not 0 <= threshold <= 1:
        raise ValueError(f'Threshold must be between 0 and 1, got {threshold}')

    # Identify numerical features
    num_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]

    # Handle any case where not enough numerical features are present
    if len(num_cols) <= 1:
        if verbose:
            print(
                'Insufficient numerical features found. Correlation analysis requires at least two numerical features.')

        return {
            'summary': {
                'method': method,
                'num_correlated_pairs': 0,
                'max_correlation': None,
                'avg_correlation': None,
                'features_involved': []
            },
            'correlation_pairs': []
        }

    # Process/procedural information for verbose mode
    if verbose:
        print('-' * 50)
        print(f'Analyzing correlations among {len(num_cols)} numerical features...')
        print(f'Using {method} correlation with threshold: ± {threshold}')

        # Add method descriptions here
        if method == 'pearson':
            print('\nMethod description: Pearson correlation measures linear relationships between features.')
            print('Best for: Normally distributed data with linear relationships and no significant outliers.')
            print('Limitations: Sensitive to outliers and only detects linear relationships.')

        elif method == 'spearman':
            print('\nMethod description: Spearman correlation measures monotonic (consistently increasing or '
                  'decreasing) relationships using ranks.')
            print('Best for: Data with non-linear but monotonic relationships or when outliers are present.')
            print('Limitations: Less powerful than Pearson for detecting truly linear relationships.')

        elif method == 'kendall':
            print('\nMethod description: Kendall\'s Tau measures concordance (agreement in ranking direction) between '
                  'feature pairs.')
            print('Best for: Small samples, ordinal data, or when robustness to outliers is essential.')
            print('Limitations: More computationally intensive and typically has lower values than other methods.')
        print('-' * 50)

    # Build correlation matrix
    corr_matrix = df[num_cols].corr(method=method)

    # Initialize results dictionary and tracking lists
    results = {
        'summary': {
            'method': method,
            'num_correlated_pairs': 0,
            'max_correlation': 0.0,
            'avg_correlation': 0.0,
            'features_involved': []
        },
        'correlation_pairs': []
    }

    # Instantiate empty data structures for tracking correlated features
    corr_features = set()
    corr_values = []

    # Find highly correlated pairs
    # I'll iterate only through the 'upper triangle' of the correlation matrix
    # The idea is to avoid duplicating pairs (e.g. corr of A,B is same as corr of B,A)
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            corr_value = corr_matrix.iloc[i, j]
            abs_corr = abs(corr_value)

            # Check if correlation exceeds threshold
            if abs_corr >= threshold:
                feature1 = num_cols[i]
                feature2 = num_cols[j]

                # Add features to tracking set
                corr_features.add(feature1)
                corr_features.add(feature2)

                # Add correlation value to tracking list
                corr_values.append(abs_corr)

                # Add pair details to results
                results['correlation_pairs'].append({
                    'feature_1': feature1,
                    'feature_2': feature2,
                    'correlation': corr_value,
                    'absolute_correlation': abs_corr
                })

    # Update summary information
    num_pairs = len(results['correlation_pairs'])
    results['summary']['num_correlated_pairs'] = num_pairs
    results['summary']['features_involved'] = sorted(list(corr_features))

    if num_pairs > 0:
        results['summary']['max_correlation'] = max(corr_values)
        results['summary']['avg_correlation'] = sum(corr_values) / num_pairs

    # Sort correlation pairs by absolute correlation value (descending)
    results['correlation_pairs'] = sorted(
        results['correlation_pairs'],
        key=lambda x: x['absolute_correlation'],
        reverse=True
    )

    # Print results if verbose is True
    if verbose:
        if num_pairs > 0:
            print(f'Found {num_pairs} feature pairs with {method} correlations of {threshold} or higher:')
            print('-' * 50)

            # Print each highly correlated pair
            for pair in results['correlation_pairs']:
                corr_sign = '+' if pair['correlation'] >= 0 else '-'
                print(f"{pair['feature_1']} and {pair['feature_2']}: {corr_sign}{pair['absolute_correlation']:.4f}")

            print('-' * 50)
            print(f'Maximum correlation found: {results["summary"]["max_correlation"]:.4f}')
            print(f'Average absolute correlation among highly-correlated pairs: '
                  f'{results["summary"]["avg_correlation"]:.4f}')
            print(f'Total count of unique features involved in high correlations: {len(corr_features)}')

            if len(corr_features) > 0:
                print('\nIndividual features participating in high correlations:')
                for feature in sorted(corr_features):
                    # Count how many correlations involve this feature
                    count = sum(1 for pair in results['correlation_pairs']
                                if pair['feature_1'] == feature or pair['feature_2'] == feature)
                    print(f'- {feature} (appears in {count} high-correlation pair{"s" if count > 1 else ""})')

        else:
            print(f'No feature pairs found with {method} correlation of ≥ {threshold}')

        print('-' * 50)

    # Apply Numpy-to-Python conversion to the results dictionary
    results = numpy_to_python(results)

    # Return results dictionary
    return results


def _reshape_core(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Core function for reshaping a DataFrame by handling missing values, dropping features, and subsetting data.

    Args:
        df (pd.DataFrame): Input DataFrame to reshape
        verbose (bool): Whether to print detailed information about operations. Defaults to True.

    Returns:
        pd.DataFrame: Reshaped DataFrame

    Raises:
        ValueError: If invalid indices are provided for column dropping
        ValueError: If an invalid subsetting proportion is provided
    """
    if verbose:
        print('-' * 50)  # Visual separator
        print('Beginning data reshape process.')
        print('-' * 50)  # Visual separator

    row_missing_cnt = df.isnull().any(axis=1).sum()  # Compute count
    # Ask if the user wants to delete *all* instances with any missing values, if any exist
    if row_missing_cnt > 0:
        user_drop_na = input('Do you want to drop all instances with *any* missing values? (Y/N): ')
        if user_drop_na.lower() == 'y':
            df = df.dropna()
            if verbose:
                print(f'After deletion of {row_missing_cnt} instances with missing values, {len(df)} instances remain.')

    # Ask if the user wants to drop any of the columns/features in the dataset
    user_drop_cols = input('\nDo you want to drop any of the features in the dataset? (Y/N): ')
    if user_drop_cols.lower() == 'y':
        print('The full set of features in the dataset is:')
        for col_idx, column in enumerate(df.columns, 1):  # Create enumerated list of features starting at 1
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                drop_cols_input = input('\nEnter the index integers of the features you wish to drop '
                                        '(comma-separated) or enter "C" to cancel: ')

                # Check for user cancellation
                if drop_cols_input.lower() == 'c':
                    if verbose:
                        print('Feature deletion cancelled.')
                    break

                # Create list of column indices to drop
                drop_cols_idx = [int(idx.strip()) for idx in drop_cols_input.split(',')]  # Splitting on comma

                # Verify that all index numbers of columns to be dropped are valid/in range
                if not all(1 <= idx <= len(df.columns) for idx in drop_cols_idx):  # Using a generator
                    raise ValueError('Some feature index integers entered are out of range/invalid.')

                # Convert specified column numbers to actual column names
                drop_cols_names = [df.columns[idx - 1] for idx in drop_cols_idx]  # Subtracting 1 from indices

                # Drop the columns
                df = df.drop(columns=drop_cols_names)
                if verbose:
                    print('-' * 50)  # Visual separator
                    print(f'Dropped features: {",".join(drop_cols_names)}')  # Note dropped columns
                    print('-' * 50)  # Visual separator
                break

            # Catch invalid user input
            except ValueError:
                print('Invalid input. Please enter valid feature index integers separated by commas.')
                continue  # Restart the loop

    # Ask if the user wants to sub-set the data
    user_subset = input('Do you want to sub-set the data by randomly deleting a specified proportion of '
                        'instances? (Y/N): ')
    if user_subset.lower() == 'y':
        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                subset_input = input('Enter the proportion of instances to DROP (0.0-1.0) or '
                                     'enter "C" to cancel: ')

                # Check for user cancellation
                if subset_input.lower() == 'c':
                    if verbose:
                        print('Random sub-setting cancelled.')
                    break

                subset_rate = float(subset_input)  # Convert string input to float
                if 0 < subset_rate < 1:  # If the float is valid (i.e. between 0 and 1)
                    retain_rate = 1 - subset_rate  # Compute retention rate
                    retain_row_cnt = int(len(df) * retain_rate)  # Select count of rows to keep in subset

                    df = df.sample(n=retain_row_cnt)  # No random state set b/c we want true randomness
                    if verbose:
                        print(f'Randomly dropped {subset_rate}% of instances. {retain_row_cnt} instances remain.')
                    break

                # Catch user input error for invalid/out-of-range float
                else:
                    print('Enter a value between 0.0 and 1.0.')

            # Catch outer-level user input errors
            except ValueError:
                print('Invalid input. Enter a float value between 0.0 and 1.0 or enter "C" to cancel.')
                continue  # Restart the loop
    if verbose:
        print('-' * 50)  # Visual separator
        print('Data reshape complete. Returning modified dataframe.')
        print('-' * 50)  # Visual separator

    return df  # Return the trimmed dataframe


def _subset_core(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Core function for subsetting a DataFrame via random sampling (with or without seed), stratified sampling,
    or time-based instance selection for timeseries data.

    Args:
        df (pd.DataFrame): Input DataFrame to subset
        verbose (bool): Whether to print detailed information about operations. Defaults to True.

    Returns:
        pd.DataFrame: Subsetted DataFrame

    Raises:
        ValueError: If invalid subsetting proportion is provided
        ValueError: If invalid date range is provided for timeseries subsetting
    """
    if verbose:
        print('-' * 50)
        print('Beginning data subset process.')
        print('-' * 50)

    # Build list of categorical columns, if any exist
    cat_cols = [column for column in df.columns
                if pd.api.types.is_object_dtype(df[column]) or
                isinstance(df[column].dtype, type(pd.Categorical.dtype))]

    # Check if data might be timeseries in form
    datetime_cols = []

    for column in df.columns:
        # Check if column is already datetime type
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            datetime_cols.append(column)
        # For string columns, try to parse as datetime
        elif pd.api.types.is_object_dtype(df[column]):
            try:
                # Try to parse first non-null value
                first_valid = df[column].dropna().iloc[0]
                pd.to_datetime(first_valid)
                datetime_cols.append(column)

            # If the process doesn't work, move to the next column
            except (ValueError, IndexError):
                continue

    is_datetime_index = pd.api.types.is_datetime64_any_dtype(df.index)
    has_datetime = bool(datetime_cols or is_datetime_index)

    # Convert any detected datetime string columns to datetime type
    if datetime_cols:
        for column in datetime_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                try:
                    df[column] = pd.to_datetime(df[column])

                # If conversion fails, remove that feature from the list of datetime features
                except ValueError:
                    datetime_cols.remove(column)

    if verbose:
        print('NOTE - TADPREP supports:'
              '\n - Seeded and unseeded random sampling'
              '\n - Stratified random sampling (if categorical features are present)'
              '\n - Date-based instance deletion (if the data are a timeseries).')

    # Present subset options based on data characteristics
    print('\nAvailable subset methods for this dataset:')
    print('1. Unseeded Random sampling (For true randomness)')
    print('2. Seeded Random sampling (For reproducibility)')

    # Only offer stratified sampling if categorical columns are present
    if cat_cols:
        print('3. Stratified random sampling (Preserves ratio of categories in data)')

    # Only offer time-based subsetting if datetime elements exist
    if has_datetime:
        print('4. Time-based boundaries for subset (Select only data within a specified time range)')

    while True:
        try:
            method = input('\nSelect a subset method or enter "C" to cancel: ').strip()

            # Handle cancellation
            if method.lower() == 'c':
                if verbose:
                    print('Subsetting cancelled. Input dataframe was not modified.')
                return df

            # Validate input based on available options
            valid_options = ['1', '2']

            # Show option for stratified sampling only if categorical features exist
            if cat_cols:
                valid_options.append('3')

            # Show option for time-based deletion only if data are a timeseries
            if has_datetime:
                valid_options.append('4')

            # Handle invalid user input
            if method not in valid_options:
                raise ValueError('Invalid selection.')
            break

        # Handle non-numerical user input
        except ValueError:
            print('Invalid input. Please select from the options provided.')
            continue

    # Handle true random sampling (unseeded)
    if method == '1':
        if verbose:
            print('-' * 50)  # Visual separator
            print('Proceeding with true (unseeded) random sampling...')
            print('-' * 50)  # Visual separator

        while True:
            try:
                # Fetch subsetting proportion
                subset_input = input('Enter the proportion of instances to DROP (0.0-1.0) or enter "C" to cancel: ')

                # Handle user cancellation
                if subset_input.lower() == 'c':
                    if verbose:
                        print('Random subsetting cancelled. Input dataframe was not modified.')
                    return df

                subset_rate = float(subset_input)
                if 0 < subset_rate < 1:
                    retain_rate = 1 - subset_rate  # Calculate retention rate
                    retain_row_cnt = int(len(df) * retain_rate)  # Construct integer count for instance deletion
                    df = df.sample(n=retain_row_cnt)  # Perform random sampling

                    if verbose:
                        print(f'Randomly dropped {subset_rate:.1%} of instances. {retain_row_cnt} instances remain.')
                    break

                # Handle invalid user input
                else:
                    print('Enter a value between 0.0 and 1.0.')

            # Handle non-numerical user input
            except ValueError:
                print('Invalid input. Enter a float value between 0.0 and 1.0 or enter "C" to cancel.')
                continue

    # Handle random sampling (seeded)
    elif method == '2':
        if verbose:
            print('-' * 50)  # Visual separator
            print('Proceeding with reproducible (seeded) random sampling...')
            print('-' * 50)  # Visual separator

        while True:
            try:
                # Fetch subsetting proportion
                subset_input = input('Enter the proportion of instances to DROP (0.0-1.0) or enter "C" to cancel: ')

                # Handle user cancellation
                if subset_input.lower() == 'c':
                    if verbose:
                        print('Random subsetting cancelled. Input dataframe was not modified.')
                    return df

                subset_rate = float(subset_input)
                if 0 < subset_rate < 1:
                    # Use a fixed seed for reproducibility
                    seed = 4  # Because 4 is my lucky number and using 42 is cringe
                    retain_rate = 1 - subset_rate  # Calculate retention rate
                    retain_row_cnt = int(len(df) * retain_rate)  # Construct integer count for instance deletion
                    df = df.sample(n=retain_row_cnt, random_state=seed)  # Perform seeded random sampling

                    if verbose:
                        print(f'Randomly dropped {subset_rate:.1%} of instances using reproducible sampling.')
                        print(f'{retain_row_cnt} instances remain.')
                    break

                # Handle invalid user input
                else:
                    print('Enter a value between 0.0 and 1.0.')

            # Handle non-numerical user input
            except ValueError:
                print('Invalid input. Enter a float value between 0.0 and 1.0 or enter "C" to cancel.')
                continue

    # Handle stratified sampling
    elif method == '3':
        if verbose:
            print('-' * 50)  # Visual separator
            print('Proceeding with stratified random sampling...')
            print('-' * 50)  # Visual separator

            user_explain = input('Would you like to see a brief explanation of stratified sampling? (Y/N): ')
            if user_explain.lower() == 'y':
                print('-' * 50)  # Visual separator
                print('Overview of Stratified Random Sampling:'
                      '\n- This method samples data while maintaining proportions in a specific feature.'
                      '\n- You will select a single categorical feature to "stratify by."'
                      '\n- The proportions in that feature (and *only* in that feature) will be preserved in your '
                      'sampled data.'
                      '\n- Example: If you stratify by gender in a customer dataset that is 80% male and 20% female,'
                      '\n  your sample will maintain this 80/20 split, even if other proportions in the data change.'
                      '\n- This process prevents underrepresentation of minority categories in your chosen feature.'
                      '\n- This method is most useful when you have important categories that are imbalanced.')
                print('-' * 50)  # Visual separator

        # Display available categorical features
        print('\nAvailable categorical features:')
        for idx, col in enumerate(cat_cols, 1):
            print(f'{idx}. {col}')

        while True:
            try:
                strat_input = input('\nEnter the number of the feature to stratify by: ')
                strat_idx = int(strat_input) - 1

                # Handle bad user input
                if not 0 <= strat_idx < len(cat_cols):
                    raise ValueError('Invalid feature number.')

                strat_col = cat_cols[strat_idx]  # Set column to stratify by

                # Fetch subsetting proportion
                subset_input = input('Enter the proportion of instances to DROP (0.0-1.0) or "C" to cancel: ')

                # Handle user cancellation
                if subset_input.lower() == 'c':
                    if verbose:
                        print('Stratified subsetting cancelled. Input dataframe was not modified.')
                    return df

                subset_rate = float(subset_input)
                if 0 < subset_rate < 1:
                    retain_rate = 1 - subset_rate  # Calculate retention rate
                    start_size = len(df)  # Store original dataset size

                    # Perform stratified sampling with minimum instance guarantee
                    stratified_dfs = []
                    unique_values = df[strat_col].unique()

                    # Check if we can maintain at least one instance per category
                    min_instances_needed = len(unique_values)
                    total_retain = int(len(df) * retain_rate)

                    if total_retain < min_instances_needed:
                        print(f'\nWARNING: Cannot maintain stratification with {retain_rate:.1%} retention rate.')
                        print(f'Need at least {min_instances_needed} instances to maintain one per category.')
                        user_proceed = input('Would you like to retain one instance per category instead? (Y/N): ')
                        if user_proceed.lower() != 'y':
                            continue
                        # Adjust to keep one per category
                        retain_rate = min_instances_needed / len(df)
                        print(f'\nAdjusted retention rate to {retain_rate:.1%} to maintain stratification.')

                    for value in unique_values:
                        subset = df[df[strat_col] == value]
                        retain_count = max(1, int(len(subset) * retain_rate))  # Ensure at least 1 instance
                        sampled = subset.sample(n=retain_count)
                        stratified_dfs.append(sampled)

                    df = pd.concat(stratified_dfs)

                    # Calculate actual retention rate achieved
                    true_retain_rate = len(df) / start_size
                    true_drop_rate = 1 - true_retain_rate

                    if abs(subset_rate - true_drop_rate) > 0.01:  # If difference is more than 1%
                        print(f'\nNOTE: To maintain proportional stratification, the actual drop rate was '
                              f'adjusted to {true_drop_rate:.1%}.')
                        print(f'(You requested: {subset_rate:.1%})')

                    if verbose:
                        print(f'\nPerformed stratified sampling by "{strat_col}".')
                        print(f'{len(df)} instances remain.')
                    break

                # Handle invalid user input
                else:
                    print('Enter a value between 0.0 and 1.0.')

            # Handle all other input problems
            except ValueError as exc:
                print(f'Invalid input: {str(exc)}')
                continue

    # Handle time-based subsetting
    elif method == '4':
        if verbose:
            print('-' * 50)  # Visual separator
            print('Proceeding with time-based subsetting...')
            print('-' * 50)  # Visual separator

        # Identify datetime column
        if is_datetime_index:
            time_col = df.index
            col_name = 'index'

        # If there's more than one datetime feature, allow user to select which one to use
        else:
            if len(datetime_cols) > 1:
                print('\nAvailable datetime features:')
                for idx, col in enumerate(datetime_cols, 1):
                    print(f'{idx}. {col}')

                while True:
                    try:
                        col_input = input('\nSelect the feature you want to use to create your subset "boundaries": ')
                        col_idx = int(col_input) - 1
                        if not 0 <= col_idx < len(datetime_cols):
                            raise ValueError('Invalid feature number.')
                        col_name = datetime_cols[col_idx]
                        time_col = df[col_name]
                        break

                    # Handle bad user input
                    except ValueError:
                        print('Invalid input. Please enter a valid number.')
            else:
                col_name = datetime_cols[0]
                time_col = df[col_name]

        # Determine time frequency from data
        if isinstance(time_col, pd.Series):
            time_diffs = time_col.sort_values().diff().dropna().unique()
        else:
            time_diffs = pd.Series(time_col).sort_values().diff().dropna().unique()

        # Fetch most common temporal difference between instances to identify the timeseries frequency
        if len(time_diffs) > 0:
            most_common_diff = pd.Series(time_diffs).value_counts().index[0]
            freq_hours = most_common_diff.total_seconds() / 3600  # Use hours as a baseline computation

            # Define frequency type/level using most common datetime typologies
            if freq_hours < 1:
                freq_str = 'sub-hourly'
            elif freq_hours == 1:
                freq_str = 'hourly'
            elif freq_hours == 24:
                freq_str = 'daily'
            elif 24 * 28 <= freq_hours <= 24 * 31:
                freq_str = 'monthly'
            elif 24 * 89 <= freq_hours <= 24 * 92:
                freq_str = 'quarterly'
            elif 24 * 365 <= freq_hours <= 24 * 366:
                freq_str = 'yearly'

            # In case the data are just weird
            else:
                freq_str = 'irregular'

        # In case we literally couldn't compute the time deltas
        else:
            freq_str = 'undetermined'

        # Fetch data range and some example timestamps
        min_date = time_col.min()
        max_date = time_col.max()
        example_stamps = sorted(time_col.sample(n=min(3, len(time_col))).dt.strftime('%Y-%m-%d %H:%M:%S'))

        if verbose:
            print(f'\nData frequency/time aggregation level appears to be {freq_str}.')
        print(f'Your timeseries spans from {min_date} to {max_date}.')
        if verbose:
            print('\nExample timestamps from your data:')
            for stamp in example_stamps:
                print(f'- {stamp}')
            print('\nEnter time boundaries in any standard format (YYYY-MM-DD, MM/DD/YYYY, etc.) or press Enter to use '
                  'the earliest/latest timestamp')

        while True:
            try:
                # Fetch start boundary
                start_input = input('\nEnter the earliest/starting time boundary for the subset, or press Enter '
                                    'to use the earliest timestamp: ').strip()
                if start_input:
                    start_date = pd.to_datetime(start_input)
                else:
                    start_date = min_date

                # Fetch end boundary
                end_input = input('\nEnter the latest/ending time boundary for the subset, or press Enter to use '
                                  'the latest timestamp: ').strip()
                if end_input:
                    end_date = pd.to_datetime(end_input)
                else:
                    end_date = max_date

                # Validate the user's date range
                if start_date > end_date:
                    raise ValueError('Start time must be before end time.')

                # Apply the time filter
                if is_datetime_index:
                    df_filtered = df[start_date:end_date]
                else:
                    df_filtered = df[(df[col_name] >= start_date) & (df[col_name] <= end_date)]

                # Check if any data remains after filtering
                if df_filtered.empty:
                    print(f'WARNING: No instances found between {start_date} and {end_date}.')
                    print('Please provide a different date range.')
                    continue  # Go back to date input

                df = df_filtered  # Only update the df if we actually found data in the range defined by the user

                if verbose:
                    print(f'\nRetained instances from {start_date} to {end_date}.')
                    print(f'{len(df)} instances remain after subsetting.')
                break

            except ValueError as exc:
                print(f'Invalid input: {str(exc)}')
                continue

    if verbose:
        print('-' * 50)  # Visual separator
        print('Data subsetting complete. Returning modified dataframe.')
        print('-' * 50)  # Visual separator

    return df  # Return modified dataframe


def _rename_and_tag_core(df: pd.DataFrame, verbose: bool = True, tag_features: bool = False) -> pd.DataFrame:
    """
    Core function to rename features and to append the '_ord' and/or '_target' suffixes to ordinal or target features,
    if desired by the user.

    Args:
        df (pd.DataFrame): Input DataFrame to reshape.
        verbose (bool): Whether to print detailed information about operations. Defaults to True.
        tag_features (bool): Whether to activate the feature-tagging process. Defaults to False.

    Returns:
        pd.DataFrame: Reshaped DataFrame

    Raises:
        ValueError: If invalid indices are provided for column renaming
        ValueError: If any other invalid input is provided
    """
    def valid_identifier_check(name: str) -> tuple[bool, str]:
        """Helper function to validate whether a proposed feature name follows Python naming conventions."""
        # Check using valid identifier method
        if not name.isidentifier():
            return False, ('New feature name must be a valid Python identifier (i.e. must contain only letters, '
                           'numbers, and underscores, and cannot start with a number)')
        else:
            return True, ''

    def problem_chars_check(name: str) -> list[str]:
        """Helper function to identify potentially problematic characters in a proposed feature name."""
        # Instantiate empty list to hold warnings
        char_warnings = []
        # Space check
        if ' ' in name:
            char_warnings.append('Contains spaces')

        # Special character check
        if any(char in name for char in '!@#$%^&*()+=[]{}|\\;:"\',.<>?/'):
            char_warnings.append('Contains special characters')

        # Double underscore check
        if '__' in name:
            char_warnings.append('Contains double underscores')

        # Return warning list
        return char_warnings

    def antipattern_check(name: str) -> list[str]:
        """Helper function to check for common naming anti-patterns in a proposed feature name."""
        # Instantiate empty list to hold warnings
        pattern_warnings = []

        # Check for common problematic anti-patterns
        if name[0].isdigit():
            pattern_warnings.append('Name starts with a number')
        if name.isupper():
            pattern_warnings.append('Name is all uppercase')
        if len(name) <= 2:
            pattern_warnings.append('Name is very short')
        if len(name) > 30:
            pattern_warnings.append('Name is very long')

        # Return warning list
        return pattern_warnings

    def python_keyword_check(name: str) -> bool:
        """Helper function to check if a proposed feature name conflicts with Python keywords."""
        import keyword  # We can do this in one step with the keyword library
        return keyword.iskeyword(name)

    # Track all rename operations for end-of-process summary
    rename_tracker = []

    if verbose:
        print('-' * 50)  # Visual separator
        print('Beginning feature renaming process.')
        print('-' * 50)  # Visual separator

    while True:  # We can justify 'while True' because we have a cancel-out input option
        try:
            if verbose:
                print('The list of features currently present in the dataset is:')
            else:
                print('Features:')

            for col_idx, column in enumerate(df.columns, 1):  # Create enumerated list of features starting at 1
                print(f'{col_idx}. {column}')

            rename_cols_input = input('\nEnter the index integer of the feature you want to rename, enter "S" to skip '
                                      'this step, or enter "E" to exit the renaming/tagging process: ')

            # Check for user skip
            if rename_cols_input.lower() == 's':
                if verbose:
                    print('Feature renaming skipped.')
                break

            # Check for user process exit
            elif rename_cols_input.lower() == 'e':
                if verbose:
                    print('Exiting process. Dataframe was not modified.')
                return df

            col_idx = int(rename_cols_input)  # Convert input to integer
            if not 1 <= col_idx <= len(df.columns):  # Validate entry
                raise ValueError('Column index is out of range.')

            # Get new column name from user
            col_name_old = df.columns[col_idx - 1]
            col_name_new = input(f'Enter new name for feature "{col_name_old}": ').strip()

            # Run Python identifier validation
            valid_status, valid_msg = valid_identifier_check(col_name_new)
            if not valid_status:
                print(f'ERROR: Invalid feature name: {valid_msg}')
                continue

            # Check if proposed name is a Python keyword
            if python_keyword_check(col_name_new):
                print(f'"{col_name_new}" is a Python keyword and cannot be used as a feature name. '
                      f'Please choose a different name.')
                continue

            # Check for problematic characters
            char_warnings = problem_chars_check(col_name_new)
            if char_warnings and verbose:
                print('\nWARNING: Potential character-choice issues with proposed feature name:')
                for warning in char_warnings:
                    print(f'- {warning}')
                if input('Do you want to use your proposed feature name anyway? (Y/N): ').lower() != 'y':
                    continue

            # Check for naming anti-patterns
            pattern_warnings = antipattern_check(col_name_new)
            if pattern_warnings and verbose:
                print('\nWARNING: Proposed feature name does not follow best practices for anti-pattern avoidance:')
                for warning in pattern_warnings:
                    print(f'- {warning}')
                if input('Do you want to use your proposed feature name anyway? (Y/N): ').lower() != 'y':
                    continue

            # Validate name to make sure it doesn't already exist
            if col_name_new in df.columns:
                print(f'Feature name "{col_name_new}" already exists. Choose a different name.')
                continue

            # Preview and approve each change in verbose mode only
            if verbose:
                print(f'\nProposed feature name change: "{col_name_old}" -> "{col_name_new}"')
                if input('Apply this change? (Y/N): ').lower() != 'y':
                    continue

            # Rename column in-place
            df = df.rename(columns={col_name_old: col_name_new})

            # Track the rename operation
            rename_tracker.append({'old_name': col_name_old, 'new_name': col_name_new, 'type': 'rename'})

            # Print renaming summary message in verbose mode only
            if verbose:
                print('-' * 50)  # Visual separator
                print(f'Successfully renamed feature "{col_name_old}" to "{col_name_new}".')
                print('-' * 50)  # Visual separator

            # Ask if user wants to rename another column
            if input('Do you want to rename another feature? (Y/N): ').lower() != 'y':
                break

        # Catch input errors
        except ValueError as exc:
            print(f'Invalid input: {exc}')

    if tag_features:
        if verbose:
            print('-' * 50)  # Visual separator
            print('Beginning ordinal feature tagging process.')
            print('-' * 50)  # Visual separator
            print('You may now select any ordinal features which you know to be present in the dataset and append the '
                  '"_ord" suffix to their feature names.')
            print('If no ordinal features are present in the dataset, enter "S" to skip this step.')

        else:
            print('\nOrdinal feature tagging:')

        print('\nFeatures:')
        for col_idx, column in enumerate(df.columns, 1):
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                ord_input = input('\nEnter the index integers of ordinal features (comma-separated), enter "S" to skip '
                                  'this step, or enter "E" to exit the renaming process: ')

                # Check for user skip
                if ord_input.lower() == 's':
                    if verbose:
                        print('Ordinal feature tagging skipped.')  # Note the cancellation
                    break  # Exit the loop

                # Check for user process exit
                elif ord_input.lower() == 'e':
                    if verbose:
                        print('Exiting process. Dataframe was not modified.')
                    return df

                ord_idx_list = [int(idx.strip()) for idx in ord_input.split(',')]  # Create list of index integers

                # Validate that all entered index integers are in range
                if not all(1 <= idx <= len(df.columns) for idx in ord_idx_list):
                    raise ValueError('Some feature indices are out of range.')

                ord_names_pretag = [df.columns[idx - 1] for idx in ord_idx_list]  # Create list of pretag names

                # Generate mapper for renaming columns with '_ord' suffix
                ord_rename_map = {name: f'{name}_ord' for name in ord_names_pretag if not name.endswith('_ord')}

                # Validate that tags for the selected columns are not already present (i.e. done pre-import)
                if not ord_rename_map:  # If the mapper is empty
                    print('WARNING: All selected features are already tagged as ordinal.')  # Warn the user
                    print('Skipping ordinal tagging.')
                    break

                df.rename(columns=ord_rename_map, inplace=True)  # Perform tagging
                # Track ordinal tagging operations
                for old_name, new_name in ord_rename_map.items():
                    rename_tracker.append({'old_name': old_name, 'new_name': new_name, 'type': 'ordinal_tag'})

                # Print ordinal tagging summary message in verbose mode only
                if verbose:
                    print('-' * 50)  # Visual separator
                    print(f'Tagged the following features as ordinal: {", ".join(ord_rename_map.keys())}')
                break

            # Catch invalid input
            except ValueError as exc:
                print(f'Invalid input: {exc}')
                continue

        if verbose:
            print('-' * 50)  # Visual separator
            print('Beginning target feature tagging process.')
            print('-' * 50)  # Visual separator
            print('You may now select any target features which you know to be present in the dataset and append the '
                  '"_target" suffix to their feature names.')
            print('If no target features are present in the dataset, enter "S" to skip this step.')

        else:
            print('\nTarget feature tagging:')

        print('\nFeatures:')
        for col_idx, column in enumerate(df.columns, 1):
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                target_input = input('\nEnter the index integers of target features (comma-separated), enter "S" to '
                                     'skip this step, or enter "E" to exit the renaming process: ')

                # Check for user cancellation
                if target_input.lower() == 's':
                    if verbose:
                        print('Target feature tagging skipped.')
                    break

                # Check for user process exit
                elif target_input.lower() == 'e':
                    if verbose:
                        print('Exiting process. Dataframe was not modified.')
                    return df

                target_idx_list = [int(idx.strip()) for idx in target_input.split(',')]  # Create list of index integers

                # Validate that all entered index integers are in range
                if not all(1 <= idx <= len(df.columns) for idx in target_idx_list):
                    raise ValueError('Some feature indices are out of range.')

                target_names_pretag = [df.columns[idx - 1] for idx in target_idx_list]  # List of pretag names

                # Generate mapper for renaming columns with '_target' suffix
                target_rename_map = {name: f'{name}_target' for name in target_names_pretag if
                                     not name.endswith('_target')}

                # Validate that tags for the selected columns are not already present (i.e. done pre-import)
                if not target_rename_map:  # If the mapper is empty
                    print('WARNING: All selected features are already tagged as targets.')  # Warn the user
                    print('Skipping target tagging.')
                    break

                df = df.rename(columns=target_rename_map)  # Perform tagging
                # Track target tagging operations
                for old_name, new_name in target_rename_map.items():
                    rename_tracker.append({'old_name': old_name, 'new_name': new_name, 'type': 'target_tag'})

                # Print ordinal tagging summary message in verbose mode only
                if verbose:
                    print('-' * 50)  # Visual separator
                    print(f'Tagged the following features as targets: {", ".join(target_rename_map.keys())}')
                    print('-' * 50)  # Visual separator
                break

            # Catch invalid input
            except ValueError as exc:
                print(f'Invalid input: {exc}')
                continue  # Restart the loop

    if verbose and rename_tracker:  # Only show summary if verbose mode is active and changes were made
        print('\nSUMMARY OF CHANGES:')
        print('-' * 50)

        # Create summary DataFrame
        summary_df = pd.DataFrame(rename_tracker)

        # Group by operation type and display changes
        for op_type in summary_df['type'].unique():
            op_changes = summary_df[summary_df['type'] == op_type]
            if op_type == 'rename':
                print('Features Renamed:')
            elif op_type == 'ordinal_tag':
                print('\nOrdinal Tags Added:')
            else:  # target_tag
                print('\nTarget Tags Added:')

            # The .iterrows() method makes the summary printing easy
            for _, row in op_changes.iterrows():
                print(f'  {row["old_name"]} -> {row["new_name"]}')

        print('-' * 50)
        print('Feature renaming/tagging complete. Returning modified dataframe.')
        print('-' * 50)

    elif verbose:
        print('No changes were made to the dataframe.')
        print('-' * 50)

    return df  # Return dataframe with renamed and tagged columns


def _feature_stats_core(df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Core function to provide feature-level descriptive and analytical statistical values.

    Args:
        df (pd.DataFrame): The input dataframe for analysis.
        verbose (bool): Whether to print more detailed information/explanations for each feature. Defaults to True.

    Returns:
        None. This is a void function which prints information to the console.
    """
    if verbose:
        print('Displaying feature-level information for all features in dataset...')
        print('-' * 50)  # Visual separator

    # Begin with four-category feature detection process
    # Boolean features (either logical values or 0/1 integers with exactly two unique values)
    bool_cols = []
    for column in df.columns:
        # Check for explicit boolean datatypes
        if pd.api.types.is_bool_dtype(df[column]):
            bool_cols.append(column)

        # Check for 0/1 integers which function as boolean
        elif (pd.api.types.is_numeric_dtype(df[column])  # Numeric check
              and df[column].dropna().nunique() == 2  # Unique value count check
              and set(df[column].dropna().unique()).issubset({0, 1})):  # Check if unique values are a subset of {0, 1}
            bool_cols.append(column)  # If all checks are passed, this is a boolean feature

        # Check for explicit Python True/False values
        elif (df[column].dropna().nunique() == 2  # Unique value count check
              and set(df[column].dropna().unique()) == {True, False}):  # Check for True/False values
            bool_cols.append(column)  # If all checks are passed, this is a boolean feature

    # Datetime features
    datetime_cols = []
    for column in df.columns:
        # Check for explicit datetime datatypes
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            datetime_cols.append(column)
        # For string columns, try to parse them as datetime and assess results
        elif pd.api.types.is_object_dtype(df[column]):
            try:
                # Fetch the first non-null value to check if the parsing is possible
                first_valid = df[column].dropna().iloc[0]

                # Skip boolean values - they can't be converted to datetime
                if isinstance(first_valid, bool):
                    continue

                # Attempt to parse the first value as a datetime - raise ValueError if not possible
                pd.to_datetime(first_valid)

                # If we get this far, the first value is a valid datetime
                # Now we check a sample of values to confirm most are valid dates
                # I'll limit the sample to 100 values which should be enough
                sample_size = min(100, len(df[column].dropna()))

                # Take a random sample if we have values, otherwise use an empty list
                sample = df[column].dropna().sample(sample_size) if sample_size > 0 else []

                # Count how many values in the sample can be parsed as dates
                # NOTE: pd.to_datetime with errors='coerce' returns NaT for invalid dates
                valid_dates = sum(pd.to_datetime(value, errors='coerce') is not pd.NaT for value in sample)

                # Calculate the proportion of valid dates
                # I use max(1, len) to avoid dividing by zero
                valid_prop = valid_dates / max(1, len(sample))

                # If more than 80% of sampled values are valid dates, classify the feature as datetime
                # This threshold helps avoid classifying text fields that occasionally contain dates as datetime
                if valid_prop > 0.8:
                    datetime_cols.append(column)

            # ValueError if failure to parse first value, IndexError if the feature is empty
            except (ValueError, IndexError):
                continue

    # Categorical features (*excluding* those identified as boolean or datetime)
    cat_cols = [column for column in df.columns
                # Datatype check
                if (pd.api.types.is_object_dtype(df[column])
                    or isinstance(df[column].dtype, type(pd.Categorical.dtype)))
                # Membership check
                and column not in datetime_cols and column not in bool_cols]

    # Numerical features (excluding those identified as boolean)
    num_cols = [column for column in df.columns
                # Datatype check
                if pd.api.types.is_numeric_dtype(df[column])
                # Membership check
                and column not in bool_cols]

    # Instantiate empty arrays for potential data quality issues
    zero_var_cols = []
    near_const_cols = []
    dup_cols = []

    # Detect zero-variance and near-constant features
    for column in df.columns:
        # Skip columns with all NaN values
        if df[column].isna().all():
            continue

        # Detect zero variance features
        if df[column].nunique() == 1:
            zero_var_cols.append(column)
            continue

        # Detect near-constant features (>95% single value)
        if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            value_counts = df[column].value_counts(normalize=True)
            if value_counts.iloc[0] >= 0.95:
                near_const_cols.append((column, value_counts.index[0], value_counts.iloc[0] * 100))

    # Detect potentially duplicated features (again using sampling for efficiency)
    # I'll compare the first 1000 rows of each feature to check for exact matches
    sample_df = df.head(1000)
    cols_to_check = df.columns

    for idx, feature_1 in enumerate(cols_to_check):
        for feature_2 in cols_to_check[idx + 1:]:
            # Only compare features of the same type
            if df[feature_1].dtype != df[feature_2].dtype:
                continue

            # Check if the columns are fully identical in the sample
            if sample_df[feature_1].equals(sample_df[feature_2]):
                dup_cols.append((feature_1, feature_2))

    # Display detection results if verbose mode is on
    if verbose:
        # Display feature type counts
        print(f'FEATURE TYPE DISTRIBUTION:')
        print(f'- Boolean features: {len(bool_cols)}')
        print(f'- Datetime features: {len(datetime_cols)}')
        print(f'- Categorical features: {len(cat_cols)}')
        print(f'- Numerical features: {len(num_cols)}')
        print('-' * 50)

        # Display the names of each feature type if any exist
        if bool_cols:
            print('The boolean features are:')
            print(', '.join(bool_cols))
            print('-' * 50)

        if datetime_cols:
            print('The datetime features are:')
            print(', '.join(datetime_cols))
            print('-' * 50)

        if cat_cols:
            print('The categorical features are:')
            print(', '.join(cat_cols))
            print('-' * 50)

        if num_cols:
            print('The numerical features are:')
            print(', '.join(num_cols))
            print('-' * 50)

        # Display data quality indicators if they exist
        if zero_var_cols:
            print('ALERT: The following features have zero variance (i.e. only a single unique value):')
            for column in zero_var_cols:
                print(f'- Feature: "{column}" (Single unique value: "{df[column].iloc[0]}")')
            print('-' * 50)

        if near_const_cols:
            print('ALERT: The following features have near-constant values (>=95% single value):')
            for column, val, rate in near_const_cols:
                print(f' - Feature "{column}": the value "{val}" appears in {rate:.2f}% of non-null instances.')
            print('-' * 50)

        if dup_cols:
            print('ALERT: Based on an analysis of the first 1000 instances, the following features may be duplicates:')
            for col1, col2 in dup_cols:
                print(f'- "{col1}" and "{col2}"')
            print('-' * 50)

    # Helper functions to calculate useful statistical measures
    def calculate_entropy(series):
        """Calculate the Shannon entropy/information content of a series"""
        value_counts = series.value_counts(normalize=True)
        return -sum(p * np.log2(p) for p in value_counts if p > 0)

    def format_large_nums(num):
        """Format any large numbers with comma separators"""
        return f'{num:,}'

    def format_numerics(num):
        """Format numerical values appropriately based on their type and magnitude"""
        if isinstance(num, (int, np.integer)) or (isinstance(num, float) and num.is_integer()):
            return format_large_nums(int(num))
        else:
            return f'{num:.4f}'

    def format_percents(part, whole):
        """Calculate and format percentage values"""
        if whole == 0:
            return '0.00%'
        return f'{(part / whole * 100):.2f}%'

    def show_key_vals(column: str, df: pd.DataFrame, feature_type: str):
        """This helper function calculates and prints key values and missingness info at the feature level."""
        if verbose:
            print('-' * 50)
            print(f'Key values for {feature_type} feature "{column}":')
            print('-' * 50)
        else:
            print(f'\nFeature: "{column}" - ({feature_type})')

        # Calculate missingness at feature level
        missing_cnt = df[column].isnull().sum()  # Total count
        print(f'Missing values: {format_large_nums(missing_cnt)} ({format_percents(missing_cnt, len(df))})')

        # Ensure the feature is not fully null before producing statistics
        if not df[column].isnull().all():
            # Boolean features statistics
            if feature_type == 'Boolean':
                value_cnt = df[column].value_counts()
                true_cnt = int(df[column].sum())
                false_cnt = int(len(df[column].dropna()) - true_cnt)
                print(f'True values: {format_large_nums(true_cnt)} ({format_percents(true_cnt, 
                                                                                     len(df[column].dropna()))})')
                print(f'False values: {format_large_nums(false_cnt)} ({format_percents(false_cnt, 
                                                                                       len(df[column].dropna()))})')

                if verbose:
                    print('\nValue counts:')
                    print(value_cnt)

            # DateTime features statistics
            elif feature_type == 'Datetime':
                # Convert to datetime if necessary
                if not pd.api.types.is_datetime64_any_dtype(df[column]):
                    series = pd.to_datetime(df[column], errors='coerce')
                else:
                    series = df[column]

                # Check if we have any valid dates after conversion
                if series.notna().any():
                    # Get minimum and maximum dates
                    min_date = series.min()
                    max_date = series.max()

                    # Print basic range information
                    print(f'Date range: {min_date} to {max_date}')
                else:
                    print('No valid datetime values found in this feature.')

                # Identify time granularity if possible (verbose mode only)
                if verbose and not series.dropna().empty:
                    try:
                        # Sort data and calculate time differences
                        sorted_dates = series.dropna().sort_values()
                        time_diffs = sorted_dates.diff().dropna()

                        if len(time_diffs) > 0:
                            # Convert time differences to days for more consistent analysis
                            days_diffs = time_diffs.dt.total_seconds() / (24 * 3600)
                            median_days = days_diffs.median()

                            # Determine time granularity based on median difference
                            if median_days < 0.1:  # Less than 2.4 hours
                                level = 'sub-hourly'
                            elif median_days < 0.5:  # Less than 12 hours
                                level = 'hourly'
                            elif median_days < 3:  # Between 0.5 and 3 days
                                level = 'daily'
                            elif median_days < 20:  # Between 3 and 20 days
                                level = 'weekly'
                            elif median_days < 45:  # Between 20 and 45 days
                                level = 'monthly'
                            elif median_days < 120:  # Between 45 and 120 days
                                level = 'quarterly'
                            elif median_days < 550:  # Between 120 and 550 days
                                level = 'yearly'
                            else:
                                level = 'multi-year'

                            if level:
                                print(f'This datetime feature appears to be organized at the "{level}" level.')

                            else:
                                print(f'This datetime feature\'s temporal aggregation level is unknown.')
                    except (AttributeError, TypeError, IndexError):
                        pass

            elif feature_type == 'Categorical':
                # Check for and handle empty strings before calculating statistics
                empty_str_cnt = (df[column] == '').sum()
                if empty_str_cnt > 0:
                    print(f'\nALERT: Found {empty_str_cnt} empty strings in this feature.')
                    if verbose:
                        user_convert = input('Would you like to convert empty strings to NaN values? (Y/N): ').lower()
                        if user_convert == 'y':
                            # Create a temporary copy of the column for display purposes only
                            temp_series = df[column].replace('', np.nan)
                            print(f'Converted {empty_str_cnt} empty strings to NaN for analysis.')
                        else:
                            temp_series = df[column]
                            print('Empty strings will be treated as a distinct category.')
                    else:
                        temp_series = df[column]
                        print('NOTE: Empty strings are being treated as a distinct category.')
                else:
                    temp_series = df[column]

                # Use temp_series for all calculations
                value_counts = temp_series.value_counts()
                unique_values = temp_series.nunique()
                mode_val = temp_series.mode().iloc[0] if not temp_series.mode().empty else 'No mode exists'

                print(f'Unique values: {format_large_nums(unique_values)}')
                print(
                    f'Mode: {mode_val} (appears {format_large_nums(value_counts.iloc[0])} times, '
                    f'{format_percents(value_counts.iloc[0], len(temp_series.dropna()))})')

                # Add entropy calculation if appropriate
                if len(temp_series.dropna()) > 0:
                    entropy = calculate_entropy(temp_series.dropna())
                    max_entropy = np.log2(unique_values) if unique_values > 0 else 0

                    if verbose:
                        print(f'Information entropy: {entropy:.2f} bits')
                        if max_entropy > 0:
                            print(f'Normalized entropy: {entropy / max_entropy:.2f} '
                                  f'(Where 0=constant, 1=uniform distribution)')
                            print('\nExplanation: Entropy measures the unpredictability of the feature\'s values.')
                            print('Low entropy (near 0) means the feature is highly predictable or skewed.')
                            print('High entropy (near the maximum) means values are evenly distributed.')

                if verbose:
                    if len(value_counts) <= 10:
                        print('\nComplete distribution of all values:')
                        print(value_counts)
                    else:
                        print(f'\nDistribution of most common values (showing top 10 out of {len(value_counts)} '
                              f'total unique values):')
                        print(value_counts.head(10))

                        # Calculate the percentage of total data represented by the displayed values
                        top_cnt = value_counts.head(10).sum()
                        total_cnt = len(temp_series.dropna())
                        top_rate = (top_cnt / total_cnt) * 100

                        print(f'These top 10 values represent {top_rate:.1f}% of all non-null data in this feature.')
                        print(f'There are {len(value_counts) - 10} additional unique values not shown.')

                    # Show top frequency ratios
                    if len(value_counts) >= 2:
                        ratio = value_counts.iloc[0] / value_counts.iloc[1]
                        print(f'\nTop-to-second value frequency ratio: {ratio:.2f}:1')

                    # Show distribution pattern
                    if len(value_counts) > 5:
                        print('\nDistribution pattern:')
                        total_cnt = len(temp_series.dropna())
                        coverage = 0
                        for idx in range(min(5, len(value_counts))):
                            val_freq = value_counts.iloc[idx]
                            percent = val_freq / total_cnt * 100
                            coverage += percent  # Addition assignment
                            print(f'- Top {idx + 1} values cover: {coverage:.1f}% of data')

            # Numerical features statistics
            elif feature_type == 'Numerical':
                stats = df[column].describe()

                # Basic statistics
                print(f'Mean: {format_numerics(stats["mean"])}')
                print(f'Min: {format_numerics(stats["min"])}')
                print(f'Max: {format_numerics(stats["max"])}')

                # Enhanced statistics when verbose is True
                if verbose:
                    print(f'Median: {format_numerics(stats["50%"])}')
                    print(f'Std Dev: {format_numerics(stats["std"])}')

                    # Provide quartile information
                    print(f'25th percentile: {format_numerics(stats["25%"])}')
                    print(f'75th percentile: {format_numerics(stats["75%"])}')
                    print('NOTE: The Interquartile range (IQR) represents the middle 50% of the data.')
                    print(f'IQR: {format_numerics(stats["75%"] - stats["25%"])}')

                    # Provide skewness
                    print('\nCalculating skew...')
                    print('NOTE: Skewness measures the asymmetry of a numerical distribution.')
                    skew = df[column].skew()
                    print(f'Skewness: {format_numerics(skew)}')
                    if abs(skew) < 0.5:
                        print('NOTE: This is an approximately symmetric distribution.')
                    elif abs(skew) < 1:
                        print('NOTE: This is a moderately skewed distribution.')
                    else:
                        print('NOTE: This is a highly skewed distribution.')

                    # Provide kurtosis
                    print('\nCalculating kurtosis...')
                    print('NOTE: Kurtosis measures the "tailedness" of a numerical distribution.')
                    kurt = df[column].kurtosis()
                    print(f'Kurtosis: {format_numerics(kurt)}')
                    if kurt < -0.5:
                        print('  - The feature is platykurtic - flatter than a normal distribution.')
                    elif kurt > 0.5:
                        print('  - The feature is leptokurtic - more peaked than a normal distribution.')
                    else:
                        print('  - The feature is mesokurtic - similar to a normal distribution.')

                    # Coefficient of variation
                    if stats["mean"] != 0:
                        cv = stats["std"] / stats["mean"]
                        print(f'Coefficient of Variation (CV): {format_numerics(cv)}')
                        print(f'NOTE: The CV of a feature indicates its relative variability across the dataset.')

                        # Contextual interpretation for feature-level variability
                        if cv < 0.1:
                            print('  - The feature\'s values are consistently similar across samples.')
                        elif cv < 0.5:
                            print('  - The feature shows noticeable but not extreme variation across samples.')
                        else:
                            print('  - The feature\'s values differ substantially across different samples.')

    if bool_cols:
        if verbose:
            print('\nKEY VALUES FOR BOOLEAN FEATURES:')
        for column in bool_cols:
            show_key_vals(column, df, 'Boolean')

    if datetime_cols:
        if verbose:
            print('\nKEY VALUES FOR DATETIME FEATURES:')
        for column in datetime_cols:
            show_key_vals(column, df, 'Datetime')

    if cat_cols:
        if verbose:
            print('\nKEY VALUES FOR CATEGORICAL FEATURES:')
        for column in cat_cols:
            show_key_vals(column, df, 'Categorical')

    if num_cols:
        if verbose:
            print('\nKEY VALUES FOR NUMERICAL FEATURES:')
        for column in num_cols:
            show_key_vals(column, df, 'Numerical')

    if verbose:
        print('-' * 50)
        print('Feature-level analysis complete.')
        print('-' * 50)


def _impute_core(df: pd.DataFrame, verbose: bool = True, skip_warnings: bool = False) -> pd.DataFrame:
    """
    Core function to perform simple imputation for missing values at the feature level.
    Supports mean, median, mode, constant value, random sampling, and time-series imputation methods
    based on feature type and data characteristics.

    Args:
        df (pd.DataFrame): Input DataFrame containing features to impute.
        verbose (bool): Whether to display detailed guidance and explanations. Defaults to True.
        skip_warnings (bool): Whether to skip missingness threshold warnings. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with imputed values where specified by user.

    Raises:
        ValueError: If an invalid imputation method is selected
    """
    def plot_dist_comps(original: pd.Series, imputed: pd.Series, feature_name: str) -> None:
        """
        This helper function creates and displays a side-by-side comparison of pre- and post- imputation distributions.

        Args:
            original: Original series with missing values
            imputed: Series after imputation
            feature_name: Name of the feature being visualized
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

            # Pre-imputation distribution
            sns.histplot(data=original.dropna(), ax=ax1)
            ax1.set_title(f'Pre-imputation Distribution of "{feature_name}"')
            ax1.set_xlabel(feature_name)

            # Post-imputation distribution
            sns.histplot(data=imputed, ax=ax2)
            ax2.set_title(f'Post-imputation Distribution of "{feature_name}"')
            ax2.set_xlabel(feature_name)

            plt.tight_layout()
            plt.show()
            plt.close()

        except Exception as exc:
            print(f'Could not create distribution visualizations: {str(exc)}')
            if plt.get_fignums():
                plt.close('all')

    def low_variance_check(series: pd.Series, threshold: float = 0.01) -> bool:
        """
        This helper function checks if a series has a variance below the specified threshold.

        Args:
            series: Numerical series to check
            threshold: Variance threshold, default 0.01

        Returns:
            bool: True if variance is below threshold
        """
        return series.var() < threshold

    def outlier_cnt_check(series: pd.Series, threshold: float = 0.1) -> bool:
        """
        This helper function checks if a series has more outliers than the specified threshold.

        Args:
            series: Numerical series to check
            threshold: Maximum allowable proportion of outliers, default 0.1

        Returns:
            bool: True if proportion of outliers exceeds threshold
        """
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = series < (q1 - 1.5 * iqr)
        upper_bound = series > (q3 + 1.5 * iqr)
        outlier_mask = pd.Series(lower_bound | upper_bound, index=series.index)
        return (outlier_mask.sum() / len(series)) > threshold

    def high_corr_check(df: pd.DataFrame, columns: list[str], threshold: float = 0.9) -> list[tuple]:
        """
        This helper function identifies pairs of highly correlated numerical features.

        Args:
            df: DataFrame containing the features
            columns: List of numerical column names to check
            threshold: Correlation threshold, default 0.9

        Returns:
            list[tuple]: List of tuples containing (col1, col2, correlation)
        """
        if len(columns) < 2:
            return []

        corr_matrix = df[columns].corr()
        high_corr_pairs = []

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append((
                        columns[i],
                        columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        return high_corr_pairs

    if verbose:
        print('-' * 50)  # Visual separator
        print('Beginning imputation process.')
        print('-' * 50)  # Visual separator

    # Check if there are no missing values - if no missing values exist, skip imputation
    if not df.isnull().any().any():
        print('WARNING: No missing values are present in dataset. Skipping imputation. Dataframe was not modified.')
        return df

    # Identify initial feature types using list comprehensions
    cat_cols = [column for column in df.columns if pd.api.types.is_object_dtype(df[column])
                or isinstance(df[column].dtype, type(pd.Categorical.dtype))]

    num_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]

    # Check for presence of datetime columns and index
    datetime_cols = []

    for column in df.columns:
        # Check if column is already datetime type
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            datetime_cols.append(column)
        # For string columns, try to parse as datetime
        elif pd.api.types.is_object_dtype(df[column]):
            try:
                # Try to parse first non-null value
                first_valid = df[column].dropna().iloc[0]
                pd.to_datetime(first_valid)
                datetime_cols.append(column)
            except (ValueError, IndexError):
                continue

    is_datetime_index = pd.api.types.is_datetime64_any_dtype(df.index)
    has_datetime = bool(datetime_cols or is_datetime_index)

    # Convert detected datetime string columns to datetime type
    if datetime_cols:
        for col in datetime_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except ValueError:
                    datetime_cols.remove(col)  # Remove if conversion fails

    # Top-level feature classification information if Verbose is True
    if verbose:
        print('Initial feature type identification:')
        print(f'Categorical features: {", ".join(cat_cols) if cat_cols else "None"}')
        print(f'Numerical features: {", ".join(num_cols) if num_cols else "None"}')
        if has_datetime:
            print('ALERT: Datetime elements detected in dataset.')
        print('-' * 50)

    # Check for numeric features that might actually be categorical in function/intent
    true_nums = num_cols.copy()  # Start with all numeric columns
    for column in num_cols:
        unique_vals = sorted(df[column].dropna().unique())

        # We suspect that any all-integer column with five or fewer unique values is actually categorical
        if len(unique_vals) <= 5 and all(float(x).is_integer() for x in unique_vals if pd.notna(x)):
            if verbose:
                print(f'Feature "{column}" has only {len(unique_vals)} unique integer values: '
                      f'{[int(val) for val in unique_vals]}')
                print('ALERT: This could be a categorical feature encoded as numbers, e.g. a 1/0 representation of '
                      'Yes/No values.')

            # Ask the user to assess and reply
            user_cat = input(f'\nShould "{column}" actually be treated as categorical? (Y/N): ')

            # If user agrees, recast the feature to string-type and append to list of categorical features
            if user_cat.lower() == 'y':
                df[column] = df[column].apply(lambda value: str(value) if pd.notna(value) else value)
                cat_cols.append(column)
                true_nums.remove(column)  # Remove from numeric if identified as categorical
                if verbose:
                    print(f'Converted numerical feature "{column}" to categorical type.')
                    print('-' * 50)

    # Check input dataframe's numerical features for potential data quality problems
    qual_probs = {}
    for column in true_nums:
        problems = []

        if low_variance_check(df[column]):
            problems.append('Near-zero variance')

        if outlier_cnt_check(df[column]):
            problems.append('High outlier count')

        if abs(df[column].skew()) > 2:
            problems.append('High skewness')

        if problems:
            qual_probs[column] = problems

    high_corr_pairs = high_corr_check(df, true_nums)

    # Print data quality warnings if warnings are not skipped
    if not skip_warnings and (qual_probs or high_corr_pairs):
        print('-' * 50)
        print('WARNING: POTENTIAL DATA QUALITY ISSUES DETECTED!')
        print('-' * 50)

        if qual_probs:
            print('Feature-specific issues:')
            for col, issues in qual_probs.items():
                print(f'- {col}: {", ".join(issues)}')

        if high_corr_pairs:
            print('\nHighly correlated feature pairs:')
            for col1, col2, corr in high_corr_pairs:
                print(f'- {col1} & {col2}: correlation = {corr:.2f}')

        if verbose:
            print('\nNOTE: These issues may affect imputation quality:')
            print('- Near-zero variance: Feature may not be usefully informative.')
            print('- High outlier count: Mean-substitution imputation may be inappropriate.')
            print('- High correlation: Features may contain redundant information.')
            print('- High skewness: Transforming data may be necessary prior to imputation.')
        print('-' * 50)

    # Final feature classification info if Verbose is True
    if verbose:
        print('Final feature type classification:')
        print(f'Categorical features: {", ".join(cat_cols) if cat_cols else "None"}')
        print(f'Numerical features: {", ".join(true_nums) if true_nums else "None"}')
        print('-' * 50)

    # For each feature, calculate missingness statistics
    if verbose:
        print('Count and rate of missingness for each feature:')

    # Calculate missingness values
    missingness_vals = {}
    for column in df.columns:
        missing_cnt = df[column].isnull().sum()
        missing_rate = (missing_cnt / len(df) * 100).round(2)
        missingness_vals[column] = {'count': missing_cnt, 'rate': missing_rate}

    if verbose:
        for column, vals in missingness_vals.items():
            print(f'Feature {column} has {vals["count"]} missing value(s). ({vals["rate"]}% missing)')

    # Check if this is time series data if datetime elements exist
    is_timeseries = False  # Assume the data are not timeseries as the naive case
    if has_datetime:
        if verbose:
            print('\nALERT: Datetime elements detected in the dataset.')
            print('Time series data may benefit from forward/backward fill imputation.')

        ts_response = input('Are these time series data? (Y/N): ').lower()
        is_timeseries = ts_response == 'y'

        if verbose and is_timeseries:
            print('\nTime series imputation methods available in TADPREP:')
            print('- Forward fill: Carries the last valid observation forward')
            print('- Backward fill: Carries the next valid observation backward')
            print('-' * 50)

    if not skip_warnings and verbose:
        print('\nWARNING: Imputing missing values for features with a missing rate over 10% is not recommended '
              'due to potential bias introduction.')

    # Build list of good candidates for imputation based on missingness and data quality
    if not skip_warnings:
        imp_candidates = [
            key for key, value in missingness_vals.items()
            # Check for missingness problems
            if 0 < value['rate'] <= 10]

    else:
        imp_candidates = [key for key, value in missingness_vals.items() if 0 < value['rate']]

    # We only walk through the imputation missingness-rate guidance if warnings aren't skipped
    if not skip_warnings:
        if imp_candidates and verbose:
            print('Based on missingness rates and data quality assessments, the following features are good '
                  'candidates for imputation:')

            for key in imp_candidates:
                print(f'- {key}: {missingness_vals[key]["rate"]}% missing')

        elif verbose:
            print('No features fall within the recommended criteria for imputation.')
            print('WARNING: Statistical best practices indicate you should not perform imputation.')

    # Store imputation records for summary
    imp_records = []

    # Ask if user wants to override the recommendation
    while True:
        try:
            user_override = input('\nDo you wish to:\n'
                                  '1. Impute only for recommended features (<= 10% missing)\n'
                                  '2. Override the warning and consider all features with missing values\n'
                                  '3. Skip imputation\n'
                                  'Enter choice: ').strip()

            # Catch bad input
            if user_override not in ['1', '2', '3']:
                raise ValueError('Please enter 1, 2, or 3.')

            if user_override == '3':
                print('Skipping imputation. No changes made to dataset.')
                return df

            # Build list of features to be imputed
            imp_features = (imp_candidates if user_override == '1'
                            else [key for key, value in missingness_vals.items() if value['count'] > 0])

            # If user wants to follow missingness guidelines and no good candidates exist, skip imputation
            if not imp_features:
                print('No features available for imputation given user input. Skipping imputation.')
                return df

            # Exit the loop once the imp_features list is built
            break

        # Catch all other input errors
        except ValueError as exc:
            print(f'Invalid input: {exc}')
            continue  # Restart loop

    # Offer methodology explanations if desired by user if Verbose is True
    if verbose:
        print('\nWARNING: TADPREP supports the following imputation methods:')
        print('- Mean/Median/Mode imputation')
        print('- Constant value imputation')
        print('- Random sampling from non-null within-feature values')

        if is_timeseries:
            print('- Forward/backward fill (for time series data)')

        print('\nFor more sophisticated methods (e.g. imputation-by-modeling), skip this step and write '
              'your own imputation code.')

        user_impute_refresh = input('Do you want to see a brief refresher on these imputation methods? (Y/N): ')
        if user_impute_refresh.lower() == 'y':
            print('\nImputation Methods Overview:')
            print('Statistical Methods:')
            print('- Mean: Best for normally-distributed data. Theoretically practical. Sensitive to outliers.')
            print('- Median: Better for skewed numerical data. Robust to outliers.')
            print('- Mode: Useful for categorical and fully-discrete numerical data.')
            print('\nOther Methods:')
            print('- Constant: Replaces missing values with a specified value. Good when default values exist.')
            print('- Random Sampling: Maintains feature distribution by sampling from non-null values.')

            if is_timeseries:
                print('\nTime Series Imputation Methods:')
                print('- Forward Fill: Carries last valid value forward. Good for continuing trends.')
                print('- Backward Fill: Carries next valid value backward. Good for establishing history.')

    # Begin imputation at feature level
    for feature in imp_features:
        print(f'\nProcessing feature "{feature}"...')
        print('-' * 50)
        if verbose:
            print(f'- Datatype: {df[feature].dtype}')
            print(f'- Missingness rate: {missingness_vals[feature]["rate"]}%')

            # Show pre-imputation distribution if numerical
            if feature in true_nums:
                print('\nPre-imputation distribution:')
                print(df[feature].describe().round(2))

                # Offer visualization
                if input('\nWould you like to view a plot of the current feature distribution? (Y/N): ').lower() == 'y':
                    try:
                        plt.figure(figsize=(12, 8))
                        sns.histplot(data=df, x=feature)
                        plt.title(f'Distribution of {feature} (Pre-imputation)')
                        plt.show()
                        plt.close()

                    except Exception as exc:
                        print(f'Could not create visualization: {exc}')
                        if plt.get_fignums():
                            plt.close('all')

        # Build list of available/valid imputation methods based on feature characteristics
        val_methods = ['Skip imputation for this feature']

        if feature in true_nums:
            val_methods = ['Mean', 'Median', 'Mode', 'Constant', 'Random Sample'] + val_methods

        else:
            val_methods = ['Mode', 'Constant', 'Random Sample'] + val_methods

        if is_timeseries:
            val_methods = ['Forward Fill', 'Backward Fill'] + val_methods

        # Prompt user to select an imputation method
        while True:
            method_items = [f'{idx}. {method}' for idx, method in enumerate(val_methods, 1)]
            method_prompt = f'\nChoose imputation method:\n{"\n".join(method_items)}\nEnter choice: '
            user_imp_choice = input(method_prompt)

            try:
                method_idx = int(user_imp_choice) - 1
                if 0 <= method_idx < len(val_methods):
                    imp_method = val_methods[method_idx]
                    break
                else:
                    print('Invalid input. Enter a valid number.')
            except ValueError:
                print('Invalid input. Enter a valid number.')

        # Exit current feature if user chooses to skip
        if imp_method == 'Skip imputation for this feature':
            if verbose:
                print(f'Skipping imputation for feature: "{feature}"')
            continue

        # Begin actual imputation process
        try:
            # Store original values for distribution comparison
            original_values = df[feature].copy()
            feature_missing_cnt = missingness_vals[feature]['count']

            # Calculate imputation values based on method selection
            if imp_method == 'Mean':
                imp_val = df[feature].mean()
                df[feature] = df[feature].fillna(imp_val)
                method_desc = f'Mean value ({imp_val:.4f})'

            elif imp_method == 'Median':
                imp_val = df[feature].median()
                df[feature] = df[feature].fillna(imp_val)
                method_desc = f'Median value ({imp_val:.4f})'

            elif imp_method == 'Mode':
                mode_vals = df[feature].mode()
                if len(mode_vals) == 0:
                    print(f'No mode value exists for feature {feature}. Skipping imputation.')
                    continue

                imp_val = mode_vals[0]  # Select first mode

                df[feature] = df[feature].fillna(imp_val)
                method_desc = f'Mode value ({imp_val})'

            elif imp_method == 'Constant':
                while True:
                    try:
                        if feature in true_nums:
                            user_input = input('Enter constant numerical value for imputation: ')
                            imp_val = float(user_input)

                        else:
                            # For categorical features, show existing categories
                            existing_cats = df[feature].dropna().unique()
                            print('\nExisting categories in feature:')
                            for idx, cat in enumerate(existing_cats, 1):
                                print(f'{idx}. {cat}')

                            user_input = input('\nEnter the number of the category to use for imputation: ').strip()

                            # Validate category selection
                            try:
                                cat_idx = int(user_input) - 1
                                if 0 <= cat_idx < len(existing_cats):
                                    imp_val = existing_cats[cat_idx]

                                else:
                                    raise ValueError('Selected category number is out of range')

                            except ValueError:
                                raise ValueError('Please enter a valid category number')

                        break  # Exit loop if input is valid

                    except ValueError as exc:
                        print(f'Invalid input: {str(exc)}. Please try again.')

                df[feature] = df[feature].fillna(imp_val)
                method_desc = f'Constant value ({imp_val})'

            elif imp_method == 'Random Sample':
                # Get non-null values to sample from
                valid_values = df[feature].dropna()

                if len(valid_values) == 0:
                    print('No valid values to sample from. Skipping imputation.')
                    continue

                imp_vals = valid_values.sample(n=feature_missing_cnt, replace=True)  # Sample with replacement

                df.loc[df[feature].isna(), feature] = imp_vals.values  # Fill NaN values with sampled values

                method_desc = 'Random sampling from non-null values'

            elif imp_method == 'Forward Fill':
                df[feature] = df[feature].ffill()
                method_desc = 'Forward fill'

            elif imp_method == 'Backward Fill':
                df[feature] = df[feature].bfill()
                method_desc = 'Backward fill'

            # Add record for summary table
            imp_records.append({
                'feature': feature,
                'count': feature_missing_cnt,
                'method': imp_method,
                'description': method_desc
            })

            if verbose:
                print(f'\nSuccessfully imputed {feature_missing_cnt} missing values using {method_desc}.')

                # Show post-imputation distribution for numerical features
                if feature in true_nums:
                    print('\nPost-imputation distribution:')
                    print(df[feature].describe().round(2))

                    if input('\nWould you like to see comparison plots of the pre- and post-imputation feature '
                             'distribution? (Y/N): ').lower() == 'y':
                        plot_dist_comps(original_values, df[feature], feature)

        except Exception as exc:
            print(f'Error during imputation for feature {feature}: {exc}')
            print('Skipping imputation for this feature.')
            continue

    # Print imputation summary if any features were imputed
    if verbose:
        if imp_records:
            print('\nIMPUTATION SUMMARY:')
            print('-' * 50)

            for record in imp_records:
                print(f'Feature: {record["feature"]}')
                print(f'- {record["count"]} values imputed')
                print(f'- Method: {record["method"]}')
                print(f'- Description: {record["description"]}')
                print('-' * 50)

        print('Imputation complete. Returning modified dataframe.')
        print('-' * 50)

    return df  # Return the modified dataframe with imputed values


def _encode_core(
        df: pd.DataFrame,
        features_to_encode: list[str] | None = None,
        verbose: bool = True,
        skip_warnings: bool = False,
        preserve_features: bool = False
) -> pd.DataFrame:
    """
    Core function to encode categorical features using one-hot or dummy encoding, as specified by user.
    The function also looks for false-numeric features (e.g. 1/0 representations of 'Yes'/'No') and asks
    if they should be treated as categorical and therefore be candidates for encoding.

    Args:
        df : pd.DataFrame
            Input DataFrame containing features to encode.
        features_to_encode : list[str] | None, default=None
            Optional list of features to encode. If None, function will help identify categorical features.
        verbose : bool, default=True
            Whether to display detailed guidance and explanations.
        skip_warnings : bool, default=False
            Whether to skip best-practice-related warnings about null values and cardinality issues.
        preserve_features : bool, default=False
            Whether to keep original features in the DataFrame alongside encoded ones.

    Returns:
        pd.DataFrame: DataFrame with encoded values as specified by user.
    """

    def clean_col_name(name: str) -> str:
        """
        This helper function cleans column names to ensure they're valid Python identifiers.
        It replaces spaces and special characters with underscores.
        """
        # Replace spaces, special characters, and any non-alphanumeric character with underscores
        clean_name = re.sub(r'\W', '_', str(name))

        # Ensure name doesn't start with a number
        if clean_name and clean_name[0].isdigit():
            clean_name = 'feature_' + clean_name

        # Ensure no double underscores exist
        while '__' in clean_name:
            clean_name = clean_name.replace('__', '_')

        # Remove any trailing underscores
        clean_name = clean_name.rstrip('_')

        # Return cleaned feature name
        return clean_name

    if verbose:
        print('-' * 50)  # Visual separator
        print('Beginning encoding process.')
        print('-' * 50)  # Visual separator

    if verbose and features_to_encode:
        print('Encoding only features in user-provided list:')
        print(features_to_encode)
        print('-' * 50)  # Visual separator

    # If no features are specified in the to_encode list, identify potential categorical features
    if features_to_encode is None:
        # Identify obvious categorical features (i.e. those which are object or categorical in data-type)
        cat_cols = [column for column in df.columns
                    if pd.api.types.is_object_dtype(df[column]) or
                    isinstance(df[column].dtype, type(pd.Categorical.dtype))]

        # Check for numeric features which might actually be categorical in function/role
        numeric_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]

        for column in numeric_cols:
            # Skip columns with all NaN values
            if df[column].isna().all():
                continue

            unique_vals = sorted(df[column].dropna().unique())  # Get sorted unique values excluding nulls

            # We suspect that any all-integer column with five or fewer unique values is actually categorical
            if len(unique_vals) <= 5 and all(float(x).is_integer() for x in unique_vals if pd.notna(x)):

                # Ask user to assess and reply
                if verbose:
                    print(f'ALERT: Feature "{column}" has only {len(unique_vals)} unique integer values: '
                          f'{[int(val) for val in unique_vals]}')
                    print('This could be a categorical feature encoded as numbers, e.g. a 1/0 representation of '
                          'Yes/No values.')

                user_cat = input(f'Should "{column}" actually be treated as categorical? (Y/N): ')

                if user_cat.lower() == 'y':  # If the user says yes
                    df[column] = df[column].astype(str)  # Convert to string type for encoding
                    cat_cols.append(column)  # Add that feature to list of categorical features
                    if verbose:
                        print('-' * 50)
                        print(f'Converted numerical feature "{column}" to categorical type.')
                        print(f'"{column}" is now a candidate for encoding.')
                        print('-' * 50)

        final_cat_cols = cat_cols

    else:
        final_cat_cols = features_to_encode

    # Print verbose reminder about not encoding target features
    if features_to_encode is None and verbose:
        print('REMINDER: Target features (prediction targets) should not be encoded.')
        print('If any of your features are known targets, do not encode them.')

    # Validate that all specified features exist in the dataframe
    missing_cols = [column for column in final_cat_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f'Features not found in dataframe: {missing_cols}')

    if not final_cat_cols:
        if verbose:
            print('No features were identified as candidates for encoding.')
            print('-' * 50)  # Visual separator
        print('Skipping encoding. Dataset was not modified.')
        return df

    # For memory efficiency, I'll aim to modify the dataframe in place
    columns_to_drop = []  # Track original features to be dropped after encoding
    encoded_features = []  # Track features and their encoding methods for reporting
    encoding_performed = False  # Track if any encoding was performed

    # Offer explanation of encoding methods if in verbose mode
    if verbose:
        user_encode_refresh = input('\nWould you like to see an explanation of encoding methods? (Y/N): ')
        if user_encode_refresh.lower() == 'y':
            print('\nOverview of One-Hot vs. Dummy Encoding:'
                  '\nOne-Hot Encoding: '
                  '\n- Creates a new binary column for every unique category.'
                  '\n- No information is lost, which preserves interpretability, but more features are created.'
                  '\n- This method is preferred for non-linear models like decision trees.'
                  '\n'
                  '\nDummy Encoding:'
                  '\n- Creates n-1 binary columns given n categories in the feature.'
                  '\n- One category becomes the reference class, and is represented by all zeros.'
                  '\n- Dummy encoding is preferred for linear models, as it avoids perfect multi-collinearity.'
                  '\n- This method is more computationally- and space-efficient, but is less interpretable.')

            # Explain missing value handling options
            print('\nMissing Value Handling Options:'
                  '\n- Ignore: Leave missing values as NaN in encoded columns'
                  '\n- Treat as category: Create a separate indicator column for missing values'
                  '\n- Drop instances: Remove instances with missing values before encoding is performed')

    if verbose and not features_to_encode:
        print('\nFinal candidate features for encoding are:')
        print(final_cat_cols)
        print()

    # Process each feature in our list
    for column in final_cat_cols:
        if verbose:
            print('-' * 50)  # Visual separator
        print(f'Processing feature: "{column}"')
        if verbose:
            print('-' * 50)  # Visual separator

        # Check for nulls if warnings aren't suppressed
        null_count = df[column].isnull().sum()
        null_detected = null_count > 0

        # Define default missing value strategy, which is to ignore them and leave them as NaN values
        missing_strat = 'ignore'

        if null_detected and not skip_warnings:
            # Check to see if user wants to proceed
            print(f'Warning: "{column}" contains {null_count} null values.')

            if verbose:
                print('\nHow would you like to handle missing values?')
                print('1. Ignore (leave as NaN in encoded columns)')
                print('2. Treat as separate category')
                print('3. Drop rows with missing values')
                print('4. Skip encoding this feature')

                # Have user define missing values handling strategy
                while True:
                    choice = input('Enter choice (1-4): ')
                    if choice == '1':
                        missing_strat = 'ignore'
                        break

                    elif choice == '2':
                        missing_strat = 'category'
                        break

                    elif choice == '3':
                        missing_strat = 'drop'
                        break

                    # Implement feature skip
                    elif choice == '4':
                        print(f'Skipping encoding for feature "{column}".')
                        continue

                    # Handle bad user input
                    else:
                        print('Invalid choice. Please enter 1, 2, 3, or 4.')

            # Default to 'ignore' strategy in non-verbose mode
            else:
                missing_strat = 'ignore'
                if input('Continue encoding this feature? (Y/N): ').lower() != 'y':
                    continue

        # Perform cardinality checks if warnings aren't suppressed
        unique_count = df[column].nunique()
        if not skip_warnings:
            # Check for high cardinality
            if unique_count > 20:
                print(f'WARNING: "{column}" has high cardinality ({unique_count} unique values)')
                if verbose:
                    print('Consider using dimensionality reduction techniques instead of encoding this feature.')
                    print('Encoding high-cardinality features can lead to issues with the curse of dimensionality.')
                # Check to see if user wants to proceed
                if input('Continue encoding this feature? (Y/N): ').lower() != 'y':
                    continue

            # Skip constant features (those with only one unique value)
            elif unique_count <= 1:
                if verbose:
                    print(f'WARNING: Feature "{column}" has only one unique value and thus provides no '
                          f'meaningful information.')
                print(f'Skipping encoding for "{column}".')
                continue

            # Check for low-frequency categories
            value_counts = df[column].value_counts()
            low_freq_cats = value_counts[value_counts < 10]  # Categories with fewer than 10 instances
            if not low_freq_cats.empty:
                if verbose:
                    print(f'\nWARNING: Found {len(low_freq_cats)} categories with fewer than 10 instances:')
                    print(low_freq_cats)
                print('You should consider grouping rare categories before encoding.')
                if input('Continue encoding this feature? (Y/N): ').lower() != 'y':
                    continue

        if verbose:
            # Show current value distribution
            print(f'\nCurrent values in "{column}":')
            print(df[column].value_counts(dropna=False))  # Include NA in count

            # Offer to show user a distribution plot
            if input('\nWould you like to see a plot of the value distribution? (Y/N): ').lower() == 'y':
                try:
                    plt.figure(figsize=(12, 10))
                    value_counts = df[column].value_counts(dropna=False)
                    plt.bar(range(len(value_counts)), value_counts.values)
                    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
                    plt.title(f'Distribution of "{column}"')
                    plt.xlabel(column)
                    plt.ylabel('Count')
                    plt.tight_layout()
                    plt.show()
                    plt.close()  # Explicitly close the figure

                # Catch plotting errors
                except Exception as exc:
                    print(f'Error creating plot: {str(exc)}')
                    if plt.get_fignums():  # If any figures are open
                        plt.close('all')  # Close all figures

        # Let user customize the encoding prefix for a feature
        prefix = column  # Default prefix is the column name
        if verbose:
            user_prefix = input(f'\nWould you like to use a custom prefix for the encoded columns? (Y/N): ')
            if user_prefix.lower() == 'y':
                prefix_value = input(f'Enter custom prefix (default: "{column}"): ').strip()
                if not prefix_value:  # If user enters empty string, use default
                    prefix_value = column
                prefix = prefix_value  # Set prefix to user-provided value

        # Clean the prefix regardless of whether it was customized
        prefix = clean_col_name(prefix)

        if verbose:
            # Fetch encoding method preference from user
            print('\nTADPREP-Supported Encoding Methods:')

        # Show encoding options
        print('1. One-Hot Encoding (builds a new column for each category)')
        print('2. Dummy Encoding (builds n-1 columns, drops one category)')
        print('3. Skip encoding for this feature')

        # Fetch user encoding method choice
        while True:
            method = input('Select encoding method or skip encoding (Enter 1, 2 or 3): ')
            if method in ['1', '2', '3']:
                break
            print('Invalid choice. Please enter 1, 2, or 3.')

        try:
            # Skip encoding if user enters the skip option
            if method == '3':
                if verbose:
                    print(f'\nSkipping encoding for feature "{column}".')
                continue

            # Determine feature to encode based on missing value strategy
            feature_data = df[column]
            temp_df = df

            # Handle missing values according to strategy
            if null_detected:
                if missing_strat == 'drop':
                    if verbose:
                        print(f'Dropping {null_count} rows with missing values for encoding.')
                    # Create a temporary subset without null values
                    temp_df = df.dropna(subset=[column])
                    feature_data = temp_df[column]

            # Apply selected encoding method
            if method == '1':  # One-hot encoding
                # Set parameters based on missing value strategy
                dummy_na = missing_strat == 'category'

                # Create a temporary version for encoding
                encoded = pd.get_dummies(
                    feature_data,
                    prefix=prefix,
                    prefix_sep='_',
                    dummy_na=dummy_na
                )

                # Sanitize column names
                encoded.columns = [clean_col_name(col) for col in encoded.columns]

                # Reindex to match original dataframe if we used a temporary subset
                if missing_strat == 'drop':
                    # Use reindex to create a DataFrame with same index as original, filling missing values with 0
                    encoded = encoded.reindex(df.index, fill_value=0)

                # Apply encoding directly to dataframe
                df = pd.concat([df, encoded], axis=1)
                columns_to_drop.append(column)
                encoded_features.append(f'{column} (One-Hot)')  # or Dummy equivalent
                encoding_performed = True

                # Note successful encoding action
                if verbose:
                    print(f'\nSuccessfully one-hot encoded "{column}" with prefix "{prefix}".')
                    print(f'Created {len(encoded.columns)} new columns.')

            else:
                # Get unique values for reference category selection
                unique_vals = df[column].dropna().unique()

                # Check if there are any non-null values to encode
                if len(unique_vals) == 0:
                    print(f'Feature "{column}" has only null values. Skipping encoding for this feature.')
                    continue

                # Check for single-value features
                if len(unique_vals) <= 1:
                    print(f'Feature "{column}" has too few unique values for dummy encoding. Skipping feature.')
                    continue

                # Let user select reference category
                if verbose:
                    print('\nSelect reference category (the category that will be dropped):')
                    for idx, val in enumerate(unique_vals, 1):
                        print(f'{idx}. {val}')

                    while True:
                        try:
                            choice = input(
                                f'Enter category number (1-{len(unique_vals)}) or press Enter for default: ')
                            if not choice:  # Use first category as default
                                reference_cat = unique_vals[0]
                                break

                            # Convert user input to idx value
                            idx = int(choice) - 1
                            if 0 <= idx < len(unique_vals):
                                reference_cat = unique_vals[idx]
                                break
                            else:
                                print(f'Please enter a number between 1 and {len(unique_vals)}.')

                        # Catch bad user input
                        except ValueError:
                            print('Invalid input. Please enter a number.')
                else:
                    # Use first category as default reference
                    reference_cat = unique_vals[0]

                # Set parameters based on missing value strategy
                dummy_na = missing_strat == 'category'

                # For the reference category, I need to ensure proper categorical order
                if reference_cat is not None:
                    # Create ordered categorical type with the reference first
                    cat_order = [reference_cat] + [cat for cat in unique_vals if cat != reference_cat]
                    feature_data = pd.Categorical(feature_data, categories=cat_order)

                # Create dummy variables
                encoded = pd.get_dummies(
                    feature_data,
                    prefix=prefix,
                    prefix_sep='_',
                    dummy_na=dummy_na,
                    drop_first=True
                )

                # Clean column names with helper function
                encoded.columns = [clean_col_name(col) for col in encoded.columns]

                # Reindex to match original dataframe if I had to use a temporary subset
                if missing_strat == 'drop':
                    # Use reindex to create a DataFrame with same index as original, filling missing values with 0
                    encoded = encoded.reindex(df.index, fill_value=0)

                # Apply encoding directly to dataframe
                df = pd.concat([df, encoded], axis=1)
                columns_to_drop.append(column)
                encoded_features.append(f'{column} (Dummy, reference: "{reference_cat}")')
                encoding_performed = True

                # Note successful encoding action
                if verbose:
                    print(f'\nSuccessfully dummy encoded "{column}" with prefix "{prefix}".')
                    print(f'Using "{reference_cat}" as reference category.')
                    print(f'Created {len(encoded.columns)} new columns.')

        # Catch all other errors
        except Exception as exc:
            print(f'Error encoding feature "{column}": {str(exc)}')
            continue

    # Drop original columns if any encoding was performed and preserve_features is False
    if encoding_performed and not preserve_features:
        df = df.drop(columns=columns_to_drop)  # Remove original columns

    # Print summary of encoding if in verbose mode
    if encoding_performed and verbose:
        print('\nENCODING SUMMARY:')
        for feature in encoded_features:
            print(f'- {feature}')

        if preserve_features:
            print('\nNOTE: Original features were preserved alongside encoded columns.')

    if verbose:
        print('-' * 50)  # Visual separator
        print('Encoding complete. Returning modified dataframe.')
        print('-' * 50)  # Visual separator

    # Return the modified dataframe with encoded values
    return df


def _scale_core(
        df: pd.DataFrame,
        features_to_scale: list[str] | None = None,
        verbose: bool = True,
        skip_warnings: bool = False,
        preserve_features: bool = False
) -> pd.DataFrame:
    """
    Core function to scale numerical features using standard, robust, or minmax scaling methods.

    Args:
        df : pd.DataFrame
            Input DataFrame containing features to scale.
        features_to_scale : list[str] | None, default=None
            Optional list of features to scale. If None, function will help identify numerical features.
        verbose : bool, default=True
            Whether to display detailed guidance and explanations.
        skip_warnings : bool, default=False
            Whether to skip all best-practice-related warnings about nulls, outliers, etc.
        preserve_features : bool, default=False
            Whether to preserve original features by creating new columns with scaled values.
            When True, new columns are created with the naming pattern '{original_column}_scaled'.
            If a column with that name already exists, a numeric suffix is added: '{original_column}_scaled_1'.

    Returns:
        pd.DataFrame: DataFrame with scaled values as specified by user.

    Raises:
        ValueError: If scaling fails due to data structure issues.
    """

    def plot_comp(original: pd.Series, scaled: pd.Series, feature_name: str) -> None:
        """
        This helper function creates and displays a side-by-side comparison of pre- and post-scaling distributions.

        Args:
            original: Original series before scaling
            scaled: Series after scaling
            feature_name: Name of the feature being visualized
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

            # Pre-scaling distribution
            sns.histplot(data=original.dropna(), ax=ax1)
            ax1.set_title(f'Pre-scaling Distribution of "{feature_name}"')
            ax1.set_xlabel(feature_name)

            # Post-scaling distribution
            sns.histplot(data=scaled, ax=ax2)
            ax2.set_title(f'Post-scaling Distribution of "{feature_name}"')
            ax2.set_xlabel(f'{feature_name} (scaled)')

            plt.tight_layout()
            plt.show()
            plt.close()

        except Exception as exc:
            print(f'Could not create distribution visualizations: {str(exc)}')
            if plt.get_fignums():
                plt.close('all')

    def handle_inf(series: pd.Series, method: str, value: float = None) -> pd.Series:
        """
        This helper function replaces infinite values in a series based on user-specified method.

        Args:
            series: Series to handle infinities in
            method: Method to use ('nan', 'min', 'max', or 'value')
            value: Custom value to use if method is 'value'

        Returns:
            Series with infinite values properly handled
        """
        # Make a copy to avoid modifying the original
        result = series.copy()

        # Build a mask of infinite values
        inf_mask = np.isinf(result)

        # Skip if no infinite values exist
        if not inf_mask.any():
            return result

        # Apply replacement based on method
        if method == 'nan':
            result[inf_mask] = np.nan

        elif method == 'mean':
            # Find mean of non-infinite values
            mean_val = result[~np.isinf(result)].mean()
            # Replace negative infinities with mean value
            result[result == -np.inf] = mean_val
            # Replace positive infinities with mean value
            result[result == np.inf] = mean_val

        elif method == 'min':
            # Find min of non-infinite values
            min_val = result[~np.isinf(result)].min()
            # Replace negative infinities with min value
            result[result == -np.inf] = min_val
            # Replace positive infinities with min value too (unusual but consistent)
            result[result == np.inf] = min_val

        elif method == 'max':
            # Find max of non-infinite values
            max_val = result[~np.isinf(result)].max()
            # Replace positive infinities with max value
            result[result == np.inf] = max_val
            # Replace negative infinities with max value too (unusual but consistent)
            result[result == -np.inf] = max_val

        elif method == 'value':
            # Replace all infinities with specified value
            result[inf_mask] = value

        return result

    if verbose:
        print('-' * 50)
        print('Beginning scaling process.')
        print('-' * 50)

    if verbose and features_to_scale:
        print('Scaling only features in user-provided list:')
        print(features_to_scale)
        print('-' * 50)

    # If no features are specified, identify potential numerical features
    if features_to_scale is None:
        # Identify all numeric features
        numeric_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]

        # Check for numeric features which might actually be categorical
        for column in numeric_cols:
            unique_vals = sorted(df[column].dropna().unique())

            # We suspect any all-integer column with five or fewer unique values is categorical
            if len(unique_vals) <= 5 and all(float(x).is_integer() for x in unique_vals if pd.notna(x)):
                if verbose:
                    print(f'ALERT: Feature "{column}" has only {len(unique_vals)} unique integer values: '
                          f'{[int(val) for val in unique_vals]}')
                    print('This could be a categorical feature encoded as numbers, '
                          'e.g. a 1/0 representation of Yes/No values.')

                user_cat = input(f'Should "{column}" be treated as categorical and excluded from scaling? (Y/N): ')
                if user_cat.lower() == 'y':
                    numeric_cols.remove(column)
                    if verbose:
                        print(f'Excluding "{column}" from scaling consideration.')

        final_numeric_cols = numeric_cols

    else:
        final_numeric_cols = features_to_scale

    # Validate that all specified features exist in the dataframe
    missing_cols = [column for column in final_numeric_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f'Features not found in DataFrame: {missing_cols}')

    if not final_numeric_cols:
        if verbose:
            print('No features were identified as candidates for scaling.')
            print('-' * 50)
        print('Skipping scaling. Dataset was not modified.')
        return df

    if not isinstance(preserve_features, bool):
        print('Invalid value for preserve_original. Using default (False).')
        preserve_features = False

    # Print reminder about not scaling target features
    if features_to_scale is None and verbose:
        print('-' * 50)
        print('REMINDER: Target features (prediction targets) should not be scaled.')
        print('If any of the identified features are targets, do not scale them.')
        print('-' * 50)

    # Track scaled features for reporting
    scaled_features = []

    # Offer explanation of scaling methods if verbose
    if verbose:
        user_scale_refresh = input('Would you like to see an explanation of scaling methods? (Y/N): ')
        if user_scale_refresh.lower() == 'y':
            print('\nOverview of the Standard, Robust, and MinMax Scalers:'
                  '\nStandard Scaler (Z-score normalization):'
                  '\n- Transforms features to have zero mean and unit variance.'
                  '\n- Best choice for comparing measurements in different units.'
                  '\n- Good for methods that assume normally distributed data.'
                  '\n- Not ideal when the data have many outliers.'
                  '\n'
                  '\nRobust Scaler (Uses median and IQR):'
                  '\n- Scales using statistics that are resistant to outliers.'
                  '\n- Great for data where outliers are meaningful.'
                  '\n- Useful for survey data with extreme ratings.'
                  '\n- Good when outliers contain important information.'
                  '\n'
                  '\nMinMax Scaler (scales to a custom range, default 0-1):'
                  '\n- Scales all values to a fixed range (default: between 0 and 1).'
                  '\n- Good for neural networks that expect bounded inputs.'
                  '\n- Works well with sparse data.'
                  '\n- Preserves zero values in sparse data.')

    if verbose and not features_to_scale:
        print('\nFinal candidate features for scaling are:')
        print(final_numeric_cols)

    # Process each feature
    for column in final_numeric_cols:
        print()
        if verbose:
            print('-' * 50)
        print(f'Processing feature: "{column}"')
        if verbose:
            print('-' * 50)

        # Check for nulls if warnings aren't suppressed
        null_count = df[column].isnull().sum()
        if null_count > 0 and not skip_warnings:
            print(f'Warning: "{column}" contains {null_count} null values.')
            print('Scaling with null values present may produce unexpected results.')
            if input('Continue scaling this feature? (Y/N): ').lower() != 'y':
                continue

        # Check for infinite values and offer handling options if warnings aren't suppressed
        inf_count = np.isinf(df[column]).sum()
        if inf_count > 0:
            if not skip_warnings:
                print(f'Warning: "{column}" contains {inf_count} infinite values.')
                print('Scaling with infinite values present may produce unexpected results.')

                # Only ask for handling if continuing with scaling
                if input('Continue scaling this feature? (Y/N): ').lower() != 'y':
                    continue

                # Offer options for handling infinities
                print('\nHow would you like to handle infinite values?')
                print('1. Replace with NaN (missing values)')
                print('2. Replace with mean feature value')
                print('3. Replace with minimum feature value')
                print('4. Replace with maximum feature value')
                print('5. Replace with a custom value')

                while True:
                    inf_choice = input('Enter your choice (1-5): ')
                    if inf_choice == '1':
                        df[column] = handle_inf(df[column], 'nan')
                        print(f'Replaced {inf_count} infinite value(s) with NaN.')
                        break

                    elif inf_choice == '2':
                        df[column] = handle_inf(df[column], 'mean')
                        print(f'Replaced {inf_count} infinite value(s) with mean feature value.')
                        break

                    elif inf_choice == '3':
                        df[column] = handle_inf(df[column], 'min')
                        print(f'Replaced {inf_count} infinite value(s) with minimum feature value.')
                        break

                    elif inf_choice == '4':
                        df[column] = handle_inf(df[column], 'max')
                        print(f'Replaced {inf_count} infinite value(s) with maximum feature value.')
                        break

                    elif inf_choice == '5':
                        try:
                            custom_val = float(input('Enter the value to replace infinities with: '))
                            df[column] = handle_inf(df[column], 'value', custom_val)
                            print(f'Replaced {inf_count} infinite value(s) with {custom_val}.')
                            break

                        except ValueError:
                            print('Invalid input. Please enter a valid number.')

                    else:
                        print('Invalid choice. Please enter 1, 2, 3, 4, or 5.')

        # Skip constant features
        if df[column].nunique() <= 1:
            print(f'ALERT: Skipping "{column}" - this feature has no variance.')
            continue

        # Check for extreme skewness if warnings aren't suppressed
        if not skip_warnings:
            skewness = df[column].skew()
            if abs(skewness) > 2:  # Common threshold for "extreme" skewness
                print(f'Warning: "{column}" is highly skewed (skewness={skewness:.2f}).')
                print('Consider applying a transformation before scaling.')
                if input('Continue scaling this feature? (Y/N): ').lower() != 'y':
                    continue

        # Store original values for visualization and preservation
        original_values = df[column].copy()

        if verbose:
            # Show current distribution statistics
            print(f'\nCurrent statistics for "{column}":')
            print(df[column].describe())

            # Offer distribution plot
            if input('\nWould you like to see a distribution plot of the feature? (Y/N): ').lower() == 'y':
                try:
                    plt.figure(figsize=(12, 8))
                    sns.histplot(data=df, x=column)
                    plt.title(f'Distribution of "{column}"')
                    plt.xlabel(column)
                    plt.ylabel('Count')
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                except Exception as exc:
                    print(f'Error creating plot: {exc}')
                    if plt.get_fignums():
                        plt.close('all')

        # Show scaling options
        print('\nSelect scaling method:')
        print('1. Standard Scaler (Z-score normalization)')
        print('2. Robust Scaler (median and IQR based)')
        print('3. MinMax Scaler (range can be specified)')
        print('4. Skip scaling for this feature')

        while True:
            method = input('Enter choice (1, 2, 3, or 4): ')
            # Exit loop if valid input provided
            if method in ['1', '2', '3', '4']:
                break

            # Catch invalid input
            print('Invalid choice. Please enter 1, 2, 3, or 4.')

        # Skip scaling for a given feature if user decides to do so
        if method == '4':
            if verbose:
                print(f'\nSkipping scaling for feature "{column}".')
            continue

        try:
            # Reshape data for scikit-learn
            reshaped_data = df[column].values.reshape(-1, 1)

            # Target column name (either original or new)
            target_column = column
            if preserve_features:
                target_column = f'{column}_scaled'
                # Check if target column already exists
                counter = 1
                while target_column in df.columns:
                    target_column = f'{column}_scaled_{counter}'
                    counter += 1

            # Apply selected scaling method
            if method == '1':
                scaler = StandardScaler()
                method_name = 'Standard'

            elif method == '2':
                scaler = RobustScaler()
                method_name = 'Robust'

            else:  # MinMax Scaler with custom range option
                feature_range = (0, 1)  # Default range

                custom_range = input('\nDo you want to use a custom MinMax scaler range instead of the default '
                                     '0-1? (Y/N): ').lower() == 'y'

                if custom_range:
                    while True:
                        try:
                            min_val = float(input('Enter minimum value for range: '))
                            max_val = float(input('Enter maximum value for range: '))

                            if min_val >= max_val:
                                print('ERROR: Minimum value must be less than maximum value.')
                                continue

                            feature_range = (min_val, max_val)
                            break
                        except ValueError:
                            print('Invalid input. Please enter valid numbers.')

                scaler = MinMaxScaler(feature_range=feature_range)
                method_name = f'MinMax (range: {feature_range[0]}-{feature_range[1]})'

            # Perform scaling
            scaled_values = scaler.fit_transform(reshaped_data).flatten()

            # Apply to dataframe (either replacing or adding new column)
            if preserve_features:
                df[target_column] = scaled_values
            else:
                df[column] = scaled_values

            scaled_features.append(f'{column} -> {target_column} ({method_name})')

            # Offer before/after visualization comparison
            if verbose:
                print(f'\nSuccessfully scaled "{column}" using {method_name} scaler.')

                if input('\nWould you like to see a comparison plot of the feature before and after scaling? '
                         '(Y/N): ').lower() == 'y':
                    # Get the scaled values from the dataframe
                    scaled_series = df[target_column]
                    # Display comparison
                    plot_comp(original_values, scaled_series, column)

        except Exception as exc:
            print(f'Error scaling feature "{column}": {exc}')
            continue

    # Print summary if features were scaled
    if scaled_features:
        if verbose:
            print('-' * 50)
            print('SCALING SUMMARY:')
            for feature in scaled_features:
                print(f'- {feature}')
            print('-' * 50)

    if verbose:
        print('Scaling complete. Returning modified dataframe.')
        print('-' * 50)

    # Return the modified dataframe with scaled values
    return df


def _prep_df_core(
        df: pd.DataFrame,
        features_to_encode: list[str] | None = None,
        features_to_scale: list[str] | None = None
) -> pd.DataFrame:
    """Core function implementing the TADPREP pipeline with parameter control."""

    def get_bool_param(param_name: str, default: bool = True) -> bool:
        """Get boolean parameter values from user input."""
        while True:
            print(f'\nSet {param_name} parameter:')
            print(f'1. True (default: {default})')
            print('2. False')
            choice = input('Enter choice (1/2) or press Enter to accept default setting: ').strip()

            # Handle the three valid input cases
            if not choice:  # User accepts default
                return default

            elif choice == '1':  # User selects True
                return True

            elif choice == '2':  # User selects False
                return False

            else:  # Invalid input - notify and retry
                print('Invalid choice. Please enter 1, 2, or press Enter.')

    def get_feature_selection(df: pd.DataFrame, operation: str) -> list[str] | None:
        """Get feature selections for encoding and scaling from user input."""
        while True:
            # Present options for feature selection
            print(f'\nFeature Selection for {operation}:')
            print('1. Auto-detect features')
            print('2. Manually select features')
            choice = input('Enter choice (1/2): ').strip()

            if choice == '1':  # Auto-detect
                return None

            elif choice == '2':  # Manual selection
                # Show available features
                print('\nAvailable features:')
                for idx, col in enumerate(df.columns, 1):
                    print(f'{idx}. {col}')

                # Get and validate feature selection
                while True:
                    selections = input(
                        '\nEnter feature numbers (comma-separated) or press Enter to auto-detect: ').strip()

                    if not selections:  # User wants auto-detect
                        return None

                    try:
                        # Convert input to indices then validate
                        indices = [int(idx.strip()) - 1 for idx in selections.split(',')]
                        if not all(0 <= idx < len(df.columns) for idx in indices):
                            print('Error: Invalid feature numbers')
                            continue
                        # Return selected feature names
                        return [df.columns[i] for i in indices]

                    except ValueError:
                        print('Invalid input. Please enter comma-separated numbers.')

            else:  # Invalid choice
                print('Invalid choice. Please enter 1 or 2.')

    # Step 1: File Info
    user_choice = input('\nSTAGE 1: Display file info? (Y/N/Q): ').lower()
    if user_choice == 'y':
        verbose = get_bool_param('verbose')
        _df_info_core(df, verbose=verbose)

    elif user_choice == 'q':
        return df

    # Step 2: Reshape - handles missing values and feature dropping
    user_choice = input('\nSTAGE 2: Run file reshape process? (Y/N/Q): ').lower()

    if user_choice == 'y':
        verbose = get_bool_param('verbose')
        df = _reshape_core(df, verbose=verbose)

    elif user_choice == 'q':
        return df

    # Step 3: Rename and Tag - handles feature renaming and classification
    user_choice = input('\nSTAGE 3: Run feature renaming and tagging process? (Y/N/Q): ').lower()
    if user_choice == 'y':
        verbose = get_bool_param('verbose')
        tag_features = get_bool_param('tag_features', default=False)
        df = _rename_and_tag_core(df, verbose=verbose, tag_features=tag_features)

    elif user_choice == 'q':
        return df

    # Step 4: Feature Stats - calculates and displays feature statistics
    user_choice = input('\nSTAGE 4: Show feature-level information? (Y/N/Q): ').lower()
    if user_choice == 'y':
        verbose = get_bool_param('verbose')
        summary_stats = get_bool_param('summary_stats', default=False)
        _feature_stats_core(df, verbose=verbose, summary_stats=summary_stats)

    elif user_choice == 'q':
        return df

    # Step 5: Impute - handles missing value imputation
    user_choice = input('\nSTAGE 5: Perform imputation? (Y/N/Q): ').lower()
    if user_choice == 'y':
        verbose = get_bool_param('verbose')
        skip_warnings = get_bool_param('skip_warnings', default=False)
        df = _impute_core(df, verbose=verbose, skip_warnings=skip_warnings)

    elif user_choice == 'q':
        return df

    # Step 6: Encode - handles categorical feature encoding
    user_choice = input('\nSTAGE 6: Encode categorical features? (Y/N/Q): ').lower()
    if user_choice == 'y':
        # Use provided features or get interactive selection
        features_to_encode_final = features_to_encode
        if features_to_encode_final is None:
            features_to_encode_final = get_feature_selection(df, 'encoding')
        verbose = get_bool_param('verbose')
        skip_warnings = get_bool_param('skip_warnings', default=False)
        df = _encode_core(
            df,
            features_to_encode=features_to_encode_final,
            verbose=verbose,
            skip_warnings=skip_warnings
        )

    elif user_choice == 'q':
        return df

    # Step 7: Scale - handles numerical feature scaling
    user_choice = input('\nSTAGE 7: Scale numerical features? (Y/N/Q): ').lower()
    if user_choice == 'y':
        # Use provided features or get interactive selection
        features_to_scale_final = features_to_scale
        if features_to_scale_final is None:
            features_to_scale_final = get_feature_selection(df, 'scaling')
        verbose = get_bool_param('verbose')
        skip_warnings = get_bool_param('skip_warnings', default=False)
        df = _scale_core(
            df,
            features_to_scale=features_to_scale_final,
            verbose=verbose,
            skip_warnings=skip_warnings
        )

    elif user_choice == 'q':
        return df

    return df


def _transform_core(
        df: pd.DataFrame,
        features_to_transform: list[str] | None = None,
        verbose: bool = True,
        preserve_features: bool = False,
        skip_warnings: bool = False
) -> pd.DataFrame:
    """
    Core function to transform numerical features in a DataFrame using various mathematical transformations.
    Supports transformations to improve data distributions for modeling, with a focus on normalization and
    linearization.

    Args:
        df (pd.DataFrame): Input DataFrame containing features to transform.
        features_to_transform (list[str] | None, default=None): Optional list of features to transform.
            If None, function will help identify numerical features.
        verbose (bool, default=True): Whether to display detailed guidance and explanations.
        preserve_features (bool, default=False): Whether to keep original features in the DataFrame
            alongside transformed ones.
        skip_warnings (bool, default=False): Whether to skip distribution and outlier warnings.

    Returns:
        pd.DataFrame: DataFrame with transformed numerical features.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If DataFrame is empty or specified features don't exist.
    """
    # Define constants
    MIN_SAMPLES = 10
    NORMAL_SKEW = 0.5
    HIGH_SKEW = 1.0

    def plot_dist(series: pd.Series, feature_name: str, title_suffix: str = '') -> None:
        """
        This helper function creates and displays a distribution plot for a feature.

        Args:
            series: Series to visualize
            feature_name: Name of the feature being visualized
            title_suffix: Optional suffix for the plot title
        """
        try:
            plt.figure(figsize=(16, 10))
            sns.histplot(data=series.dropna(), kde=True)
            plt.title(f'Distribution of "{feature_name}"{title_suffix}')
            plt.xlabel(feature_name)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()
            plt.close()

        except Exception as exc:
            print(f'Could not create distribution visualization: {str(exc)}')
            if plt.get_fignums():
                plt.close('all')

    def plot_comp(original: pd.Series, transformed: pd.Series, feature_name: str,
                  transform_name: str) -> None:
        """
        This helper function creates and displays a side-by-side comparison of original and transformed distributions.

        Args:
            original: Original series before transformation
            transformed: Series after transformation
            feature_name: Name of the feature being visualized
            transform_name: Name of the transformation applied
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

            # Original distribution
            sns.histplot(data=original.dropna(), kde=True, ax=ax1)
            ax1.set_title(f'Original Distribution of "{feature_name}"')
            ax1.set_xlabel(feature_name)

            # Transformed distribution
            sns.histplot(data=transformed.dropna(), kde=True, ax=ax2)
            ax2.set_title(f'"{feature_name}" after {transform_name} Transform')
            ax2.set_xlabel(f'{feature_name} (Transformed)')

            plt.tight_layout()
            plt.show()
            plt.close()

        except Exception as exc:
            print(f'Could not create distribution comparison: {str(exc)}')
            if plt.get_fignums():
                plt.close('all')

    def suggest_transform(series: pd.Series) -> list[str]:
        """
        This helper function suggests appropriate transformations based on data characteristics.

        Args:
            series: Numerical series to analyze

        Returns:
            list of suggested transformation methods
        """
        suggestions = []

        # Skip if too few non-null values are present
        if len(series.dropna()) < MIN_SAMPLES:
            return ['WARNING: Too few samples present. Do not transform this feature.']

        # Calculate statistics
        skew = series.skew()
        has_zeros = (series == 0).any()
        has_negs = (series < 0).any()

        # Based on skewness, suggest transformations
        if abs(skew) < NORMAL_SKEW:
            # Data is already approximately normal
            suggestions.append('boxcox' if not has_negs and not has_zeros else 'yeojohnson')

        else:
            # For right-skewed data (positive skew)
            if skew > HIGH_SKEW:
                # Group transformations by data requirements

                # Transformations for strictly positive data (no zeros, no negatives)
                if not has_negs and not has_zeros:
                    suggestions.append('log')  # Natural log
                    suggestions.append('boxcox')  # Box-Cox transformation

                # Transformations for non-negative data (allows zeros)
                if not has_negs:  # Note: this includes the previous case
                    suggestions.append('sqrt')  # Square root
                    suggestions.append('log1p')  # log(1+x) handles zeros

                # Transformations for any data (including negatives and zeros)
                suggestions.append('yeojohnson')  # Yeo-Johnson works with all values

            # For left-skewed data (negative skew)
            elif skew < -HIGH_SKEW:
                # Power transformations for left skew
                suggestions.append('square')
                suggestions.append('cube')

                # Yeo-Johnson works for all data
                suggestions.append('yeojohnson')

                # Reciprocal for left skew (if no zeros or negatives)
                if not has_zeros and not has_negs:
                    suggestions.append('reciprocal')

            # For moderately skewed data
            else:
                suggestions.append('yeojohnson')  # Safe choice for most distributions

        # If no specific suggestions, include generally-applicable options
        if not suggestions:
            suggestions = ['yeojohnson']

        return suggestions

    def apply_transform(series: pd.Series, method: str) -> tuple[pd.Series, str]:
        """
        This helper function applies the requested transformation to a series.

        Args:
            series: Series to transform
            method: Transformation method name

        Returns:
            tuple of (transformed series, description of transformation)
        """
        # Create a copy to prevent modifying the original
        result = series.copy()

        # Handle each transformation type
        if method == 'log':
            # Check for valid values first
            if (result <= 0).any():
                raise ValueError('Log transform requires all positive values')
            # Natural log transform
            result = result.transform(np.log)
            desc = 'Natural logarithm'

        elif method == 'log10':
            # Check for valid values first
            if (result <= 0).any():
                raise ValueError('Log10 transform requires all positive values')
            # Base-10 log transform
            result = result.transform(np.log10)
            desc = 'Base-10 logarithm'

        elif method == 'log1p':
            # Check for valid values first
            if (result < 0).any():
                raise ValueError('Log1p transform requires non-negative values')
            # Log(1+x) transform (handles zeros)
            result = result.transform(np.log1p)
            desc = 'Natural logarithm of (1+x)'

        elif method == 'sqrt':
            # Check for valid values first
            if (result < 0).any():
                raise ValueError('Square root transform requires non-negative values')
            # Square root transform
            result = result.transform(np.sqrt)
            desc = 'Square root'

        elif method == 'square':
            # Square transform
            result = result.transform(np.square)
            desc = 'Square (x²)'

        elif method == 'cube':
            # Cube transform
            result = result.transform(lambda x: np.power(x, 3))
            desc = 'Cube (x³)'

        elif method == 'reciprocal':
            # Check for valid values first
            if (result == 0).any():
                raise ValueError('Reciprocal transform cannot handle zero values')
            # Reciprocal transform
            result = result.transform(lambda x: 1 / x)
            desc = 'Reciprocal (1/x)'

        elif method == 'boxcox':
            # Check for valid values first
            if (result <= 0).any():
                raise ValueError('Box-Cox transform requires strictly positive values')
            # Box-Cox transform (finds optimal lambda parameter)
            np_array = result.to_numpy()
            np_transformed, lambda_val = stats.boxcox(np_array)
            # Convert back to Series with original index
            result = pd.Series(np_transformed, index=result.index)
            desc = f'Box-Cox (lambda={lambda_val:.4f})'

        elif method == 'yeojohnson':
            # Yeo-Johnson transform (works with negative values)
            np_array = result.to_numpy()
            np_transformed, lambda_val = stats.yeojohnson(np_array)
            # Convert back to Series with original index
            result = pd.Series(np_transformed, index=result.index)
            desc = f'Yeo-Johnson (lambda={lambda_val:.4f})'

        else:
            raise ValueError(f'Unknown transformation method: {method}')

        return result, desc

    # Begin main function execution
    if df.empty:
        print('DataFrame is empty. No transformation possible.')
        return df

    if verbose:
        print('-' * 50)
        print('Beginning feature transformation process.')
        print('-' * 50)

    # Feature identification
    if features_to_transform is None:
        # Identify numerical features
        numeric_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]

        # Filter out likely categorical numeric features (binary/boolean)
        filtered_cols = []
        for column in numeric_cols:
            unique_vals = sorted(df[column].dropna().unique())
            # Skip binary features (0/1 values)
            if len(unique_vals) <= 2 and all(float(x).is_integer() for x in unique_vals if pd.notna(x)):
                if verbose:
                    print(f'Excluding "{column}" from transformation - appears to be binary/categorical.')
                continue
            filtered_cols.append(column)

        if not filtered_cols:
            if verbose:
                print('No suitable numerical features found for transformation.')
            return df

        # In verbose mode, allow user to select features interactively
        if verbose:
            print(f'Identified {len(filtered_cols)} potential numerical features for transformation:')
            print(', '.join(filtered_cols))

            # Allow user to select which features to transform
            print('\nWhich features would you like to transform?')
            print('1. All identified numerical features')
            print('2. Select specific features')
            print('3. Cancel transformation process')

            final_features = None
            while final_features is None:  # Keep looping until we have valid features
                selection = input('Enter your choice (1, 2, or 3): ').strip()

                if selection == '1':
                    final_features = filtered_cols
                    # No break needed - loop will exit since final_features is no longer None

                elif selection == '2':
                    # Show features for selection
                    print('\nAvailable numerical features:')
                    for idx, col in enumerate(filtered_cols, 1):
                        print(f'{idx}. {col}')

                    while True:  # Loop for feature selection
                        # Get user selection
                        user_input = input(
                            '\nEnter the feature numbers to transform (comma-separated) or "C" to cancel: ')

                        if user_input.lower() == 'c':
                            print('Feature selection cancelled.')
                            break  # Break out of inner loop, return to main menu

                        try:
                            # Parse selected indices
                            indices = [int(idx.strip()) - 1 for idx in user_input.split(',')]

                            # Validate indices
                            if not all(0 <= idx < len(filtered_cols) for idx in indices):
                                print('Invalid feature number(s). Please try again.')
                                continue  # Try feature selection again

                            # Get selected feature names
                            final_features = [filtered_cols[idx] for idx in indices]
                            break  # Successfully got features, break inner loop

                        except ValueError:
                            print('Invalid input. Please enter comma-separated numbers.')
                            continue  # Try feature selection again

                elif selection == '3':
                    print('Transformation cancelled.')
                    return df

                else:
                    print('Invalid choice. Please enter 1, 2, or 3.')
        else:
            # In non-verbose mode, automatically use all identified numerical features
            final_features = filtered_cols
    else:
        # Validate provided features
        missing_cols = [col for col in features_to_transform if col not in df.columns]
        if missing_cols:
            raise ValueError(f'Features not found in DataFrame: {missing_cols}')

        # Verify features are numeric
        non_numeric = [col for col in features_to_transform if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            raise ValueError(f'The following features are not numeric and cannot be transformed: {non_numeric}')

        final_features = features_to_transform

    if verbose:
        print(f'\nPreparing to transform {len(final_features)} features:')
        print(', '.join(final_features))

        # Offer explanation of transformation methods
        explain = input('\nWould you like to see an explanation of transformation methods? (Y/N): ').lower()
        if explain == 'y':
            print('\nTransformation Methods Overview:')
            print('- Log Transformation: Best for right-skewed data, compresses large values')
            print('- Square Root: Less aggressive than log, works well for moderately skewed data')
            print('- Box-Cox: Finds optimal power transformation for normalization (requires positive values)')
            print('- Yeo-Johnson: Similar to Box-Cox but works with zero and negative values')
            print('- Square/Cube: For left-skewed data, amplifies differences in larger values')
            print('- Reciprocal (1/x): Reverses the order of values, transforms very skewed distributions')
            print('\nNOTE: Different transformations are appropriate for different data distributions.')
            print('Skewness will be analyzed to recommend suitable transformation methods.')

    # Track transformations for summary
    transform_records = []

    # Process each feature
    for feature in final_features:
        if verbose:
            print('-' * 50)  # Visual separator
        print(f'Processing feature: "{feature}"')
        if verbose:
            print('-' * 50)  # Visual separator

        # Skip constant features
        if df[feature].nunique() <= 1:
            print(f'Skipping "{feature}" - this feature has no variance.')
            continue

        # Check for minimum sample size requirement
        if len(df[feature].dropna()) < MIN_SAMPLES:
            print(f'Skipping "{feature}" - insufficient samples (minimum required: {MIN_SAMPLES}).')
            continue

        # Data quality checks
        null_count = df[feature].isnull().sum()
        if null_count > 0 and not skip_warnings:
            pct_null = (null_count / len(df) * 100)
            print(f'Warning: "{feature}" contains {null_count} null values ({pct_null:.2f}%).')
            print('Transformations will be applied only to non-null values.')
            if verbose:
                if input('Continue with transforming this feature? (Y/N): ').lower() != 'y':
                    continue
            # In non-verbose mode, ask for confirmation on high-nullity features
            elif pct_null > 30:  # If more than 30% of values are null
                if input('Continue with transforming this feature? (Y/N): ').lower() != 'y':
                    continue

        # Feature analysis
        original_values = df[feature].copy()

        if verbose:
            # Show current distribution statistics
            print(f'Current statistics for "{feature}":')
            desc_stats = df[feature].describe()
            print(desc_stats)

            # Show skewness and kurtosis
            skew = df[feature].skew()
            kurt = df[feature].kurtosis()
            print(f'\nSkewness: {skew:.4f}')
            print(f'Kurtosis: {kurt:.4f}')

            if abs(skew) > HIGH_SKEW:
                print(f'This feature is{"" if skew > 0 else " negatively"} skewed.')
                if not skip_warnings:
                    print('A transformation may help normalize the distribution.')

            # Show zeros and negatives
            has_zeros = (df[feature] == 0).any()
            has_negs = (df[feature] < 0).any()
            if has_zeros:
                print(f'This feature contains zeros. Some transformations (like log) may not be appropriate.')
            if has_negs:
                print(f'This feature contains negative values. Some transformations require positive data only.')

            # Offer to show distribution plot
            if input('\nWould you like to see the current distribution? (Y/N): ').lower() == 'y':
                plot_dist(df[feature], feature)

        # Get transformation suggestions based on data characteristics
        suggestions = suggest_transform(df[feature])

        # Check if there's a warning about sample size
        if any(sugg.startswith('WARNING:') for sugg in suggestions):
            if verbose:
                print(f"\n{suggestions[0]}")
                print(f'Skipping transformation for feature "{feature}".')
            else:
                print(f'Skipping "{feature}" - insufficient data (Minimum instances required: {MIN_SAMPLES}).')
            continue

        if verbose:
            print('\nBased on feature characteristics, the following transformations might be appropriate:')
            for suggestion in suggestions:
                print(f'- {suggestion}')

        # Show transformation options
        print('\nAvailable transformation methods:')

        # Build set of valid methods based on data characteristics
        has_zeros = (df[feature] == 0).any()
        has_negs = (df[feature] < 0).any()

        # Use set to avoid duplicates
        methods_set = set()

        # Add methods appropriate for the data characteristics
        if not has_negs and not has_zeros:
            methods_set.update(['log', 'log10', 'sqrt', 'reciprocal', 'boxcox'])
        elif not has_negs:
            methods_set.update(['log1p', 'sqrt'])

        # These work with all data including negatives
        methods_set.update(['yeojohnson', 'square', 'cube'])

        # Convert to sorted list for presentation
        methods = sorted(methods_set)

        # Show options to user
        for idx, method in enumerate(methods, 1):
            print(f'{idx}. {method.title()}')

        # Add skip option
        print(f'{len(methods) + 1}. Skip transformation for this feature')

        # Get user's transformation choice
        if verbose:
            method_idx = None
            while True:
                try:
                    choice = input('\nSelect transformation method: ')
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(methods):
                            method_idx = idx
                            selected_method = methods[idx]
                            break
                        elif idx == len(methods):
                            # User chose to skip
                            method_idx = idx
                            break
                        else:
                            print(f'Please enter a number between 1 and {len(methods) + 1}.')
                    except ValueError:
                        print('Invalid input. Please enter a number.')
                except Exception as exc:
                    print(f'Error in selection: {str(exc)}')
                    print('Please try again.')

            # Skip to next feature if user chose to skip
            if method_idx is None or method_idx == len(methods):
                print(f'Skipping transformation for feature "{feature}".')
                continue

            selected_method = methods[method_idx]
        else:
            # In non-verbose mode, select the first recommended transformation that's valid
            valid_suggs = [sugg for sugg in suggestions if sugg in methods]
            if valid_suggs:
                selected_method = valid_suggs[0]
                # Skip features with no valid transformations
            else:
                print(f'Skipping "{feature}" - no appropriate transformation available.')
                continue

        # Apply transformation
        try:
            # Target column name (either original or new)
            target_column = feature
            if preserve_features:
                target_column = f'{feature}_transformed'
                # Check if target column already exists
                counter = 1
                while target_column in df.columns:
                    target_column = f'{feature}_transformed_{counter}'
                    counter += 1

            # Apply the transformation
            transformed_values, desc = apply_transform(df[feature], selected_method)

            # Update the dataframe
            df[target_column] = transformed_values

            # Track the transformation
            transform_records.append({
                'feature': feature,
                'target': target_column,
                'method': selected_method,
                'desc': desc
            })

            if verbose:
                print(f'\nSuccessfully transformed "{feature}" using {desc}.')

                # Calculate and show new skewness
                new_skew = df[target_column].skew()
                print(f'New skewness: {new_skew:.4f} (was {skew:.4f})')

                # Show before/after comparison
                if input('\nWould you like to see a comparison of the distributions? (Y/N): ').lower() == 'y':
                    plot_comp(original_values, df[target_column], feature, selected_method)
            else:
                print(f'Transformed "{feature}" using {desc}.')

        except Exception as exc:
            print(f'Error transforming feature "{feature}": {exc}')
            print('Skipping this feature.')
            continue

    # Print summary if features were transformed
    if transform_records:
        if verbose:
            print('\n' + '-' * 50)
            print('TRANSFORMATION SUMMARY:')
            print('-' * 50)

            for record in transform_records:
                print(f'Feature: {record["feature"]}')
                print(f'- Method: {record["method"].title()}')
                print(f'- Description: {record["desc"]}')
                if record["feature"] != record["target"]:
                    print(f'- New column: {record["target"]}')
                print('-' * 50)

            if preserve_features:
                print('NOTE: Original features were preserved alongside transformed columns.')

    if verbose:
        print('-' * 50)
        print('Transformation complete. Returning modified dataframe.')
        print('-' * 50)

    # Return the modified dataframe
    return df


def _extract_datetime_core(
        df: pd.DataFrame,
        datetime_features: list[str] | None = None,
        verbose: bool = True,
        preserve_features: bool = False,
        components: list[str] | None = None
) -> pd.DataFrame:
    """
    Core function to extract useful features from datetime columns in a dataframe.

    Args:
        df: Input DataFrame containing datetime features.
        datetime_features: Optional list of datetime features to process. If None, function will
            identify datetime features interactively.
        verbose: Whether to display detailed guidance and explanations.
        preserve_features: Whether to keep original datetime features in the DataFrame.
        components: Optional list of specific datetime components to extract. If None, function
            will help identify components interactively.

    Returns:
        DataFrame with extracted datetime features.
    """
    # Standard datetime components available for extraction
    STANDARD_COMPONENTS = [
        'year', 'month', 'day', 'dayofweek', 'hour', 'minute',
        'quarter', 'weekofyear', 'dayofyear'
    ]

    # Time series specific components
    TS_COMPONENTS = [
        'is_weekend', 'is_month_start', 'is_month_end',
        'is_quarter_start', 'is_quarter_end'
    ]

    # Component mapping between name and pandas datetime accessor method
    COMPONENT_MAP = {
        'year': 'year',
        'month': 'month',
        'day': 'day',
        'dayofweek': 'dayofweek',
        'hour': 'hour',
        'minute': 'minute',
        'second': 'second',
        'quarter': 'quarter',
        'weekofyear': 'isocalendar().week',
        'dayofyear': 'dayofyear',
        'is_weekend': lambda x: x.dayofweek >= 5,
        'is_month_start': 'is_month_start',
        'is_month_end': 'is_month_end',
        'is_quarter_start': 'is_quarter_start',
        'is_quarter_end': 'is_quarter_end'
    }

    def parse_datetime_col(df: pd.DataFrame, col: str) -> bool:
        """This helper function parses a column as datetime if it's not already a datetime dtype."""
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return True

        try:
            # Try to convert to datetime
            df[col] = pd.to_datetime(df[col])
            return True
        except (ValueError, TypeError):
            if verbose:
                print(f'Could not convert column "{col}" to datetime type.')
            return False

    def is_timeseries(df: pd.DataFrame, date_col: str) -> bool:
        """This helper function detects whether a datetime column represents a time series."""
        # Check if we have at least 10 non-null values
        if df[date_col].count() < 10:
            return False

        # Sort values and check if differences are mostly consistent
        try:
            sorted_dates = df[date_col].dropna().sort_values()
            diff = sorted_dates.diff().dropna()

            # Calculate mode of differences (most common interval)
            most_common_diff = diff.value_counts().index[0]

            # If more than 60% of intervals are the same, likely a time series
            if (diff == most_common_diff).mean() > 0.6:
                return True

            return False

        except:
            return False

    def extract_components(df: pd.DataFrame, col: str, comp_list: list[str], cyclical: bool = False) -> dict:
        """This helper function extracts datetime components from a single column."""
        extracted = {}

        for comp in comp_list:
            try:
                # Generate column name for the extracted component
                new_col = f'{col}_{comp}'

                # Extract the component
                if comp in COMPONENT_MAP:
                    if callable(COMPONENT_MAP[comp]):
                        df[new_col] = COMPONENT_MAP[comp](df[col])
                    else:
                        # Handle nested property access (like isocalendar().week)
                        if '.' in COMPONENT_MAP[comp]:
                            parts = COMPONENT_MAP[comp].split('.')
                            temp = getattr(df[col].dt, parts[0])
                            for part in parts[1:]:
                                if '()' in part:
                                    method_name = part.split('(')[0]
                                    temp = getattr(temp, method_name)()
                                else:
                                    temp = getattr(temp, part)
                            df[new_col] = temp
                        else:
                            df[new_col] = getattr(df[col].dt, COMPONENT_MAP[comp])

                # Apply cyclical encoding if requested
                if cyclical and comp in ['month', 'dayofweek', 'day', 'hour']:
                    if comp == 'month':
                        max_val = 12
                    elif comp == 'dayofweek':
                        max_val = 7
                    elif comp == 'day':
                        max_val = 31
                    elif comp == 'hour':
                        max_val = 24
                    else:
                        continue

                    # Create sin and cos transformations
                    sin_col = f'{col}_{comp}_sin'
                    cos_col = f'{col}_{comp}_cos'
                    df[sin_col] = np.sin(2 * np.pi * df[new_col] / max_val)
                    df[cos_col] = np.cos(2 * np.pi * df[new_col] / max_val)

                    # Record these new columns
                    extracted[sin_col] = 'sin'
                    extracted[cos_col] = 'cos'

                # Record the extracted component
                extracted[new_col] = comp

            except Exception as e:
                if verbose:
                    print(f'Error extracting {comp} from {col}: {str(e)}')

        return extracted

    if verbose:
        print('-' * 50)
        print('Beginning datetime feature extraction process.')
        print('-' * 50)

    # Identify datetime columns if not provided
    if datetime_features is None:
        # Look for explicit datetime columns
        explicit_dt_cols = [col for col in df.columns
                            if pd.api.types.is_datetime64_any_dtype(df[col])]

        # Look for potential string datetime columns
        potential_dt_cols = []
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                # Try to parse the first non-null value
                sample = df[col].dropna().iloc[:5] if not df[col].dropna().empty else []
                try:
                    for val in sample:
                        # If any value is parseable as datetime, add to potential list
                        pd.to_datetime(val)
                        potential_dt_cols.append(col)
                        break
                except:
                    continue

        if verbose and explicit_dt_cols:
            print(f'Found {len(explicit_dt_cols)} explicit datetime columns:')
            for col in explicit_dt_cols:
                print(f'- {col}')

        if verbose and potential_dt_cols:
            print(f'Found {len(potential_dt_cols)} potential datetime columns:')
            for col in potential_dt_cols:
                print(f'- {col}')

        all_dt_cols = explicit_dt_cols + potential_dt_cols

        if not all_dt_cols:
            print('No datetime columns found in the DataFrame.')
            return df

        # Let user select from discovered datetime columns
        if verbose:
            print('\nWhich datetime columns would you like to extract features from?')
            for idx, col in enumerate(all_dt_cols, 1):
                print(f'{idx}. {col}')
            print(f'{len(all_dt_cols) + 1}. All columns')
            print(f'{len(all_dt_cols) + 2}. Cancel extraction')

        while True:
            try:
                user_choice = input('Enter your choice: ')

                # Check if user wants to use all columns
                if user_choice == str(len(all_dt_cols) + 1):
                    datetime_features = all_dt_cols
                    break

                # Check if user wants to cancel
                elif user_choice == str(len(all_dt_cols) + 2):
                    print('Extraction cancelled. DataFrame was not modified.')
                    return df

                # Check if user wanted specific columns
                else:
                    # Parse comma-separated list of indices
                    if ',' in user_choice:
                        indices = [int(idx.strip()) for idx in user_choice.split(',')]
                    else:
                        indices = [int(user_choice)]

                    # Validate indices
                    if not all(1 <= idx <= len(all_dt_cols) for idx in indices):
                        print(f'Please enter valid numbers between 1 and {len(all_dt_cols)}')
                        continue

                    # Convert indices to column names
                    datetime_features = [all_dt_cols[idx - 1] for idx in indices]
                    break

            except ValueError:
                print('Invalid input. Please enter a valid number or comma-separated list.')

    # Ensure all specified datetime columns are valid
    valid_dt_cols = []
    for col in datetime_features:
        if col not in df.columns:
            print(f'Warning: Column "{col}" not found in the DataFrame.')
            continue

        # Convert to datetime if needed
        if parse_datetime_col(df, col):
            valid_dt_cols.append(col)

    if not valid_dt_cols:
        print('No valid datetime columns to process.')
        return df

    # Determine which components to extract
    if components is None:
        available_components = STANDARD_COMPONENTS.copy()

        # Check if any column is likely a time series
        is_ts_data = any(is_timeseries(df, col) for col in valid_dt_cols)

        if is_ts_data:
            available_components.extend(TS_COMPONENTS)
            if verbose:
                print('\nTime series data detected. Additional time series components available.')

        if verbose:
            print('\nSelect components to extract:')
            for idx, comp in enumerate(available_components, 1):
                print(f'{idx}. {comp}')
            print(f'{len(available_components) + 1}. All components')
            print(f'{len(available_components) + 2}. Cancel extraction')

            # Ask about cyclical encoding
            cyclical_enc = input(
                '\nApply cyclical encoding to periodic components (month, day, hour)? (Y/N): ').lower() == 'y'
        else:
            cyclical_enc = False

        while True:
            try:
                user_choice = input('Enter your choice: ')

                # Check if user wants all components
                if user_choice == str(len(available_components) + 1):
                    components = available_components
                    break

                # Check if user wants to cancel
                elif user_choice == str(len(available_components) + 2):
                    print('Extraction cancelled. DataFrame was not modified.')
                    return df

                # Check if user wanted specific components
                else:
                    # Parse comma-separated list of indices
                    if ',' in user_choice:
                        indices = [int(idx.strip()) for idx in user_choice.split(',')]
                    else:
                        indices = [int(user_choice)]

                    # Validate indices
                    if not all(1 <= idx <= len(available_components) for idx in indices):
                        print(f'Please enter valid numbers between 1 and {len(available_components)}')
                        continue

                    # Convert indices to component names
                    components = [available_components[idx - 1] for idx in indices]
                    break

            except ValueError:
                print('Invalid input. Please enter a valid number or comma-separated list.')
    else:
        # With user-provided components, cyclical encoding is off by default
        cyclical_enc = False

        # Validate user-provided components
        valid_components = []
        for comp in components:
            if comp in COMPONENT_MAP:
                valid_components.append(comp)
            else:
                if verbose:
                    print(f'Warning: Unknown component "{comp}". Skipping.')

        if not valid_components:
            print('No valid datetime components to extract.')
            return df

        components = valid_components

    # Track columns created during extraction
    extracted_columns = {}

    # Process each datetime column
    for col in valid_dt_cols:
        if verbose:
            print(f'\nExtracting components from "{col}"...')

        # Extract specified components
        new_cols = extract_components(df, col, components, cyclical_enc)
        extracted_columns.update(new_cols)

    # Remove original datetime columns if not preserving
    if not preserve_features:
        df = df.drop(columns=valid_dt_cols)
        if verbose:
            print('\nRemoved original datetime columns.')

    # Print summary if verbose
    if verbose and extracted_columns:
        print('\nEXTRACTION SUMMARY:')
        print('-' * 50)
        print(f'Created {len(extracted_columns)} new columns:')

        # Group by original datetime column
        by_source = {}
        for new_col, comp in extracted_columns.items():
            source = new_col.split('_')[0]  # This assumes column naming convention
            if source not in by_source:
                by_source[source] = []
            by_source[source].append((new_col, comp))

        # Print grouped summary
        for source, cols in by_source.items():
            print(f'\nFrom {source}:')
            for col, comp in cols:
                print(f'- {col} ({comp})')

    if verbose:
        print('-' * 50)
        print('Datetime feature extraction complete. Returning modified dataframe.')
        print('-' * 50)

    return df
