import re
import numpy as np
import pandas as pd
import matplotlib
from collections import defaultdict
from itertools import combinations

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


def _reshape_core(
        df: pd.DataFrame,
        features_to_reshape: list[str] | None = None,
        verbose: bool = True
) -> pd.DataFrame:
    """
    Core function for reshaping a DataFrame by identifying missing values and dropping rows and columns.

    Args:
        df (pd.DataFrame): Input DataFrame to reshape.
        features_to_reshape (list[str]): User-provided list of features to constrain TADPREP behavior.
        verbose (bool): Whether to print detailed information about operations. Defaults to True.

    Returns:
        pd.DataFrame: Reshaped DataFrame

    Raises:
        ValueError: If invalid indices are provided for column dropping
        ValueError: If an invalid subsetting proportion is provided
    """
    if verbose:
        print('-' * 50)  # Visual separator
        print(f'Beginning data reshape process. \nInput data of {df.shape[0]} instances x {df.shape[1]} features.')
        print('-' * 50)  # Visual separator

    ## Helper func to identify current level of row-based missingness by threshold
    # Default threshold is 25% "real" values
    def rows_missing_by_thresh(df: pd.Dataframe, threshold: float = 0.25) -> int:
        """
        Helper function to determine missingness of data by row for a given percentage threshold.

        Args:
            df (pd.Dataframe): Input DataFrame in process of 'reshape'.
            threshold (float): Populated data threshold. Defaults to 0.25.

        Returns:
            int: Count of instances in 'df' with 'threshold' or less populated data.
            
        Raises:
            ...
        """
        # Determine how many NA's at each row and encode by threshold if 
        sum_by_missing = df.isna().sum(axis=1).tolist()
        encode_by_thresh = [1 if ((df.shape[1] - row_cnt) / df.shape[1]) <= (threshold)
                            else 0
                            for row_cnt in sum_by_missing]
        # Sum count of rows that meet threshold
        row_missing_cnt = sum(encode_by_thresh)
        
        return row_missing_cnt

    def recommend_thresholds(df: pd.DataFrame) -> list:
        """
        Helper function generates recommended degree-of-population thresholds based on size of user data.

        Args:
            df (pd.DataFrame): Input DataFrame in process of 'reshape'.

        Returns:
            list: Array of recommended degree-of-population thresholds.
        """
        print('Degree-of-population thresholds adjust based on # of Features in DataFrame.')
        print('Consider custom thresholds based on your understanding of data requirements.')
        
        feature_cnt = df.shape[1]
        
        if feature_cnt <= 5:
            print(f'\nFeature space is {feature_cnt}: Evaluated as "very small".')
            
            print(f'Recommend very high thresholds\n{[]}')
            
        ##  if feature_cnt  


    ## Helper func to identify rows with pre-defined column values missing
    def rows_missing_by_feature(df: pd.DataFrame, features_to_reshape: list[str]) -> dict:
        """
        Helper function generates counts of missingness by features in 'features_to_reshape'

        Args:
            df (pd.DataFrame): Input DataFrame in process of 'reshape'.
            features_to_reshape (list[str]): User-provided list of features to constrain TADPREP behavior.

        Returns:
            missing_cnt_by_feature (dict): Keyed by feature, Val count of missing per-key
        """
        # Straighforward dict comprehension to store 'features_to_reshape' and corresponding
        # missingness counts as key:val pairs
        missing_cnt_by_feature = {feature: df[feature].isna().sum() for feature in features_to_reshape}
        
        # Pandas-native approach to counting rows missing ALL 'features_to_reshape'
        missing_all_feature_cnt = df[features_to_reshape].isna().all().sum()
        
        # Add count of rows missing ALL 'features_to_reshape'
        missing_cnt_by_feature['ALL'] = missing_all_feature_cnt
        
        return missing_cnt_by_feature
        
    ## Core Operation 1
    def row_based_row_remove(df: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
        """
        Function to perform row-based row removal from input DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame in process of 'reshape'.
            threshold (float | None): Decimal percent degree-of-population threshold to apply to row removal process.

        Returns:
            df (pd.DataFrame): DataFrame in process of 'reshape' with rows removed by degree-of-population threshold.
        """
        
        # Here we rework the threshold by rounding for communication and df.dropna(thresh=)
        final_thresh = int(round(df.shape[1] * threshold))
        
        # rows_missing_by_thresh defaults to 25%
        print(f'Identified {rows_missing_by_thresh(df, threshold)} instances with {(threshold * 100):.2f}% or less populated data.')
        print(f'Rounding {(threshold * 100):.2f}% threshold to {final_thresh} features out of {df.shape[1]}.')
        
        # User confirmation to drop instances with fewer than 'final_tresh' populated features
        while True:
            proceed = input(f'Drop all instances with {final_thresh} or fewer populated features? (Y/N): ')
        
            if proceed.lower() == 'y':
                print('Dropping\n')
                df.dropna(thresh=final_thresh, inplace=True)
                
            elif proceed.lower() == 'n':
                print('Aborting drop operation. Input DataFrame not modified.\n')
                break
                
            else:
                print('Invalid input, please enter "Y" or "N.\n')
                continue
        
        return df
    
    ## Core Operation 2
    def column_based_row_remove(df: pd.DataFrame, features_to_reshape: list[str]) -> pd.DataFrame:
        """
        Function to perform column-based row removal from input DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame in process of 'reshape'.
            features_to_reshape (list[str]): DataFrame columns by which to apply row removal process.

        Returns:
            df (pd.DataFrame): DataFrame in process of 'reshape' with rows removed by column-missingness.
        """
        
        ## This build assumes that input arg 'features_to_reshape' will be the way user provides
        ## what features they wish to analyze and drop by.
        ## It is also the way this func determines relevant missingness
        
        # Create dict of missings-by-feature
        missing_cnt_by_feature = rows_missing_by_feature(df, features_to_reshape)
        
        print('Counts of instances missing by feature:')
        for pair in sorted(missing_cnt_by_feature.items()):
            print(pair)

        # User confirmation to drop instances missingness in 'features_to_reshape'
        while True:
            proceed = input(f'Drop all instances with missing values in {features_to_reshape} ? (Y/N): ')
        
            if proceed.lower() == 'y':
                print('Dropping\n')
                df.dropna(subset=features_to_reshape, inplace=True)
                
            elif proceed.lower() == 'n':
                print('Aborting drop operation. Input DataFrame not modified.\n')
                break
                
            else:
                print('Invalid input, please enter "Y" or "N.\n')
                continue
        
        return df
    
    ## Core Operation 3
    def drop_columns(df: pd.DataFrame, features_to_reshape: list[str]) -> pd.DataFrame:
        
        ## This theoretically could be just a pandas wrapper.
        ## Check with Don about how it might integrate with UX ideas.

        # Drop columns in 'features_to_reshape'
        input = input(f'Drop columns {features_to_reshape}? (Y/N): ')
        if input.lower() == 'y':
            print('Dropping columns\n')
            df.drop(columns=features_to_reshape, inplace=True)
        elif input.lower() == 'n':
            print('Aborting column drop operation. Input DataFrame not modified.\n')
        else:
            print('Invalid input, please enter "Y" or "N.\n')
        
        return df
    
    return df

# #-------New OOP-----------------------------------

### Working through this has me thinking that there may be benefits to
### handling the entire process of TADPREP through OOP, considering the
### user's DataFrame will exist in a variety of states throughout.

class PlotHandler:
    def __init__(self, palette: str = 'colorblind'):
        
        self.palette = palette
        sns.set_palette(self.palette)

        # Triple-layer defaultdict() allows for automatic data structuring
        # when we expect all plots to be stored in the same way
        self.plot_storage = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Basic structure for data snapshot storage
        # {
        #     'col_name': {
        #         'plot_type': {
        #             'plot_num': [int],
        #             'data': [pd.Series] (pd.Series will contain indexing info)
        #         }
        #     }
        # } 
        
    def det_plot_type(self, data: pd.DataFrame | pd.Series, col_name: str) -> tuple:
        """
        Determines an "appropriate" plot type for a set of data (pd.Series)
        based on the pandas dtype of the user's DataFrame column.

        Args:
            col_name (str): Specified DataFrame column to determine plotting info for.
        
        Raises:
            ValueError: _description_

        Returns:
            plot_type (tuple (pd.dtype, str)): A tuple containing dtype and its corresponding "plot type" string (hist/line/etc.).
        """

        #####################################
        # This method may not be necessary, but is currently a placeholder method in the case
        # we decide to separate plot-type determination from method-arg-input.
        #####################################
        
        # Determine if input is pd.DataFrame or pd.Series
        if isinstance(data, pd.DataFrame):
            dtype = data[col_name].dtype
        elif isinstance(data, pd.Series):
            dtype = data.dtype
        # Empty str variable, will be re-valued by this method
        plot_type = ""
        
        if pd.api.types.is_numeric_dtype(dtype):
            plot_type = 'hist'  # Define 'hist' as plot type for numeric data
            
        elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            plot_type = 'box'   # Define 'box' as plot type for numeric data
            
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            # '.is_datetime64_any_dtype' allows for TZ-aware DataFrames
            plot_type = 'line'  # Define 'line' for time-series data
            
        else:
            plot_type = 'scatter'   # For all other types, assume mixed data and assign 'scatter'
        
        return (dtype, plot_type)
    
    def plot(self, df: pd.DataFrame, col_name: str, plot_type: str):
        """
        Create a specified Seaborn plot for a specified pandas DataFrame column.
        Copies current data state to PlotHandler class instance self.plot_storage.

        Args:
            data (pd.DataFrame): DataFrame to reference.
            col_name (str): Name of DataFrame column for plotting and archiving.
            plot_type (str): Type of plot to create (hist, box, line, scatter).

        Raises:
            ValueError: _description_
        """

        data = df[col_name]

        ### LIKELY UNNECESSARY if giving user control over method arg
        # # Check dtype of pd.Series for indexing in self.plot_storage
        # plot_type = self.det_plot_type(data, col_name)[1]
        
        # Create 'plot_num' list with first value 1 if not already present
        if not self.plot_storage[col_name][plot_type]['plot_num']:
            self.plot_storage[col_name][plot_type]['plot_num'].append(1)
        # Otherwise, append the next number in the sequence
        else:
            self.plot_storage[col_name][plot_type]['plot_num'].append(
                self.plot_storage[col_name][plot_type]['plot_num'][-1] + 1
            )

        self.plot_storage[col_name][plot_type]['data'].append(data)


        if plot_type == 'hist':
            plot = sns.histplot(data=data, kde=True)
            plot.set_title(f"Histogram for '{col_name}'")
            plt.show()  # Assume viz is desired on creation for now
        
            return
        
        elif plot_type == 'box':
            plot = sns.boxplot(data=data)
            plot.set_title(f"Box Plot for '{col_name}'")
            plt.show()  # Assume viz is desired on creation for now
        
            return
        
        elif plot_type == 'line':
            plot = sns.lineplot(data=data)
            plot.set_title(f"Line Plot for '{col_name}'")
            plt.show()  # Assume viz is desired on creation for now
        
            return
        
        elif plot_type == 'scatter':
            plot = sns.scatterplot(data=data)
            plot.set_title(f"Scatter Plot for '{col_name}'")
            plt.show()  # Assume viz is desired on creation for now
            
            return
        
        else:
            print(f'Unsupported plot type: {plot_type}')
            return
    
    def recall_plot(self, col_name: str, plot_type: str):
        """
        Recall a previously-created stored plot for a given dtype and DataFrame column.

        Args:
            dtype (_type_): dtype to be fetched from defaultdict.
            col_name (str): Name of DataFrame column for fetching from defaultdict.

        Raises:
            ValueError: _description_
        """
        # Could be implemented to fetch by plot number, but currently fetches most recent plot
        if not self.plot_storage[col_name][plot_type]:
            print(f"No plot found for '{col_name}' with dtype '{plot_type}'")
        else:
            # Fetch data for single most recent plot of specified dtype and col_name
            stored_data = self.plot_storage[col_name][plot_type]['data'][-1]
            stored_order = self.plot_storage[col_name][plot_type]['plot_num'][-1]

            print(f"Redrawing {plot_type} plot #{stored_order} for '{col_name}'")

            if plot_type == 'hist':
                plot = sns.histplot(data=stored_data, kde=True)
                plot.set_title(f"Histogram #{stored_order} for '{col_name}'")
            elif plot_type == 'box':
                plot = sns.boxplot(data=stored_data)
                plot.set_title(f"Box Plot #{stored_order} for '{col_name}'")
            elif plot_type == 'line':
                plot = sns.lineplot(data=stored_data)
                plot.set_title(f"Line Plot #{stored_order} for '{col_name}'")
            elif plot_type == 'scatter':
                plot = sns.scatterplot(data=stored_data)
                plot.set_title(f"Scatter Plot #{stored_order} for '{col_name}'")
            plt.show()
        
        return

    def compare_plots(self, col_name: str):
        """
        Compare all stored plots for a given column.

        Args:
            col_name (str): Name of DataFrame column for which plots will be compared.

        Raises:
            ValueError: _description_
        """

        types_list = ['hist', 'box', 'line', 'scatter']

        # Determine the number of rows and columns for subplots
        subplots_nrows = max(len(self.plot_storage[col_name][plot_type]['data']) for plot_type in types_list)
        
        subplots_ncols = sum(1 for plot_type in types_list if self.plot_storage[col_name][plot_type]['data'])

        # debug
        print(f"nrows: {subplots_nrows}, ncols: {subplots_ncols}")

        # Create subplots
        fig, ax = plt.subplots(nrows=subplots_nrows,
                               ncols=subplots_ncols,
                               sharex=True
                               )

        # Ensure ax is always a 2D array for consistent indexing
        if subplots_nrows == 1:
            ax = [ax]  # Convert to a list for 1D case
        if subplots_ncols == 1:
            ax = [[a] for a in ax]  # Convert to a nested list for 2D case

        # Iterate over columns (plot types)
        for col, curr_plot_type in enumerate(types_list):
            if curr_plot_type in self.plot_storage[col_name]:
                # Iterate over rows (individual plots)
                for row, data in enumerate(self.plot_storage[col_name][curr_plot_type]['data']):
                    # Validate data presence
                    if data.dropna().empty:
                        print(f"No valid data available for {curr_plot_type} at position: {row, col}")
                        continue
                    
                    ### BUG HERE: sns.histplot does not populate for 1st column in 2D .subplots() case
                    print(f"Plotting {curr_plot_type} at position: {row, col}")
                    if curr_plot_type == 'hist':
                        sns.histplot(data=data, kde=True, ax=ax[row][col])
                    elif curr_plot_type == 'box':
                        sns.boxplot(data=data, ax=ax[row][col])
                    elif curr_plot_type == 'line':
                        sns.lineplot(data=data, ax=ax[row][col])
                    elif curr_plot_type == 'scatter':
                        sns.scatterplot(data=data, ax=ax[row][col])

                    ax[row][col].set_title(f"{curr_plot_type.capitalize()} Plot {row+1}")

        fig.suptitle(f"Comparison of all plots for '{col_name}'")
        plt.tight_layout()
        plt.show()

        return
    
def _build_interactions_core(
        df: pd.DataFrame,
        f1: str | None = None,
        f2: str | None = None,
        features_list: list[str] | None = None,
        interact_types: list[str] | None = None,
        verbose: bool = True,
        preserve_features: bool = True
        # max_features: int | None = None
) -> pd.DataFrame:
    """
    Core function to build interaction terms between specified features in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to build interactions from.
        f1 (str): Feature 1 to interact in "focused" paradigm.
        f2 (str): Feature 2 to interact in "focused" paradigm.
        features_list (list[str]): List of features to interact in "exploratory" paradigm.
        interact_types (list[str]): List of interaction types to apply.
        verbose (bool): Whether to print detailed information about operations. Defaults to True.
        preserve_features (bool): Whether to retain original features in the DataFrame. Defaults to True.
        max_features (int): Optional maximum number of interaction features to create.

    Returns:
        pd.DataFrame: DataFrame with interaction terms appended and user-specified columns removed.

    Raises:
        ValueError: If invalid interaction types are provided.
    """
    #TODO: Data Validation steps to be moved to package.py
    #TODO: Implement feature selection checks, including max_features

    # Check for existence of input DataFrame
    if df.empty:
        print('DataFrame is empty. No interactions can be created.')
        return df

    if verbose:
        print('-' * 50)  # Visual separator
        print('Beginning feature interaction process.')
        print('-' * 50)  # Visual separator

    # Remove dupes from 'interact_types' if they exists
    # Replace "categorical calls" if included
    # and preserve order:
    def clean_interact_types(interact_types: list[str]):
        # Replace "categorical calls" if included, including dupes
        if 'arithmetic' in interact_types:
            interact_types = [x for x in interact_types if x != 'arithmetic']
            interact_types = ['+', '-', '*', '/'] + interact_types
        if 'exponential' in interact_types:
            interact_types = [x for x in interact_types if x != 'exponential']
            interact_types = ['^2', '^3', '^1/2', '^1/3', 'e^x'] + interact_types
        if 'distance' in interact_types:
            interact_types = [x for x in interact_types if x != 'distance']
            interact_types = ['magnitude','magsum', 'magdiff'] + interact_types
        if 'polynomial' in interact_types:
            interact_types = [x for x in interact_types if x != 'polynomial']
            interact_types = ['poly', 'prod^1/2', 'prod^1/3'] + interact_types
        if 'logexp' in interact_types:
            interact_types = [x for x in interact_types if x != 'logexp']
            interact_types = ['log_inter', 'exp_inter'] + interact_types
        if 'stat' in interact_types:
            interact_types = [x for x in interact_types if x != 'stat']
            interact_types = ['mean_diff', 'mean_ratio'] + interact_types

        # Remove dupes from 'interact_types' if they exists
        seen = set()
        result_list = []
        for type in interact_types:
            if type in seen:
                if verbose:
                    print(f"Duplicate interaction type '{type}' detected. Removing...")
            if type not in seen:
                result_list.append(type)
                seen.add(type)
        interact_types = result_list

        return interact_types

    if interact_types:
        interact_types = clean_interact_types(interact_types)


    # Check for valid interaction categories or types
    valid_types = ['arithmetic',                        # Categorical interaction calls
                   'exponential',                       #
                   'distance',                          #
                   'polynomial',                        #
                   'logexp',                            #
                   'stat',                              #
                   '+', '-', '*', '/',                  # Arithmetic
                   '^2', '^3', '^1/2', '^1/3', 'e^x',   # Exponential
                   'magnitude','magsum', 'magdiff',     # Distance
                   'poly', 'prod^1/2', 'prod^1/3',      # Polynomial and other roots
                   'log_inter', 'exp_inter',            # Logarithmic and exponential interactions
                   'mean_diff', 'mean_ratio']           # Statistical interactions

    manual_abort = 0
    if (interact_types is None) or (not interact_types):
        print(f"""
                No interaction types specified.
                Available interaction types are:
                Arithmetic    ->  '+', '-', '*', '/'
                Exponential  ->  '^2', '^3', '^1/2', '^1/3', 'e^x'
                Distance     ->  'magnitude', 'magsum', 'magdiff'
                Polynomial   ->  'poly', 'prod^1/2', 'prod^1/3'
                Logarithmic  ->  'log_inter', 'exp_inter'
                Statistical  ->  'mean_diff', 'mean_ratio'
                """)
        input0 = input("Would you like further detail on the available interaction types? (Y/N): ")
        if input0.lower() == 'y':
            print(f"""
                    '+' : Column-wise sum of features            ->  df[f1] + df[f2]
                    '-' : Column-wise difference of features     ->  df[f1] - df[f2]
                    '*' : Column-wise product of features        ->  df[f1] * df[f2]
                    '/' : Column-wise quotient of features       ->  df[f1] / df[f2]
                    '^2'   : Single-column square of feature     ->  df[f1] ** 2
                    '^3'   : Single-column cube of feature       ->  df[f1] ** 3
                    '^1/2' : Single-column sqrt of feature       ->  np.sqrt(df[f1])
                    '^1/3' : Single-column cbrt of feature       ->  np.cbrt(df[f1])
                    'e^x'  : Single-column exponent of feature   ->  np.exp(df[f1])
                    'magnitude' : Column-wise sqrt of squares    ->  np.sqrt(df[f1] ** 2 + df[f2] ** 2)
                    'magsum'   : Column-wise absolute diff       ->  np.abs(df[f1] + df[f2])
                    'magdiff'   : Column-wise absolute diff      ->  np.abs(df[f1] - df[f2])
                    'poly'     : Column-wise binomial square     ->  df[f1] * df[f2] + df[f1] ** 2 + df[f2] ** 2
                    'prod^1/2' : Column-wise sqrt of product     ->  np.sqrt(np.abs(df[f1] * df[f2]))
                    'prod^1/3' : Column-wise cbrt of product     ->  np.cbrt(df[f1] * df[f2])
                    'log_inter' : Product of offset logarithms   ->  np.log(df[f1] + 1) * np.log(df[f2] + 1)
                    'exp_inter' : Product of feature exponents   ->  np.exp(df[f1]) * np.exp(df[f2])
                    'mean_diff'  : f1 difference from f1,f2 mean ->  df[f1] - df[[f1, f2]].mean(axis=1)
                    'mean_ratio' : Ratio of f1 to f1,f2 mean     ->  df[f1] / df[[f1, f2]].mean(axis=1)
                """)
        elif input0.lower() == 'n':
            pass
        else:
            print("Invalid input. Select 'Y' or 'N'.")

        iter_input = 0
        while True:
            if iter_input == 0:
                input1 = input(f"""
                                Interaction types must be specified for .build_interactions() operation.
                                Please select one of the following:
                                1. Provide a list of custom interaction types   (comma-separated format. i.e. [+,^2,magnitude,...])
                                2. Apply some common "default" interactions     (Arithmetic interactions [+,-,*,/])
                                3. Exit method
                                -> """)
            elif iter_input > 0:
                input1 = input("-> ")

            if input1.lower() == '1':
                input10 = input("Interaction types: ")
                interact_types = input10.lower().split(",")
                print(f"\nContinuing with new interact_types: {interact_types}")
                break
            elif input1.lower() == '2':
                print("Defaulting to ALL Arithmetic interactions (+, -, *, /).")
                interact_types = ['+', '-', '*', '/']
                break

            elif input1.lower() == '3':
                print("Aborting .build_interactions() operation. Input DataFrame not modified")
                manual_abort = 1
                break
            else:
                print("Invalid input. Select '1', '2', or '3'")
                iter_input += 1
                continue

    if manual_abort == 1:
        return df

    for type in interact_types:
        if type not in valid_types:
            raise ValueError(f'Invalid interaction type: {type}')
        
    # Clean up runtime user inputs
    interact_types = clean_interact_types(interact_types)

    if features_list and len(features_list) == 1 and not set(interact_types).issubset(set(['^2', '^3', '^1/2', '^1/3', 'e^x'])):
        print("Only 1 feature provided and multi-feature interaction types specified. INVALID OPERATION SET. Aborting method. Input DataFrame not modified.")
        return df
    
    # Record of actions for 'verbose' state
    action_summary = []

    ## Handle error from incorrect 'f1', 'f2' arg input
    if f2 and not f1:
        print("Only 'f2' feature provided for TADPREP 'Focused' _build_interactions method. Please provide only 'f1' if single-column interactions desired.")
        return df

    ### "focused" interaction creation paradigm
    ### i.e. specific interactions between two specifically-provided features
    if f1:
        if not f2:
            print("'f2' feature not provided for multi-column interaction. Aborting method. Input DataFrame not modified.")
            return df
        if set(interact_types).issubset(set(['^2', '^3', '^1/2', '^1/3', 'e^x'])):
            print("WARNING: All provided interact_types are single-column interactions. Only f1-argument-sourced interactions will be created in 'Focused' paradigm.")
        # Perform interaction term creation
        for interact in interact_types:
            
            ## Arithmetic interactions
            if interact == '+':                     # Sum
                new_feature = f'{f1}_+_{f2}'
                df[new_feature] = df[f1] + df[f2]
            elif interact == '-':                   # Difference
                new_feature = f'{f1}_-_{f2}'
                df[new_feature] = df[f1] - df[f2]
            elif interact == '*':                   # Product
                new_feature = f'{f1}_*_{f2}'
                df[new_feature] = df[f1] * df[f2]
            elif interact == '/':                   # Quotient
                new_feature = f'{f1}_/_{f2}'
                df[new_feature] = df[f1] / df[f2]
                # Replace infinite values with NaN
                df[new_feature] = df[new_feature].replace([np.inf, -np.inf], np.nan)

                if verbose:
                    print('***Div-by-zero errors are replaced with NaN. Take care to handle these and propagated-NaNs in your analysis.***')

            ## Exponential interactions
            elif interact == '^2':                  # Square
                new_feature = f'{f1}^2'
                df[new_feature] = df[f1] ** 2
            elif interact == '^3':                  # Cube
                new_feature = f'{f1}^3'
                df[new_feature] = df[f1] ** 3
            elif interact == '^1/2':                # Square root
                new_feature = f'{f1}^1/2'
                df[new_feature] = np.sqrt(df[f1])
            elif interact == '^1/3':                # Cube root
                new_feature = f'{f1}^1/3'
                df[new_feature] = np.cbrt(df[f1])
            elif interact == 'e^x':                 # Exponential
                new_feature = f'e^{f1}'
                df[new_feature] = np.exp(df[f1])

            ## Distance interactions
            elif interact == 'magnitude':           # Magnitude
                new_feature = f'magnitude_{f1}_{f2}'
                df[new_feature] = np.sqrt(df[f1] ** 2 + df[f2] ** 2)
            elif interact == 'magsum':              # Magnitude sum
                new_feature = f'magsum_{f1}_{f2}'
                df[new_feature] = np.abs(df[f1] + df[f2])
            elif interact == 'magdiff':             # Magnitude difference
                new_feature = f'magdiff_{f1}_{f2}'
                df[new_feature] = np.abs(df[f1] - df[f2])
            
            ## Polynomial and other roots
            elif interact == 'poly':                # Polynomial
                new_feature = f'poly_{f1}_{f2}'
                df[new_feature] = df[f1] * df[f2] + df[f1] ** 2 + df[f2] ** 2
            elif interact == 'prod^1/2':            # Product square root
                new_feature = f'prod^1/2_{f1}_{f2}'
                df[new_feature] = np.sqrt(np.abs(df[f1] * df[f2]))
            elif interact == 'prod^1/3':            # Product cube root
                new_feature = f'prod^1/3_{f1}_{f2}'
                df[new_feature] = np.cbrt(df[f1] * df[f2])
            
            ## Logarithmic and exponential interactions
            elif interact == 'log_inter':           # Logarithmic interaction
                new_feature = f'log_inter_{f1}_{f2}'
                df[new_feature] = np.log(df[f1] + 1) * np.log(df[f2] + 1)
            elif interact == 'exp_inter':           # Exponential interaction
                new_feature = f'exp_inter_{f1}_{f2}'
                df[new_feature] = np.exp(df[f1]) * np.exp(df[f2])

            ## Statistical interactions
            elif interact == 'mean_diff':           # Mean difference
                new_feature = f'mean_diff_{f1}_{f2}'
                df[new_feature] = df[f1] - df[[f1, f2]].mean(axis=1)
            elif interact == 'mean_ratio':          # Mean ratio
                new_feature = f'mean_ratio_{f1}_{f2}'
                df[new_feature] = df[f1] / df[[f1, f2]].mean(axis=1)

            if verbose:
                summary_str = f'Created new feature: {new_feature} ({interact} interaction)'
                action_summary.append(summary_str)
                print(summary_str)

        if not preserve_features:
            if verbose:
                drop_str = f'Dropping original features {f1} and {f2} from DataFrame.'
                action_summary.append(drop_str)
                print(drop_str)
            df.drop(columns=[f1,f2], inplace=True)

    ### "exploratory" interaction creation paradigm
    ### i.e. all possible interactions between all features in provided list
    if features_list:
        # Perform interaction term creation with itertools.combinations
        if len(features_list) == 1:
            feature_combinations = [(features_list[0], None)]
        else:
            feature_combinations = list(combinations(set(features_list), 2))
        if verbose:
            print("All combinations of submitted 'features_list' elements:")
            print(feature_combinations)
        # Create flag for "only single-feature interactions present"
        single_feat_interactions_only = 0
        # Adjust if necessary
        if set(interact_types).issubset(set(['^2', '^3', '^1/2', '^1/3', 'e^x'])):
            single_feat_interactions_only = 1
            
        for feature, other_feature in feature_combinations:
            for interact in interact_types:
                
                # Arithmetic interactions
                if interact == '+':           # Sum
                    new_feature = f'{feature}_+_{other_feature}'
                    df[new_feature] = df[feature] + df[other_feature]
                elif interact == '-':           # Difference
                    new_feature = f'{feature}_-_{other_feature}'
                    df[new_feature] = df[feature] - df[other_feature]
                elif interact == '*':             # Product
                    new_feature = f'{feature}_*_{other_feature}'
                    df[new_feature] = df[feature] * df[other_feature]
                elif interact == '/':           # Quotient
                    new_feature = f'{feature}_/_{other_feature}'
                    df[new_feature] = df[feature] / df[other_feature]
                    # Replace infinite values with NaN
                    df[new_feature] = df[new_feature].replace([np.inf, -np.inf], np.nan)
                    
                    if verbose:
                        print('***Div-by-zero errors are replaced with NaN. Take care to handle these and propagated-NaNs in your analysis.***')

                # Exponential interactions
                elif interact == '^2':          # Square
                    new_feature = f'{feature}^2'
                    if new_feature not in df.columns:
                        df[new_feature] = df[feature] ** 2
                    if (len(feature_combinations) == 1) and (feature_combinations[0][1] != None):
                        new_feature_2 = f'{other_feature}^2'
                        df[new_feature_2] = df[other_feature] ** 2
                elif interact == '^3':          # Cube
                    new_feature = f'{feature}^3'
                    if new_feature not in df.columns:
                        df[new_feature] = df[feature] ** 3
                    if (len(feature_combinations) == 1) and (feature_combinations[0][1] != None):
                        new_feature_2 = f'{other_feature}^3'
                        df[new_feature_2] = df[other_feature] ** 3
                elif interact == '^1/2':        # Square root
                    new_feature = f'{feature}^1/2'
                    if new_feature not in df.columns:
                        df[new_feature] = np.sqrt(df[feature])
                    if (len(feature_combinations) == 1) and (feature_combinations[0][1] != None):
                        new_feature_2 = f'{other_feature}^1/2'
                        df[new_feature_2] = np.sqrt(df[other_feature])
                elif interact == '^1/3':        # Cube root
                    new_feature = f'{feature}^1/3'
                    if new_feature not in df.columns:
                        df[new_feature] = np.cbrt(df[feature])
                    if (len(feature_combinations) == 1) and (feature_combinations[0][1] != None):
                        new_feature_2 = f'{other_feature}^1/3'
                        df[new_feature_2] = np.cbrt(df[other_feature])
                elif interact == 'e^x':         # Exponential
                    new_feature = f'e^{feature}'
                    if new_feature not in df.columns:
                        df[new_feature] = np.exp(df[feature])
                    if (len(feature_combinations) == 1) and (feature_combinations[0][1] != None):
                        new_feature_2 = f'e^{other_feature}'
                        df[new_feature_2] = np.exp(df[other_feature])

                # Distance interactions
                elif interact == 'magnitude':   # Magnitude
                    new_feature = f'magnitude_{feature}_{other_feature}'
                    df[new_feature] = np.sqrt(df[feature] ** 2 + df[other_feature] ** 2)
                elif interact == 'magsum':      # Magnitude sum
                    new_feature = f'magsum_{feature}_{other_feature}'
                    df[new_feature] = np.abs(df[feature] + df[other_feature])
                elif interact == 'magdiff':     # Magnitude difference
                    new_feature = f'magdiff_{feature}_{other_feature}'
                    df[new_feature] = np.abs(df[feature] - df[other_feature])

                # Polynomial and other roots
                elif interact == 'poly':        # Polynomial
                    new_feature = f'poly_{feature}_{other_feature}'
                    df[new_feature] = df[feature] * df[other_feature] + df[feature] ** 2 + df[other_feature] ** 2
                elif interact == 'prod^1/2':    # Product square root
                    new_feature = f'prod^1/2_{feature}_{other_feature}'
                    df[new_feature] = np.sqrt(np.abs(df[feature] * df[other_feature]))
                elif interact == 'prod^1/3':    # Product cube root
                    new_feature = f'prod^1/3_{feature}_{other_feature}'
                    df[new_feature] = np.cbrt(df[feature] * df[other_feature])

                # Logarithmic and exponential interactions
                elif interact == 'log_inter':   # Logarithmic interaction
                    new_feature = f'log_inter_{feature}_{other_feature}'
                    df[new_feature] = np.log(df[feature] + 1) * np.log(df[other_feature] + 1)
                elif interact == 'exp_inter':   # Exponential interaction
                    new_feature = f'exp_inter_{feature}_{other_feature}'
                    df[new_feature] = np.exp(df[feature]) * np.exp(df[other_feature])

                # Statistical interactions
                elif interact == 'mean_diff':   # Mean difference
                    new_feature = f'mean_diff_{feature}_{other_feature}'
                    df[new_feature] = df[feature] - df[[feature, other_feature]].mean(axis=1)
                elif interact == 'mean_ratio':  # Mean ratio
                    new_feature = f'mean_ratio_{feature}_{other_feature}'
                    df[new_feature] = df[feature] / df[[feature, other_feature]].mean(axis=1)

                if verbose:
                    summary_str = f'Created new feature: {new_feature} ({interact} interaction)'
                    action_summary.append(summary_str)
                    print(summary_str)
                    if (single_feat_interactions_only == 1) and (feature_combinations[0][1] != None):
                        summary_str_2 = f'Created new feature: {new_feature_2} ({interact} interaction)'
                        action_summary.append(summary_str_2)
                        print(summary_str_2)

        # Drop original features if user specifies
        if not preserve_features:
            if verbose:
                drop_str = f'Dropping original features from DataFrame:\n{features_list}'
                action_summary.append(drop_str)
                print(drop_str)
            df.drop(columns=features_list, inplace=True)

    # Closing verbosity
    if verbose:
        iter_input = 0
        while True:
            if iter_input == 0:
                input2 = input("\nOperations complete. Would you like an action summary? (Y/N): ")
            elif iter_input > 0:
                input2 = input("-> ")

            if input2.lower() == 'y':
                print("\nThe following actions were performed by build_interactions():")
                for action in action_summary:
                    print(action)
                break
            elif input2.lower() == 'n':
                break
            else:
                print("Invlid input. Please enter either 'Y' or 'N'.")
                iter_input += 1
    if verbose:
        print('-' * 50)  # Visual separator
        print('Feature interaction complete. Returning modified dataframe.')
        print('-' * 50)  # Visual separator

    #TODO: Implement verbosity conditions
    #TODO: Implement warnings about large feature space and allow for cancellation

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
        min_val = series.min()
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
