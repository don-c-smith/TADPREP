import pandas as pd
from .core.transforms import (
    _df_info_core,
    _reshape_core,
    _subset_core,
    _rename_and_tag_core,
    _feature_stats_core,
    _impute_core,
    _encode_core,
    _scale_core,
    _prep_df_core
)


def df_info(df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Prints comprehensive information about a DataFrame's structure, contents, and potential data quality issues.

    Basic information (always shown):
    - Total number of instances (rows)
    - Total number of features (columns)
    - Count and percentage of instances containing any missing values
    - Count and percentage of duplicate instances, if any exist

    When verbose=True, the method also displays:
    - Features with very low variance (>95% single value)
    - Features containing infinite values (in numeric columns)
    - Features containing empty strings (distinct from NULL/NaN values)
    - Names and data types of all features
    - Non-null value count for each feature

    The method also provides contextual warnings about data quality issues that may require investigation.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be analyzed
    verbose : bool, default=True
        Controls whether detailed feature information and data quality checks are displayed

    Returns
    -------
    None
        This is a void method which prints information to the console.

    Notes
    -----
    Use verbose=True (default) to receive full information about data structure and potential quality issues.
    Use verbose=False when you only need basic information about dataset size and contents.

    Different from NULL/NaN values, empty strings ('') are valid string values that may indicate data quality
    issues in text columns.

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, None, 4],
    ...     'B': ['x', 'y', 'z', 'w']
    ... })
    >>> tadprep.df_info(df, verbose=True)  # Shows full dataframe information
    >>> tadprep.df_info(df, verbose=False)  # Shows reduced dataframe information
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    _df_info_core(df, verbose)


def reshape(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Interactively reshapes the input DataFrame according to user specification.

    Allows deletion of missing values, dropping columns, and random sub-setting of instances.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be reshaped
    verbose : bool, default=True
        Controls whether detailed process information is displayed

    Returns
    -------
    pandas.DataFrame
        The reshaped DataFrame as modified by the user's specifications

    Examples
    --------
        >>> import pandas as pd
        >>> import tadprep
        >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, 5, 6]})
        >>> df_reshaped = tadprep.reshape(df)  # Shows detailed status messages
        >>> df_reshaped_quiet = tadprep.reshape(df, verbose=False)  # Shows only necessary user prompts
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    return _reshape_core(df, verbose)


def subset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Interactively subsets the input DataFrame according to user specification. Supports random sampling
    (with or without a seed), stratified sampling, and time-based instance selection for timeseries data.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be reshaped
    verbose : bool, default=True
        Controls whether detailed process information and methodological guidance is displayed

    Returns
    -------
    pandas.DataFrame
        The modified DataFrame as subset by the user's specifications

    Examples
    --------
        >>> import pandas as pd
        >>> import tadprep
        >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, 5, 6]})
        >>> df_subset = tadprep.subset(df)  # Shows detailed status messages and guidance
        >>> df_subset_quiet = tadprep.subset(df, verbose=False)  # Shows only necessary user prompts
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    return _subset_core(df, verbose)


def rename_and_tag(df: pd.DataFrame, verbose: bool = True, tag_features: bool = False) -> pd.DataFrame:
    """
    Interactively renames features and allows user to tag them as ordinal or target features, if desired.

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

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({'feature1': [1,2,3], 'feature2': ['a','b','c']})
    >>> df_renamed = tadprep.rename_and_tag(df)  # Only rename features
    >>> df_tagged = tadprep.rename_and_tag(df, tag_features=True)  # Rename and tag features
    >>> df_renamed_quiet = tadprep.rename_and_tag(df, verbose=False, tag_features=False)  # Show minimal output
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    return _rename_and_tag_core(df, verbose=verbose, tag_features=tag_features)


def feature_stats(df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Displays feature-level statistics for each feature in the DataFrame.

    For each feature, displays missingness information and appropriate descriptive statistics
    based on the feature's datatype (boolean, datetime, categorical, or numerical).
    Features are automatically classified by type for appropriate statistical analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze
    verbose : bool, default=True
        Whether to print detailed statistical information and more extensive visual formatting

    Returns
    -------
    None
        This is a void method that prints information to the console.

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, None, 4],
    ...     'B': ['x', 'y', 'z', 'w']
    ... })
    >>> tadprep.feature_stats(df)  # Show detailed statistics with formatting
    >>> tadprep.feature_stats(df, verbose=False)  # Show only key feature-level statistics
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    _feature_stats_core(df, verbose=verbose)


def impute(df: pd.DataFrame, verbose: bool = True, skip_warnings: bool = False) -> pd.DataFrame:
    """
    Interactively imputes missing values in the DataFrame using simple imputation methods.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing missing values to impute
    verbose : bool, default = True
        Controls whether detailed process information is displayed
    skip_warnings : bool, default = False
        Controls whether missingness threshold warnings are displayed

    Returns
    -------
    pandas.DataFrame
        The DataFrame with imputed values

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({'A': [1, None, 3], 'B': ['x', 'y', None]})
    >>> df_imputed = tadprep.impute(df)  # Full guidance and warnings
    >>> df_imputed_quiet = tadprep.impute(df, verbose=False)  # Minimize output
    >>> df_imputed_nowarn = tadprep.impute(df, skip_warnings=True)  # Skip missingness warnings
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    return _impute_core(df, verbose=verbose, skip_warnings=skip_warnings)


def encode(
    df: pd.DataFrame,
    features_to_encode: list[str] | None = None,
    verbose: bool = True,
    skip_warnings: bool = False,
    preserve_features: bool = False
) -> pd.DataFrame:
    """
    Interactively encodes categorical features in the DataFrame using specified encoding methods.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing features to encode.
    features_to_encode : list[str] | None, default=None
        Optional list of features to encode - if None, method will help identify categorical features.
    verbose : bool, default=True
        Controls whether detailed guidance and explanations are displayed.
    skip_warnings : bool, default=False
        Controls whether all best-practice-related warnings about encoding are skipped.
    preserve_features : bool, default=False
        Whether to keep original features in the DataFrame alongside encoded ones.
        When True, original categorical columns are retained after encoding.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with encoded categorical features

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({'A': ['cat', 'dog', 'horse'], 'B': [1, 2, 3]})
    >>> df_encoded = tadprep.encode(df)  # Let function identify categorical features
    >>> df_encoded_specified = tadprep.encode(df, features_to_encode=['A'])  # Specify features to encode
    >>> df_encoded_quiet = tadprep.encode(df, verbose=False)  # Minimize output
    >>> df_encoded_nowarn = tadprep.encode(df, skip_warnings=True)  # Skip best-practice warnings
    >>> df_encoded_preserved = tadprep.encode(df, preserve_features=True)  # Keep original features
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    # Validate features_to_encode if provided by user
    if features_to_encode is not None:
        if not isinstance(features_to_encode, list):
            raise TypeError('features_to_encode must be a list of strings')

        if not all(isinstance(col, str) for col in features_to_encode):
            raise TypeError('All feature names in features_to_encode must be strings')

        if not all(col in df.columns for col in features_to_encode):
            missing = [col for col in features_to_encode if col not in df.columns]
            raise ValueError(f'Features not found in DataFrame: {missing}')

    # Validate preserve_features parameter
    if not isinstance(preserve_features, bool):
        raise TypeError('preserve_features must be a boolean')

    return _encode_core(
        df,
        features_to_encode=features_to_encode,
        verbose=verbose,
        skip_warnings=skip_warnings,
        preserve_features=preserve_features
    )


def scale(
    df: pd.DataFrame,
    features_to_scale: list[str] | None = None,
    verbose: bool = True,
    skip_warnings: bool = False,
    preserve_features: bool = False
) -> pd.DataFrame:
    """
    Interactively scales numerical features in the DataFrame using standard scaling methods.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing features to scale.
    features_to_scale : list[str] | None, default=None
        Optional list of features to scale - if None, method will help identify numerical features.
    verbose : bool, default=True
        Controls whether detailed guidance and explanations are displayed.
    skip_warnings : bool, default=False
        Controls whether all best-practice-related warnings about scaling are skipped.
    preserve_features : bool, default=False
        Controls whether original features are preserved when scaling. When True, creates new columns
        with the naming pattern '{original_column}_scaled'. If a column with that name already exists,
        a numeric suffix is added: '{original_column}_scaled_1'.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with scaled numerical features. If preserve_features=True, original
        features are retained and new columns are added with scaled values.

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    >>> # Basic usage - let function identify numerical features
    >>> df_scaled = tadprep.scale(df)
    >>> # Specify features to scale
    >>> df_scaled_specified = tadprep.scale(df, features_to_scale=['A'])
    >>> # Minimize output
    >>> df_scaled_quiet = tadprep.scale(df, verbose=False)
    >>> # Skip best-practice warnings
    >>> df_scaled_nowarn = tadprep.scale(df, skip_warnings=True)
    >>> # Preserve original features, creating new scaled columns
    >>> df_with_both = tadprep.scale(df, preserve_features=True)
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    # Validate features_to_scale if provided
    if features_to_scale is not None:
        if not isinstance(features_to_scale, list):
            raise TypeError('features_to_scale must be a list of strings')

        if not all(isinstance(col, str) for col in features_to_scale):
            raise TypeError('All feature names in features_to_scale must be strings')

        if not all(col in df.columns for col in features_to_scale):
            missing = [col for col in features_to_scale if col not in df.columns]
            raise ValueError(f'Features not found in DataFrame: {missing}')

    # Validate preserve_features parameter
    if not isinstance(preserve_features, bool):
        raise TypeError('preserve_features must be a boolean')

    return _scale_core(
        df,
        features_to_scale=features_to_scale,
        verbose=verbose,
        skip_warnings=skip_warnings,
        preserve_features=preserve_features
    )


def prep_df(
        df: pd.DataFrame,
        features_to_encode: list[str] | None = None,
        features_to_scale: list[str] | None = None
) -> pd.DataFrame:
    """
    Run the complete TADPREP pipeline with user control over each step.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to process using the TADPREP pipeline
    features_to_encode : list[str] | None, default=None
        Optional pre-defined list of features to encode. If None, the function facilitates user selection.
    features_to_scale : list[str] | None, default=None
        Optional pre-defined list of features to scale. If None, the function facilitates user selection.

    Returns
    -------
    pandas.DataFrame
        The processed DataFrame after running the specified pipeline steps.

    Notes
    -----
    - At each step, users can:
      - Enter 'Y' to run a given step
      - Enter 'N' to skip a given step
      - Enter 'Q' to quit the pipeline altogether

    - For boolean parameters:
      - Press Enter to accept the default value
      - Enter '1' to set the parameter to True
      - Enter '2' to set the parameter to False

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, None, 4],
    ...     'B': ['x', 'y', 'z', 'w']
    ... })
    >>> processed_df = tadprep.prep_df(df)  # Run pipeline with interactive feature selection
    >>> processed_df = tadprep.prep_df(df, features_to_encode=['B'])  # Run pipeline with pre-defined features
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    # Validate features_to_encode if provided
    if features_to_encode is not None:
        if not isinstance(features_to_encode, list):
            raise TypeError('The "features_to_encode" object must be a list of strings')

        if not all(isinstance(col, str) for col in features_to_encode):
            raise TypeError('All feature names in features_to_encode must be strings')

        if not all(col in df.columns for col in features_to_encode):
            missing = [col for col in features_to_encode if col not in df.columns]
            raise ValueError(f'Features not found in DataFrame: {missing}')

    # Validate features_to_scale if provided
    if features_to_scale is not None:
        if not isinstance(features_to_scale, list):
            raise TypeError('The "features_to_scale" object must be a list of strings')

        if not all(isinstance(col, str) for col in features_to_scale):
            raise TypeError('All feature names in features_to_scale must be strings')

        if not all(col in df.columns for col in features_to_scale):
            missing = [col for col in features_to_scale if col not in df.columns]
            raise ValueError(f'Features not found in DataFrame: {missing}')

    print('Welcome to the TADPREP pipeline.')
    print('At each step you can:')
    print("- Enter 'Y' to execute the step")
    print("- Enter 'N' to skip the step")
    print("- Enter 'Q' to quit the pipeline")
    print('\nFor boolean parameters:')
    print('- Press Enter to accept the default value')
    print("- Enter '1' to set the parameter to True")
    print("- Enter '2' to set the parameter to False")

    return _prep_df_core(df, features_to_encode=features_to_encode, features_to_scale=features_to_scale)
