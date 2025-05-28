import pandas as pd
from .core.transforms import (
    _method_list_core,
    _summary_core,
    _reshape_core,
    _find_outliers_core,
    _find_corrs_core,
    _subset_core,
    _rename_and_tag_core,
    _feature_stats_core,
    _impute_core,
    _encode_core,
    _scale_core,
    _transform_core,
    _extract_datetime_core,
    _make_plots_core,
    _build_interactions_core
)


# OVERALL LIBRARY INFORMATION METHODS
def method_list():
    """
    Prints the names and brief descriptions of all callable methods in the TADPREP library.

    Parameters
    ----------
    None
        This is a nullary method which functions only as an information source.

    Returns
    -------
    None
        This is a void method which prints information to the console.

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> tadprep.method_list()  # Shows names and descriptions of available methods in the TADPREP library
    """
    _method_list_core()


# DATAFRAME-LEVEL INFORMATION AND MANIPULATIONS
def summary(df: pd.DataFrame) -> None:
    """
    Prints comprehensive information about a DataFrame's structure, contents, and potential data quality issues.

    Basic information:
    - Total number of instances (rows)
    - Total number of features (columns)
    - Count and percentage of instances containing any missing values
    - Names, non-Null counts, and datatypes of individual features
    - Counts of features aggregated at the datatype level
    - Dataframe physical memory usage

    The method also runs data quality cheks for:
    - Count and percentage of duplicate instances, if any exist
    - Features with very low variance (>95% single value)
    - Features containing infinite values (in numeric columns)
    - Features containing empty strings (distinct from NULL/NaN values)
    - Features which are actually numerical (integers or floats) but are typed as strings
    - Instances which are all-Null (i.e. 'empty rows')

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be analyzed

    Returns
    -------
    None
        This is a void method which prints information to the console.

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, None, 4],
    ...     'B': ['x', 'y', 'z', 'w']
    ... })
    >>> tadprep.summary(df, verbose=True)  # Shows dataframe information
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    _summary_core(df)


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


# EXPLORATORY DATA ANALYSIS (EDA)
def find_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = None, verbose: bool = True) -> dict:
    """
    Detects outliers in numerical features of a DataFrame using a specified detection method.

    Analyzes numerical features in the dataframe and identifies outliers using the specified detection method.

    Supports three common approaches for outlier detection: IQR-based detection, Z-score method, and
    Modified Z-score method.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze for outliers
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

    Returns
    -------
    dict
        A dictionary containing outlier information with summary and feature-specific details

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep as tp
    >>> df = pd.DataFrame({'A': [1, 2, 3, 100], 'B': ['x', 'y', 'z', 'w']})
    >>> outlier_results = tp.find_outliers(df)  # Use default IQR method
    >>> outlier_results = tp.find_outliers(df, method='zscore')  # Use Z-score method
    >>> outlier_results = tp.find_outliers(df, verbose=False)  # Hide detailed output
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    return _find_outliers_core(df, method=method, threshold=threshold, verbose=verbose)


def find_corrs(df: pd.DataFrame, method: str = 'pearson', threshold: float = 0.8, verbose: bool = True) -> dict:
    """
    Detects highly-correlated features in a DataFrame using the specified correlation method.

    Analyzes numerical features in the dataframe and identifies feature pairs with correlation coefficients exceeding
    the specified threshold.

    High correlations often indicate redundant features that could be simplified or removed to improve model
    performance and interpretability.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze for feature correlations
    method : str, default='pearson'
        Correlation method to use.
        Options:
          - 'pearson': Standard correlation coefficient (default, best for linear relationships)
          - 'spearman': Rank correlation (robust to outliers and non-linear relationships)
          - 'kendall': Rank correlation (more robust for small samples, handles ties differently)
    threshold : float, default=0.8
        Absolute correlation coefficient threshold above which features are considered highly correlated.
        Values should be between 0 and 1.
    verbose : bool, default=True
        Whether to print detailed information about detected correlations

    Returns
    -------
    dict
        A dictionary containing correlation information with summary statistics and detailed pair information.
        Structure:
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
            ]}

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep as tp
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [1, 3, 5, 7, 9],
    ...     'D': [2, 4, 6, 8, 10]
    ... })
    >>> # Use default Pearson correlation
    >>> corr_results = tp.find_corrs(df)
    >>> # Use Spearman correlation with lower threshold
    >>> corr_results = tp.find_corrs(df, method='spearman', threshold=0.6)
    >>> # Hide detailed output
    >>> corr_results = tp.find_corrs(df, verbose=False)
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    return _find_corrs_core(df, method=method, threshold=threshold, verbose=verbose)


def make_plots(df: pd.DataFrame, features_to_plot: list[str] | None = None) -> None:
    """
    Interactively creates and displays plots for features in a DataFrame.

    This method guides users through the process of creating feature-level plots based on the data characteristics
    of the user-selected features.

    It supports visualizations of numerical, categorical, and datetime features using appropriate plot types.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing features to plot.
    features_to_plot : list[str] | None, default=None
        Optional list of specific features to consider for plotting. If None, the function will use all features
        in the DataFrame.

    Returns
    -------
    None
        This function displays plots but does not return any values.

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep as tp
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': ['x', 'y', 'z', 'z', 'y'],
    ...     'C': [0.1, 0.2, 0.3, 0.4, 0.5]
    ... })
    >>> # Plot using all available features
    >>> tp.make_plots(df)
    >>>
    >>> # Plot using only specified features
    >>> tp.make_plots(df, features_to_plot=['A', 'B'])

    Notes
    -----
    The function supports various plot types including:
    - For single features: histograms, box plots, violin plots, count plots, bar plots
    - For two features: scatter plots, line plots, heat maps, bar plots
    - For three features: scatter plots with hue, line plots with hue, pair plots

    Plot types are suggested based on feature data types (numerical, categorical, datetime), and the user can select
    which plot type to create.
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    # Validate features_to_plot if provided
    if features_to_plot is not None:
        if not isinstance(features_to_plot, list):
            raise TypeError('features_to_plot must be a list of strings')

        if not all(isinstance(col, str) for col in features_to_plot):
            raise TypeError('All feature names in features_to_plot must be strings')

    # Call the core function
    _make_plots_core(df, features_to_plot=features_to_plot)


# FEATURE-LEVEL INFORMATION AND MANIPULATIONS
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

    For each feature, displays missingness information and appropriate descriptive statistics based on the feature's
    datatype (boolean, datetime, categorical, or numerical).

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
    Interactively imputes missing values in the DataFrame using user-specified simple imputation methods.

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
    Interactively encodes categorical features in the DataFrame using user-specified encoding methods.

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


def transform(
        df: pd.DataFrame,
        features_to_transform: list[str] | None = None,
        verbose: bool = True,
        preserve_features: bool = False,
        skip_warnings: bool = False
) -> pd.DataFrame:
    """
    Transforms numerical features using various mathematical transformations.

    Applies transformations to improve data distributions for modeling, with a focus on normalization and linearization.

    The method analyzes data characteristics and suggests appropriate transformations based on distribution properties.

    Supports transformations including:
    - Logarithmic: log, log10, log1p (for right-skewed data)
    - Power: sqrt, square, cube, reciprocal
    - Statistical: Box-Cox, Yeo-Johnson (for normalization)

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing features to transform.
    features_to_transform : list[str] | None, default=None
        Optional list of features to transform. If None, method will identify
        numerical features automatically.
    verbose : bool, default=True
        Controls whether detailed guidance, explanations, and visualizations are displayed.
    preserve_features : bool, default=False
        When True, creates new columns with transformed values instead of replacing
        original features.
    skip_warnings : bool, default=False
        Controls whether distribution and data quality warnings are skipped.

    Returns
    -------
    pandas.DataFrame
        DataFrame with transformed features.

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep as tp
    >>> df = pd.DataFrame({'A': [1, 2, 100, 200], 'B': ['x', 'y', 'z', 'w']})
    >>> # Basic usage with automatic feature detection
    >>> df_transformed = tp.transform(df)
    >>> # Specify features to transform
    >>> df_transformed = tp.transform(df, features_to_transform=['A'])
    >>> # Preserve original features alongside transformed versions
    >>> df_transformed = tp.transform(df, preserve_features=True)
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    # Validate features_to_transform if provided
    if features_to_transform is not None:
        if not isinstance(features_to_transform, list):
            raise TypeError('features_to_transform must be a list of strings')

        if not all(isinstance(col, str) for col in features_to_transform):
            raise TypeError('All feature names in features_to_transform must be strings')

        if not all(col in df.columns for col in features_to_transform):
            missing = [col for col in features_to_transform if col not in df.columns]
            raise ValueError(f'Features not found in DataFrame: {missing}')

    # Validate boolean parameters
    if not isinstance(preserve_features, bool):
        raise TypeError('preserve_features must be a boolean')

    if not isinstance(skip_warnings, bool):
        raise TypeError('skip_warnings must be a boolean')

    # Call the core implementation
    return _transform_core(
        df,
        features_to_transform=features_to_transform,
        verbose=verbose,
        preserve_features=preserve_features,
        skip_warnings=skip_warnings
    )


def extract_datetime(
        df: pd.DataFrame,
        datetime_features: list[str] | None = None,
        verbose: bool = True,
        preserve_features: bool = False
) -> pd.DataFrame:
    """
    Extracts useful features from datetime columns in a dataframe.

    The method identifies datetime columns and automatically extracts standard component features
    (year, month, day, dayofweek, hour, minute, quarter, dayofyear) for each datetime column.

    It then creates new extracted component columns using the naming pattern '{original_column}_{component}'.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing datetime features to extract.
    datetime_features : list[str] | None, default=None
        Optional list of datetime features to process. If None, the function will
        identify datetime features automatically.
    verbose : bool, default=True
        Controls whether detailed guidance and explanations are displayed.
    preserve_features : bool, default=False
        Controls whether original datetime features are preserved in the DataFrame.
        When False, original datetime columns are removed after extraction.

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame with extracted datetime features. If preserve_features=True,
        original datetime columns are retained.

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep as tp
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2021-01-01', periods=5),
    ...     'value': [10, 20, 30, 40, 50]
    ... })
    >>> # Basic usage - let function identify datetime columns automatically
    >>> df_extracted = tp.extract_datetime(df)
    >>> # Specify datetime columns to process
    >>> df_extracted = tp.extract_datetime(df, datetime_features=['date'])
    >>> # Keep original datetime columns
    >>> df_extracted = tp.extract_datetime(df, preserve_features=True)
    >>> # Minimize output
    >>> df_extracted = tp.extract_datetime(df, verbose=False)
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    # Validate datetime_features if provided
    if datetime_features is not None:
        if not isinstance(datetime_features, list):
            raise TypeError('datetime_features must be a list of strings')

        if not all(isinstance(col, str) for col in datetime_features):
            raise TypeError('All feature names in datetime_features must be strings')

        if not all(col in df.columns for col in datetime_features):
            missing = [col for col in datetime_features if col not in df.columns]
            raise ValueError(f'Features not found in DataFrame: {missing}')

    # Validate preserve_features
    if not isinstance(preserve_features, bool):
        raise TypeError('preserve_features must be a boolean')

    # Call core implementation, passing datetime_features as dt_feats
    return _extract_datetime_core(
        df,
        datetime_features=datetime_features,
        verbose=verbose,
        preserve_features=preserve_features)


def build_interactions(
    df: pd.DataFrame,
    f1: str | None = None,
    f2: str | None = None,
    features_list: list[str] | None = None,
    interact_types: list[str] | None = None,
    verbose: bool = True,
    preserve_features: bool = True
) -> pd.DataFrame:
    """
    Creates mathematical interaction terms between user-specified features in a DataFrame.

    Supports two UX paradigms:
    - Focused: Create specific interactions between two features (f1 and f2)
    - Exploratory: Create all possible interactions between features in features_list

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing features to create interactions from.
    f1 : str | None, default=None
        First feature for focused interactions.
    f2 : str | None, default=None
        Second feature for focused interactions.
    features_list : list[str] | None, default=None
        List of features for exploratory interactions.
    interact_types : list[str] | None, default=None
        List of interaction types to apply. If None, user will be prompted.
    verbose : bool, default=True
        Whether to display detailed information about the process.
    preserve_features : bool, default=True
        Whether to keep original features in the DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with interaction terms appended.

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep as tp
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> # Focused paradigm
    >>> df_interact = tp.build_interactions(df, f1='A', f2='B', interact_types=['+', '*'])
    >>> # Exploratory paradigm
    >>> df_interact = tp.build_interactions(df, features_list=['A', 'B'], interact_types=['+', '*'])
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    # Validate features_list if provided
    if features_list is not None:
        if not isinstance(features_list, list):
            raise TypeError('features_list must be a list of strings')

        if not all(isinstance(col, str) for col in features_list):
            raise TypeError('All feature names in features_list must be strings')

        if not all(col in df.columns for col in features_list):
            missing = [col for col in features_list if col not in df.columns]
            raise ValueError(f'Features not found in DataFrame: {missing}')

    # Validate f1 and f2 if provided
    if f1 is not None:
        if not isinstance(f1, str):
            raise TypeError('f1 must be a string')
        if f1 not in df.columns:
            raise ValueError(f'Feature f1="{f1}" not found in DataFrame')

    if f2 is not None:
        if not isinstance(f2, str):
            raise TypeError('f2 must be a string')
        if f2 not in df.columns:
            raise ValueError(f'Feature f2="{f2}" not found in DataFrame')

    # Validate interact_types if provided
    if interact_types is not None:
        if not isinstance(interact_types, list):
            raise TypeError('interact_types must be a list of strings')

        if not all(isinstance(t, str) for t in interact_types):
            raise TypeError('All interaction types must be strings')

    return _build_interactions_core(
        df,
        f1=f1,
        f2=f2,
        features_list=features_list,
        interact_types=interact_types,
        verbose=verbose,
        preserve_features=preserve_features
    )
