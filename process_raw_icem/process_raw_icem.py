import json
import polars as pl
from polars.exceptions import InvalidOperationError
import polars.selectors as cs

def replace_numeric_null_values(df, cols_to_ignore=["recid", "hid"], null_value=999999):
    """
    Replaces numeric null values (999999) in integer columns of I-CeM data stored in a Polars DataFrame with actual nulls (None).
    The I-CeM documentation states that "For numeric variables missing data are indicated by 999999 values." (https://www.campop.geog.cam.ac.uk/research/projects/icem/fields.html)
    Removing the 999999 values ensures the dataframe requires less memory because it allows integer columns (like 'Age') to be downcast below Int32. 
    With '999999' values no integer column can be downcast below Int32.

    Args:
        df (pl.DataFrame): The input Polars DataFrame to process.
        cols_to_ignore (list[str]): Column names to exclude from null replacement,
            integer columns that contain valid '999999' entries that should not be considered Null.
            Defaults to ["recid", "hid"].
        null_value (int): The value used to represent nulls in the dataset.
            Defaults to 999999, as per I-CeM documentation.

    Returns:
        pl.DataFrame: A new DataFrame with the same columns as the input DataFrame but with all
            signed integer columns (except `cols_to_ignore`) `null_values` replaced with Nulls.

    Example:
        >>> import polars as pl
        >>> data = {"recid": [1, 2], "hid": [10, 20], "age": [25, 999999]}
        >>> df = pl.DataFrame(data)
        >>> replace_numeric_null_values(df)
        shape: (2, 3)
        ┌───────┬─────┬──────┐
        │ recid ┆ hid ┆ age  │
        │ ---   ┆ --- ┆ ---  │
        │ i64   ┆ i64 ┆ i64  │
        ╞═══════╪═════╪══════╡
        │ 1     ┆ 10  ┆ 25   │
        │ 2     ┆ 20  ┆ null │
        └───────┴─────┴──────┘

    Notes:
        - Only Int64 columns are processed; other numeric types (e.g. Float64) are excluded.
        - The ignored columns are prepended to the result via a horizontal concat,
          preserving their original values unchanged.
    """
    # Isolate columns with valid 999999 values to remain unchanged.
    recid_hid = df.select(pl.col(cols_to_ignore))

    # On all remaining columns, select only signed integer types and replace `null_value` (e.g. 999999) with None
    df = (
        df.with_columns(pl.exclude(cols_to_ignore))
          .with_columns(cs.signed_integer().replace({null_value: None}))
    )

    # Rejoin the isolated columns with the cleaned integer columns
    # df = pl.concat([recid_hid, df], how="horizontal")

    return df



def downcast_integers(df):
    """
    Downcasts integer columns in a Polars DataFrame to the smallest integer type
    that can safely represent their values, reducing memory usage.

    Iterates through Int32, Int16, and Int8 ranges in descending order. For each
    type, any column whose minimum and maximum values both fit within that type's
    range is cast to it. Columns are cast repeatedly, so a column may be reduced
    from Int64 → Int32 → Int16 → Int8 if its values permit.

    Args:
        df (pl.DataFrame): The input Polars DataFrame. All integer columns are
            candidates for downcasting; non-integer columns are left unchanged.

    Returns:
        pl.DataFrame: A new DataFrame with eligible integer columns cast to a
            smaller integer type. The shape and column names are unchanged.

    Raises:
        polars.exceptions.InvalidOperationError: Caught internally per iteration.
            If a cast fails unexpectedly, the error is printed and the column
            retains its current type; processing continues for remaining columns.

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "big":   [1_000_000, 2_000_000],  # exceeds Int16/Int8 — stays Int32
        ...     "mid":   [200, 300],               # fits Int16, not Int8 — cast to Int16
        ...     "small": [1, 2],                   # fits Int8 — cast to Int8
        ...     "name":  ["a", "b"],               # non-integer — unchanged
        ... })
        >>> downcast_integers(df).dtypes
        [Int32, Int16, Int8, String]

    Notes:
        - Only the value range is checked — not the original dtype. A column that
          is already Int16 will still be evaluated against the Int8 range.
        - The ranges used are:
            Int32 — -2,147,483,648 to 2,147,483,647
            Int16 — -32,768 to 32,767
            Int8  — -128 to 127
        - Int64 is not included as a downcast target; it is the assumed starting type.
    """

    int_ranges = {
        "int32_range": {"integer": pl.Int32, "int_range": range(-2147483648, 2147483647)},
        "int16_range": {"integer": pl.Int16, "int_range": range(-32768, 32767)},
        "int8_params": {"integer": pl.Int8,  "int_range": range(-128, 127)},
    }

    dtype_dict = {}

    for key, integer_params in int_ranges.items():

        # Restrict to integer columns only to avoid errors on other dtypes
        df_intonly = df.select(cs.integer())

        # Identify columns whose min and max both fall within the target type's range
        subset = df_intonly.select(
            col.name for col in
            df_intonly.select(
                (pl.min(df_intonly.columns) > integer_params["int_range"].start) &
                (pl.max(df_intonly.columns) < integer_params["int_range"].stop)
            )
            if col.all()
        )

        cols_to_downcast = subset.columns
        
        # Overwrite any earlier entry — later iterations map to smaller types,
        # so the final value for each column is always its smallest qualifying type
        for col in cols_to_downcast:
            dtype_dict[col] = integer_params["integer"]

    try:
        # Apply all casts in one operation
        df = df.cast(dtype_dict)

    except InvalidOperationError as err:
        print(err)

    return df

def downcast_floats(df, floatdtypes=[pl.Float64, pl.Float32]):
    """
    Downcasts float columns in a Polars DataFrame to successively smaller float
    types, reducing memory usage (possibly at the cost of reduced precision, though in 
    reality none of the I-CeM columns will lose precision).

    Iterates through consecutive pairs in `floatdtypes`, casting all columns of
    the earlier type to the later type. With the default argument, this means
    every Float64 column is cast to Float32.

    Args:
        df (pl.DataFrame): The input Polars DataFrame. Only columns whose dtype
            matches an entry in `floatdtypes` are affected; all others are left
            unchanged.
        floatdtypes (list[pl.DataType]): An ordered list of Polars float types
            defining the casting chain. Each type is cast to the one immediately
            following it. Defaults to [pl.Float64, pl.Float32], giving a single
            Float64 → Float32 pass. A longer chain such as
            [pl.Float64, pl.Float32, pl.Float16] would perform two successive
            downcasts, but result in 'inf' values for some columns so has not been included.

    Returns:
        pl.DataFrame: A new DataFrame with eligible float columns cast to a
            smaller float type. Shape and column names are unchanged.

    Raises:
        polars.exceptions.InvalidOperationError: Caught internally per iteration.
            If a cast fails, the error is printed and the column retains its
            current type; processing continues for remaining pairs.

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "a": pl.Series([1.5, 2.5], dtype=pl.Float64),
        ...     "b": pl.Series([0.1, 0.2], dtype=pl.Float32),
        ...     "c": pl.Series([1,   2  ], dtype=pl.Int32),
        ... })
        >>> downcast_floats(df).dtypes
        [Float32, Float32, Int32]  # Float64 → Float32; Float32 and Int32 unchanged

    Notes:
        - Unlike downcast_integers(), no range checking is performed — every
          column matching the source dtype is cast unconditionally. Values that
          cannot be represented exactly in the target type will be silently
          rounded or lose precision.
    """


    for previous, current in zip(floatdtypes, floatdtypes[1:]):

        try:
            # Cast all columns of the current source type to the next smaller type
            df = df.with_columns(pl.col(previous).cast(current))

        except InvalidOperationError as err:
            # Log the failure and continue — the column retains its current type
            print(err)

    return df


def create_categoricals(df, categorical_threshold=1000):
    """
    Converts low-cardinality string columns in a Polars DataFrame to the
    Categorical dtype, reducing memory usage and improving performance on
    group-by and join operations.

    Scans all string columns and builds a mapping of any column with fewer
    unique values than `categorical_threshold` to pl.Categorical, then applies
    all casts in a single operation.

    Args:
        df (pl.DataFrame): The input Polars DataFrame. Only string columns are
            evaluated; all other dtypes are left unchanged.
        categorical_threshold (int): The maximum number of unique values a string
            column may have to be eligible for conversion. A column with fewer
            than this many unique values is cast to Categorical. Defaults to 1000.

    Returns:
        pl.DataFrame: A new DataFrame with eligible string columns cast to
            pl.Categorical. Shape and column names are unchanged.

    Example:
            >>> import polars as pl
            >>> df = pl.DataFrame({
            ...     "country": ["ENG", "WAL", "SCT", "ENG"],  # 3 unique — cast to Categorical
            ...     "value":   [1, 2, 3, 4],                  # non-string — unchanged
            ... })
            >>> create_categoricals(df, categorical_threshold=5).dtypes
            [Categorical, Int64]
    """

    dtype_dict = {}

    # Restrict evaluation to string columns only
    df_stronly = df.select(cs.string())

    # Identify string columns with fewer unique values than the threshold
    all_categoricals = df_stronly.select(
        col.name for col in
        df_stronly.select(pl.all().n_unique() < categorical_threshold)
        if col.all()
    )

    cols_to_categorical = all_categoricals.columns

    # Build a column → Categorical mapping for all eligible columns
    for col in cols_to_categorical:
        dtype_dict[col] = pl.Categorical

    # Apply all casts in one operation
    df = df.cast(dtype_dict)

    return df

def fix_recid_errors(
    df,
    recid_corrections_file,
    census_year,
    recid_field="recid",
    hid_field="hid",
    country_field="Country",
):
    """
    Corrects known errors in the unqiue id (recid) field by
    replacing them with the correct recid value, using a JSON corrections file as ground truth.

    Some census files have duplicate recids of 999999 possibly introduced at some point due to the use of 999999 
    as a Null signifier (see `replace_numeric_null_values()`.
    
    This function loads a pre-defined set of corrections
    keyed by census year and country, identifies affected rows by matching on 
    'recid', 'hid', and 'Country'. It replaces the erroneous 999999 with the
    correct value of 99999. After each correction, uniqueness of recid is
    verified via `verify_unique_recid()`.

    Args:
        df (pl.DataFrame): The input Polars DataFrame containing the records to
            be corrected.
        recid_corrections_file (str | Path): Path to a JSON file mapping census
            years to the specific 'hid' record for each country that needs its 'recid' changing.

            {"1851": 
                {"ENG": 51418, 
                    "SCT":45837}
                    }

        census_year (int | str): The census year whose corrections should be
            applied.
        recid_field (str): Name of the record ID column. Defaults to "recid".
        hid_field (str): Name of the household ID column used to identify the
            affected row within a country. Defaults to "hid".
        country_field (str): Name of the country column. Defaults to "Country".

    Returns:
        pl.DataFrame: A new DataFrame with erroneous recid 999999 values
            replaced by 99999 for all matching rows. All other columns and
            rows are unchanged.

    """

    # Load the full corrections dictionary from the JSON file
    with open(recid_corrections_file) as f:
        recid_corrections_dct = json.load(f)

    # Extract corrections relevant to the given census year
    country_dict = recid_corrections_dct.get(str(census_year))

    for country, hid_value in country_dict.items():

        # Replace recid 999999 with 99999 for rows matching all three conditions
        df = df.with_columns(
            pl.when(
                (pl.col(country_field) == country) &
                (pl.col(hid_field) == hid_value) &
                (pl.col(recid_field) == 999999)
            )
            .then(99999)
            .otherwise(pl.col("recid"))
            .alias("recid")
        )

        # Verify recid uniqueness after each correction to check fix applied correctly
        verify_unique_recid(df)

    return df


def verify_unique_recid(df, recid_field = "recid", country_field = "Country", ):
    for country in ["SCT", "ENG", "WAL", "IBS"]:

        recid_check_df = df.filter((pl.col(recid_field).is_duplicated()) & (pl.col(country_field) == country))
    if not recid_check_df.is_empty():
        print(recid_check_df)
        raise ValueError("Duplicate RecID values found")
    

def verify_unique_recid(df, recid_field="recid", country_field="Country"):
    """
    Verifies that the record ID (recid) field contains no duplicate values
    within each country, raising an error if any are found.

    Iterates over a fixed list of country codes and filters for rows where
    recid is duplicated within that country. If any duplicates are found,
    the offending rows are printed and a ValueError is raised.

    Args:
        df (pl.DataFrame): The input Polars DataFrame to validate.
        recid_field (str): Name of the record ID column to check for
            duplicates. Defaults to "recid".
        country_field (str): Name of the country column used to scope each
            duplicate check. Defaults to "Country".

    Returns:
        None: This function is called for its side effects (validation and
            error raising) only. It does not return a modified DataFrame.

    Raises:
        ValueError: If any duplicate recid values are found in any country.
            The offending rows are printed to stdout before the error is
            raised, to aid diagnosis.

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "recid":   [1, 1, 3, 4],
        ...     "Country": ["ENG", "ENG", "SCT", "WAL"],
        ... })
        >>> verify_unique_recid(df)
        shape: (2, 2)
        ┌───────┬─────────┐
        │ recid ┆ Country │
        │ ---   ┆ ---     │
        │ i64   ┆ str     │
        ╞═══════╪═════════╡
        │ 1     ┆ ENG     │
        │ 1     ┆ ENG     │
        └───────┴─────────┘
        ValueError: Duplicate RecID values found

    """

    for country in ["SCT", "ENG", "WAL", "IBS"]:

        recid_check_df = df.filter(
            (pl.col(recid_field).is_duplicated()) & (pl.col(country_field) == country)
        )

    if not recid_check_df.is_empty():
        print(recid_check_df)
        raise ValueError("Duplicate RecID values found")
    
def fix_encoding_errors(df):
    """
    Removes non-printable and invisible control characters from all string
    columns in a Polars DataFrame. These are present in the I-CeM downloads (https://icem.ukdataservice.ac.uk/).

    Applies a regex replacement across every string column, stripping any
    character matching the pattern of common encoding problems that aren't read by UTF-8 encoding.

    Args:
        df (pl.DataFrame): The input Polars DataFrame. Only columns with the
            String dtype are affected; all other dtypes are left
            unchanged.

    Returns:
        pl.DataFrame: A new DataFrame with the target characters removed from
            all string columns. Shape and column names are unchanged, though
            string values may be shorter where characters were removed.

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "name":  ["Eng\\x00land", "Sco\\x1Ftland", "Wale\\xs"],
        ...     "recid": [1, 2, 3],
        ... })
        >>> fix_encoding_errors(df)
        shape: (3, 2)
        ┌──────────┬───────┐
        │ name     ┆ recid │
        │ ---      ┆ ---   │
        │ str      ┆ i64   │
        ╞══════════╪═══════╡
        │ England  ┆ 1     │
        │ Scotland ┆ 2     │
        │ Wales    ┆ 3     │
        └──────────┴───────┘

    Notes:
        - The regex pattern [\\x00-\\x1F\\x7F\\xA0] targets three character ranges:
              \x00-\x1F:the 32 ASCII control characters (null, tab, carriage
                           return, line feed, escape, etc.)
              \\x7F: the DEL character
              \\xA0: non-breaking space (a common artefact in data copied
                           from web sources or PDFs)
    """

    # Strip non-printable and invisible control characters from all string columns
    df = df.with_columns(pl.col(pl.String).str.replace('[\x00-\x1F\x7F\xA0]', ""))

    return df
