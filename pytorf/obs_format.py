def obs_format(
    dt_frame: Frame,
    spf: List[str] = ["month", "day", "hour", "minute", "second",
                     "month_end", "day_end", "hour_end", "minute_end", "second_end"],
    spffmt: str = "{:02d}", # Python format spec mini-language
    rnd: List[str] = ["latitude", "longitude"],
    rndn: int = 4,
    spfrnd: bool = True, # Apply sprintf-style formatting after rounding?
    spf_rnd_fmt: str = "{:.4f}", # Format for rounded columns if spfrnd=True
    out: Optional[Union[str, Path]] = None,
    **kwargs
) -> Frame:
    """
    Formats specific columns in a datatable Frame (rounding, padding).

    Args:
        dt_frame: The input datatable Frame.
        spf: List of columns to format using `spffmt` (typically integers needing padding).
        spffmt: Python format string specifier (e.g., '{:02d}' for 2-digit padding).
        rnd: List of columns to round using `rndn` decimals.
        rndn: Number of decimal places for rounding.
        spfrnd: If True, format the rounded columns using `spf_rnd_fmt`.
        spf_rnd_fmt: Python format string specifier for rounded columns (e.g., '{:.4f}').
        out: Optional output path to save the formatted Frame using `fwrite`.
        **kwargs: Additional arguments for `fwrite`.

    Returns:
        The modified datatable Frame (changes are done in-place).
    """
    if not DT_AVAILABLE:
        raise ImportError("datatable library is required for obs_format.")
    if not isinstance(dt_frame, Frame):
        raise TypeError("Input dt_frame must be a datatable Frame.")

    # Format columns in spf (padding integers)
    for col_name in spf:
        if col_name in dt_frame.names:
            try:
                # Ensure column is integer-like first, handle NAs
                temp_list = [spffmt.format(int(x)) if x is not None and not math.isnan(x) else None
                             for x in dt_frame[col_name].to_list()[0]]
                # Update column - will likely become string type
                dt_frame[:, update(**{col_name: dt.Frame(temp_list)[f[0]]})]
            except (ValueError, TypeError) as e:
                warnings.warn(f"Could not apply format '{spffmt}' to column '{col_name}': {e}. Skipping.")
        else:
            warnings.warn(f"Column '{col_name}' not found for formatting. Skipping.")

    # Round columns in rnd
    for col_name in rnd:
        if col_name in dt_frame.names:
            try:
                # datatable doesn't have a direct round function easily usable in update
                # Extract, round using numpy/math, then update
                col_data = dt_frame[col_name].to_numpy()
                # Use np.round for potentially better handling of different types/NAs
                rounded_data = np.round(col_data.astype(float), rndn) # Convert to float first

                if spfrnd:
                     # Format after rounding
                     formatted_list = [spf_rnd_fmt.format(x) if x is not None and not np.isnan(x) else None
                                       for x in rounded_data]
                     # Update column - will likely become string type
                     dt_frame[:, update(**{col_name: dt.Frame(formatted_list)[f[0]]})]
                else:
                     # Update with rounded numeric data
                     dt_frame[:, update(**{col_name: dt.Frame(rounded_data)[f[0]]})]

            except Exception as e: # Catch broader errors during numpy conversion/rounding
                 warnings.warn(f"Could not round or format column '{col_name}': {e}. Skipping.")
        else:
            warnings.warn(f"Column '{col_name}' not found for rounding. Skipping.")

    # Save if output path provided
    if out:
        try:
            out_path = Path(out)
            # Note: fwrite doesn't exist in datatable. Use to_csv.
            dt_frame.to_csv(str(out_path), **kwargs)
        except Exception as e:
            print(f"Error saving formatted data to {out}: {e}")

    return dt_frame # Return the modified frame
