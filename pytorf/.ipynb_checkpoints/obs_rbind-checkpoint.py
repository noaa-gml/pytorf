
def obs_rbind(dt1: Frame, dt2: Frame, verbose: bool = True) -> Optional[Frame]:
    """
    Binds two datatable Frames by row, aligning columns.

    Args:
        dt1: First datatable Frame.
        dt2: Second datatable Frame.
        verbose: If True, print information about common names and type mismatches.

    Returns:
        A combined datatable Frame, or None if errors occur.
    """
    if not DT_AVAILABLE:
        raise ImportError("datatable library is required for obs_rbind.")
    if not isinstance(dt1, Frame) or not isinstance(dt2, Frame):
        raise TypeError("Inputs must be datatable Frames.")

    names1 = set(dt1.names)
    names2 = set(dt2.names)
    common_names = sorted(list(names1.intersection(names2)))
    all_names = sorted(list(names1.union(names2)))

    if verbose:
        print("Identifying common names...")
        print(f"Common: {common_names}")
        print(f"Only in dt1: {sorted(list(names1 - names2))}")
        print(f"Only in dt2: {sorted(list(names2 - names1))}")

    # Check for type mismatches in common columns (basic check)
    mismatched_types = []
    if verbose: print("\nComparing classes for common columns:")
    for name in common_names:
        type1 = dt1.stypes[name]
        type2 = dt2.stypes[name]
        if type1 != type2:
            mismatched_types.append((name, type1, type2))
            if verbose: print(f"- {name}: {type1} vs {type2} <-- MISMATCH")
        # else:
        #     if verbose: print(f"- {name}: {type1} (OK)")

    if mismatched_types:
        warnings.warn(f"Type mismatches found for columns: {[m[0] for m in mismatched_types]}. rbind might coerce types or fail.")

    try:
        # datatable's rbind handles column alignment and fills missing with NA
        combined_dt = dt.rbind(dt1, dt2, force=True) # force=True might help with type coercion
        return combined_dt
    except Exception as e:
        print(f"Error during rbind: {e}")
        # Optional: Implement manual alignment as a fallback if needed,
        # similar to the fallback logic in obs_read/obs_read_nc,
        # but rbind itself is generally robust.
        return None
