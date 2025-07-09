
def obs_write_csvy(
    dt_frame: Frame,
    notes: List[str],
    out: Union[str, Path],
    sep: str = ",",
    # nchar_max: int = 80, # Less relevant for Python structure printing
    **kwargs
):
    """
    Writes a datatable Frame to a CSVY file with YAML header.

    Args:
        dt_frame: The datatable Frame to write.
        notes: A list of strings to include as notes in the YAML header.
        out: The output file path.
        sep: The CSV separator.
        **kwargs: Additional arguments passed to datatable's `to_csv`.
    """
    if not DT_AVAILABLE:
        raise ImportError("datatable library is required for obs_write_csvy.")
    if not isinstance(dt_frame, Frame):
        raise TypeError("Input dt_frame must be a datatable Frame.")

    out_path = Path(out)
    metadata = {
        'name': 'Metadata',
        'notes': notes,
        'structure': {
            'rows': dt_frame.nrows,
            'columns': dt_frame.ncols,
            'names': dt_frame.names,
            'types': [str(t) for t in dt_frame.stypes] # Store datatable stypes
        },
        'generated_by': 'rtorf Python package',
        'timestamp': pydt.datetime.now(pydt.timezone.utc).isoformat()
    }

    try:
        with out_path.open('w', encoding='utf-8') as f:
            f.write("---\n")
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, indent=2)
            f.write("---\n")

        # Append the data using datatable's efficient writer
        dt_frame.to_csv(str(out_path), sep=sep, header=True, append=True, **kwargs)

    except Exception as e:
        print(f"Error writing CSVY file {out_path}: {e}")
