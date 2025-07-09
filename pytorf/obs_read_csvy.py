def obs_read_csvy(f: Union[str, Path], n_header_lines: int = 100, **kwargs) -> Tuple[Optional[Dict], Optional[Frame]]:
    """
    Reads a CSVY file, prints YAML header, and returns header dict and data frame.

    Args:
        f: Path to the CSVY file.
        n_header_lines: Max lines to search for YAML delimiters.
        **kwargs: Additional arguments passed to datatable's `fread`.

    Returns:
        A tuple containing: (YAML header as dict or None, data as datatable Frame or None).
    """
    if not DT_AVAILABLE:
        raise ImportError("datatable library is required for obs_read_csvy.")

    f_path = Path(f)
    yaml_content = []
    in_yaml = False
    header_data = None
    dt_frame = None
    skip_rows = 0 # 0-based line number *after* the second '---'

    try:
        with f_path.open('r', encoding='utf-8') as file:
            found_delimiters = 0
            for i, line in enumerate(file):
                current_line_num = i + 1 # 1-based line number
                if line.strip() == '---':
                    found_delimiters += 1
                    if found_delimiters == 1:
                        in_yaml = True
                    elif found_delimiters == 2:
                        skip_rows = current_line_num # Next line is data header/start
                        break # Found end of YAML
                elif in_yaml:
                    yaml_content.append(line)

                if current_line_num >= n_header_lines * 2: # Safety break
                    warnings.warn(f"YAML delimiters '---' not found within expected lines in {f_path}. Reading might be incorrect.")
                    # Attempt to read from beginning if delimiters not found? Risky.
                    skip_rows = 0 # Assume no header if delimiters not found
                    yaml_content = [] # Discard potentially partial YAML
                    break
            else: # Loop finished without break (EOF before second '---')
                if found_delimiters == 1:
                     warnings.warn(f"YAML end delimiter '---' not found in {f_path}. Reading might be incorrect.")
                     skip_rows = current_line_num # Read after the last line scanned
                     yaml_content = [] # Discard partial YAML


        if yaml_content:
            try:
                # Use safe_load for security
                header_data = yaml.safe_load("".join(yaml_content))
                print("--- YAML Header ---")
                print("".join(yaml_content).strip())
                print("-------------------")
            except yaml.YAMLError as ye:
                 warnings.warn(f"Error parsing YAML header in {f_path}: {ye}")
                 header_data = {"error": "YAML parsing failed", "raw": "".join(yaml_content)}
                 print("--- Raw Header ---")
                 print("".join(yaml_content).strip())
                 print("------------------")

        # Read the data part using datatable, skipping the YAML header
        # skip_to_line expects 1-based line number
        dt_frame = dt.fread(str(f_path), skip_to_line=skip_rows + 1, **kwargs)

    except FileNotFoundError:
        print(f"Error: File not found {f_path}")
    except Exception as e:
        print(f"Error reading CSVY file {f_path}: {e}")

    return header_data, dt_frame
