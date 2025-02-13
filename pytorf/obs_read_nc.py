def obs_read_nc(index, 
                categories="flask",
                solar_time=False,
                as_list=False,
                verbose=False):
    
    if len(index) == 0:
        raise ValueError("empty index")
    
    if verbose:
        print(f"Searching {categories}...")
        
    df = index[index["sector"] == categories]
    
    # x1 has all filess with full name
    x1 = df["id"].tolist()
    
    
    # time names
    time_names = ["year", "month", "day", "hour", "minute", "second"]
    nt = time_names
    for i in range(len(time_names)):
        nt[i] = nt[i] + "_sl"
        
    # list will store different DataFrames
    lx = []
    
    # loop over the NetCDF files
    for j in range(len(x1)):
    
        agl = df.iloc[j]["agl"]
        n = df.iloc[j]["n"]
        print(f"agl: {agl} n: {n}")
        
        nc = Dataset(x1[j])
        
        if verbose:
                print('NC data:')
                #print(nc)
                
        vars_names = [var for var in nc.variables]        
        
        if verbose:
                print('NC vars:')
                print(vars_names)
                
        
        na = pd.DataFrame({"vars": vars_names})

        la = {var: nc.variables[var].__dict__ for var in vars_names}
        
        lv = {var: nc.variables[var][:] for var in vars_names}
        
        d = pd.DataFrame({"dim": [len(lv[var].shape) for var in lv], 
                  "names": list(lv.keys())})
        if verbose:
                print(d)
                
        x2 = nc.variables["time_components"][:]
        if verbose:
                print(x2)
                
        dt = pd.DataFrame(x2, columns = time_names)
        if verbose:
                print(dt)
                
        if solar_time:
                x3 = nc.variables["solartime_components"][:]
                dtsl = pd.DataFrame(x3, columns=nt)

        xx = d[d['dim'] == 1]
        if verbose:
                print(xx)
                
        # DataFrame
        for i, row in xx.iterrows():
                dt[row["names"]] = lv[row["names"]]
                
        if verbose:
                print(dt.columns.values)
        
        # Adding scale
        dt["scale"] = la["value"]["scale_comment"]

        if solar_time:
                for col in dtst.columns:
                        dt[col] = dtsl[col]
            
        # adding NetCDF global attributes            
        global_attrs = {key: getattr(nc, key) for key in nc.ncattrs()}
        if verbose:
                print(global_attrs)
                
        for key, value in global_attrs.items():
                dt[key] = value
                
                
        
        lx.append(dt)
    
    # identifying unique names in each DataFrame
    unames = pd.unique([col for sublist in lx for col in sublist.columns])
    
    # Add missing columns to each dataframe
    for i in range(len(lx)):
        ly = [col for col in unames if col not in lx[i].columns]
        if ly:
            for col in ly:
                lx[i][col] = pd.NA

    type_altitude = None

    if as_list:
        return lx
    else:
        dt = pd.concat(lx, ignore_index=True)
    
    # No need to add id column as pandas automatically assigns index
    return dt