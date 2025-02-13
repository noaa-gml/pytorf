
def obs_summary(obs, 
                categories=["aircraft-pfp", 
                            "aircraft-insitu", 
                            "surface-insitu",
                            "tower-insitu", 
                            "aircore",
                            "surface-pfp",
                            "shipboard-insitu",
                            "flask"], 
                out=None, 
                verbose=True):
    """Adds two numbers together.
    Args:
    obs: path to obspack.
    categories: sectors to be identified in file name.
    out: File name to write the obspack file names.
    verbose: Logical to print more information.
    
    Returns:
    DataFrame with ObsPack file names.
    """
    
    # read full path for each archive
    x = [os.path.join(obs, f) for f in os.listdir(obs)]
    
    # read each archive
    na = [f for f in os.listdir(obs)]
    
    # generate dataframe, order by name and add id
    index = pd.DataFrame({'id': x, 'name': na})
    index = index.sort_values(by='name')
    
    # Add column for number of files in each category
    index['n'] = range(1, len(index) + 1)
    
    # identify category names in index
    for category in categories:
        index.loc[index["name"].str.contains(category), "sector"] = category

    # print information for verbose
    if verbose:
        print(f"Number of files of index: {len(index)}")
        xx = index.groupby("sector").size().reset_index(name="N")
        dx = pd.DataFrame({"sector": ["Total sectors"], "N": [xx["N"].sum()]})
        print(pd.concat([xx, dx], ignore_index=True))
    
    # detecting file extension
    fiex = os.path.splitext(index["id"][0])[1][1:]
    
    idfiex = 10 if fiex == "nc" else 11
    
    # add the last 11 characters of each name file'
    index.loc[index["id"].str.contains("magl"), "agl"] = index.loc[index["id"].str.contains("magl"), "id"].apply(lambda x: x[-idfiex:])
    
    # This line replaces removes the characters magl.txt
    # for instance, remove "magl.txt" from -11magl.txt
    
    index["agl"] = index["agl"].str.replace(f"magl.{fiex}", "", regex=True)
    
    # assuming d-{number}magl.txt
    index["agl"] = index["agl"].str.replace("d", "", regex=True)

    # Now we transform the column
    # then get the absolute number and now we have magl
    index["agl"] = index["agl"].apply(lambda x: abs(float(x)) if pd.notnull(x) else x)
    
    if out is not None:
        index.to_csv(out, index=False)
        if verbose:
            print(f"index in: {out}")

    # identifying agl when reading the data instead of name file
    # 2023/09/13
    # it mauy be removed, because the defualt approach will be
    # using only NetCDF, which 
    if verbose:
        yai = index["agl"].notnull()
        nai = index["agl"].isnull()
        print(f"Detected {yai.sum()} files with agl")
        print(f"Detected {nai.sum()} files without agl")

    return index