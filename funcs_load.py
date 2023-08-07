import xarray as xr
import xagg as xa
import numpy as np
import pandas as pd
import os
import glob
import re
import warnings

from funcs_support import get_params,get_filepaths
dir_list = get_params()


def load_raws(search_params,
              subset_params = {},
              source = 'raw',
              manually_decode_dates = False,
              drop_list = ['lon_bounds','lat_bounds','time_bounds',
                           'lat_bnds','lon_bnds','time_bnds'],
              key_hierarchy = ['exp','time'],
              force_return_dict = False,
              force_key = 'model',
              silent=False,
              **open_kwargs):
    """ Load data from multiple models
    
    Workflow:
        1. Get all filepaths in dir_list[source] through
           `get_filepaths()`
        2. Subset filepaths using `search_params`
        3. Put into a `pd.DataFrame`, with a (multi-)index
           based on which filename parameters can't be 
           used to merge or concatenate the resultant files
        4. For each unique index combination, call
           `xr.open_mfdataset()` with an index for all 
           `concat_columns`
        5. Output a dictionary, with keys as the unique 
           index combinations
    
    Parameters:
    -------------
    search_params : dict
        Of the form, e.g.,: 
            ```
            search_params = {'varname':'T',
                             'run':'r1i1p1',
                             'exp':'historical'}
            ```
        Use the `get_cam6_filepaths()` .csv file's columns
        as dictionary keys.
        
    subset_params : dict, by default {}
        Files are individually subset before concatenation using
        `.sel(**subset_params)`.
        
    source : str, by default 'raw'
        Piped into `get_filepaths(source_dir=source)`
        
    drop_list : list, by default ['lon_bounds','lat_bounds','time_bounds']
        Drops variables (with `errors='ignore'`) from loaded datasets
        
    silent : bool, by default False
        If True, suppresses std out printing (currently not in use) 
        
    Returns:
    -------------
    dss : dict of `xr.Dataset`s
    
    """

    #----------- Setup -----------
    # Define which columns can be easily merged over, 
    # either because of variable names or because they
    # can be dimensions
    merge_columns = ['varname']
    concat_columns = ['run']


    # Function to flatten list of arbitrary depth, from 
    # https://stackoverflow.com/questions/2158395/flatten-an-irregular-arbitrarily-nested-list-of-lists
    from collections.abc import Iterable
    def flatten(xs):
        for x in xs:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from flatten(x)
            else:
                yield x

    #----------- Find files -----------
    # Get all mods
    mods = [re.split('\/',mod)[-1] for mod in glob.glob(dir_list['raw']+'*')]

    # Get all filepaths for all files
    fns_all = pd.concat([get_filepaths(source_dir='raw',mod=mod) for mod in mods])

    fns = fns_all.loc[np.product(np.array([fns_all[k] == v for k,v in search_params.items()]),axis=0).astype(bool),:]
    # Get which columns only have one item and therefore don't need to be dims
    fns = fns.drop(columns = fns.columns[fns.apply(lambda x: np.all([v is None for v in x]) 
                                                       or (len(np.unique(x))==1),axis=0)])


    # Get columns which will be dictionary keys instead of 
    # dimensions in the dataset (since they would affect 
    # the merge - different grids, areas, timeframes,...)
    key_columns = [col for col in fns.columns if col not in ['path',*merge_columns,*concat_columns]]
    
    # If some of the key columns don't provide unique information
    # (i.e., if for example a certain experiment only shows up 
    # together with a certain timeframe), then only keep as keys
    # unique values, following the `key_hierarchy` parameter to 
    # decide which key to keep 
    if len(np.unique([fns[key_columns].drop_duplicates().shape[0],
                      *[fns[k].drop_duplicates().shape[0] for k in key_columns]]))==1:
        # Find the highest ranked (earliest-appearing in key_hierarchy) 
        # column of the columns desired 
        try:
            keep_key = key_hierarchy[next(index for index, item in enumerate(key_hierarchy) if item in key_columns)]
        except:
            if not silent:
                warnings.warn("Tried to declutter the ouptut dictionary keys using the `key_hierarchy`, "+
                              "but none of the `key_columns` ("+', '.join(key_columns)+") are in the `key_hierarchy`.")
            pass
        fns = fns.drop(columns=[c for c in key_columns if c != keep_key])

    # Set multi-index to all non-path variables
    index_cols = list(np.sort([c for c in fns.columns if c not in ['path',*merge_columns,*concat_columns]]))
    if len(index_cols)>0:
        fns = fns.set_index(index_cols)
    
    # For speed
    fns = fns.sort_index()

    #----------- Load files -----------
    # Now load
    if len(index_cols)>0:
        # Use indices to load data into separate dictionary
        # keys based on which parameters can't be merged
        # or concatenated
        dss = {idx:(xr.open_mfdataset(flatten(fns.loc[idx][['path']].values.tolist()),
                                     concat_dim=[pd.Index(flatten(fns.loc[idx][[col]].values),name=col) 
                                                 for col in concat_columns],
                                     combine='nested',**open_kwargs).drop(drop_list,errors='ignore').
                    sel(**subset_params))
                for idx in np.unique(fns.index)}
    else:
        # Return one dataset if dictionary structure
        # not needed (i.e., all data have compatible
        # dimensions)
        dss = (xr.open_mfdataset(flatten(fns[['path']].values.tolist()),
                                     concat_dim=[pd.Index(flatten(fns[[col]].values),name=col) 
                                                 for col in concat_columns],
                                     combine='nested',**open_kwargs).drop(drop_list,errors='ignore').
                    sel(**subset_params))
        
        # If the data have to be returned as a dict, 
        # use the desired search_params key to turn
        # it into a dict of one key, value pair
        if force_return_dict:
            if force_key in search_params:
                dss = {search_params[force_key]:dss}
            else:
                warnings.warn("through force_return_dict = True, no '"+force_key+
                              "' entry was found in `search_params`. A dataset will be returned instead.")
            

    #----------- Return -----------
    return dss