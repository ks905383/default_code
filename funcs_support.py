import xarray as xr
import xesmf as xe
import xagg as xa
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
import os
import glob
import warnings

class NotUniqueFile(Exception):
    """ Exception for when one file needs to be loaded, but the search returned multiple files """
    pass


def get_params():
    ''' Get parameters 
    
    Outputs necessary general parameters. 
    
    Parameters:
    ----------------------
    (none)
    
    
    Returns:
    ----------------------
    dir_list : dict()
        a dictionary of directory names for file system 
        managing purposes: 
            - 'raw':   where raw climate files are stored, in 
                        subdirectories by model/product name
            - 'proc':  where processed climate files are stored,
                        in subdirectories by model/product name
            - 'aux':   where aux files (e.g. those that transcend
                        a single data product/model) are stored
    '''

    # Dir_list
    dir_list = pd.read_csv('dir_list.csv')
    dir_list = {d:dir_list.set_index('dir_name').loc[d,'dir_path'] for d in dir_list['dir_name']}


    # Return
    return dir_list

dir_list = get_params()

def get_filepaths(source_dir = 'proc' ,
                  mod = 'CAM6',
                  dir_list = dir_list,
                  col_namer = {'(hadley$)|(CMIP[0-9]$)':'forcing_dataset',
                                 'PDO$':'pdo_state',
                                     'AMO$':'amo_state'}):
    ''' Get filepaths of climate data, split up by CMIP filename component
    
    
    Uses modified CMIP5/6 filename standards used by Kevin Schwarzwald's 
    filesystem - in other words, with the additional optional "suffix" 
    between the daterange and the filetype extension. 
    
    Returns
    ------------
    df : pd.DataFrame
        A dataframe containing information for all files in 
        `dir_list[source_dir]/mod/*.nc`, with the full filepath in the
        column `path`, and filename components `varname`, `freq`, 
        `model`, `exp`, `run`, `grid`, `time`, `suffix`, in their own
        columns. `grid` may be Nones if files use CMIP5 conventions, 
        `suffix` may be Nones if no suffixes are found. 
        
        If `exp` has a match for the regex "\-", then additionally
        extra columns for each experiment name component will be 
        created, if possible, using the `col_namer` input.
    
    
    '''

    #---------- Get list of files ----------
    # Get list of subdirectories
    fns = glob.glob(dir_list[source_dir]+mod+'/*.nc')

    def id_fncomps(comps,col_namer=col_namer):
        # Make sure there are enough components 
        if len(comps)<6:
            # For now - but there has to be a better way to 
            # flag this
            slots = {'varname':None}
        else:
            # Prepopulate set components
            slots = {s:n for n,s in zip(np.arange(0,5),['varname','freq','model','exp','run'])}

            # Get which slot is the timeframe 
            slots['time'] = np.where([re.search('[0-9]{4,8}\-[0-9]{4,8}',comp) for comp in comps])[0][0]

            # Use the time position to determine whether
            # there's a grid slot (CMIP6) or not (CMIP5)
            if slots['time'] == 5:
                slots['grid'] = None
            elif slots['time'] == 6:
                slots['grid'] = 5

            # Use whether the file extension is in the time
            # or one after slot to determine whether there's a 
            # suffix slot
            if np.where([re.search('\.nc',comp) for comp in comps])[0][0] == slots['time']:
                slots['suffix'] = None
            elif np.where([re.search('\.nc',comp) for comp in comps])[0][0] == (slots['time']+1):
                slots['suffix'] = slots['time']+1

            # Now, assign slots to the components
            slots = {k:re.sub('\.nc$','',comps[s]) if s is not None else None for k,s in slots.items()}

            # If the experiment slot has multiple sub-experiments,
            # save them seperately using the column namer dict
            exp_comps = re.split('\-',slots['exp'])
            if len(exp_comps)>1:
                for exp_comp in exp_comps:
                    if np.any([re.search(k,exp_comp) for k in col_namer]):
                        match_type = [v for k,v in col_namer.items() if re.search(k,exp_comp)]
                        if len(match_type) > 1:
                            warnings.warn('More than one column match found for '+exp_comp+'. Check col_namer, no exp has been split.')
                        else:
                            slots[match_type[0]] = exp_comp

        return slots

    # Split up filename by components
    fn_comps = [re.split('\_',re.split('\/',fn)[-1]) for fn in fns]
    # Identify components, concatenate with path
    df = pd.DataFrame([id_fncomps(comps) for comps in fn_comps])
    df = pd.concat([df,pd.DataFrame([{'path':fn} for fn in fns])],axis=1)

    #---------- Return ----------
    return df


# The next two are from https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)

    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5)
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5
        )

    return r

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda

def area_mean(ds):
    """ Calculate area-weighted mean of all variables in a  dataset
    
    Mean over lat / lon, weighted by the relative size of each
    pixel, dependent on latitude. Only weights by latitude, does
    not take into account lat/lon bounds, if present. 
    
    Parameters
    ------------------
    ds : xr.Dataset
    
    Returns
    ------------------
    dsm : xr.Dataset
        The input dataset, `ds`, averaged.
    
    """

    # Calculate area in each pixel
    weights = area_grid(ds.lat,ds.lon)

    # Remove nans, to make weight sum have the right magnitude
    weights = weights.where(~np.isnan(ds))

    # Calculate mean
    ds = ((ds*weights).sum(('lat','lon'))/weights.sum(('lat','lon')))

    # Return 
    return ds


def utility_print(output_fn,formats=['pdf','png']):
    if 'pdf' in formats:
        plt.savefig(output_fn+'.pdf')
        print(output_fn+'.pdf saved!')

    if 'png' in formats:
        plt.savefig(output_fn+'.png',dpi=300)
        print(output_fn+'.png saved!')

    if 'svg' in formats:
        plt.savefig(output_fn+'.svg')
        print(output_fn+'.svg saved!')


def get_varlist(source_dir=None,var=None,varsub='all',
                experiment=None,freq=None,
                empty_warnings=False):
    ''' Get a list of which models have which variables
    
    Searches the filesystem for all models (directory names) and 
    all variables (first part of filenames, before the first 
    underscore), and returns either that information for all 
    models and variables, or an array of models that have 
    files for specified variables. 
    
    NB: if no experiment or frequency is specified, and the
    full dataframe is returned (`var=None`), then the fields
    have True whenever any file with that variable in the filename
    for that model is present (and potentially more than one). 
    In general, the code does not differentiate between multiple
    files for a single model/variable combination. 
    
    Parameters
    ---------------
    source_dir : str; default dir_list['raw']
        a path to the directory with climate data (all 
        subdirectories are assumed to be models, all files in
        these directories are assumed to be climate data files
        in rough CMIP format).
        
    var : str, list; default `None`
        one variable name or a list of variables for which to 
        subset the model list of. If not `None`, then only a list
        of models for which this variable(s) is present is returned
        (instead of the full Dataframe).
        
    varsub : str; default 'all'
        - if 'all', then if `var` has multiple variables, 
          only models that have files for all of the variables 
          are returned
        - if 'any', then if `var` has multiple variables, 
          models that have files for any of the variables are 
          returned
          
    experiment : str; default `None`
        if not None, then only returns models / True if files
        for the given 'experiment' (in CMIP6 parlance, the 
        fourth filename component) are found. If not None, the
        variable is piped into re.search(), allowing for re
        searches for the experiment. 
        
    freq : str; default `None`
        if not None, then only returns models / True if files
        for the given 'frequency' (in CMIP6 parlance, the 
        second filename component) are found. If not None, the
        variable is piped into re.search(), allowing for re
        searches for the frequency. 
        
    empty_warnings : bool; default `False`
        if True, a warning is thrown if no files at all (before 
        subsetting) are found for a model. 
    
    
    Returns
    ---------------
    varindex : pd.DataFrame()
        if `var` is None, then a models x variables pandas
        DataFrame is returned, with `True` if that model has 
        a file with that variable, and `False` otherwise.
        
    mods : list
        if `var` is not None, then a list of model names 
        that have the variables, subject to the subsetting above
    
    
    '''
    if source_dir is None:
        dir_list = get_params()
        source_dir = dir_list['raw']
    
    
    ##### Housekeeping
    # Ensure the var input is a list of strings, and not a string
    if type(var) == str:
        var = [var]
    
    ##### Identify models
    # Figure out in which position of the filename path the model name
    # directory is located (based on how many directory levels there 
    # are in the parent directory)
    modname_idx = len(re.split('/',source_dir)) - 1
    # Get list of all the models (the directory names in source_dir)
    all_mods = [re.split('/',x)[modname_idx] for x in [x[0] for x in os.walk(source_dir)] if re.split('/',x)[modname_idx]!='']
    all_mods = [mod for mod in list(np.unique(all_mods)) if 'ipynb' not in mod]
    
    ##### Identify variables
    # Get list of all variables used and downloaded
    # Make this a pandas dataarray - mod x var
    varlist = []
    for mod in all_mods[:]:
        varlist.append([re.split('\_',fn)[0] for fn in [x for x in os.walk(source_dir+mod+'/')][0][2]])
    varlist = [item for sublist in varlist for item in sublist]

    varlist = list(np.unique(varlist))

    # Remove "README" and ".nc" files 
    varlist = [var for var in [var for var in varlist if 'READ' not in var] if '.nc' not in var]
    
    ##### Populate dataframe
    # Create empty dataframe to populate with file existence
    varindex = pd.DataFrame(columns=['model',*varlist])

    # Populate the model column
    varindex['model'] = all_mods

    # Actually, just set the models as the index
    varindex = varindex.set_index('model')
    
    # Now populate the dataframe with Trues if that model has that variable as a file
    for mod in all_mods:
        # Get variable name of each file 
        file_varlist = [re.split('\_',fn)[0] for fn in [x for x in os.walk(source_dir+mod+'/')][0][2]]

        if len(file_varlist) == 0:
            if empty_warnings:
                warnings.warn('No relevant files found for model '+mod)
            varindex.loc[mod] = False
        else:
            # Subset by frequency, or experiment, if desired
            if freq is not None:
                try:
                    freq_bools = [(re.search(freq,re.split('\_',fn)[1]) != None) for fn in [x for x in os.walk(source_dir+mod+'/')][0][2]]
                except IndexError:
                    freq_bools = [False]*len(file_varlist)
                    if empty_warnings:
                        warnings.warn('Model '+mod+' has files not in CMIP format.')
                    continue
            else:
                freq_bools = [True]*len(file_varlist)

            if experiment is not None:
                try:
                    exp_bools = [(re.search(experiment,re.split('\_',fn)[3]) != None) for fn in [x for x in os.walk(source_dir+mod+'/')][0][2]]
                except IndexError:
                    exp_bools = [False]*len(file_varlist)
                    if empty_warnings:
                        warnings.warn('Model '+mod+' has files not in CMIP format.')
                    continue
            else:
                exp_bools = [True]*len(file_varlist)

            # Remove from list if it doesn't fit the frequency/experiment subset
            file_varlist = list(np.asarray(file_varlist)[np.asarray(freq_bools) & np.asarray(exp_bools)])

            # Add to dataframe
            varindex.loc[mod] = [var in file_varlist for var in varlist]

    # Fill NaNs with False
    varindex = varindex.fillna(False)

    ##### Return
    if var is None: 
        return varindex
    else:
        if type(var) == str:
            var = [var]
        if varsub == 'all':
            # (1) is to ensure the `all` is across variables/columns, not rows/models
            return list(varindex.index[varindex[var].all(1)].values)
        elif varsub == 'any':
            return list(varindex.index[varindex[var].any(1)].values)
        else:
            raise KeyError(str(varsub) + ' is not a supported variable subsetting method, choose "all" or "any".')
            