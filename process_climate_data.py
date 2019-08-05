#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code that analyses the daily high resolution Australian Gridded Climate Data
(5 x 5 km) at site level. Different time analyses are performed and extreme
climate event metrics are also computed.

For more information on the climate data, see:
    * http://ozewex.org/database-single-record/?pdb=34
    * https://data.gov.au/dataset/ds-bom-ANZCW0503900567/details?q=

For more information on some common extreme climate metrics, see:
    * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.593.3104&rep=rep1&type=pdf
    * https://journals.ametsoc.org/doi/full/10.1175/JCLI-D-12-00383.1
    * https://htmlpreview.github.io/?https://raw.githubusercontent.com/ARCCSS-extremes/climpact2/master/user_guide/html/appendixA.htm

"""

__title__ = "AGCD site level analysis"
__author__ = "Manon Sabot"
__version__ = "1.0 (16.04.2019)"
__email__ = "m.e.b.sabot@gmail.com"


#==============================================================================

# import general modules
import os, sys # check for files, paths, version on the system
import gc # free memory when reading netcdfs
import warnings # deal with potential different numpy compiler versions

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    import numpy as np # array manipulations, math operators
    import numpy.ma as ma # masked arrays
    import pandas as pd # read/write dataframes, csv files
    import xarray as xr # read netcdf

    pd.options.mode.chained_assignment = None # deactivate annoying warning


#==============================================================================

def main(repo):

    """
    Main function: creates csv output files of the same form as the
                   species_list_location file. Specifically, analyses the timing
                   of meteorogical forcings on various scales (from day to
                   multi-year) and computes heatwave, dryness and rainfall
                   metrics for each of the sites.

    Arguments:
    ----------
    repo: string
        path to where input files are

    Returns:
    --------
    16 output files, 13 of which are timely analyses of the met forcings, 3 of
    which are index / metric files.

    """

    # site info file stored in an "input" dir in current directory
    filename = 'species_list_location_dates.csv'

    # read in csv
    df1 = read_csv(os.path.join(repo, os.path.join('input', filename)))

    # first collection date, in pandas interpretable format
    df1['Old seed collection date'] = [pd.to_datetime(dt).date() for dt in \
                                       df1['Old seed collection date']] 

    # modern collection, in pandas interpretable format
    df1['Modern seed collection date'] = [pd.to_datetime(dt).date() for dt in \
                                          df1['Modern seed collection date']]

    # sort by coor to avoid reading same climate netcdf file several times 
    df1.sort_values(by=['Lat', 'Lon'], inplace=True)

    # initialise output files
    df3 = df1.copy() # climate info of day-1
    df4 = df1.copy() # week climate leading onto the day
    df5 = df1.copy() # month climate leading onto the day
    df6 = df1.copy() # mid term climate leading onto the day (months)
    df7 = df1.copy() # five year climate leading in
    df8 = df1.copy() # climate metrics

    # make sure the appropriate columns are there, depending on output type
    df3 = prepare_new_df(df3, which='day')
    df4 = prepare_new_df(df4, which='week')
    df5 = prepare_new_df(df5, which='month')
    df6 = prepare_new_df(df6, which='3 months')
    df7 = prepare_new_df(df7, which='5 years')
    df8 = prepare_new_df(df8, which='metrics')

    for isite in range(len(df1)): # loop over all the sites

        lat = df1.iloc[isite].Lat
        lon = df1.iloc[isite].Lon

        # first time loop over a given coord? then read AWAP data (netcdf)
        if (isite == 0) or ((isite > 0) and (((lat != df1.iloc[isite - 1].Lat)
           or (lon != df1.iloc[isite - 1].Lon)) or 
           ((lat != df1.iloc[isite - 1].Lat) and
           (lon != df1.iloc[isite - 1].Lon)))):

            # file stored in an "input" dir in current directory
            filename = 'AWAP_met_%s_%s.nc' % (str(lat).rstrip('0'),
                                              str(lon).rstrip('0'))

            # read in netcdf
            df2 = read_netcdf(os.path.join(repo, os.path.join('input',
                                                              filename)))

            # time dimension in the netcdfs is corrupted, fix it
            oldest_collection = np.amin(df1[np.logical_and(df1.Lat == lat,
                                                           df1.Lon == lon)]
                                           ['Old seed collection date'])
            first_year = int(oldest_collection.year) - 5 # actual year
            df2.index = pd.date_range(start=pd.datetime(first_year, 1, 1),
                                      periods=len(df2), freq='D') # new index

            # vpd, convert hPa to kPa
            df2['vprp3pm'] /= 10.

            # using threshold for "no precip" (BoM def.) to mask no precip
            BoM_thr = 0.200000002980232 # mm
            df2['pre'] = ma.masked_where(df2['pre'] <= BoM_thr, df2['pre'])

            # adding midway temperature, i.e. median of tmin & tmax
            df2['tmdw'] = df2[['tmin', 'tmax']].median(axis=1)

        # fill in day climate df
        df3.iloc[isite] = day_forcings(df3.iloc[isite], df2)

        # fill in week climate df
        df4.iloc[isite] = other_period(df4.iloc[isite], df2)

        # fill in month climate df
        df5.iloc[isite] = other_period(df5.iloc[isite], df2, which='month')

        # fill in 3 months climate df
        df6.iloc[isite] = other_period(df6.iloc[isite], df2, which='3 months')

        # fill in 5 years climate df
        df7.iloc[isite] = other_period(df7.iloc[isite], df2, which='5 years')

        # fill in climate metrics
        df8.iloc[isite] = metrics(df8.iloc[isite], df2)

    # reset orginal order (by species name)
    df3.sort_values(by=['Species'], inplace=True)
    df4.sort_values(by=['Species'], inplace=True)
    df5.sort_values(by=['Species'], inplace=True)
    df6.sort_values(by=['Species'], inplace=True)
    df7.sort_values(by=['Species'], inplace=True)
    df8.sort_values(by=['Species'], inplace=True)

    # reset the date format to something nicer :)
    df3['Old seed collection date'] = [dt.strftime('%d/%m/%Y') for dt in \
                                       df3['Old seed collection date']] 
    df3['Modern seed collection date'] = [dt.strftime('%d/%m/%Y') for dt in \
                                          df3['Modern seed collection date']]
    df4['Old seed collection date'] = [dt.strftime('%d/%m/%Y') for dt in \
                                       df4['Old seed collection date']] 
    df4['Modern seed collection date'] = [dt.strftime('%d/%m/%Y') for dt in \
                                          df4['Modern seed collection date']]
    df5['Old seed collection date'] = [dt.strftime('%d/%m/%Y') for dt in \
                                       df5['Old seed collection date']] 
    df5['Modern seed collection date'] = [dt.strftime('%d/%m/%Y') for dt in \
                                          df5['Modern seed collection date']]
    df6['Old seed collection date'] = [dt.strftime('%d/%m/%Y') for dt in \
                                       df6['Old seed collection date']] 
    df6['Modern seed collection date'] = [dt.strftime('%d/%m/%Y') for dt in \
                                          df6['Modern seed collection date']]
    df7['Old seed collection date'] = [dt.strftime('%d/%m/%Y') for dt in \
                                       df7['Old seed collection date']] 
    df7['Modern seed collection date'] = [dt.strftime('%d/%m/%Y') for dt in \
                                          df7['Modern seed collection date']]
    df8['Old seed collection date'] = [dt.strftime('%d/%m/%Y') for dt in \
                                       df8['Old seed collection date']] 
    df8['Modern seed collection date'] = [dt.strftime('%d/%m/%Y') for dt in \
                                          df8['Modern seed collection date']]

    # save new dataframes to csv files
    filename = os.path.join(repo, os.path.join('output',
                                               'leading_day_all_met.csv'))
    df3.to_csv(filename, index=False, na_rep='', encoding='utf-8')

    filename = os.path.join(repo, os.path.join('output',
                                               'week_temperature.csv'))
    (df4[df4.columns[~df4.columns.str.contains('vpd|ppt')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output', 'week_vpd.csv'))
    (df4[df4.columns[~df4.columns.str.contains('tmdw|tmin|tmax|tair|ppt')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output', 'week_ppt.csv'))
    (df4[df4.columns[~df4.columns.str.contains('tmdw|tmin|tmax|tair|vpd')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output',
                                               'month_temperatude.csv'))
    (df5[df5.columns[~df5.columns.str.contains('vpd|ppt')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output', 'month_vpd.csv'))
    (df5[df5.columns[~df5.columns.str.contains('tmdw|tmin|tmax|tair|ppt')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output', 'month_ppt.csv'))
    (df5[df5.columns[~df5.columns.str.contains('tmdw|tmin|tmax|tair|vpd')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output',
                                               '3_months_temperatude.csv'))
    (df6[df6.columns[~df6.columns.str.contains('vpd|ppt')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output',
                                               '3_months_vpd.csv'))
    (df6[df6.columns[~df6.columns.str.contains('tmdw|tmin|tmax|tair|ppt')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output',
                                               '3_months_ppt.csv'))
    (df6[df6.columns[~df6.columns.str.contains('tmdw|tmin|tmax|tair|vpd')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output',
                                               '5_years_temperatude.csv'))
    (df7[df7.columns[~df7.columns.str.contains('vpd|ppt')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output',
                                               '5_years_vpd.csv'))
    (df7[df7.columns[~df7.columns.str.contains('tmdw|tmin|tmax|tair|ppt')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output',
                                               '5_years_ppt.csv'))
    (df7[df7.columns[~df7.columns.str.contains('tmdw|tmin|tmax|tair|vpd')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output',
                                               'heatwave_metrics.csv'))
    (df8[df8.columns[~df8.columns.str.contains('dry spell|ppt')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output', 'dry_atm_metrics.csv'))
    (df8[df8.columns[~df8.columns.str.contains('heatwave|ppt')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    filename = os.path.join(repo, os.path.join('output', 'precip_metrics.csv'))
    (df8[df8.columns[~df8.columns.str.contains('dry spell|heatwave')]]
        .to_csv(filename, index=False, na_rep='', encoding='utf-8'))

    return


def read_csv(fname):

    """
    Reads csv file with one header.

    Arguments:
    ----------
    fname: string
        input filename (with path)

    Returns:
    --------
    df: pandas dataframe
        dataframe containing all the csv data

    """

    df = (pd.read_csv(fname, header=[0]).dropna(axis=0, how='all')
          .dropna(axis=1, how='all').squeeze()) # open and drop missing values

    return df


def prepare_new_df(df, which='day'):

    """
    Appends empty column to the new dataset, depending on which period of time
    it is designed to analyse.

    Arguments:
    ----------
    df: pandas dataframe
        df containing the data

    which: string
        period to analyse before data collection. Can be week, month, 3 months,
        or 5 years. If which is metrics, then it is about climate metrics rather
        than temporality

    Returns:
    --------
    df: pandas dataframe
        df containing new empty columns

    """

    if which == 'day': # day immediately before collection

        # insert the relevant columns if they're not already there
        for collect in ['Old', 'Modern']:

            if ('%s prev day tmdw (degC)' % (collect) not in
                df.columns.tolist()): # day midway temp. (median tmin & tmax)
                df['%s prev day tmdw (degC)' % (collect)] = pd.np.nan

            if ('%s prev day tmin (degC)' % (collect) not in
                df.columns.tolist()):
                df['%s prev day tmin (degC)' % (collect)] = pd.np.nan

            if ('%s prev day tmax (degC)' % (collect) not in
                df.columns.tolist()):
                df['%s prev day tmax (degC)' % (collect)] = pd.np.nan

            if ('%s prev day vpd 3pm (kPa)' % (collect) not in
                df.columns.tolist()):
                df['%s prev day vpd 3pm (kPa)' % (collect)] = pd.np.nan

            if ('%s prev day ppt (mm day-1)' % (collect) not in
                df.columns.tolist()):
                df['%s prev day ppt (mm day-1)' % (collect)] = pd.np.nan

    elif which != 'metrics': # period before collection >= 1 week

        # insert the relevant columns if they're not already there
        for collect in ['Old', 'Modern']:

            # all the temperature columns
            if ('%s prev %s avg tmdw (degC)' % (collect, which) not in
                df.columns.tolist()): # average midway temperature over period
                df['%s prev %s avg tmdw (degC)' % (collect, which)] = pd.np.nan

            if ('%s prev %s avg tmin (degC)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s avg tmin (degC)' % (collect, which)] = pd.np.nan

            if ('%s prev %s avg tmax (degC)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s avg tmax (degC)' % (collect, which)] = pd.np.nan

            # N.B.: since there is daily min max temp, the amplitude in this
            # case is the daily difference between max and min, which can then
            # be an avg / the min / the max amplitude over the time period
            if ('%s prev %s avg amplitude tair (degC)' % (collect, which) not in
                df.columns.tolist()): # average (tmax - tmin)(daily)
                df['%s prev %s avg amplitude tair (degC)' \
                   % (collect, which)] = pd.np.nan

            if ('%s prev %s min tmdw (degC)' % (collect, which) not in
                df.columns.tolist()): # minimum midway temperature over period
                df['%s prev %s min tmdw (degC)' % (collect, which)] = pd.np.nan

            if ('%s prev %s min tmin (degC)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s min tmin (degC)' % (collect, which)] = pd.np.nan

            if ('%s prev %s min tmax (degC)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s min tmax (degC)' % (collect, which)] = pd.np.nan

            if ('%s prev %s min amplitude tair (degC)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s min amplitude tair (degC)' \
                   % (collect, which)] = pd.np.nan

            if ('%s prev %s max tmdw (degC)' % (collect, which) not in
                df.columns.tolist()): # maximum midway temperature over period
                df['%s prev %s max tmdw (degC)' % (collect, which)] = pd.np.nan

            if ('%s prev %s max tmin (degC)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s max tmin (degC)' % (collect, which)] = pd.np.nan

            if ('%s prev %s max tmax (degC)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s max tmax (degC)' % (collect, which)] = pd.np.nan

            if ('%s prev %s max amplitude tair (degC)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s max amplitude tair (degC)' \
                   % (collect, which)] = pd.np.nan

            if ('%s prev %s var tmdw (-)' % (collect, which) not in
                df.columns.tolist()): # variability of midway temp. over period
                df['%s prev %s var tmdw (-)' % (collect, which)] = pd.np.nan

            if ('%s prev %s var tmin (-)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s var tmin (-)' % (collect, which)] = pd.np.nan

            if ('%s prev %s var tmax (-)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s var tmax (-)' % (collect, which)] = pd.np.nan

            # monthly averaged temperatures?
            if (which != 'week') and (which != 'month'):
                if ('%s prev %s avg monthly tmdw (degC)' % (collect, which) 
                    not in df.columns.tolist()):
                    df['%s prev %s avg monthly tmdw (degC)' \
                       % (collect, which)] = pd.np.nan

                if ('%s prev %s avg monthly tmin (degC)' % (collect, which) not
                    in df.columns.tolist()):
                    df['%s prev %s avg monthly tmin (degC)' \
                       % (collect, which)] = pd.np.nan

                if ('%s prev %s avg monthly tmax (degC)' % (collect, which) not
                    in df.columns.tolist()):
                    df['%s prev %s avg monthly tmax (degC)' \
                       % (collect, which)] = pd.np.nan

                # N.B.: since there is monthly min max temp, the amplitude in
                # this case is the monthly difference between max and min which
                # can then be an avg / the min / the max amplitude over the time
                # period
                if ('%s prev %s avg amplitude monthly tair (degC)'
                    % (collect, which) not in df.columns.tolist()):
                    df['%s prev %s avg amplitude monthly tair (degC)' \
                       % (collect, which)] = pd.np.nan

                if ('%s prev %s min amplitude monthly tair (degC)'
                    % (collect, which) not in df.columns.tolist()):
                    df['%s prev %s min amplitude monthly tair (degC)' \
                       % (collect, which)] = pd.np.nan

                if ('%s prev %s max amplitude monthly tair (degC)'
                    % (collect, which) not in df.columns.tolist()):
                    df['%s prev %s max amplitude monthly tair (degC)' \
                       % (collect, which)] = pd.np.nan

                # variability
                if ('%s prev %s var monthly tmdw (-)' % (collect, which) not in
                    df.columns.tolist()): # var of midway temp.
                    df['%s prev %s var monthly tmdw (-)' % (collect, which)] = \
                                                                       pd.np.nan

                if ('%s prev %s var monthly tmin (-)' % (collect, which) not in
                    df.columns.tolist()):
                    df['%s prev %s var monthly tmin (-)' % (collect, which)] = \
                                                                       pd.np.nan

                if ('%s prev %s var monthly tmax (-)' % (collect, which) not in
                    df.columns.tolist()):
                    df['%s prev %s var monthly tmax (-)' % (collect, which)] = \
                                                                       pd.np.nan

            # all the vpd columns
            if ('%s prev %s avg vpd 3pm (kPa)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s avg vpd 3pm (kPa)' % (collect, which)] = \
                                                                       pd.np.nan

            if ('%s prev %s min vpd 3pm (kPa)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s min vpd 3pm (kPa)' % (collect, which)] = \
                                                                       pd.np.nan

            if ('%s prev %s max vpd 3pm (kPa)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s max vpd 3pm (kPa)' % (collect, which)] = \
                                                                       pd.np.nan

            # N.B.: there is no daily min max vpd, so we compute the absolute
            # range between max of time period and min of time period rather
            # than the timely amplitude of the variable strictly speaking
            if ('%s prev %s range vpd 3pm (kPa)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s range vpd 3pm (kPa)' % (collect, which)] = \
                                                                       pd.np.nan

            if ('%s prev %s var vpd 3pm (-)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s var vpd 3pm (-)' % (collect, which)] = pd.np.nan

            # monthly averaged vpd?
            if (which != 'week') and (which != 'month'):
                if ('%s prev %s avg monthly vpd 3pm (kPa)' % (collect, which)
                    not in df.columns.tolist()):
                    df['%s prev %s avg monthly vpd 3pm (kPa)' \
                       % (collect, which)] = pd.np.nan 

                if ('%s prev %s min monthly vpd 3pm (kPa)' % (collect, which)
                    not in df.columns.tolist()):
                    df['%s prev %s min monthly vpd 3pm (kPa)' \
                       % (collect, which)] = pd.np.nan 

                if ('%s prev %s max monthly vpd 3pm (kPa)' % (collect, which)
                    not in df.columns.tolist()):
                    df['%s prev %s max monthly vpd 3pm (kPa)' \
                       % (collect, which)] = pd.np.nan 

                # N.B.: the range in this case is the difference between max and
                # min across months
                if ('%s prev %s range monthly vpd 3pm (kPa)' % (collect, which)
                    not in df.columns.tolist()):
                    df['%s prev %s range monthly vpd 3pm (kPa)' \
                       % (collect, which)] = pd.np.nan                        

                if ('%s prev %s var monthly vpd 3pm (-)' % (collect, which) not 
                    in df.columns.tolist()):
                    df['%s prev %s var monthly vpd 3pm (-)' \
                       % (collect, which)] = pd.np.nan

            # all the precipitation columns
            if ('%s prev %s total ppt (mm)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s total ppt (mm)' % (collect, which)] = pd.np.nan

            if ('%s prev %s avg ppt (mm day-1)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s avg ppt (mm day-1)' % (collect, which)] = \
                                                                       pd.np.nan

            if ('%s prev %s min ppt (mm day-1)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s min ppt (mm day-1)' % (collect, which)] = \
                                                                       pd.np.nan

            if ('%s prev %s max ppt (mm day-1)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s max ppt (mm day-1)' % (collect, which)] = \
                                                                       pd.np.nan

            # N.B.: there is no daily min max ppt, so we compute the absolute
            # range between max of time period and min of time period rather
            # than the timely amplitude of the variable strictly speaking
            if ('%s prev %s range ppt (mm)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s range ppt (mm)' % (collect, which)] = \
                                                                       pd.np.nan

            if ('%s prev %s var ppt (-)' % (collect, which) not in
                df.columns.tolist()):
                df['%s prev %s var ppt (-)' % (collect, which)] = pd.np.nan

            # monthly rainfall?
            if (which != 'week') and (which != 'month'):
                if ('%s prev %s avg monthly ppt (mm month-1)' % (collect, which)
                    not in df.columns.tolist()):
                    df['%s prev %s avg monthly ppt (mm month-1)' \
                       % (collect, which)] = pd.np.nan 

                if ('%s prev %s min monthly ppt (mm month-1)' % (collect, which)
                    not in df.columns.tolist()):
                    df['%s prev %s min monthly ppt (mm month-1)' \
                       % (collect, which)] = pd.np.nan 

                if ('%s prev %s max monthly ppt (mm month-1)' % (collect, which)
                    not in df.columns.tolist()):
                    df['%s prev %s max monthly ppt (mm month-1)' \
                       % (collect, which)] = pd.np.nan 

                # N.B.: the range in this case is the difference between max and
                # min across months
                if ('%s prev %s range monthly ppt (mm)' % (collect, which) not
                    in df.columns.tolist()):
                    df['%s prev %s range monthly ppt (mm)' \
                       % (collect, which)] = pd.np.nan      

                if ('%s prev %s var monthly ppt (-)' % (collect, which) not in
                    df.columns.tolist()):
                    df['%s prev %s var monthly ppt (-)' % (collect, which)] = \
                                                                       pd.np.nan

            # for month & + timescale, add the longest period of no rain
            if (which != 'week') and (which != '5 years'):
                if ('%s prev %s max Ndays no ppt (-)' % (collect, which) not in
                    df.columns.tolist()):
                    df['%s prev %s max Ndays no ppt (-)' \
                       % (collect, which)] = pd.np.nan

            # for several years, add the (year and) season's info
            if which == '5 years':

                for yr in ['y1', 'y2', 'y3', 'y4', 'y5']:

                    if ('%s prev %s total ppt in %s (mm)' % (collect, which, yr)
                        not in df.columns.tolist()):
                        df['%s prev %s total ppt in %s (mm)' \
                           % (collect, which, yr)] = pd.np.nan

                    if ('%s prev %s total ppt in the season %s (mm)' % 
                        (collect, which, yr) not in df.columns.tolist()):
                        df['%s prev %s total ppt in the season %s (mm)' \
                           % (collect, which, yr)] = pd.np.nan

                    # add the season's (> relevant than year's) min & max precip
                    if ('%s prev %s min ppt in the season %s (mm day-1)' % 
                        (collect, which, yr) not in df.columns.tolist()):
                        df['%s prev %s min ppt in the season %s (mm day-1)' \
                           % (collect, which, yr)] = pd.np.nan

                    if ('%s prev %s max ppt in the season %s (mm day-1)' % 
                        (collect, which, yr) not in df.columns.tolist()):
                        df['%s prev %s max ppt in the season %s (mm day-1)' \
                           % (collect, which, yr)] = pd.np.nan

    else: # heat / dryness indices / metrics

        # insert the relevant columns if they're not already there
        for collect in ['Old', 'Modern']:

            for yr in ['y1', 'y2', 'y3', 'y4', 'y5']: # yearly heatwave indexes
    
                if ('%s total N heatwaves %s (-)' % (collect, yr) not in
                    df.columns.tolist()):
                    df['%s total N heatwaves %s (-)' % (collect, yr)] = \
                                                                       pd.np.nan

                if ('%s avg Ndays heatwave %s (-)' % (collect, yr) not in
                    df.columns.tolist()):
                    df['%s avg Ndays heatwave %s (-)' % (collect, yr)] = \
                                                                       pd.np.nan

                if ('%s max Ndays heatwave %s (-)' % (collect, yr) not in
                    df.columns.tolist()):
                    df['%s max Ndays heatwave %s (-)' % (collect, yr)] = \
                                                                       pd.np.nan

                if ('%s dates max Ndays heatwave %s (-)' % (collect, yr) not in
                    df.columns.tolist()):
                    df['%s dates max Ndays heatwave %s (-)' % (collect, yr)] = \
                                                                       pd.np.nan

                if yr == 'y1':
                    if ('%s dates most recent heatwave (-)' % (collect) not in
                        df.columns.tolist()):
                        df['%s dates most recent heatwave (-)' % (collect)] = \
                                                                       pd.np.nan

            # heatwave indexes, but seasonnal interannual (across 5 years) 
            if ('%s interannual total N heatwaves in the season (-)' % (collect)
                not in df.columns.tolist()):
                df['%s interannual total N heatwaves in the season (-)' \
                   % (collect)] = pd.np.nan

            if ('%s interannual avg Ndays heatwave in the season (-)'
                % (collect) not in df.columns.tolist()):
                df['%s interannual avg Ndays heatwave in the season (-)' \
                   % (collect)] = pd.np.nan

            if ('%s interannual max Ndays heatwave in the season (-)' 
                % (collect) not in df.columns.tolist()):
                df['%s interannual max Ndays heatwave in the season (-)' \
                   % (collect)] = pd.np.nan

            for yr in ['y1', 'y2', 'y3', 'y4', 'y5']: # atm. dryness (VPD)

                if ('%s total N dry spells %s (-)' % (collect, yr) not in
                    df.columns.tolist()):
                    df['%s total N dry spells %s (-)' % (collect, yr)] = \
                                                                       pd.np.nan

                if ('%s avg Ndays dry spell %s (-)' % (collect, yr) not in
                    df.columns.tolist()):
                    df['%s avg Ndays dry spell %s (-)' % (collect, yr)] = \
                                                                       pd.np.nan

                if ('%s max Ndays dry spell %s (-)' % (collect, yr) not in
                    df.columns.tolist()):
                    df['%s max Ndays dry spell %s (-)' % (collect, yr)] = \
                                                                       pd.np.nan

                if ('%s dates max Ndays dry spell %s (-)' % (collect, yr) not in
                    df.columns.tolist()):
                    df['%s dates max Ndays dry spell %s (-)' \
                       % (collect, yr)] = pd.np.nan

                if yr == 'y1':
                    if ('%s dates most recent dry spell (-)' % (collect) not in
                        df.columns.tolist()):
                        df['%s dates most recent dry spell (-)' % (collect)] = \
                                                                       pd.np.nan

            # atmospheric dryness, seasonnal interannual (across 5 years) 
            if ('%s interannual total N dry spells in the season (-)'
                % (collect) not in df.columns.tolist()):
                df['%s interannual total N dry spells in the season (-)' \
                   % (collect)] = pd.np.nan

            if ('%s interannual avg Ndays dry spell in the season (-)'
                % (collect) not in df.columns.tolist()):
                df['%s interannual avg Ndays dry spell in the season (-)' \
                   % (collect)] = pd.np.nan

            if ('%s interannual max Ndays dry spell in the season (-)'
                % (collect) not in df.columns.tolist()):
                df['%s interannual max Ndays dry spell in the season (-)' \
                   % (collect)] = pd.np.nan

            for yr in ['y1', 'y2', 'y3', 'y4', 'y5']: # consecutive no rain days
 
                if ('%s max Ndays no ppt %s (-)' % (collect, yr) not in
                    df.columns.tolist()):
                    df['%s max Ndays no ppt %s (-)' % (collect, yr)] = pd.np.nan

                if ('%s dates max Ndays no ppt %s (-)' % (collect, yr) not in
                    df.columns.tolist()):
                    df['%s dates max Ndays no ppt %s (-)' % (collect, yr)] = \
                                                                       pd.np.nan

                if yr == 'y1':
                    if ('%s dates most recent no ppt (-)' % (collect) not in
                        df.columns.tolist()):
                        df['%s dates most recent no ppt (-)' % (collect)] = \
                                                                       pd.np.nan

            # consecutive no rain days, seasonnal interannual (across 5 years) 
            if ('%s interannual max Ndays no ppt in the season (-)' % (collect)
                not in df.columns.tolist()):
                df['%s interannual max Ndays no ppt in the season (-)' \
                   % (collect)] = pd.np.nan

    return df   


def read_netcdf(fname, var_list=None):

    """
    Retrieves netcdf data & stores it into a dataframe

    Arguments:
    ----------
    fname: string
        input filename (with path)

    var_list: array
        variables to slice from the netcdf 

    Returns:
    --------
    df: pandas dataframe
        df containing the data

    """

    ds = xr.open_dataset(fname, autoclose=True) # access the data
    dates = pd.to_datetime(ds.time.values) # retrieve dates

    if var_list is None: # drop grid
        df = ds.squeeze(dim=('lon', 'lat'), drop=True).to_dataframe()

    else:
        df = ds[var_list].squeeze(dim=('lon', 'lat'), drop=True).to_dataframe()

    # reindex
    try:
        df['dates'] = dates
        df = df.set_index('dates')

    except ValueError: # this can happen when there are multiple z dimension > 1 
        pass

    # free memory
    ds.close()
    del ds
    ds = None
    gc.collect()

    return df


def get_season(p, df, dt):

    """
    Finds the season dates for the collection date, i.e. the climatical season
    the collection date is in.

    Arguments:
    ----------
    p: pandas series
        site info

    df: pandas dataframe
        df containing the data

    dt: pandas date
        collection date

    Returns:
    --------
    Data restricted to the season of collection.

    """

    if (p.Lat > -23.45) and (p.Lat < 23.45): # sub-tropics
        if (dt.month > 4) and (dt.month < 11): # dry season

            return df[np.logical_and(df.index.month > 4, df.index.month < 11)]

        else: # wet season

            return df[np.logical_or(df.index.month > 10, df.index.month < 5)]


    else: # mid-latitudes
        if (dt.month > 2 ) and (dt.month < 6): # autumn

            return df[np.logical_and(df.index.month > 2, df.index.month < 6)]

        elif (dt.month > 5 ) and (dt.month < 9): # winter

            return df[np.logical_and(df.index.month > 5, df.index.month < 9)]

        elif (dt.month > 8 ) and (dt.month < 12): # spring

            return df[np.logical_and(df.index.month > 8, df.index.month < 12)]

        else: # summer

            return df[np.logical_or(df.index.month > 11, df.index.month < 3)]


def consecutive(indexes):

    """
    Finds subsets of consecutive indexes from an index input array.

    Arguments:
    ----------
    indexes: array
        array or list of indexes

    Returns:
    --------
    Nested array containing all consecutive arrays, with each subarray
    containing a suite of indexes.

    """

    # first calculate the difference between next and current 
    count = [indexes[i] - indexes[i-1] for i in range(1, len(indexes))]
    count = np.asarray([1] + count) # do not forget first element of indexes!

    # where count is 1, the previous and current are consecutive
    start = [0] + list(np.where(count > 1.)[0]) # start of consecutive
    finish = [e for e in list(np.where(count > 1.)[0])] + [len(count)] # end of

    # return all the indexes in sublists of consecutive, i.e. not single indexes
    indexes = np.asarray(indexes) # makes sure the index list can be sliced
    idxs = [indexes[start[i]:finish[i]] for i in range(len(start))]
    idxs = [e for e in idxs if len(e) > 1] # do not return single index

    return idxs


def running_mean(var, N):

    """
    Calculates a N-day running mean on the data.

    Arguments:
    ----------
    var: array
        variable on which the running mean is applied

    N: int
        length of the running mean [d]

    Returns:
    --------
    var: array
        initial variable smoothed via a running mean

    """

    var = var.rolling(N, min_periods=1).mean()

    return var


def day_forcings(p, df):

    """
    Finds out the forcing information on the day right before collection.

    Arguments:
    ----------
    p: pandas series
        site info

    df: pandas dataframe
        df containing the data

    Returns:
    --------
    p: pandas series
        site info containing data on the day right before collection

    """

    # start one day before, i.e. with leading conditions
    old_date = p['Old seed collection date'] - pd.to_timedelta(1, unit='D')
    modern_date = p['Modern seed collection date'] - pd.to_timedelta(1,
                                                                       unit='D')

    for collect in ['Old', 'Modern']:

        if collect == 'Old':
            dt = old_date

        else:
            dt = modern_date

        # midway temperature: between tmin and tmax
        p['%s prev day tmdw (degC)' % (collect)] = df['tmdw'].loc[dt]
        p['%s prev day tmin (degC)' % (collect)] = df['tmin'].loc[dt]
        p['%s prev day tmax (degC)' % (collect)] = df['tmax'].loc[dt]
        p['%s prev day vpd 3pm (kPa)' % (collect)] = df['vprp3pm'].loc[dt]
        p['%s prev day ppt (mm day-1)' % (collect)] = df['pre'].loc[dt]

    return p


def other_period(p, df, which='week'):

    """
    Analyses the forcing information depending on the length of a lead up period
    before collection.

    Arguments:
    ----------
    p: pandas series
        site info

    df: pandas dataframe
        df containing the data

    which: string
        period to analyse before data collection. Can be week, month, 3 months,
        or 5 years

    Returns:
    --------
    p: pandas series
        site info containing data about "which" period before collection

    """

    # start one day before, i.e. with leading conditions
    old_dt2 = p['Old seed collection date'] - pd.to_timedelta(1, unit='D')
    modern_dt2 = p['Modern seed collection date'] - pd.to_timedelta(1, unit='D')

    if which == 'week': # extra 6 days lead up
        old_dt1 = old_dt2 - pd.to_timedelta(6, unit='D')
        modern_dt1 = modern_dt2 - pd.to_timedelta(6, unit='D')

    if which == 'month': # extra month lead up
        old_dt1 = old_dt2 - pd.to_timedelta(1, unit='M')
        modern_dt1 = modern_dt2 - pd.to_timedelta(1, unit='M')

    if which == '3 months': # extra 3 months lead up
        old_dt1 = old_dt2 - pd.to_timedelta(3, unit='M')
        modern_dt1 = modern_dt2 - pd.to_timedelta(3, unit='M')

    if which == '5 years': # extra 5 years lead up
        old_dt1 = old_dt2 - pd.to_timedelta(5, unit='Y')
        modern_dt1 = modern_dt2 - pd.to_timedelta(5, unit='Y')
        

    for collect in ['Old', 'Modern']:

        if collect == 'Old':
            dt1 = old_dt1 # start date (i.e up to 5 years prior to collection)
            dt2 = old_dt2 # end date

        else:
            dt1 = modern_dt1 # start date
            dt2 = modern_dt2 # end date

        # periods's data, df restricted to the relevant dates
        dfr = df.loc[dt1:dt2]

        # temperature averages
        p['%s prev %s avg tmdw (degC)' % (collect, which)] = \
                                                            np.mean(dfr['tmdw'])
        p['%s prev %s avg tmin (degC)' % (collect, which)] = \
                                                            np.mean(dfr['tmin'])
        p['%s prev %s avg tmax (degC)' % (collect, which)] = \
                                                            np.mean(dfr['tmax'])

        # N.B.: since there is daily min max temp, the amplitude in this case is
        # the daily difference between max and min, which can then be an avg /
        # the min / the max amplitude over the time period
        p['%s prev %s avg amplitude tair (degC)' % (collect, which)] = \
                                              np.mean(dfr['tmax'] - dfr['tmin'])

        # temperature minimums
        p['%s prev %s min tmdw (degC)' % (collect, which)] = \
                                                            np.amin(dfr['tmdw'])
        p['%s prev %s min tmin (degC)' % (collect, which)] = \
                                                            np.amin(dfr['tmin'])
        p['%s prev %s min tmax (degC)' % (collect, which)] = \
                                                            np.amax(dfr['tmax'])
        p['%s prev %s min amplitude tair (degC)' % (collect, which)] = \
                                              np.amin(dfr['tmax'] - dfr['tmin'])

        # temperature maximums
        p['%s prev %s max tmdw (degC)' % (collect, which)] = \
                                                            np.amax(dfr['tmdw'])
        p['%s prev %s max tmin (degC)' % (collect, which)] = \
                                                            np.amax(dfr['tmin'])
        p['%s prev %s max tmax (degC)' % (collect, which)] = \
                                                            np.amax(dfr['tmax'])
        p['%s prev %s max amplitude tair (degC)' % (collect, which)] = \
                                              np.amax(dfr['tmax'] - dfr['tmin'])

        # variability of temperature
        try:
            p['%s prev %s var tmdw (-)' % (collect, which)] = \
                                      np.std(dfr['tmdw']) / np.mean(dfr['tmdw'])

        except ZeroDivisionError: # mean is null
            pass

        try:
            p['%s prev %s var tmin (-)' % (collect, which)] = \
                                      np.std(dfr['tmin']) / np.mean(dfr['tmin'])

        except ZeroDivisionError: # mean is null
            pass

        try:
            p['%s prev %s var tmax (-)' % (collect, which)] = \
                                      np.std(dfr['tmax']) / np.mean(dfr['tmax'])

        except ZeroDivisionError: # mean is null
            pass


        # monthly averaged temperatures?
        if (which != 'week') and (which != 'month'):
            mtmdw = (dfr['tmdw'].groupby(by=[dfr.index.month, dfr.index.year])
                                  .mean())
            mtmin = (dfr['tmin'].groupby(by=[dfr.index.month, dfr.index.year])
                                .mean())
            mtmax = (dfr['tmax'].groupby(by=[dfr.index.month, dfr.index.year])
                                .mean())

            p['%s prev %s avg monthly tmdw (degC)' % (collect, which)] = \
                                                                  np.mean(mtmdw)
            p['%s prev %s avg monthly tmin (degC)' % (collect, which)] = \
                                                                  np.mean(mtmin)
            p['%s prev %s avg monthly tmax (degC)' % (collect, which)] = \
                                                                  np.mean(mtmax)

            p['%s prev %s avg amplitude monthly tair (degC)' \
              % (collect, which)] = np.mean(mtmax - mtmin)
            p['%s prev %s min amplitude monthly tair (degC)' \
              % (collect, which)] = np.amin(mtmax - mtmin)
            p['%s prev %s max amplitude monthly tair (degC)' \
              % (collect, which)] = np.amax(mtmax - mtmin)

            try: # variability
                p['%s prev %s var monthly tmdw (-)' % (collect, which)] = \
                                                 np.std(mtmdw) /  np.mean(mtmdw)

            except ZeroDivisionError: # mean is null
                pass

            try:
                p['%s prev %s var monthly tmin (-)' % (collect, which)] = \
                                                  np.std(mtmin) / np.mean(mtmin)

            except ZeroDivisionError: # mean is null
                pass

            try:
                p['%s prev %s var monthly tmax (-)' % (collect, which)] = \
                                                  np.std(mtmax) / np.mean(mtmax)

            except ZeroDivisionError: # mean is null
                pass

        # all the vpd columns
        p['%s prev %s avg vpd 3pm (kPa)' % (collect, which)] = \
                                                         np.mean(dfr['vprp3pm'])
        p['%s prev %s min vpd 3pm (kPa)' % (collect, which)] = \
                                                         np.amin(dfr['vprp3pm'])
        p['%s prev %s max vpd 3pm (kPa)' % (collect, which)] = \
                                                         np.amax(dfr['vprp3pm'])

        # N.B.: there is no daily min max vpd, so the range in this case is the
        # difference between max of time period and min (non zero) of time
        # period
        p['%s prev %s range vpd 3pm (kPa)' % (collect, which)] = \
                               np.amax(dfr['vprp3pm']) - np.amin(dfr['vprp3pm'])

        try:
            p['%s prev %s var vpd 3pm (-)' % (collect, which)] = \
                                np.std(dfr['vprp3pm']) / np.mean(dfr['vprp3pm'])

        except ZeroDivisionError: # mean is null
            pass

        # monthly averaged vpd?
        if (which != 'week') and (which != 'month'):
            monthly = (dfr['vprp3pm']
                          .groupby(by=[dfr.index.month, dfr.index.year])
                          .mean())

            p['%s prev %s avg monthly vpd 3pm (kPa)' % (collect, which)] = \
                                                                np.mean(monthly)
            p['%s prev %s min monthly vpd 3pm (kPa)' % (collect, which)] = \
                                                                np.amin(monthly)
            p['%s prev %s max monthly vpd 3pm (kPa)' % (collect, which)] = \
                                                                np.amax(monthly)

            p['%s prev %s range monthly vpd 3pm (kPa)' % (collect, which)] = \
                                             np.amax(monthly) - np.amin(monthly)

            try: # variability
                p['%s prev %s var monthly vpd 3pm (-)' % (collect, which)] = \
                                             np.std(monthly) /  np.mean(monthly)

            except ZeroDivisionError: # mean is null
                pass

        # all the precipitation columns
        BoM_thr = 0.200000002980232 # mm
        p['%s prev %s total ppt (mm)' % (collect, which)] = ma.sum(dfr['pre'])
        p['%s prev %s avg ppt (mm day-1)' % (collect, which)] = \
                                                             ma.mean(dfr['pre'])
        p['%s prev %s min ppt (mm day-1)' % (collect, which)] = \
                                 ma.amin(dfr['pre'].where(dfr['pre'] > BoM_thr))
        p['%s prev %s max ppt (mm day-1)' % (collect, which)] = \
                                                             ma.amax(dfr['pre'])

        # N.B.: there is no daily min max ppt, so the range in this case is the
        # difference between max of time period and min (non zero) of time
        # period
        p['%s prev %s range ppt (mm)' % (collect, which)] = \
           ma.amax(dfr['pre']) - ma.amin(dfr['pre'].where(dfr['pre'] > BoM_thr))

        try:
            p['%s prev %s var ppt (-)' % (collect, which)] = \
                                        ma.std(dfr['pre']) / ma.mean(dfr['pre'])

        except ZeroDivisionError: # mean is null
            pass

        # monthly averaged precip?
        if (which != 'week') and (which != 'month'):
            monthly = (dfr['pre']
                          .groupby(by=[dfr.index.month, dfr.index.year])
                          .mean())

            p['%s prev %s avg monthly ppt (mm month-1)' % (collect, which)] = \
                                                                ma.mean(monthly)
            p['%s prev %s min monthly ppt (mm month-1)' % (collect, which)] = \
                                       ma.amin(monthly.where(monthly > BoM_thr))
            p['%s prev %s max monthly ppt (mm month-1)' % (collect, which)] = \
                                                                ma.amax(monthly)

            p['%s prev %s range monthly ppt (mm)' % (collect, which)] = \
                    ma.amax(monthly) - ma.amin(monthly.where(monthly > BoM_thr))

            try: # variability
                p['%s prev %s var monthly ppt (-)' % (collect, which)] = \
                                             ma.std(monthly) /  ma.mean(monthly)

            except ZeroDivisionError: # mean is null
                pass

        # max Ndays without rain for periods up to 3 months?
        if (which != 'week') and (which != '5 years'):
            ind = [i for i in range(len(dfr['pre'])) if \
                   dfr['pre'].iloc[i] <= BoM_thr]
            ind = np.asarray([e for e in consecutive(ind)])

            try:
                p['%s prev %s max Ndays no ppt (-)' % (collect, which)] = \
                                                  np.amax([len(e) for e in ind])

            except ValueError: # no rainless consecutive days
                pass

        # total min max precip in the year and season of collection?
        if which == '5 years': 
            season = get_season(p, dfr, dt2 + pd.to_timedelta(1, unit='D'))

            # restrict by N years lead up, i.e. season3 is a 3 years lead up 
            idx1 = dfr.index[-1] - pd.to_timedelta(1, unit='Y')
            year1 = dfr['pre'].loc[idx1:]
            season1 = season['pre'].loc[idx1:]

            idx1 = dfr.index[-1] - pd.to_timedelta(2, unit='Y')
            idx2 = dfr.index[-1] - pd.to_timedelta(1, unit='Y')
            year2 = dfr['pre'].loc[idx1:idx2]
            season2 = season['pre'].loc[idx1:idx2]

            idx1 = dfr.index[-1] - pd.to_timedelta(3, unit='Y')
            idx2 = dfr.index[-1] - pd.to_timedelta(2, unit='Y')
            year3 = dfr['pre'].loc[idx1:idx2]
            season3 = season['pre'].loc[idx1:idx2]

            idx1 = dfr.index[-1] - pd.to_timedelta(4, unit='Y')
            idx2 = dfr.index[-1] - pd.to_timedelta(3, unit='Y')
            year4 = dfr['pre'].loc[idx1:idx2]
            season4 = season['pre'].loc[idx1:idx2]

            idx2 = dfr.index[-1] - pd.to_timedelta(4, unit='Y')
            year5 = dfr['pre'].loc[:idx2]
            season5 = season['pre'].loc[:idx2]

            years = [year1, year2, year3, year4, year5]
            seasons = [season1, season2, season3, season4, season5]
            yrs = ['y1', 'y2', 'y3', 'y4', 'y5']

            # using threshold of 0.2 mm day-1 for "no precip" (BoM definition)
            for i in range(len(yrs)):

                p['%s prev %s total ppt in %s (mm)' \
                  % (collect, which, yrs[i])] = ma.sum(years[i])

                p['%s prev %s total ppt in the season %s (mm)' \
                  % (collect, which, yrs[i])] = ma.sum(seasons[i])

                p['%s prev %s min ppt in the season %s (mm day-1)' \
                  % (collect, which, yrs[i])] = \
                                 ma.amin(seasons[i].where(seasons[i] > BoM_thr))

                p['%s prev %s max ppt in the season %s (mm day-1)' \
                  % (collect, which, yrs[i])] = ma.amax(seasons[i])

    return p


def metrics(p, df):

    """
    Calculates heatwave indexes, atmospheric dryness information, and info on
    rainfall deficit going back 5 years before collection, both on a yearly
    basis and within the season of collection only.

    Arguments:
    ----------
    p: pandas series
        site info

    df: pandas dataframe
        df containing the data

    Returns:
    --------
    p: pandas series
        site info containing the computed metrics

    """

    # start one day before, i.e. with leading conditions
    old_dt2 = p['Old seed collection date'] - pd.to_timedelta(1, unit='D')
    modern_dt2 = p['Modern seed collection date'] - pd.to_timedelta(1, unit='D')

    # extra 5 years lead up necessary for metric calculation
    old_dt1 = old_dt2 - pd.to_timedelta(5, unit='Y')
    modern_dt1 = modern_dt2 - pd.to_timedelta(5, unit='Y')     

    for collect in ['Old', 'Modern']:

        if collect == 'Old':
            dt1 = old_dt1
            dt2 = old_dt2

        else:
            dt1 = modern_dt1
            dt2 = modern_dt2

        # for both the heatwave and vpd index, get 3 and 30 day running means
        mdwts3 = running_mean(df['tmdw'], 3)
        mdwts30 = running_mean(df['tmdw'], 30)
        vpd3 = running_mean(df['vprp3pm'], 3)
        vpd30 = running_mean(df['vprp3pm'], 30)

        # shift the indexes of the 30 day running means by -3 days to match the
        # definition of the EHF. This is a trick to avoid having to deal with
        # missaligned indexes!
        mdwts30.index = mdwts30.index - pd.to_timedelta(3, unit='D')
        mdwts30.name = 'tmdw30'
        vpd30.index = vpd30.index - pd.to_timedelta(3, unit='D')
        vpd30.name = 'vpd30'
        dftmp30 = pd.concat([mdwts30, vpd30], axis=1) # df for the 30 day window

        # temporarily add those variables to the dataframe
        dftmp = df.copy()
        dftmp['tmdw3'] = mdwts3
        dftmp['vpd3'] = vpd3
        dftmp = dftmp.merge(dftmp30, how='inner', left_index=True,
                            right_index=True)

        # periods's data, restricted to the relevant dates
        dfr = dftmp.loc[dt1:dt2]
        season = get_season(p, dfr, dt2 + pd.to_timedelta(1, unit='D'))

        # restrict by N years lead up, i.e. season3 is a 3 years lead up
        idx1 = dfr.index[-1] - pd.to_timedelta(1, unit='Y') # 1 year
        year1 = dfr.loc[idx1:]
        season1 = season.loc[idx1:]

        idx1 = dfr.index[-1] - pd.to_timedelta(2, unit='Y') # 2 years
        idx2 = dfr.index[-1] - pd.to_timedelta(1, unit='Y')
        year2 = dfr.loc[idx1:idx2]
        season2 = season.loc[idx1:idx2]

        idx1 = dfr.index[-1] - pd.to_timedelta(3, unit='Y') # 3 years
        idx2 = dfr.index[-1] - pd.to_timedelta(2, unit='Y')
        year3 = dfr.loc[idx1:idx2]
        season3 = season.loc[idx1:idx2]

        idx1 = dfr.index[-1] - pd.to_timedelta(4, unit='Y') # 4 years
        idx2 = dfr.index[-1] - pd.to_timedelta(3, unit='Y')
        year4 = dfr.loc[idx1:idx2]
        season4 = season.loc[idx1:idx2]

        idx2 = dfr.index[-1] - pd.to_timedelta(4, unit='Y') # 5 years
        year5 = dfr.loc[:idx2]
        season5 = season.loc[:idx2]

        # each lead up
        yrs = ['y1', 'y2', 'y3', 'y4', 'y5']

        # each lead up's midway temperatures, to calculate excess heat factor
        mdwts = [year1[['tmdw3', 'tmdw30']], year2[['tmdw3', 'tmdw30']], \
                 year3[['tmdw3', 'tmdw30']], year4[['tmdw3', 'tmdw30']], \
                 year5[['tmdw3', 'tmdw30']]]

        # heatwave duration index: restrict by relative excess heat factor, i.e.
        # dates where average mdwt over 3 days > average mdwt over the 30 days
        # preceding the 3 days
        for i in range(len(yrs)): # loop over the 5 years

            ind = [j for j in range(len(mdwts[i])) if \
                   mdwts[i]['tmdw3'].iloc[j] > mdwts[i]['tmdw30'].iloc[j]]
            ind = np.asarray([e for e in consecutive(ind) if len(e) >= 3])

            try:
                # how many heatwaves that year?
                p['%s total N heatwaves %s (-)' % (collect, yrs[i])] = len(ind)

                # avg duration of heatwave?
                p['%s avg Ndays heatwave %s (-)' % (collect, yrs[i])] = \
                                                  np.mean([len(e) for e in ind])

                # longest heatwave duration?
                p['%s max Ndays heatwave %s (-)' % (collect, yrs[i])] = \
                                                  np.amax([len(e) for e in ind])

                # dates of longest heatwave?
                max_dates = np.argmax([len(e) for e in ind])

                if i == 0:
                    dts = year1.iloc[(ind[max_dates])]

                if i == 1:
                    dts = year2.iloc[(ind[max_dates])]
                if i == 2:
                    dts = year3.iloc[(ind[max_dates])]

                if i == 3:
                    dts = year4.iloc[(ind[max_dates])]

                if i == 4:
                    dts = year5.iloc[(ind[max_dates])]

                p['%s dates max Ndays heatwave %s (-)' % (collect, yrs[i])] = \
                          '%s--%s' % (dts.index[0].date().strftime('%d/%m/%Y'),
                                      dts.index[-1].date().strftime('%d/%m/%Y'))

                if i == 0: # dates of most recent heatwave?

                    if len(ind) > 1:
                        dts = year1.iloc[(ind[-1])]
                        recent = '%s--%s' \
                                 % (dts.index[0].date().strftime('%d/%m/%Y'),
                                    dts.index[-1].date().strftime('%d/%m/%Y'))
                        p['%s dates most recent heatwave (-)' % (collect)] = \
                                                                          recent

            except ValueError: # no heatwaves
                pass

        # the interannual season's heatwave indexes
        mdwts = [season1[['tmdw3', 'tmdw30']], season2[['tmdw3', 'tmdw30']], \
                 season3[['tmdw3', 'tmdw30']], season4[['tmdw3', 'tmdw30']], \
                 season5[['tmdw3', 'tmdw30']]]

        ind = []

        for i in range(len(yrs)): # loop over the 5 years

            # restrict by excess heat factor
            ii = [j for j in range(len(mdwts[i])) if \
                  mdwts[i]['tmdw3'].iloc[j] > mdwts[i]['tmdw30'].iloc[j]]
            ind += [e for e in consecutive(ii) if len(e) >= 3]

        try:
            # how many heatwaves over all years in that specific season?
            p['%s interannual total N heatwaves in the season (-)' \
              % (collect)] = len(ind)
            # avg duration of heatwave over all years in that specific season?
            p['%s interannual avg Ndays heatwave in the season (-)' \
              % (collect)] = np.mean([len(e) for e in ind])

            # longest heatwave duration over all years in that specific season?
            p['%s interannual max Ndays heatwave in the season (-)' \
              % (collect)] = np.amax([len(e) for e in ind])

        except ValueError: # no heatwaves
            pass

        # each lead up's vpd, for calculation of atmospheric dryspells
        vpds = [year1[['vpd3', 'vpd30']], year2[['vpd3', 'vpd30']], \
                year3[['vpd3', 'vpd30']], year4[['vpd3', 'vpd30']], \
                year5[['vpd3', 'vpd30']]]

        # atmospheric dry spell duration: restrict by relative excess dryness,
        # i.e. dates where average vpd over 3 days > average vpd over the 30 
        # days preceding the 3 days
        for i in range(len(yrs)): # loop over the 5 years

            ind = [j for j in range(len(vpds[i])) if \
                   vpds[i]['vpd3'].iloc[j] > vpds[i]['vpd30'].iloc[j]]
            ind = np.asarray([e for e in consecutive(ind) if len(e) >= 3])

            try:
                # how many dry spells that year?
                p['%s total N dry spells %s (-)' % (collect, yrs[i])] = len(ind)

                # avg duration of dry spell?
                p['%s avg Ndays dry spell %s (-)' % (collect, yrs[i])] = \
                                                  np.mean([len(e) for e in ind])

                # longest dry spell duration?
                p['%s max Ndays dry spell %s (-)' % (collect, yrs[i])] = \
                                                  np.amax([len(e) for e in ind])

                # dates of longest dry spell?
                max_dates = np.argmax([len(e) for e in ind])

                if i == 0:
                    dts = year1.iloc[(ind[max_dates])]

                if i == 1:
                    dts = year2.iloc[(ind[max_dates])]
                if i == 2:
                    dts = year3.iloc[(ind[max_dates])]

                if i == 3:
                    dts = year4.iloc[(ind[max_dates])]

                if i == 4:
                    dts = year5.iloc[(ind[max_dates])]

                p['%s dates max Ndays dry spell %s (-)' % (collect, yrs[i])] = \
                          '%s--%s' % (dts.index[0].date().strftime('%d/%m/%Y'),
                                      dts.index[-1].date().strftime('%d/%m/%Y'))

                if i == 0: # dates of most recent dry spell?

                    if len(ind) > 1:
                        dts = year1.iloc[(ind[-1])]
                        recent = '%s--%s' \
                                 % (dts.index[0].date().strftime('%d/%m/%Y'),
                                    dts.index[-1].date().strftime('%d/%m/%Y'))
                        p['%s dates most recent dry spell (-)' % (collect)] = \
                                                                          recent

            except ValueError: # no dry spells
                pass

        # the interannual season's dry spell indexes
        vpds = [season1[['vpd3', 'vpd30']], season2[['vpd3', 'vpd30']], \
                season3[['vpd3', 'vpd30']], season4[['vpd3', 'vpd30']], \
                season5[['vpd3', 'vpd30']]]

        ind = []

        for i in range(len(yrs)): # loop over the 5 years

            # restrict by excess heat factor
            ii = [j for j in range(len(vpds[i])) if \
                  vpds[i]['vpd3'].iloc[j] > vpds[i]['vpd30'].iloc[j]]
            ind += [e for e in consecutive(ii) if len(e) >= 3]

        try:
            # how many dry spells over all years in that specific season?
            p['%s interannual total N dry spells in the season (-)' \
              % (collect)] = len(ind)
            # avg duration of dry spell over all years in that specific season?
            p['%s interannual avg Ndays dry spell in the season (-)' \
              % (collect)] = np.mean([len(e) for e in ind])

            # longest dry spell duration over all years in that specific season?
            p['%s interannual max Ndays dry spell in the season (-)' \
              % (collect)] = np.amax([len(e) for e in ind])

        except ValueError: # no dry spells
            pass

        # each lead up's precip, for calculation of no rain days
        ppts = [year1['pre'], year2['pre'], year3['pre'], year4['pre'], \
                year5['pre']]

        for i in range(len(yrs)): # loop over the 5 years

            # using threshold of 0.2 mm day-1 for "no precip" (BoM definition)
            BoM_thr = 0.200000002980232 # mm
            ind = [j for j in range(len(ppts[i])) if ppts[i].iloc[j] <= BoM_thr]
            ind = np.asarray([e for e in consecutive(ind)])

            try:
                # longest consecutive no rain days?
                p['%s max Ndays no ppt %s (-)' % (collect, yrs[i])] = \
                                                  np.amax([len(e) for e in ind])

                # dates of longest consecutive no rain days?
                max_dates = np.argmax([len(e) for e in ind])

                if i == 0:
                    dts = year1.iloc[(ind[max_dates])]

                if i == 1:
                    dts = year2.iloc[(ind[max_dates])]
                if i == 2:
                    dts = year3.iloc[(ind[max_dates])]

                if i == 3:
                    dts = year4.iloc[(ind[max_dates])]

                if i == 4:
                    dts = year5.iloc[(ind[max_dates])]

                p['%s dates max Ndays no ppt %s (-)' % (collect, yrs[i])] = \
                          '%s--%s' % (dts.index[0].date().strftime('%d/%m/%Y'),
                                      dts.index[-1].date().strftime('%d/%m/%Y'))

                if i == 0: # dates of most recent dry spell?

                    if len(ind) > 1:
                        dts = year1.iloc[(ind[-1])]
                        recent = '%s--%s' \
                                 % (dts.index[0].date().strftime('%d/%m/%Y'),
                                    dts.index[-1].date().strftime('%d/%m/%Y'))
                        p['%s dates most recent no ppt (-)' % (collect)] = \
                                                                          recent

            except ValueError: # no rainless consecutive days
                pass

        # the interannual season's longest consecutive no rain days
        ppts = [season1['pre'], season2['pre'], season3['pre'], \
                season4['pre'], season5['pre']]

        ind = []

        for i in range(len(yrs)): # loop over the 5 years

            ii = [j for j in range(len(ppts[i])) if ppts[i].iloc[j] <= BoM_thr]
            ind += [e for e in consecutive(ii)]

        try:
            # longest dry spell duration over all years in that specific season?
            p['%s interannual max Ndays no ppt in the season (-)' \
              % (collect)] = np.amax([len(e) for e in ind])

        except ValueError: # no rainless consecutive days
            pass

    return p


#==============================================================================

if __name__ == "__main__":

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="numpy.dtype size changed")
        warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

        # get current working directory
        cwd = os.path.dirname(os.path.realpath(sys.argv[0]))

        # run the main in current working dir
        main(cwd)

