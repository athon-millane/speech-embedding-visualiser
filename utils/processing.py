import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.impute import SimpleImputer, KNNImputer


def sort(df):
    """Sort features of df based on count of populated values.
    """
    return df[df.isna().sum().sort_values().index.tolist()]


def drop_empty(df, cutoff=0.95):
    """Drop any columns where a large fraction (> cutoff) is NaN.
    """
    return df[df.isna().sum().sort_values()[df.isna().sum().sort_values() < int(cutoff*(len(df)))].index.tolist()]


def replace_inf(df):
    """Replace inf and -inf with NaN.
    """
    return df.replace([np.inf, -np.inf], np.nan)
    
    
def drop_by_nunique(df,lower=0, upper=1e9):
    """Remove columns that only contain a single value other than NaN.
    """
    col_counts = df.nunique()
    drop_cols = col_counts[(col_counts <= lower) | (col_counts > upper)].index.tolist()
    return df[list(set(df.columns) - set(drop_cols))]
    

def select(df, dtype):
    """Select subset of features based on dtype. Can be 'numerical', 'categorical', or 'binary'
    """
    if dtype == 'numerical':
        return df.select_dtypes(['float64', 'int64'])
    elif dtype == 'binary':
        return df.select_dtypes('object')
    elif dtype == 'categorical':
        return df.select_dtypes('object')
    elif drype == 'date':
        return df.pipe(get_dates)


def impute(df,type='simple', strategy='most_frequent'):
    """Impute values of df according to imputation type and strategy.
    """
    if type=='simple':
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    elif type=='knn':
        imp = KNNImputer(n_neighbors=10, weights="uniform")
    return pd.DataFrame(imp.fit_transform(df), index=df.index, columns=df.columns)


def scale(df):
    """Apply minmax scaling to dataframe.
    """
    return df.transform(lambda x: minmax_scale(x))


def melt(df):
    """Apply melt operation to dataframe, assumes muli-indexed dataframe.
    """
    return df.reset_index().pipe(pd.melt,id_vars=['pin','date_daily'])


def unmelt(df,**kwargs):
    """Apply pivot table operation to 'unmelt' dataframe. Should allow melt to be reversible if used with indices.
    """
    return df.pivot_table(index=['pin','date_daily'],columns='variable',values='value',**kwargs)


def onehot(df,**kwargs):
    """One hot encode categorical features. Different to unmelt in that values of variables are broken out too. Infilled with zero.
    """
    return df.pipe(melt).pivot_table(index=['pin','date_daily'],columns=['variable','value'], aggfunc=[len],fill_value=0)
    
    
def add_binary(df):
    """Add flag for binary variables to melted df. 
    """
    nunique = df.groupby('variable')['value'].nunique().sort_values()
    binary = nunique[nunique <= 2].index.tolist()
    df['binary'] = df['variable'].isin(binary)
    return df


def remove_outliers(df_in, lower_quantile=0.01, upper_quantile=0.99):
    """Remove outlier rows for each paramater where below lower quantile and above upper quantile.
    Apply to melted df.
    """
    df = df_in.pipe(melt)
    upper = df.groupby('variable')['value'].quantile(upper_quantile).to_frame('upper')
    lower = df.groupby('variable')['value'].quantile(lower_quantile).to_frame('lower')
    nunique = df.groupby('variable')['value'].nunique().to_frame('nunique')

    df = df.merge(lower, how='left', right_index=True, left_on='variable')\
                      .merge(upper, how='left', right_index=True, left_on='variable')\
                      .merge(nunique, how='left', right_index=True, left_on='variable')
    
    df_out = df[((df['value'] >= df['lower']) & (df['value'] <= df['upper'])) | (df['nunique'] < 4)]
    return df_out.pipe(unmelt)


def filter_categorical(df, cutoff=10, plot=True):
    """Filter categorical features based on a cuttoff number of value counts.
    Apply to melted df.
    """
    from utils.vis import plot_filter
    df_vcs = df.pipe(melt).groupby(['variable'])['value'].value_counts()
    df_vcs_sub = df_vcs[df_vcs > cutoff]

    if plot:
        plot_filter(df_vcs, df_vcs_sub)
    return df.pipe(melt).set_index(['variable','value']).loc[df_vcs_sub.index].reset_index().pipe(unmelt,aggfunc='first')


def match(df_X, df_y):
    """Ensure X and y dfs contain the same rows on the same index.
    """
    merge = df_y.merge(df_X, left_index=True, right_index=True, how='inner')
    merge = merge.dropna(subset=df_y.columns)
    return merge[df_X.columns], merge[df_y.columns]


def get_dates(df):
    """Pull out date and time features and parse as type (dt64[ms]). Currently only 1 time feature - is aggregated with date conterpart and converted to dt dtype.
    """
    reg_dates = r'date[^{time}]'
    reg_times = r'time'
    dates,times = [df.columns[df.columns.str.contains(r)].tolist() for r in [reg_dates,reg_times]]
    df['datetime_admission'] = df['date_admission'].str.cat(df['datetime_admission_time'], sep=' ').astype('datetime64[ms]')
    dates.append('datetime_admission')
    return df[dates].astype('datetime64[ms]')


def get_categories(df, drop_units=True):
    """Currently all features in objects that aren't dates are parsed - this seems to have been cleaned up since the last run.
    """
    reg_units      = r'unit'
    reg_categories = r'type|site|unit|mode|source|sex|ethnicity|country|outcome'
    reg_other      = r'treatment_other_value|transfer_from_other_facility|bacteria|treatment_antiviral_value|eotd|ventilatory|cause|severity|position|AVPU|antibiotics'
    reg_rest       = r'ecmo_location_cannulation|other_infection|O2_saturation_on|complication_other_value|ability_to_self_care|rsv'
    features       = df.columns[df.columns.str.contains(reg_categories+'|'+reg_other+'|'+reg_rest)].tolist()
    if drop_units:
        blacklist  = df.columns[df.columns.str.contains(reg_units)].tolist()
    else:
        blacklist  = []
    return df[list(set(features) - set(blacklist))].astype('category')


def cap(df):
    """Suppress output of dataframe if required.
    """
    return


def filter_regex(df, columns):
    """Filter columns based on list of columns.
    """
    return df[set(df.columns.tolist()) - set(df.filter(regex='|'.join(columns)).columns.tolist())]