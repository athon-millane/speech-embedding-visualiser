import matplotlib.pyplot as plt
import seaborn as sns
from utils.processing import scale, melt
from utils.manifold import get_manifold, supervised_umap

# sns.set_palette("husl")

def spy(df,title='Matrix non-zero values',**kwargs):
    """Wrapper for matplotlib spy function on matrix.
    """
    fig,ax=plt.subplots(1,1,**kwargs)
    fig.suptitle(title)
    ax.spy(df.isna(), aspect='auto')
    fig.tight_layout(); 
    if kwargs and 'figsize' in kwargs.keys(): 
        fig.subplots_adjust(top=1-0.8*(1/kwargs['figsize'][1]))
    else:
        fig.subplots_adjust(top=0.85)
    return df
        
        
def look(df_in,title="Values in df",scale_df=True,**kwargs):
    """Visualise values in df.
    """
    fig = plt.figure(**kwargs)
    fig.suptitle(title)
    df = df_in.pipe(scale) if scale_df else df_in
    _=plt.imshow(df, interpolation='nearest', aspect='auto')
    _=plt.colorbar(label='Values')
    fig.tight_layout(); fig.subplots_adjust(top=0.9)
    fig.show()
    return df_in
        
        
def plot_nunique(df,title='Number of unique values',log=True):
    """Plot count of unqiue values for non-binary columns.
    """
    fig = plt.figure(figsize=[16,4])
    fig.suptitle(title)
    ax = df.pipe(melt).groupby('variable')['value'].nunique().sort_values().plot.bar(logy=log)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    if (len(ax.get_xticklabels()) > 50): # turn of axis labels
        ax.set_xticklabels('')
    return df
    
    
def plot_dists(df_in, title='Signals Distribution'):
    """Plot distribution of signals for non-binary columns. Expects melted df.
    """
    df=df_in.pipe(melt).copy()
    df['value'] = df['value']+1e-4
    fig,axs=plt.subplots(2,1,figsize=[16,8],sharex=True)
    fig.suptitle(title)
    sns.boxenplot(data=df, x='variable', y='value',ax=axs[0]).set_title('Linear')
    sns.boxenplot(data=df, x='variable', y='value',ax=axs[1]).set_title('Logarithmic')
    axs[1].set_yscale('log')
    [ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right') for ax in axs]
    plt.tight_layout(); plt.subplots_adjust(top=0.9)
    return df_in
    
    
def plot_filter(df_vcs,df_vcs_sub):
    """Plot outcome of categorical filter.
    """
    fig,axs=plt.subplots(2,1,figsize=[32,12])
    _=df_vcs.plot.bar(logy=True, ax=axs[0]).set_title('Before Categorical Filter')
    _=df_vcs_sub.plot.bar(logy=True, ax=axs[1]).set_title('After Categorical Filter')
    [ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right') for ax in axs]
    if (len(df_vcs) > 50): # turn of axis labels
        axs[0].set_xticklabels('')
    plt.tight_layout(); plt.subplots_adjust(wspace=0.5)
    plt.show()
    
    
def eda(df):
    """Spy+summary plot shortcut.
    """
    spy(df,figsize=[16,4])
    summary(df)
    return df