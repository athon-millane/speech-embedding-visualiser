# Manifold Learning Utils
#
# Utils to understand and visualise data using manifold learning in cudf.
#
# Authors: Athon Millane, 2020
#
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

try:
    import cudf
    from cuml import PCA, TSNE, UMAP
    CPU = False
except:
    from umap import UMAP
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import warnings
    warnings.filterwarnings('ignore')
    CPU = True
    

if CPU:
    def t(df):
        r = df.astype(np.float32).values
        if r.shape[1] == 1:
            r = r.reshape(-1, )
        return r
    def f(res):
        return pd.DataFrame(res)
else:
    def t(df):
        return df


algos = {'PCA':PCA, 'tSNE':TSNE, 'UMAP':UMAP}


def get_manifold(df, algo, dim=2):
    """
    auth:ajm
    Fit-transform the chosen manifold algo (PCA, tSNE or UMAP) on the chosen df.
    Default dimension is 2.
    """
    return f(algo(n_components=dim).fit_transform(t(df))).set_index(df.index)
        

def supervised_umap(df_X, df_y, labels_dict, ax=None, label='Label', emb=False):
    """
    auth:ajm
    Plotting tool for supervised umap, currently designed for binarised versions of continuous label spaces.
    Will split labels to boolean map based on cutoff point then colour points in learned manifold accordingly.
    """
    from matplotlib.colors import ListedColormap

    # sns.set_palette("coolwarm")
    cmap = ListedColormap(sns.hls_palette(n_colors=len(labels_dict.values())).as_hex())

    # feed in raw data
    if emb:
        m = df_X
    else:
        m = f(UMAP(target_metric = "categorical").fit_transform(t(df_X), t(df_y)))
    
    if not ax:
        fig,ax = plt.subplots(figsize=[16,8])
    _=ax.scatter(m.values[:,0],m.values[:,1],alpha=1,s=3,c=t(df_y),cmap=cmap,label=label)
    cbar = plt.colorbar(_, boundaries=np.array(list(labels_dict.values())+[2])-0.5, ax=ax)
    cbar.set_ticks(list(labels_dict.values())); cbar.set_ticklabels(list(labels_dict.keys()))
    
    
def umap_embed(X_train, y_train, X_test, dim=2, cpu=CPU):  
        
    # get UMAP embeddings on train set
    X_train_umap  = f(UMAP(target_metric = "categorical", n_components=dim).fit(t(X_train), t(y_train)).transform(t(X_train)))
    X_test_umap   = f(UMAP(target_metric = "categorical", n_components=dim).fit(t(X_train), t(y_train)).transform(t(X_test)))
    
    return f(X_train_umap), f(X_test_umap)


def plot_manifolds(manifolds, titles, labels=None, savefig=False, alpha=1, figsize=[24,8]):
    """
    auth:ajm
    Plot n manifolds - 3 works best. If labels are provided (as a cudf series), then colour points by label space.
    """
    from matplotlib import colors
    from matplotlib.colors import ListedColormap
    
    # Discrete colour map
    # cmap = ListedColormap(sns.color_palette('coolwarm').as_hex())
    
    # Continuous colour map
    cmap = 'viridis'


    size=2000/len(manifolds[0].values[:,0])
    if labels is not None:
        h,w = len(labels.columns),len(titles)
        fig,ax = plt.subplots(h,w,figsize=figsize)
        for i,l in enumerate(labels.columns):
            for j,t in enumerate(titles):
                m = manifolds[j].reset_index(drop=True)
                mask = ~(labels[l].isna())
                c = labels[l]
                if h == 1:
                    ax = [ax]
                if w == 1:
                    ax = [ax]
                _=ax[i][j].scatter(m.values[:,0],m.values[:,1],alpha=alpha,s=5,c=c,label=labels,cmap=cmap,norm=colors.DivergingNorm(vmin=c.min(), vcenter=c.median(), vmax=c.max()))
                ax[i][j].set_title(t)
                ax[i][j].set_xlabel('PC1')
                ax[i][j].set_ylabel('PC2')
                # legend
                # legend1 = ax[j].legend(*_.legend_elements(),
                #     loc="lower left", title="Compliance")
                # ax[j].add_artist(legend1)
    else:
        fig,ax = plt.subplots(1,len(titles),figsize=[20,6])
        for i,m in enumerate(zip(manifolds, titles)):
            t = m[1]
            m = m[0].reset_index(drop=True)
            ax[i].scatter(m.values[:,0],m.values[:,1],alpha=1,s=5)
            ax[i].set_title(t)
            
    if savefig:
        try:
            fignum = max([int(x.split('_')[-1].split('.')[0]) for x in glob.glob('../figures/*.png')]) + 1
        except:
            fignum = 0
        fig.tight_layout();
        plt.savefig('figures/figure_{}'.format(fignum))


def plot_umap_transform(X_train_umap, X_test_umap, y_train, y_test, labels_dict):
    fig,ax=plt.subplots(1,2,figsize=[16,6])
    supervised_umap(X_train_umap, y_train, labels_dict=labels_dict, ax=ax[0], emb=True)
    supervised_umap(X_test_umap, y_test, labels_dict=labels_dict, ax=ax[1], emb=True)
    
    
def summary(df, exps=['PCA'], labels=None, **kwargs):
    """
    """
    algos = {k:v for k,v in {'PCA':PCA, 'tSNE':TSNE, 'UMAP':UMAP}.items() if k in exps}
    manifolds = [get_manifold(df.astype('float32'), a) for a in algos.values()]
    plot_manifolds(manifolds, algos.keys(), labels=labels, savefig=False, **kwargs)
    return df
    
    
def binary_summary(df_X, df_y):
    """
    """
    cutoffs = [6, 12, 24, 36, 48, 72]
    fig,axs = plt.subplots(2,3,figsize=[16,8])
    [supervised_umap(axs[i//3, i%3], df_X.astype('float32'), df_y, cutoff=c) for i,c in enumerate(cutoffs)]
    fig.legend(loc='center right'); fig.tight_layout(); fig.subplots_adjust(top=0.9, right=0.82)