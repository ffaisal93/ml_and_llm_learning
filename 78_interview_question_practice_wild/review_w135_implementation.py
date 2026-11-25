import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import completeness_score,v_measure_score


def pca_cluster_scores(df,n_components,clusters,y_true=None,seed=0):
    if y_true is None:
        raise ValueError('reference labels y_true are required for external clustering scores')
    if len(y_true)!=len(df):raise ValueError('one reference label per row required')
    if not 1<=n_components<=min(df.shape) or not 1<=clusters<=len(df):
        raise ValueError('invalid component/cluster count')
    # PCA centers but does not standardize: raw feature units are the stated choice.
    reduced=PCA(n_components=n_components,svd_solver='full').fit_transform(df.to_numpy(dtype=float))
    predicted=KMeans(n_clusters=clusters,n_init=10,random_state=seed).fit_predict(reduced)
    return completeness_score(y_true,predicted),v_measure_score(y_true,predicted)

df=pd.DataFrame({'feature1':[1,2,3,4],'feature2':[5,6,7,8]})
try:pca_cluster_scores(df,2,2)
except ValueError:pass
else:raise AssertionError('missing labels were silently invented')
assert np.allclose(pca_cluster_scores(df,2,2,[0,0,1,1]),[1.,1.])
assert completeness_score([0,0,1,1],[0,0,0,0])==1.
assert v_measure_score([0,0,1,1],[0,0,0,0])==0.
print('W135: missing-label rejection and labelled demonstration scores (1,1) passed')
