"""Explicit demonstration contract: iid numeric regression, numeric target.
Usage: python3 review_w142_implementation.py --csv data.csv --target outcome --out results
Without --csv: a synthetic fixture is used. Other tasks need a different contract.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error


def analyze(df,target,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if target not in df:raise ValueError('target column absent')
    missing=df.isna().sum();duplicates=int(df.duplicated().sum())
    missing.to_csv(out/'missingness.csv',header=['missing'])
    df.describe(include='all').to_csv(out/'descriptive.csv')
    # Demonstration assumption: exact duplicate rows are ingestion duplicates.
    df=df.drop_duplicates().copy()
    df[target]=pd.to_numeric(df[target],errors='raise')
    dropped=int(df[target].isna().sum());df=df.dropna(subset=[target])
    y=df.pop(target);Xtr,Xte,ytr,yte=train_test_split(df,y,test_size=.25,random_state=0)
    num=Xtr.select_dtypes(include='number').columns.tolist();cat=[c for c in Xtr if c not in num]
    prep=ColumnTransformer([
       ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
       ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    model=Pipeline([('prep',prep),('model',Ridge(alpha=1.))]).fit(Xtr,ytr)
    pred=model.predict(Xte);baseline=np.full(len(yte),ytr.mean())
    metrics={'test_n':len(yte),'model_mae':mean_absolute_error(yte,pred),
       'baseline_mae':mean_absolute_error(yte,baseline),'model_rmse':float(np.sqrt(mean_squared_error(yte,pred))),
       'duplicates_removed':duplicates,'missing_targets_removed':dropped}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame({'row_index':yte.index,'actual':yte.to_numpy(),'prediction':pred}).to_csv(out/'predictions.csv',index=False)
    (out/'report.txt').write_text('Chosen iid regression demonstration. Inspect descriptive/missingness tables.\n'
        'Median/mode imputation fitted on train only; exact duplicates assumed ingestion repeats.\n'
        'No outliers removed automatically. Target missing rows excluded and counted.\n'
        'Metrics: '+json.dumps(metrics)+'\n'
        'Limitations: split invalid for repeated entities/time series; target/feature availability needs review.\n')
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv');p.add_argument('--target',default='outcome');p.add_argument('--out',default='w142_results');a=p.parse_args()
    rng=np.random.default_rng(0)
    df=pd.read_csv(a.csv) if a.csv else pd.DataFrame({'x':np.arange(80.),'category':['a','b']*40,'outcome':2*np.arange(80.)+rng.normal(size=80)})
    result=analyze(df,a.target,a.out)
    assert result['test_n']>0 and np.isfinite(result['model_mae'])
    print('ACTUAL CSV' if a.csv else 'SYNTHETIC DEMONSTRATION',result)
