"""W089: deterministic cleaning + train-only learned preprocessing."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support

def baseline(df,numeric,categorical,mixed_column,identifier,aliases=None):
    features=list(numeric)+list(categorical)
    if len(set(features))!=len(features) or 'target' in features or identifier in features:
        raise ValueError('Features must be unique and exclude target/ID')
    if mixed_column not in numeric: raise ValueError('Mixed column must be declared numeric')
    if df['target'].isna().any() or set(df['target'].unique())!={0,1}:
        raise ValueError('Need nonmissing binary target with both classes')
    profile={'types':df.dtypes.astype(str).to_dict(),
             'missing':df.isna().sum().to_dict(),
             'categorical_cardinality':df[categorical].nunique(dropna=False).to_dict(),
             'target_counts':df['target'].value_counts().to_dict()}
    X_train,X_test,y_train,y_test=train_test_split(df[features],df['target'],
        test_size=.25,random_state=7,stratify=df['target'])
    def clean(frame):
        frame=frame.copy()
        frame[mixed_column]=pd.to_numeric(frame[mixed_column],errors='coerce')
        for col in categorical:
            values=frame[col].astype('string').str.strip().str.casefold()
            values=values.replace((aliases or {}).get(col,{}))
            # Use ordinary object values and None for consistent sklearn missing handling.
            frame[col]=values.astype(object).where(values.notna(),None)
        return frame
    numeric_pipe=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),
                           ('scale',StandardScaler())])
    category_pipe=Pipeline([('impute',SimpleImputer(strategy='most_frequent',missing_values=None)),
                            ('onehot',OneHotEncoder(handle_unknown='ignore'))])
    prep=ColumnTransformer([('numeric',numeric_pipe,numeric),('category',category_pipe,categorical)])
    model=Pipeline([('preprocess',prep),('classifier',LogisticRegression(max_iter=1000))])
    model.fit(clean(X_train),y_train)
    predicted=model.predict(clean(X_test))
    minority=int(y_train.value_counts().idxmin())
    precision,recall,_,_=precision_recall_fscore_support(y_test,predicted,
        labels=[minority],average=None,zero_division=0)
    metrics={'accuracy':float(accuracy_score(y_test,predicted)),
             'confusion_matrix_labels':[0,1],
             'confusion_matrix':confusion_matrix(y_test,predicted,labels=[0,1]).tolist(),
             'minority_label':minority,'minority_precision':float(precision[0]),
             'minority_recall':float(recall[0]),'test_rows':len(y_test)}
    return model,profile,metrics,clean

if __name__=='__main__':
    # A declared fixture, not the unavailable interview dataset.
    rows=[]
    for i in range(80):
        rows.append(dict(id=i,income=None if i%9==0 else i,
            balance='bad' if i%13==0 else str(i%17),
            city=['NY','ny ','New York',None][i%4],target=int(i%5==0)))
    data=pd.DataFrame(rows)
    model,profile,metrics,clean=baseline(data,['income','balance'],['city'],'balance','id',
                                       aliases={'city':{'ny':'new york'}})
    assert metrics['test_rows']==20 and 'id' not in model.feature_names_in_
    assert clean(pd.DataFrame({'income':[1],'balance':['bad'],'city':[' NY ']}))['city'][0]=='new york'
    print('Synthetic held-out metrics:',metrics)
