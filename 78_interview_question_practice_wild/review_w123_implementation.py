"""Demonstration schema ONLY: client_id, session_id, session_at, attended.
One decision at cutoff t; label=any attendance in [t,t+7d).
"""
import pandas as pd


def coaching_table(events,cutoff,observed_through):
    required={'client_id','session_id','session_at','attended'}
    if not required<=set(events):raise ValueError(f'missing columns: {required-set(events)}')
    e=events.copy()
    if e[['client_id','session_id']].isna().any().any():raise ValueError('missing identity')
    e['session_at']=pd.to_datetime(e.session_at,utc=True,errors='raise')
    if not e.attended.isin([0,1]).all():raise ValueError('attended must be 0/1')
    # Same id with conflicting payload is not silently accepted.
    if (e.groupby('session_id').nunique().max(axis=1)>1).any():raise ValueError('conflicting duplicate session')
    e=e.drop_duplicates('session_id')
    t=pd.Timestamp(cutoff);t=t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    observed=pd.Timestamp(observed_through);observed=observed.tz_localize('UTC') if observed.tzinfo is None else observed.tz_convert('UTC')
    if observed<t+pd.Timedelta(days=7):raise ValueError('future label window has not matured')
    # Eligibility: only clients known BEFORE the prediction cutoff.
    clients=pd.Index(e.loc[e.session_at<t,'client_id'].unique(),name='client_id')
    past=e[(e.session_at>=t-pd.Timedelta(days=28)) & (e.session_at<t)]
    features=past.groupby('client_id').agg(sessions_28d=('session_id','size'),attended_28d=('attended','sum'))
    result=features.reindex(clients,fill_value=0)
    last=past.groupby('client_id').session_at.max()
    result['days_since_session']=(t-last.reindex(clients)).dt.total_seconds()/86400
    future=e[(e.session_at>=t)&(e.session_at<t+pd.Timedelta(days=7))]
    result['return_7d']=future.groupby('client_id').attended.max().reindex(clients,fill_value=0).astype(int)
    return result.reset_index()

if __name__=='__main__':
    events=pd.DataFrame({'client_id':['a','a','b','b','new'], 'session_id':[1,2,3,4,5],
        'session_at':['2026-01-30','2026-02-03','2026-01-20','2026-02-02','2026-02-04'],
        'attended':[1,1,1,0,1]})
    out=coaching_table(events,'2026-02-01','2026-02-08')
    assert out.client_id.tolist()==['a','b'] and out.return_7d.tolist()==[1,0]
    assert out.sessions_28d.tolist()==[1,1] and out.days_since_session.tolist()==[2.,12.]
    assert coaching_table(pd.concat([events,events.iloc[:1]]),'2026-02-01','2026-02-08').equals(out)
    try:coaching_table(events,'2026-02-01','2026-02-05')
    except ValueError:pass
    else:raise AssertionError('immature labels accepted')
    print(out.to_string(index=False))
