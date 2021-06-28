import pandas as pd
from pathlib import Path


def create():
    val_path = Path('data/validated.tsv')
    df = pd.read_csv(val_path, sep='\t')
    
    clients = set(df['client_id'].value_counts()[:40].index)
    
    clients_list = df[df['client_id'].map(lambda x: x in clients)]
    
    dn = clients_list.groupby('client_id').apply(lambda x: x.sample(200, random_state=42))
    d_newm = dn.reset_index(level=1, drop=True)

    d_newm.to_csv('data/new_main_train.csv', index=False)
    
    return d_newm