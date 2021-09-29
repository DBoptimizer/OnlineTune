import argparse
import pandas as pd
import numpy as np
THREAD = 64



parser = argparse.ArgumentParser(description='input the inputfileName')
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()


query_type =  ['select', 'insert', 'update', 'delete']
def one_hot_query_type(y):
    y_one_hot = np.zeros((len(y), len(query_type )))
    for i, label in enumerate(y):
        if label in query_type:
            y_one_hot[i, query_type.index(label)] = 1
        else:
            y_one_hot[i, 0] = 1
    return y_one_hot


df =  pd.read_csv(args.input, delimiter = "\t", names=['id', 'select_type', 'table', 'partitions', 'type', 'possible_keys', 'key', 'key_len', 'ref', 'rows', 'filtered' ,'Extra'])
try:
    df.rows = df.rows.astype('float')
except:
    index_drop_mask = np.array(df.rows.str.isnumeric()==False)
    df = df.drop(np.array(df.index)[index_drop_mask])
    df['rows'] = df['rows'].astype('float')

try:
    df['filtered'] = df['filtered'].astype('float')
except:
    index_drop_mask = np.array(df.filtered.str.isnumeric() == False)
    df = df.drop(np.array(df.index)[index_drop_mask])
    df['filtered'] = df['filtered'].astype('float')

df['select_type'] = df['select_type'].str.lower()
df['rows_filtered'] = df['rows'] * df['filtered']/100

y_query_type = one_hot_query_type(df['select_type'])
num_query = float(df.shape[0])
vecL = list(sum(y_query_type) / num_query)
rows = df['rows'].sum() / num_query
vecL.append(rows)
rows_filtered = df['rows_filtered'].sum() / num_query
vecL.append(rows_filtered)
num_index = (df.shape[0] - df['key'].isna().sum() )/ num_query
vecL.append(num_index)
vecL.append(num_query)

print (vecL)


