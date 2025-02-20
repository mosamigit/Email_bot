import pandas as pd


df_1 = pd.read_parquet('dataset_850prod_updated.parquet.gzip') ##working
# df = pd.read_parquet('Askiris_package.parquet.gzip')
# df[['Company Name', 'Company_specification', 'n_tokens', 'embeddings']]= ['name', 'text', 'n_tokens', 'embeddings']
print(df_1)