import pandas as pd

file_name='src/backend/data/images_dataset.csv'

tsc_file='src/backend/data/photos.tsv000'

df=pd.read_csv(tsc_file,sep='\t',header=0)
dataset=df.to_csv(file_name)
data=pd.read_csv(file_name)

final_df=data[["photo_id","photo_image_url"]]

def get_df(start_index,end_index):
    df = final_df[start_index:end_index] 
    return df