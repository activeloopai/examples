import os
import pandas as pd
import numpy as np
import deeplake
from tqdm import tqdm
from pathlib import Path

token = "{YOUR_TOKEN}"

columns = [('SAMPLE_ID', deeplake.htype.DEFAULT), 
            ('URL', 'link[image]'),
            ('TEXT', deeplake.htype.TEXT),
            ('HEIGHT', deeplake.htype.DEFAULT), 
            ('WIDTH', deeplake.htype.DEFAULT), 
            ('LICENSE', deeplake.htype.TEXT),
            ('NSFW', deeplake.htype.TEXT),
            ('similarity', deeplake.htype.DEFAULT)]

def create_dataset(path: str, columns: list) -> deeplake.Dataset:
    if deeplake.exists(path, token=token):
        return deeplake.load(path, token=token)
    
    ds = deeplake.dataset(path, overwrite=True, token=token)
    for key, htype in columns:
        ds.create_tensor(key, htype=htype, verify=False, create_shape_tensor=False, create_sample_info_tensor=False, create_id_tensor=False)
        
    return ds


@deeplake.compute
def process(df: pd.DataFrame, ds_out: deeplake.Dataset):
    # vectorized extend default tensors in one line
    for key, _ in tqdm(filter(lambda x: x[1] == deeplake.htype.DEFAULT, columns)):
        ds_out[key].extend(df[key].values)
        
    # append image links
    for i in range(len(df)):
        link_sample = df.iloc[i,:]['URL']
        if link_sample is None: 
            link_sample = ''
        ds_out.URL.append(deeplake.link(link_sample))
    
        # append other tensors
        for key, _ in filter(lambda x: x[1] in (deeplake.htype.TEXT, deeplake.htype.CLASS_LABEL), columns):
            el = df.iloc[i,:][key]
            if el is None: # class_label -> Nones  (?)
                el = ""
            ds_out[key].append(el)

def ingest(ds: deeplake.Dataset, columns: list, source: str):
    
    df = pd.read_parquet(source)
    step = 5000
    # df = df.head(10000)
    process().eval([df[step*i:step*i+step] for i in range(len(df)//step)]+[df[step*(len(df)//step):]], ds, scheduler='processed', num_workers=16)
    
    ds.commit("uploading dataset")          
    print(len(ds))
    
if __name__ == "__main__":
    path = 'hub://laion/laion-400M'
    source_path = '/data2/laion/'
    
    ds = create_dataset(path, columns)
    
    data_dir = Path(source_path)
    paths = list(data_dir.glob('*.parquet'))
    paths.sort()
    
    # ingest every parquet file
    for source in paths:
        print(f"processing: {source}")
        ingest(ds, columns, source)
