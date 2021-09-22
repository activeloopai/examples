import time
import argparse
import tqdm
import numpy as np
from PIL import Image

import hub

SAMPLES = 500
DS_OUT_PATH = "./data/cars_out" # "s3://snark-test/testing" 

parser = argparse.ArgumentParser(description='PyTorch RPC Batch RL example')
parser.add_argument('--samples', type=int, default=SAMPLES, metavar='S',
                    help='how many samples dataset should have')
parser.add_argument('--ds_out', type=str, default=DS_OUT_PATH, metavar='O',
                    help='dataset path to be transformed into')

args = parser.parse_args()


def define_dataset(path: str, n_samples: int = 100) -> hub.Dataset:
    """ Define the dataset """
    ds = hub.empty(path, overwrite=True)
    
    ds.create_tensor("labels", htype="class_label")
    ds.create_tensor("images", htype="image", sample_compression="jpeg")
    ds.create_tensor("images_downsampled")
    
    # Define tensor with customized htype, compression and chunk_size
    ds['images_downsampled'].meta.htype = ds['images'].meta.htype
    ds['images_downsampled'].meta.sample_compression = ds['images'].meta.sample_compression
    ds['images_downsampled'].meta.max_chunk_size = 1 * 1024 * 1024
    
    return ds

# Define the remote compute
@hub.compute
def downsample(index, samples_out):
    """ Takes image from a sample_in, downsamples it and pushes to a new tensor """
    array = (255*np.random.random((100,100,3))).astype(np.uint8)
    img = Image.fromarray(array)
    max_d = max(img.size[0], img.size[1])
    min_s = min(100, max_d)
    ratio = max_d // min_s
    img_downsampled = img.resize((img.size[0] // ratio, img.size[1] // ratio))
    array_downsampled = np.array(img_downsampled)
    
    samples_out.images.append(array)
    samples_out.images_downsampled.append(array_downsampled)
    samples_out.labels.append(index)

    
if __name__ == "__main__":
    
    # Define a dataset and fill in random images
    ds_out = define_dataset(args.ds_out, args.samples)

    # Run the distributed computation
    t1 = time.time()
    downsample().eval(list(range(args.samples)), ds_out, num_workers=12, scheduler="ray")
    t2 = time.time()
    print(f"The processing took {t2-t1}")