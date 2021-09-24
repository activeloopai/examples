<img src="https://static.scarf.sh/a.png?x-pxid=bc3c57b0-9a65-49fe-b8ea-f711c4d35b82" /><p align="center">
    <img src="https://www.linkpicture.com/q/hub_logo-1.png" width="35%"/>
    </br>
    <h1 align="center">Examples for <a href="https://github.com/activeloopai/Hub">Hub</a> - Dataset Format for AI
 </h1>
 
 
A repository showcasing examples of using [Hub](https://github.com/pytorch/pytorch)
 - [Uploading Dataset Places365](datasets/places365)
 
 
## Getting Started with Hub 🚀 


### Installation
Hub is written in 100% python and can be quickly installed using pip.
```sh
pip3 install hub
```


### Creating Datasets

A hub dataset can be created in various locations (Storage providers). This is how the paths for each of them would look like:

| Storage provider | Example path                  |
| ---------------- | ----------------------------- |
| Hub cloud        | hub://user_name/dataset_name  |
| AWS S3           | s3://bucket_name/dataset_name |
| GCP              | gcp://bucket_name/dataset_name|
| Local storage    | path to local directory       |
| In-memory        | mem://dataset_name            |



Let's create a dataset in the Hub cloud. Create a new account with Hub from the terminal using `activeloop register` if you haven't already. You will be asked for a user name, email id and passowrd. The user name you enter here will be used in the dataset path.

```sh
$ activeloop register
Enter your details. Your password must be atleast 6 characters long.
Username:
Email:
Password:
```

Initialize an empty dataset in the hub cloud:

```python
import hub

ds = hub.empty("hub://<USERNAME>/test-dataset")
```

Next, create a tensor to hold images in the dataset we just initialized:

```python
images = ds.create_tensor("images", htype="image", sample_compression="jpg")
```

Assuming you have a list of image file paths, lets upload them to the dataset:

```python
image_paths = ...
with ds:
    for image_path in image_paths:
        image = hub.read(image_path)
        ds.images.append(image)
```

Alternatively, you can also upload numpy arrays. Since the `images` tensor was created with `sample_compression="jpg"`, the arrays will be compressed with jpeg compression.


```python
import numpy as np

with ds:
    for _ in range(1000):  # 1000 random images
        radnom_image = np.random.randint(0, 256, (100, 100, 3))  # 100x100 image with 3 channels
        ds.images.append(image)
```



### Loading Datasets


You can load the dataset you just created with a single line of code:

```python
import hub

ds = hub.load("hub://<USERNAME>/test-dataset")
```

You can also access other publicly available hub datasets, not just the ones you created. Here is how you would load the [Objectron Bikes Dataset](https://github.com/google-research-datasets/Objectron):

```python
import hub

ds = hub.load('hub://activeloop/objectron_bike_train')
```

To get the first image in the Objectron Bikes dataset in numpy format:


```python
image_arr = ds.image[0].numpy()
```



## Documentation
Getting started guides, examples, tutorials, API reference, and other usage information can be found on our [documentation page](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme). 
