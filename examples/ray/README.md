# Distributed processing with Ray


## Execute locally
Install hub+ray and run the script locally to create a dataset of size `num_samples`
```
pip3 install -r requirements.txt
python3 transform.py --num_workers 2 --ds_out ./tmp/cars --num_samples 1000
```


## Execute on a cluster
#### 1. Start the cluster
Requires AWS credentials. If you skip this step, it will start a ray cluster on your machine. You can further modify the cluster in cluster.yaml
```
ray up ./cluster.yaml
```

#### 2. Execute the code, dataset created on head node
```
ray exec ./cluster.yaml "python3 ~/hub/transform.py --num_workers 2 --ds_out s3://bucket/dataset_name --num_samples 1000"
```
Change number of workers to 6 once all workers are up.


#### 4. Shut down the cluster
```
ray down ./cluster.yaml
```

Notes

* To monitor workers, utilization and jobs use this command
```
ray dashboard ./cluster.yaml
```

* Update locally the code and sync the cluster

```
ray rsync-up ./cluster.yaml
```

* Attach and execute locally
```
ray attach ./cluster.yaml
> python3 ~/hub/transform.py --num_workers 2
```
