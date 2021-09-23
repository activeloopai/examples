## Distributed processing with Ray

### 1. Install ray on your local machine
```
pip3 install ray==1.6
```

### 2. Start a ray cluster and attach to it (Optional)
Requires AWS credentials. If you skip this step, it will start a ray cluster on your machine.
```
ray up ./cluster.yaml
ray attach ./cluster.yaml
```
You can further modify the cluster in cluster.yaml

### 3. Execute the code
```
python3 ~/hub/transform.py --num_workers 2
```

or to store on an S3
```
python3 ~/hub/transform.py --num_workers 2 --ds_out s3://bucket/dataaset
```
Change number of workers to 6 once all workers are up.


### 4. Once you are done please shut down the cluster.
```
exit
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

* Directly execute code on the cluster from local machine
```
ray exec ./cluster.yaml "python3 ~/hub/transform.py --num_workers 2"
```
