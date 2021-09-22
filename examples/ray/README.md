## Distributed processing with Ray

1. Install ray on your local machine
```
pip3 install ray==1.6
```

2. Start a ray cluster (Optional)
* Optional, if you skip it will start a local ray cluster

Start the cluster and attach to it
```
ray up ./cluster.yaml
ray attach ./cluster.yaml
```

3. Execute the code
```
python3 transform.py
```

4. Once you are done please shut down the cluster.
```
ray down ./cluster.yaml
```