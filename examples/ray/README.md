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
You can configure further

### 3. Execute the code
```
python3 transform.py
```

### 4. Once you are done please shut down the cluster.
```
ray down ./cluster.yaml
```
