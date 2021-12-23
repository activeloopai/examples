## Run Hub on AIStore

Start a production-ready standalone docker of aistore. More information can be found [here](https://nvidia.github.io/aistore/deploy/prod/docker/single).
```sh
docker run \
    -p 51080:51080 \
    -v $(mktemp -d):/ais/disk0 \
    aistore/cluster-minimal:latest
```

Run the script
```
python3 examples/aistore/simple.py
```