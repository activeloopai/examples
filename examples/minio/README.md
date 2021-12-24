
## Run Hub on MinIO

Run MinIO container and create a bucket called `myBucket` using UI at `http://localhost:9001`. 
```
docker run \
  -p 9000:9000 \
  -p 9001:9001 \
  -e "MINIO_ROOT_USER=username" \
  -e "MINIO_ROOT_PASSWORD=password" \
  quay.io/minio/minio server /data --console-address ":9001"
```

Then run the following script to test hub
```
python3 examples/minio/minio.py
```