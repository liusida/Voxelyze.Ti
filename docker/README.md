# Run in docker

Not ready yet.

```bash
docker build -t voxelyze .
docker run --mount type=bind,source=/home/liusida/code/,target=/code/ -it --rm --gpus all voxelyze
```
