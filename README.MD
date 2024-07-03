

## Segment
#### Usage:
segment.py \<path:dataset> (optional: nx=\<int>) (optional: ny=\<int>) (optional: nz=\<int>) (optional: out=\<path:output>) (optional: slices_from=\<int>) (optional: slices_to=\<int>)

#### Supports:
- any raw files with volume size specified as arguments
- raw files with equally named metadata .json file in same folder

#### Requires:
- around 250MB of free Ram per 2000x2000 slice => around 310GB for full volume 
- cuda capable GPU with sufficient VRAM
- 5GB of free disk space

## Show Segmentation
#### Usage:
show_segmentation.py <path:dataset> (optional: nx=<int>) (optional: ny=<int>) (optional: nz=<int>) (optional: out=<path:output>) (optional: slices_from=<int>) (optional: slices_to=<int>)


## Show Dataset
#### Usage:
show_dataset.py <path:dataset> (optional: nx=<int>) (optional: ny=<int>) (optional: nz=<int>) (optional: out=<path:output>) (optional: slices_from=<int>) (optional: slices_to=<int>)
