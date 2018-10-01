# Learnable Pooling Methods for Video Classification
The repository is based on the starter code provided by Google AI. It contains a code for training and evaluating models for [YouTube-8M](https://research.google.com/youtube8m/) dataset. The detailed table of contents and descriptions can be found at [original repository](https://github.com/google/youtube-8m).

Our approach was accepted in [ECCV - The 2nd Workshop on YouTube-8M Large-Scale Video Understanding](https://research.google.com/youtube8m/workshop2018/index.html). The presentation is accessable in official ECCV Workshop page.

At the moment, we are refactoring our code for reusuability.

# Changes
- **1.00** (31 August 2018)
    - Initial public release
- **2.00** (30 September 2018)
    - Code cleaning
    - Model usage
    
# Usage
In frame_level_models.py, Prototype 1, 2 and 3 refer to Sections 3.1, 3.2 and 3.3 in our paper. Detailed instructions for performing model verification and inference can be found in the [YT8M repository](https://github.com/google/youtube-8m). Below we provide example usage of training these models:

### Prototype 1
```
python train.py --train_data_pattern="<path to tr>" --model=NetVladV1 --train_dir="<path for model checkpoints>" --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=80 --base_learning_rate=0.0002 --netvlad_cluster_size=256 --netvlad_hidden_size=512 --iterations=256 --learning_rate_decay=0.85
```
### Prototype 2
```
python train.py --train_data_pattern="<path to tr>" --model=NetVladV2 --train_dir="<path for model checkpoints>" --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=80 --base_learning_rate=0.0002 --netvlad_cluster_size=256 --netvlad_hidden_size=512 --iterations=256 --learning_rate_decay=0.85
```
### Prototype 3
```
TBD
```

# Contributors (Alphabetical Order)
- [Ruijian An](https://github.com/RuijianSZ)
- [Juhan Bae](https://github.com/pomonam)
- [Sebastian Kmiec](https://github.com/sebastiankmiec)

