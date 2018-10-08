# Learnable Pooling Methods for Video Classification
The repository is based on the starter code provided by Google AI. It contains a code for training and evaluating models for [YouTube-8M](https://research.google.com/youtube8m/) dataset. The detailed table of contents and descriptions can be found at [original repository](https://github.com/google/youtube-8m).

The repository contains models from team "Deep Topology". Our approach was accepted in [ECCV - The 2nd Workshop on YouTube-8M Large-Scale Video Understanding](https://research.google.com/youtube8m/workshop2018/index.html). The presentation is accessible in ECCV Workshop page.

Presentation: TBA \
Paper: [Link](paper/Learnable_Pooling_Methods_for_Video_Classification.pdf), [Arxiv](https://arxiv.org/abs/1810.00530)
    
# Usage
In [frame_level_models.py](frame_level_models.py), prototype 1, 2 and 3 refer to sections 3.1, 3.2 and 3.2 in the paper. The detailed instructions instructions to train and evaluate the model can be found at [YT8M repository](https://github.com/google/youtube-8m). The following is the example training command to reproduce the result.
### Prototype 1 (Attention Enhanced NetVLAD)
```
python train.py --train_data_pattern="<path to train .tfrecord>" --model=NetVladV1 --train_dir="<path for model checkpoints>" --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=80 --base_learning_rate=0.0002 --netvlad_cluster_size=256 --netvlad_hidden_size=512 --iterations=256 --learning_rate_decay=0.85
```
### Prototype 2 (NetVLAD with Attention Based Cluster Similarities)
```
python train.py --train_data_pattern="<path to train .tfrecord>" --model=NetVladV2 --train_dir="<path for model checkpoints>" --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=80 --base_learning_rate=0.0002 --netvlad_cluster_size=256 --netvlad_hidden_size=512 --iterations=256 --learning_rate_decay=0.85
```
### Prototype 3 (Regularized Function Approximation Approach)
```
TBD
```

# Changes
- **1.00** (31 August 2018)
    - Initial public release
- **2.00** (30 September 2018)
    - Code cleaning
    - Model usage
    
# Citations
If you find our apporaches useful, please cite our paper.
```
@article{kmiec2018learnable,
  title={Learnable Pooling Methods for Video Classification},
  author={Kmiec, Sebastian and Bae, Juhan and An, Ruijian},
  journal={arXiv preprint arXiv:1810.00530},
  year={2018}
}
```

# Contributors (Alphabetical Order)
- [Ruijian An](https://github.com/RuijianSZ)
- [Juhan Bae](https://github.com/pomonam)
- [Sebastian Kmiec](https://github.com/sebastiankmiec)

