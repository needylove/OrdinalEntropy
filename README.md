# OrdinalEntropy
The official code of "Improving Deep Regression with Ordinal Entropy" in ICLR 2023. [[PDF]](https://openreview.net/forum?id=raU07GpP0P).

We currently provide a detailed code for experiments on the synthetic dataset, with a new visualization experiments for easy reproduction.

For the real-world tasks, we now only provide the code of the ordinal entropy regularizer, the trained models and some corresponding details will be provided later.
## Experiments on the synthetic dataset

### Obtain experiments results on the synthetic dataset
- run main.py

### Visualization experiment on the synthetic dataset
We add a new visualization experiment with the synthetic dataset for easy reproduction, as the visualization experiments in our paper is on depth estimation task, which may take some effort to reproduce. 

- run vis_tsne.py to obtain the features
- run vis_sphere.py to visualize the obtained features on a sphere

### Dataset

For the Linear task:
- train.npy : the traning set 
- test.npy: the test set

For the non-linear task:
- train_sde.npy : the traning set 
- test_sde.npy: the test set, please download it [here](https://drive.google.com/file/d/19gmrPb2PG8LTp_Lz5b7S0QGXdlEyVpNc/view?usp=sharing).

The dataset above is generated with this code: [DeepONet](https://github.com/lululxvi/deeponet). 


## Experiments on the Depth Estimation and Crowd Counting

The code for the Depth Baseline can be found here:
- [NeW-CRFs](https://github.com/aliyun/NeWCRFs). 

The code for the Crowd Counting Baseline can be found here:
- [CSRNet](https://github.com/leeyeehoo/CSRNet-pytorch). 

The ordinal entropy code for the two tasks can be found here: 
- ./DepthEstimation&CrowdCounting/OrdinalEntropy.py

The ordinal entropy can be added into the New-CRFs and CSRNet baselines by:
- change the output of models from
```
        returen x
```

to 
```
        if self.training:
            return x, encoding
        else:
            return x
```

- add the ordinal entropy into the loss:
change
```
outputs = model(inputs, targets, epoch)
```
to 
```
outputs, features = model(inputs, targets, epoch)
oe_loss = ordinalentropy(features, targets)
loss = loss + oe_loss
```

### Visualization results on depth-estimation with NYU-v2
The visualization results can be obtained by:

- run vis_sphere.py to visualize the obtained features on a sphere


## Experiments on the Age Estimation

The code for the Baseline can be found here:
- [Imbalanced Regression](https://github.com/YyzHarry/imbalanced-regression/tree/main/agedb-dir). 

The ordinal entropy code for Age Estimation can be found here: 
- ./AgeEstimation/OrdinalEntropy.py

The ordinal entropy can be added into the Age Estimation baselines in a similar way shown above.

## Reference

S. Zhang, L. Yang, M. Bi Mi, X. Zheng, A. Yao, "Improving Deep Regression with Ordinal Entropy," in ICLR, 2023. [[PDF]](https://openreview.net/forum?id=raU07GpP0P).
