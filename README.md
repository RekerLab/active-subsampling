
![ActiveSubsampling3](https://user-images.githubusercontent.com/127516906/229308666-bfce275d-7e73-410f-901e-7ec408a3cf07.png)



## Improving Molecular Machine Learning Through Adaptive Subsampling with Active Learning

![toc_graphic_white](https://user-images.githubusercontent.com/56095399/227727724-404e2bfb-fcd4-497d-bc77-3acff026ff2b.png)

We use active machine learning as an autonomous and adaptive data subsampling strategy and show that active learning-based subsampling can lead to better molecular machine learning performance when compared to both training models on the complete training data and 19 state-of-the-art subsampling strategies. We find that active learning is robust to errors in the data, highlighting the utility of this approach for low-quality datasets. Taken together, we here describe a new, adaptive machine learning pre-processing approach and provide novel insights into the behavior and robustness of active machine learning for molecular sciences.

For more information, please refer to: [Improving Molecular Machine Learning Through Adaptive Subsampling with Active Learning](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/63e5c76e1d2d18406337135d/original/improving-molecular-machine-learning-through-adaptive-subsampling-with-active-learning.pdf)

If you use this data or code, please kindly cite: Wen, Y., Li, Z., Xiang, Y., & Reker, D. (2023). Improving Molecular Machine Learning Through Adaptive Subsampling with Active Learning.

<br>

## Files

- **Example_workflow_for_AL_Subsampling.ipynb** contains an example notebook that runs BBBP but can be run out of the box on a local machine or on Google Colab to apply this technique to new datasets

<br>


## Installation
```
pip install git+https://github.com/RekerLab/active-subsampling
```

<br>

## Quickstart

Datasets can be loaded from DeepChem
```
#load data
import deepchem as dc
tasks, data, transformers = dc.molnet.load_bbbp(splitter=None)
bbbp = data[0]
```

Model and performance metric need to be initialized, we recommend random forest models and Matthew's correlation coefficient (MCC)
```
# initialize model and performance metric
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.ensemble import RandomForestClassifier as RF
model = RF()
metric = mcc
```

Active learning subsampling can be directly called using the al_subsampling function
```
# run active learning
from active_subsampling import ALSubsampling
result = ALSubsampling.al_subsampling(model, bbbp, metric, 5 )
```

Results can be visualized by plotting the learning curve
```
# visualize learning curve (result[0] is all MCC values on validation set)
pl.plot(np.mean(result[0],axis=0))
pl.savefig("learning_curve.pdf")
pl.close()
```

Delta performance can be directly calculated from the resulting curves
```
# report deltaPerformance 
print(ALSubsampling.calc_deltaPerformances(result))
```

Subsampled data can be extracted by calling the subsample_data function
```
# extract AL subsample data
subsample = ALSubsampling.subsample_data(model, data, metric, 5)
```

