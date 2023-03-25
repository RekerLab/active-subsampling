# Active Subsampling
Using active learning for data curation.
![toc_graphic](https://user-images.githubusercontent.com/56095399/227727685-043c5ca6-27c8-4f7e-9565-fd146a1c31cf.png)


## Files
- **code.py** contains all code and functions to run and evaluate active learning subsampling
- **Example_workflow_for_AL_Subsampling.ipynb** contains an example notebook that runs BBBP but can be run out of the box on a local machine or on Google Colab to apply this technique to new datasets

## Quickstart

Datasets can be loaded from DeepChem
```
#load data
tasks, data, transformers = dc.molnet.load_bbbp(splitter=None)
bbbp = data[0]
```

Model and performance metric need to be initialized, we recommend random forest models and Matthews correlation coefficient
```
# initialize model and performance metric
model = RF()
metric = mcc
```

Active learning subsampling can be directly called using the al_subsampling function
```
# run active learning
result = al_subsampling(model, bbbp, metric, 5 )
```

Results can be visualized by plotting the learning curve
```
# visualize learning curve (result[0] is all MCC values on validation set)
pl.plot(np.mean(result[0],axis=0))
pl.savefig("learning_curve.pdf")
pl.close()
```

Delta Performance can be directly calculated from the resulting curves
```
# report deltaPerformance 
print(calc_deltaPerformances(result))
```

Subsampled data can be extracted by calling the subsample_data function
```
# extract AL subsample data
subsample = subsample_data(model, data, metric, 5)
```

## Dependencies
This code uses numpy, scipy, sklearn, numpy, deepchem, and matplotlib.

