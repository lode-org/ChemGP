# GP Core

Core types and functions for Gaussian process regression with derivative observations.

## Types

```@docs
GPModel
TrainingData
```

## Data Management

```@docs
add_point!
npoints
normalize
```

## Training and Prediction

```@docs
train_model!
predict
predict_with_variance
```

## Covariance Assembly

```@docs
build_full_covariance
kernel_blocks
```
