# TalkingData Click Fraud Detection

Binary classification pipeline to detect mobile ad click fraud using the [TalkingData AdTracking dataset](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection) (184M rows).

## Results
| Model | ROC-AUC |
|-------|---------|
| LightGBM (early stopping) | 0.9584 |
| XGBoost (hist) | 0.9563 |
| CatBoost | 0.9325 |
| Rank Ensemble (LGB + XGB) | **0.9586** |

## Approach

### Data
- 184M click records across 10 days of mobile ad traffic
- Binary target: `is_attributed` (1 = legitimate conversion, 0 = fraud proxy)
- Time-based train/test split (last day = test) to prevent leakage
- 5M row undersample: all 456K legitimate clicks + sampled fraud

### Feature Engineering (182 features)
- **Temporal**: hour, is_night
- **Count aggregations**: 8 group combinations (ip_app, ip_device, ip_os, etc.)
- **Rolling windows**: 5-minute and 1-hour click counts per ip, app, channel
- **Time deltas**: previous/next click intervals for ip_app and ip_device groups
- **Diversity metrics**: unique channels per IP, unique IPs per device
- **Conversion rates**: app, channel, and ip_app level attribution rates
- **Featuretools DFS**: 152 depth-2 automated features

### Models
- **LightGBM**: leaf-wise boosting, `colsample_bytree=0.4`, `is_unbalance=True`, early stopping
- **XGBoost**: histogram method, matched hyperparameters
- **CatBoost**: symmetric trees, ordered boosting, `class_weights` for imbalance handling
- **Ensemble**: rank averaging of LightGBM and XGBoost to handle calibration differences between models

## Usage
Run `report.ipynb` top to bottom. Requires `train.csv` and `test.csv` from the Kaggle competition placed in the project root.

## Limitations

- **Training data size**: Models trained on a 5M row undersample of the full 184M row dataset.
- **Fraud label proxy**: `is_attributed=0` is not a direct fraud label — it means the click did not result in an app download. Some legitimate clicks may never convert, introducing label noise.
- **Feature generation complexity**: 182 features including Featuretools depth-2 aggregations require significant preprocessing before scoring. In a production system, many of these features would need to be precomputed and served from a feature store rather than computed on the fly.
- **Training speed**: Full pipeline including feature engineering and model training runs in under 10 minutes on a consumer GPU, making iteration fast but also limiting the depth of hyperparameter search conducted.



*Built with assistance from Claude*

