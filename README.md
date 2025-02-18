# Suppress and Rebalance: Towards Generalized Multi-Modal Face Anti-Spoofing
This is an implementation of the paper [Suppress and Rebalance: Towards Generalized Multi-Modal Face Anti-Spoofing](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_Suppress_and_Rebalance_Towards_Generalized_Multi-Modal_Face_Anti-Spoofing_CVPR_2024_paper.html).

## How to use
- `train.py`: Define cross entropy loss and SSP loss here.
- `test.py`: Perform inference per **model_save_epoch** or **model_save_step**.
- `balanceloader.py`: Dataloader for training.
- `intradataloader.py`: Dataloader for testing.

One can run the following command to train the model:
```console
python train.py --train_dataset [dataset] --weighted_factor [weight] 
```

To test the model and log the results in APCER, BPCER, ACER, AUC:
```console
python test.py --train_dataset [dataset] --test_dataset [dataset] --missing [dataset/none]
```
## TODO
Calculate prototypes for all source domains. Current implementation only trains on single source domain.