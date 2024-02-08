# Training & Testing

```
CUDA_VISIBLE_DEVICES=0 python train.py --start_channel 4 --using_l2 2 --smth_labda 5.0 --lr 1e-4 --trainingset 4 --checkpoint 403 --iteration 201501

CUDA_VISIBLE_DEVICES=0 python infer_bilinear.py --start_channel 4 --using_l2 2 --smth_labda 5.0 --lr 1e-4 --trainingset 4 --checkpoint 403 --iteration 201501

python compute_dsc_jet_from_quantiResult.py
```
