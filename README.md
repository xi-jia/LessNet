# Training & Testing

We note that the pre-trained models are provided for easier reproduction of our reported results. 

```
CUDA_VISIBLE_DEVICES=0 python train.py --start_channel 4 --using_l2 2 --smth_labda 5.0 --lr 1e-4 --trainingset 4 --checkpoint 403 --iteration 201501

CUDA_VISIBLE_DEVICES=0 python infer_bilinear.py --start_channel 4 --using_l2 2 --smth_labda 5.0 --lr 1e-4 --trainingset 4 --checkpoint 403 --iteration 201501

python compute_dsc_jet_from_quantiResult.py
```

# Acknowledgment

We note that parts of the code are adopted from [IC-Net](https://github.com/zhangjun001/ICNet), [SYM-Net,](https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) (in chronological order of publication).

The preprocessed IXI data can be found in [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration).
