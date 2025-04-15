# Training & Testing

We note that the pre-trained models are provided for easier reproduction of our reported results. 

```
CUDA_VISIBLE_DEVICES=0 python train.py --start_channel 8 --using_l2 2 --smth_labda 5.0 --lr 1e-4 --trainingset 4 --checkpoint 403 --iteration 201501

CUDA_VISIBLE_DEVICES=0 python infer_bilinear.py --start_channel 8 --using_l2 2 --smth_labda 5.0 --lr 1e-4 --trainingset 4 --checkpoint 403 --iteration 201501

python compute_dsc_jet_from_quantiResult.py
```
# Changes

In the original arXiv draft, I used the notations LessNet_4, 6, 8, 12, and 16 (LessNet_C) to denote different variants of LessNet, assuming a starting channel of 4C that is progressively reduced to 3C, 2C, and C.
However, I recently noticed that in the code implementation, the starting channels were actually set to 8×, 6×, 4×, and 2× a start channel.
To align the notation with the actual implementation, the model names should be updated to LessNet_8, 12, 16, 24, and 32. But the computational results remain unchanged.

For instance, the changes in TABLE IV will be:

| Was | Now | Unchanged | Unchanged |  Unchanged  |  Unchanged  |Unchanged  |
|-----|-----|:---------:|:---------:|:------:|:-----------:|:------------:|
|  C  |  C  | Parameter | Mult-Adds | Memory |    Dice     |       J<0    |
|  4  |  8  |   44,336  |    152    |  10.31 | 0.749±0.040 |  0.808±0.389 |
|  6  |  12 |   96,264  |    327    |  15.23 | 0.757±0.040 |  0.749±0.376 |
|  8  |  16 |  168,032  |    570    |  20.16 | 0.761±0.039 |  0.742±0.353 |
|  12 |  24 |  371,088  |    1250   |  30.00 | 0.766±0.039 |  0.852±0.392 |
|  16 |  32 |  653,504  |    2200   |  39.84 | 0.768±0.039 |  0.830±0.397 |

 **Only the naming was incorrect; the values reported are correct.**
I have updated the code accordingly to reflect these changes.

# Acknowledgment

We note that parts of the code are adopted from [IC-Net](https://github.com/zhangjun001/ICNet), [SYM-Net,](https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) (in chronological order of publication).

The preprocessed IXI data can be found in [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration).
