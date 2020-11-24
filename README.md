# Efficient-CV

## Train the cifar-100 baseline model

1. install dependency

```
pip install -r requirements.txt
```

2. train model
```
python efficient_cv/train.py --do_train --do_eval --n_gpu 1 --optim sgd --num_epochs 200 --batch_size 128 --scheduler multistep
```

## TODO
1. add evaluation model loading
2. quantization
3. distillation
4. pruning
