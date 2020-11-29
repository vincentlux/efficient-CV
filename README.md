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

3. evaluate model
```
python efficient_cv/train.py --do_eval --test_model_path snap/2020-11-24T14-13-16/best.pt --n_gpu 1 --benchmarks baseline,quantization,fp16
```

## TODO
1. add evaluation model loading [done]
2. quantization [done]
3. distillation
4. pruning

## NOTES
* best baseline model:
  * location: `snap/2020-11-24T14-13-16/best.pt`
  * eval accuracy: 0.7617
