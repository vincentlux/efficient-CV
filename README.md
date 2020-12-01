# Efficient-CV

## Train the cifar-100 baseline model

1. install dependency

```
pip install -r requirements.txt
```

2. train model
```
python efficient_cv/train.py --do_train --do_eval --n_gpu 1 --optim sgd --num_epochs 100 --batch_size 128 --scheduler multistep --lr 0.1 --model_name resnet18
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
* best baseline model (resnet18):
  * location: `snap/2020-12-01T17-04-17/best.pt`
  * eval accuracy: 0.7513

* resnet10 model (to compare with resnet10 model distilled from resnet 18):
  * location: `snap/2020-12-01T17-02-09/best.pt`
  * eval accuracy: 0.7344 
