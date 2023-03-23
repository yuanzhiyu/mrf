# 使用说明
python run.py [dataset (数据集)] [frac_num (0至1的小数)] [cross_validation (1或者0)] [round_num] [fold_num]

## 对adult数据集进行0.5采样，进行交叉验证，1轮10-fold
python3 run.py adult 0.5 1 1 10

## 对wine数据集进行全量训练和测试，不进行交叉验证
python3 run.py wine 1 0 1 10

```
