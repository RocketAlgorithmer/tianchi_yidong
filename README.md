# tianchi_yidong
天池移动推荐算法
运行环境 python2.7

提取了约180维特征，包括：
用户特征，商品特征，类目特征，用户商品特征，用户类目特征，商品类目特征等

分别运行：
data_split.py
feature_extract.py
gen_data.py
xgb.py

forfast文件夹下run1，run2,run3的python文件是吧gen_data.py中训练集，验证集和测试集的分割分成3个程序，这样在多核电脑中可以分别运行三个文件以更快切分数据集


