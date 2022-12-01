import numpy as np
from math import log2
from collections import Counter

# 传入特征取第一个值的概率p，则取另一个值的概率为1-p
def entropy(y):
  counter = Counter(y) # 统计标签中每个值出现的次数
  res = 0.0
  for num in counter.values():
    p = num/len(y)
    res += -p * log2(p)
  return res

def split_by_val(X_i, y):
  counter = Counter(X_i)  # 一个特征可能有多个取值
  groups = []  # 返回一个字典列表
  print(X_i)
  print(y)
  for val in counter.keys():
    indexs = X_i[:] == val
    x_i = X_i[indexs]
    y_i = y[indexs]
    groups.append({'value': val, 'num': counter[val], 'y': y_i, 'indexs': indexs})
  return groups

def try_split(X, y):
  all_entropy = entropy(y)  # 先计算总的信息熵
  flag = -1  # 代表最终选用的特征的编号
  inc_entropy = 0
  fin_groups = []
  for i in range(len(X[0])):  # 依次计算各个特征的条件熵
    con_entropy = 0
    groups = split_by_val(X[:, i], y)
    # print(groups)
    for group in groups:  # 计算条件熵的公式
      con_entropy += group['num']/len(y) * entropy(group['y'])
    temp = all_entropy - con_entropy
    print(f"计算得到的条件增益为{temp}")
    if temp > inc_entropy:
      inc_entropy = temp
      fin_groups = groups
      flag = i
  
  return flag, fin_groups
      
X = np.array([[0, 1, 'T'], [0, 1, 'S'], [0, 1, 'S'], [0, 0, 'T'], [0, 1, 'T'],
     [0, 0, 'T'], [0, 0, 'D'], [1, 0, 'T'], [1, 0, 'T'], [1, 0, 'D'], 
     [1, 1, 'D'], [1, 1, 'T'], [1, 1, 'T'], [1, 0, 'S'], [1, 0, 'S']])
y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1])
flag, groups = try_split(X, y)
print(f"选择特征X{flag}作为根结点\n")

# 第二次划分
col = list(range(0, flag)) + list(range(flag+1, len(X[0])))
for group in groups:
  flag2, temp = try_split(X.take(col, 1)[group['indexs']], group['y'])
  if(flag2>=flag): flag2 += 1
  print(f"特征X{flag}取值为{group['value']}时，选择特征X{flag2}\n")
  