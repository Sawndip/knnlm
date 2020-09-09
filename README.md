# knnlm
k-NN based language model

**Aim**

This is a striking simple language model based on k-NN yet providing good results. The distance measure of two prefixes is their weighted hamming distance. Distances are linearly transformed feeding into a softmax re-weighter. The next char is sampled from softmax weighted next-char of all text prefixes.

**Install**

make

**Training**

./train -t COVID-19

**Generate sample**

./knnlm -t COVID-19 The coronavirus

**Benchmark language model**

./knnlm -t COVID-19 -b 1000

**Enjoy!**

By Wang Yi @ Fudan University

2020-09-09
