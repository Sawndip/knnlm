# knnlm
k-NN based language model without training

**Aim**

This is a striking simple language model based on k-NN yet providing good results. The distance measure of two prefixes is their weighted hamming distance. Distances are linearly transformed feeding into a softmax re-weighter. The next char is sampled from softmax weighted next-char of all text prefixes.

**Install**

make

**Generate sample from input.txt with prefixes (I have a dream that one day) with 32 mer**

./knnlm -t input.txt -k 32 I have a dream that one day

**Benchmark language model with BPC(Bits per Char)**

./knnlm -t input.txt -b 1000

**Enjoy!**

By Wang Yi @ Fudan University

2020-08-25
