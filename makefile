all:	train knnlm
train: train.cpp
	g++ train.cpp -o train -O3 -Wall -static -s -fopenmp -march=native
knnlm:	knnlm.cpp
	g++ knnlm.cpp -o knnlm -O3 -Wall -static -s -fopenmp -march=native

