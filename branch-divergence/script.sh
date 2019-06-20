#!/bin/bash

#inicializa arquivos de dados:
echo -n > data.dat
for i in {1..50}
do
	./divergence >> data.dat
done
