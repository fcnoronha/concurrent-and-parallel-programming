#!/bin bash

MAXP=15			# Numero maximo de processos
MAXS=1000000000	# Numero maximo de pontos

PSTEP=1			# Incremento por loop da quantidade de processos (aditivo)
SSTEP=10		# Incremento por loop da quantidade de pontos (multiplicativo)

P=1				# Quantidade inicial de processos
S=1			 	# Quantidade inicial de pontos

ZERO=0

# Processos x Pontos

# Just to reset
./pi_process 0 0
rm resultado.csv

while [ $P -le $MAXP ]
do

	while [ $S -lt $MAXS ]
	do
		./pi_process $P $S | tr '\n' ';' >> resultado.csv
		#echo ""            | tr '\n' '|'  >> resultado.txt
		S=$(( $S * $SSTEP ))
	done
	S=1

	P=$(( $P + PSTEP ))
	echo "" >> resultado.csv
done

echo "" >> resultado.csv
