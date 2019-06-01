#!/bin/bash

MAXT=3200		# Numero maximo de threads
MAXB=960		# Numero maximo de blocos

TSTEP=320		# Incremento por loop da quantidade de threads (aditivo)
BSTEP=320		# Incremento por loop da quantidade de blocos (aditivo)

T=0				# Quantidade inicial de threads
B=0				# Quantidade inicial de blocos
C=1				# Number of simulations made so far

simulation_rep(){
	# Realiza 50 vezes a função e guarda resultados em um arquivo .dat

	echo -e "" > in.dat
	for i in {1..50}
	do
		./mbrot_test -2.0 1.5 1.0 -1.5 1000 1000 cpu $T $B >> in.dat
	done
}

echo -e "" > resultado.md
while [ $T -le $MAXT ]
do

	while [ $B -lt $MAXB ]
	do
		simulation_rep

		./calc | tr '\n' '|' >> resultado.md
		#echo ""            | tr '\n' '|'  >> resultado.txt
		B=$(( $B + $BSTEP ))
		mv in.dat in$C.dat
		let C=$C+1
	done
	S=B


	T=$(( $T + $TSTEP ))
	B=0
	echo "" >> resultado.md
done
