#!/bin/bash

MAXT=960
MAXB=640

TSTEP=320
BSTEP=320

MODE=gpu
T=1
B=1
C=1


	# Realiza 50 vezes a função e guarda resultados em um arquvo.dat
simulation_rep() {
	#inicializa arquivos de dados:
	echo -n > data/"$MODE"\_"$B"\_"$T".dat
	for i in {1..50}
	do
		./mbrot_test -2.0 1.5 1.0 -1.5 1000 1000 "$MODE" "$T" "$B" >> data/"$MODE"\_"$B"\_"$T".dat
	done
}

#sequencial
MODE=cpu
#one thread
B=1
T=1
simulation_rep
echo -n > stats/"$MODE"\_"$B"\_"$T"
./calc data/"$MODE"_"$B"_"$T".dat >> stats/"$MODE"\_"$B"\_"$T".stats

#omp
for ((T=320; T<=MAXT; T=T+TSTEP))
do
	simulation_rep
	#inicializa arquivos de dados:
	echo -n > stats/"$MODE"\_"$B"\_"$T".stats
	./calc data/"$MODE"_"$B"_"$T".dat >> stats/"$MODE"\_"$B"\_"$T".stats
done

#gpu
MODE=gpu
#one thread
B=1
T=1
simulation_rep
echo -n > stats/"$MODE"\_"$B"\_"$T".stats
./calc data/"$MODE"_"$B"_"$T".dat >> stats/"$MODE"\_"$B"\_"$T".stats

for ((B=320; B<=MAXB; B=B+BSTEP))
do
	for ((T=320; T<=MAXT; T=T+TSTEP))
	do
		simulation_rep
		#inicializa arquivos de dados:
		echo -n > stats/"$MODE"\_"$B"\_"$T".stats
		./calc data/"$MODE"_"$B"_"$T".dat >> stats/"$MODE"\_"$B"\_"$T".stats
	done
done
