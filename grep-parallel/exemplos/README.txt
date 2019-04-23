
Alguns exemplos de execução para o EP1 de MAC0219/5742

Abaixo estão exemplos de execuções e possíveis outputs para o código
do ep1. Considere todos os comandos como sendo invocados a partir do
diretório pai de A.

===== exp1
$ ./ep1 8 [ab] A

<saida>
A/a.txt: 0
A/a.txt: 1
A/C/c.txt: 0
A/C/c.txt: 1
A/C/D/d.txt: 0
A/C/D/d.txt: 1
A/C/D/d.txt: 2
A/B/b2.txt: 0
A/B/b2.txt: 1


===== exp2
$ ./ep1 8 [a0]0 A

<saida>
A/C/c.txt: 0
A/C/D/d.txt: 0
A/B/b2.txt: 0
A/B/b.txt: 0


===== exp3
$ ./ep1 8 [0-9][a-b]0 A

<saida>
A/C/D/d.txt: 0
