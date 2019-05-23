pi_process : pi_process.o
	g++ -o pi_process pi_process.o -lm -lrt
pi_process.o : pi_process.cpp
	g++ -c pi_process.cpp -lm -lrt

clean:
	rm pi_process.o
	rm pi_process
