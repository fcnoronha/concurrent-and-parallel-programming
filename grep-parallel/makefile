pgrep : pgrep.o
	g++ -o pgrep pgrep.o -lpthread
pgrep.o : pgrep.cpp
	g++ -c pgrep.cpp -lpthread

clean:
	rm pgrep.o
	rm pgrep
