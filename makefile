network/network.o: network/network.h network/network.cpp
	g++ -c network/network.cpp -o network/network.o
network/trainer.o: network/trainer.h network/trainer.cpp
	g++ -c network/trainer.cpp -o network/trainer.o
reader/reader.o: reader/reader.h reader/reader.cpp
	g++ -c reader/reader.cpp -o reader/reader.o
main.o: main.cpp
	g++ -c main.cpp
main: reader/reader.o network/network.o network/trainer.o main.o
	g++ -o main reader/reader.o network/network.o network/trainer.o main.o
run: main
	./main
clean:
	rm */*.o main
