main.o: main.cpp
	g++ -c main.cpp
network.o: network/network.cpp network/network.h
	g++ -c network/network.cpp
trainer.o: network/trainer.cpp network/trainer.h
	g++ -c network/trainer.cpp
reader.o: reader/reader.cpp reader/reader.h
	g++ -c reader/reader.cpp
main: main.o network.o trainer.o reader.o
	g++ -o main main.o network.o trainer.o reader.o
run: main
	./main
