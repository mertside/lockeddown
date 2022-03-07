CC=nvcc

all: program1 program2

program1:
	$(CC) receiver.cu -o receiver.out

program2:
	$(CC) sender.cu -o sender.out

test:
	./receiver.out
	./receiver.out 1
	./sender.out
	./sender.out 1

clean:
	rm -rf *.out
