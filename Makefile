NVCC = nvcc
NVCC_FLAGS = -arch=sm_61 -O2
TARGET = test_parallel
SOURCE = test_parallel.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCE)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all run clean
