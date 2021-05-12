CPP=g++
NVCC=/usr/local/cuda/bin/nvcc
CFLAGS=-Wall -O3 -std=c++17
INCLUDE=/usr/local/cuda/include ./internal .
LDFLAGS=
LIBS=
LIBDIRS=/usr/local/cuda/lib64
HEADERS=$(shell find . -name '*.hpp')
TARGET=potential


.PHONY: all clear clean purge

all: $(TARGET)



# Building Targets

$(TARGET): $(TARGET).cpp $(HEADERS) kernels.obj
	@echo Compiling and linking executable "$@" ...
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) -lcudart kernels.obj $< -o $@


kernels.obj: kernels.cu
	@echo Compiling CUDA kernels "$@" ...
	@$(NVCC) $(addprefix -I,$(INCLUDE)) -arch sm_70 --compile -cudart static $< -o $@



# Cleaning Stuff

clear:
	@echo Removing object files ...
	-@rm -f *.obj

clean: clear

purge: clear
	@echo Removing executable ...
	-@rm -f $(TARGETS)
