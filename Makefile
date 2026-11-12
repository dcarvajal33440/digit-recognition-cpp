CXX      ?= g++
WARN     := -std=c++17 -Wall -Wextra -Wpedantic
OPT      := -O3 -march=native -fopenmp
# "baseline" build: what you get with no tuning at all - no SIMD ISA flags, no threads.
BASE     := -O2 -Wno-unknown-pragmas

SRC := src
BLD := build
HDRS := $(wildcard $(SRC)/*.hpp)

all: $(BLD)/neuron $(BLD)/train_mlp $(BLD)/train_mlp_base $(BLD)/bench_gemm $(BLD)/test_gemm $(BLD)/test_grad

$(BLD):
        mkdir -p $(BLD)

$(BLD)/%: $(SRC)/%.cpp $(HDRS) | $(BLD)
        $(CXX) $(WARN) $(OPT) $< -o $@

$(BLD)/train_mlp_base: $(SRC)/train_mlp.cpp $(HDRS) | $(BLD)
        $(CXX) $(WARN) $(BASE) $< -o $@

test: $(BLD)/test_gemm $(BLD)/test_grad
        $(BLD)/test_gemm
        $(BLD)/test_grad

clean:
        rm -rf $(BLD)

.PHONY: all test clean
