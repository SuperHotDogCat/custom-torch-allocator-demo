# Compiler
CXX = g++

# Compiler flags
CXXFLAGS ?= -fPIC -O3

# CUDA and system include directories
CUDA_PATH = /usr/local/cuda
INCLUDES = -I$(CUDA_PATH)/include

# Source files
SRCS = allocator.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target shared library
LIBRARY = allocator.so

# Rule for building CPU object files
%.o: %.cpp
	$(CXX) -shared $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link object files into a shared library
$(LIBRARY): $(OBJS)
	$(CXX) -shared $(OBJS) -L$(CUDA_PATH)/lib64 -lcudart -o $(LIBRARY)

# Clean up object files and the shared library
clean:
	rm -f $(OBJS) $(LIBRARY)