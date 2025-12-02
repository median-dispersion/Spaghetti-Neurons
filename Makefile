# Directories
INCLUDE_DIRECTORY = ./Include
SOURCE_DIRECTORY = ./Source
OBJECT_DIRECTORY = ./Build

# Compiler
COMPILER = g++
COMPILER_FLAGS = -Wall -Wextra -Werror -I$(INCLUDE_DIRECTORY)

# Libraries
LIBRARIES =

# Files
SOURCES = $(wildcard $(SOURCE_DIRECTORY)/*.cpp)
OBJECTS = $(patsubst $(SOURCE_DIRECTORY)/%.cpp, $(OBJECT_DIRECTORY)/%.o, $(SOURCES))
TARGET = ./main.out

# Default build rule
all: clean $(TARGET)

# Compile a fast heavily optimized build
fast: COMPILER_FLAGS += -O3 -march=native -flto -funroll-loops
fast: all

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(COMPILER) $(COMPILER_FLAGS) $^ $(LIBRARIES) -o $@

# Compile source files into object files
$(OBJECT_DIRECTORY)/%.o: $(SOURCE_DIRECTORY)/%.cpp | $(OBJECT_DIRECTORY)
	$(COMPILER) $(COMPILER_FLAGS) -c $< -o $@

# Create build directory if missing
$(OBJECT_DIRECTORY):
	mkdir -p $(OBJECT_DIRECTORY)

# Clean build artifacts
clean:
	rm -rf $(OBJECT_DIRECTORY) $(TARGET)