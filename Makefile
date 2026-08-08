CXX     = g++
PYTHIA ?= /home/tiaan/pythia8316

CXXFLAGS = -O2 $(shell $(PYTHIA)/bin/pythia8-config --cxxflags) -std=c++17
LDFLAGS  = $(shell $(PYTHIA)/bin/pythia8-config --ldflags --libs) -pthread

# --------------------------------------------------
# Directories
# --------------------------------------------------
SRC_DIR = src
BIN_DIR = bin

# Targets (executables)
TARGETS = chargedGenerationCorrelation
BINS    = $(addprefix $(BIN_DIR)/, $(TARGETS))

# Default target
all: $(BIN_DIR) $(BINS)

# Ensure bin directory exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build rule: src/foo.cc -> bin/foo
$(BIN_DIR)/%: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

# --------------------------------------------------
# Run helpers
# --------------------------------------------------
run: $(BIN_DIR)/chargedGenerationCorrelation
	./$(BIN_DIR)/chargedGenerationCorrelation $(ARGS)

# --------------------------------------------------
# Clean
# --------------------------------------------------
clean:
	rm -rf $(BIN_DIR)



