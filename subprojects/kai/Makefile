# Build the Kai QAOA grid viewer with Raylib.

CXX := c++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  RAYLIB_PREFIX := $(shell brew --prefix raylib 2>/dev/null)
  ifneq ($(RAYLIB_PREFIX),)
    CXXFLAGS += -I$(RAYLIB_PREFIX)/include
    LDFLAGS := -L$(RAYLIB_PREFIX)/lib -lraylib -framework IOKit -framework Cocoa -framework OpenGL
  else
    CXXFLAGS += -I/opt/homebrew/include -I/usr/local/include
    LDFLAGS := -L/opt/homebrew/lib -L/usr/local/lib -lraylib -framework IOKit -framework Cocoa -framework OpenGL
  endif
else
  CXXFLAGS += $(shell pkg-config --cflags raylib 2>/dev/null)
  LDFLAGS := $(shell pkg-config --libs raylib 2>/dev/null)
  ifeq ($(LDFLAGS),)
    LDFLAGS := -lraylib -lGL -lm -lpthread -ldl -lrt
  endif
endif

TARGET := qaoa_grid_viewer

.PHONY: all clean run

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) qaoa_grid_history.txt
