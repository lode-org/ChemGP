DOT_FILES := $(wildcard *.dot)
PNG_FILES := $(DOT_FILES:.dot=.png)
SVG_FILES := $(DOT_FILES:.dot=.svg)

DPI ?= 150

.PHONY: all clean

all: $(PNG_FILES) $(SVG_FILES)

%.png: %.dot
	dot -Tpng $< -o $@ -Gdpi=$(DPI)

%.svg: %.dot
	dot -Tsvg $< -o $@

clean:
	rm -f $(PNG_FILES) $(SVG_FILES)
