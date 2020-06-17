all: html pdf

html:: presentation.cfg sycltrain.md
	landslide presentation.cfg

pdf::
	landslide sycltrain.md -d presentation.pdf

clean::
	rm -f sycltrain.html presentation.pdf

