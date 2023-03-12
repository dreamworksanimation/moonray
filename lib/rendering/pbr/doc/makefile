#run pdflatex twice to be sure citations/references are correctly resolved
all: SamplingStrategy.pdf

clean:
	rm -f SamplingStrategy.aux SamplingStrategy.bbl SamplingStrategy.blg SamplingStrategy.log SamplingStrategy.pdf

SamplingStrategy.pdf: SamplingStrategy.aux SamplingStrategy.bbl *.tex
	echo "Make pdf"
	pdflatex SamplingStrategy.tex
	pdflatex SamplingStrategy.tex

SamplingStrategy.aux: SamplingStrategy.tex
	echo "Make aux"
	pdflatex SamplingStrategy.tex

SamplingStrategy.bbl: SamplingStrategy.bib SamplingStrategy.aux
	echo "Make bib"
	bibtex SamplingStrategy
