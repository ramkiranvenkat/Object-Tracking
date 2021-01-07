all:
	nvcc -o pr_bs.o pr_bs.cu
	nvcc -o pr_mo.o pr_mo.cu
bs:
	nvcc -o pr_bs.o pr_bs.cu
mo:
	nvcc -o pr_mo.o pr_mo.cu
clean:
	rm -f pr_bs.o pr_mo.o *.csv *.tsv
	