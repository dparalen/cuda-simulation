SRCDIR=c/src
SRCFILE=num_sim_kernel.cu
BUILDDIR=c/build
OUTFILE=num_sim_kernel.cubin
DEBUG_FLAGS=-G -g -keep
FLAGS=-O2 -gencode=arch=compute_20,code=\"sm_21\" -gencode=arch=compute_20,code=\"sm_21\"
CPPFFLAGS="-DWITH_FILTERING"
CUBIN_FLAGS=-cubin
CC=nvcc

.PHONY: 
compile:
	$(CC) $(CPPFFLAGS) $(CUBIN_FLAGS) $(FLAGS) -o $(BUILDDIR)/$(OUTFILE) $(SRCDIR)/$(SRCFILE);

.PHONY: debug
debug:
	$(CC) $(CPPFFLAGS) $(DEBUG_FLAGS) $(CUBIN_FLAGS) $(FLAGS) -o $(BUILDDIR)/$(OUTFILE) $(SRCDIR)/$(SRCFILE);
