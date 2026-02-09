# Minimal makefile for Sphinx documentation
#
# Usage:  make html          (from docs/)
#         make clean         (remove build artifacts)

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help html clean doxygen

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

doxygen:
	@echo "Running Doxygen..."
	cd .. && doxygen docs/Doxyfile

html: doxygen
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "Build finished. HTML is in $(BUILDDIR)/html/"

clean:
	rm -rf $(BUILDDIR) _doxygen
