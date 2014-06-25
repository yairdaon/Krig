SUFFIXES :=
%.py:

#        Compilers, linkers, compile options, link options, paths 

PYTHON  = python2.6
FFMPEG  = ffmpeg           
PLAY    = vlc

#          Lists of files


KERNEL_SOURCES   = kernel/*.py 
TEST_SOURCES     = Test_*.py

#           Stuff that every single program depends on

FOR_ALL   = Makefile 

ALL_SOURCES = $(KERNEL_SOURCES) $(TEST_SOURCES) Makefile MovieFrames
MOVIE_SOURCES = $(KERNEL_SOURCES) Makefile Test_Movie1D.py Test_Movie2D.py

#               Makin Movies!

movie1:
	clear
	$(PYTHON) Test_Movie1D.py  # the python script creates and plays the movie on its own

movie2:
	clear
	$(PYTHON) Test_Movie2D.py
	
short:
	clear
	$(PYTHON) Test_Sampler.py
	$(PYTHON) Test_Solver.py
	$(PYTHON) Test_Simple.py
	$(PYTHON) Test_Uniform.py
	$(PYTHON) Test_Reproducible.py
	$(PYTHON) Test_Plots.py

overnight:
	clear
	$(PYTHON) Test_Movie1D.py
	$(PYTHON) Test_Movie2D.py
	$(PYTHON) Test_Gaussian.py
	
tests:
	clear
	$(PYTHON) Test_Sampler.py
	$(PYTHON) Test_Solver.py
	$(PYTHON) Test_Simple.py
	$(PYTHON) Test_Uniform.py
	$(PYTHON) Test_Reproducible.py
	mkdir -p graphics
	$(PYTHON) Test_Plots.py
	$(PYTHON) Test_Movie1D.py
	$(PYTHON) Test_Movie2D.py
	$(PYTHON) Test_Gaussian.py
