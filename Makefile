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
MOVIE_SOURCES = $(KERNEL_SOURCES) Makefile MovieFrames Test_Movie1D.py

#               Makin Movies!

movie1:
	clear
	$(PYTHON) Test_Movie1D.py  # the python script creates and plays the movie on its own

tests:
	clear
	$(PYTHON) Test_Sampler.py
	$(PYTHON) Test_Solver.py
	$(PYTHON) Test_Simple.py
	$(PYTHON) Test_Uniform.py
	$(PYTHON) Test_Reproducible.py
	$(PYTHON) Test_MinMax.py

test_graphics:
	clear
	$(PYTHON) Test_Plots.py
	$(PYTHON) Test_Movie1D.py
	$(PYTHON) Test_Gaussian.py

#	Makin tarball

tarball: $(All_SOURCES)  
	rm -f MovieFrames/*.png        #  delete any ex$(FFMPEG) -i MovieFrames/Frame%d.png Movie.mpg
	$(PLAY) Movie.mpgisting frames
	rm -f Movie.mpg                #  delete the existing movie, if there is one
	tar -cvf KrigingSampler.tar $(ALL_SOURCES) 

kernelball: $(KERNEL_SOURCES)
	tar -cvf kernel.tar $(KERNEL_SOURCES)
	 
movieball: $(MOVIE_SOURCES)
	rm -f MovieFrames/*.png        #  delete any existing frames
	tar -cvf MovieMaker1.tar $(MOVIE_SOURCES) 
