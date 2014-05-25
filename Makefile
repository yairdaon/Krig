SUFFIXES :=
%.py:

#        Compilers, linkers, compile options, link options, paths 

PYTHON  = python2.6
FFMPEG  = ffmpeg           

#          Lists of files


KERNEL_SOURCES   = kernel/*.py 
TEST_SOURCES     = Test_*.py

#           Stuff that every single program depends on

FOR_ALL   = Makefile 

ALL_SOURCES = $(KERNEL_SOURCES) $(TEST_SOURCES) Makefile MovieFrames
MOVIE_SOURCES = $(KERNEL_SOURCES) Makefile MovieFrames Test_Movie1.py

#               Makin Movies!

movie1:
	clear
	rm -f MovieFrames/*.png        #  delete any existing frames
	rm -f Movie.mpg                #  delete the existing movie, if there is one
	$(PYTHON) Test_Movie1.py
	$(FFMPEG) -i MovieFrames/Frame%d.png Movie.mpg


#	Makin tarball

tarball: $(All_SOURCES)  
	rm -f MovieFrames/*.png        #  delete any existing frames
	rm -f Movie.mpg                #  delete the existing movie, if there is one
	tar -cvf KrigingSampler.tar $(ALL_SOURCES) 

movieball: $(MOVIE_SOURCES)
	rm -f MovieFrames/*.png        #  delete any existing frames
	tar -cvf MovieMaker1.tar $(MOVIE_SOURCES) 
	