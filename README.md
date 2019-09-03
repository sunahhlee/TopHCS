# TopHCS: TopH Count Sketch 

## Installation
Dependencies: `pytorch`, `numpy` and `CSVec` (from Sketched SGD paper, see description). Tested with `torch==1.0.1` and `numpy==1.15.3`, but this should work with a wide range of versions.

`git clone` the repository to your local machine, move to the directory containing `setup.py`, then run
```
pip install -e .
```
to install this package.

## Description

This package contains one main class, `TopHCS`, which computes the Count Sketch of input vectors, and can extract heavy hitters from a Count Sketch. To account for collisions, it keeps track of the top H elements (by magnitude).

Link to the Count Sketch paper -> http://www.mathcs.emory.edu/~cheung/Courses/584-StreamDB/Syllabus/papers/Frequency-count/FrequentStream.pdf
Link to the Sketched SGD paper -> http://arxiv.org/pdf/1903.04488.pdf 
