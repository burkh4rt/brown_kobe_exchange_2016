We can use pygments (https://pygments.org) for code highlighting in LaTeX.

pip install Pygments
pygmentize -f latex -l cython -o filter_cythonized.tex -O full,style=emacs filter_cythonized.pyx
