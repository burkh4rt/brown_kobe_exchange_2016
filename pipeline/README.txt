1. learn model parameters with model_learn.py
2. compile particle filter with: python setup_filter_cythonized_with_neural_nets.py build_ext --inplace
3. run particle filter with: python -c "import filter_cythonized_with_nerual_nets"
4. data is now in file: filter_run.npz
5. install PyEVTK: https://bitbucket.org/pauloh/pyevtk/downloads
6. run npz_to_vtk.py
7. open result in paraview
