"""
Steps to display these files:
	1: Change line 15 to contain the correct directory. Use absolute directories.
	2: Open Paraview
	3: Go to Sources (in the top tab) and select "Programmable Source"
	4: In the Script box, paste the code below
	5: Click Apply
	6: Adjust the color of the line and background as necessary
"""

import numpy as np

#Read the data
#TODO: Change the next line to a correct absolute directory!!!!
data_file = np.load('C:\\Users\\GuestUser\\Documents\\Project\\brown_kobe_exchange_2016\\pipeline\\filter_run.npz')
all_particles = data_file['all_particles']
all_weights = data_file['all_weights']
all_est = data_file['all_est']
all_true = data_file['all_true']

#Number of points
numPts = all_true.shape[0]

#Create an array of x,y,z values
all_true_t = np.zeros((all_true.shape[0], 3))
all_true_t[:,0:2] = all_true*100
for t in range(all_true.shape[0]):
	all_true_t[t,2] = t

xs = all_true_t[:,0].flatten()
ys = all_true_t[:,1].flatten()
zs = all_true_t[:,2].flatten()

#Initialize the polydata output
pdo = self.GetPolyDataOutput()

#Create a vtkPoints instance
newPts = vtk.vtkPoints()

#Add all points to the vtkPoints
for i in range(numPts):
	x = xs[i]
	y = ys[i]
	z = zs[i]
	newPts.InsertPoint(i,x,y,z)

#Save points to the output
pdo.SetPoints(newPts)

#Create a PolyLine instance to connect all lines
aPolyLine = vtk.vtkPolyLine()

#Set number of points in polyline
aPolyLine.GetPointIds().SetNumberOfIds(numPts)

#Set point ID's and connection order
for i in range(numPts):
	aPolyLine.GetPointIds().SetId(i,i)
	
#Allocate a single cell (one line)
pdo.Allocate(1,1)

#Feed lines to output
pdo.InsertNextCell(aPolyLine.GetCellType(), aPolyLine.GetPointIds())
