"""
Steps to display these files:
	1: Open Paraview
	2: Go to Sources (in the top tab) and select "Programmable Source"
	3: In the Script box, paste the code below
	4: Click Apply
	5: Adjust the color of the line and background as necessary
"""

numPts = 2
#Initialize the polydata output
pdo = self.GetPolyDataOutput()

#Create a vtkPoints instance
newPts = vtk.vtkPoints()

#Add all points to the vtkPoints
newPts.InsertPoint(0,0,0,0)
newPts.InsertPOint(1,0,0,999)

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
