#this is the script that takes 4 heat diffusion results and
#produces layer IDs in layers.vtk file
#and  gradients information, that is necessary for fiber
#orientation calculations, in gradients.vtk file

#Oguz Ziya Tikenogullari-Fall 2018-Stanford

import vtk 
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

tol=1e-10
filenames=glob.glob("result*")
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

filenames = sorted(filenames, key=numericalSort)
print filenames[:]

#create the layers array that will be filled in the
#following for loop
tmp_reader= vtk.vtkXMLUnstructuredGridReader()
tmp_reader.SetFileName(filenames[0])
tmp_reader.Update()

tmp_read_out= tmp_reader.GetOutput()
ncell = tmp_read_out.GetNumberOfCells()
npts =  tmp_read_out.GetNumberOfPoints()
print 'number of cells:', ncell
print 'number of points:', npts
layers= np.zeros(npts)
fibers= np.zeros((ncell,3))
points=vtk.vtkPoints()

for i in range(npts):
    coord = tmp_read_out.GetPoint(i)
    x,y,z = coord[:3]
    points.InsertPoint(i,x,y,z)

output_layers=vtk.vtkUnstructuredGrid()
output_layers.CopyStructure(tmp_read_out)
output_layers.SetPoints(points)

output_gradients=vtk.vtkUnstructuredGrid()
output_gradients.CopyStructure(tmp_read_out)
output_gradients.SetPoints(points)

vtkphi_epi = vtk.vtkDoubleArray()
vtkphi_epi.SetNumberOfComponents(1)
vtkphi_epi.Allocate(npts,128)
vtkphi_epi.SetNumberOfTuples(npts)

vtkphi_rv = vtk.vtkDoubleArray()
vtkphi_rv.SetNumberOfComponents(1)
vtkphi_rv.Allocate(npts,128)
vtkphi_rv.SetNumberOfTuples(npts)

vtkphi_lv = vtk.vtkDoubleArray()
vtkphi_lv.SetNumberOfComponents(1)
vtkphi_lv.Allocate(npts,128)
vtkphi_lv.SetNumberOfTuples(npts)

vtkpsi = vtk.vtkDoubleArray()
vtkpsi.SetNumberOfComponents(1)
vtkpsi.Allocate(npts,128)
vtkpsi.SetNumberOfTuples(npts)

vtkphigrad_epi = vtk.vtkDoubleArray()
vtkphigrad_epi.SetNumberOfComponents(3)
#TUN T_epiHIS TO CELL DATA
vtkphigrad_epi.Allocate(npts,128)
vtkphigrad_epi.SetNumberOfTuples(npts)

vtkphigrad_rv = vtk.vtkDoubleArray()
vtkphigrad_rv.SetNumberOfComponents(3)
#TUN T_rvHIS TO CELL DATA
vtkphigrad_rv.Allocate(npts,128)
vtkphigrad_rv.SetNumberOfTuples(npts)

vtkphigrad_lv = vtk.vtkDoubleArray()
vtkphigrad_lv.SetNumberOfComponents(3)
#TUN T_lvHIS TO CELL DATA
vtkphigrad_lv.Allocate(npts,128)
vtkphigrad_lv.SetNumberOfTuples(npts)

vtkpsigrad = vtk.vtkDoubleArray()
vtkpsigrad.SetNumberOfComponents(3)
#TUpsi_psiTO CELL DATA
vtkpsigrad.Allocate(npts,128)
vtkpsigrad.SetNumberOfTuples(npts)

#for idx,fname in enumerate(filenames) :
fnameA=filenames[0]
fnameB=filenames[1]
fnameC=filenames[2]
fnameD=filenames[3]
print fnameA

readerA= vtk.vtkXMLUnstructuredGridReader()
readerB= vtk.vtkXMLUnstructuredGridReader()
readerC= vtk.vtkXMLUnstructuredGridReader()
readerD= vtk.vtkXMLUnstructuredGridReader()
readerA.SetFileName(fnameA)
readerB.SetFileName(fnameB)
readerC.SetFileName(fnameC)
readerD.SetFileName(fnameD)
readerA.Update()
readerB.Update()
readerC.Update()
readerD.Update()
read_outA= readerA.GetOutput()
read_outB= readerB.GetOutput()
read_outC= readerC.GetOutput()
read_outD= readerD.GetOutput()
tempA= read_outA.GetPointData().GetScalars('HS_Temperature')
tempB= read_outB.GetPointData().GetScalars('HS_Temperature')
tempC= read_outC.GetPointData().GetScalars('HS_Temperature')
tempD= read_outD.GetPointData().GetScalars('HS_Temperature')
tempA_num = vtk_to_numpy(tempA)
tempB_num = vtk_to_numpy(tempB)
tempC_num = vtk_to_numpy(tempC)
tempD_num = vtk_to_numpy(tempD)

# ----- layer id calculations at cell center
ctrA=np.zeros(ncell)
ctrB=np.zeros(ncell)
ctrC=np.zeros(ncell)
ctrD=np.zeros(ncell)

for icell in xrange(0,ncell):
    curcell=read_outA.GetCell(icell)
    pts_cell = curcell.GetPointIds()

    ptids=np.zeros(4,dtype=int)         
    ptids[0]=pts_cell.GetId(0)
    ptids[1]=pts_cell.GetId(1)
    ptids[2]=pts_cell.GetId(2)
    ptids[3]=pts_cell.GetId(3)
    
    ctrA[icell] = 0.25*( tempA_num[ptids[0]]
                        +tempA_num[ptids[1]]
                        +tempA_num[ptids[2]]
                        +tempA_num[ptids[3]])
    ctrB[icell] = 0.25*( tempB_num[ptids[0]]
                        +tempB_num[ptids[1]]
                        +tempB_num[ptids[2]]
                        +tempB_num[ptids[3]])
    ctrC[icell] = 0.25*( tempC_num[ptids[0]]
                        +tempC_num[ptids[1]]
                        +tempC_num[ptids[2]]
                        +tempC_num[ptids[3]])
    ctrD[icell] = 0.25*( tempD_num[ptids[0]]
                        +tempD_num[ptids[1]]
                        +tempD_num[ptids[2]]
                        +tempD_num[ptids[3]])

    
epi_cells=ctrA>=0.667
epi_id=epi_cells*7
myo_cells=(ctrA<0.667)&(ctrB>0.333)&(ctrC>0.333)
myo_id=myo_cells*4
end_cells=(ctrA<0.667)&((ctrB<0.333)|(ctrC<0.333))
end_id=end_cells*1

apex_cells=ctrD<0.90
apex_id=apex_cells*2
mid_cells=(ctrD>=0.90)&(ctrD<0.95)
mid_id=mid_cells*1
base_cells=ctrD>=0.95
base_id=base_cells*0

layers=end_id+myo_id+epi_id+ apex_id+mid_id+base_id
# ----- layer id calculations at cell center

# ----- temperature and gradients calculation at nodes
psi    =np.zeros(npts)
phi_epi=np.zeros(npts)
phi_rv=np.zeros(npts)
phi_lv=np.zeros(npts)

d0=np.zeros(3)
d1=np.zeros(3)
d2=np.zeros(3)
d3=np.zeros(3)

count = np.zeros(npts)
gradsum_epi = np.zeros((npts,3))
gradsum_rv = np.zeros((npts,3))
gradsum_lv = np.zeros((npts,3))
gradsum_psi = np.zeros((npts,3))

for icell in xrange(0,ncell):
    curcell=read_outA.GetCell(icell)
    pts_cell = curcell.GetPointIds()

    ptids=np.zeros(4,dtype=int)         
    ptids[0]=pts_cell.GetId(0)
    ptids[1]=pts_cell.GetId(1)
    ptids[2]=pts_cell.GetId(2)
    ptids[3]=pts_cell.GetId(3)

    for i in xrange(0,4):
        count[pts_cell.GetId(i)] = count[pts_cell.GetId(i)] + 1 

    p0 = read_outA.GetPoint(ptids[0])
    p1 = read_outA.GetPoint(ptids[1])
    p2 = read_outA.GetPoint(ptids[2])
    p3 = read_outA.GetPoint(ptids[3])

    ctr_x = 0.25*(p0[0]+p1[0]+p2[0]+p3[0])
    ctr_y = 0.25*(p0[1]+p1[1]+p2[1]+p3[1])    
    ctr_z = 0.25*(p0[2]+p1[2]+p2[2]+p3[2])

    #el-x  = el-x - ctrx    
    d0[0] = p0[0]-ctr_x
    d0[1] = p0[1]-ctr_y    
    d0[2] = p0[2]-ctr_z
    d1[0] = p1[0]-ctr_x
    d1[1] = p1[1]-ctr_y    
    d1[2] = p1[2]-ctr_z
    d2[0] = p2[0]-ctr_x
    d2[1] = p2[1]-ctr_y    
    d2[2] = p2[2]-ctr_z
    d3[0] = p3[0]-ctr_x
    d3[1] = p3[1]-ctr_y    
    d3[2] = p3[2]-ctr_z
    
    for i in xrange(0,4):
        phi_epi[ptids[i]] =  tempA.GetTuple1(ptids[i])
        phi_rv [ptids[i]] =-(tempB.GetTuple1(ptids[i])-1)
        phi_lv [ptids[i]] =-(tempC.GetTuple1(ptids[i])-1)
        psi    [ptids[i]] =  tempD.GetTuple1(ptids[i])

    D = np.matrix([[1.0,d0[0],d0[1],d0[2]],
                   [1.0,d1[0],d1[1],d1[2]],
                   [1.0,d2[0],d2[1],d2[2]],
                   [1.0,d3[0],d3[1],d3[2]]])
        
    dphi_epi=np.linalg.solve(D,[phi_epi[ptids[0]],
                                phi_epi[ptids[1]],
                                phi_epi[ptids[2]],
                                phi_epi[ptids[3]],])
    dphi_rv =np.linalg.solve(D,[phi_rv [ptids[0]],
                                phi_rv [ptids[1]],
                                phi_rv [ptids[2]],
                                phi_rv [ptids[3]],])
    dphi_lv =np.linalg.solve(D,[phi_lv [ptids[0]],
                                phi_lv [ptids[1]],
                                phi_lv [ptids[2]],
                                phi_lv [ptids[3]],])
    dpsi    =np.linalg.solve(D,[psi    [ptids[0]],
                                psi    [ptids[1]],
                                psi    [ptids[2]],
                                psi    [ptids[3]],])
    
    for i in xrange(0,4):
        gradsum_epi[ptids[i],0] = (gradsum_epi[ptids[i],0]
        + dphi_epi[1] )
        gradsum_epi[ptids[i],1] = (gradsum_epi[ptids[i],1]
        + dphi_epi[2] )
        gradsum_epi[ptids[i],2] = (gradsum_epi[ptids[i],2]
        + dphi_epi[3] )

    for i in xrange(0,4):
        gradsum_rv[ptids[i],0] = (gradsum_rv[ptids[i],0]
        + dphi_rv[1])
        gradsum_rv[ptids[i],1] = (gradsum_rv[ptids[i],1]
        + dphi_rv[2])
        gradsum_rv[ptids[i],2] = (gradsum_rv[ptids[i],2]
        + dphi_rv[3])       

    for i in xrange(0,4):
        gradsum_lv[ptids[i],0] = (gradsum_lv[ptids[i],0]
        + dphi_lv[1])     
        gradsum_lv[ptids[i],1] = (gradsum_lv[ptids[i],1]
        + dphi_lv[2] )                           
        gradsum_lv[ptids[i],2] = (gradsum_lv[ptids[i],2]
        + dphi_lv[3])

    for i in xrange(0,4):
        gradsum_psi[ptids[i],0] = (gradsum_psi[ptids[i],0]
        + dpsi[1])                                 
        gradsum_psi[ptids[i],1] = (gradsum_psi[ptids[i],1]
        + dpsi[2] )                                
        gradsum_psi[ptids[i],2] = (gradsum_psi[ptids[i],2]
        + dpsi[3]  )      

gradavg_epi=np.zeros((npts,3))
gradavg_rv =np.zeros((npts,3))
gradavg_lv =np.zeros((npts,3))
gradavg_psi=np.zeros((npts,3))

#average the sums of gradients at nodes
for i in xrange(0,npts):
    for j in xrange(0,3):
        gradavg_epi[i,j]=gradsum_epi[i,j]/count[i]
        gradavg_rv [i,j]=gradsum_rv [i,j]/count[i]
        gradavg_lv [i,j]=gradsum_lv [i,j]/count[i]
        gradavg_psi[i,j]=gradsum_psi[i,j]/count[i]

for i in xrange(0,npts):
    vtkphi_epi.SetTuple1(i,phi_epi[i])
    vtkphi_rv.SetTuple1(i,phi_rv[i])
    vtkphi_lv.SetTuple1(i,phi_lv[i])
    vtkpsi.SetTuple1(i,psi[i])
    vtkphigrad_epi.SetTuple3(i,gradavg_epi[i,0]
                             ,gradavg_epi[i,1],gradavg_epi[i,2])
    vtkphigrad_rv.SetTuple3(i,gradavg_rv[i,0]
                            ,gradavg_rv[i,1],gradavg_rv[i,2])
    vtkphigrad_lv.SetTuple3(i,gradavg_lv[i,0]
                            ,gradavg_lv[i,1],gradavg_lv[i,2])
    vtkpsigrad.SetTuple3(i,gradavg_psi[i,0]
                            ,gradavg_psi[i,1],gradavg_psi[i,2])

vtkphi_epi.SetName('phi_epi')
vtkphi_rv.SetName('phi_rv')
vtkphi_lv.SetName('phi_lv')
vtkpsi.SetName('psi')
vtkphigrad_epi.SetName('grad_epi')
vtkphigrad_rv.SetName('grad_rv')
vtkphigrad_lv.SetName('grad_lv')
vtkpsigrad.SetName('grad_psi')
#
layers_vtk =numpy_to_vtk(layers)
layers_vtk.SetName('layer ids')
output_layers.GetCellData().AddArray(layers_vtk)
#
output_gradients.GetPointData().AddArray(vtkphi_epi)
output_gradients.GetPointData().AddArray(vtkphi_rv)
output_gradients.GetPointData().AddArray(vtkphi_lv)
output_gradients.GetPointData().AddArray(vtkpsi)
output_gradients.GetPointData().AddArray(vtkphigrad_epi)
output_gradients.GetPointData().AddArray(vtkphigrad_rv)
output_gradients.GetPointData().AddArray(vtkphigrad_lv)
output_gradients.GetPointData().AddArray(vtkpsigrad)
#
layers_out=vtk.vtkXMLUnstructuredGridWriter()
layers_out.SetFileName('layers.vtu')
layers_out.SetInputData(output_layers)
layers_out.Update()
#
gradients_out=vtk.vtkXMLUnstructuredGridWriter()
gradients_out.SetFileName('gradients.vtu')
gradients_out.SetInputData(output_gradients)
gradients_out.Update()
#
#also create domain.dat file that keep layer ids.
domainfile=open("layers.dat", "w+")
for  icell in xrange(0,ncell):
    domainfile.write("%d\n" %(10+layers[icell]))
domainfile.close()


