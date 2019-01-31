#this is the script that takes 4 heat diffusion results
#and produces layer IDs in layers.dat file
#and fiber orientations, in fibers.vtu file
#implementations from bayer et al 2012 and perotti et al 2015
#9 layer ids are between 11-19
#diffusion result files should be result_A.vtu,...,result_D.vtu
#and temperature field should have the name HS_Temperature
#this code is only tested with simvascular's heat cond. output
#layer ids look like
#                 base
#              1   4   7
# endocardium  2   5   8  epicardium
#              3   6   9
#                 apex


#Oguz Ziya Tikenogullari-Fall 2018-Stanford

import vtk 
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

#threshold values that seperate segments
epi_myo=0.667
mod_end=0.333
apex_mid=0.90
mid_base=0.95
#these values are valid for canine heart
#bayer et al 2012 implementation
alpha_end=( 40/180.0)*3.1416
alpha_epi=(-50/180.0)*3.1416
beta_end =(-65/180.0)*3.1416
beta_epi =( 25/180.0)*3.1416

def alpha_s(d):
    return alpha_end*(1-d)-alpha_end*d

def alpha_w(d):
    return alpha_end*(1-d)+ alpha_epi*d

def beta_s(d):
    return beta_end*(1-d)-beta_end*d

def beta_w(d):
    return beta_end*(1-d)+ beta_epi*d

def quat_mul(q0, q1):
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1

    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])

def axis (u,v):
    #u is grad(psi)
    #v is gradphi_something
    e1=u/np.linalg.norm(u) 
    #v is grad(phi)
    #in the bayer paper e0 is used instead if e1!!!!!
    e2=(v-np.dot(e1,v)*e1)/np.linalg.norm(v-np.dot(e1,v)*e1)
    e0=np.cross(e1,e2)
    Q=np.zeros((3,3))
    Q[:,0]=e0
    Q[:,1]=e1
    Q[:,2]=e2
    return Q

def orient(Q, alpha, beta):
    R1=np.array([[ np.cos(alpha),-np.sin(alpha), 0],
                 [ np.sin(alpha), np.cos(alpha), 0],
                 [ 0,             0,             1]])
    R2=np.array([[1, 0,            0           ],
                 [0, np.cos(beta), np.sin(beta)],
                 [0,-np.sin(beta), np.cos(beta)]])
    Qprime= np.matmul(np.matmul(Q,R1),R2)
    return Qprime

def rot2quat(R):
    """
    ROT2QUAT - Transform Rotation matrix into normalized quaternion.
    Usage: q = rot2quat(R)
    Input:
    R - 3-by-3 Rotation matrix
    Output:
    q - 4-by-1 quaternion, with form [w x y z], where w is the scalar term.
    """
    # By taking certain sums and differences of the elements
    # of R we can obtain all products of pairs a_i a_j with
    # i not equal to j. We then get the squares a_i^2 from
    # the diagonal of R.
    a2_a3 = (R[0,1] + R[1,0]) / 4
    a1_a4 = (R[1,0] - R[0,1]) / 4
    a1_a3 = (R[0,2] - R[2,0]) / 4
    a2_a4 = (R[0,2] + R[2,0]) / 4
    a3_a4 = (R[1,2] + R[2,1]) / 4
    a1_a2 = (R[2,1] - R[1,2]) / 4
    D = np.array([[+1, +1, +1, +1],
                  [+1, +1, -1, -1],
                  [+1, -1, +1, -1],
                  [+1, -1, -1, +1]]) * 0.25
    aa = np.dot(D, np.r_[np.sqrt(np.sum(R**2) / 3), np.diag(R)])
    # form 4 x 4 outer product a \otimes a:
    a_a = np.array([[aa[0], a1_a2, a1_a3, a1_a4],
                 [a1_a2, aa[1], a2_a3, a2_a4],
                 [a1_a3, a2_a3, aa[2], a3_a4],
                 [a1_a4, a2_a4, a3_a4, aa[3]]])
    # use rank-1 approximation to recover a, up to sign.
    U, S, V = np.linalg.svd(a_a)
    q = U[:, 0] 
    # q = np.dot(_math.sqrt(S[0]), U[:, 0])
    # Use this if you want unnormalized quaternions 
    return q

def quat2rot(q):
    """
    QUAT2ROT - Transform quaternion into rotation matrix
    Usage: R = quat2rot(q)
    Input:
    q - 4-by-1 quaternion, with form [w x y z], where w is the scalar term.
    Output:
    R - 3-by-3 Rotation matrix
    """    
    q = q / np.linalg.norm(q)
    w = q[0]; x = q[1];  y = q[2];  z = q[3]
    x2 = x*x;  y2 = y*y;  z2 = z*z;  w2 = w*w
    xy = 2*x*y;  xz = 2*x*z;  yz = 2*y*z
    wx = 2*w*x;  wy = 2*w*y;  wz = 2*w*z
    R = np.array([[w2+x2-y2-z2, xy-wz, xz+wy],
               [xy+wz, w2-x2+y2-z2, yz-wx],
               [xz-wy, yz+wx, w2-x2-y2+z2]])
    return R

def slerp(v0, v1, t):
    # >>> slerp([1,0,0,0],[0,0,0,1],np.arange(0,1,0.001))
    v0 = np.array(v0)
    v0 = v0/np.linalg.norm(v0)
    v1 = np.array(v1)
    v1 = v1/np.linalg.norm(v1)

    dot = np.sum(v0*v1)
    if (dot < 0.0):
        v1 = -v1
        dot = -dot
        
    DOT_THRESHOLD = 0.9995
    if (dot > DOT_THRESHOLD):
        result = v0+ t*(v1 - v0)
        result = result/np.linalg.norm(result)
        return result
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0*t
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * v0)+ (s1 * v1)

def bislerp(QA, QB, t):
    qA=rot2quat(QA)
    qB=rot2quat(QB)
    i=np.array([0, 1, 0, 0])
    j=np.array([0, 0, 1, 0])
    k=np.array([0, 0, 0, 1])
    qtrial=np.zeros((4,8))
    qtrial[:,0]=qA
    qtrial[:,1]=-qA
    qtrial[:,2]= quat_mul(i,qA)
    qtrial[:,3]=-quat_mul(i,qA)
    qtrial[:,4]= quat_mul(j,qA)
    qtrial[:,5]=-quat_mul(j,qA)
    qtrial[:,6]= quat_mul(k,qA)
    qtrial[:,7]=-quat_mul(k,qA)
    norms=np.zeros(8)
    for i in xrange(0,8):
        norms[i]=np.linalg.norm(np.dot(qtrial[:,i],qB))

    index=np.argmax(norms)
    qM=qtrial[:,index]

    return quat2rot(slerp(qM,qB,t))

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
points=vtk.vtkPoints()

for i in range(npts):
    coord = tmp_read_out.GetPoint(i)
    x,y,z = coord[:3]
    points.InsertPoint(i,x,y,z)

output_fibers=vtk.vtkUnstructuredGrid()
output_fibers.CopyStructure(tmp_read_out)
output_fibers.SetPoints(points)

#set output vectors: longitudinal, normal and transverse
vtklong = vtk.vtkDoubleArray()
vtklong.SetNumberOfComponents(3)
vtklong.Allocate(npts,128)
vtklong.SetNumberOfTuples(npts)
vtknorm = vtk.vtkDoubleArray()
vtknorm.SetNumberOfComponents(3)
vtknorm.Allocate(npts,128)
vtknorm.SetNumberOfTuples(npts)
vtktran = vtk.vtkDoubleArray()
vtktran.SetNumberOfComponents(3)
vtktran.Allocate(npts,128)
vtktran.SetNumberOfTuples(npts)

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
    
epi_cells=ctrA>=epi_myo
epi_id=epi_cells*7
myo_cells=(ctrA<epi_myo)&(ctrB>myo_end)&(ctrC>myo_end)
myo_id=myo_cells*4
end_cells=(ctrA<epi_myo)&((ctrB<myo_end)|(ctrC<myo_end))
end_id=end_cells*1

apex_cells=ctrD<apex_mid
apex_id=apex_cells*2
mid_cells=(ctrD>=apex_mid)&(ctrD<mid_base)
mid_id=mid_cells*1
base_cells=ctrD>=mid_base
base_id=base_cells*0

layers=end_id+myo_id+epi_id+ apex_id+mid_id+base_id
# ----- layer id calculations at cell center ends here

# ----- temperature and gradients calculation at nodes
psi    =np.zeros(npts)
phi_epi=np.zeros(npts)
phi_rv =np.zeros(npts)
phi_lv =np.zeros(npts)

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

gradphi_epi=np.zeros(3)
gradphi_rv =np.zeros(3)
gradphi_lv =np.zeros(3)
gradpsi    =np.zeros(3)


for i in xrange(0,npts):
    #average the sums of gradients at nodes
    for j in xrange(0,3):
        gradphi_epi[j]=gradsum_epi[i,j]/count[i]
        gradphi_rv [j]=gradsum_rv [i,j]/count[i]
        gradphi_lv [j]=gradsum_lv [i,j]/count[i]
        gradpsi    [j]=gradsum_psi[i,j]/count[i]

    denom = max(tol,phi_rv[i]+phi_lv[i])
    Q_lv=orient(axis(gradpsi,-gradphi_lv),
                alpha_s(phi_rv[i]/denom),
                beta_s(phi_rv[i]/denom))
    Q_rv=orient(axis(gradpsi,gradphi_rv),
                alpha_s(phi_rv[i]/denom),
                beta_s(phi_rv[i]/denom))
    Q_endo=bislerp(Q_lv,Q_rv,phi_rv[i]/denom)
    Q_epi= orient(axis(gradpsi,gradphi_epi),
                  alpha_w(phi_epi[i]),
                  beta_w(phi_epi[i]))
    fib_coor = bislerp(Q_endo,Q_epi,phi_epi[i])
    F=fib_coor[:,0]
    S=fib_coor[:,1]
    T=fib_coor[:,2]
    
    vtklong.SetTuple3(i,F[0],F[1],F[2])
    vtknorm.SetTuple3(i,S[0],S[1],S[2])
    vtktran.SetTuple3(i,T[0],T[1],T[2])    

# create domain.dat file that keep layer ids.
domainfile=open("layers.dat", "w+")
for  icell in xrange(0,ncell):
    domainfile.write("%d\n" %(10+layers[icell]))
domainfile.close()

# create fibers.vtu file that shows the fiber directions.
vtklong.SetName('FIB_DIR')
#vtknorm.SetName('FIB_DIR_2')
#check if simvascular understands the name FIB_DIR_2 

#there is only logitudinal orientations loaded rigth now
#uncomment the next line to load normal direction too
output_fibers.GetPointData().AddArray(vtklong)
#output_fibers.GetPointData().AddArray(vtknorm)

fibers_out=vtk.vtkXMLUnstructuredGridWriter()
fibers_out.SetFileName('fibers.vtu')
fibers_out.SetInputData(output_fibers)
fibers_out.Update()

