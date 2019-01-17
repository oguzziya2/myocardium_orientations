# this is the script for calculation of heart fiber orientations
# input: gradients.vtk, output: fibers.vtk

#Oguz Ziya Tikenogullari-Fall 2018-Stanford


import vtk 
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

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
filename=('gradients.vtu')
reader= vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(filename)
reader.Update()

model=vtk.vtkUnstructuredGrid()
model=reader.GetOutput()
ncell = model.GetNumberOfCells()
npts = model.GetNumberOfPoints()
points=vtk.vtkPoints()
for i in range(npts):
    coord = model.GetPoint(i)
    x,y,z = coord[:3]
    points.InsertPoint(i,x,y,z)
output_fibers=vtk.vtkUnstructuredGrid()
output_fibers.CopyStructure(model)
output_fibers.SetPoints(points)
# Read in temperature fields from the vtk
vtkphi_epi=vtk.vtkDoubleArray()
vtkphi_epi=model.GetPointData().GetArray("phi_epi")
vtkphi_rv =vtk.vtkDoubleArray()
vtkphi_rv =model.GetPointData().GetArray("phi_rv")
vtkphi_lv =vtk.vtkDoubleArray()
vtkphi_lv =model.GetPointData().GetArray("phi_lv")
vtkpsi    =vtk.vtkDoubleArray()
vtkpsi    =model.GetPointData().GetArray("psi")
# Read in temperature gradient fields from the vtk
#SET NUMBER OF COMPONENTS FOR GRADIENT ARRAY?????
vtkgrad_epi=vtk.vtkDoubleArray()
vtkgrad_epi=model.GetPointData().GetArray("grad_epi")
vtkgrad_rv =vtk.vtkDoubleArray()
vtkgrad_rv =model.GetPointData().GetArray("grad_rv")
vtkgrad_lv =vtk.vtkDoubleArray()
vtkgrad_lv =model.GetPointData().GetArray("grad_lv")
vtkgrad_psi=vtk.vtkDoubleArray()
vtkgrad_psi=model.GetPointData().GetArray("grad_psi")
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

gradphi_epi=np.zeros(3)
gradphi_rv =np.zeros(3)
gradphi_lv =np.zeros(3)
gradpsi    =np.zeros(3)

for i in xrange(0,npts):
    phi_epi=vtkphi_epi.GetTuple1(i)
    phi_rv =vtkphi_rv.GetTuple1(i)
    phi_lv =vtkphi_lv.GetTuple1(i)
    psi    =vtkpsi.GetTuple1(i)
    gradphi_epi=vtkgrad_epi.GetTuple3(i)
    gradphi_rv =vtkgrad_rv.GetTuple3(i)
    gradphi_lv =vtkgrad_lv.GetTuple3(i)
    gradpsi    =vtkgrad_psi.GetTuple3(i)
    gradphi_epi=np.array(gradphi_epi)
    gradphi_rv =np.array(gradphi_rv )
    gradphi_lv =np.array(gradphi_lv )
    gradpsi    =np.array(gradpsi    )
    denom = max(tol,phi_rv+phi_lv)
    Q_lv=orient(axis(gradpsi,-gradphi_lv),
                alpha_s(phi_rv/denom),
                beta_s(phi_rv/denom))
    Q_rv=orient(axis(gradpsi,gradphi_rv),
                alpha_s(phi_rv/denom),
                beta_s(phi_rv/denom))
    Q_endo=bislerp(Q_lv,Q_rv,phi_rv/denom)
    Q_epi= orient(axis(gradpsi,gradphi_epi),
                  alpha_w(phi_epi),
                  beta_w(phi_epi))
    fib_coor = bislerp(Q_endo,Q_epi,phi_epi)
    F=fib_coor[:,0]
    S=fib_coor[:,1]
    T=fib_coor[:,2]
    
    vtklong.SetTuple3(i,F[0],F[1],F[2])
    vtknorm.SetTuple3(i,S[0],S[1],S[2])
    vtktran.SetTuple3(i,T[0],T[1],T[2])

#vtklong.SetName('longitudinal')
#vtknorm.SetName('normal')
#vtktran.SetName('transverse')
#
#output_fibers.GetPointData().AddArray(vtklong)
#output_fibers.GetPointData().AddArray(vtknorm)
#output_fibers.GetPointData().AddArray(vtktran)

vtklong.SetName('FIB_DIR')

output_fibers.GetPointData().AddArray(vtklong)

fibers_out=vtk.vtkXMLUnstructuredGridWriter()
fibers_out.SetFileName('fibers.vtu')
fibers_out.SetInputData(output_fibers)
fibers_out.Update()
