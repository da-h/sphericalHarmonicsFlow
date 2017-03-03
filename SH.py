#!/usr/bin/python3

import tensorflow as tf
import os
import numpy as np
from sympy import symbols
from sympy.utilities.lambdify import lambdify
import pickle
from math import sqrt,pi,factorial
import numbers

class SH:
    def emptyfunc(*args,**kwargs): pass
    def printfunc(*args,**kwargs): pass

    def setVerbose(b):
        if b:
            SH.printfunc = print
        else:
            SH.printfunc = SH.emptyfunc

    # ============ #
    # basis helpes #
    # ============ #

    # needed for shbasis
    _P_saved = []
    def _P(l,m,z):
        if l==0 and m==0:
            return 1
        if l==m:
            if len(SH._P_saved)==m:
                return SH._P_saved[m-1]
            new = (1-2*m)*SH._P(m-1,m-1,z)
            SH._P_saved.append(new)
            return new
        elif l==m+1:
            return (2*m+1)*z*SH._P(m,m,z)
        else:
            return ( (2*l-1)*z*SH._P(l-1,m,z) - (l+m-1)*SH._P(l-2,m,z) ) / (l-m)
    def _K(l,m):
        return sqrt((2*l+1)*factorial(l-abs(m))/(4*pi*factorial(l+abs(m))))

    _S_saved = []
    def _S(m, x, y):
        if m==0:
            return 0
        if len(SH._S_saved)==m:
            return SH._S_saved[m-1]
        new = x*SH._S(m-1,x,y)+y*SH._C(m-1,x,y)
        SH._S_saved.append(new)
        return new

    _C_saved = []
    def _C(m, x, y):
        if m==0:
            return 1
        if len(SH._C_saved)==m:
            return SH._C_saved[m-1]
        new = x*SH._C(m-1,x,y)-y*SH._S(m-1,x,y)
        SH._C_saved.append(new)
        return new


    # basis-generation
    # (used by sympy to squeeze calculations)
    x,y,z = symbols('x,y,z')
    _B_saved = {}
    def _basis_elem_sympy(l,m,x=None,y=None,z=None):
        if x==None: x=SH.x
        if y==None: y=SH.y
        if z==None: z=SH.z

        # check if already known
        key = str(l)+":"+str(m)
        if key in SH._B_saved:
            return SH._B_saved[key]

        # calculate
        if m>0:
            new = sqrt(2)*SH._K(l,m)*SH._C(m,x,y) * SH._P(l,m, z)
        elif m<0:
            new = sqrt(2)*SH._K(l,m)*SH._S(-m,x,y) * SH._P(l,-m, z )
        else:
            new = SH._K(l,0)*SH._P(l,0,z)

        # save to db
        if not isinstance(new, numbers.Number):
            new = new.simplify()
        SH._B_saved[key] = new
        return new


    # basis-generation
    # (used with tensorflow placeholders)
    def _basis_elem(l, m,x,y,z):
        calculation = SH._basis_elem_sympy(l,m)
        func = lambdify([SH.x,SH.y,SH.z],calculation)
        return func(x,y,z)


    # basis functions of spherical harmonics
    # (with projected coordinates)
    def _basis(x,y,z, numcoeffs, suffix=""):
        with tf.name_scope("SH_basis_all"):
            with tf.name_scope("SH_basis"+suffix):
                # -l ≤ m ≤ l
                sh_basis_list = [ SH._basis_elem(l, m,x,y,z) for l in range(numcoeffs) for m in range(numcoeffs) if l*(l+1)+m<numcoeffs**2 and m <= l and m!=0 and l!=0 ]
                sh_basis_list += [ SH._basis_elem(l,-m,x,y,z) for l in range(numcoeffs) for m in range(numcoeffs) if l*(l+1)+m<numcoeffs**2 and m <= l and m!=0 and l!=0 ]
                sh_basis_list = [ sh_basis_list[0]*0 + SH._basis_elem(0, 0,x,y,z) ] + sh_basis_list
                return tf.stack(sh_basis_list, axis=1, name="sh_basis"+suffix)


    # ------------------- #
    # Preparing the Basis #
    # ------------------- #
    sh_basis_file = os.path.join(os.path.dirname(__file__),"sh_basis")
    def saveBasis():
        if SH.sh_basis_file!=None:
            with open(SH.sh_basis_file+".pkl", "wb") as f:
                pickle.dump(SH._B_saved, f, pickle.HIGHEST_PROTOCOL)
    def loadBasis():
        if SH.sh_basis_file!=None and os.path.exists(SH.sh_basis_file+".pkl"):
            with open(SH.sh_basis_file+".pkl", "rb") as f:
                SH._B_saved = pickle.load(f)
    def prepareBasis(numcoeffs, verbose=False):
        SH.loadBasis()
        if verbose:
            numbasis = 1 + 2*np.sum([ 1 for l in range(numcoeffs) for m in range(numcoeffs) if l*(l+1)+m<numcoeffs**2 and m <= l and m!=0 and l!=0 ])
            i = 1
            print("Preparing Basis "+str(i)+" of "+str(numbasis)+" ( "+str(i/numbasis)*100+" % )",end="")
            for m in range(numcoeffs):
                for l in range(numcoeffs):
                    if l*(l+1)+m<numcoeffs**2 and m<=l and m!=0 and l!=0:
                        i+=1
                        print("\rPreparing Basis "+str(i)+" of "+str(numbasis)+" ( "+str(i/numbasis)+" % )",end="")
                        SH._basis_elem_sympy(l, m)
                        i+=1
                        print("\rPreparing Basis "+str(i)+" of "+str(numbasis)+" ( "+str(i/numbasis)+" % )",end="")
                        SH._basis_elem_sympy(l,-m)
        else:
            SH._basis_elem_sympy(0,0)
            [ (SH._basis_elem_sympy(l, m), SH._basis_elem_sympy(l,-m)) for l in range(numcoeffs) for m in range(numcoeffs) if l*(l+1)+m<numcoeffs**2 and m <= l and m!=0 and l!=0 ]
        SH.saveBasis()



    # define sh_approximation based on a given function
    def approximate(self, x, basis, f, name=""):

        with tf.name_scope("SH_approx_"+name):

            # solve linear equation (<b_i,b_j>)_ij c = (<b_i,f_i>)_i <=> Bc=v
            B = tf.matmul(tf.transpose(basis), basis)
            a = tf.reduce_sum(tf.expand_dims(basis,2)*tf.expand_dims(f,1), axis=0)
            S,U,V = tf.svd(B, full_matrices=True)
            Dinv = tf.diag(tf.where(S < self.inv_eps, S*0, 1/S))
            # alternative: Dinv = tf.diag(1/S)
            Binv = tf.matmul(tf.matmul(V,Dinv), tf.transpose(U))
            # alternative: Binv = tf.matrix_inverse(B)
            coeffs = tf.matmul(Binv, a)

            # ensure non-nans
            alternative_coeffs = coeffs*0 + 1

            return tf.where(tf.equal(tf.shape(x)[0],0), alternative_coeffs, coeffs, name="coeffs")




    def __init__(self, pts, center, channels, numcoeffs=10, numshells=0, radius=1, inv_eps=0.00001, autosave=True):
        self.numcoeffs = numcoeffs
        self.numshells = numshells
        self.inv_eps = inv_eps
        numchannels = channels.get_shape().as_list()[1]

        if autosave:
            SH.prepareBasis(numcoeffs)



        # =================================== #
        # Generate Spherical Harmonics Basis  #
        # =================================== #

        # get local coordinates for SH representation
        with tf.name_scope("local_coords"):
            self.x_local = tf.subtract(pts, center, name="x_local")

        # get spherical coordinates (on unit sphere!)
        with tf.name_scope("sphere_coords"):

            # project onto unit sphere & get distance
            self.x_norm = tf.sqrt(tf.reduce_sum(self.x_local*self.x_local, axis=1), name="x_norm")
            self.x_proj = tf.nn.l2_normalize(self.x_local,1, name="x_proj")

            # calc angles on sphere coordinates
            self.x_theta   = tf.acos( self.x_local[:,2] , name="x_theta")
            self.x_phi     = tf.asin( self.x_local[:,1] / tf.sin( self.x_theta ), name="x_phi")

        # get corresponding shell-numbers for each point
        with tf.name_scope("shell_subset"):
            self.x_shell_no = tf.to_int32(tf.floordiv( self.x_norm - radius/2 , radius ), name="x_shell_no")
            self.max_shell_no = tf.reduce_max(self.x_shell_no)




        # ------------ #
        # single shell #  - project all points onto one single shell
        # ------------ #

        if numshells==0:

            # build basis and approximate this subset
            SH.printfunc("building coeffs ... (general): ", end="")
            with tf.name_scope("sh_coeffs"):
                basis = SH._basis(self.x_proj[:,0],self.x_proj[:,1],self.x_proj[:,2],numcoeffs)
                coeffs = self.approximate(self.x_proj, basis, channels)
            self.coeffs = tf.stack([coeffs])
            SH.printfunc("done.")


            # use coefficients of approximation to redefine function
            self.numbasis = basis.get_shape().as_list()[1]
            self.coeffs_input = tf.placeholder(tf.float32, shape=(self.numbasis,numchannels), name="coeffs_input")
            with tf.name_scope("SH_approx_fn"):

                # remerge coeffs into function
                self.approx_func = tf.stack([tf.matmul(basis, self.coeffs_input, name="shell_fn")])


        # --------------- #
        # multiple shells #  - project all points on their next shell
        # --------------- #     (ignores points that are too far away)

        # project all shells seperately
        if numshells>0:
            coeffs = []
            basis = []

            # helper to get points, that lies on one of the shells
            # (only used for approx_func)
            with tf.name_scope("shell_subset"):
                self.pts_on_shell_ind = tf.reshape(tf.where(tf.less_equal(self.x_shell_no,numshells)), [-1], name="x_on_shell_ind")
                self.pts_on_shell = tf.gather(pts,self.pts_on_shell_ind, name="x_on_shell")

            SH.printfunc("building coeffs ... ", end="")
            self.xs_all = []
            for s in range(numshells):
                SH.printfunc("\rbuilding coeffs ... shell no."+str(s),end="")

                # subset x and channels
                with tf.name_scope("shell_subset"):
                    x_shell_ind = tf.reshape(tf.where(tf.equal(self.x_shell_no,s)), [-1], name="x_ind_S"+str(s))
                    xs_proj = tf.gather(self.x_proj,x_shell_ind, name="xs_S"+str(s))
                    xs_channels = tf.gather(channels,x_shell_ind, name="xs_channels_S"+str(s))

                    self.xs_all.append(xs_proj)

                # build basis and approximate this subset
                with tf.name_scope("sh_coeffs"):
                    s_basis = SH._basis(xs_proj[:,0],xs_proj[:,1],xs_proj[:,2],numcoeffs)
                    s_coeffs = self.approximate(xs_proj, s_basis, xs_channels)

                coeffs.append(s_coeffs)
                basis.append(s_basis)
            SH.printfunc("\rbuilding coeffs ... done.")

            # stack to coeff-cubus
            self.coeffs = tf.stack(coeffs)
            basis = tf.stack(basis)


            # use coefficients of approximation to redefine function
            self.numbasis = basis.get_shape().as_list()[2]
            self.coeffs_input = tf.placeholder(tf.float32, shape=(numshells,self.numbasis,numchannels), name="coeffs_input")
            with tf.name_scope("SH_approx_fn"):

                # remerge coeffs into function
                self.approx_func = tf.matmul(basis, coeffs, name="approx_fn_shells")


        if autosave:
            SH.saveBasis()
