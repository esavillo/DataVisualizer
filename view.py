# Evan Savillo
# 04/11/16


import numpy
import math


class View:
    def __init__(self, vrp=numpy.matrix([0.5,0.5,1]), vpn=numpy.matrix([0,0,-1]),
                 vup=numpy.matrix([0,1,0]), u=numpy.matrix([-1,0,0]),
                 extent=[1.0,1.0,1.0], screen=None, offset=[50,50]):

        (self.vrp, self.vpn, self.vup, self.u,
         self.extent, self.offset) = (vrp, vpn, vup, u,
                                                   extent, offset)

        if screen is None:
            self.screen = [400,400]
            self.offset = [150,70]
        else:
            self.screen = screen

    # Resets the view
    def reset(self, screen=None):
        self.vrp = numpy.matrix([0.5,0.5,1])
        self.vpn = numpy.matrix([0,0,-1])
        self.vup = numpy.matrix([0,1,0])
        self.u = numpy.matrix([-1,0,0])
        self.extent = [1.0,1.0,1.0]
        if screen is None:
            self.screen = [400,400]
            self.offset = [150,70]
        else:
            self.screen = screen
            self.offset = [50,50]

    # Uses the current viewing parameters to return a view matrix
    def build(self):
        # basis for the view matrix
        vtm = numpy.identity(4, float)
        # translation matrix which moves the VRP to the origin
        t1 = numpy.matrix( [ [1, 0, 0, -self.vrp[0,0]],
                             [0, 1, 0, -self.vrp[0,1]],
                             [0, 0, 1, -self.vrp[0,2]],
                             [0, 0, 0, 1] ] )
        vtm = t1 * vtm

        # Calculate the view reference axes
        tu = numpy.cross(self.vup, self.vpn)
        tvup = numpy.cross(self.vpn, tu)
        tvpn = numpy.copy(self.vpn)

        # Normalize the view reference axes and copy orthonormal axes back to VRAs
        self.u, self.vup, self.vpn = self.normalize([tu,tvup,tvpn])

        # align the axes
        # Use normalized VRAs to generate the rotation matrix to align to the VRA
        r1 = numpy.matrix( [[tu[0,0], tu[0,1], tu[0,2], 0.0],
                            [tvup[0,0], tvup[0,1], tvup[0,2], 0.0],
                            [tvpn[0,0], tvpn[0,1], tvpn[0,2], 0.0],
                            [0.0, 0.0, 0.0, 1.0]] )
        vtm = r1 * vtm

        # Translate the lower left corner of the view space to the origin
        t2 = numpy.matrix( [[1, 0, 0, 0.5*self.extent[0]],
                            [0, 1, 0, 0.5*self.extent[1]],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]] )
        vtm = t2 * vtm

        # Use the extent and screen size values to scale to the screen
        s1 = numpy.matrix( [[-self.screen[0]/self.extent[0], 0, 0, 0],
                            [0, -self.screen[1]/self.extent[1], 0, 0],
                            [0, 0, 1.0/self.extent[2], 0],
                            [0, 0, 0, 1]] )
        vtm = s1 * vtm

        # Translate the lower left corner to the origin and add the view offset
        t3 = numpy.matrix( [[1, 0, 0, self.screen[0]+self.offset[0]],
                            [0, 1, 0, self.screen[1]+self.offset[1]],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]] )
        vtm = t3 * vtm

        return vtm

    def normalize(self, matrix_list):
        norm_list = []
        for matrix in matrix_list:
            l = math.sqrt( matrix[0,0]**2 + matrix[0,1]**2 + matrix[0,2]**2 )
            for i in range( len(matrix[0]) ):
                matrix[0, i] /= l
            norm_list += [ matrix.copy() ]

        return norm_list

    # Duplicates and returns the View object
    def clone(self):
        return View( self.vrp.copy(), self.vpn.copy(), self.vup.copy(), self.u.copy(),
                     list(self.extent), list(self.screen), list(self.offset) )

    # Rotations; takes in two angles, ang1e:VUP/yaw ang2e:U/pitch
    def rotateVRC(self, ang1e, ang2e):

        # transl matrix to move the point ( VRP + VPN * extent[Z] * 0.5 ) to the origin
        T = ( self.vrp + self.vpn * self.extent[2] * 0.5 )
        t1 = numpy.matrix( [[1, 0, 0, -T[0,0]],
                            [0, 1, 0, -T[0,1]],
                            [0, 0, 1, -T[0,2]],
                            [0, 0, 0, 1]] )
        # Make an axis alignment matrix Rxyz using u, vup and vpn.
        Rxyz = numpy.matrix( [[self.u[0,0], self.u[0,1], self.u[0,2], 0.0],
                              [self.vup[0,0], self.vup[0,1], self.vup[0,2], 0.0],
                              [self.vpn[0,0], self.vpn[0,1], self.vpn[0,2], 0.0],
                              [0.0, 0.0, 0.0, 1.0]] )
        # Make a rotation matrix about the Y axis by the VUP angle, put it in r1.
        r1 = numpy.matrix( [[math.cos(ang1e), 0, math.sin(ang1e), 0],
                            [0, 1, 0, 0],
                            [-math.sin(ang1e), 0, math.cos(ang1e), 0],
                            [0, 0, 0, 1]] )
        # Make a rotation matrix about the X axis by the U angle. Put it in r2.
        r2 = numpy.matrix( [[1, 0, 0, 0],
                            [0, math.cos(ang2e), -math.sin(ang2e), 0],
                            [0, math.sin(ang2e), math.cos(ang2e), 0],
                            [0, 0, 0, 1]] )
        # Make a translation matrix that has the opposite translation from step 1.
        t2 = numpy.matrix( [[1, 0, 0, T[0,0]],
                            [0, 1, 0, T[0,1]],
                            [0, 0, 1, T[0,2]],
                            [0, 0, 0, 1]] )
        # Make a numpy matrix where the VRP is on the first row,
        # with a 1 in the homogeneous coordinate, and u, vup, and vpn are the next three
        # rows, with a 0 in the homogeneous coordinate.
        tvrc = numpy.matrix( [[self.vrp[0,0], self.vrp[0,1], self.vrp[0,2], 1],
                              [self.u[0,0], self.u[0,1], self.u[0,2], 0],
                              [self.vup[0,0], self.vup[0,1], self.vup[0,2], 0],
                              [self.vpn[0,0], self.vpn[0,1], self.vpn[0,2], 0]] )
        # Execute the following:
        tvrc = (t2 * Rxyz.T * r2 * r1 * Rxyz * t1 * tvrc.T).T

        # Then copy the values from tvrc back into the VPR, U, VUP, and VPN fields and
        # normalize U, VUP, and VPN.
        self.vrp = numpy.resize(tvrc[0], (1,3))
        self.u, self.vup, self.vpn = self.normalize([numpy.resize(tvrc[1], (1,3)),
                                                     numpy.resize(tvrc[2], (1,3)),
                                                     numpy.resize(tvrc[3], (1,3))])


if __name__ == "__main__":
    #view = View(extent=[0.5,0.5,0.5])
    #print view.build()
    #view = View(extent=[1,1,1])
    #print view.build()
    #view = View(extent=[2,2,2])
    #print view.build()
    view = View()
    print view.vrp, view.u, view.vup, view.vpn
    view.rotateVRC(0, math.radians(-90))
    print view.vrp, view.u, view.vup, view.vpn
