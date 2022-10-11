import typing as T
import math
import numpy as np
from utils import wrapToPi

# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1

class PoseController:
    """ Pose stabilization controller """
    def __init__(self, k1: float, k2: float, k3: float,
                 V_max: float = 0.5, om_max: float = 1) -> None:
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max

    def load_goal(self, x_g: float, y_g: float, th_g: float) -> None:
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g

    def compute_control(self, x: float, y: float, th: float, t: float) -> T.Tuple[float, float]:
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs:
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########
        x_diff = self.x_g-x
        y_diff = self.y_g-y
       
        rho = np.sqrt(x_diff**2 + y_diff**2)

        
        #th_al = 0
        #if y_diff < 0 and x_diff < 0:
        #    th_al = -np.pi/2
        #if x_diff < 0 and y_diff > 0:
        #    th_al = np.pi/2
        
        th_al = np.arctan2(y_diff,1e-8+x_diff)
        al = wrapToPi(th_al - th)
        delta = wrapToPi(th_al - self.th_g)
        if rho < RHO_THRES and al < ALPHA_THRES and delta < DELTA_THRES:
            V = 0
            om = 0 
        else: 
            V = self.k1*rho*np.cos(al)
            om = self.k2*al + self.k1*(np.sinc(al/np.pi)*np.cos(al))*(al+self.k3*delta)
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om
