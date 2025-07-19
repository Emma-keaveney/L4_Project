import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import BSpline, splrep
import Auriga_Analysis as A


def use_splines(times, data,a, result):

    '''
    Uses cubic splines to model the motion of a galaxy

    Args:
        times (array): the time-steps at which the data is takes
        data (array): the data for which splines will be fit
        a (array): the cosmological scale factor corresponding to each time step
        result (string): the output required
            'position'
            'velocity'
            'acceleration'
            'peculiar_velocity'
            'peculiar_acceleration'
            'times'

    Returns:
        data_to_retrieve (array): the spline fitting of the result required

    '''

    splines = [CubicSpline(times, data[:,0]), CubicSpline(times, data[:,1]), CubicSpline(times, data[:,2])]
    position = [splines[0](times,0), splines[1](times,0),splines[2](times,0)]
    velocity = [splines[0](times,1), splines[1](times,1),splines[2](times,1)]
    acceleration = [splines[0](times,2), splines[1](times,2),splines[2](times,2)]

    peculiar_velocity = [[],[],[]]      #make velocity peculiar and in units of km/s
    peculiar_acceleration = [[],[],[]]    #make acceleration peculiar and in units of cm/s/yr
    
    for i in range(3):
        vel = velocity[i]
        accel = acceleration[i]
    
        for j in range(len(vel)):
            pec_vel = A.util.conversion_factors(0,vel[j]*a[j])
            pec_accel = A.util.conversion_factors(1,accel[j]*a[j])
    
            peculiar_velocity[i].append(pec_vel)
            peculiar_acceleration[i].append(pec_accel)

    all_values = [position, velocity, acceleration, peculiar_velocity, peculiar_acceleration, times]
    values = ['position', 'velocity', 'acceleration', 'peculiar_velocity', 'peculiar_acceleration', 'times']
    data_to_retrieve = all_values[values.index(result)]

    return data_to_retrieve
    

def moving_average(n,times,data, a, result):

    '''
    Finds the moving average of a set of data

    Args:
        n (int): the number of time-steps the moving average will be applied over
        times (array): the time-steps at which the data is takes
        data (array): the data over which the moving average will be applied
        a (array): the cosmological scale factor corresponding to each time step
        result (string): the output required
            'position'
            'velocity'
            'acceleration'
            'peculiar_velocity'
            'peculiar_acceleration'
            'times'

    Returns:
        data_to_retrieve (array): the moving average of the data required by result
    '''
    
    m = int((n-1)/2)
    print('Smoothing over', 5*n, 'Myr')

    time = times[m:-m]
    a = a[m:-m]
    
    smoothed_splines = [CubicSpline(time, A.util.moving_average(data[:,0],n)), CubicSpline(time, A.util.moving_average(data[:,1],n)), CubicSpline(time, A.util.moving_average(data[:,2],n))]
    smoothed_pos = [smoothed_splines[0](time,0), smoothed_splines[1](time,0),smoothed_splines[2](time,0)]
    smoothed_velocity = [smoothed_splines[0](time,1), smoothed_splines[1](time,1),smoothed_splines[2](time,1)]
    smoothed_acceleration = [smoothed_splines[0](time,2), smoothed_splines[1](time,2),smoothed_splines[2](time,2)]

    peculiar_velocity = [[],[],[]]      #make velocity peculiar and in units of km/s
    peculiar_acceleration = [[],[],[]]     #make acceleration peculiar and in units of cm/s/yr
    
    for i in range(3):
        vel = smoothed_velocity[i]
        accel = smoothed_acceleration[i]
    
        for j in range(len(vel)):
            pec_vel = A.util.conversion_factors(0,vel[j]*a[j])
            pec_accel = A.util.conversion_factors(1,accel[j]*a[j])
    
            peculiar_velocity[i].append(pec_vel)
            peculiar_acceleration[i].append(pec_accel)

    all_values = [smoothed_pos, smoothed_velocity, smoothed_acceleration, peculiar_velocity, peculiar_acceleration, time]
    values = ['position', 'velocity', 'acceleration', 'peculiar_velocity', 'peculiar_acceleration', 'times']
    data_to_retrieve = all_values[values.index(result)]

    return data_to_retrieve


def b_splines(degree, k, data, times,a, result): 

    '''
    Models data using b-splines
    
    Args:
        degree (int): the degree of b-spline
        k (int): the number of time-steps each b-spline spans
        data (array): the data over which the b-splines will be fit
        times (array): the time-steps at which the data is taken
        a (array): the cosmological scale factor corresponding to each time step
        result (string): the output required
            'velocity'
            'acceleration'
            'peculiar_velocity'
            'peculiar_acceleration'
            'times'

    Returns:
        data_to_retrieve (array): the b-spline fitting to the data required by result
    '''

    n_interior_knots = int(len(data)/k)
    print('Number of interior knots = ', n_interior_knots)

    qs = np.linspace(0, 1, n_interior_knots+2)[1:-1]
    knots = np.quantile(times, qs)

    velocity = []
    acceleration = []
    peculiar_velocity = []
    peculiar_acceleration = []

    for i in range(3):
        tck = splrep(times, data[:,i], t=knots, k = degree)
        bspl = BSpline(*tck)(times,1)
        velocity.append(bspl)
        subspline = []
        for j in range(len(bspl)):
            spline = A.util.conversion_factors(0,bspl[j]*a[j])
            subspline.append(spline)
            
        peculiar_velocity.append(subspline)
    
        tck = splrep(times, data[:,i], t=knots, k = degree)
        bspl = BSpline(*tck)(times,2)
        acceleration.append(bspl)
        subspline = []
        for j in range(len(bspl)):
            spline = A.util.conversion_factors(1,bspl[j]*a[j])
            subspline.append(spline)
            
        peculiar_acceleration.append(subspline)

    all_values = [velocity, acceleration, peculiar_velocity, peculiar_acceleration, times]
    values = ['velocity', 'acceleration', 'peculiar_velocity', 'peculiar_acceleration', 'times']
    data_to_retrieve = all_values[values.index(result)]

    return data_to_retrieve

def use_1Dsplines(times, data,a, result):

    '''
    Models 1D data using splines
    
    Args:
        times (array): the time-steps at which the data is taken
        data (array): the data over which the b-splines will be fit
        a (array): the cosmological scale factor corresponding to each time step
        result (string): the output required
            'position'
            'velocity'
            'acceleration'
            'peculiar_velocity'
            'peculiar_acceleration'
            'times'

    Returns:
        data_to_retrieve (array): the 1D spline fitting to the data required by result
    '''

    splines = CubicSpline(times, data)
    position = splines(times,0)
    velocity = splines(times,1)
    acceleration = splines(times,2)

    peculiar_velocity = []      #make velocity peculiar and in units of km/s
    peculiar_acceleration = []    #make acceleration peculiar and in units of cm/s/yr
    
    for j in range(len(velocity)):
        pec_vel = A.util.conversion_factors(0,velocity[j]*a[j])
        pec_accel = A.util.conversion_factors(1,acceleration[j]*a[j])

        peculiar_velocity.append(pec_vel)
        peculiar_acceleration.append(pec_accel)

    all_values = [position, velocity, acceleration, peculiar_velocity, peculiar_acceleration, times]
    values = ['position', 'velocity', 'acceleration', 'peculiar_velocity', 'peculiar_acceleration', 'times']
    data_to_retrieve = all_values[values.index(result)]

    return data_to_retrieve

def b_1Dsplines(degree, k, data, times,a, result):  

    '''
    Models 1D data using b-splines
    
    Args:
        degree (int): the degree of b-spline
        k (int): the number of time-steps each b-spline spans
        data (array): the data over which the b-splines will be fit
        times (array): the time-steps at which the data is taken
        a (array): the cosmological scale factor corresponding to each time step
        result (string): the output required
            'velocity'
            'acceleration'
            'peculiar_velocity'
            'peculiar_acceleration'
            'times'

    Returns:
        data_to_retrieve (array): the 1D b-spline fitting to the data required by result
    '''

    n_interior_knots = int(len(data)/k)
    print('Number of interior knots = ', n_interior_knots)

    qs = np.linspace(0, 1, n_interior_knots+2)[1:-1]
    knots = np.quantile(times, qs)

    tck = splrep(times, data, t=knots, k = degree)
    bspl = BSpline(*tck)(times,0)
    position = bspl
    
    tck = splrep(times, data, t=knots, k = degree)
    bspl = BSpline(*tck)(times,1)
    velocity = bspl
    
    peculiar_velocity = []
    for j in range(len(bspl)):
        spline = A.util.conversion_factors(0,bspl[j]*a[j])
        peculiar_velocity.append(spline)
        

    tck = splrep(times, data, t=knots, k = degree)
    bspl = BSpline(*tck)(times,2)
    acceleration = bspl
    peculiar_acceleration = []
    for j in range(len(bspl)):
        spline = A.util.conversion_factors(1,bspl[j]*a[j])
        peculiar_acceleration.append(spline)

    all_values = [position, velocity, acceleration, peculiar_velocity, peculiar_acceleration, times]
    values = ['position','velocity', 'acceleration', 'peculiar_velocity', 'peculiar_acceleration', 'times']
    data_to_retrieve = all_values[values.index(result)]

    return data_to_retrieve


#EXAMPLE

# unsmoothed_subfind_velocity = A.motion.use_splines(times, subfind,a, 'peculiar_velocity')
# ma_subfind_velocity = A.motion.moving_average(20,times,subfind, a, 'peculiar_velocity')
# ma_time = A.motion.moving_average(20,times,subfind, a, 'times')                               #as MA changes the time interval, we need to return a new time. Use times to plot unsmoothed + bspline
# bspline_subfind_velocity = A.motion.b_splines(4, 20, subfind, times,a, 'peculiar_velocity')
