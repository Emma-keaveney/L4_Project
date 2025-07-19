import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.linalg import svd


def plot_circle_LS(coords, number_points):

    ''' 
    Find the resulting circle from a non-linear least squares fitting

    Args:
        coords (array): the coordinates of the particles to which a circle fitting will be performed
        number_points (int): the number of points on the resulting circle to be returned

    Returns:
        x_test (list): the x-coordinates of the circle of best fit
        y_test (list): the y-coordinates of the circle of best fit
        centre (array): the centre of the circle of best fit
    '''
    
    def fit_circle(x, y):

        '''
        Uses non-linear least squares fitting to fit the best fit circle to a set of 2D data

        Args:
            x,y: the x and y coordinates

        Returns:
            result: object containing the parameters of the best fit circle - the centre and radius
        '''

        x_m, y_m = np.mean(x), np.mean(y) 
        r_guess = np.sqrt((x - x_m)**2 + (y - y_m)**2).mean()
        initial_guess = [x_m, y_m, r_guess]
    
        def residuals(params):
            a, b, r = params
            return np.sqrt((x - a)**2 + (y - b)**2) - r
    
        result = least_squares(residuals, initial_guess)
    
        a, b, r = result.x
        return a, b, r

    a,b,r = fit_circle(coords[:,0],coords[:,1])
    centre = np.array([a,b])

    theta = np.linspace(0, 2*np.pi,number_points)
    x_test = [a+r*np.cos(angle) for angle in theta]
    y_test = [b+r*np.sin(angle) for angle in theta]

    return x_test, y_test, centre


def plot_ellipse_LS(coords, number_points):

    ''' 
    Find the resulting ellipse from a non-linear least squares fitting

    Args:
        coords (array): the coordinates of the particles to which a circle fitting will be performed
        number_points (int): the number of points on the resulting circle to be returned

    Returns:
        x_test (list): the x-coordinates of the ellipse of best fit
        y_test (list): the y-coordinates of the ellipse of best fit
        centre (array): the centre of the ellipse of best fit
    '''

    def fit_ellipse(x, y):

        '''
        Uses non-linear least squares fitting to fit the best fit ellipse to a set of 2D data

        Args:
            x,y: the x and y coordinates

        Returns:
            result: object containing the parameters of the best fit ellipse - the centre, semi-major axis and semi-minor axis
        '''

        x_m, y_m = np.mean(x), np.mean(y) 
        r_guess = np.sqrt((x - x_m)**2 + (y - y_m)**2).mean()
        initial_guess = [x_m, y_m, r_guess, r_guess]
    
        def residuals(params):
            h, k, a, b = params
            return (x - h)**2/(a**2) + (y - k)**2/(b**2) - 1
    
        result = least_squares(residuals, initial_guess)
    
        h, k, a, b = result.x
        return h, k, a,b

    h,k,a,b = fit_ellipse(coords[:,0], coords[:,1])
    centre = np.array([h,k])
    e = np.sqrt(1-(b/a)**2)

    def r(b,e,angle):
        r = b/np.sqrt(1-(e*np.cos(angle))**2)
        return r

    theta = np.linspace(0, 2*np.pi,number_points)
    x_test = [h+r(b,e,angle)*np.cos(angle) for angle in theta]
    y_test = [k+r(b,e,angle)*np.sin(angle) for angle in theta]
    

    return x_test, y_test, centre


def tilted_ellipse_LS(coords, number_points):

    ''' 
        Find the resulting ellipse from a non-linear least squares fitting

        Args:
            coords (array): the coordinates of the particles to which a circle fitting will be performed
            number_points (int): the number of points on the resulting circle to be returned

        Returns:
            x_test (list): the x-coordinates of the ellipse of best fit
            y_test (list): the y-coordinates of the ellipse of best fit
            centre (array): the centre of the ellipse of best fit
        '''
        
    def fit_ellipse(x, y):

        '''
        Uses non-linear least squares fitting to fit the best fit ellipse tilted at an angle
        to a set of 2D data

        Args:
            x,y: the x and y coordinates

        Returns:
            result: object containing the parameters of the best fit ellipse - centre, semi-major axis
            semi-minor axis, and angle tilted from x-axis.
        '''

        x_m, y_m = np.mean(x), np.mean(y) 
        r_guess = np.sqrt((x - x_m)**2 + (y - y_m)**2).mean()
        initial_guess = [x_m, y_m, r_guess, r_guess, 0]
    
        def residuals(params):
            h, k, a, b, theta = params
            return ((x - h)*np.cos(theta)+(y-k)*np.sin(theta))**2/(a**2) + ((x - h)*np.sin(theta)-(y-k)*np.cos(theta))**2/(b**2) - 1

        if abs(max(x)-min(x)) > abs(max(y)-min(y)):
            r_max = abs(max(x)-min(x))/1.5
        else:
            r_max = abs(max(y)-min(y))/1.5

        bounds = ([min(x), min(y), 0, 0, -5], [max(x), max(y), r_max, r_max, 5])
        result = least_squares(residuals, initial_guess, bounds = bounds)
    
        h, k, a, b, theta = result.x
        return h, k, a,b, theta

    h,k,a,b,theta = fit_ellipse(coords[:,0], coords[:,1])
    centre = np.array([h,k])
    
    if b>a:
        olda = a
        oldb = b
        b = olda
        a = oldb
        theta += np.pi/2
        
    e = np.sqrt(1-(b/a)**2)

    def r(b,e,angle):
        r = b/np.sqrt(1-(e*np.cos(angle))**2)
        return r

    angles = np.linspace(0, 2*np.pi,number_points)
    x_test = h + a * np.cos(angles) * np.cos(theta) - b * np.sin(angles) * np.sin(theta)
    y_test = k + a * np.cos(angles) * np.sin(theta) + b * np.sin(angles) * np.cos(theta)


    return x_test, y_test, centre


def isophotes(n, xc, yc, tolerance, magnitude):

    '''
    Finds the isophotes (contours of constant surface brightness) from 2D brightness data
    
    Args:
        n (array): the 2D surface brightness histogram
        xc (array): x-coordinates of histogram bins
        yc (array): y-coordinates of histogram bins
        tolerance (float): the tolerance required for a bin to belong to the isophote
        magnitude (float): the magnitude of bin to return to that isophote

    Returns:
        coords (array): the coordinates of all the histogram bins belonging to the isophote
    '''

    matches = np.isclose(n, magnitude, atol=tolerance)
    coords = np.column_stack((xc[matches], yc[matches]))

    return coords


def make_SB_map(coords, histogram_width, bin_no, magnitudes, band, sigma, f, time, show_plot = True):

    '''
    From raw star particle data created a 2D surface brightness map

    Args:
        coords (array): coordinates of all the star particles
        histogram_width (float): the span of coordinates the histogram should span
        bin_no (int): the number of bins in the histogram
        magnitudes (array): the magnitudes of all the star particles across all EM bands
        band (int): the index for the EM band data to be selected. Bands are [U, B, V, K, g, r, i, z']
        sigma (float): the smoothing kernel to be applied, in units of bins
        f (int): the scaling index of the resulting histogram:
            1: Mpc
            10**3: kpc
            10**6: pc
        time (float): the time in Gyr of the snapshot
        show_plot (bool): if show_plot = True, resulting histogram is shown, else histogram is not shown.

    Returns:
        n_smooth (array): the 2D smoothed surface brightness histogram data
        xc (array): the x-coordinates of the histogram bins
        yc (array): the y-coordinates of the histogram bins
    '''
    
    fluxes = 10**(-0.4*(magnitudes[:,band]))

    b = histogram_width
    n_flux, xedges, yedges = np.histogram2d(coords[:,1], coords[:,2], bins=(bin_no,bin_no), range=[[-b, b],[-b, b]], weights = fluxes)
    n_flux = gaussian_filter(n_flux, sigma=sigma)

    pix_size = ((b)*(10**6)/bin_no)**2
    n = 22.5-2.5*np.log10(n_flux/pix_size)
    n_smooth = n
    
    xbin = 0.5 * (xedges[:-1] + xedges[1:])
    ybin = 0.5 * (yedges[:-1] + yedges[1:])
    xc, yc = np.meshgrid(xbin, ybin)

    if show_plot == True:
        fig, ax = plt.subplots(figsize=(6, 5))

        if f == 1:
            ax.set_xlabel('x / Mpc', fontsize = 12)
            ax.set_ylabel('y / Mpc', fontsize = 12)

        if f == 10**3:
            ax.set_xlabel('x / kpc', fontsize = 12)
            ax.set_ylabel('y / kpc', fontsize = 12)
        
        if f == 10**6:
            ax.set_xlabel('x / pc', fontsize = 12)
            ax.set_ylabel('y / pc', fontsize = 12)
            
        contour = ax.contourf( xc*f, yc*f,n_smooth , cmap='binary')
        cbar = fig.colorbar(contour, ax=ax, label = r'Magnitude')

    plt.title('Time = {} Gyr'.format(np.round(time,2)), fontsize = 14)

    return n_smooth, xc, yc


def radial_profile_func(n, xc, yc, num_bins=100):

    '''
    Returns the average radial profile from a 2D histogram

    Args:
        n (array): the 2D histogram data
        xc (array): the x-coordinates of the histogram bins
        yc (array): the y-coordinates of the histogram bins
        num_bins (int): the number of radial bins to be taken

    Returns:
        r_bin_centers (array): radial coordinates
        radial_profile (array): radial average
    '''
    
    r = np.sqrt(xc**2 + yc**2).flatten()  # Flatten to 1D array
    intensity = n.flatten()  # Flatten intensity array

    # Define radial bins
    r_max = np.max(r)
    radial_bins = np.linspace(0, r_max, num_bins)
    
    # Compute the mean intensity in each radial bin
    radial_profile, bin_edges, _ = binned_statistic(r, intensity, statistic='mean', bins=radial_bins)
    
    # Compute bin centers
    r_bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    return r_bin_centers, radial_profile
