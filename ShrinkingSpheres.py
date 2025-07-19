import numpy as np
import h5py

def particles_within_sphere(centre, radius, pos, masses):

    '''
    Filters the particles so only particles within a sphere are kept.

    Args:
        centre (list): the coordinates of the centre of the sphere
        radius (float): the radius of the sphere
        pos (array): the coordinates of all the particles
        masses (array): the masses of all the particles

    Returns:
        filtered_pos (array): the coordinates of particles contained within the sphere
        filtered_mass (array): the masses of the particles contained within the sphere

    '''

    distances = np.linalg.norm(pos - centre, axis=1)  
    mask = distances <= radius 
    
    filtered_pos = pos[mask]
    filtered_mass = masses[mask]
    return filtered_pos, filtered_mass


def mask(centre, radius, pos, details):

    '''
    Filters the particles so only particles within a sphere are kept.

    Args:
        centre (list): the coordinates of the centre of the sphere
        radius (float): the radius of the sphere
        pos (array): the coordinates of all the particles
        details (list): list containing certain attributes of all the particles. E.g. may be [masses, velocities] the arrays of masses and velocities of all the particles

    Returns:
        filtered_pos (array): the coordinates of particles contained within the sphere
        filtered_detail (list): the attributes described in details of the particles contained within the sphere.
        
    '''

    distances = np.linalg.norm(pos - centre, axis=1)  
    mask = distances <= radius 

    filtered_details = [detail[mask] for detail in details]
    filtered_pos = pos[mask]

    return filtered_pos, filtered_details

def mask_annulus(centre, radius1, radius2, pos,details):

    '''
    Filters the particles so only particles within an annulus are kept.

    Args:
        centre (list): the coordinates of the centre of the annulus
        radius1 (float): the inner radius of the annulus
        radius2 (float): the outer radius of the annulus
        pos (array): the coordinates of all the particles
        details (list): list containing certain attributes of all the particles. E.g. may be [masses, velocities] the arrays of masses and velocities of all the particles

    Returns:
        filtered_pos (array): the coordinates of particles contained within the sphere
        filtered_detail (list): the attributes described in details of the particles contained within the sphere.
        
    '''

    distances = np.linalg.norm(pos - centre, axis=1)  
    mask = (radius1 <= distances) & (distances <= radius2)

    filtered_details = [detail[mask] for detail in details]
    filtered_pos = pos[mask]

    return filtered_pos, filtered_details


def barycentre(positions, masses):

    '''
    Returns the barycentre (centre of mass) of a set of particles

    Args:
        positions (array): the coordinates of the particles
        masses (array): the masses of the particles

    Returns:
        centres (array): the barycentre of the particles

    '''

    centres = []
    
    for i in range(len(positions[0])): 
        cm = np.sum(positions[:,i]*masses)/sum(masses)
        centres.append(cm)
    return np.array(centres)


def shrinking_spheres(pos, masses,radius, r,no_part, centre, mass):

    '''
    This function is used to find the centre of mass distribution of a set of particles. Works by 
    iterating over progressively shrinking spheres, with each sphere centred at the barycentre of the
    previous sphere, until a given number of particles is reached. 

    Args:
        pos (array): the positions of all the particles
        masses (array): the masses of all the particles
        radius (float): the starting radius of the sphere
        r (float): the rate at which sphere shrinks, given as a fraction of the radius
        no_part (int): the number of particles contained within the sphere at which the algorithm stops
        centre (array): the initial centre of the sphere
        mass (bool): If mass = True, spheres are centred on barycentre. If mass = False, spheres are centred on mean position.

    Returns:
        pos (array): the coordinates of all the particles within the final sphere
        masses (array): the masses of all the particles within the final sphere
        no_iterations (int): the number of iterations required for no_part to be reached
        radius (float): the radius of the final sphere
        centre(list): the centre of the final sphere

    '''

    num_particles = len(pos)
    no_iterations = 0
    
    while num_particles > no_part:
        
        pos,masses = particles_within_sphere(centre, radius, pos,masses)

        num_particles = len(pos)

        if mass == True:
            centre = barycentre(pos,masses)
        elif mass == False:
            centre = np.mean(pos)
        radius = radius*r
    
        no_iterations +=1
        
    return(pos,masses, no_iterations, radius,centre)


def shrinking_circles(pos, masses,radius, r,no_part, centre):

    '''
    This function is used to find the 2D centre of mass distribution of a set of particles. Works by 
    iterating over progressively shrinking circles, with each circle centred at the barycentre of the
    previous circle, until a given number of particles is reached. 

    Args:
        pos (array): the positions of all the particles
        masses (array): the masses of all the particles
        radius (float): the starting radius of the circle
        r (float): the rate at which circle shrinks, given as a fraction of the radius
        no_part (int): the number of particles contained within the circle at which the algorithm stops
        centre (array): the initial centre of the circle
        mass (bool): If mass = True, circles are centred on barycentre. If mass = False, circles are centred on mean position.

    Returns:
        pos (array): the coordinates of all the particles within the final circle
        masses (array): the masses of all the particles within the final circle
        no_iterations (int): the number of iterations required for no_part to be reached
        radius (float): the radius of the final circle
        centre(list): the centre of the final circle

    '''

    num_particles = len(pos)
    no_iterations = 0
    
    while num_particles > no_part:
        
        pos,masses = particles_within_sphere(centre, radius, pos,masses)
        num_particles = len(pos)

        centres = []
    
        for i in range(2):    #used to be for i in range(3) in case shrinking spheres in 3D stops working
            cm = np.dot(pos[:,i],masses)/sum(masses)
            centres.append(cm[0])
            
        centre = np.array(centres)
        radius = radius*r
    
        no_iterations +=1
        
    return(pos,masses, no_iterations, radius,centre)


def barycentre_shrinking_spheres(positions,masses):

    '''
    Returns the barycentre of a set of particles

    Args:
        positions (array): coordinates of the particles
        masses (array): the masses of the particles
    '''

    centre = barycentre(positions,masses)
    return centre
    

def median_shrinking_spheres(positions):

    '''
    Returns the median centre of a set of particles

    Args:
        positions (array): coordinates of the particles

    Returns:
        [x,y,z] (array): the median 3D position
    '''

    x = np.median(positions[:,0])
    y = np.median(positions[:,1])
    z = np.median(positions[:,2])

    return np.array([x,y,z])


def rotation_matrix(a,b,c):

    '''
    Gives the rotation matrix for rotations around all 3 axes

    Args:
        a (float): angle around which x axis is rotated
        b (float): angle around which y axis is rotated
        c (float): angle around which z axis is rotated

    Returns:
        rotationMatrix (array): the resulting rotationMatrix    
    '''

    rotationMatrix = np.array([[np.cos(a)*np.cos(b), np.cos(a)*np.sin(b)*np.sin(c)-np.sin(a)*np.cos(c), np.cos(a)*np.sin(b)*np.cos(c)+np.sin(a)*np.sin(c)],
                            [np.sin(a)*np.sin(b), np.sin(a)*np.sin(b)*np.sin(c)+np.cos(a)*np.cos(c),np.sin(a)*np.sin(b)*np.cos(c)-np.cos(a)*np.sin(c)],
                           [-np.sin(b), np.cos(b)*np.sin(c), np.cos(b)*np.cos(c)]])
    return rotationMatrix


def medianRotated(positions,centre, a,b,c):

    '''
    
    Returns the mean median position of particles, where the median is taken across multiple sets of axes.
    
    Args:
        positions (array): coordinates of the particles
        centre (array): the centre around which the axes are rotated
        a (array): set of angles around which x axis is rotated
        b (array): set of angles around which y axis is rotated
        c (array): set of angles around which z axis is rotated
    
    Returns:
        centres (array): the mean position of all the medians
    '''

    centres = []
    for i in range(len(a)):
    
        matrix = rotation_matrix(a[i],b[i],c[i])
    
        pos = []
        for position in positions:
            rotated = np.dot(matrix, (position-centre))+centre
            pos.append(rotated)
        pos = np.array(pos)  
        x = np.median(pos[:,0])
        y = np.median(pos[:,1])
        z = np.median(pos[:,2])

        centres.append(np.array([x,y,z]))
                       
    return np.mean(centres,axis = 0)


def minimum_potential(pos, masses):

    '''
    Finds the position of the particle with minimum potental.

    Args:
        pos (array): coordinates of all the particles
        masses (array): masses of all the particles

    Returns:
        pos[min_index] (array): the coordinate of the particle with least potential
    
    '''
    pos = np.array(pos)
    masses = np.array(masses)
    distances = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    
    np.fill_diagonal(distances, np.inf)
    
    potentials = np.sum(masses[:, None] * masses[None, :] / distances, axis=1)
    
    min_index = np.argmin(potentials)
    return pos[min_index]


def interparticle_distances(pos, centre,n):    #centre given by subfind_centre, as we want centre with particle, n is the number of particles around subfind centre to average over
   
   '''
   Calculates the mean interparticle distance of a group of particles

   Args:
        pos (array): positions of all the particles
        centre (array): the centre of the particles 
        n (int): the number of particles closest to the centre from which mean interparticle distance is found

    Returns
        average(float): the mean interparticle distance
   '''

    differences = [np.linalg.norm(position-centre) for position in pos]
    differences = np.sort(differences)

    average = np.mean(differences[:n])

    return average