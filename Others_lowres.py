import numpy as np
import h5py

class load_snapshot():

    '''
    Loads in the snapshot data from supercomputer database
    '''

    def __init__(self, simulation_number, snapshot, partType, details):

        '''
        Sets up file path for the data required

        Args:
            simulation_number (int): Auriga halo to be accessed
            snapshot (int): snapshot of data required
            partType (int): particle type required
                0: Gas
                1: Low mass dark matter
                2: High mass dark matter
                3: High mass dark matter
                4: Star and wind
                5: Blackhole
            details (string): the particle attribution required
                e.g. 'Mass', 'Coordinates', 'Velocity' etc..
        '''
        
        self.fileroot = "halo_{}/output/snapdir_{}/snapshot_{}.".format(simulation_number, str(snapshot).zfill(3),str(snapshot).zfill(3))
        self.loadtype = partType
        self.details = details

        self.load_header()
        self.load_snapshot_data()

    def load_header(self):

        '''
        Loads data from the header of the hdf file, including number of particles, time of snapshot,
        number of files across which data is split
        '''

        self.filename = self.fileroot+"0.hdf5"
        
        with h5py.File(self.filename, "r") as snap:
            self.num_files = snap["Header"].attrs["NumFilesPerSnapshot"]
            self.boxsize = snap["Header"].attrs["BoxSize"]
            self.masses = snap["Header"].attrs["MassTable"]
            self.NumPart_Total = snap["Header"].attrs["NumPart_Total"]
            self.Time = snap["Header"].attrs["Time"]

        return

    def load_snapshot_data(self):

        '''
        Accesses the data required from the file
        '''

        partType = self.loadtype
        self.results = []
        
        self.filenames = []
        for i in range(self.num_files):
            self.filenames.append(self.fileroot+"{}.hdf5".format(i))

        for detail in self.details:
            detail_result = []

            if detail == "Masses" and partType == 1:
                detail_result.append([self.masses[1]]*self.NumPart_Total[1])
                
            else: 
                for file in self.filenames:
                    with h5py.File(file, "r") as snap:
                        NumPart_ThisFile = snap["Header"].attrs["NumPart_ThisFile"]
                        if NumPart_ThisFile[partType]>0:
                            information = snap["PartType{}/{}".format(partType, detail)][...]
                            detail_result.append(information)
                        
            if len(detail_result)>0:
                detail_result =  np.concatenate(detail_result, axis =0)
            
            self.results.append(detail_result)

    def get_results(self):

        '''
        Returns required data
        '''

        return self.results


class group_data():

    '''
    Loads in group data from the supercomputer database
    '''

    def __init__(self, simulation_number, snapshot, n, detail):

        '''
        Sets up file path for the data required

        Args:
            simulation_number (int): Auriga halo to be accessed
            snapshot (int): snapshot of data required
            n (int): the group number of the subhalo being accessed, assume always looking at halo 0
            details (string): the particle attribution required
                e.g. 'Mass', 'Radius', 'Composition' etc..
        '''

        self.n = n
        self.detail = detail
        self.filename = "halo_{}/output/groups_{}/fof_subhalo_tab_{}.0.hdf5".format(simulation_number, str(snapshot).zfill(3),str(snapshot).zfill(3))
        self.load_header()
        self.get_detail()

    def load_header(self):

        '''
        Loads data from the header of the hdf file, including number of particles, number of haloes and
        subhaloes contained in each file
        '''
        
        with h5py.File(self.filename, "r") as snap:
            self.num_files = snap["Header"].attrs["NumFiles"]
            self.boxsize = snap["Header"].attrs["BoxSize"]
            self.Ngroups_ThisFile = snap["Header"].attrs["Ngroups_ThisFile"] 
            self.Nsubgroups_ThisFile = snap["Header"].attrs["Nsubgroups_ThisFile"] 
        return


    def get_detail(self):

        '''
        Accesses the data required from the file
        '''

        with h5py.File(self.filename, "r") as snap:
            result = snap["{}".format(self.detail)][...]
            if len(result)>self.n:
                self.result = result[self.n]
            else:
                print('Unable to extract information for this subhalo in this group')
        return

    
    def return_detail(self):

        '''
        Returns the data
        '''

        return self.result


class mergertree():

    '''
    Uses merger trees to track the same subhalo either backwards or forwards in time over the course
    of the subhalo's existence
    '''

    def __init__(self, simulation_number, snapshot, end, subhaloID, groupID, detail):

        '''
        Initialises details of merger tree required and path required

        Args:
            simulation_number (int): Auriga halo to be accessed
            snapshot (int): starting snapshot for tree
            end (int): end snapshot for tree
            subhaloID(int): subhalo number
            groupID(int): group number
            details (string): the particle attribution required
                e.g. 'Mass', 'Radius', 'Composition' etc..
        '''
        
        self.filename = "mergertrees_new/Au-{}/trees_sf1_127.0.hdf5".format(simulation_number)
        self.snapshot = snapshot
        self.subhaloID = subhaloID
        self.groupID = groupID
        self.detail = detail
        self.endSnap = end

        self.load_data()

        return
        
    def load_data(self):

        '''
        Loads data about the merger tree at the starting snapshot and creates data to be stored
        '''

        with h5py.File(self.filename, "r") as snap:
            self.SubhaloGrNr = snap['Tree0/SubhaloGrNr'][...]
            self.SubhaloNumber = snap['Tree0/SubhaloNumber'][...]
            self.SnapNum = snap['Tree0/SnapNum'][...]
            self.Data = snap['Tree0/{}'.format(self.detail)][...]
            self.FirstProgen = snap['Tree0/FirstProgenitor'][...]
            self.Descend = snap['Tree0/Descendant'][...]
            self.NextProgen = snap['Tree0/NextProgenitor'][...]
            self.MassType = snap['Tree0/SubhaloMassType'][...]

        return

    
    def walk_the_tree(self, mainprogonly = True):

        '''
        Starts the process of walking the tree BACKWARDS in time
        
        Args:
            mainprogonly (bool): If True, only follow history of subhalo required
        
        Returns:
            data (list): data required for subhalo at each snapshot
            snapshots (list):  snapshots over which merger tree extends
 
        '''

        ref_index = np.where((self.SubhaloGrNr == self.groupID) & (self.SubhaloNumber == self.subhaloID) & (self.SnapNum == self.snapshot))[0]
        if len(ref_index) ==0:
            print('subhalo not in this tree!')
            return None
        else:
            return self.object_history(ref_index, mainprogonly = mainprogonly)
            
        return

    def object_history(self, ref_index, mainprogonly = False):
        data = []
        snapshots =[]
        self.ReturnTreeNode(ref_index, data, snapshots, mainprogonly)
        return data, snapshots
        

    def ReturnTreeNode(self, index, data, snapshots, mainprogonly):
        data.append(self.Data[index])
        snapshots.append(self.SnapNum[index])
        self.ReturnProgenitor(index, data, snapshots, mainprogonly)


    def ReturnProgenitor(self, index, data, snapshots, mainprogonly):
        first_prog = self.FirstProgen[index]        

        if first_prog < 0 or self.endSnap == self.SnapNum[index]:
            return
            
        if not mainprogonly:
            next_prog = self.NextProgen[first_prog]

            while next_prog >= 0:
                self.ReturnTreeNode(next_prog, data, snapshots, mainprogonly)
                next_prog = self.NextProgen[next_prog]
        
        self.ReturnTreeNode(first_prog, data, snapshots,mainprogonly)
        
        first_prog = self.FirstProgen[first_prog]

        return 


    def walk_the_tree_ascending(self, mainprogonly = True):

        '''
        Starts the process of walking the tree FORWARDS in time
        
        Args:
            mainprogonly (bool): If True, only follow history of subhalo required
        
        Returns:
            data (list): data required for subhalo at each snapshot
            snapshots (list):  snapshots over which merger tree extends
 
        '''

        ref_index = np.where((self.SubhaloGrNr == self.groupID) & (self.SubhaloNumber == self.subhaloID) & (self.SnapNum == self.snapshot))[0]
        if len(ref_index) ==0:
            print('subhalo not in this tree!')
            return None
        else:
            return self.object_history_descending(ref_index, mainprogonly = mainprogonly)
            
        return

    def object_history_descending(self, ref_index, mainprogonly = False):
        data = []
        snapshots =[]
        self.ReturnTreeNode_descending(ref_index, data, snapshots, mainprogonly)
        return data, snapshots
        

    def ReturnTreeNode_descending(self, index, data, snapshots, mainprogonly):
        data.append(self.Data[index])
        snapshots.append(self.SnapNum[index])
        self.ReturnProgenitor_descending(index, data, snapshots, mainprogonly)


    def ReturnProgenitor_descending(self, index, data, snapshots, mainprogonly):
        descendant = self.Descend[index]        

        if descendant < 0 or self.endSnap == self.SnapNum[index]:
            return
        
        self.ReturnTreeNode_descending(descendant, data, snapshots,mainprogonly)
        
        descendant = self.Descend[descendant]

        return 


hubbleparam = 0.6777
OmegaMatter = 0.307
OmegaLambda = 0.693
alpha = OmegaLambda/OmegaMatter


def time_lookback(a):

    '''
    Finds the lookback time for a given cosmological scale factor
    
    Args:
        a (float): cosmological scale factor

    Returns:
        time (float): lookback time in Gyr
    '''

    convert_in_Gyr = 3.085678e10 / 3.1536e7
    H = 100
    x = np.sqrt(alpha)+np.sqrt(1+alpha)
    y = np.sqrt(alpha*a**3) + np.sqrt(1+a**3*alpha)

    time = 2*np.log(x/y)/(3*H*hubbleparam*np.sqrt(OmegaLambda))

    return time* convert_in_Gyr
    

def time(simulation_number, s):

    '''
    Finds the time since the start of a simulation for a given snapshot number

    Args:
        simulation_number(int): Auriga halo to be accessed
        s (int): snapshot number
    
    Returns:
        time (float): time since the start of the simulation to that snapshot in Gyr
    '''

    filename="halo_{}/output/snapdir_{}/snapshot_{}.{}.hdf5".format(simulation_number, str(s).zfill(3),str(s).zfill(3), 0)
            
    with h5py.File(filename, "r") as snap:
        a = snap["Header"].attrs["Time"]

    filename="halo_{}/output/snapdir_000/snapshot_000.0.hdf5".format(simulation_number)
            
    with h5py.File(filename, "r") as snap:
        a_0 = snap["Header"].attrs["Time"]

    time = time_lookback(a_0)-time_lookback(a)

    return time


def scale_fac(n, s):

    '''
    Finds the cosmological scale factor for a given snapshot

    Args:
        n (int): Auriga halo to be accessed
        s (int): snapshot number
    
    Returns:
        a (float): cosmological scale factor
    '''

    filename="halo_{}/output/snapdir_{}/snapshot_{}.{}.hdf5".format(n,str(s).zfill(3),str(s).zfill(3), 0)
            
    with h5py.File(filename, "r") as snap:
        a = snap["Header"].attrs["Time"]
    return a

def redshift(n, s):

    '''
    Finds the redshift for a given snapshot

    Args:
        n (int): Auriga halo to be accessed
        s (int): snapshot number
    
    Returns:
        z (float): redshift
    '''

    filename="halo_{}/output/snapdir_{}/snapshot_{}.{}.hdf5".format(n,str(s).zfill(3),str(s).zfill(3), 0)
            
    with h5py.File(filename, "r") as snap:
        z = snap["Header"].attrs["Redshift"]
    return z


def fourier_mode(m, coord, masses):

    '''
    Finds the Fourier mode of the mass distribution of particles

    Args:
        m (int): Fourier mode required (1 for non-axisymmetry)
        coord (array): particle coordinates
        masses (array): particle masses
    
    Returns:
        A_m (float): fourier mode
    '''

    angles = np.arctan(coord[:,1]/coord[:,0])
    result = masses*np.exp(1j * angles*m)
    
    mask = ~np.isnan(result)
    masses = masses[mask]
    result = result[mask]
    
    A_m = np.abs(np.sum(result))/np.abs(np.sum(masses))
    return A_m