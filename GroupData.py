import numpy as np
import h5py 

class group_data():

    '''
    Loads in group data from the supercomputer database
    '''

    def __init__(self, snapshot, n, detail):

        '''
        Sets up file path for the data required

        Args:
            snapshot (int): snapshot of data required
            n (int): the group number of the subhalo being accessed, assume always looking at halo 0
            details (string): the particle attribution required
                e.g. 'Mass', 'Radius', 'Composition' etc..
        '''

        self.n = n
        self.detail = detail
        self.filename = "groups_{}/fof_subhalo_tab_{}.0.hdf5".format(str(snapshot).zfill(3),str(snapshot).zfill(3))
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


#EXAMPLE:
#group_object = A.GroupData.group_data(2999,0, 'Subhalo/SubhaloPos')
#results = group_object.return_detail()
        