import numpy as np
import h5py

class load_snapshot():

    '''
    Loads in the snapshot data from supercomputer database
    '''

    def __init__(self, snapshot, partType, details):

        '''
        Sets up file path for the data required

        Args:
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
        
        self.fileroot = "{}/snapshot_{}.".format(str(snapshot).zfill(3),str(snapshot).zfill(3))
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
        Returns data required
        '''

        return self.results


    def get_time(self):
        '''
        Returns the time of the snapshot in terms of the cosmological scale factor
        '''
        return self.Time

#EXAMPLE
# results = A.SnapshotData.load_snapshot(snap, 0, ["Coordinates", "Masses"])
# coords0,masses0 = results.get_results()
