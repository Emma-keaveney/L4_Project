import h5py
import numpy as np


class load_snapshot():

    '''
    Uses merger trees to track the same subhalo either backwards or forwards in time over the course
    of the subhalo's existence
    '''

    def __init__(self, snapshot, end, subhaloID, groupID, detail):

        '''
        Initialises details of merger tree required and path required

        Args:
            snapshot (int): starting snapshot for tree
            end (int): end snapshot for tree
            subhaloID(int): subhalo number
            groupID(int): group number
            details (string): the particle attribution required
                e.g. 'Mass', 'Radius', 'Composition' etc..
        '''
        
        self.filename = "trees_sf1_2999.0.hdf5"
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
            print('subhalo present!')
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
            print('Tree ended at: ', self.SnapNum[index])
            print('Subhalo number: ',self.SubhaloNumber[first_prog])
            print('Group number: ', self.SubhaloGrNr[first_prog])
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
        Starts the process of walking the tree FORWARD in time
        
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
            print('subhalo present!')
            return self.object_history_ascending(ref_index, mainprogonly = mainprogonly)
            
        return

    def object_history_ascending(self, ref_index, mainprogonly = False):
        data = []
        snapshots =[]
        self.ReturnTreeNode_ascending(ref_index, data, snapshots, mainprogonly)
        return data, snapshots
        

    def ReturnTreeNode_ascending(self, index, data, snapshots, mainprogonly):
        data.append(self.Data[index])
        snapshots.append(self.SnapNum[index])
        self.ReturnProgenitor_ascending(index, data, snapshots, mainprogonly)


    def ReturnProgenitor_ascending(self, index, data, snapshots, mainprogonly):
        descendant = self.Descend[index]        

        if descendant < 0 or self.endSnap == self.SnapNum[index]:
            print('Tree ended at: ', self.SnapNum[index])
            print('Subhalo number: ',self.SubhaloNumber[descendant])
            print('Group number: ', self.SubhaloGrNr[descendant])
            return
        
        self.ReturnTreeNode_acending(descendant, data, snapshots,mainprogonly)
        
        descendant = self.Descend[descendant]

        return 


    
    def number_mergers(self, m, Type):   

        '''
        Gives the number of mergers experienced by a halo over the course of its history

        Args:
            m (float): the mass ratio between subhalo and merging dwarf galaxy for merger to be counted
            Type (int): particle type over which mass ratio requirement applies
        '''

        self.Type = Type
        
        ref_index = np.where((self.SubhaloGrNr == self.groupID) & (self.SubhaloNumber == self.subhaloID) & (self.SnapNum == self.snapshot))[0]
        if len(ref_index) ==0:
            print('subhalo not in this tree!')
            return None
        else:
            print('subhalo present!')
            return self.object_history_mergers(ref_index,m)
            
        return

    def object_history_mergers(self, ref_index,m):
        snapshots =[]
        data = []                                                       #here data is number of mergers
        self.ReturnTreeNodeMergers(ref_index, data, snapshots,m)
        return data, snapshots
        

    def ReturnTreeNodeMergers(self, index, data, snapshots,m):
        merger = False
        nextprogen = self.NextProgen[index]
        snapshots.append(self.SnapNum[index])

        main_mass = self.MassType[index][0]
        sub_mass = self.MassType[nextprogen][0]

        
        if self.SnapNum[index] == self.SnapNum[nextprogen]:
            if m ==0:
                data.append(len(nextprogen))
                merger = True

            else:
                num = 0
                for progen in nextprogen:
                    if self.Type == True:
                        mass_main = np.sum(main_mass)
                        mass_sub = np.sum(sub_mass)
                    else:
                        mass_main = main_mass[self.Type]
                        mass_sub = sub_mass[self.Type]
                        
                    if m*mass_main <= mass_sub:
                        num+=1
                data.append(num)
                merger = True
                
        if merger == False:
            data.append(0)
                
        self.ReturnProgenitorMergers(index, data, snapshots,m)


    def ReturnProgenitorMergers(self, index, data, snapshots,m):
        first_prog = self.FirstProgen[index]        

        if first_prog < 0 or self.endSnap == self.SnapNum[index]:
            print('Tree ended at: ', self.SnapNum[index])
            print('Subhalo number: ',self.SubhaloNumber[first_prog])
            print('Group number: ', self.SubhaloGrNr[first_prog])
            return
        
        self.ReturnTreeNodeMergers(first_prog, data, snapshots,m)
        
        first_prog = self.FirstProgen[first_prog]

        return 


    def mergers_data(self):
        
        ref_index = np.where((self.SubhaloGrNr == self.groupID) & (self.SubhaloNumber == self.subhaloID) & (self.SnapNum == self.snapshot))[0]
        if len(ref_index) ==0:
            print('subhalo not in this tree!')
            return None
        else:
            print('subhalo present!')
            return self.object_history_formergers(ref_index)
            
        return

    def object_history_formergers(self, ref_index):
        snapshots =[]
        data = []                                                       #here data is number of mergers
        self.ReturnTreeNode_formergers(ref_index, data, snapshots)
        return data, snapshots
        

    def ReturnTreeNode_formergers(self, index, data, snapshots):
        nextprogen = self.NextProgen[index]
        if nextprogen > -1:
            data.append(self.Data[nextprogen])
            snapshots.append(self.SnapNum[nextprogen])
        self.ReturnProgenitor_formergers(index, data, snapshots)


    def ReturnProgenitor_formergers(self, index, data, snapshots):
        first_prog = self.FirstProgen[index]        

        if first_prog < 0 or self.endSnap == self.SnapNum[index]:
            print('Tree ended at: ', self.SnapNum[index])
            print('Subhalo number: ',self.SubhaloNumber[first_prog])
            print('Group number: ', self.SubhaloGrNr[first_prog])
            return
            
        self.ReturnTreeNode_formergers(first_prog, data, snapshots)
        
        first_prog = self.FirstProgen[first_prog]

        return 


#EXAMPLE

#Walk the tree backward
# detail = 'SubhaloNumber'
# tree1  = A.MergerTrees.load_snapshot(2999,1600,0,0, detail)
# results, snaps = tree1.walk_the_tree(mainprogonly = True)

#Walk the tree forward
# detail = 'SubhaloPos'
# tree1  = A.MergerTrees.load_snapshot(1000,1400,1,0, detail)   #for ascending end snap > beginning snap
# results, snaps = tree1.walk_the_tree_ascending()
