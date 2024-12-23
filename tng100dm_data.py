# This is code to extract the subhalo data of surviving subhalos from the simulation for dark counterparts

import numpy as np
import matplotlib.pyplot as plt
import illustris_python as il
from tqdm import tqdm
import os
import pandas as pd
from joblib import Parallel, delayed #This is to parallelize the code


h = 0.6774
mdm = 8.9e6

filepath = '/rhome/psadh003/bigdata/tng100dark/tng_files/'
outpath  = '/rhome/psadh003/bigdata/tng100dark/output_files/'
baseUrl = 'https://www.tng-project.org/api/TNG100-1-Dark/'
headers = {"api-key":"894f4df036abe0cb9d83561e4b1efcf1"}
basePath = '/rhome/psadh003/bigdata/TNG100-1-Dark/output'

ages_df = pd.read_csv('/rhome/psadh003/bigdata/tng50/tng_files/ages_tng.csv', comment = '#')

all_snaps = np.array(ages_df['snapshot'])
all_redshifts = np.array(ages_df['redshift'])
all_ages = np.array(ages_df['age(Gyr)'])

#Let fofno be the input to the code from the terminal
fofno = int(os.sys.argv[1])  #This is an integer
this_fof = il.groupcat.loadSingle(basePath, 99, haloID = fofno)
central_id_at99 = this_fof['GroupFirstSub']
this_fof_nsubs = this_fof['GroupNsubs']


central_fields = ['GroupFirstSub', 'SubhaloGrNr', 'SnapNum', 'GroupNsubs', 'SubhaloPos', 'Group_R_Crit200', 'SubhaloVel', 'SubhaloCM', 'Group_M_Crit200']
central_tree = il.sublink.loadTree(basePath, 99, central_id_at99, fields = central_fields, onlyMPB = True)
central_snaps = central_tree['SnapNum']
central_redshift = all_redshifts[central_snaps]
central_x =  central_tree['SubhaloPos'][:, 0]/(1 + central_redshift)/h
central_y =  central_tree['SubhaloPos'][:, 1]/(1 + central_redshift)/h
central_z =  central_tree['SubhaloPos'][:, 2]/(1 + central_redshift)/h
central_vx = central_tree['SubhaloVel'][:, 0]
central_vy = central_tree['SubhaloVel'][:, 1]
central_vz = central_tree['SubhaloVel'][:, 2]
Rvir = central_tree['Group_R_Crit200']/(1 + central_redshift)/h #This is the virial radius of the group
ages_rvir = all_ages[central_snaps] #Ages corresponding to the virial radii



def get_grnr(snap):
	'''
	This function returns the group number at a given snapshot 
	snap: input snapshot
	'''
	grnr_arr = central_tree['SubhaloGrNr']
	grnr = grnr_arr[np.where(central_snaps == snap)]
	return grnr

central_grnr = np.zeros(0)

for csnap in central_snaps:
	central_grnr = np.append(central_grnr, get_grnr(csnap))





def get_subhalo_data(sfid):  
    this_subh = il.groupcat.loadSingle(basePath, 99, subhaloID = sfid)
    pos = np.array(this_subh['SubhaloPos'])/h - (central_x[0], central_y[0], central_z[0])
    cm = np.array(this_subh['SubhaloCM'])/h - (central_x[0], central_y[0], central_z[0])

    vel = np.array(this_subh['SubhaloVel'] - (central_vx[0], central_vy[0], central_vz[0]))
    
    
    pos_ar = pos.reshape(1, -1)
    cm_ar = cm.reshape(1, -1)
    vel_ar = vel.reshape(1, -1)

    # Let us now get the vmax_if by looking at the merger tree
    if_tree = il.sublink.loadTree(basePath, int(99), int(sfid),
            fields = ['SubfindID', 'SnapNum', 'SubhaloMass', 'SubhaloGrNr', 'GroupFirstSub', 'SubhaloVmax'], 
            onlyMPB = True) #Progenitor tree
    if_snap = if_tree['SnapNum']
    snap_len = len(if_snap)
    if_grnr = if_tree['SubhaloGrNr']

    vpeak = np.max(if_tree['SubhaloVmax']) #This is the peak value of Vmax for the subhalo across all snapshots

    i1 = 0
    i2 = 0
    inf1_snap = -1
    inf1_sfid = -1
    matching_snap = -1


    for ix in range(len(if_snap)):
        '''
        This loop is to go through all the snaps in order to obtain the snap where infall happened
        '''
        snap_ix = if_snap[snap_len - ix - 1] #Go in an ascending order of snapshots for this variable
        
        # print(if_snap, if_grnr, central_grnr)
        
        if (i1 == 0) and (if_tree['SubfindID'][ix] == if_tree['GroupFirstSub'][ix]):
            inf1_snap = if_snap[ix] #This would be the last time when it made transition from central to a satellite
            i1 = 1
            # print(snap, subid, inf1_snap)
        # if (i2 == 0) and (if_grnr[snap_len - ix - 1] == central_grnr[central_snaps == snap_ix]):
        if if_grnr[snap_len - ix - 1].size * central_grnr[central_snaps == snap_ix].size > 0: #What is this for? Assuming this is a check if subhalo existed at this snapshot
            if (i2 == 0) and (if_grnr[snap_len - ix - 1] == central_grnr[central_snaps == snap_ix]):
                matching_snap = snap_ix #This would be the first time when this entered FoF halo 0
                i2 = 1
        if i1*i2 == 1:
            # print(pos_ar[:, 0][0]) 
            # column_names = ['posx_ar', 'posy_ar', 'posz_ar', 'cmx_ar', 'cmy_ar', 'cmz_ar', 'vmax_ar', 'mass_ar', 'len_ar', 'sfid_ar', 
                # 'snap_if_ar', 'sfid_if_ar', 'vmax_if_ar', 'mdm_if_ar', 'inf1_snap_ar', 'inf1_sfid_ar', 'vpeak']
            return pos_ar[:, 0][0], pos_ar[:, 1][0], pos_ar[:, 2][0], cm_ar[:, 0][0], cm_ar[:, 1][0], cm_ar[:, 2][0], this_subh['SubhaloVmax'], this_subh['SubhaloMass']  * 1e10/h, this_subh['SubhaloLen'], sfid, matching_snap, if_tree['SubfindID'][if_snap == matching_snap], if_tree['SubhaloVmax'][if_snap == matching_snap], if_tree['SubhaloMass'][if_snap == matching_snap]*1e10/h, inf1_snap, if_tree['SubfindID'][if_snap == inf1_snap], vpeak, vel_ar[:, 0][0], vel_ar[:, 1][0], vel_ar[:, 2][0]
            

            
        
    if i2 ==1 and i1 == 0: # This would be the case wherewe have shifting to the FoF, but it was never a central. It has to be in one of the two cases.
        return pos_ar[:, 0][0], pos_ar[:, 1][0], pos_ar[:, 2][0], cm_ar[:, 0][0], cm_ar[:, 1][0], cm_ar[:, 2][0], this_subh['SubhaloVmax'], this_subh['SubhaloMass']  * 1e10/h, this_subh['SubhaloLen'], sfid, matching_snap, if_tree['SubfindID'][if_snap == matching_snap], if_tree['SubhaloVmax'][if_snap == matching_snap], if_tree['SubhaloMass'][if_snap == matching_snap]*1e10/h, -1, -1, vpeak, vel_ar[:, 0][0], vel_ar[:, 1][0], vel_ar[:, 2][0]

    


def get_merged_ids(tree):
	'''
	Expecting a full merger tree as the input. 
	This function returns the list of IDs which merged into an FoF halo at a given snap

	-o- Assumptions: All the 'NextProgenitorID's are the ones which merged into a given subhalos
	-o- However, NextProgenitorID is given for the merger tree, we need to get its SubFind ID
	-o- How to get the snapshot of merger? Current snapshot is the last snapshot where the subhalos are going to exist.
	-o- We are not going to consider subhalos which merged outside the FoF halo. 
	'''
	merged_ids = np.zeros(0)
	merged_snaps = np.zeros(0)

	snaps = tree['SnapNum']
	grnrs = tree['SubhaloGrNr']
	sfids = tree['SubfindID']
	subids = tree['SubhaloID']
	mstar = tree['SubhaloMassInRadType'][:, 4]*1e10/h
	npids = tree['NextProgenitorID']
	npids = npids[npids != -1]
	# print(np.all(np.diff(subids) == 1))

	# print(subids)
	for ix in (range(len(npids))):
		npid = npids[ix]
		this_sh_ix = npid - subids[0]
		sh_grnr = grnrs[this_sh_ix]
		sh_merger_snap = snaps[this_sh_ix]
		this_snap_central_grnr = get_grnr(sh_merger_snap)

		if sh_grnr == this_snap_central_grnr:
			# continue
			sh_merger_snap = snaps[this_sh_ix] #This would be the last snapshot where the subhalo existed
			sh_sfid = sfids[this_sh_ix] #This is the SubFind ID of the subhalo
			merged_ids = np.append(merged_ids, sh_sfid)
			merged_snaps = np.append(merged_snaps, sh_merger_snap)
		
	return merged_snaps, merged_ids 


def get_tree_for_mrger_data(ix):
    '''
    -o- This function is to obtain the merger tree for a given subhalos that is still existing in the simulation at z = 0.
    -o- This merger tree can then be passed on to the above function to extract the satellites which get merged. 
    ''' 



# this_fof_nsubs = 10        
    
        
results = Parallel(n_jobs=32, pre_dispatch='1.5*n_jobs')(delayed(get_subhalo_data)(ix) for ix in tqdm(range(central_id_at99 + 1, central_id_at99 + this_fof_nsubs)))
results = [value for value in results if value is not None] #Getting rid of all the None entries

#Let us now save all of these files in a single .csv file
column_names = ['posx_ar', 'posy_ar', 'posz_ar', 'cmx_ar', 'cmy_ar', 'cmz_ar', 'vmax_ar', 'mass_ar', 'len_ar', 'sfid_ar', 
                'snap_if_ar', 'sfid_if_ar', 'vmax_if_ar', 'mdm_if_ar', 'inf1_snap_ar', 'inf1_sfid_ar', 'vpeak', 'velx_ar', 'vely_ar', 'velz_ar']
df = pd.DataFrame(columns=column_names)
for ix in range(len(results)):
    print(df)
    print(results[ix])
    df.loc[len(df)] = results[ix]


df['fof'] = fofno

#Let us now save the file
df.to_csv(filepath + 'fofno_' + str(fofno) + '.csv', index = False)







# pos_ar = np.zeros(0)
# vmax_ar = np.zeros(0)
# cm_ar = np.zeros(0)
# mass_ar = np.zeros(0)
# len_ar = np.zeros(0)

# for sfid in tqdm(range(central_id_at99, central_id_at99 + this_fof_nsubs)):
#     this_subh = il.groupcat.loadSingle(basePath, 99, subhaloID = sfid)
#     pos = np.array(this_subh['SubhaloPos'])/h
#     cm = np.array(this_subh['SubhaloCM'])/h
#     vmax_ar = np.append(vmax_ar, this_subh['SubhaloVmax'])
#     mass_ar = np.append(mass_ar, this_subh['SubhaloMass']  * 1e10/h)
#     len_ar = np.append(len_ar, this_subh['SubhaloLen'])
    
#     if len(pos_ar) == 0:
#         pos_ar = pos.reshape(1, -1)
#         cm_ar = cm.reshape(1, -1)
#     else:
#         pos_ar = np.append(pos_ar, pos.reshape(1, -1), axis = 0)
#         cm_ar = np.append(cm_ar, cm.reshape(1, -1), axis = 0)


# column_names = ['posx_ar', 'posy_ar', 'posz_ar', 'cmx_ar', 'cmy_ar', 'cmz_ar', 'vmax_ar', 'mass_ar', 'len_ar', 'sfid_ar']
# df = pd.DataFrame(columns=column_names)
# df['posx_ar'] = pos_ar[:, 0]
# df['posy_ar'] = pos_ar[:, 1]
# df['posz_ar'] = pos_ar[:, 2]
# df['cmx_ar'] = cm_ar[:, 0]
# df['cmy_ar'] = cm_ar[:, 1]
# df['cmz_ar'] = cm_ar[:, 2]
# df['vmax_ar'] = vmax_ar
# df['mass_ar'] = mass_ar
# df['len_ar'] = len_ar
# df['sfid_ar'] = np.arange(central_id_at99, central_id_at99 + this_fof_nsubs)

# #Let us now save the file
# df.to_csv(filepath + 'fofno_' + str(fofno) + '.csv', index = False)



