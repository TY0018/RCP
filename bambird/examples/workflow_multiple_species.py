#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.close("all")

# import the package bambird
import bambird
import bambird.config as cfg

# %%
# Define constants
# ----------------

# Temporary saved directory
TEMP_DIR        = "/home/users/ntu/ytong005/scratch/sg_bird_dataset"
DATASET_NAME    = Path('SG_Birds')
ROIS_NAME       = Path(str(DATASET_NAME) +'_ROIS')

# List of species to build a clean dataset
SCIENTIC_NAME_LIST = [ 
                        "Acridotheres tristis", #commyn
                        "Aethopyga siparaja", #eacsun1
                        "Amaurornis phoenicurus", #whbwat1
                        "Anthracoceros albirostris", #orphor1
                        "Aplonis panayensis", #asgsta1
                        "Columba livia", #rocpig
                        "Copsychus saularis", #magrob, cnt download, not in model
                        "Corvus splendens", # houcro1
                        "Dicaeum cruentatum", #scbflo1
                        "Gallus gallus", # redjun
                        "Hirundo tahitica", #pacswa1
                        "Oriolus chinensis", #blnori1, cnt download, not in model
                        "Orthotomus sutorius", #comtai1
                        "Passer montanus", #eutspa
                        "Pycnonotus jocosus", #rewbul
                        "Spilopelia chinensis", # not in XCL
                        "Todiramphus chloris", # colkin1
                        "Treron vernans", # pinpig3
                        "Acridotheres javanicus", #cnt download, whvmyn
                        "Gracula religiosa" #cnt, hilmyn
                      ]

CONFIG_FILE = '/home/users/ntu/ytong005/bambird/config_default.yaml' 

# After the process, remove the audio that were saved during the process ?
CLEAN = False

# %%
if __name__ == '__main__':

    # Load the configuration file    
    params = cfg.load_config(CONFIG_FILE)
    # bambird.check_url_download()

#%%    
    # Query Xeno-Canto
    # ----------------
    df_dataset = bambird.query_xc(
                        species_list    = SCIENTIC_NAME_LIST,
                        params          = params['PARAMS_XC'],
                        random_seed     = params['RANDOM_SEED'],
                        verbose         = True
                        )
    
    # Download audio Xeno-Canto
    # -------------------------
    df_xc, csv_xc  = bambird.download_xc (
                        df_dataset      = df_dataset,
                        rootdir         = TEMP_DIR, 
                        dataset_name    = DATASET_NAME, 
                        csv_filename    = params['PARAMS_XC']['CSV_XC_FILE'],
                        overwrite       = True,
                        verbose         = True
                        )
    
#%% 
    
    # Extract ROIS
    # -------------------------------
    
    # ROIS extraction of the full dataset
    df_rois, csv_rois = bambird.multicpu_extract_rois(
                        dataset     = df_xc,
                        params      = params['PARAMS_EXTRACT'],
                        save_path   = TEMP_DIR / ROIS_NAME,
                        overwrite   = True,
                        nb_cpu      = 10,
                        verbose     = True
                        )
    
#%%
    # Compute features for each ROIS
    # -------------------------------
        
    # Test if at least 1 ROI was found     
    if len(df_rois) > 0 :    
        # compute the features on the full dataset       
        df_features, csv_features = bambird.multicpu_compute_features(
                        dataset     = df_rois,
                        params      = params['PARAMS_FEATURES'],
                        save_path   = TEMP_DIR / ROIS_NAME,
                        overwrite   = True,
                        nb_cpu      = 10,
                        verbose     = True)
        
#%%        
    #  Cluster ROIS
    # -------------------------------
    
    # with dataframe or csv file
    
    dataset = df_features 
    
    try : 
        df_cluster, csv_clusters = bambird.find_cluster(
                        dataset = dataset,
                        params  = params['PARAMS_CLUSTER'],
                        display = False,
                        verbose = True
                        )
    except:
       df_cluster = df_features 
       df_cluster['auto_label'] = 0
       df_cluster['cluster_number'] = -1
       
#%%    
    # Display the ROIS
    # -------------------------------
    
    bambird.overlay_rois(
                        cluster         = df_cluster,
                        params          = params['PARAMS_EXTRACT'],
                        column_labels   = 'cluster_number', #auto_label cluster_number
                        unique_labels   = np.sort(df_cluster.cluster_number.unique()),
                        filename        = None,
                        random_seed     = None,
                        verbose         = True
                        )

#%%    
    # Remove files
    # -------------------------------
    
    if CLEAN :
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
            