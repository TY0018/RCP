#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Collection of functions to query metadata from xeno-canto and download the audio
to build a dataset.
"""

#%%
# general packages
import os
from pathlib import Path
from dotenv import load_dotenv
# basic packages
import pandas as pd
import glob
import requests 
import time
# Scikit-Maad (ecoacoustics functions) package
import maad
from tqdm import tqdm

#
from bambird import config as cfg
# cfg.get_config()
import urllib.request
import json
#%%

###############################################################################
def check_url_download():
    url = "https://xeno-canto.org/38422/download"
    urllib.request.urlretrieve(url, "/home/users/ntu/ytong005/test_audio.mp3")
    return 

def download_xeno_canto(df, root_dir, dataset_name, delay=0.5):
    rootdir = Path(root_dir)
    print("Downloading...")
    output_dir = rootdir / dataset_name
    failed = []

    print(f"Starting download of {len(df)} recordings...\n")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        url = row.get("file")
        file_name = row.get("file-name")
        gen = row.get("gen", "UnknownGen")
        sp = row.get("sp", "UnknownSp")

        if not file_name:
            failed.append((i, url, "Missing file-name"))
            continue

        # Split to keep only ID and extension
        base_id = file_name.split("-")[0]  # e.g. "XC1035372"
        ext = os.path.splitext(file_name)[1]  # e.g. ".wav" or ".mp3"
        fname = f"{base_id}{ext}"  # e.g. "XC1035372.wav"

        if not url or not fname:
            failed.append((i, url, "Missing file or file-name"))
            continue

        species_dir = Path(output_dir) / f"{gen}_{sp}"
        species_dir.mkdir(parents=True, exist_ok=True)

        save_path = species_dir / fname

        # Skip if already downloaded
        if os.path.exists(save_path):
            continue

        try:
            urllib.request.urlretrieve(url, save_path)
            time.sleep(delay)
        except Exception as e:
            failed.append((i, url, str(e)))
            print(f"⚠️ Failed: {url} → {e}")

    print(f"\n✅ Finished. {len(df) - len(failed)} succeeded, {len(failed)} failed.\n")
    print(failed)

def xc_query_v3(gen, sp, max_pages=500, delay=1.0, **kwargs):
    """
    Query Xeno-Canto API v3 for recordings of a given species.
    Returns a pandas DataFrame with recording metadata.
    """
    base_url = "https://xeno-canto.org/api/3/recordings"
    all_records = []
    
    load_dotenv(Path(".env"))
    api_key = os.getenv("XC_TOKEN")

    for page in range(1, max_pages + 1):
        url = f"{base_url}?query=gen:{gen}+sp:{sp}"
        # url = f"{base_url}?query=area:asia"
        # url += f"+q:A" # to adjust quality to recordings
        url += f"&page={page}&key={api_key}"
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"⚠️ Error fetching page {page}: HTTP {resp.status_code}, url: {url}")
            break

        data = resp.json()
        if "recordings" not in data or not data["recordings"]:
            print(f"No recordings in {url}")
            break

        all_records.extend(data["recordings"])

        # Stop if last page reached
        if page >= data.get("numPages", 1):
            break

        time.sleep(delay)  # small delay to avoid rate limiting

    if not all_records:
        print(f"No results found for query: {gen} {sp}")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Handle lat/lon safely (convert invalid or empty to NaN)
    for col in ["lat", "lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df = df[df["file"].notnull() & (df["file"] != "")]

    return df

def query_xc (species_list, 
           params=cfg.PARAMS['PARAMS_XC'],
           format_time=False,
           format_date=False,
           random_seed=cfg.RANDOM_SEED, 
           verbose=False):
    """
    Query metadata from Xeno-Canto website with audiofile depending on the search terms. 
    The audio recordings metadata are grouped and stored in a dataframe.

    Parameters
    ----------
    species_list : list
        List of scientific name of birds (e.g. 'Columba palumbus')
    
    params : list
        list of search terms to perform the query
        The main seach terms are :
        - q   : quality
        - cnt : country
        - len : length
        - area : continent (europe, africa, america, asia)
        see more here : https://www.xeno-canto.org/help/search
    format_time : boolean, optional
        Time in Xeno-Canto is not always present neither correctly formated. 
        If true, time will be correctly formated to be processed as DateTime 
        format. When formating is not possible, the row is dropped. 
        The default is False
    format_date : boolean, optional
        Date in Xeno-Canto is not always present neither correctly formated. 
        If true, rows with uncorrect format of date are dropped.
    verbose : boolean, optional
        Print messages during the execution of the function. The default is False.

    Returns
    -------
    df_dataset : pandas DataFrame
        Dataframe containing all the recordings metadata matching search terms
    """
    
    # separate the genus and the species from the scientific name

    df_dataset = pd.DataFrame()
    JSON_FILE = "/home/users/ntu/ytong005/dataset_json/singapore_birds_extra.json"
    with open(JSON_FILE, "r") as f:
        data = json.load(f)
        species_list = [bird["scientific_name"] for bird in data]

    for species in species_list:
        gen, _, sp = species.partition(" ")
        print(f"Fetching recordings for species {gen} {sp}")

        df_part = xc_query_v3(
                gen, sp
                # len=params.get("len"),
                # q=params.get("q"),
                # area=params.get("area"),
                # cnt=params.get("cnt")
            )

        if not df_part.empty:
            df_part.insert(4, 'categories', f"{gen}_{sp}")
            df_dataset = pd.concat([df_dataset, df_part], ignore_index=True)
    # df_part = xc_query_v3()
    # if not df_part.empty:
    #     # 2. Create 'categories' by combining Genus and Species columns
    #     # Vectorized string concatenation: fast and handles the whole column at once
    #     df_part['categories'] = df_part['gen'] + '_' + df_part['sp']
        
    #     # 3. Add to main dataset
    #     df_dataset = pd.concat([df_dataset, df_part], ignore_index=True)
    if df_dataset.empty and verbose:
        print(f"No audio recordings found with parameters: {params}")

    return df_dataset

###############################################################################
def download_xc (df_dataset,
                rootdir, 
                dataset_name, 
                csv_filename="xenocanto_metadata.csv",
                overwrite=False,
                verbose=True):
    """
    Download the audio files from Xeno-Canto based on the input dataframe
    It will create directories for each species if needed

    Parameters
    ----------
    df_dataset : pandas DataFrame
        Dataframe containing the selected recordings metadata to be downloaded.
        It could be the output of dataset_query, or a subset of this
        dataframe
    rootdir : string
        Path to the directory where the whole dataset will be saved
    dataset_name : string
        Name of the dataset that will be created as a parent directory . 
    csv_filename : string, optional
        Name of the csv file where the dataframe (df_dataset) will be saved. if
        the file already exists, data will be appended at the end of the csv file.
        The default is bambird_metadata.csv
    overwrite : boolean, optional
        Test if the directory where the audio files will be downloaded already
        exists. if True, it will download the data in the directory anyway.
        Otherwise, if False, it will not download audio files.
    verbose : boolean, optional
        Print messages during the execution of the function. The default is False.

    Returns
    -------
    df_dataset : pandas DataFrame
        Dataframe containing all the selected audio recordings that were 
        resquested as input AND downloaded. If for some reasons, some of the 
        audio recordings cannot be downloaded, they do not appear in the output
        dataframe. 
    csv_fullfilename : string
        Full path to the csv file where the dataframe (df_dataset) was saved. 
        If the file already exists, data was appended at the end of the csv file.
        /!\ the csv file contains all the audio files that were recorded NOW and
        BEFPRE while the dataframe df_dataset contains ONLY the files that
        were recorded NOW.
    """
    # format rootdir as path
    rootdir = Path(rootdir)
    print("Downloading...")
    #-------------------------------------------------------------------
    # download all the audio files into a directory with a subdirectory for each
    # species
    # df_dataset = maad.util.xc_download(df           =df_dataset,
    #                                    rootdir      =rootdir,
    #                                    dataset_name =dataset_name,
    #                                    overwrite    =overwrite,
    #                                    save_csv     =True,
    #                                    verbose      =verbose)

    download_xeno_canto(df_dataset, rootdir, dataset_name)
    
    #-------------------------------------------------------------------
    def fix_filename(filename):
        try:
            # Get base ID and extension
            base_id = filename.split("-")[0]              # e.g. "XC1051079"
            ext = os.path.splitext(filename)[1]           # e.g. ".mp3" or ".wav"
            # Construct new path
            new_path = str(f"{base_id}{ext}")
            return new_path
        except Exception:
            return old_path  # fallback if parsing fails

    # Apply to column
    df_dataset["filename"] = df_dataset["file-name"].apply(fix_filename)
    df_dataset['fullfilename'] = df_dataset.apply(
        lambda row: str(rootdir / dataset_name / f"{row['gen']}_{row['sp']}" / row['filename'])
        if pd.notna(row.get('file-name')) else '', axis=1
    )

    df_dataset['categories']   = df_dataset['fullfilename'].apply(lambda path : 
                                                                  Path(path).parts[-2])
    keep_cols = [
        "id", "gen", "sp", "en", "cnt", "lat", "lon", "length", "url", "file",
        "file-name", "filename", "fullfilename", "categories", "q"
    ]
    existing_cols = [col for col in keep_cols if col in df_dataset.columns]
    df_final = df_dataset[existing_cols].copy()
    print(df_final.head(10))
    #--------------------------------------------------------------------------    
    # test if the csv file with all the metadata already exists and append the
    # df_dataset otherwise write a new csv file
    csv_fullfilename = rootdir / dataset_name / csv_filename
    if os.path.exists(csv_fullfilename):              
       # try to read the file and add the new rows (no duplicate)
        try :        
            # remove from df_data the audio files that were already downloaded
            mask = df_final['filename'].isin(pd.read_csv(csv_fullfilename,
                                                            sep=';',
                                                            index_col='id')['filename'].unique().tolist())
            # append the new audio to the dataframe and save it
            df_final[~mask].to_csv(csv_fullfilename, 
                                     sep=";", 
                                     index=False,
                                     header = False,
                                     mode = 'a')                                                   
        except Exception as e:
            print(f"Could not append to CSV - {e}")
                                                                      
    else:
        # try to create a file and add a row corresponding to the index
        try :
            df_final.to_csv(csv_fullfilename, 
                              sep=";", 
                              index=False) 
        except Exception as e:
            print(f"Could not write CSV - {e}")
    #--------------------------------------------------------------------------
    # display information about the downloaded audio files
    if verbose :        
        if len(df_final)>0 :
            print('\n' + '*' * 55)
            print(f'Total files in current batch: {len(df_final)}')
            print(f'Total unique species: {df_final["categories"].nunique()}')
            print('-' * 55)
            print('RECORDINGS PER SPECIES (Descending Order):')
            
            # This prints: Species_Name    Count
            print(df_final['categories'].value_counts().to_string())
            
            print('*' * 55 + '\n')
        else :
            print('*** WARNING *** The dataframe is empty.')  
                        
    return df_final, csv_fullfilename

#%%

def query_download_xc(
        species_list, 
        rootdir, 
        dataset_name, 
        params=cfg.PARAMS['PARAMS_XC'],
        csv_filename="bambird_metadata.csv",
        increment=True,
        only_new=True,
        format_time=False,
        format_date=False,
        random_seed=cfg.RANDOM_SEED,  
        overwrite=False,
        verbose=False
        ):
    """

    Parameters
    ----------
    species_list : TYPE
        DESCRIPTION.
    rootdir : TYPE
        DESCRIPTION.
    dataset_name : TYPE
        DESCRIPTION.
    params : TYPE, optional
        DESCRIPTION. The default is cfg.PARAMS['PARAMS_XC'].
    csv_filename : TYPE, optional
        DESCRIPTION. The default is "bambird_metadata.csv".
    format_time : TYPE, optional
        DESCRIPTION. The default is False.
    format_date : TYPE, optional
        DESCRIPTION. The default is False.
    random_seed : TYPE, optional
        DESCRIPTION. The default is cfg.RANDOM_SEED.
    overwrite : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    # Set the full path to the dataset containing all the metadata downloaded
    # from XC and then, the filename and fullfilename of the recordings downloaded
    # with the function download_xc
    csv_fullfilename = Path(rootdir) / Path(dataset_name) / csv_filename
    
    # number of file to download
    NUM_FILES = params['NUM_FILES']

    # Test if a csv with the metadata already exists
    # if not :
    if os.path.isfile(csv_fullfilename) == False :
        # set NUM_FILES to None in order to grab all the metadata for the species
        params['NUM_FILES'] = None
        df_dataset = query_xc (species_list  =species_list, 
                                params       =params,
                                format_time  =format_time,
                                format_date  =format_date,
                                random_seed  =random_seed, 
                                verbose      =verbose)
        # set NUM_FILES to it original value
        params['NUM_FILES'] = NUM_FILES
        
        # add a new column fullfilename with default value : None
        df_dataset['fullfilename'] = None
        df_dataset['filename'] = None
        df_dataset['categories'] = None
        
        # save the csv with all the requested metadata
        # first create the directory where to save the csv if it does not exist
        os.makedirs(os.path.split(csv_fullfilename)[0])
        df_dataset.set_index('id', inplace=True)
        df_dataset.to_csv(csv_fullfilename, sep=";", index=True, index_label='id') 
        
        if (NUM_FILES is not None) and (len(df_dataset)>0): 
               
            # number of audio files per species
            df_num = df_dataset.groupby(['gen','sp'], group_keys=False).apply(lambda x: len(x))
            
            # extract the combination (gen, sp) from df_groups
            keys = df_num.keys()
            
            # init the subdf_dataset
            subdf_dataset = pd.DataFrame()
            
            # test if the number of files is greater than the maximum number of
            # resquested files per species
            # Do a stratified sampling in order to download the same number of files
            # for each species (combination of gen and sp).
            # If the number of files is lower than the requested number, take all the files
            for n, k in zip( df_num, keys):
                mask = df_dataset[(df_dataset[['gen','sp']] == k)['gen']]
                if n >= NUM_FILES :
                    subdf_dataset = subdf_dataset.append(mask.apply(lambda x: x.sample(n=NUM_FILES, 
                                                                                       random_state=random_seed)))
                else:
                    subdf_dataset = subdf_dataset.append(mask)
        else:
            subdf_dataset = df_dataset
       
        # download all the audio files into a directory with a subdirectory for each
        # species
        try :
            df_dl = maad.util.xc_download(df         =subdf_dataset,
                                        rootdir      =rootdir,
                                        dataset_name =dataset_name,
                                        overwrite    =overwrite,
                                        save_csv     =False,
                                        verbose      =verbose)
        except :
            df_dl = pd.DataFrame()
        
        # add columns        
        df_dl['filename']  = df_dl['fullfilename'].apply(os.path.basename)
        df_dl['categories']= df_dl['fullfilename'].apply(lambda path : Path(path).parts[-2])
                
        # update df_dataset with df_dl
        df_dataset.update(df_dl)
        
        # save the updated dataframe
        df_dataset.to_csv(csv_fullfilename, sep=";", index=True, index_label='id') 
        
        # keep only the rows with audio already downloaded
        df = df_dataset[~df_dataset['fullfilename'].isna()]
        
    # if the csv with metadata already exists
    else:
        # read the csv
        df_dataset = pd.read_csv(csv_fullfilename, sep=';', index_col='id')
        
        if verbose:
            print((("The metadata file {} already exits\n") +
                  ("with a total of {} audio metadata\n") + 
                  ("and with {} already downloaded audio recordings")).format(csv_fullfilename,
                                                                             len(df_dataset),
                                                                             len(df_dataset[~df_dataset['fullfilename'].isna()])))
                                                    
        # Test if we ask for new files to be downloaded (increment == True)
        if increment :
            
            # Keep only the xc recordings that have not been yet downloaded
            subdf_dataset = df_dataset[df_dataset['fullfilename'].isna()]
            
            # if limit in the number of files
            if (NUM_FILES is not None) and (len(subdf_dataset)>0):
               
                # number of audio files per species
                df_num = subdf_dataset.groupby(['gen','sp'], group_keys=False).apply(lambda x: len(x))
                
                # extract the combination (gen, sp) from df_groups
                keys = df_num.keys()
                
                # init the subdf_dataset
                df_to_dl = pd.DataFrame()
                
                # test if the number of files is greater than the maximum number of
                # resquested files per species
                # Do a stratified sampling in order to download the same number of files
                # for each species (combination of gen and sp).
                # If the number of files is lower than the requested number, take all the files
                for n, k in zip( df_num, keys):
                    mask = subdf_dataset[(subdf_dataset[['gen','sp']] == k)['gen']]
                    if n >= NUM_FILES :
                        df_to_dl = df_to_dl.append(mask.apply(lambda x: x.sample(n=NUM_FILES, 
                                                                                 random_state=random_seed)))
                    else:
                        df_to_dl = df_to_dl.append(mask) 

            else :
                df_to_dl = subdf_dataset                                                         
            
            # download all the audio files into a directory with a subdirectory for each
            # species
            df_dl = maad.util.xc_download(df         =df_to_dl,
                                        rootdir      =rootdir,
                                        dataset_name =dataset_name,
                                        overwrite    =overwrite,
                                        save_csv     =False,
                                        verbose      =verbose)
            
            # add columns        
            df_dl['filename']  = df_dl['fullfilename'].apply(os.path.basename)
            df_dl['categories']= df_dl['fullfilename'].apply(lambda path : Path(path).parts[-2])
                    
            # update df_dataset with df_dl
            df_dataset.update(df_dl)
            
            # save the updated dataframe
            df_dataset.to_csv(csv_fullfilename, sep=";", index=True, index_label='id') 
            
            # output with only the new files that were just downloaded (not the older)
            if only_new == True :
                df = df_dl
            else:
                # keep the rows with audio already downloaded
                df = df_dataset[~df_dataset['fullfilename'].isna()]
        
        # if we do not want to increment dataset                 
        else:
            # Keep the xc recordings that have already been downloaded
            df = df_dataset[~df_dataset['fullfilename'].isna()]
            
            # if limit in the number of files
            if (NUM_FILES is not None) and (len(df_dataset)>0) :
               
                # number of audio files per species
                df_num = df_dataset.groupby(['gen','sp'], group_keys=False).apply(lambda x: len(x))
                
                # extract the combination (gen, sp) from df_groups
                keys = df_num.keys()
                
                # init the df
                df = pd.DataFrame()
                
                # test if the number of files is greater than the maximum number of
                # resquested files per species
                # Do a stratified sampling in order to download the same number of files
                # for each species (combination of gen and sp).
                # If the number of files is lower than the requested number, take all the files
                for n, k in zip( df_num, keys):
                    mask = df_dataset[(df_dataset[['gen','sp']] == k)['gen']]
                    if n >= NUM_FILES :
                        df = df.append(mask.apply(lambda x: x.sample(n=NUM_FILES, 
                                                                     random_state=random_seed)))
                    else:
                        df = df.append(mask)
            else:
                df = df_dataset
                
    return df, csv_fullfilename

#%%
def grab_audio_to_df (path, 
                      audio_format, 
                      verbose=False) :
    """
    
    columns_name :
        First column name corresponds to full path to the filename
        Second column name correspond to the filename alone without the extension
    """
    print("Grabbing audio")
    # create a dataframe with all recordings in the directory
    filelist = glob.glob(os.path.join(path,
                                      '**/*.'+audio_format), 
                         recursive=True)
    
    df_dataset = pd.DataFrame()
    for file in filelist:
        categories = Path(file).parts[-2]
        iden = Path(file).parts[-1]
            
        df_dataset = df_dataset.append({
                                      'fullfilename':file,
                                      'filename'    :Path(file).parts[-1],
                                      'categories'  :categories,
                                      'id'          :iden},
                                    ignore_index=True)
        
    # set id as index
    df_dataset.set_index('id', inplace = True)
        
    if verbose :
        if len(df_dataset)>0 :
            print('*******************************************************')
            print('number of files : %2.0f' % len(df_dataset))
            print('number of categories : %2.0f' % len(df_dataset.categories.unique()))
            print('unique categories : {}'.format(df_dataset['categories'].unique()))
            print('*******************************************************')
        else :
            print('No {} audio file was found in {}'.format(audio_format,
                                                            path))    
    
    return df_dataset

#%%
def change_path (dataset_csv,
                 old_path,
                 new_path,
                 column_name,
                 verbose = False,
                 ) :
    
    # Read the csv
    df = pd.read_csv(dataset_csv)

    try:
        df[column_name] = df[column_name].str.replace(old_path, new_path)
        done = True
    except:
        done = False
        if verbose:
            print("**WARNING*** : No {} column is present in the dataframe".format(column_name))
        return done
    
    # save the dataframe with the new paths
    df.to_csv(dataset_csv, index=False)
    
    if verbose:
        print(' DONE ')
        print("Current path is {}".format (old_path)) 
        print(">>> New path is {}".format (new_path))
        
    return done
