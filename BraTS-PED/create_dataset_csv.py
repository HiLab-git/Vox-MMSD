"""
split the patients into training, validation and testing
"""
import os
import csv 
import random
import shutil 
import pandas as pd
import numpy as np 
import SimpleITK as sitk 
from PIL import Image 
from skimage import exposure


def get_2d_crop_bounding_box(mask, margin = 5):
    D, H, W = mask.shape
    ds, hs, ws = np.where(mask > 0)
    hmin = max(hs.min() - margin, 0)
    hmax = min(hs.max() + margin, H)
    wmin = max(ws.min() - margin, 0)
    wmax = min(ws.max() + margin, W)
    return hmin, hmax, wmin, wmax

def get_3d_crop_bounding_box(mask, margin = 5):
    D, H, W = mask.shape
    ds, hs, ws = np.where(mask > 0)
    dmin = max(ds.min() , 0)
    dmax = min(ds.max() , D)
    hmin = max(hs.min() - margin, 0)
    hmax = min(hs.max() + margin, H)
    wmin = max(ws.min() - margin, 0)
    wmax = min(ws.max() + margin, W)
    return dmin, dmax, hmin, hmax, wmin, wmax

def preprocess():
    data_dir='../BraTS_PED/data/source_data'
    pnames    =  os.listdir(data_dir)
    for pid in pnames:
        t1c_name = pid + "/" + pid + "-t1c.nii.gz"
        seg_name = pid + "/" + pid + "-seg.nii.gz"
        t1c_obj = sitk.ReadImage(data_dir + '/' + t1c_name)
        seg_obj = sitk.ReadImage(data_dir + '/' + seg_name)
        t1c = sitk.GetArrayFromImage(t1c_obj)
        seg = sitk.GetArrayFromImage(seg_obj)
        dmin, dmax, hmin, hmax, wmin, wmax = get_3d_crop_bounding_box(t1c > 0)
            
    
        for mod in ["t1n", "t1c", "t2w", "t2f", "seg"]:
            img_name = pid + "/" + pid + "-{0:}.nii.gz".format(mod)
            img_obj = sitk.ReadImage(data_dir + '/' + img_name)
            img = sitk.GetArrayFromImage(img_obj)
            img = img[dmin:dmax, hmin:hmax, wmin:wmax]
            if mod != "seg":
                img_p99 = np.percentile(img, 99.5)
                img = img / img_p99 * 255
                img[img > 255] = 255
                img = np.asarray(img, np.uint8)
            
            img=sitk.GetImageFromArray(img)
            sitk.WriteImage(img,'../BraTS_PED/data/preprocessed_data/'+ "{0:}-{1:}.nii.gz".format(pid, mod))
            

def create_dataset_csv():
    data_dir='../BraTS_PED/data/source_data'
    preprocessed_data_dir='../BraTS_PED/data/preprocessed_data'
    pnames    =  os.listdir(data_dir)
    t1ns=[]
    t1cs=[]
    t2ws=[]
    t2fs=[]
    segs=[]
    for pid in pnames:
        t1ns.append(pid + '-t1n.nii.gz')
        t1cs.append(pid + '-t1c.nii.gz')
        t2ws.append(pid + '-t2w.nii.gz')
        t2fs.append(pid + '-t2f.nii.gz')
        segs.append(pid + '-seg.nii.gz')
    
    df_dict = {"t1n": t1ns, "t1c":t1cs, "t2w":t2ws, "t2f":t2fs, "label": segs}
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv("./data/data_patient_level.csv", index = False)

    df = df.sample(frac = 1, random_state=2023)
    n_train = int(len(pnames) * 0.7)
    n_valid = int(len(pnames) * 0.1)     

    df_train = df.iloc[:n_train:]
    df_valid = df.iloc[n_train: n_train + n_valid, :]
    df_test  = df.iloc[n_train + n_valid::]

    #构建测试集
    df_train =df_train.sort_values(by=['t1n'])
    df_train.to_csv("./data/train.csv", index = False)
    df_test  = df_test.sort_values(by = ['t1n'])
    df_test.to_csv("./data/test.csv", index = False)
    df_valid =df_valid.sort_values(by= ['t1n'])
    df_valid.to_csv("./data/valid.csv", index = False)


if __name__ == "__main__":
    preprocess()
    create_dataset_csv()