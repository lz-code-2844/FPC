import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import json

def read_ti(fn):
    data = []
    with open(fn) as f:
        file = f.readlines()
        for line in file:
            line = line.strip()
            dd = line.split()
            temp = list(dd)
            data.append(temp)
    data = np.delete(data, 0, axis=0)
    data = np.array(data, dtype='float64')
    return data

def norm_data(datapath, file_list=None):

    min_val = np.array([1e20,])
    max_val = np.array([-1e20,])
    num_data = 0
    num_points = []
    
    if file_list is None:
        f_list = os.listdir(datapath)
        f_list.sort(key=lambda x: int(x[:-4]))
    else:
        f_list = file_list

    for f in f_list:
        if f[-3:] == 'txt':
            data = read_ti(os.path.join(datapath, f))
            if min_val.shape[0] == 1:
                min_val = min_val.repeat(data.shape[1])
                max_val = max_val.repeat(data.shape[1])
            min_val = np.min([min_val, data.min(axis=0)], axis=0)
            max_val = np.max([max_val, data.max(axis=0)], axis=0)
            num_data += 1
            num_points.append(data.shape[0])
    return min_val, max_val, num_data, num_points

def split_and_save_data(input_path, pkl_output_dir, train_ratio=0.9, random_seed=42):

    # 获取所有txt文件
    all_files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
    all_files.sort(key=lambda x: int(x[:-4]))

    # 分割训练集和验证集
    train_files, val_files = train_test_split(
        all_files,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True
    )

    train_dir = os.path.join(input_path, 'train')
    val_dir = os.path.join(input_path, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_new_files = []
    val_new_files = []
    train_mapping = []
    val_mapping = []
    
    for i, f in enumerate(train_files):
        new_name = f"{i}.txt"
        src = os.path.join(input_path, f)
        dst = os.path.join(train_dir, new_name)
        shutil.copy(src, dst)
        train_new_files.append(new_name)
        train_mapping.append({"original_name": f, "new_name": new_name})
    
    for i, f in enumerate(val_files):
        new_name = f"{i}.txt"
        src = os.path.join(input_path, f)
        dst = os.path.join(val_dir, new_name)
        shutil.copy(src, dst)
        val_new_files.append(new_name)
        val_mapping.append({"original_name": f, "new_name": new_name})


    train_min_val, train_max_val, train_num_data, train_num_points = norm_data(train_dir, train_new_files)
    train_data_info = {
        'input_min': train_min_val,
        'input_max': train_max_val,
        'num_data': train_num_data,
        'num_points': train_num_points,
    }

    val_min_val, val_max_val, val_num_data, val_num_points = norm_data(val_dir, val_new_files)
    val_data_info = {
        'input_min': val_min_val,
        'input_max': val_max_val,
        'num_data': val_num_data,
        'num_points': val_num_points,
    }

    os.makedirs(pkl_output_dir, exist_ok=True)
    train_pkl_path = os.path.join(pkl_output_dir, 'train_data.pkl')
    val_pkl_path = os.path.join(pkl_output_dir, 'val_data.pkl')

    pickle.dump(train_data_info, open(train_pkl_path, 'wb'))
    pickle.dump(val_data_info, open(val_pkl_path, 'wb'))

    train_df = pd.DataFrame([list(range(len(train_files))), train_num_points]).T
    val_df = pd.DataFrame([list(range(len(val_files))), val_num_points]).T
    
    train_csv_path = os.path.join(os.path.dirname(pkl_output_dir), 'outputs/train_out.csv')
    val_csv_path = os.path.join(os.path.dirname(pkl_output_dir), 'outputs/val_out.csv')
    
    os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)
    train_df.to_csv(train_csv_path, header=False, index=False)
    val_df.to_csv(val_csv_path, header=False, index=False)

    train_map_path = os.path.join(os.path.dirname(pkl_output_dir), 'outputs/train_mapping.json')
    val_map_path = os.path.join(os.path.dirname(pkl_output_dir), 'outputs/val_mapping.json')
    with open(train_map_path, 'w', encoding='utf-8') as f:
        json.dump(train_mapping, f, ensure_ascii=False, indent=2)
    with open(val_map_path, 'w', encoding='utf-8') as f:
        json.dump(val_mapping, f, ensure_ascii=False, indent=2)

    print(f"Training files copied and renamed to: {train_dir}")
    print(f"Validation files copied and renamed to: {val_dir}")
    print(f"Training data info saved to: {train_pkl_path}")
    print(f"Validation data info saved to: {val_pkl_path}")
    print(f"Training CSV saved to: {train_csv_path}")
    print(f"Validation CSV saved to: {val_csv_path}")
    print(f"Training mapping saved to: {train_map_path}")
    print(f"Validation mapping saved to: {val_map_path}")
    print(f"Train samples: {train_num_data}, Val samples: {val_num_data}")

if __name__ == "__main__":
    input_path = '/data/dataserver01/public/zli/workspaces/pinn/FlowPredictor-Corrector_task2_4096/pv_txt/'
    pkl_output_dir = '/data/dataserver01/public/zli/workspaces/pinn/FlowPredictor-Corrector_task2_4096/pv_pkl/'

    split_and_save_data(
        input_path=input_path,
        pkl_output_dir=pkl_output_dir,
        train_ratio=0.8,  
        random_seed=42    
    )

    train_data_info = pickle.load(open(os.path.join(pkl_output_dir, 'train_data.pkl'), 'rb'))
    val_data_info = pickle.load(open(os.path.join(pkl_output_dir, 'val_data.pkl'), 'rb'))
    print("\nTrain data info:", train_data_info)
    print("Val data info:", val_data_info)
