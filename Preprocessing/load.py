import os
import numpy as np
import pandas as pd

def load_selected(path):
    """
    读取专家筛选的数据和标签
    """
    df_cu = pd.read_excel(path,sheet_name='Cu',header=0)
    df_zn = pd.read_excel(path,sheet_name='Zn',header=0)
    df_pb = pd.read_excel(path,sheet_name='Pb',header=0)
    df_v = pd.read_excel(path,sheet_name='V',header=0)

    cu_data = df_cu.iloc[:,:-2].values
    cu_label = df_cu.iloc[:,-1:].values

    zn_data = df_zn.iloc[:,:-2].values
    zn_label = df_zn.iloc[:,-1:].values

    pb_data = df_pb.iloc[:,:-2].values
    pb_label = df_pb.iloc[:,-1:].values

    v_data = df_v.iloc[:,:-2].values
    v_label = df_v.iloc[:,-1:].values

    return cu_data, cu_label, zn_data, zn_label, pb_data, pb_label, v_data, v_label

def load_full(path):
    """
    读取全光谱数据和标签
    """
    df_data = pd.read_excel(path,sheet_name='Sheet1',header=None)
    df_label = pd.read_excel(path,sheet_name='Sheet2',header=None)

    # 全光谱数据是sheet1的第1列到第801列
    data = df_data.iloc[:,1:801].values

    # Cu标签是sheet2的第0列
    cu_label = df_label.iloc[:,0].values.reshape(-1,1)
    # Zn标签是sheet2的第1列
    zn_label = df_label.iloc[:,1].values.reshape(-1,1)
    # Pb标签是sheet2的第2列
    pb_label = df_label.iloc[:,2].values.reshape(-1,1)
    # V标签是sheet2的第3列
    v_label = df_label.iloc[:,3].values.reshape(-1,1)
    return data,cu_label, zn_label, pb_label, v_label



