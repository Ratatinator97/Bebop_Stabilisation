import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    session_name = input("Enter the EXACT session_name: ")
    ts_odom = pd.read_csv('../data/'+session_name+'/odometry.csv')['Tismestamp']
    dx_odom = pd.read_csv('../data/'+session_name+'/odometry.csv')['x']
    dy_odom = pd.read_csv('../data/'+session_name+'/odometry.csv')['y']
    da_odom = pd.read_csv('../data/'+session_name+'/odometry.csv')['z']

    ts_img = pd.read_csv('../data/'+session_name+'/transforms.csv')['Tismestamp']
    dx_img = pd.read_csv('../data/'+session_name+'/transforms.csv')['dx']
    dy_img = pd.read_csv('../data/'+session_name+'/transforms.csv')['dy']
    da_img = pd.read_csv('../data/'+session_name+'/transforms.csv')['da']

    plt.plot(ts_odom,dx_odom)
    plt.plot(ts_odom,dy_odom)
    plt.plot(ts_odom,da_odom)

    plt.show()



