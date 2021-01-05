import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    session_name = raw_input("Enter the EXACT session_name: ")
    ts_odom = pd.read_csv('../data/'+session_name+'/odometry.csv')['Timestamp']
    dx_odom = pd.read_csv('../data/'+session_name+'/odometry.csv')['x']
    dy_odom = pd.read_csv('../data/'+session_name+'/odometry.csv')['y']
    da_odom = pd.read_csv('../data/'+session_name+'/odometry.csv')['z']

    ts_img = pd.read_csv('../data/'+session_name+'/transforms.csv')['Timestamp']
    dx_img = pd.read_csv('../data/'+session_name+'/transforms.csv')['dx']
    dy_img = pd.read_csv('../data/'+session_name+'/transforms.csv')['dy']
    da_img = pd.read_csv('../data/'+session_name+'/transforms.csv')['da']

    plt.plot(ts_odom,dx_odom, label="x odom")
    plt.plot(ts_odom,dy_odom, label="y odom")
    plt.plot(ts_odom,da_odom, label="teta odom")

    plt.plot(ts_img, dx_img, label='x image')
    plt.plot(ts_img,dy_img, label='y image')
    plt.plot(ts_img,da_img, label='teta image')
    plt.legend()

    plt.show()



