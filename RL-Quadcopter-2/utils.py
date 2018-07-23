import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def plot_run(results, standalone=True):
    if standalone:
        plt.subplots(figsize=(15, 15))

    plt.subplot(3, 3, 1)
    plt.title('Position Graph')
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.xlabel('time, seconds')
    plt.ylabel('Position')
    plt.grid(True)
    if standalone:
        plt.legend()

    plt.subplot(3, 3, 2)
    plt.title('Velocity Graph')
    plt.plot(results['time'], results['x_velocity'], label='x')
    plt.plot(results['time'], results['y_velocity'], label='y')
    plt.plot(results['time'], results['z_velocity'], label='z')
    plt.xlabel('time, seconds')
    plt.ylabel('Velocity')
    plt.grid(True)
    if standalone:
        plt.legend()

    plt.subplot(3, 3, 3)
    plt.title('Orientation Graph')
    plt.plot(results['time'], results['phi'], label='phi')
    plt.plot(results['time'], results['theta'], label='theta')
    plt.plot(results['time'], results['psi'], label='psi')
    plt.xlabel('time, seconds')
    plt.grid(True)
    if standalone:
        plt.legend()

    plt.subplot(3, 3, 4)
    plt.title('Angular Velocity Graph')
    plt.plot(results['time'], results['phi_velocity'], label='phi')
    plt.plot(results['time'], results['theta_velocity'], label='theta')
    plt.plot(results['time'], results['psi_velocity'], label='psi')
    plt.xlabel('time, seconds')
    plt.grid(True)
    if standalone:
        plt.legend()

    plt.subplot(3, 3, 5)
    plt.title('Rotor Speed Graph')
    plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1')
    plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2')
    plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3')
    plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4')
    plt.xlabel('time, seconds')
    plt.ylabel('Rotor Speed, revolutions / second')
    plt.grid(True)
    if standalone:
        plt.legend()

    plt.subplot(3, 3, 6)
    plt.title('Reward Graph')
    plt.plot(results['time'], results['reward'], label='Reward')
    plt.xlabel('time, seconds')
    plt.ylabel('Reward')
    if standalone:
        plt.legend(loc=3)
    ax2 = plt.twinx()
    ax2.plot(results['time'], np.cumsum(results['reward']), color='xkcd:red', label='Accum. Reward')
    ax2.set_ylabel('Accumulated Reward')
    if standalone:
        ax2.legend(loc=4)
    plt.grid(True)

    if standalone:
        plt.tight_layout()
        plt.show()

class SetupPlot():
    def __init__(self, fileName,task):
        # Setup
        self.labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
                'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
                'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4','reward']
        self.results = {x : [] for x in self.labels}
        self.task=task
        self.fileName=fileName
        # Run the simulation, and save the results.
        with open(self.fileName, 'w') as csvfile:
            self.writer = csv.writer(csvfile)
            self.writer.writerow(self.labels)
    def writeResults(self,reward,action):
        action=list(action)*4
        with open(self.fileName, 'a') as csvfile:
            self.writer = csv.writer(csvfile)
            to_write = [self.task.sim.time] + list(self.task.sim.pose) + list(self.task.sim.v) + list(self.task.sim.angular_v) + list(action)+[reward]
            for ii in range(len(self.labels)):
                self.results[self.labels[ii]].append(to_write[ii])
            self.writer.writerow(to_write)
            return self.results