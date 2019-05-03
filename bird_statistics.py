import matplotlib.pyplot as plt
from statistics import mean 
import time

class StatTracker:
    def __init__(self):
        self.generations = []
        self.top_avgs = []
        self.avgs = []


    def add_generation_data(self, generation):
        self.generations.append(generation)
        self.avgs.append(mean(generation))
        generation.sort(reverse = True) 
        self.top_avgs.append(mean(generation[:10]))



    def plot_progress(self):
        plt.plot(self.avgs, label='all avg') 
        plt.plot(self.top_avgs, label='top10 avg') 
        print(self.top_avgs)
        plt.xlabel('generations') 
        # naming the y axis 
        plt.ylabel('fitness (seconds') 
        # giving a title to my graph 
        plt.title('Generational Fitness') 
        
        # show a legend on the plot 
        plt.legend()
        mgr = plt.get_current_fig_manager()
        mgr.window.geometry("+1200+200")
        plt.pause(0.05)
        time.sleep(0.1)


