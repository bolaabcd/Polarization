from math import ceil
import cli_utils as cli
from matplotlib import pyplot as plt
from Simulation import MAX_TIME, Simulation
from belief_states import build_belief
from influence_graphs import build_influence,Influence
from update_functions import Update_Functions
from rationality_graphs import build_rationality
 
class Raw_Sim:
    def __init__(self,sim,blf,inf,fun,rat):
        self.sim=sim
        self.blf=blf
        self.inf=inf
        self.fun=fun
        self.rat = rat
    

class ManySimulations:
    def __init__(self,update_keys,belief_keys,influence_keys,rationality_keys,cmax_time=MAX_TIME,smart_stop=False,up_funs=Update_Functions(), num_agents=100):
        self.update_keys      = update_keys
        self.belief_keys      = belief_keys
        self.influence_keys   = influence_keys
        self.rationality_keys = rationality_keys

        self.cmax_time=cmax_time
        self.smart_stop=smart_stop

        self.up_funs=up_funs
        self.completed_sims=None
        self.num_agents=num_agents
    def run(self):
        raw_simulations = [Raw_Sim(
                sim=Simulation(build_belief(blf, self.num_agents), build_influence(inf, self.num_agents),build_rationality(rat), self.up_funs.get_function(upFun)),
                inf=inf,
                blf=blf,
                fun=upFun,
                rat=rat,
            ) 
            for inf in self.influence_keys 
            for blf in self.belief_keys
            for upFun in self.update_keys
            for rat in self.rationality_keys
        ]
        for i in cli.ProgressRange(len(raw_simulations), "running"):
            if raw_simulations[i].inf is Influence.CIRCULAR:
                raw_simulations[i].sim = raw_simulations[i].sim.run(max_time=self.cmax_time)
            else:
                raw_simulations[i].sim = raw_simulations[i].sim.run(smart_stop=self.smart_stop)
        self.completed_sims= raw_simulations
    
    def plot_polarization(self,saveto=None):
        nlines=ceil(len(self.completed_sims)/(2.0*len(self.update_keys)))
        ncols=2
        plt.rcParams.update({'font.size': 16})
        fig=plt.figure(num=1, figsize=(18*ncols,10*nlines))
        plt.subplots_adjust(hspace=0.3)
        amt_plotted=len(self.update_keys)
        wich_plot=1
        for i,sim in enumerate(self.completed_sims):
            if i%amt_plotted == 0:
                plt.subplot(nlines,ncols,wich_plot)
                wich_plot+=1
            
            plt.plot(sim.sim[0][:self.cmax_time],label=str(sim.fun)+" "+str(sim.rat))
            
            if i%amt_plotted==amt_plotted-1:
                plt.title(str(sim.blf)+" "+str(sim.inf))
                plt.legend(loc="upper right")            
        if saveto !=None:
            fig.savefig(saveto,bbox_inches='tight')
        plt.close()
        
    def plot_agents(self,saveto=None):
        nlines=ceil(len(self.completed_sims)/2.0)
        ncols=2
        plt.rcParams.update({'font.size': 16})
        fig=plt.figure(num=1, figsize=(18*ncols,10*nlines))
        plt.subplots_adjust(hspace=0.3)

        for i,sim in enumerate(self.completed_sims):
            plt.subplot(nlines,ncols,i+1)                
            plt.plot(sim.sim[1][:self.cmax_time])
            plt.title(str(sim.blf)+" "+str(sim.inf)+" "+str(sim.fun)+" "+str(sim.rat))
        if saveto !=None:
            fig.savefig(saveto,bbox_inches='tight')
        plt.close()

