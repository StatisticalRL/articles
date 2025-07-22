from statisticalrl_experiments.fullExperiment import runLargeMulticoreExperiment as xp
from statisticalrl_experiments.plotResults import plot_results_from_dump

#######################
# Import registered environments
import statisticalrl_environments.register as bW

#######################
# Import some learners
from statisticalrl_learners.Generic.Qlearning import Qlearning as ql
from statisticalrl_learners.MDPs_discrete.IMED_RL import IMEDRL as imedrl
from statisticalrl_learners.MDPs_discrete.PSRL import PSRL as psrl
from statisticalrl_learners.MDPs_discrete.UCRL3 import UCRL3_lazy as ucrl3
import statisticalrl_learners.MDPs_discrete.Optimal.OptimalControl  as opt



#######################
def run_experiment(env_name,timeHorizon=10000000,nbReplicates=50):
    # Instantiate one environment
    env = bW.make(env_name)
    nS = env.observation_space.n
    nA = env.action_space.n
    delta= 0.05
    #    Define a few learners to be compared:
    agents= []
    agents += [[ql, {"nS":nS, "nA":nA}]]
    agents += [[imedrl, {"nbr_states":nS, "nbr_actions":nA}]]
    agents += [[psrl, {"nS":nS, "nA":nA,"delta":delta}]]
    agents += [[ucrl3, {"nS":nS, "nA":nA,"delta":delta}]]
    #############################
    # Compute oracle policy:
    oracle = opt.build_opti(env.name, env, env.observation_space.n, env.action_space.n)

    #######################
    # Run a full experiment
    #######################
    print("-"*30+"\n Running Experiment for environment " + env.name +"\n"+"-"*30)
    xp(env, agents, oracle, timeHorizon=timeHorizon, nbReplicates=nbReplicates,root_folder="results/")


#######################

envs= ['river-swim-6','ergo-river-swim-6','grid-4-room','river-swim-25','ergo-river-swim-25','grid-2-room','grid-random-88','grid-random-1616','random-rich','nasty']
def preparerun_XpsArticle():
    print("-"*30+"\n Testing all experiments. This may take a few minutes...\n"+"-"*30)
    for env in envs:
        run_experiment(env, timeHorizon=1500,nbReplicates=6)
def run_XpsArticle():
    print("-"*30+"\n Runnning all experiments. This may take a few hours...\n"+"-"*30)
    for env in envs:
        run_experiment(env)

#######################
# Assuming data has been generated in previous runs, dumped in results/ in following files:
#cumRegret_Gridworld-4-room-v0_UCRL2_10000000
#cumRegret_Gridworld-4-room-v0_KL-UCRL_10000000
#cumRegret_Gridworld-4-room-v0_UCRL2B_10000000
#cumRegret_Gridworld-4-room-v0_PSRL_10000000
#cumRegret_Gridworld-4-room-v0_UCRL3_10000000
#Choose tplot<=tmax=10000000
#######################
def plot_XpsArticle():
    tmax=10000000

    plot_results_from_dump('river-swim-6',tplot=tmax/10)
    for env in envs[1:]:
        plot_results_from_dump(env,tplot=tmax)

#bW.print_envlist()
preparerun_XpsArticle()
#run_XpsArticle()
#plot_XpsArticle()