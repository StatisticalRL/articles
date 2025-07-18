
'''
    Companion code for the article
    "Logarithmic regret in communicating MDPs: Leveraging known dynamics with bandits"
    H. Saber, F. Pesquerel, O-A. Maillard, M. S. Talebi
    Asian Conference of Machine Learning, 2023.
'''
from statisticalrl_experiments.fullExperiment import runLargeMulticoreExperiment as xp
from statisticalrl_experiments.plotResults import plot_results_from_dump

#######################
# Import registered environments
import statisticalrl_environments.register as bW

#######################
# Import some learners
from statisticalrl_learners.MDPs_discrete.UCRL2 import UCRL2 as ucrl2
from statisticalrl_learners.MDPs_discrete.UCRL2B import UCRL2B as ucrl2b
from statisticalrl_learners.MDPs_discrete.KLUCRL import KLUCRL as klucrl
from statisticalrl_learners.MDPs_discrete.PSRL import PSRL as psrl
from statisticalrl_learners.MDPs_discrete.UCRL3 import UCRL3_lazy as ucrl3
import statisticalrl_learners.MDPs_discrete.Optimal.OptimalControl  as opt

from statisticalrl_learners.Generic.Qlearning import Qlearning as ql
from statisticalrl_learners.MDPs_discrete.KnownDynamics.IMEDKTP import IMEDKTP as imedktp
from statisticalrl_learners.MDPs_discrete.KnownDynamics.UCB_KD import UCBKD as  ucb_KD
from statisticalrl_learners.MDPs_discrete.KnownDynamics.TS_KD import TSKD as ts_KD




#######################
def run_experiment(env_name,learner_names,timeHorizon=10000000,nbReplicates=50):
    # Instantiate one environment
    env = bW.make(env_name)
    nS = env.observation_space.n
    nA = env.action_space.n
    #    Define a few learners to be compared:
    agents= [[ql, {"nS":nS, "nA":nA}]]+ [ [agent, {"env": env}] for agent in learner_names]

    #############################
    # Compute oracle policy:
    oracle = opt.build_opti(env.name, env, env.observation_space.n, env.action_space.n)

    #######################
    # Run a full experiment
    #######################
    print("-"*30+"\n Running Experiment for environment " + env.name +"\n"+"-"*30)
    xp(env, agents, oracle, timeHorizon=timeHorizon, nbReplicates=nbReplicates,root_folder="results/")

#######################

envs= ['ergo-river-swim-6','grid-4-room','grid-2-room','nasty','ergo-river-swim-25']
learner_names = [imedktp, ucb_KD, ts_KD]
def preparerun_XpsArticle():
    print("-"*30+"\n Testing all experiments. This may take a few minutes...\n"+"-"*30)
    for env in envs:
        run_experiment(env, learner_names,timeHorizon=2000,nbReplicates=10)
def run_XpsArticle():
    print("-"*30+"\n Runnning all experiments. This may take a few hours...\n"+"-"*30)
    for env in envs:
        run_experiment(env, learner_names)

#######################
# Assuming data has been generated in previous runs, dumped in results/ in following files:
#cumRegret_Gridworld-4-room-v0_IMED-KD_10000000
#cumRegret_Gridworld-4-room-v0_Q-learning_10000000
#Choose tplot<=tmax=10000000
#######################
def plot_XpsArticle():
    tmax=10000000

    plot_results_from_dump('river-swim-6',tplot=tmax/10)
    for env in envs[1:]:
        plot_results_from_dump(env,tplot=tmax)

preparerun_XpsArticle()
#run_XpsArticle()
#plot_XpsArticle()