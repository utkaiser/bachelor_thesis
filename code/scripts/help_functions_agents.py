import logging
import importlib

def get_agent_by_name(agent_name, trading_env, decay_rate=None,layer_size=None,gamma=None,memory_size=None,dropout_rate=None, learning_rate=None):

    '''
        Goal: initializes selected agent

        Input:
        - agent_name: name of agent to initialize

        Output:
        - agent: initialized agent, if agent not found, raise Exception
    '''

    try:
        Agent_class = getattr(importlib.import_module('asset_trading_agents.repository_1.'+agent_name),"Agent")
    except:
        raise ValueError("Agent " + agent_name + " not found! Please give correct name of agent.")

    if agent_name == "improved_q_learning_agent":
        Agent = Agent_class(trading_env,decay_rate,layer_size,gamma,memory_size,dropout_rate, learning_rate)
    else:
        Agent = Agent_class(trading_env, decay_rate, layer_size, gamma, memory_size)
    logging.info("Initialized " + Agent.name)
    return Agent



