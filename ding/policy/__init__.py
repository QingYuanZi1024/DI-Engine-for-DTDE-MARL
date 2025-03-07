from .base_policy import Policy, CommandModePolicy, create_policy, get_policy_cls
from .common_utils import single_env_forward_wrapper, single_env_forward_wrapper_ttorch, default_preprocess_learn
from .dqn import DQNSTDIMPolicy, DQNPolicy
from .mdqn import MDQNPolicy
from .iqn import IQNPolicy
from .fqf import FQFPolicy
from .qrdqn import QRDQNPolicy
from .c51 import C51Policy
from .rainbow import RainbowDQNPolicy
from .ddpg import DDPGPolicy
from .d4pg import D4PGPolicy
from .td3 import TD3Policy
from .td3_vae import TD3VAEPolicy
from .td3_bc import TD3BCPolicy
from .dt import DTPolicy

from .pg import PGPolicy
from .a2c import A2CPolicy
from .ppo import PPOPolicy, PPOPGPolicy, PPOOffPolicy
from .sac import SACPolicy, DiscreteSACPolicy, SQILSACPolicy
from .cql import CQLPolicy, DiscreteCQLPolicy
from .edac import EDACPolicy
from .impala import IMPALAPolicy
from .ngu import NGUPolicy
from .r2d2 import R2D2Policy
from .r2d2_gtrxl import R2D2GTrXLPolicy
from .ppg import PPGPolicy, PPGOffPolicy
from .sqn import SQNPolicy
from .bdq import BDQPolicy

from .qmix import QMIXPolicy
from .wqmix import WQMIXPolicy
from .coma import COMAPolicy
from .collaq import CollaQPolicy
from .atoc import ATOCPolicy
from .acer import ACERPolicy
from .qtran import QTRANPolicy
from .ciallo import CIALLOPolicy

from .il import ILPolicy

from .r2d3 import R2D3Policy

from .command_mode_policy_instance import *

from .policy_factory import PolicyFactory, get_random_policy
from .pdqn import PDQNPolicy

from .bc import BehaviourCloningPolicy
from .ibc import IBCPolicy

from .pc import ProcedureCloningBFSPolicy

from .bcq import BCQPolicy

# new-type policy
from .ppof import PPOFPolicy
from .prompt_pg import PromptPGPolicy
