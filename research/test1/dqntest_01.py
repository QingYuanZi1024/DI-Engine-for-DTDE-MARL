import gym # 载入 gym 库，用于标准化强化学习环境
import torch # 载入 PyTorch 库，用于加载 Tensor 模型，定义计算网络
from easydict import EasyDict # 载入 EasyDict，用于实例化配置文件
from ding.config import compile_config # 载入DI-engine config 中配置相关组件
from ding.envs import DingEnvWrapper # 载入DI-engine env 中环境相关组件
from ding.policy import DQNPolicy, single_env_forward_wrapper # 载入DI-engine policy 中策略相关组件
from ding.model import DQN # 载入DI-engine model 中模型相关组件
from dizoo.box2d.lunarlander.config.lunarlander_dqn_config import main_config, create_config # 载入DI-zoo lunarlander 环境与 DQN 算法相关配置


def main(main_config: EasyDict, create_config: EasyDict, ckpt_path: str):
    main_config.exp_name = 'lunarlander_dqn_deploy' # 设置本次部署运行的实验名，即为将要创建的工程文件夹名
    cfg = compile_config(main_config, create_cfg=create_config, auto=True) # 编译生成所有的配置
    env = DingEnvWrapper(gym.make(cfg.env.env_id), EasyDict(env_wrapper='default')) # 在gym的环境实例的基础上添加DI-engine的环境装饰器
    env.enable_save_replay(replay_path='./lunarlander_dqn_deploy/video') # 开启环境的视频录制，设置视频存放位置
    model = DQN(**cfg.policy.model) # 导入模型配置，实例化DQN模型
    state_dict = torch.load(ckpt_path, map_location='cpu') # 从模型文件加载模型参数
    model.load_state_dict(state_dict['model']) # 将模型参数载入模型
    policy = DQNPolicy(cfg.policy, model=model).eval_mode # 导入策略配置，导入模型，实例化DQN策略，并选择评价模式
    forward_fn = single_env_forward_wrapper(policy.forward) # 使用简单环境的策略装饰器，装饰DQN策略的决策方法
    obs = env.reset() # 重置初始化环境，获得初始观测
    returns = 0. # 初始化总奖励
    while True: # 让智能体的策略与环境，循环交互直到结束
        action = forward_fn(obs) # 根据观测状态，决定决策动作
        obs, rew, done, info = env.step(action) # 执行决策动作，与环境交互，获得下一步的观测状态，此次交互的奖励，是否结束的信号，以及其它环境信息
        returns += rew # 累计奖励回报
        if done:
            break
    print(f'Deploy is finished, final epsiode return is: {returns}')

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(main_config=main_config, create_config=create_config, ckpt_path='./Downloads/final.pth.tar')