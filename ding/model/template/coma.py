from typing import Dict, Union
import torch
import torch.nn as nn

from functools import reduce
from ding.torch_utils import one_hot, MLP
from ding.utils import squeeze, list_split, MODEL_REGISTRY, SequenceType
from .q_learning import DRQN


class COMAActorNetwork(nn.Module):
    """
    Overview:
        Decentralized actor network in COMA algorithm.
    Interface:
         ``__init__``, ``forward``
    """

    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        hidden_size_list: SequenceType = [128, 128, 64],
    ):
        """
        Overview:
            Initialize COMA actor network
        Arguments:
            - obs_shape (:obj:`int`): the dimension of each agent's observation state
            - action_shape (:obj:`int`): the dimension of action shape
            - hidden_size_list (:obj:`list`): the list of hidden size, default to [128, 128, 64]
        """
        super(COMAActorNetwork, self).__init__()
        self.main = DRQN(obs_shape, action_shape, hidden_size_list)

    def forward(self, inputs: Dict) -> Dict:
        """
        Overview:
            The forward computation graph of COMA actor network
        Arguments:
            - inputs (:obj:`dict`): input data dict with keys ['obs', 'prev_state']
            - agent_state (:obj:`torch.Tensor`): each agent local state(obs)
            - action_mask (:obj:`torch.Tensor`): the masked action
            - prev_state (:obj:`torch.Tensor`): the previous hidden state
        Returns:
            - output (:obj:`dict`): output data dict with keys ['logit', 'next_state', 'action_mask']
        ArgumentsKeys:
            - necessary: ``obs`` { ``agent_state``, ``action_mask`` }, ``prev_state``
        ReturnsKeys:
            - necessary: ``logit``, ``next_state``, ``action_mask``
        Examples:
            >>> T, B, A, N = 4, 8, 3, 32
            >>> embedding_dim = 64
            >>> action_dim = 6
            >>> data = torch.randn(T, B, A, N)
            >>> model = COMAActorNetwork((N, ), action_dim, [128, embedding_dim])
            >>> prev_state = [[None for _ in range(A)] for _ in range(B)]
            >>> for t in range(T):
            >>>     inputs = {'obs': {'agent_state': data[t], 'action_mask': None}, 'prev_state': prev_state}
            >>>     outputs = model(inputs)
            >>>     logit, prev_state = outputs['logit'], outputs['next_state']
        """
        agent_state = inputs['obs']['agent_state']
        prev_state = inputs['prev_state']
        if len(agent_state.shape) == 3:  # B, A, N
            agent_state = agent_state.unsqueeze(0)
            unsqueeze_flag = True
        else:
            unsqueeze_flag = False
        T, B, A = agent_state.shape[:3]
        agent_state = agent_state.reshape(T, -1, *agent_state.shape[3:])
        prev_state = reduce(lambda x, y: x + y, prev_state)
        output = self.main({'obs': agent_state, 'prev_state': prev_state, 'enable_fast_timestep': True})
        logit, next_state = output['logit'], output['next_state']
        next_state, _ = list_split(next_state, step=A)
        logit = logit.reshape(T, B, A, -1)
        if unsqueeze_flag:
            logit = logit.squeeze(0)
        return {'logit': logit, 'next_state': next_state, 'action_mask': inputs['obs']['action_mask']}


class COMACriticNetwork(nn.Module):
    """
    Overview:
        Centralized critic network in COMA algorithm.
    Interface:
         ``__init__``, ``forward``
    """

    def __init__(
        self,
        input_size: int,
        action_shape: int,
        hidden_size: int = 128,
    ):
        """
        Overview:
            initialize COMA critic network
        Arguments:
            - input_size (:obj:`int`): the size of input global observation
            - action_shape (:obj:`int`): the dimension of action shape
            - hidden_size_list (:obj:`list`): the list of hidden size, default to 128
        Returns:
            - output (:obj:`dict`): output data dict with keys ['q_value']
        Shapes:
            - obs (:obj:`dict`): ``agent_state``: :math:`(T, B, A, N, D)`, ``action_mask``: :math:`(T, B, A, N, A)`
            - prev_state (:obj:`list`): :math:`[[[h, c] for _ in range(A)] for _ in range(B)]`
            - logit (:obj:`torch.Tensor`): :math:`(T, B, A, N, A)`
            - next_state (:obj:`list`): :math:`[[[h, c] for _ in range(A)] for _ in range(B)]`
            - action_mask (:obj:`torch.Tensor`): :math:`(T, B, A, N, A)`
        """
        super(COMACriticNetwork, self).__init__()
        self.action_shape = action_shape
        self.act = nn.ReLU()
        self.mlp = nn.Sequential(
            MLP(input_size, hidden_size, hidden_size, 2, activation=self.act), nn.Linear(hidden_size, action_shape)
        )

    def forward(self, data: Dict) -> Dict:
        """
        Overview:
            forward computation graph of qmix network
        Arguments:
            - data (:obj:`dict`): input data dict with keys ['obs', 'prev_state', 'action']
            - gent_state (:obj:`torch.Tensor`): each agent local state(obs)a
            - global_state (:obj:`torch.Tensor`): global state(obs)
            - action (:obj:`torch.Tensor`): the masked action
        ArgumentsKeys:
            - necessary: ``obs`` { ``agent_state``, ``global_state`` }, ``action``, ``prev_state``
        ReturnsKeys:
            - necessary: ``q_value``
        Examples:
            >>> agent_num, bs, T = 4, 3, 8
            >>> obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
            >>> coma_model = COMACriticNetwork(
            >>>     obs_dim - action_dim + global_obs_dim + 2 * action_dim * agent_num, action_dim)
            >>> data = {
            >>>     'obs': {
            >>>         'agent_state': torch.randn(T, bs, agent_num, obs_dim),
            >>>         'global_state': torch.randn(T, bs, global_obs_dim),
            >>>     },
            >>>     'action': torch.randint(0, action_dim, size=(T, bs, agent_num)),
            >>> }
            >>> output = coma_model(data)
        """
        x = self._preprocess_data(data)
        q = self.mlp(x)
        return {'q_value': q}

    def _preprocess_data(self, data: Dict) -> torch.Tensor:
        """
        Overview:
            preprocess data to make it can be used by MLP net
        Arguments:
            - data (:obj:`dict`): input data dict with keys ['obs', 'prev_state', 'action']
            - agent_state (:obj:`torch.Tensor`): each agent local state(obs)
            - global_state (:obj:`torch.Tensor`): global state(obs)
            - action (:obj:`torch.Tensor`): the masked action
        ArgumentsKeys:
            - necessary: ``obs`` { ``agent_state``, ``global_state``} , ``action``, ``prev_state``
        Return:
            - x (:obj:`torch.Tensor`): the data can be used by MLP net, including \
                ``global_state``, ``agent_state``, ``last_action``, ``action``, ``agent_id``
        """
        t_size, batch_size, agent_num = data['obs']['agent_state'].shape[:3]
        agent_state_ori, global_state = data['obs']['agent_state'], data['obs']['global_state']

        # splite obs, last_action and agent_id
        agent_state = agent_state_ori[..., :-self.action_shape - agent_num]
        last_action = agent_state_ori[..., -self.action_shape - agent_num:-agent_num]
        last_action = last_action.reshape(t_size, batch_size, 1, -1).repeat(1, 1, agent_num, 1)
        agent_id = agent_state_ori[..., -agent_num:]

        action = one_hot(data['action'], self.action_shape)  # T, B, A，N
        action = action.reshape(t_size, batch_size, -1, agent_num * self.action_shape).repeat(1, 1, agent_num, 1)
        action_mask = (1 - torch.eye(agent_num).to(action.device))
        action_mask = action_mask.view(-1, 1).repeat(1, self.action_shape).view(agent_num, -1)  # A, A*N
        action = (action_mask.unsqueeze(0).unsqueeze(0)) * action  # T, B, A, A*N
        global_state = global_state.unsqueeze(2).repeat(1, 1, agent_num, 1)

        x = torch.cat([global_state, agent_state, last_action, action, agent_id], -1)
        return x


@MODEL_REGISTRY.register('coma')
class COMA(nn.Module):
    """
    Overview:
        The network of COMA algorithm, which is QAC-type actor-critic.
    Interface:
        ``__init__``, ``forward``
    Properties:
        - mode (:obj:`list`): The list of forward mode, including ``compute_actor`` and ``compute_critic``
    """

    mode = ['compute_actor', 'compute_critic']

    def __init__(
            self, agent_num: int, obs_shape: Dict, action_shape: Union[int, SequenceType],
            actor_hidden_size_list: SequenceType
    ) -> None:
        """
        Overview:
            initialize COMA network
        Arguments:
            - agent_num (:obj:`int`): the number of agent
            - obs_shape (:obj:`Dict`): the observation information, including agent_state and \
                global_state
            - action_shape (:obj:`Union[int, SequenceType]`): the dimension of action shape
            - actor_hidden_size_list (:obj:`SequenceType`): the list of hidden size
        """
        super(COMA, self).__init__()
        action_shape = squeeze(action_shape)
        actor_input_size = squeeze(obs_shape['agent_state'])
        critic_input_size = squeeze(obs_shape['agent_state']) + squeeze(obs_shape['global_state']) + \
            agent_num * action_shape + (agent_num - 1) * action_shape
        critic_hidden_size = actor_hidden_size_list[-1]
        self.actor = COMAActorNetwork(actor_input_size, action_shape, actor_hidden_size_list)
        self.critic = COMACriticNetwork(critic_input_size, action_shape, critic_hidden_size)

    def forward(self, inputs: Dict, mode: str) -> Dict:
        """
        Overview:
            forward computation graph of COMA network
        Arguments:
            - inputs (:obj:`dict`): input data dict with keys ['obs', 'prev_state', 'action']
            - agent_state (:obj:`torch.Tensor`): each agent local state(obs)
            - global_state (:obj:`torch.Tensor`): global state(obs)
            - action (:obj:`torch.Tensor`): the masked action
        ArgumentsKeys:
            - necessary: ``obs`` { ``agent_state``, ``global_state``, ``action_mask`` }, ``action``, ``prev_state``
        ReturnsKeys:
            - necessary:
                - compute_critic: ``q_value``
                - compute_actor: ``logit``, ``next_state``, ``action_mask``
        Shapes:
            - obs (:obj:`dict`): ``agent_state``: :math:`(T, B, A, N, D)`, ``action_mask``: :math:`(T, B, A, N, A)`
            - prev_state (:obj:`list`): :math:`[[[h, c] for _ in range(A)] for _ in range(B)]`
            - logit (:obj:`torch.Tensor`): :math:`(T, B, A, N, A)`
            - next_state (:obj:`list`): :math:`[[[h, c] for _ in range(A)] for _ in range(B)]`
            - action_mask (:obj:`torch.Tensor`): :math:`(T, B, A, N, A)`
            - q_value (:obj:`torch.Tensor`): :math:`(T, B, A, N, A)`
        Examples:
            >>> agent_num, bs, T = 4, 3, 8
            >>> agent_num, bs, T = 4, 3, 8
            >>> obs_dim, global_obs_dim, action_dim = 32, 32 * 4, 9
            >>> coma_model = COMA(
            >>>     agent_num=agent_num,
            >>>     obs_shape=dict(agent_state=(obs_dim, ), global_state=(global_obs_dim, )),
            >>>     action_shape=action_dim,
            >>>     actor_hidden_size_list=[128, 64],
            >>> )
            >>> prev_state = [[None for _ in range(agent_num)] for _ in range(bs)]
            >>> data = {
            >>>     'obs': {
            >>>         'agent_state': torch.randn(T, bs, agent_num, obs_dim),
            >>>         'action_mask': None,
            >>>     },
            >>>     'prev_state': prev_state,
            >>> }
            >>> output = coma_model(data, mode='compute_actor')
            >>> data= {
            >>>     'obs': {
            >>>         'agent_state': torch.randn(T, bs, agent_num, obs_dim),
            >>>         'global_state': torch.randn(T, bs, global_obs_dim),
            >>>     },
            >>>     'action': torch.randint(0, action_dim, size=(T, bs, agent_num)),
            >>> }
            >>> output = coma_model(data, mode='compute_critic')
        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        if mode == 'compute_actor':
            return self.actor(inputs)
        elif mode == 'compute_critic':
            return self.critic(inputs)
