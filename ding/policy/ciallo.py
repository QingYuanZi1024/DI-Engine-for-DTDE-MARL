from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
import torch
import copy

from ding.torch_utils import Adam, to_device
from ding.rl_utils import coma_data, coma_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate, timestep_collate
from .base_policy import Policy


@POLICY_REGISTRY.register('ciallo')
class CIALLOPolicy(Policy):

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='ciallo',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        learn=dict(
            update_per_collect=20,
            batch_size=32,
            learning_rate=0.0005,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) target network update weight, theta * new_w + (1 - theta) * old_w, defaults in [0, 0.1]
            target_update_theta=0.001,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.99,
            # (float) the trade-off factor of td-lambda, which balances 1step td and mc(nstep td in practice)
            td_lambda=0.8,
            # (float) the loss weight of policy network network
            policy_weight=0.001,
            # (float) the loss weight of value network
            value_weight=1,
            # (float) the loss weight of entropy regularization
            entropy_weight=0.01,
        ),
        collect=dict(
            unroll_len=20,
        ),
        eval=dict(),
    )

    def default_model(self) -> Tuple[str, List[str]]:

        return 'ciallo', ['ding.model.template.ciallo']

    def _init_learn(self) -> None:

        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority, "not implemented priority in COMA"
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._gamma = self._cfg.learn.discount_factor
        self._lambda = self._cfg.learn.td_lambda
        self._policy_weight = self._cfg.learn.policy_weight
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight

        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_update_theta}
        )
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        )
        self._learn_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        )
        self._learn_model.reset()
        self._target_model.reset()

    def _data_preprocess_learn(self, data: List[Any]) -> dict:

        # data preprocess
        data = timestep_collate(data)
        assert set(data.keys()) > set(['obs', 'action', 'reward'])
        if self._cuda:
            data = to_device(data, self._device)
        data['weight'] = data.get('weight', None)
        data['done'] = data['done'].float()
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:

        data = self._data_preprocess_learn(data)
        # forward
        self._learn_model.train()
        self._target_model.train()
        self._learn_model.reset(state=data['prev_state'][0])
        self._target_model.reset(state=data['prev_state'][0])
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        with torch.no_grad():
            target_q_value = self._target_model.forward(data, mode='compute_critic')['q_value']
        logit = self._learn_model.forward(data, mode='compute_actor')['logit']
        logit[data['obs']['action_mask'] == 0.0] = -9999999

        data = coma_data(logit, data['action'], q_value, target_q_value, data['reward'], data['weight'])
        coma_loss = coma_error(data, self._gamma, self._lambda)
        total_loss = self._policy_weight * coma_loss.policy_loss + self._value_weight * coma_loss.q_value_loss - \
            self._entropy_weight * coma_loss.entropy_loss

        # update
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        # after update
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': coma_loss.policy_loss.item(),
            'value_loss': coma_loss.q_value_loss.item(),
            'entropy_loss': coma_loss.entropy_loss.item(),
        }

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        self._learn_model.reset(data_id=data_id)

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:

        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.collect.env_num,
            save_prev_state=True,
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        )
        self._collect_model = model_wrap(self._collect_model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float) -> dict:

        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps, data_id=data_id, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        self._collect_model.reset(data_id=data_id)

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:

        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'prev_state': model_output['prev_state'],
            'action': model_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _init_eval(self) -> None:

        self._eval_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.eval.env_num,
            save_prev_state=True,
            init_fn=lambda: [None for _ in range(self._cfg.model.agent_num)]
        )
        self._eval_model = model_wrap(self._eval_model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:

        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, data_id=data_id, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        self._eval_model.reset(data_id=data_id)

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:

        return get_train_sample(data, self._unroll_len)

    def _monitor_vars_learn(self) -> List[str]:

        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss']
