from gym import spaces
import torch
import torch as th
import numpy as np
from controllers.basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
class CQMixMAC(BasicMAC):

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, past_actions=None, critic=None,
                       target_mac=False, explore_agent_ids=None):
        avail_actions = ep_batch["avail_actions"][bs, t_ep]

        if t_ep is not None and t_ep > 0:
            past_actions = ep_batch["actions"][:, t_ep-1]

        if getattr(self.args, "agent", "cqmix") == "cqmix":
            raise Exception("No CQMIX agent selected (naf, icnn, qtopt)!")

        # Note batch_size_run is set to be 1 in our experiments
        if self.args.agent in ["naf", "mlp", "rnn"]:
            chosen_actions = self.forward(ep_batch[bs],
                                          t_ep,
                                          hidden_states=self.hidden_states[bs],
                                          test_mode=test_mode,
                                          select_actions=True)["actions"] # just to make sure detach
            chosen_actions = chosen_actions.view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).detach()

        # Now do appropriate noising
        exploration_mode = getattr(self.args, "exploration_mode", "gaussian")

        if not test_mode:  # do exploration
            if exploration_mode == "gaussian":
                start_steps = getattr(self.args, "start_steps", 0)
                act_noise = getattr(self.args, "act_noise", 0.1)
                if t_env >= start_steps:
                    if explore_agent_ids is None:
                        x = chosen_actions.clone().zero_()
                        chosen_actions += act_noise * x.clone().normal_()
                    else:
                        for idx in explore_agent_ids:
                            x = chosen_actions[:, idx].clone().zero_()
                            chosen_actions[:, idx] += act_noise * x.clone().normal_()
                            
        chosen_continuous_actions = chosen_actions[:, :, :self.args.action_continuous_spaces[0].shape[0]].clone()
        chosen_discrete_actions = chosen_actions[:, :, self.args.action_continuous_spaces[0].shape[0]:].clone()
        # For continuous actions, now clamp actions to permissible action range (necessary after exploration)
        if all([isinstance(act_space, spaces.Box) for act_space in self.args.action_continuous_spaces]):
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_continuous_spaces[_aid].shape[0]):
                    chosen_continuous_actions[:, _aid, _actid].clamp_((self.args.action_continuous_spaces[_aid].low[_actid]).item(),
                                                           (self.args.action_continuous_spaces[_aid].high[_actid]).item())
                    chosen_continuous_actions[:, _aid, _actid] += 1.0e-6  # to avoid numerical issues

        # For discrete actions, now sample from the distribution
        if all([isinstance(act_space, spaces.Tuple) for act_space in self.args.action_discrete_spaces[0]]):
            for _aid in range(self.n_agents):
                for _actid in range(len(self.args.action_discrete_spaces[_aid])):
                    # tuple for discrete action space
                    discrete_values = self.args.action_discrete_spaces[_aid][_actid]
                    tanh_outputs = chosen_discrete_actions[:, _aid, _actid] + 1.0e-6  # to avoid numerical issues

                    # 基于分桶的思想将连续值映射到离散值
                    num_buckets = len(discrete_values.spaces)
                    bin_width = 2.0 / num_buckets

                    # 使用torch操作保持梯度（如果需要）并提高效率
                    chosen_discrete_actions[:, _aid, _actid] = torch.clamp(
                        torch.floor((tanh_outputs + 1) / bin_width), 
                        0, 
                        num_buckets - 1
                    ).long()
                    
                    try:
                      # 将索引映射到实际离散值
                      chosen_discrete_actions[:, _aid, _actid] = discrete_values.spaces[
                          int(chosen_discrete_actions[:, _aid, _actid][0].item())
                      ]
                    except Exception as e:
                      print(f"Error in mapping discrete actions: {tanh_outputs}")
                      chosen_discrete_actions[:, _aid, _actid] = 0
        
        # combine continuous and discrete actions
        new_chosen_actions = th.cat([chosen_continuous_actions, chosen_discrete_actions], dim=2)

        return new_chosen_actions, chosen_actions

    def get_weight_decay_weights(self):
        return self.agent.get_weight_decay_weights()

    def forward(self, ep_batch, t, actions=None, hidden_states=None, select_actions=False, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        ret = self.agent(agent_inputs, self.hidden_states, actions=actions)
        if select_actions:
            self.hidden_states = ret["hidden_state"]
            return ret
        agent_outs = ret["Q"]
        self.hidden_states = ret["hidden_state"]

        if self.agent_output_type == "pi_logits":
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/agent_outs.size(-1))
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), actions

    def _build_inputs(self, batch, t, target_mac=False, last_target_action=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            if getattr(self.args, "discretize_actions", False):
                input_shape += scheme["actions_onehot"]["vshape"][0]
            else:
                input_shape += scheme["actions"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
