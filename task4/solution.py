import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation=None):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        if activation is not None:
            self.activation = activation
        else:
            self.activation = nn.ReLU()
        
        layers = [nn.Linear(input_dim, hidden_size), self.activation] # input layer
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), self.activation] # hidden layers
        self.output_layer = nn.Linear(hidden_size, output_dim) # output layer
        self.model = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor):
        # TODO: Implement the forward pass for the neural network you have defined.
        #print(self.model)
        x = self.model(s)
        return x, self.output_layer(x)
    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py. 
        self.actor = NeuralNetwork(self.state_dim, self.action_dim, self.hidden_size, self.hidden_layers)

        self.mu = nn.Linear(self.hidden_size, self.action_dim, device=self.device)
        self.sigma = nn.Linear(self.hidden_size, self.action_dim, device=self.device)

        self.optimizer = optim.Adam(list(self.actor.parameters())+list(self.mu.parameters())+list(self.sigma.parameters()), self.actor_lr)


    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        
        x, action = self.actor.forward(state)
        #print(x.shape, action.shape)
        
        mu = self.mu(x)

        if deterministic: torch.tanh(mu), log_prob # TODO maybe change?
        
        sigma = self.clamp_log_std(self.sigma(x))
        std = torch.exp(sigma)
        
        prob = Normal(mu, std)
        action = prob.rsample()
        action = torch.tanh(action)

        #print(log_prob)
        log_prob = prob.log_prob(action) #- torch.log(1 - action.pow(1) + 1e-10)
        #print(action, log_prob)
        #print(log_prob.shape, (state.shape[0], self.action_dim), action.shape)
        
        #assert action.shape == (state.shape[0], self.action_dim) and \
        #    log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        
        return action, log_prob


class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.

        self.critic1 = NeuralNetwork(self.state_dim + self.action_dim, self.action_dim, self.hidden_size, self.hidden_layers).to(self.device)
        self.critic2 = NeuralNetwork(self.state_dim + self.action_dim, self.action_dim, self.hidden_size, self.hidden_layers).to(self.device)
        self.critic_target1 = NeuralNetwork(self.state_dim + self.action_dim, self.action_dim, self.hidden_size, self.hidden_layers).to(self.device)
        self.critic_target2 = NeuralNetwork(self.state_dim + self.action_dim, self.action_dim, self.hidden_size, self.hidden_layers).to(self.device)
        
        self.value = NeuralNetwork(self.state_dim, self.action_dim, self.hidden_size, self.hidden_layers).to(self.device)
        self.target_value = NeuralNetwork(self.state_dim, self.action_dim, self.hidden_size, self.hidden_layers).to(self.device)

        # Optimizers
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)

        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.critic_lr)
        self.target_value_optimizer = optim.Adam(self.target_value.parameters(), lr=self.critic_lr)

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temperature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.
        
        self.actor = Actor(128, 2, 6e-4, self.state_dim, self.action_dim)
        self.critic = Critic(128, 2, 6e-4, self.state_dim, self.action_dim)
        self.gamma = TrainableParameter(init_param=0.99, lr_param=3e-4, train_param=True)
        self.alpha = TrainableParameter(init_param=1, lr_param=3e-4, train_param=True)
        self.tau = TrainableParameter(init_param=5*1e-3, lr_param=1e-5, train_param=True)

        self.target_entropy = -1

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        s = torch.from_numpy(s).to(self.device)
        #print(s)
        action, log_prob = self.actor.get_action_and_log_prob(s, train)
        action = action.detach().numpy()
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray'
        #print(action) 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in an object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        # value function:
        #with torch.no_grad():
        actions_, log_probs_ = self.actor.get_action_and_log_prob(s_batch, False)
        
        state_action = torch.cat([s_batch, actions_], 1)
        q1_new = self.critic.critic1(state_action)[1]
        q2_new = self.critic.critic2(state_action)[1]
        
        #Q_target1_next = self.critic.critic_target1(state_action)[1]
        #Q_target2_next = self.critic.critic_target2(state_action)[1]
        v = self.critic.value(s_batch)[1]
        v_ = self.critic.target_value(s_prime_batch)[1]

        Q_target = torch.min(q1_new, q2_new) - self.alpha.get_log_param() * log_probs_
        #Q_target =  #self.critic.value(s_batch) #r_batch #+ self.gamma.get_param() * Q_target_next
    
        value_loss = 0.5*F.mse_loss(v, Q_target)
        self.critic.value_optimizer.zero_grad()
        value_loss.backward()
        self.critic.value_optimizer.step()

        # Actor loss
        new_actions, log_probs = self.actor.get_action_and_log_prob(s_batch, False)
        state_action = torch.cat([s_batch, new_actions], 1)
        Q1_new = self.critic.critic1(state_action)[1]
        Q2_new = self.critic.critic2(state_action)[1]
        Q_new = torch.min(Q1_new, Q2_new)
        actor_loss = (self.alpha.get_log_param() * log_probs - Q_new).mean()

        # Update actor
        self.actor.optimizer.zero_grad()
        #print(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        # Critic loss
        state_action = torch.cat([s_batch, a_batch], 1)
        Q1 = self.critic.critic1(state_action)[1]
        Q2 = self.critic.critic2(state_action)[1]

        Q_target = v_ * self.gamma.get_param() + r_batch * 2
        critic_loss1 = 0.5 * F.mse_loss(Q1, Q_target)
        critic_loss2 = 0.5 * F.mse_loss(Q2, Q_target)
        
        c_loss = torch.add(critic_loss1, critic_loss2)
        # Update critics
        self.critic.critic1_optimizer.zero_grad()
        self.critic.critic2_optimizer.zero_grad()
        
        c_loss.backward()
        self.critic.critic1_optimizer.step()
        self.critic.critic2_optimizer.step()

        # Update target networks
        self.critic_target_update(self.critic.value, self.critic.target_value, self.tau.get_param(), True)
        #self.critic_target_update(self.critic.critic2, self.critic.critic_target2, self.tau.get_param(), True)
        
        alpha_loss = -(self.alpha.get_log_param() * (log_probs + 1*self.target_entropy).detach()).mean()
        self.alpha.optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha.optimizer.step()
        
        #actions , log_probs = self.actor.get_action_and_log_prob(s_batch, deterministic=False)
        #log_probs = log_probs.view(-1)
        #q1_new_policy = self.critic.critic1.forward(s_batch, actions)
        #q2_new_policy = self.critic_2.forward(s_batch, actions_)
        #q_min = q_min.view(-1)
        #v = self.value.forward(states).view(-1)
        #v_ = self.target_value.forward(states_).view(-1)
        #v_[done] = 0.0

        #target_value = q_min - log_probs
        #value_loss = 0.5 * F.mse_loss(v, target_value)
        #self.value.optimizer.zero_grad()
        #value_loss.backward(retain_graph=True)
        #self.value.optimizer.step()


        # # TODO: Implement Critic(s) update here.
        # value_1 = self.critic.critic1.forward(s_batch)
        # critic_loss = torch.mse_loss(value_1, r_batch)

        # # TODO: Implement Policy update here
        # actions, log_probs = self.actor.get_action_and_log_prob(s_batch, deterministic=False)
        # q_1 = self.critic.critic1.forward(s_batch)
        # #value_2 = self.critic.critic2.forward()

        # self.actor.optimizer.zero_grad()
        # actor_loss = torch.mean(log_probs - q_1)
        # actor_loss.backward(retain_graph=True)
        # self.actor.optimizer.step()


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
