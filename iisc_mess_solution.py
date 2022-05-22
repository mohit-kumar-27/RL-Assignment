import numpy as np


class IIScMess:
    def __init__(self):
        self.demand_values = [100, 200, 300, 400, 500]
        self.demand_probs = [0.15, 0.05, 0.3, 0.25, 0.25]
        self.capacity = self.demand_values[-1]
        self.days = ['Monday', 'Tuesday', 'Wednesday',
                     'Thursday', 'Friday', 'Weekend']
        self.cost_price = 10
        self.selling_price = 12
        self.action_space = [0, 100, 200, 300, 400, 500]
        self.state_space = [('Monday', 0)] + [(d, i)
                                              for d in self.days[1:] for i in [0, 100, 200, 300, 400]]

    def get_next_state_reward(self, state, action, demand):
        day, inventory = state
        result = {}
        result['next_day'] = self.days[self.days.index(day) + 1]
        result['starting_inventory'] = min(self.capacity, inventory + action)
        result['cost'] = self.cost_price * action
        result['sales'] = min(result['starting_inventory'],  demand)
        result['revenue'] = self.selling_price * result['sales']
        result['next_inventory'] = result['starting_inventory'] - result['sales']
        result['reward'] = result['revenue'] - result['cost']
        return result

    def get_transition_prob(self, state, action):
        next_s_r_prob = {}
        for ix, demand in enumerate(self.demand_values):
            result = self.get_next_state_reward(state, action, demand)
            next_s = (result['next_day'], result['next_inventory'])
            reward = result['reward']
            prob = self.demand_probs[ix]
            if (next_s, reward) not in next_s_r_prob:
                next_s_r_prob[next_s, reward] = prob
            else:
                next_s_r_prob[next_s, reward] += prob
        return next_s_r_prob

    def is_terminal(self, state):
        day, inventory = state
        if day == "Weekend":
            return True
        else:
            return False


class IIScMessSolution:

    def example_policy(self, states):
        policy = {}
        for s in states:
            day, inventory = s
            prob_a = {}
            if inventory >= 200:
                prob_a[0] = 1
            else:
                prob_a[100 - inventory] = 0.4
                prob_a[300 - inventory] = 0.6
            policy[s] = prob_a
        return policy

    def iterative_policy_evaluation(self, env, policy, max_iter=1000, v=None, eps=0.01, gamma=1):
        v = {}
        old_v = {}
        for state in env.state_space:
            v[state] = 0
        for i in range(max_iter):

            for state in env.state_space:
                old_v[state] = v[state]
            for state in env.state_space:
                if(env.is_terminal(state) == False):
                    prob_a = policy[state]
                    v[state] = sum([prob_a[action]*sum([env.get_transition_prob(state, action)[next_s, reward]*(
                        reward+gamma*old_v[next_s])for (next_s, reward) in env.get_transition_prob(state, action)])for action in prob_a])
            delta = 0
            for state in env.state_space:
                delta = max(delta, abs(v[state]-old_v[state]))
            if(delta < eps):
                break
        return v

    def value_iteration(self, env, max_iter=1000, eps=0.01, gamma=1):
        v = {}
        old_v = {}
        for state in env.state_space:
            v[state] = 0
        for i in range(max_iter):

            for state in env.state_space:
                old_v[state] = v[state]
            for state in env.state_space:
                if(env.is_terminal(state) == False):
                    v[state] = max([sum([env.get_transition_prob(state, action)[next_s, reward]*(
                        reward+gamma*old_v[next_s])for (next_s, reward) in env.get_transition_prob(state, action)])for action in env.action_space])
            delta = 0
            for state in env.state_space:
                delta = max(delta, abs(v[state]-old_v[state]))
            if(delta < eps):
                break
        return v


if __name__ == "__main__":

    # check the value_iteration output
    mess = IIScMess()
    solution = IIScMessSolution()
    v = solution.value_iteration(mess)
    assert(int(v[('Monday', 0)]) == 2884)

    # check the policy evaluation output
    mess = IIScMess()
    solution = IIScMessSolution()
    policy = solution.example_policy(mess.state_space)
    v = solution.iterative_policy_evaluation(mess, policy)
    assert(int(v[('Monday', 0)]) == 1775)
