import random
import gym
import numpy as np
import keras
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from gym import wrappers

EPISODES_COUNT = 100000 #TODO: Hyperparameter #1
EPSILON_START = 1 #TODO: Hyperparameter #2
EPSILON_FINISH = 0.05 #TODO: Hyperparameter #3
EPSILON_ENDS = 50000 #TODO: Hyperparameter #4
DISCOUNT_RATE = 0.99 #TODO: Hyperparameter #5
LEARNING_RATE = 0.01 #TODO: Hyperparameter #6
NEURON_COUNT = 20 #TODO: Hyperparameter #7

episodes_done = 0

class Agent:
    def __init__(self, ACTION_COUNT, STATE_COUNT): #initing our agent
        self.act_count = ACTION_COUNT
        self.state_count = STATE_COUNT
        self.model = self.initialize_model()
        
    def initialize_model(self): #we'll have two layers, both of them with @param NEURON_COUNT /@ neurons inside them, both activations are defaultly set to be RelU
        input_layer = Input(shape = (self.state_count, ), name = 'input_layer')
        
        hidden_layer_1 = Dense(NEURON_COUNT, activation = 'relu', name = 'hidden_layer_1')(input_layer)
        hidden_layer_2 = Dense(NEURON_COUNT, activation = 'relu', name = 'hidden_layer_2')(hidden_layer_1)
        
        output_layer = Dense(self.act_count, activation = 'linear', name = 'output_layer')(hidden_layer_2)
        
        model = Model(inputs = input_layer, outputs = output_layer)
        model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
        return model
        
    def get_epsilon(self): #getting the epsilon (exploration probability) in that exact moment
        global episodes_done
        if episodes_done >= EPSILON_ENDS:
            return EPSILON_ENDS
        else:
            return EPSILON_START - episodes_done / EPSILON_ENDS * (EPSILON_START - EPSILON_ENDS)
            
    def load(self, path): #loads the model
        self.model.load_weights(path)
        
    def save(self, path):
        self.model.save_weights(path)
        
    def act(self, state): #returns the next action to be played, given what state agent is currently in
        eps_current = self.get_epsilon()
        rand_generated = np.random.rand()
        if rand_generated < eps_current: #we should do a random action then
            return random.randrange(0, self.act_count)
        else: #we should listen to our policy
            current_policy = self.model.predict(state)
            return np.random.choice(a = self.act_count, p = current_policy[0])
     
     
GAME = "CartPole-v1"       
def main():
    global episodes_done
    env = gym.make(GAME) #creating a new game
    env = wrappers.Monitor(env, './logs/' + GAME, force = True) #logs to be made
    STATE_COUNT = env.observation_space.shape[0] #dimension of our state
    ACTION_COUNT = env.action_space.n #number of possible actions we can make
    agent = Agent(ACTION_COUNT, STATE_COUNT)
    
    tmpModel = agent.model

    while episodes_done < EPISODES_COUNT: #we are playing EPISODES_COUNT (hyperparameter) episodes before finishing our training
        state = env.reset() #beginning new instance of the game
        state = np.reshape(state, [1, -1]) #unfortunately, we have to reshape this thing here
        steps_made = 0
        while True:
            steps_made += 1
            action = agent.act(state) #finds the next action we are supposed to play
            next_state, current_reward, done, useless = env.step(action) #we are actually playing that action right now
            
            if done:
                current_reward = -100 #punish the agent if the game is already lost
            
            next_state = np.reshape(next_state, [1, -1]) #and this thing here, again
            if done:
                episodes_done += 1
                print('Episode ' + str(episodes_done) + ', score achieved ' + str(steps_made))
                break

            #target = current_reward + DISCOUNT_RATE * np.amax(agent.model.predict(next_state)[0]) #`training` part of the code
            target = current_reward + DISCOUNT_RATE * np.amax(tmpModel.predict(next_state)[0])
            #target_f = agent.model.predict(next_state)
            target_f = tmpModel.predict(next_state)
            target_f[0][action] = target
            agent.model.fit(state, target_f, epochs = 3, verbose = 0) #epochs is a hyperparameter

        if (episodes_done % 100 == 99):
            tmpModel = Model(inputs = agent.model.input, outputs = agent.model.output)
            agent.model.save('saved_models/model_' + str(episodes_done) + '.h5')

if __name__ == '__main__':
    main()
