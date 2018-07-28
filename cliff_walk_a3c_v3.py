from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import sys
from threading import Thread

from pycolab import ascii_art
#from pycolab import human_ui
from pycolab.prefab_parts import sprites as prefab_sprites

import numpy as np
import tensorflow as tf

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K

#-- constants
RUN_TIME = 30
THREADS = 22
OPTIMIZERS = 20
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.9
EPS_STOP  = 0.001
EPS_STEPS = 50000000

MIN_BATCH = 64
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

#---------
class Brain:
  train_queue = [ [], [], [], [], [] ]	# s, a, r, s', s' terminal mask
  lock_queue = threading.Lock()

  def __init__(self):
    self.session = tf.Session()
    K.set_session(self.session)
    K.manual_variable_initialization(True)

    self.model = self._build_model()
    self.graph = self._build_graph(self.model)

    self.session.run(tf.global_variables_initializer())
    self.default_graph = tf.get_default_graph()

    self.default_graph.finalize()	# avoid modifications

  def _build_model(self):

    self.l_input = Input( batch_shape=(None, NUM_STATE) )
    
    l_dense1 = Dense(32, activation='relu')(self.l_input)
    l_dense2 = Dense(32, activation='relu')(l_dense1)
    l_dense3 = Dense(32, activation='relu')(l_dense2)

    self.out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense3)
    self.out_value   = Dense(1, activation='linear')(l_dense3)

    model = Model(inputs=[self.l_input], outputs=[self.out_actions, self.out_value])
    model._make_predict_function()	# have to initialize before threading

    return model

  def _build_graph(self, model):
    s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
    a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
    r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward
    
    p, v = model(s_t)

    log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
    advantage = r_t - v

    loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
    loss_value  = LOSS_V * tf.square(advantage)												# minimize value error
    entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)

    loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
    minimize = optimizer.minimize(loss_total)

    return s_t, a_t, r_t, minimize

  def optimize(self):
    if len(self.train_queue[0]) < MIN_BATCH:
      time.sleep(0)	# yield
      return

    with self.lock_queue:
      if len(self.train_queue[0]) < MIN_BATCH:	# more thread could have passed without lock
        return 									# we can't yield inside lock

      s, a, r, s_, s_mask = self.train_queue
      self.train_queue = [ [], [], [], [], [] ]

    s = np.vstack(s)
    a = np.vstack(a)
    r = np.vstack(r)
    s_ = np.vstack(s_)
    s_mask = np.vstack(s_mask)

    if len(s) > 5*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

    v = self.predict_v(s_)
    r = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state
    
    s_t, a_t, r_t, minimize = self.graph
    self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

  def train_push(self, s, a, r, s_):
    with self.lock_queue:
      self.train_queue[0].append(s)
      self.train_queue[1].append(a)
      self.train_queue[2].append(r)

      if s_ is None:
        self.train_queue[3].append(NONE_STATE)
        self.train_queue[4].append(0.)
      else:	
        self.train_queue[3].append(s_)
        self.train_queue[4].append(1.)

  def predict(self, s):
    with self.default_graph.as_default():
      p, v = self.model.predict(s)
      return p, v

  def predict_p(self, s):
    with self.default_graph.as_default():
      p, v = self.model.predict(s)		
      return p

  def predict_v(self, s):
    with self.default_graph.as_default():
      p, v = self.model.predict(s)		
      return v

#---------
frames = 0
class Agent:
  def __init__(self, eps_start, eps_end, eps_steps):
    self.eps_start = eps_start
    self.eps_end   = eps_end
    self.eps_steps = eps_steps

    self.memory = []	# used for n_step return
    self.R = 0.

  def getEpsilon(self):
    #if(frames >= self.eps_steps):
    global episodes_total
    if episodes_total >= 10000:
      return self.eps_end
    else:
      return self.eps_start + episodes_total * (self.eps_end - self.eps_start) / 10000	# linearly interpolate
      
  def act(self, s):
    eps = self.getEpsilon()			
    global frames; frames = frames + 1
    if random.random() < eps:
      return random.randint(0, NUM_ACTIONS-1)

    else:
      s = np.array([s])
      p = brain.predict_p(s)[0]
      
      a = np.argmax(p)
      #a = np.random.choice(NUM_ACTIONS, p=p)

      return a
  
  def act_fake(self, s):
      s = np.array([s])
      p = brain.predict_p(s)[0]
      
      a = np.argmax(p)
      #a = np.random.choice(NUM_ACTIONS, p=p)

      return a

  def train(self, s, a, r, s_):
    def get_sample(memory, n):
      s, a, _, _  = memory[0]
      _, _, _, s_ = memory[n-1]

      return s, a, self.R, s_

    a_cats = np.zeros(NUM_ACTIONS)	# turn action into one-hot representation
    a_cats[a] = 1 

    self.memory.append( (s, a_cats, r, s_) )

    self.R = ( self.R + r * GAMMA_N ) / GAMMA

    if s_ is None:
      while len(self.memory) > 0:
        n = len(self.memory)
        s, a, r, s_ = get_sample(self.memory, n)
        brain.train_push(s, a, r, s_)

        self.R = ( self.R - self.memory[0][2] ) / GAMMA
        self.memory.pop(0)		

      self.R = 0

    if len(self.memory) >= N_STEP_RETURN:
      s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
      brain.train_push(s, a, r, s_)

      self.R = self.R - self.memory[0][2]
      self.memory.pop(0)	
  
  # possible edge case - if an episode ends in <N steps, the computation is incorrect
    
episodes_total = 0
def fnd(s):
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            if s[i][j] == 80:
                return np.array([i, j])
#---------
def conv(s):
    return fnd(s[0])

class Environment(threading.Thread):
  stop_signal = False

  def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
    threading.Thread.__init__(self)

    self.render = render
    self.env = make_game()
    self.agent = Agent(eps_start, eps_end, eps_steps)

  def runEpisode(self):
    global episodes_total
    episodes_total += 1
    # if episodes_total % 100 == 0:
        #print("running")
        #tester.run()
    if (episodes_total % 1500 == 0):
        global brain
        brain.model.save("saved_model/model" + str(episodes_total))

    self.env = make_game()
    s, _, _ = self.env.its_showtime()
    #print('s shape ' + str(s[0].shape))
    R = 0
    while True:
      time.sleep(THREAD_DELAY)
      convs = conv(s)
      #print('s ' + str(conv(s)))
      a = self.agent.act(convs)
      #print('a ' + str(a))
      observation, reward, done = self.env.play(a)
      if done == 0.0:
        observation = None
      if observation is not None:
        self.agent.train(convs, a, reward, conv(observation))
      else:
        self.agent.train(convs, a, reward, None)
      s = observation
      R += reward
      if observation is None or self.stop_signal:
        break
      
    #print('Total score ' + str(R) + ' in episode ' + str(episodes_total))

  def run(self):
    while not self.stop_signal:
      self.runEpisode()

  def stop(self):
    self.stop_signal = True

#---------
class Optimizer(threading.Thread):
  stop_signal = False

  def __init__(self):
    threading.Thread.__init__(self)

  def run(self):
    while not self.stop_signal:
      brain.optimize()

  def stop(self):
    self.stop_signal = True


#class Tester(threading.Thread):
class Tester(threading.Thread):
    stop_signal = False
    def __init__(self):
        #threading.Thread.__init__(self)
        self.agent = Agent(EPS_START, EPS_STOP, EPS_STEPS)

    def run(self):
        self.env = make_game()
        s, _, _ = self.env.its_showtime()
        #print('s shape ' + str(s[0].shape))
        R = 0
        finished = False
        it = 0
        while it < 100:
            convs = conv(s)
            #print('s ' + str(conv(s)))
            a = self.agent.act_fake(convs)
            #print('a ' + str(a))
            observation, reward, done = self.env.play(a)
            if done == 0.0:
                observation = None
            #if observation is not None:
            #    self.agent.train(convs, a, reward, conv(observation))
            #else:
            #    self.agent.train(convs, a, reward, None)
            s = observation
            R += reward
            if observation is None or self.stop_signal:
                finished = true
                break
            it += 1
      
        print('TESTER Total score ' + str(R) + ' finished? ' + str(finished))

#-- main
# Copyright 2017 the pycolab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example implementation of the classic cliff-walk problem."""


GAME_ART = ['............',
            '............',
            '............',
            'P...........']


def make_game():
  """Builds and returns a cliff-walk game."""
  return ascii_art.ascii_art_to_game(
      GAME_ART, what_lies_beneath='.',
      sprites={'P': PlayerSprite})


class PlayerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player.

  This `Sprite` ties actions to going in the four cardinal directions. If it
  walks into all but the first and last columns of the bottom row, it receives a
  reward of -100 and the episode terminates. Moving to any other cell yields a
  reward of -1; moving into the bottom right cell terminates the episode.
  """

  def __init__(self, corner, position, character):
    """Inform superclass that we can go anywhere, but not off the board."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable='', confined_to_board=True)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del layers, backdrop, things   # Unused.

    # Apply motion commands.
    if actions == 0:    # walk upward?
      self._north(board, the_plot)
    elif actions == 1:  # walk downward?
      self._south(board, the_plot)
    elif actions == 2:  # walk leftward?
      self._west(board, the_plot)
    elif actions == 3:  # walk rightward?
      self._east(board, the_plot)
    else:
      # All other actions are ignored. Although humans using the CursesUi can
      # issue action 4 (no-op), agents should only have access to actions 0-3.
      # Otherwise staying put is going to look like a terrific strategy.
      return

   # See what reward we get for moving where we moved.
    if (self.position[0] == (self.corner[0] - 1) and
        0 < self.position[1] < (self.corner[1] - 2)):
      the_plot.add_reward(-100.0)  # Fell off the cliff.
    else:
      the_plot.add_reward(-1.0)

    # See if the game is over.
    if self.position[0] == (self.corner[0] - 1) and 0 < self.position[1]:
      the_plot.terminate_episode()

env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = 2
NUM_ACTIONS = 4
NONE_STATE = np.zeros(NUM_STATE) #TODO
print(str(NUM_STATE) + str(NUM_ACTIONS))

brain = Brain()	# brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

tester = Tester()

for o in opts:
  o.start()

for e in envs:
  e.start()

time.sleep(RUN_TIME * 1200) # 10 hours

for e in envs:
  e.stop()
for e in envs:
  e.join()

for o in opts:
  o.stop()
for o in opts:
  o.join()

print("Training finished")
print(episodes_total)

print("Saving model")