import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pyglet

class MyEnv(gym.Env):
    """
    Able to build required model based on gym.
    Example: Cart
    Description:
        A cart moves along a frictionless track.The goal is to move it to the origin with 0 velocity.

    Observation: 
        Type: Box(2)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf

    Actions:
        Type: Box(1)
        Num	Action
        0	force to push cart to the left (negative value)
                                or to the right (positive value)

    Reward:
        r(x,u) = 10*x^2+u^2

    Starting State:
        All observations are assigned a uniform random value in [-4.0,4.0]

    Episode Termination:
        Cart Position is more than 4.0 (center of the cart reaches the edge of the display)
        Episode length is greater than 100
        
    Solved Requirements:
        Considered solved when the average reward is greater or equal to ? over 100 consecutive trials.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    def __init__(self):
        self.mass = 1.0
        self.tau = 1.0  # seconds between state updates

        self.x_threshold = 4.0  # position at which to fail the episode
        state_high = np.array([
            self.x_threshold,
            np.finfo(np.float32).max])  # max number in float32, express infinity
        action_high = np.array([10.0])  # max absolute input

        self.action_space = spaces.Box(-action_high,action_high, dtype=np.float32)
        self.observation_space = spaces.Box(-state_high, state_high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action,traj_time):
        """ During one constant periode, excutes one constant action to the model.
        The return is: np.array(self.state), reward, done, {}
        where: 
            np.array(self.state): the next state
            reward: r(s,a)
            done: if the next state is the terminal state
        """
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # print(self.state)
        state = self.state
        x, x_dot = state
        [u] = action
        xacc = u/self.mass

        # computes next state
        x_next  = x + self.tau * x_dot + 0.5*xacc*self.tau*self.tau
        x_dot_next = x_dot + self.tau * xacc
        self.state = (x_next,x_dot_next)
        
        self.episode_length +=1
        done1 =  bool(x_next < -self.x_threshold
                or x_next > self.x_threshold)
                
        done2 = bool((x_next<0.1 and x_next>-0.1 and 
                    x_dot_next<0.1 and x_dot_next>-0.1)
                     or self.episode_length>=100)

        done = done1 or done2

        if not done:  # reward function
            reward = x*x + x_dot*x_dot + u*u
        elif self.steps_beyond_done is None:
            # position just outside bound
            self.steps_beyond_done = 0
            if done:
                reward =  10*(x*x + x_dot*x_dot + u*u)
#             if done2:
#                 reward =  x*x + x_dot*x_dot + u*u
#             if done1:
#                 reward = 100.0*(100.0-traj_time)
            self.episode_length = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already \
                returned done = True. You should always call 'reset()' once you receive 'done = \
                True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = x*x + x_dot*x_dot + u*u
            self.episode_length = 0

        return np.array(self.state), 0.1*reward, done, {}
    
    def reset(self):
        """resets the environment"""
        self.state = self.np_random.uniform(low=-2, high=2, size=(2,))
        self.steps_beyond_done = None
        self.episode_length = 0
        return np.array(self.state)

    def render(self,text=1, mode='human'):
        """visilization of the cart motion"""
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        label = pyglet.text.Label(text, font_size=20,
                              x=50, y=300, anchor_x='left', anchor_y='top',
                              color=(255, 0, 0, 123))
        class DrawText:
            def __init__(self, label:pyglet.text.Label):
                self.label=label
            def render(self):
                self.label.draw()

        self.viewer.add_geom(DrawText(label))
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
    def text(self, text, x=50, y=100, font_size=20, color=(255, 0, 0, 123)):
        """Draw text"""
        label = pyglet.text.Label(text, font_size=font_size,
                                  x=x, y=y, anchor_x='left', anchor_y='top',
                                  color=color)
        label.draw()
        
    def close(self):
        """closes the environment"""
        if self.viewer:
            self.viewer.close()
            self.viewer = None


