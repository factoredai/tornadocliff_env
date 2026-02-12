from contextlib import closing
from io import StringIO
from os import path
from typing import Dict, List, Literal, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class TornadoCliffEnv(Env):
    """
    Cliff Walking with Tornados is a variation of the original Cliff walking environment.
    It involves crossing a gridworld while simultaneously avoiding falling off a cliff
    and encountering a tornado which blows away the character to a random square in the
    grid (including the cliff).

    ## Description
    The game start with the Elf at the top left corner of a gridworld (i.e. [0,0]), which a
    given shape (default is (6,8)). The goal location is always at the bottom right corner
    (i.e. [-1,-1]), and if the Elf reaches the goal the episodes ends.

    A cliff runs along the middle of the grid. If the player moves to a cliff location it
    returns to the start location.

    A tornado begins from a random square, excluding the cliff, and it makes a random walk
    thorug the grid at a given pace (default if 1). If the Elf crosses the tornado, it will
    be blown away a random square in the grid including the cliff.

    The player makes moves until they reach the goal.

    Resembles to Example 6.6 (page 132) from Reinforcement Learning: An Introduction
    by Sutton and Barto [<a href="#cliffwalk_ref">1</a>].

    It is an adaptation of Gymnasium's Cliff Walking [<a href="#gymnasium_ref">2</a>].

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.

    - 0: Move up
    - 1: Move right
    - 2: Move down
    - 3: Move left

    ## Observation Space
    The observation depends of the shape of the grid. For a (6,8) grid there are
    48*48 posibles states, corresponding to the position of the Elf and the position of
    the tornado. The player cannot be at the cliff, nor at the goal as the latter
    results in the end of the episode.

    The observation is a tuple representing the player's and the tornado's current
    position as current_row * nrows + current_col (where both the row and col start at 0).

    The observation is returned as an `Tuple[int, int]`.

    ## Starting State
    The episode starts with the player in state `[0]` (location [0, 0]). And the tornado
    begins at a random state.

    ## Reward
    Each time step incurs -1 reward, unless the player stepped into the cliff,
    which incurs -100 reward.

    ## Episode End
    The episode terminates when the player reaches the goal at the bottom left corner.

    ## Information

    `step()` and `reset()` return a dict with the following keys:
    - "p" - transition proability for the state.

    As cliff walking is not stochastic, the transition probability returned always 1.0.

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('CliffWalking-v0')
    ```

    ## References
    <a id="cliffwalk_ref"></a>[1] R. Sutton and A. Barto, “Reinforcement Learning:
    An Introduction” 2020. [Online].
    Available: [http://www.incompleteideas.net/book/RLbook2020.pdf](http://www.incompleteideas.net/book/RLbook2020.pdf) # noqa :

    <a id="gymnasium_ref"></a>[2] Farama Foundation, “Gymnasium” 2023. (v0.28.1).
    See: [https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/cliffwalking.py](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/cliffwalking.py)


    ## Version History
    - v0: Initial version release

    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: Optional[Literal["human", "rgb_array", "ansi"]] = None,
        shape: Tuple[int, int] = (6, 8),
        tornado_step_size: int = 1,
    ):
        """_summary_

        Parameters
        ----------
        render_mode : Optional[Literal["human", "rgb_array", "ansi"]], optional
            Render mode, by default None
        shape : Tuple[int, int], optional
            Grid shape, by default (6, 8)
        tornado_step_size : int, optional
            Step size of tornado random walk, by default 1
        """

        self.shape = shape

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=bool)
        self._cliff[
            (shape[0] // 2) - 1 : (shape[0] // 2) + 1,
            1:-2,
        ] = True

        # State and actions count
        self.nS_elf = int(np.prod(self.shape))
        self.nS_tornado = int(np.prod(self.shape))
        self.nA = 4

        # Calculate initial state distribution
        # Tornado
        self._tornado_step_size = tornado_step_size
        self.tornado_initial_st_dist = np.ravel(~self._cliff / np.sum(~self._cliff))

        # Elf
        elf_start_row, elf_start_col = 0, 0  # elf always starts at 0, 0
        self.elf_start_state_index = np.ravel_multi_index((elf_start_row, elf_start_col), self.shape)
        self.elf_initial_st_dist = np.zeros(self.nS_elf)
        self.elf_initial_st_dist[self.elf_start_state_index] = 1.0

        # Terminal state
        self.terminal_state = (self.shape[0] - 1, self.shape[1] - 1)

        # Spaces
        self.observation_space = spaces.MultiDiscrete(
            [self.nS_elf, self.nS_tornado],
        )
        self.action_space = spaces.Discrete(self.nA)

        self.render_mode = render_mode

        # pygame utils
        self.cell_size = (60, 60)
        self.window_size = (
            self.shape[1] * self.cell_size[1],
            self.shape[0] * self.cell_size[0],
        )
        self.window_surface = None
        self.clock = None
        self.elf_images = None
        self.start_img = None
        self.goal_img = None
        self.cliff_img = None
        self.mountain_bg_img = None
        self.near_cliff_img = None
        self.tree_img = None
        self.tornado_img = None

    def _limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """Prevent the agent from falling out of the grid world."""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _update_tornado(self, tornado_current: List[int]) -> int:
        """Deternine the new state of the tornado

        Parameters
        ----------
        tornado_current : List[int]
            Current position of tornado

        Returns
        -------
        int
            New state of tornado
        """
        in_cliff = True
        while in_cliff:
            delta = np.random.randint(-self._tornado_step_size, self._tornado_step_size + 1, 2)
            new_position = tuple(self._limit_coordinates(tornado_current + delta))
            if not self._cliff[new_position]:
                in_cliff = False

        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        return new_state

    def _update_elf(self, elf_current: List[int], tornado_current: List[int], delta: List[int]) -> List[Tuple]:
        """Determine the new state of the elf

        Parameters
        ----------
        elf_current : List[int]
            Current position of the elf
        tornado_current : List[int]
            Current position of the tornado
        delta : List[int]
            Change in position for transition of the elf

        Returns
        -------
        List[Tuple]
            Tuple of ``(1.0, new_state, reward, terminated)``
        """

        if elf_current == tornado_current:
            # Send Elf to a random position
            elf_random_dist = np.ones(self.nS_elf, dtype=bool) / self.nS_elf
            random_s = categorical_sample(elf_random_dist, self.np_random)
            new_position = np.array(np.unravel_index(random_s, self.shape))

        else:
            new_position = np.array(elf_current) + np.array(delta)

        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.elf_start_state_index, -100, False)]

        is_terminated = tuple(new_position) == self.terminal_state
        return [(1.0, new_state, -1, is_terminated)]

    def step(self, a: int) -> Tuple:
        """
        Parameters
        ----------
        a : int
            Action to be performed

        Returns
        -------
        Tuple
            Tuple of ``(new_state, reward, terminated, truncated, extra_info)``

        Raises
        ------
        ValueError
            If an invalid action is given
        """

        # Update tornados
        tornado_position = np.unravel_index(self.s_tornado, self.shape)
        s_tornado = self._update_tornado(tornado_position)

        # Calculate transition probabilities and rewards
        elf_position = np.unravel_index(self.s_elf, self.shape)
        if a == UP:
            transitions = self._update_elf(elf_position, tornado_position, [-1, 0])
        elif a == RIGHT:
            transitions = self._update_elf(elf_position, tornado_position, [0, 1])
        elif a == DOWN:
            transitions = self._update_elf(elf_position, tornado_position, [1, 0])
        elif a == LEFT:
            transitions = self._update_elf(elf_position, tornado_position, [0, -1])
        else:
            raise ValueError("Invalid action.")

        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s_elf, r, t = transitions[i]

        # Perform step
        self.s_tornado = s_tornado
        self.s_elf = s_elf
        self.lastaction = a

        if self.render_mode == "human":
            self.render()

        return ([int(s_elf), int(s_tornado)], r, t, False, {"prob": p})

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[List[int], Dict]:
        """
        Parameters
        ----------
        seed : Optional[int], optional
            Random seed, by default None

        Returns
        -------
        Tuple[List[int], Dict]
            Tuple of  ``(new_state, extra_info)``
        """
        super().reset(seed=seed, **kwargs)

        self.s_tornado = categorical_sample(self.tornado_initial_st_dist, self.np_random)
        self.s_elf = categorical_sample(self.elf_initial_st_dist, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return [int(self.s_elf), int(self.s_tornado)], {"prob": 1}

    def render(self) -> Optional[Union[str, np.ndarray]]:
        """Render function

        Returns
        -------
        Optional[Union[str, np.ndarray]]
            Depends on the given render mode.
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode: Literal["human", "rgb_array"]) -> Optional[np.ndarray]:
        """Render in human or RGB array mode

        Parameters
        ----------
        mode : Literal["human", "rgb_array"]
            Rendering mode

        Returns
        -------
        Optional[np.ndarray]
            None if mode if "human", or array in "RGB array" mode

        Raises
        ------
        DependencyNotInstalled
            Raise if pygame library is not installed
        """
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled("pygame is not installed, run `pip install gymnasium[toy-text]`") from e
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("TornadoCliffWalking")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.elf_images is None:
            hikers = [
                path.join(path.dirname(__file__), "img/elf_up.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_left.png"),
            ]
            self.elf_images = [pygame.transform.scale(pygame.image.load(f_name), self.cell_size) for f_name in hikers]
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/cookie.png")
            self.goal_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
        if self.mountain_bg_img is None:
            bg_imgs = [
                path.join(path.dirname(__file__), "img/mountain_bg1.png"),
                path.join(path.dirname(__file__), "img/mountain_bg2.png"),
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size) for f_name in bg_imgs
            ]
        if self.near_cliff_img is None:
            near_cliff_imgs = [
                path.join(path.dirname(__file__), "img/mountain_near-cliff1.png"),
                path.join(path.dirname(__file__), "img/mountain_near-cliff2.png"),
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size) for f_name in near_cliff_imgs
            ]
        if self.cliff_img is None:
            file_name = path.join(path.dirname(__file__), "img/mountain_cliff.png")
            self.cliff_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)

        if self.tornado_img is None:
            file_name = path.join(path.dirname(__file__), "img/tornado.png")
            self.tornado_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)

        for s in range(self.nS_elf):
            row, col = np.unravel_index(s, self.shape)
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            self.window_surface.blit(self.mountain_bg_img[check_board_mask], pos)

            # Render Cliff
            if self._cliff[row, col]:
                self.window_surface.blit(self.cliff_img, pos)

            if row < self.shape[0] - 1 and row + 1 == np.where(self._cliff == 1)[0][0] and self._cliff[row + 1, col]:
                self.window_surface.blit(self.near_cliff_img[check_board_mask], pos)

            # Render start state
            if s == self.elf_start_state_index:
                self.window_surface.blit(self.start_img, pos)

            # Render goal
            if s == self.nS_elf - 1:
                self.window_surface.blit(self.goal_img, pos)

            # Render Elf
            if s == self.s_elf:
                elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                last_action = self.lastaction if self.lastaction is not None else 2
                self.window_surface.blit(self.elf_images[last_action], elf_pos)

            # Render Tornado
            if s == self.s_tornado:
                tornado_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                self.window_surface.blit(self.tornado_img, tornado_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2))

    def _render_text(self) -> str:
        """Render in text mode

        Returns
        -------
        str
            Text representation.
        """
        outfile = StringIO()

        for s in range(self.nS_elf):
            position = np.unravel_index(s, self.shape)
            if self.s_elf == s:
                output = " x "
            # Print terminal state
            elif position == (3, 11):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()


# Tornado from https://www.freepik.com/
# All other from by https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/toy_text/img # noqa :

if __name__ == "__main__":
    env = TornadoCliffEnv(render_mode="human")
    observation, info = env.reset()

    for _ in range(100):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
