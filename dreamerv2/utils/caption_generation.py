"""This module generates captions for the curent state of the environment of the game"""

import numpy as np
from abc import ABC, abstractmethod


class CaptionBase(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def generate_caption(self, obs):
        pass


class PomdpBreakoutCaptioner(CaptionBase):
    def __init__(self, env):
        super().__init__(env)

    def _get_paddle_caption(self, paddle_obs: np.ndarray) -> str:
        """Generates a caption for the paddle

        Args:
            paddle_obs (np.ndarray): array of shape (10, 10) where:
                dim 0 = (y)
                dim 1 = (x)
            and true if the paddle is present in the cell

        Returns:
            str: Caption for position of the paddle
        """
        # Don't care about y position as it is always at the bottom
        _, paddle_pos = np.where(paddle_obs == 1)
        # Convert from array obj to int
        paddle_pos = int(paddle_pos)
        if paddle_pos < 2:
            return "The paddle is at the far left of the screen. "
        if paddle_pos > 7:
            return "The paddle is at the far right of the screen. "
        if paddle_pos < 4:
            return "The paddle is on the left side of the screen. "
        if paddle_pos > 5:
            return "The paddle is on the right side of the screen. "
        else:
            return "The paddle is in the middle of the screen. "

    def _get_ball_caption(self, ball_obs: np.ndarray) -> str:
        """Generates a caption for the paddle

        Args:
            paddle_obs (np.ndarray): array of shape (10, 10) where:
                dim 0 = (y)
                dim 1 = (x)
            and true if the ball is present in the cell

        Returns:
            str: Captionf for the position of the ball
        """

        ball_y_pos, ball_x_pos = np.where(ball_obs == 1)
        # Convert from array obj to ints
        ball_x_pos = int(ball_x_pos)
        ball_y_pos = int(ball_y_pos)

        x_str = "The ball is on the far left side"
        y_str = "at the very top"

        if ball_x_pos == 2 or ball_x_pos == 3:
            x_str = "The ball is on the left side"
        elif ball_x_pos == 4 or ball_x_pos == 5:
            x_str = "The ball is in the centre"
        elif ball_x_pos == 6 or ball_x_pos == 7:
            x_str = "The ball is on the right side"
        elif ball_x_pos == 8 or ball_x_pos == 9:
            x_str = "The ball is on the far right side"

        if ball_y_pos == 1:
            y_str = "just below the very top"
        elif ball_y_pos == 2 or ball_y_pos == 3:
            y_str = "at the top"
        elif ball_y_pos == 4 or ball_y_pos == 5:
            y_str = "in the middle"
        elif ball_y_pos == 6 or ball_y_pos == 7:
            y_str = "at the bottom"
        elif ball_y_pos == 8:
            y_str = "just above the very bottom"
        elif ball_y_pos == 9:
            y_str = "at the very bottom"

        return f"{x_str} and is {y_str} of the screen."

    def _get_bricks_caption(self, bricks_obs: np.ndarray) -> str:
        """Generates a caption for the paddle

        Args:
            paddle_obs (np.ndarray): array of shape (10, 10) where:
                dim 0 = (y)
                dim 1 = (x)
            and true if the paddle is present in the cell

        Returns:
            str: image caption
        """

        def get_semantic_position_of_layer(layer: np.ndarray, layer_name: str) -> str:
            """Gets

            Args:
                layer (np.ndarray): _description_
                layer_name (str): _description_

            Returns:
                str: _description_
            """
            if np.count_nonzero(layer) == 0:
                return f"There are no bricks in the {layer_name} layer. "
            if np.count_nonzero(layer) == 10:
                return f"All the bricks remain in the {layer_name} layer. "
            far_left_str = (
                f"There are no bricks on the far left of the {layer_name} layer. "
            )
            left_str = f"There are no bricks on the left of the {layer_name} layer. "
            middle_str = (
                f"There are no bricks in the middle of the {layer_name} layer. "
            )
            right_str = f"There are no bricks on the right of the {layer_name} layer. "
            far_right_str = (
                f"There are no bricks on the far right of the {layer_name} layer. "
            )
            if layer[4] and layer[5]:
                middle_str = (
                    f"There are two bricks in the middle of the {layer_name} layer. "
                )
            elif layer[4] or layer[5]:
                middle_str = (
                    f"There is one brick in the middle of the {layer_name} layer. "
                )
            if layer[0] and layer[1]:
                far_left_str = (
                    f"There are two bricks on the far left of the {layer_name} layer. "
                )
            elif layer[0] or layer[1]:
                far_left_str = (
                    f"There is one brick on the far left of the {layer_name} layer. "
                )
            if layer[8] and layer[9]:
                far_right_str = (
                    f"There are two bricks on the far right of the {layer_name} layer. "
                )
            elif layer[8] or layer[9]:
                far_right_str = (
                    f"There is one brick on the far right of the {layer_name} layer. "
                )
            if layer[2] and layer[3]:
                left_str = (
                    f"There are two bricks on the left of the {layer_name} layer. "
                )
            elif layer[2] or layer[3]:
                left_str = f"There is one brick on the left of the {layer_name} layer. "
            if layer[6] and layer[7]:
                right_str = (
                    f"There are two bricks on the right of the {layer_name} layer. "
                )
            elif layer[6] or layer[7]:
                right_str = (
                    f"There is one brick on the right of the {layer_name} layer. "
                )
            return far_left_str + left_str + middle_str + right_str + far_right_str

        if np.count_nonzero(bricks_obs) == 30:
            return "All the bricks remain. "
        if np.count_nonzero(bricks_obs) == 0:
            return "No bricks remain. "

        return (
            get_semantic_position_of_layer(bricks_obs[1], "top")
            + get_semantic_position_of_layer(bricks_obs[2], "middle")
            + get_semantic_position_of_layer(bricks_obs[3], "bottom")
        )

    def generate_caption(self, obs: np.ndarray) -> str:
        """Generates a caption for the current observation image

        Args:
            obs (np.ndarray): array of shape (3, 10, 10) where:
                dim 0 = (paddle, ball, bricks)
                dim 1 = (y)
                dim 2 = (x)
            and true if the object is present in the cell

        Returns:
            str: image caption
        """
        paddle_caption = self._get_paddle_caption(obs[0])
        bricks_caption = self._get_bricks_caption(obs[2])
        ball_caption = self._get_ball_caption(obs[1])

        return paddle_caption + bricks_caption + ball_caption
