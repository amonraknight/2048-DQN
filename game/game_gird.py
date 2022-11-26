from tkinter import Frame, Label, CENTER, Button
import threading
import numpy as np
import math
import copy
import torch

import config as c
from game_nodisplay import Game2048NoDisplay


class GameGrid(Frame):
    def __init__(self):
        frame = Frame.__init__(self)
        self.step_count = 0
        self.grid()
        self.master.title('2048')

        # A button to start auto-solve
        Button(frame, text='DQN play', command=self.start_dqn_thread).grid(column=0, row=1)
        ## Button(frame, text='Refresh', command=self.refresh).grid(column=1, row=2)

        self.master.bind("<Key>", self.key_down)

        self.game = Game2048NoDisplay()
        self.grid_cells = []
        self.init_grid()
        self.game.reset()
        self.matrix = self.game.matrix

        # start main loop
        self.update_grid_cells()
        self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def refresh(self):
        self.game.reset()
        self.matrix = self.game.matrix
        self.update_grid_cells()

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()

    def key_down(self, event):
        print(event)
        print('Manual operation not allowed.')
        '''
        if key == c.KEY_QUIT:
            exit()
        elif key in self.commands:
            self.commit_move(key)
        '''

    def start_dqn_thread(self):
        thread = threading.Thread(target=self.auto_solve, args=())
        thread.setDaemon(True)
        thread.start()

    def auto_solve(self):
        print('DQN solution...')

        # Load model
        self.game.agent.read_latest_module()

        cells, status, action_l = self.game.reset()
        self.matrix = self.game.matrix
        self.update_grid_cells()

        step_count = 0

        # Prepare the state in tensor[[]]
        state = torch.from_numpy(cells).type(torch.FloatTensor)
        state = torch.unsqueeze(state, 0)
        invalid_step = False
        while status == 'not over':
            action = self.game.agent.get_action(state, action_l, c.NUM_EPISODES, is_random=invalid_step)

            if action_l[action] == 1:
                # Commit the step: tensor.item converts the tensor with a single item to a single value.
                state_next, status, action_l, reward = self.game.step(action.item())
                step_count += 1
                invalid_step = False
            else:
                invalid_step = True
                state_next = cells
            # Update the gird.
            self.matrix = self.game.matrix
            self.update_grid_cells()

            state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
            state_next = torch.unsqueeze(state_next, 0)

            state = state_next

            if status == 'win' or status == 'lose':
                print('Finished after %d steps in status %s score %d.' % (step_count, status, self.game.total_score))

                if status == 'win':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                elif status == 'lose':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                break



if __name__ == "__main__":
    game_grid = GameGrid()
