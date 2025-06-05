import tkinter as tk
from tkinter import messagebox
import random

class TicTacToe:
    def _init_(self, root):
        self.root = root
        self.root.title("AI Tic Tac Toe - Single Player")
        self.root.configure(bg="#222831")
        self.board = ["" for _ in range(9)]
        self.buttons = []
        self.player = "X"
        self.ai = "O"
        self.game_over = False
        self.create_widgets()
        self.draw_board()

    def create_widgets(self):
        title = tk.Label(self.root, text="Tic Tac Toe", font=("Arial", 28, "bold"), fg="#00adb5", bg="#222831")
        title.pack(pady=20)
        frame = tk.Frame(self.root, bg="#393e46")
        frame.pack()
        for i in range(9):
            btn = tk.Button(frame, text="", font=("Arial", 24, "bold"), width=5, height=2, bg="#eeeeee", fg="#222831", bd=0,
                            command=lambda i=i: self.player_move(i))
            btn.grid(row=i//3, column=i%3, padx=8, pady=8)
            self.buttons.append(btn)
        self.status = tk.Label(self.root, text="Your turn!", font=("Arial", 16), fg="#eeeeee", bg="#222831")
        self.status.pack(pady=10)
        reset_btn = tk.Button(self.root, text="Restart", font=("Arial", 14), bg="#00adb5", fg="#eeeeee", bd=0, command=self.reset_game)
        reset_btn.pack(pady=10)

    def draw_board(self):
        for i in range(9):
            self.buttons[i]["text"] = self.board[i]
            if self.board[i] == "X":
                self.buttons[i]["fg"] = "#222831"
            elif self.board[i] == "O":
                self.buttons[i]["fg"] = "#00adb5"
            else:
                self.buttons[i]["fg"] = "#222831"

    def player_move(self, idx):
        if self.board[idx] == "" and not self.game_over:
            self.board[idx] = self.player
            self.draw_board()
            if self.check_winner(self.player):
                self.status["text"] = "You win!"
                self.game_over = True
                messagebox.showinfo("Game Over", "Congratulations! You win!")
            elif "" not in self.board:
                self.status["text"] = "It's a draw!"
                self.game_over = True
                messagebox.showinfo("Game Over", "It's a draw!")
            else:
                self.status["text"] = "AI's turn..."
                self.root.after(500, self.ai_move)

    def ai_move(self):
        if not self.game_over:
            idx = self.best_move()
            self.board[idx] = self.ai
            self.draw_board()
            if self.check_winner(self.ai):
                self.status["text"] = "AI wins!"
                self.game_over = True
                messagebox.showinfo("Game Over", "AI wins! Try again.")
            elif "" not in self.board:
                self.status["text"] = "It's a draw!"
                self.game_over = True
                messagebox.showinfo("Game Over", "It's a draw!")
            else:
                self.status["text"] = "Your turn!"

    def best_move(self):
        best_score = -float('inf')
        move = None
        for i in range(9):
            if self.board[i] == "":
                self.board[i] = self.ai
                score = self.minimax(0, False)
                self.board[i] = ""
                if score > best_score:
                    best_score = score
                    move = i
        return move

    def minimax(self, depth, is_max):
        if self.check_winner(self.ai):
            return 10 - depth
        if self.check_winner(self.player):
            return depth - 10
        if "" not in self.board:
            return 0
        if is_max:
            best = -float('inf')
            for i in range(9):
                if self.board[i] == "":
                    self.board[i] = self.ai
                    best = max(best, self.minimax(depth+1, False))
                    self.board[i] = ""
            return best
        else:
            best = float('inf')
            for i in range(9):
                if self.board[i] == "":
                    self.board[i] = self.player
                    best = min(best, self.minimax(depth+1, True))
                    self.board[i] = ""
            return best

    def check_winner(self, player):
        win_states = [
            [0,1,2], [3,4,5], [6,7,8],
            [0,3,6], [1,4,7], [2,5,8],
            [0,4,8], [2,4,6]
        ]
        for state in win_states:
            if all(self.board[i] == player for i in state):
                for i in state:
                    self.buttons[i]["bg"] = "#ff5722"
                return True
        return False

    def reset_game(self):
        self.board = ["" for _ in range(9)]
        self.game_over = False
        for btn in self.buttons:
            btn["bg"] = "#eeeeee"
        self.draw_board()
        self.status["text"] = "Your turn!"

if __name__ == "_main_":
    root = tk.Tk()
    root.geometry("400x550")
    app = TicTacToe(root)
    root.mainloop()