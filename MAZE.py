#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import tkinter as tk
from collections import deque
from tkinter import *
import time
import heapq
import pandas as pd
from datetime import datetime

global nodes_visited,solution_path,df
df=pd.DataFrame([])

rows,col = (10,10)
start=(0,0)
goal=(9,9)
obstacle=(0,1)

maze=[]

for x in range(rows):
    lst=[]
    for y in range(col):
        if x==start[0] and y==start[1]:
            lst.append("S")
        elif x==goal[0] and y==goal[1]:
            lst.append("G")
        elif x==obstacle[0] and y==obstacle[1]:
            lst.append("X")
        else:
            lst.append(".")
    print(lst)
    maze.append(lst)
    


class Maze:
    def __init__(self,master, rows, cols):
        self.master = master
        self.master.title("AI Project 1 - Maze")
        self.master.geometry("630x700")

        self.rows = rows
        self.cols = cols
        self.cell_size = 42
        
        self.speed=0

        # Initialize a random maze
        #self.maze = [[random.choice(['W', ' ', ' ']) for _ in range(cols)] for _ in range(rows)]
        self.maze = maze
        self.current_maze=maze

        self.canvas = tk.Canvas(master, width=(cols * self.cell_size)+30, height=(rows * self.cell_size)+30)
        self.canvas.pack()
        
        
        self.perform_random = tk.Button(master, text="Performance Evaluation random maze", command=self.evaluate_performance_random,font=("Helvetica", 12))
        self.perform_random.place(relx=0.275, rely=0.9, anchor=CENTER)
        
        self.reset_button = tk.Button(master, text="Generate Maze", command=self.generate_maze,font=("Helvetica", 12))
        self.reset_button.place(relx=0.6, rely=0.75, anchor=CENTER)
        #self.reset_button.pack()
        
        self.run_dfs = tk.Button(master, text="Run A*", command=self.a_star_final,font=("Helvetica", 12))
        self.run_dfs.place(relx=0.68, rely=0.65, anchor=CENTER)
        #self.run_dfs.pack()
        
        self.run_bfs = tk.Button(master, text="Run BFS", command=self.bfs_final,font=("Helvetica", 12))
        self.run_bfs.place(relx=0.48, rely=0.65, anchor=CENTER)
        #self.run_dfs.pack()
        
        self.run_astar = tk.Button(master, text="Run DFS", command=self.Run_dfs,font=("Helvetica", 12))
        self.run_astar.place(relx=0.28, rely=0.65, anchor=CENTER)
        #self.run_dfs.pack()
        
        self.maze_reset = tk.Button(master, text="Reset Maze", command=self.reset_maze,font=("Helvetica", 12))
        self.maze_reset.place(relx=0.35, rely=0.75, anchor=CENTER)
        
        self.speed_label = tk.Label(master, text="Adjust Speed: ",font=("Helvetica", 12))
        self.speed_label.place(relx=0.85, rely=0.73, anchor=CENTER)
        
        self.display_label = tk.Label(master, text=str(self.speed), font=("Helvetica", 12))
        self.display_label.place(relx=0.8, rely=0.8, anchor=CENTER)

        # Create the increase button
        self.increase_button = tk.Button(master, text="▲", command=self.increase_number)
        self.increase_button.place(relx=0.9, rely=0.78, anchor=CENTER)

        # Create the decrease button
        self.decrease_button = tk.Button(master, text="▼", command=self.decrease_number)
        self.decrease_button.place(relx=0.9, rely=0.83, anchor=CENTER)
        
        self.is_running = False
        self.start_time = None
        self.elapsed_time = 0

        # Create the display labels for hours, minutes, and seconds        
        self.minutes_label = tk.Label(master, text="Timer \n (min,sec,ms)", font=("Helvetica", 12))
        self.minutes_label.place(relx=0.16, rely=0.78, anchor=CENTER)
        
        self.hours_label = tk.Label(master, text="00 |:|", font=("Helvetica", 12))
        self.hours_label.place(relx=0.09, rely=0.83, anchor=CENTER)

        self.minutes_label = tk.Label(master, text="00 |:|", font=("Helvetica", 12))
        self.minutes_label.place(relx=0.15, rely=0.83, anchor=CENTER)

        self.seconds_label = tk.Label(master, text="00", font=("Helvetica", 12))
        self.seconds_label.place(relx=0.25, rely=0.83, anchor=CENTER)
        
        self.draw_maze()
        #self.update_display_stopwatch()
        
    def increase_number(self):
        self.speed += 1
        self.update_display()

    def decrease_number(self):
        if self.speed > 0:  # Check if number is greater than 0
            self.speed -= 1
            self.update_display()

    def update_display(self):
        self.display_label.config(text=str(self.speed))
        
    def start_stopwatch(self):
        self.is_running=True
        if not self.is_running:
            self.start_time = time.time() - self.elapsed_time
            self.is_running = True

    def stop_stopwatch(self):
        if self.is_running:
            self.is_running = False

    def reset_stopwatch(self):
        self.start_time = time.time()
        self.elapsed_time = 0
        self.is_running=False
        self.seconds_label.config(text="0.000")
    
    def update_display_stopwatch(self):
        if self.is_running:
            self.elapsed_time = time.time() - self.start_time

        # Calculate hours, minutes, and seconds
        hours, rem = divmod(self.elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # Update the display labels
        self.hours_label.config(text="{:02d} |".format(int(hours)))
        self.minutes_label.config(text="{:02d} |".format(int(seconds)))
        #self.seconds_label.config(text="{:02d}".format(int(seconds)))
        if self.start_time is not None:
            self.seconds_label.config(text="{:.3f}".format(round((time.time() - self.start_time) * 1000,3))) 
        else:
            self.seconds_label.config(text="0.000")



        # Schedule the next update after 100 milliseconds
        self.master.after(100, self.update_display)
        
        
    def generate_maze(self):
        size = 10  # Adjust the size of the maze here
        num_obstacles = random.randint(5,15)  # Adjust the number of obstacles here
        maze = [['.' for _ in range(size)] for _ in range(size)]

        # Set start point
        start_x, start_y = random.randint(0, size - 1), random.randint(0, size - 1)
        maze[start_x][start_y] = 'S'

        # Set goal point
        goal_x, goal_y = random.randint(0, size - 1), random.randint(0, size - 1)
        while goal_x == start_x and goal_y == start_y:
            goal_x, goal_y = random.randint(0, size - 1), random.randint(0, size - 1)
        maze[goal_x][goal_y] = 'G'

        # Set obstacles
        for _ in range(num_obstacles):
            obstacle_x, obstacle_y = random.randint(0, size - 1), random.randint(0, size - 1)
            while (obstacle_x == start_x and obstacle_y == start_y) or (obstacle_x == goal_x and obstacle_y == goal_y) or maze[obstacle_x][obstacle_y] == 'X':
                obstacle_x, obstacle_y = random.randint(0, size - 1), random.randint(0, size - 1)
            maze[obstacle_x][obstacle_y] = 'X'
            
        print(maze)
        self.current_maze=maze
        self.maze=self.current_maze
        self.draw_maze()

        #return self.maze
        

        
    def reset_maze(self):
        self.reset_stopwatch()
        self.seconds_label.config(text="0.000")
        # Reset the maze to its initial state
        self.maze = self.current_maze
        print("\n")
        for row in self.current_maze:
            print(' '.join(row))
        self.draw_maze()
        
    def draw_maze(self):
        self.update_display_stopwatch()
        self.canvas.delete("all")  # Clear the canvas

        for i in range(self.rows):
            for j in range(self.cols):
                x0, y0 = j * self.cell_size, i * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size

                cell_type = self.maze[i][j]

                if cell_type == 'S':  # Wall
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill='black')
                elif cell_type == 'G':
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill='green')
                elif cell_type == 'X':
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill='red')
                elif cell_type == '-':
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill='yellow')
                elif cell_type == '*':
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill='teal')
                else:  # Path
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill='white')

        # After drawing, schedule the next update
        #self.master.after(1000, self.update_maze)

        
    def check(self,array,index):
        if array[index]=="-":
            return None
        elif array[index]==".":
            return index
        elif array[index]=="G":
            return index



    def next_step(self,arr,current):
        self.update_display_stopwatch()
        possible_pos=[]
        #top-left
        if current[0]==0 and current[1]==0:
            possible_pos.append([self.check(arr,(current[0]+1,current[1])),self.check(arr,(current[0],current[1]+1))])
        #bottom-right
        elif current[0]==(len(arr)-1) and current[1]==(len(arr)-1):
            possible_pos.append([self.check(arr,(current[0]-1,current[1])),self.check(arr,(current[0],current[1]-1))])
        #top-right
        elif current[0]==0 and current[1]==(len(arr)-1):
            possible_pos.append([self.check(arr,(current[0]+1,current[1])),self.check(arr,(current[0],current[1]-1))])
        #bottom-left
        elif current[0]==(len(arr)-1) and current[1]==0:
            possible_pos.append([self.check(arr,(current[0]-1,current[1])),self.check(arr,(current[0],current[1]+1))])

        #first column
        elif 0<current[0]<(len(arr)-1) and current[1]==0: 
            possible_pos.append([self.check(arr,(current[0]+1,current[1])),self.check(arr,(current[0],current[1]+1)),self.check(arr,(current[0]-1,current[1]))])

        #last column
        elif 0<current[0]<(len(arr)-1) and current[1]==(len(arr)-1): 
            possible_pos.append([self.check(arr,(current[0]+1,current[1])),self.check(arr,(current[0],current[1]-1)),self.check(arr,(current[0]-1,current[1]))])

        #first row
        elif current[0]==0 and 0<current[1]<(len(arr)-1):
            possible_pos.append([self.check(arr,(current[0]+1,current[1])),self.check(arr,(current[0],current[1]+1)),self.check(arr,(current[0],current[1]-1))])

        #last row
        elif current[0]==(len(arr)-1) and 0<current[1]<(len(arr)-1):
            possible_pos.append([self.check(arr,(current[0]-1,current[1])),self.check(arr,(current[0],current[1]+1)),self.check(arr,(current[0],current[1]-1))])

        elif current[0]>0 and current[1]>0:
            possible_pos.append([self.check(arr,(current[0]+1,current[1])),self.check(arr,(current[0],current[1]+1)),self.check(arr,(current[0]-1,current[1])),self.check(arr,(current[0],current[1]-1))])

        return possible_pos



    def Run_dfs(self):
        global nodes_visited,solution_path
        self.draw_maze()
        self.maze = np.array(self.current_maze)
        arr = np.array(self.maze)
        visited_nodes=[]

        goal_reached=False
        
        goal_unreachable=False
        
        goal=(np.where(arr=="G")[0][0],np.where(arr=="G")[1][0])
        pop_count=0

        self.reset_stopwatch()
        self.start_stopwatch()
        self.update_display_stopwatch()
        
        sequence_nodes=[]

        while goal_reached==False:
            self.update_display_stopwatch()
            if (np.where(arr=="S")[0][0]!=goal[0]) or (np.where(arr=="S")[1][0]!=goal[1]):

                current=(np.where(arr=="S")[0][0],np.where(arr=="S")[1][0])
                sequence_nodes.append(current)
                filtered_list = list(filter(lambda x: x is not None, self.next_step(arr,current)[0]))

                if len(filtered_list)==0:
                    goal_unreachable=True
                    print("Goal Unreachable.. backtrack and try different node")
                    if pop_count>0:
                        visited_nodes.pop()
                    arr[current]="-"
                    if len(visited_nodes)==0:
                        print("Goal cannot be reached")
                        break
                    current=visited_nodes[-1]
                    arr[current]="S"
                    pop_count=1
                    continue
                random_choice=random.choice(filtered_list)
                visited_nodes.append(current)
                arr[current]="-"
                arr[random_choice]="S"
                print(arr)
                print("------------------------------------------------------------------")
                pop_count=0
                self.maze=arr
                self.master.update()
                self.draw_maze()
                self.master.after(int(self.speed)*100)
                
            else:
                print("Goal Reached!!"+"\n")
                goal_reached=True
                for x,y in visited_nodes:
                    arr[(x,y)]='*'
                self.draw_maze()
        self.stop_stopwatch()
        #print(visited_nodes)
        print("Visited Nodes - "+"\n" + str(sequence_nodes))
        nodes_visited=sequence_nodes
        solution_path=visited_nodes
    
    
    def bfs(self, start, goal):
        global nodes_visited,solution_path
        self.draw_maze()
        queue = deque([(start, [])])
        maze=self.maze
        visited = set()
        self.update_display_stopwatch()
        
        while queue:
            current, path = queue.popleft()

            if current == goal:
                return path + [current]

            if current in visited:
                continue

            visited.add(current)
            self.bfs_mark_visited(current)  # Mark visited nodes
            #time.sleep(0.1)  # Introduce a delay for visualization (adjust as needed)

            row, col = current
            neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]

            for neighbor in neighbors:
                n_row, n_col = neighbor
                if 0 <= n_row < len(maze) and 0 <= n_col < len(maze[0]) and maze[n_row][n_col] != 'X' and (n_row, n_col) not in visited:
                    queue.append(((n_row, n_col), path + [current]))
                    self.maze=maze
                    self.master.update()
                    self.draw_maze()
                    self.master.after(int(self.speed)*100)
                    
            nodes_visited=visited
            solution_path=(path + [current])
                
            

        return None  # No path found

    def bfs_mark_visited(self, position):
        maze=self.maze
        row, col = position
        if maze[row][col] not in ('S', 'G', '*'):
            maze[row][col] = '-'  # Symbol for visited nodes
        for row in maze:
            print(' '.join(row))
        print("\n")

    def bfs_mark_path(self, path):
        maze=self.maze
        for i, (row, col) in enumerate(path):
            if maze[row][col] == 'G':
                maze[row][col] = 'G'  # Mark the goal point
            elif maze[row][col]=="-":
                maze[row][col] = '*'  # Mark the last step of the optimal path
            elif maze[row][col] not in ('S', '-', '*'):
                maze[row][col] = '+'  # Symbol for other steps in the optimal path

    # Example: Generate a 10x10 maze
    #rows = 10
    #cols = 10
    #maze = generate_maze(rows, cols)

    def bfs_final(self):
        global nodes_visited,solution_path
        # Find starting point (S) and goal point (G) in the maze
        start = None
        goal = None
        self.draw_maze()
        self.maze = np.array(self.current_maze)
        print("-------Current------")
        print(self.current_maze)
        print("_---------------")
        self.draw_maze()
        self.master.update()
        maze=np.array(self.maze)
        print(maze)
        self.reset_stopwatch()
        self.start_stopwatch()
        self.update_display_stopwatch()
        
        for row_idx, row in enumerate(maze):
            for col_idx, cell in enumerate(row):
                if cell == 'S':
                    start = (row_idx, col_idx)
                elif cell == 'G':
                    goal = (row_idx, col_idx)

        if start and goal:
            result = self.bfs(start, goal)
            if result:
                #print("Result path" +str(result))
                self.bfs_mark_path(result)
                self.draw_maze()
                print("Optimal Path:")
                #for row in maze:
                #    print(' '.join(row))
            else:
                print("No path found.")
        else:
            print("Starting point (S) or goal point (G) not found in the maze.")
        self.stop_stopwatch()
        print("-------Current------")
        print(self.current_maze)
        print("_---------------")
        
    def astar(self, start, goal):
        global nodes_visited,solution_path
        self.update_display_stopwatch()
        maze=self.maze
        # Define the heuristic function
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Define the cost function
        def cost(current, next):
            return 1

        # Initialize the open and closed lists
        open_list = []
        closed_list = set()
        heapq.heappush(open_list, (0, start))

        # Initialize the g and f scores
        g_scores = {start: 0}
        f_scores = {start: heuristic(start, goal)}

        # Initialize the came_from dictionary
        came_from = {}

        # Run the algorithm
        while open_list:
            # Get the node with the lowest f score
            current = heapq.heappop(open_list)[1]

            # Check if we have reached the goal
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            # Add the current node to the closed list
            closed_list.add(current)

            # Check the neighbors of the current node
            for neighbor in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                x = current[0] + neighbor[0]
                y = current[1] + neighbor[1]
                if 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] != 'X':
                    next = (x, y)
                    maze=np.array(maze)
                    if maze[next]==".":
                        maze[next]="-"
                    new_g_score = g_scores[current] + cost(current, next)
                    if next in g_scores and new_g_score >= g_scores[next]:
                        continue
                    g_scores[next] = new_g_score
                    f_scores[next] = new_g_score + heuristic(next, goal)
                    heapq.heappush(open_list, (f_scores[next], next))
                    came_from[next] = current
                    for row in maze:
                        print(' '.join(row))
                    print("\n")
                    self.maze=maze
                    self.master.update()
                    self.draw_maze()
                    self.update_display_stopwatch()
                    print(self.update_display_stopwatch())
                    self.master.after(int(self.speed)*100)
                    
        nodes_visited=current
        solution_path=path


        # If we get here, there is no path
        print("Goal Unreachable")
        return None

    # Run the A* algorithm
    def a_star_final(self):
        self.draw_maze()
        self.maze = np.array(self.current_maze)
        maze=np.array(self.maze)
        start_x=np.where(maze=="S")[0][0]
        start_y=np.where(maze=="S")[1][0]
        goal_x=np.where(maze=="G")[0][0]
        goal_y=np.where(maze=="G")[1][0]
        self.reset_stopwatch()
        self.start_stopwatch()
        self.update_display_stopwatch()
        
        path = self.astar((start_x, start_y), (goal_x, goal_y))
        maze=self.maze
        self.draw_maze()
        self.master.update()

        # Print the maze and the path
        
        for x in path:
            if maze[x[0]][x[1]]=="S":
                maze[x[0]][x[1]]="S"
            elif maze[x[0]][x[1]]=="G":
                maze[x[0]][x[1]]="G"
            elif maze[x[0]][x[1]]!="S" or maze[x[0]][x[1]]!="G":
                maze[x[0]][x[1]]="*"             
                
       
        self.maze=maze
        print(maze)
        self.draw_maze()
        self.stop_stopwatch()
        
    def evaluate_performance_random(self):
        global nodes_visited,solution_path,df
        self.status_p = tk.Label(self.master, text="Wait.......Running DFS", font=("Helvetica", 12))
        self.status_p.place(relx=0.7, rely=0.9, anchor=CENTER)
        data = {'Run','Algorithm','Solution_Path','Nodes_Expanded','Execution_Time(min:sec:ms)','Status'}

        # Creating a DataFrame using the dictionary
        df = pd.DataFrame(columns=data)

        # Displaying the DataFrame
        for x in range(1,11):
            self.generate_maze()
            print(x)
            self.Run_dfs()
            print("Seconds -"+self.minutes_label.cget("text")+" Mili Seconds -"+self.seconds_label.cget("text"))
            print(nodes_visited)
            print(solution_path)
            new_row = {'Run': x, 
                       'Algorithm': "DFS", 
                       'Solution_Path': len(solution_path), 
                       'Nodes_Expanded': len(nodes_visited), 
                       'Execution_Time(min:sec:ms)': str(self.hours_label.cget("text")+":"+self.minutes_label.cget("text")+":"+self.seconds_label.cget("text")), 
                       'Status': "Pass"}
            df = df.append(new_row, ignore_index=True)

            print(x)
            self.status_p.config(text="Wait.......Running BFS")
            self.bfs_final()
            print("Seconds -"+self.minutes_label.cget("text")+" Mili Seconds -"+self.seconds_label.cget("text"))
            print(nodes_visited)
            print(solution_path)
            new_row = {'Run': x, 
                       'Algorithm': "BFS", 
                       'Solution_Path': len(solution_path), 
                       'Nodes_Expanded': len(nodes_visited), 
                       'Execution_Time(min:sec:ms)': str(self.hours_label.cget("text")+":"+self.minutes_label.cget("text")+":"+self.seconds_label.cget("text")), 
                       'Status': "Pass"}
            df = df.append(new_row, ignore_index=True)

            print(x)
            self.status_p.config(text="Wait.......Running A*")
            self.a_star_final()
            print("Seconds -"+self.minutes_label.cget("text")+" Mili Seconds -"+self.seconds_label.cget("text"))
            print(nodes_visited)
            print(solution_path)
            new_row = {'Run': x, 
                       'Algorithm': "A*", 
                       'Solution_Path': len(solution_path), 
                       'Nodes_Expanded': len(nodes_visited), 
                       'Execution_Time(min:sec:ms)': str(self.hours_label.cget("text")+":"+self.minutes_label.cget("text")+":"+self.seconds_label.cget("text")), 
                       'Status': "Pass"}
            df = df.append(new_row, ignore_index=True)
        
            
        data_new = ['Run','Algorithm','Solution_Path','Nodes_Expanded','Execution_Time(min:sec:ms)','Status']
        df = df[data_new]
        self.status_p.config(text="Performance_output.xlsx downloaded")
        df_sorted = df.sort_values(by=['Algorithm', 'Run'], ascending=[False, True])
        #print(df_sorted)
        #print("---------------------")
        #print( df_sorted.groupby('Algorithm')[['Solution_Path', 'Nodes_Expanded']].describe())
        #print( df_sorted.groupby('Algorithm')[['Solution_Path', 'Nodes_Expanded']].mean())
        
        current_datetime = datetime.now()

        # Format the date and time
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        df_sorted.to_excel("Performance_output_random"+str(formatted_datetime.replace(':', '_'))+".xlsx", index=False) 

        
      
    
root = tk.Tk()
maze_gui = Maze(root, 10, 10)
root.mainloop()


# In[ ]:




