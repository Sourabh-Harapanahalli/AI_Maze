# AI Maze Solver

## 📌 Overview
AI Maze Solver is a Python-based interactive visualization tool designed to demonstrate the efficiency of different pathfinding algorithms in navigating a randomly generated maze. This project serves as an educational tool for understanding fundamental search algorithms and their real-world applications, such as robotics, game AI, and network routing.

The maze consists of a **10x10 grid** where:
- **S (Start Point)** represents the entry position of the search.
- **G (Goal Point)** is the target destination.
- **X (Obstacles)** are barriers preventing movement.
- **. (Empty Spaces)** are traversable paths.

Users can visualize and compare the behavior of three different algorithms:
1. **Depth-First Search (DFS)** - A backtracking-based approach that explores as deeply as possible before retracing steps.
2. **Breadth-First Search (BFS)** - A level-wise search that guarantees the shortest path in an unweighted maze.
3. **A* Algorithm** - A heuristic-based search using cost functions to optimize the shortest path.

The project offers a **real-time graphical interface**, allowing users to:
- Generate random mazes.
- Execute different search algorithms.
- Observe algorithmic decision-making in real time.
- Adjust execution speed for better analysis.
- Evaluate algorithm performance based on execution time, nodes visited, and solution path length.

This tool is useful for students, researchers, and developers interested in artificial intelligence, graph theory, and algorithm visualization.

## 🎯 Features
- 📍 **Maze Generation**: Generates random 10x10 mazes with start, goal, and obstacles.
- 🏃 **Pathfinding Algorithms**:
  - **DFS (Depth-First Search)**
  - **BFS (Breadth-First Search)**
  - **A* Search Algorithm**
- 📊 **Performance Evaluation**: Compares the efficiency of different algorithms.
- ⏱ **Real-Time Visualization**: Displays the search progress live.
- ⚡ **Speed Control**: Adjusts the execution speed dynamically.

## 🛠 Installation
To run this project, ensure you have Python installed.

### **1. Clone the Repository**
```bash
git clone https://github.com/sourabh-harapanahalli/AI_Maze.git
cd AI_Maze
```

### **2. Install Dependencies**
```bash
pip install numpy pandas
```

## 🚀 Usage
Run the script using:
```bash
python MAZE.py
```

### **Controls:**
- 🎲 **Generate Maze**: Creates a new random maze.
- 🔄 **Reset Maze**: Restores the initial maze.
- 🚀 **Run DFS**: Executes Depth-First Search.
- 🚀 **Run BFS**: Executes Breadth-First Search.
- 🚀 **Run A***: Executes A* Algorithm.
- ⏩ **Adjust Speed**: Increases or decreases animation speed.

## 🔬 Algorithms Used
### **1️⃣ Depth-First Search (DFS)**
DFS explores as deep as possible before backtracking. It is not guaranteed to find the shortest path.

### **2️⃣ Breadth-First Search (BFS)**
BFS explores all neighbors before moving deeper. It guarantees the shortest path if all moves have equal cost.

### **3️⃣ A* Search Algorithm**
A* is an informed search algorithm using a heuristic function to efficiently find the shortest path.

## 📊 Performance Evaluation
To provide insights into the efficiency of different search algorithms, this project includes a **performance evaluation module** that runs and compares the three algorithms—DFS, BFS, and A*. The following metrics are used for comparison:

- **Solution Path Length**: Measures the number of steps taken to reach the goal.
- **Nodes Expanded**: Counts the number of nodes visited during the search.
- **Execution Time (ms)**: Measures the total time taken for each algorithm to complete the search.

### **Evaluation Process**
The program runs each algorithm on a set of randomly generated mazes and logs the results in a structured format. The evaluation process follows these steps:
1. Generate a new random maze with obstacles, start, and goal positions.
2. Run DFS, BFS, and A* algorithms sequentially.
3. Record the number of visited nodes, solution path length, and execution time for each algorithm.
4. Save the collected performance data in a structured table.

### **Performance Analysis & Expected Results**
- **DFS** tends to explore deeper paths first and may take longer to find an optimal path, often visiting more nodes than necessary.
- **BFS** guarantees the shortest path in an unweighted maze and is usually more efficient than DFS in terms of path length.
- **A*** optimizes the search using heuristics, often reducing the number of nodes expanded and execution time.

The results of multiple runs are stored in an output file (`Performance_output_random_<timestamp>.xlsx`), allowing for comparative analysis of the algorithms over different maze configurations.

## 📜 License
This project is licensed under the **MIT License**.

