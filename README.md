# Code for “Optimizing Self-Organized Volunteer Efforts in Response to the COVID-19 Pandemic”
## Authors: Anping Zhang1†,Ke Zhang1†, Wanda Li1, Yue Wang2,Yang Li1∗, Lin Zhang1

## NCE computation: Compute self-organizational intervals on Shenzhen’s data
### Note:
1.	O_NCE.csv. T_NCE.csv and P_NCE.csv are pre-computed NCEs for Shenzhen and its district using data files “issuer_task_data.csv” and “issuer_user_data”.
2.	The “task label” column in organizer_task_data.csv represents the task type extracted from task descriptions using LDA. 
a)	Label 1: Transportational Topic tasks; 
b)	Label 2: volunteering topic tasks; 
c)	Label 3: Reopening Topic tasks;
d)	Label 4: Educational topic tasks; 
e)	Label 5: environmental topic tasks; 
f)	Label 6: Covid-19 topic tasks.
3.	“neigborhood_1.csv” and “neighborhood_2.csv” are data for case studies.

### To run:
1.	Run NCE.ipynb to generate an NCE plot with color shaded self-organization intervals

## Causality Analysis: Causality analysis on what dynamic factors have caused self-organization events.
### Note:
1.	causality_data.csv contains three types NCE, internal and external variables (policies impulse and covid-19 daily new cases)
2.	all_diff_data.csv is differencing from causality_data.csv to make sure our time-series data is stationary for causality analysis
### To run: 
1.	Install tigramite package from https://github.com/jakobrunge/tigramite.git
2.	Install graphviz package from https://graphviz.org/download/
3.	Run Causality_analysis.ipynb to obtain full causal graphs for self-organization intervals

## Simulation part: A simulation of users participating in a fixed number of tasks.
### Simulation rules:
1. Simulation is initialized with a fixed number of agents and tasks  
1. Each task is represented by a cell in a 2D grid. All tasks have a limit on
   the number of agents it can recruit: `max_agent_per_cell`;
2. At each step, each agent decides whether to participate in a task with
      probability, `p_participate.`
3. If the agent is participating, it will join the first available task
   from its recent participation history within a time window, ordered by highest frequency;
   (To simulate the user behavior of participating in the same task.)
   If no space is available for all these tasks, it will join a random available task nearby the current task.

### To run:
1. Install mesa package from https://mesa.readthedocs.io/
2. Run UserModel.py to run the simulation
3. paint_subplot_simulation.py is used to draw NCEs and gains under different parameters. People can change the parameters in UserModel.py to get NCE and gains.

