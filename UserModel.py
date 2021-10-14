from mesa import Agent, Model
from mesa.time import RandomActivation
import matplotlib.pyplot as plt
from mesa.space import MultiGrid
import numpy as np
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
from sortedcollections import ValueSortedDict
from pdb import set_trace as bp
from sklearn.preprocessing import normalize
import collections
import pandas as pd
from itertools import chain

from functools import partial


# parameter: p_participate
class UserAgent(Agent):
    """A user agent
    Parameters:
        unique_id: ID
        model: model that stores the agent
        query_size: sliding window size
        p_part: probability an agent will participate in a task (active=True)
    """
    def __init__(self, unique_id, model,queue_size=None,p_part=None):
        super().__init__(unique_id, model)
        self.queue_size =  queue_size if queue_size else model.queue_size
        self.p_participate = p_part if p_part else model.p_part

        # a buffer to record the participation pos in the recent past
        self.traj_queue =  collections.deque([], maxlen=self.queue_size)
        # whether the agent is active
        self.is_active = False
        self.current_pos = (0,0)

        #print('add agent',self.unique_id,'queue_size=',self.queue_size,
            #'p_part=',self.p_participate)

    def step(self):
        """ for a probability p, user will select an activity to participate"""
        if self.random.uniform(0,1) < self.p_participate:
            self.is_active  = True
            self.join()
        else:
            self.is_active = False
            self.leave()

    def count_in_dq(self,dq, item):
        """count the frequency of item in a deque"""
        return sum(elem == item for elem in dq)

    def find_value_in_grid(self,grid_with_values,pos):

        value = grid_with_values[pos[0],pos[1]]

        return value

    def choose_k_in_dic(self,dic):
        b = dic.copy()
        key = b.keys()
        for i, k in enumerate(key):
            #print(k)
            if dic[k] < 3:
                dic.pop(k)
        return dic

    def choose_k_in_distance(self,dic):
        #b = dic.copy()
        #key = list(b.keys())
        #for i in range (len(key)):
            #if (self.current_pos[0] - key[i][0]) ** 2 + (self.current_pos[1] - key[i][1]) ** 2 > 10:
                #dic.pop(key[i])

        possible_steps = self.model.grid.get_neighborhood(self.current_pos, moore=True,
                                                          include_center=False)
        self.random.shuffle(possible_steps)
        possible_steps_final = []
        for i in range(len(possible_steps)):
            possible_steps_final.append(self.model.grid.get_neighborhood(possible_steps[i], moore=True,
                                                          include_center=False))
        possible_steps_final = list(chain.from_iterable(possible_steps_final))
        self.random.shuffle(possible_steps_final)
        #print(possible_steps_final)
        b = dic.copy()
        key = list(b.keys())
        for i in range (len(key)):
            if key[i] not in possible_steps_final:
                dic.pop(key[i])

        #print(dic)
        return ValueSortedDict(dic)

    def join(self):
        """
        move to the first available spot from visited locations in the past time
        window, ordered by visiting frequency; if no space is available,
        move to a neighbor if model.neighbor_first is True. Otherwise,
        move to a random empty location
        """
        candidates_freq = ValueSortedDict({pos: self.count_in_dq(self.traj_queue,pos) \
            for pos in self.traj_queue} )

        grid_with_values = grid_value_changes(self.model.schedule.steps,self.model.grid_with_values)
        #grid_with_values = grid_value_changes(0, self.model.grid_with_values)

        candidates_self_org = {pos: self.find_value_in_grid(grid_with_values,pos) \
            for pos in self.traj_queue}

        candidates_self_org = ValueSortedDict(self.choose_k_in_dic(candidates_self_org))
        #candidates_self_org = ValueSortedDict(candidates_self_org)
        #print(candidates_self_org)
        candidates_self_org = self.choose_k_in_distance(candidates_self_org)



        supervisor_pos = [(i, j) for i in range(self.model.supervisor.shape[0]) for j in range(self.model.supervisor.shape[1])]

        candidates_centralized_ori = {pos: self.find_value_in_grid(grid_with_values, pos) \
                                                  for pos in supervisor_pos}
        #candidates_centralized_ori  = self.choose_k_in_distance(candidates_centralized_ori )
        candidates_centralized = ValueSortedDict(candidates_centralized_ori)
        #centralized
        '''

        if  self.model.schedule.steps < 2:
            candidates = ValueSortedDict(candidates_centralized_ori)


        #elif  self.model.schedule.steps == 20 or self.model.schedule.steps == 40 or self.model.schedule.steps == 60 or self.model.schedule.steps == 80 :

        elif  self.model.schedule.steps == 30 or self.model.schedule.steps == 70:
        #elif self.model.schedule.steps == 50:
            #candidates = candidates_self_org
            candidates = candidates_centralized
            self.queue_size = 1
            self.traj_queue.clear()
            #candidates = {self.traj_queue[0]:self.find_value_in_grid(grid_with_values, self.traj_queue[0])}
        else:
            #candidates = candidates_centralized
            #print(len(self.traj_queue))
            self.queue_size = 1
            self.traj_queue.clear()
            self.traj_queue.appendleft(self.current_pos)

            candidates = {self.traj_queue[0]: self.find_value_in_grid(grid_with_values, self.traj_queue[0])}
        
        success = False
        while candidates:
            pos,_ = candidates.popitem()
            # move to this location, if success
            #if (self.current_pos[0] - pos[0]) ** 2 + (self.current_pos[1] - pos[1]) ** 2 > 25:
                #success = False
                #break
            success = self.move(col=pos[0],row=pos[1])
            if success:
                self.traj_queue.appendleft(pos)
                self.current_pos = pos

                break
        '''

        #hybrid


        if self.model.schedule.steps <2:
            candidates = candidates_centralized
            self.queue_size = 1
            self.traj_queue.clear()
            success = False
            while candidates:
                pos, _ = candidates.popitem()
                # move to this location, if success
                success = self.move(col=pos[0], row=pos[1])
                if success:
                    self.traj_queue.appendleft(pos)
                    self.current_pos = pos

                    break

        #elif (self.model.schedule.steps < 25 and self.model.schedule.steps>=20) or (self.model.schedule.steps < 45 and self.model.schedule.steps>=40) \
               #or (self.model.schedule.steps < 65 and self.model.schedule.steps>=60) or (self.model.schedule.steps < 85 and self.model.schedule.steps>=80):
                #or (self.model.schedule.steps < 15 and self.model.schedule.steps>=10) or (self.model.schedule.steps < 35 and self.model.schedule.steps>=30) \
                #or (self.model.schedule.steps < 55 and self.model.schedule.steps>=50) or (self.model.schedule.steps < 75 and self.model.schedule.steps>=70) \
                #or (self.model.schedule.steps < 95 and self.model.schedule.steps>=90):
            #candidates = candidates_self_org
        elif (self.model.schedule.steps < 45 and self.model.schedule.steps>=30) or (self.model.schedule.steps < 85 and self.model.schedule.steps>=70):
        #elif (self.model.schedule.steps < 55 and self.model.schedule.steps >= 50):
            candidates = candidates_centralized
            self.queue_size = 1
            self.traj_queue.clear()
            success = False
            while candidates:
                pos, _ = candidates.popitem()
                # move to this location, if success
                success = self.move(col=pos[0], row=pos[1])
                if success:
                    self.traj_queue.appendleft(pos)
                    self.current_pos = pos

                    break

            #self.traj_queue.clear()
            #candidates = {self.traj_queue[0]:self.find_value_in_grid(grid_with_values, self.traj_queue[0])}
        else:
            candidates = ValueSortedDict(candidates_self_org)
            #self.queue_size = 12
            success = False
            while candidates:
                pos,_ = candidates.popitem()
                # move to this location, if success
                #if (self.current_pos[0] - pos[0]) ** 2 + (self.current_pos[1] - pos[1]) ** 2 > 25:
                    #success = False
                    #break
                success = self.move(col=pos[0],row=pos[1])
                if success:
                    self.traj_queue.appendleft(pos)
                    self.current_pos = pos

                    break
         

        # move to a neighboring cell if neighbors are not available


            if (len(self.traj_queue) <3):
                if not success and self.pos and self.model.neighbor_first:
                    possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True,
                            include_center=False)
                    self.random.shuffle(possible_steps)
                    while possible_steps:
                    #while possible_steps:
                        new_pos  = possible_steps.pop()
                        success = self.move(col=new_pos[0], row = new_pos[1])
                        if success:
                            self.traj_queue.appendleft(new_pos)
                            self.current_pos = new_pos




        # move to a neighboring cell if neighbors are not available





        #candidates = candidates_centralized

        #print(self.model.schedule.steps)
        #self_org

        '''
        candidates = candidates_self_org

        success = False
        while candidates:
            pos,_ = candidates.popitem()
            # move to this location, if success
            #if (self.current_pos[0] - pos[0]) ** 2 + (self.current_pos[1] - pos[1]) ** 2 > 25:
                #success = False
                #break
            success = self.move(col=pos[0],row=pos[1])
            if success:
                self.traj_queue.appendleft(pos)
                self.current_pos = pos

                break


        # move to a neighboring cell if neighbors are not available


        if not success and self.pos and self.model.neighbor_first:
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True,
                    include_center=False)
            self.random.shuffle(possible_steps)
            possible_steps_final = []
            for i in range(len(possible_steps)):
                possible_steps_final.append(self.model.grid.get_neighborhood(possible_steps[i], moore=True,
                                                                             include_center=False))
            possible_steps_final = list(chain.from_iterable(possible_steps_final))
            self.random.shuffle(possible_steps_final)
            while possible_steps_final:
                new_pos  = possible_steps_final.pop()
                success = self.move(col=new_pos[0], row = new_pos[1])
                if success:
                    self.traj_queue.appendleft(new_pos)
                    self.current_pos = new_pos


            # move to a random open location
        candidates = [ (i,j) for i in range(self.model.grid.width) for j in range(self.model.grid.height) ]
        time = 0
        while not success:
            time +=1
            #if (time>10):
                #break
            pos = self.random.choice(candidates)
            success = self.move( col = pos[0], row=pos[1] )

            if success:
                self.traj_queue.appendleft(pos)
                self.current_pos = pos
         '''



    def leave(self):
        """remove record from current pos"""
        if self.pos:
            self.model.grid.remove_agent(self)

    def move(self,row,col):
        """attempt to move to the specific location,
        Parameters:
            row,col: index of target position
        Return:
            False if no space is available,, otherwise return True
        """
        agents = self.model.grid.get_cell_list_contents([ (col,row)])
        if self.unique_id in agents:
            agents.remove(self.unique_id)

        success = False

        if len(agents)< self.model.grid.max_agent_per_cell:
            if not self.pos :
                self.model.grid.place_agent(self,(col,row))
            else:
                self.model.grid.move_agent(self, (  col,row ))
            success =  True

        return success


class UserModel(Model):
    """A model with some number of agents.
    Parameters:
        N: # of agents
        width, height:  width and height of grid
        max_agent_per_cell: max number of agents in each grid cell
                        (# of people necessary for each task)
        query_size: sliding window size
        p_part: probability an agent will participate in a task (active=True)
    """
    def __init__(self, N,width,height,grid_with_values,supervisor,max_agent_per_cell=1,
                 queue_size=12,p_part=1,neighbor_first=False):
        self.num_agents = N
        self.grid = MultiGrid(width=width, height=height, torus=True)
        self.grid.max_agent_per_cell = max_agent_per_cell
        self.queue_size=queue_size
        self.p_part = p_part
        self.neighbor_first = neighbor_first
        self.schedule = RandomActivation(self)
        self.running = True
        self.grid_with_values = grid_with_values
        self.supervisor = supervisor

        # Create agents
        for i in range(self.num_agents):
            a = UserAgent(i, self )
            self.schedule.add(a)

        # Create data collector
        self.datacollector = DataCollector(\
            model_reporters = {#"H(pos|user)": [ compute_cond_entropy,[self, 'None']],
            "H(pos|user)/H(pos)":[ compute_cond_entropy,[self, 'asymmetric']],
            #"H(pos|user)/(H(pos)+H(user))": [compute_cond_entropy,[self, 'symmetric']]
                },
            agent_reporters= {"agent pos":"pos"})


    def step(self):
        """ one model step  """
        self.schedule.step()
        if(self.schedule.steps>1):
            self.datacollector.collect(self)


def count_pos_from_traj(model,traj_queue):
    """ count position frequency from past agent trajectory
    Parameter:
        traj_queue: an iterable object containing agent positions
    Return:
        vectorized visit frequency at each grid location
    """
    freq = np.zeros((model.grid.width, model.grid.height))
    for pos in traj_queue:
        freq[pos[0]][pos[1]]+=1
    return freq.flatten()

def calcualte_gain_one_agent(model,current_pos):
    """ count gain for one agent
    Parameter:
        traj_queue: an iterable object containing agent positions
    Return:
        int constant for
    """
    gain_one_agent = 0
    grid_with_values = grid_value_changes(model.schedule.steps, model.grid_with_values)
    gain_one_agent += grid_with_values[current_pos[0],current_pos[1]]
    return gain_one_agent

def calcualte_gain(model):
    """ count gain for one agent
    Parameter:
        traj_queue: an iterable object containing agent positions
    Return:
        int constant for
    """
    gain = 0

    for i, agent in enumerate(model.schedule.agents):
        if(agent.pos):
            gain += calcualte_gain_one_agent(model, agent.pos)
    return gain

def compute_cond_entropy(model,normalizer= "asymmetric"):
    """
    compute the conditional entropy of position (pos) given user H(pos|user)
     in the most recent time window of length model.queue_size.

    Parameter:
        model: the simulation model
        normalizer: "None" is unnormalized entropy, "asymmetric" is normalized
            using H(pos), symmetric is normalized using H(pos)+H(user)
    """
    H_pos_given_user = np.zeros(model.num_agents)
    p_user = np.zeros(model.num_agents)
    p_pos = np.zeros(model.grid.width* model.grid.height)
    for i,agent in enumerate(model.schedule.agents):
        agent_history =  count_pos_from_traj(model,agent.traj_queue)
        p_user[i] = np.sum(agent_history)
        p_pos += agent_history
        p_pos_given_user = normalize(agent_history[:,np.newaxis],
                                     axis=0,norm='l1').ravel() + 1e-9
        H_pos_given_user[i]  = -1* p_pos_given_user.dot(np.log(p_pos_given_user))
    p_user = normalize(p_user[:,np.newaxis],axis=0,norm='l1').ravel()+ 1e-9
    p_pos = normalize(p_pos[:,np.newaxis],axis=0,norm='l1').ravel()+ 1e-9
    H_pos_given_users= p_user.dot(H_pos_given_user)
    H_pos = -1* p_pos.dot(np.log(p_pos) )
    H_user = -1 *p_user.dot(np.log(p_user))
    if normalizer=='asymmetric':
        NCE = H_pos_given_users/H_pos
    elif normalizer=='symmetric':
        NCE = H_pos_given_users/(H_pos+H_user)
    else:
        NCE = H_pos_given_users
    return NCE


def grid_value_changes(iter,grid_with_values_org):

    grid_with_values_phase1 = grid_with_values_org.copy()
    grid_with_values_phase2 = grid_with_values_org.copy()
    grid_with_values_phase3 = grid_with_values_org.copy()
    grid_with_values_phase4 = grid_with_values_org.copy()
    grid_with_values_phase5 = grid_with_values_org.copy()
    '''
    grid_with_values_phase4 = grid_with_values_org.copy()
    grid_with_values_phase5 = grid_with_values_org.copy()
    grid_with_values_phase6 = grid_with_values_org.copy()
    grid_with_values_phase7 = grid_with_values_org.copy()
    grid_with_values_phase8 = grid_with_values_org.copy()
    grid_with_values_phase9 = grid_with_values_org.copy()
    grid_with_values_phase10 = grid_with_values_org.copy()
    values = grid_with_values_org
    grid_with_values_phase1[:, :1] += 5
    grid_with_values_phase2[:, 1:2] += 5
    grid_with_values_phase3[:, 2:3] += 5
    grid_with_values_phase4[:, 3:4] += 5
    grid_with_values_phase5[:, 4:5] += 5
    grid_with_values_phase6[:, 5:6] += 5
    grid_with_values_phase7[:, 6:7] += 5
    grid_with_values_phase8[:, 7:8] += 5
    grid_with_values_phase9[:, 8:9] += 5
    grid_with_values_phase10[:, 9:10] += 5
    if (iter <10):
        values = grid_with_values_phase1
    if (iter >= 10 and iter < 20):
        values = grid_with_values_phase2
    if (iter >= 20 and iter < 30):
        values = grid_with_values_phase3
    if (iter >= 30 and iter < 40):
        values = grid_with_values_phase4
    if (iter >= 40 and iter < 50):
        values = grid_with_values_phase5
    if (iter >= 50 and iter <60):
        values = grid_with_values_phase6
    if (iter >= 60 and iter < 70):
        values = grid_with_values_phase7
    if (iter >= 70 and iter < 80):
        values = grid_with_values_phase8
    if (iter >= 80 and iter < 90):
        values = grid_with_values_phase9
    if (iter >= 90):
        values = grid_with_values_phase10

   
    grid_with_values_phase1[:, :4] += 5
    grid_with_values_phase2[:, 3:7] += 5
    grid_with_values_phase3[:, 6:10] += 5
    if (iter <30):
        values = grid_with_values_phase1
    if (iter >= 30 and iter < 70):
        values = grid_with_values_phase2
    if (iter >= 70):
        values = grid_with_values_phase3
    
    grid_with_values_phase1[:, :2] += 5
    grid_with_values_phase2[:, 2:4] += 5
    grid_with_values_phase3[:, 4:6] += 5
    grid_with_values_phase4[:, 6:8] += 5
    grid_with_values_phase5[:, 8:] += 5
    if (iter <20):
        values = grid_with_values_phase1
    if (iter >= 20 and iter < 40):
        values = grid_with_values_phase2
    if (iter >= 40 and iter < 60):
        values = grid_with_values_phase3
    if (iter >= 60 and iter < 80):
        values = grid_with_values_phase4
    if (iter >= 80):
        values = grid_with_values_phase5
    
    grid_with_values_phase1[:, :5] += 5
    grid_with_values_phase2[:, 6:] += 5
    if (iter <50):
        values = grid_with_values_phase1
    if (iter >= 50):
        values = grid_with_values_phase2
    
    grid_with_values_phase1[:, :4] += 5
    grid_with_values_phase2[:, 3:7] += 5
    grid_with_values_phase3[:, 6:10] += 5
    if (iter <30):
        values = grid_with_values_phase1
    if (iter >= 30 and iter < 70):
        values = grid_with_values_phase2
    if (iter >= 70):
        values = grid_with_values_phase3
    
    grid_with_values_phase1[:, :5] += 5
    grid_with_values_phase2[:, 6:] += 5
    if (iter <50):
        values = grid_with_values_phase1
    if (iter >= 50):
        values = grid_with_values_phase2
    '''
    grid_with_values_phase1[:, :4] += 5
    grid_with_values_phase2[:, 3:7] += 5
    grid_with_values_phase3[:, 6:10] += 5
    if (iter <30):
        values = grid_with_values_phase1
    if (iter >= 30 and iter < 70):
        values = grid_with_values_phase2
    if (iter >= 70):
        values = grid_with_values_phase3
    return values

def singleRun(ite):
    task_size = 3 # max user per task
    N= 150 # number of users
    p_part=1
    queue_size = 12
    neighbor_first = True
    width = 10
    height = 10
    grid_with_values_org = np.random.random((width, height))*5 # model.grid.
    grid_with_values = grid_with_values_org
    # print(np.random.random_integers(0,10,(5,10)))

    #print(grid_with_values)
    #super_line =10
    #super_line = 10 - (ite // 10)
    super_line = 8
    supervisor = grid_with_values[:super_line,:]
    #print(supervisor.shape)

    model = UserModel(N,width,height,grid_with_values,supervisor,
                    max_agent_per_cell = task_size,
                    queue_size=queue_size,
                    p_part=p_part,
                    neighbor_first= neighbor_first)
    #print(grid_with_values)
    gain =[]
    for iter in range(100):
        if iter % 50==0:
            print(iter)
        model.step()
        gain.append(calcualte_gain(model))

    np.save('output3/gain_self_1step_80sup.npy',np.array(gain))

    # get final position of agents
    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count
    print('max agent count=',np.max(agent_counts))
    print('system gain=', calcualte_gain(model))
    plt.imshow(agent_counts, interpolation='nearest')
    plt.colorbar()
    #plt.savefig('output3/agents_80_self.pdf')
    #plt.show()
    param_prefix= '_N%d_ts%d_w%d_p%f'  % (N,task_size, queue_size,p_part)
    if neighbor_first:
        param_prefix = param_prefix+'_n'

    # plot the conditional entropy score
    cond_entropy = model.datacollector.get_model_vars_dataframe()
    cond_entropy = pd.DataFrame(data=cond_entropy)['H(pos|user)/H(pos)']
    #cond_entropy.plot()
    #plt.savefig('output3/renew_hybrid_entropy.pdf')
    test = pd.DataFrame(data=cond_entropy)
    test.to_csv('output3/hybrid_entropy_15step_80sup.csv', index=False)
    #test.to_csv('self_50_renew/%s.csv' % ite, index=False)
    #plt.show()


    #plt.plot(np.linspace(1, 100, 100), gain, 'green')
    #plt.savefig('output2/5gain_hybrid_1.pdf')
    #plt.show()

    ##plt.colorbar()
    #plt.savefig('output2/grid_800_step.pdf')
    #plt.show()
    # save agent position to file
    dataframe= model.datacollector.get_agent_vars_dataframe()
    #print(dataframe)
    #dataframe.to_csv('output/agent_pos%s.csv' % param_prefix)
    gain_ave = np.sum(np.array(gain)) / 100
    print(gain_ave)
    return gain_ave


if __name__ == '__main__':
    gain_exp = []
    for i in range(1):
        np.random.seed(i)
        gain_exp.append(singleRun(i))
    test = pd.DataFrame(data=gain_exp)
    #test.to_csv('self_gain_renew.csv', index=False)
    #test.to_csv('5gain_exp_self_1.csv', index=False)
