import Learning
import numpy as np


class Chunking_Graph:
    """
    A class to represent the representation of a hierarchical chunk learner.

    Attributes
    ----------
    vertex_list : list of chunk object
        list of vertex with the order of addition
    vertex_location : list
        location of vertex on the visualization graph
    edge_list : list
        list of vertex tuples
    y_max: int
        the initial height of the graph, used for plotting
    x_max: int
        initial x location of the graph
    M: dict
        set of chunk objects, and the frequency that each is observed
    T: dict
        transition from chunk to chunk
    theta: float
        forgetting rate
    deletion_threshold: float
    H: int
        height of the sequence
    W: int
        width of the sequence
    zero:
        non-observational element
    """
    def __init__(self, y_max= 0, x_max=0, DT = 0.01, theta=0.75):
        """DT: deletion threshold"""
        self.vertex_list = [] # list of the chunks
        self.vertex_location = []
        self.edge_list = []
        self.y_max = y_max
        self.x_max = x_max
        self.M = {}
        self.T = {}
        self.theta = theta
        self.deletion_threshold = DT
        self.H = 1
        self.W = 1
        self.zero = None


    def update_hw(self, H, W):
        self.H = H
        self.W = W
        return

    def generate_empty(self):
        if self.zero == None:
            self.zero = Learning.arr_to_tuple(np.zeros([1, self.H, self.W]))
            self.M[self.zero] = 0
            self.add_chunk_to_vertex(self.zero)
        return

    def get_nonzeroM(self):
        """returns an M without zero component"""
        nzm = list(self.M.keys()).copy()
        nzmm = nzm.copy()
        nzmm.remove(self.zero)
        return nzmm


    def add_chunk_to_vertex(self, newc, left= None, right = None):
        """update graph configuration"""
        self.vertex_list.append(newc)
        # compute the x and y location of a new chunk based on pre-existing
        # graph configuration
        if left is None and right is None:
            x_new_c = self.x_max + 1
            y_new_c = self.y_max # no y axis for atomix chunk
            self.x_max = x_new_c
            self.y_max = y_new_c
            self.vertex_location.append([x_new_c, y_new_c])
        else:
            left_idx = self.vertex_list.index(left)
            right_idx = self.vertex_list.index(right)
            l_x, l_y = self.vertex_location[left_idx]
            r_x, r_y = self.vertex_location[right_idx]
            x_c = (l_x + r_x)*0.5
            y_c = self.y_max + 1
            self.y_max = y_c
            idx_c = len(self.vertex_list) - 1
            self.edge_list.append((left_idx, idx_c))
            self.edge_list.append((right_idx, idx_c))
            self.vertex_location.append([x_c, y_c])
        return

    def add_chunk_to_cg_class(self, chunk):
        """
        Add a new chunk to the existing representation
            Parameters:
                chunk: np.array converted to tuple format
        """

        if len(self.M) > 0:
            if chunk in list(self.M.keys()):
                self.M[chunk] = self.M[chunk] + 1
            else:
                self.M[chunk] = 1
                self.add_chunk_to_vertex(chunk)
        else:
            self.M[chunk] = 1
            self.add_chunk_to_vertex(chunk)
            return

    def transition_into_probabilities(self, prev, current):
        """returns the marginal probability of prev,
        and the transition probability from prev to current
        based on the past history of experience"""
        chunk_f = self.M
        chunk_pair_f = self.T
        if (prev in list(chunk_f.keys()) and prev in list(
                chunk_pair_f.keys())) and current in list(
            chunk_pair_f[prev].keys()):
            sum_transition = 0
            for key in list(chunk_pair_f[prev].keys()):
                sum_transition += chunk_pair_f[prev][key]
            sum_marginals = 0
            for key in list(chunk_f.keys()):
                sum_marginals = sum_marginals + chunk_f[key]
            P_prev = chunk_f[prev] / sum_marginals
            P_current_giv_prev = chunk_pair_f[prev][current] / sum_transition
        else:
            P_prev = None
            P_current_giv_prev = None

        return P_prev, P_current_giv_prev

    def forget(self):
        """ discounting past observations if the number of frequencies is beyond deletion threshold"""
        chunk_f = self.M
        chunk_pair_f = self.T
        for item in list(chunk_f.keys()):
            chunk_f[item] = chunk_f[item] * self.theta  # memory decays as a function of time
            if chunk_f[item] < self.deletion_threshold and item != self.zero:
                chunk_f.pop(item)
                self.pop_transition_matrix(tuple(item))
                # print("pop ", item, 'in marginals and symbol table because it is not used very often')
                if item == ():
                    print("is an empty item in transition matrix key? ",
                          item in list(chunk_pair_f.keys()))
        if chunk_pair_f != {}:
            for fromkey in list(chunk_pair_f.keys()):
                for dt in list(chunk_pair_f[fromkey].keys()):
                    if chunk_pair_f[fromkey][dt] != {}:
                        for tokey in list(chunk_pair_f[fromkey][dt].keys()):
                            chunk_pair_f[fromkey][dt][tokey] = chunk_pair_f[fromkey][dt][tokey] * self.theta
                            if chunk_pair_f[fromkey][dt][tokey] < self.deletion_threshold:
                                chunk_pair_f[fromkey][dt].pop(tokey)
        return


    def chunking_reorganization(self, prev, current, cat, dt):
        """ Reorganize marginal and transitional probability matrix when a new chunk is created by concatinating prev and current """
        try:
            self.add_chunk_to_vertex(cat, left=prev, right=current)
        except ValueError:
            print('some chunks are not found in the list')
        chunk_f, chunk_pair_f = self.M, self.T

        """Model hasn't seen this chunk before:"""
        if (tuple(cat) not in list(chunk_f.keys())) & (tuple(current) in list(chunk_f.keys())):
            # estimate the marginal probability of cat from P(s_last)P(s|s_last)
            #chunk_f[tuple(cat)] =  P_current_giv_prev*P_prev*sum(chunk_f.values())
            chunk_f[tuple(cat)] = 1
            if chunk_pair_f != {}:
                if tuple(current) in list(chunk_pair_f.keys()):
                    chunk_pair_f[tuple(cat)] = chunk_pair_f[tuple(current)].copy()  # inherent the transition from the last chunk element
                for key in list(chunk_pair_f.keys()):
                    if str(dt) in list(chunk_pair_f[key].keys()):
                        if tuple(prev) in list(chunk_pair_f[key][str(dt)].keys()):
                            chunk_pair_f[key][str(dt)][tuple(cat)] = 1
                            chunk_pair_f[key][str(dt)][tuple(prev)] = chunk_pair_f[key][str(dt)][tuple(prev)] - 1
                            if chunk_pair_f[key][str(dt)][tuple(prev)] <= 0:  # delete the not used entries
                                chunk_pair_f[key][str(dt)].pop(tuple(prev))
            '''reduce the estimate occurance times of joint from each component in s and s_last'''
            if tuple(prev) in list(chunk_f.keys()):
                #             chunk_f[tuple(prev)] = chunk_f[tuple(prev)]-chunk_f[tuple(cat)]
                if chunk_f[tuple(prev)] <= 0:
                    chunk_f.pop(tuple(prev))
                    self.pop_transition_matrix(tuple(prev))

            if tuple(current) in list(chunk_f.keys()):
                #             chunk_f[tuple(current)] = chunk_f[tuple(current)]-chunk_f[tuple(cat)]
                if chunk_f[tuple(current)] <= 0:
                    chunk_f.pop(tuple(current))
                    self.pop_transition_matrix(tuple(current))

        '''Model has seen this chunk before'''
        if (tuple(cat) in list(chunk_f.keys())) & (tuple(current) in list(chunk_f.keys())):
            # reduce count from subtransition:
            chunk_f[tuple(cat)] = chunk_f[tuple(cat)] + 1
            if tuple(prev) in list(chunk_f.keys()):
                chunk_f[tuple(prev)] = chunk_f[tuple(prev)] - 1
                if chunk_f[tuple(prev)] <= 0:
                    chunk_f.pop(tuple(prev))
                    self.pop_transition_matrix(tuple(prev))

            if tuple(current) in list(chunk_f.keys()):
                chunk_f[tuple(current)] = chunk_f[tuple(current)] - 1
                if chunk_f[tuple(current)] <= 0:
                    chunk_f.pop(tuple(current))
                    chunk_pair_f.pop(tuple(current))
                    self.pop_transition_matrix(tuple(current))

            if chunk_pair_f != {}:
                if tuple(prev) in list(chunk_pair_f.keys()):
                    if chunk_pair_f[tuple(prev)] != {}:
                        if str(dt) in list(chunk_pair_f[tuple(prev)].keys()):
                            if tuple(current) in list(chunk_pair_f[tuple(prev)][str(dt)].keys()):
                                chunk_pair_f[tuple(prev)][str(dt)][tuple(current)] = \
                                    chunk_pair_f[tuple(prev)][str(dt)][tuple(current)] - 1
                                if chunk_pair_f[tuple(prev)][str(dt)][tuple(current)] <= 0:
                                    chunk_pair_f[tuple(prev)][str(dt)].pop(tuple(current))
        return

    def pop_transition_matrix(self, element):
        """delete the entries of element in the transition matrix"""
        transition_matrix = self.T
        # pop an element out of a transition matrix
        if transition_matrix != {}:
            # element should be a tuple
            if element in list(transition_matrix.keys()):
                transition_matrix.pop(element)
                # print("pop ", item, 'in transition matrix because it is not used very often')
            for key in list(transition_matrix.keys()):
                if element in list(transition_matrix[
                                       key].keys()):  # also delete entry in transition matrix
                    transition_matrix[key].pop(element)
        return


    def get_transitional_p(self,prev,current):
        """returns the transitional probability form the previous to the current chunk"""
        if (prev in list(self.M.keys()) and prev in list(self.T.keys())) and current in list(self.T[prev].keys()):
            sum_transition = 0
            for key in list(self.T[prev].keys()):
                sum_transition += self.T[prev][key]

            P_s_giv_s_last = self.T[prev][current] / sum_transition
            return P_s_giv_s_last
        else: return None


    def imagination(self, n, sequential = False, spatial = False,spatial_temporal = False):
        ''' Independently sample from a set of chunks, and put them in the generative sequence
            Obly support transitional probability at the moment
            n+ temporal length of the imagination'''
        marginals = self.M
        s_last_index = np.random.choice(np.arange(0,len(list(self.M.keys())),1))
        s_last = list(self.M.keys())[s_last_index]
        # s_last = np.random.choice(list(self.M.keys()))
        s_last = tuple_to_arr(s_last)
        H, W = s_last.shape[1:]
        if sequential:
            L = 20
            produced_sequence = np.zeros([n+ L, H, W])
            produced_sequence[0:s_last.shape[0], :, :] = s_last
            t = s_last.shape[0]
            while t <= n:
                #item, p = sample(transition_matrix, marginals, arr_to_tuple(s_last))
                item, p = sample_marginal(marginals)
                s_last = tuple_to_arr(item)
                produced_sequence[t:t + s_last.shape[0], :, :] = s_last
                t = t + s_last.shape[0]
            produced_sequence = produced_sequence[0:n,:,:]
            return produced_sequence

        if spatial:
            produced_sequence = np.zeros([n, H, W])
            produced_sequence[0, :, :] = s_last
            t = 1
            while t <= n:
                # this part is not as simple, because transition matrix is spatial.
                item, p = sample_spatial(marginals)
                s_last = item
                produced_sequence[t:t+1, :, :] = s_last
                t = t + 1
            return produced_sequence

        if spatial_temporal:
            L = 20
            produced_sequence = np.zeros([n+L, H, W])
            produced_sequence[0:s_last.shape[0], :, :] = s_last
            t = 1
            while t <= n:
                # this part is not as simple, because transition matrix is spatial.
                item, p = sample_spatial_temporal(marginals)
                s_last = item
                produced_sequence[t:t+s_last.shape[0], :, :] = s_last
                t = t + s_last.shape[0]
            return produced_sequence

        else:
            return None

def sample_from_distribution(states, prob):
    prob = [k / sum(prob) for k in prob]
    cdf = [0.0]
    for s in range(0, len(states)):
        cdf.append(cdf[s] + prob[s])
    k = np.random.rand()
    for i in range(1, len(states) + 1):
        if (k >= cdf[i - 1]):
            if (k < cdf[i]):
                return states[i - 1], prob[i - 1]


def sample_marginal(marg):
    """When it returns [], it means there is no prediction,
        otherwise, returns the predicted sequence of certain length as a list
        s_last: a tuple, of last stimuli, as the key to look up in the transition probability dictionary"""
    states = list(marg.keys())
    prob = []
    for s in range(0, len(states)):
        prob.append(marg[states[s]])
    prob = [k / sum(prob) for k in prob]
    return sample_from_distribution(states, prob)


def sample_spatial(marg):
    """When it returns [], it means there is no prediction,
        otherwise, returns the predicted sequence of certain length as a list
        s_last: a tuple, of last stimuli, as the key to look up in the transition probability dictionary"""
    if marg == {}:
        return [], 0
    else:
        states = list(marg.keys())
        prob = []
        for s in range(0, len(states)):
            prob.append(marg[states[s]])
        prob = [k / sum(prob) for k in prob]
        return sample_from_distribution(states, prob)

def tuple_to_arr(tup):
    return np.array(tup)

def sample_spatial_temporal(marg):
    if marg == {}:
        return [], 0
    else:
        states = list(marg.keys())
        prob = []
        for s in range(0, len(states)):
            prob.append(marg[states[s]])
        prob = [k / sum(prob) for k in prob]
        return sample_from_distribution(states, prob)

