import numpy as np
import random
from Learning import *
from Chunking_Graph import *
import json

''''Generates a hierarchical generative model with depth d'''
class Generative_Model():
        def generate_hierarchical_sequence(marginals, s_length=1000):
            # marginals can allso be the learned marginals, in that case this function is used to produce a simulated sequence
            # spatial or temporal
            # spatial chunks: chunks that exist only in spatial domain
            # temporal chunks: chunks that exist only in temporal domain
            # spatial temporal chunks: chunks that exist both in spatial and temporal domain.
            # or temporal sequential chunks

            not_over = True
            while not_over:
                new_sample, _ = sample_from_distribution(list(marginals.keys()), list(marginals.values()))
                # what is this sample from distribution when the marginals contain spatial temporal chunks?
                sequence = sequence + new_sample
                if len(sequence) >= s_length:
                    not_over = False
                    sequence = sequence[0:s_length]
            return sequence

        # seq= generate_hierarchical_sequence(generative_marginals,s_length = 1000)

        def sample_from_distribution(states, prob):
            """
            states: a list of states to sample from
            prob: another list that contains the probability"""
            prob = [k / sum(prob) for k in prob]
            cdf = [0.0]
            for s in range(0, len(states)):
                cdf.append(cdf[s] + prob[s])
            k = np.random.rand()
            for i in range(1, len(states) + 1):
                if (k >= cdf[i - 1]):
                    if (k < cdf[i]):
                        return list(states[i - 1]), prob[i - 1]


def partition_seq(this_sequence, bag_of_chunks):
    '''one dimensional chunks, multi dimensional chunks, TODO: make them compatible to other chunks'''
    # find the maximal chunk that fits the sequence
    # so that this could be used to evaluate the learning ability of the algorithm
    # what to do when the bag of chunks does not partition the sequence??
    i = 0
    end_of_sequence = False
    partitioned_sequence = []

    while end_of_sequence == False:
        max_chunk = None
        max_length = 0
        for chunk in bag_of_chunks:
            this_chunk = json.loads(chunk)
            if this_sequence[i:i + len(this_chunk)] == this_chunk:
                if len(this_chunk) > max_length:
                    max_chunk = this_chunk
                    max_length = len(max_chunk)

        if max_chunk == None:
            partitioned_sequence.append([this_sequence[i]])
            i = i + 1
        else:
            partitioned_sequence.append(list(max_chunk))
            i = i + len(max_chunk)

        if i >= len(this_sequence): end_of_sequence = True

    return partitioned_sequence


def initialize_atomic_chunks(h, w, n):
    """value takes random number that is sampled in the range from 0 to n"""
    # temporal x height x width
    # n is the maximal range (exclusive)
    # n is also the number of alphabets. # setting subject to change later.
    keys = []
    empty = np.zeros([1, h, w])  # the empty chunk
    zero = arr_to_tuple(empty)
    keys.append(zero)
    for i in range(1, n):
        neu_a = np.zeros([1, h, w])
        loi = np.random.choice(np.arange(h))  # randomly choose a location
        loj = np.random.choice(np.arange(w))
        value = np.random.choice(np.arange(n))
        neu_a[0, loi, loj] = value
        while arr_to_tuple(neu_a) in keys:
            neu_a = np.zeros([1, h, w])
            loi = np.random.choice(np.arange(h))
            loj = np.random.choice(np.arange(w))
            value = np.random.choice(np.arange(n))
            neu_a[0, loi, loj] = value
        keys.append(arr_to_tuple(neu_a))

    alpha = tuple([1 for i in range(0, n)])  # coefficient for the flat dirichlet distribution
    values = sorted(list(np.random.dirichlet(alpha, 1)[0]), reverse=True)
    marginal = dict(zip(keys, values))  # so that empty observation is always ranked the highest.

    nzent = keys[1:]
    transition = {} # need to specify transition with different delta t
    zero = arr_to_tuple(empty)
    transition[zero] = {}
    # ==== specify zero ====
    for item in list(marginal.keys()):
        transition[zero][item] = marginal[item]


    for pre in nzent:
        transition[pre] = {}
        transition[pre][zero] = marginal[zero]
        valid = False
        while not valid:
            gen_transition = dirichlet_flat(len(nzent),sort = False)
            nzentrans = dict(zip(nzent, gen_transition))  # so that empty observation is always ranked the highest.
            valid = True
            for cr in nzent:
                if nzentrans[cr]*marginal[pre] > marginal[cr]:
                    valid = False

        transition[pre].update(nzentrans)
    return marginal, transition, zero



def sample_from_distribution(states, prob):
    """
    states: a list
    prob: another list that contains the probability"""
    prob = [k / sum(prob) for k in prob]
    cdf = [0.0]
    for s in range(0, len(states)):
        cdf.append(cdf[s] + prob[s])
    k = np.random.rand()
    for i in range(1, len(states) + 1):
        if (k >= cdf[i - 1]):
            if (k < cdf[i]):
                return states[i - 1], prob[i - 1]


def generate_hierarchical_sequence(marginals, s_length=10):
    sequence = []
    not_over = True
    H, W = tuple_to_arr(list(marginals.keys())[0]).shape[1:]
    sequence = np.zeros([s_length, H, W])
    i = 0
    while i < s_length:
        indices = np.arange(len(list(marginals.keys())))
        idx = np.random.choice(indices, p = list(marginals.values()))
        new_sample = list(marginals.keys())[idx]
        t_len = tuple_to_arr(new_sample).shape[0]
        sample_duration = min(s_length, i + t_len) - i
        sequence[i:min(s_length, i + t_len), :, :] = tuple_to_arr(new_sample)[0:sample_duration, :, :]
        i = i + t_len
    return sequence


def generate_random_hierarchical_sequence(marginals, s_length=10):
    #sequence parse should be the same as the way the sequence is generated
    clen = {}
    for chunk in list(marginals):
        clen[chunk]= tuple_to_arr(chunk).shape[0]

    def checkoverlap(sequence, i, clen, this_chunk, len_this_chunk):
        '''i: location where this chunk is installed'''
        overlap = False
        for chunk in clen.keys():# iterate through each chunk:
            if  chunk!=(((0,),),) and chunk!=this_chunk and this_chunk!=(((0,),),):
                lenc = clen[chunk]
                if lenc > len_this_chunk:
                    tuple_seqchunk = arr_to_tuple(sequence[max(0,i + len_this_chunk-lenc):min(s_length, i + len_this_chunk),
                                               :, :])
                    # print('tuple seq chunk ', tuple_seqchunk, ' this chunk ', this_chunk)
                    if tuple_seqchunk == chunk: # some other chunk in this bag can be used to explain observations
                        overlap = True
                        # print(' the current generated chunk: ', this_chunk, ' overlaps with ', chunk)
                        # print(tuple_seqchunk, this_chunk)

        return overlap

    H, W = tuple_to_arr(list(marginals.keys())[0]).shape[1:]
    sequence = np.zeros([s_length, H, W])

    i = 0
    while i < s_length:
        new_sample = np.random.choice(list(marginals.keys()), p = list(marginals.values()))
        t_len = tuple_to_arr(new_sample).shape[0]
        sample_duration = min(s_length, i + t_len) - i
        sequence[i:min(s_length, i + t_len), :, :] = tuple_to_arr(new_sample)[0:sample_duration, :, :]

        # check if there are chunk overlaps:
        overlap = checkoverlap(sequence, i,clen,new_sample,sample_duration)
        while overlap:
            # delete items
            new_sample = np.random.choice(list(marginals.keys()), p = list(marginals.values()))
            t_len = tuple_to_arr(new_sample).shape[0]
            sample_duration = min(s_length, i + t_len) - i
            sequence[i:min(s_length, i + t_len), :, :] = tuple_to_arr(new_sample)[0:sample_duration, :, :]
            overlap = checkoverlap(sequence, i, clen, new_sample,sample_duration)
        i = i + t_len
    return sequence

def dirichlet_flat(N, sort = True):
    alpha = tuple([1 for i in range(0, N)])  # coefficient for the flat dirichlet distribution
    if sort:return sorted(list(np.random.dirichlet(alpha, 1)[0]), reverse=True)
    else: return list(np.random.dirichlet(alpha, 1)[0])

def find_max_p_chunk(M, T,zero):
    #compute all possible chunks, and find the one with the maximum probabiility
    max_p = 0
    max_cl = None
    max_cr = None
    max_ck = None

    for cl in list(M.keys()):
        if cl!= zero:
            pcl = M[cl]
            for cr in list(T[cl].keys()):
                if cr!= zero:
                    pcrgivcl = T[cl][cr]
                    pclcr = pcrgivcl*pcl
                    clcr = arr_to_tuple(np.concatenate((tuple_to_arr(cl), tuple_to_arr(cr)), axis=0))
                    if pclcr > max_p:
                        max_p = pclcr
                        max_cl = cl
                        max_cr = cr
                        max_ck = clcr
    # returns chunk and the probability associated with ti
    return max_p, max_ck, max_cl, max_cr


def d_to_dplusone0(M,T,p_clcr, clcr,cl,cr,zero):
    Md = M.copy()
    Td = T.copy()

    Md1 = {}
    Td1 = {}

    # formulating new marginal probability
    for x in list(Md.keys()):
        if x != cl and x!= cr:
            Md1[x] = Md[x]/(1 - p_clcr)
    Md1[cl] = (Md[cl] - p_clcr)/(1-p_clcr)
    Md1[cr] = (Md[cr] - p_clcr)/(1-p_clcr)
    Md1[clcr] = p_clcr/(1 - p_clcr)

    if not all(i >= 0 for i in list(Md1.values())):
        print('check')
    #print(np.sum(list(Md1.values())))
    # initialize transition
    for x in list(Md1.keys()):
        Td1[x] = {}
        for y in list(Md1.keys()):
            Td1[x][y] = 0
    # formulating new transitional probability

    for x in list(Md1.keys()):
        if x == cl:
            for y in list(Md1.keys()):
                if y!=cl and y!=cr and y!= clcr:
                    if np.isclose((1 - Td[cl][cr]), 0):
                        Td1[cl][y] = 0
                    else:
                        Td1[cl][y] = Td[cl][y]/(1 - Td[cl][cr])
            Td1[cl][cr] = 0
            # yellow constraints
            if np.isclose((1 - Td[cl][cr]), 0):
                Td1[cl][cl] = 0
                Td1[cl][clcr] = 0
            else:
                #HIGH = Td[cl][cl]/(1-Td[cl][cr])
                valid = False
                n = 0
                while not valid:
                    valid = True
                    Td1[cl][clcr] = np.random.uniform(low=0,high=min(1,Td[cl][cl]/(2*(1-Td[cl][cr])),Md1[clcr]/Md1[cl]))
                    Td1[cl][cl] = Td[cl][cl]/(1-Td[cl][cr]) - Td1[cl][clcr]
                    n = 0
                    if Td1[cl][cl]>1.0 or Td1[cl][cl]< 0:
                        print('Td[cl][cl]', Td[cl][cl])
                        print('Td[cl][cr]', Td[cl][cr])
                        print('Td1[cl][clcr]', Td1[cl][clcr])
                        print('Td1[cl][cl]', Td1[cl][cl])
                        print('Td[cl][cl]/(1-Td[cl][cr]) ', Td[cl][cl]/(1-Td[cl][cr]))
                        valid = False
                        n = n + 1
                print('cl-->', n)

        elif x == cr:
            valid = False
            while not valid:
                valid = True
                sum_y_giv_cr = 0
                sum_y_giv_clcr = 0
                for y in list(Md.keys()):
                    if y!= clcr:
                        # cr to y, clcr to y
                        # red constraints
                        print('Md[cr] ', Md[cr])
                        print('p_clcr', p_clcr)
                        print('(Md[cr] - p_clcr)', (Md[cr] - p_clcr))
                        ok = False
                        while not ok:
                            ok = True
                            #HIGH = Td[cr][y]*Md[cr]
                            #cond = np.random.uniform(low=0, high=min(HIGH))
                            #Td1[cr][y] = cond/(Md[cr] - p_clcr)
                            #Td1[clcr][y] = ( HIGH - Td1[cr][y]*(Md[cr] - p_clcr) )/p_clcr
                            HIGH = Td[cr][y]*Md[cr]/(Md[cr] - p_clcr)
                            Td1[cr][y] = np.random.uniform(low=Td[cr][y], high=min([HIGH,Md1[y]/Md1[cr],1]))
                            Td1[clcr][y] = (Td[cr][y]*Md[cr] - Td1[cr][y]*(Md[cr] - p_clcr))/p_clcr
                            print(' Td1[cr][y]', Td1[cr][y])
                            print(' Td1[clcr][y]', Td1[clcr][y])

                            if Td1[cr][y] <0 or Td1[cr][y] >1 or Td1[cr][y]*Md1[cr] > Md1[y] or sum_y_giv_cr + Td1[
                                cr][y]>1:
                                ok = False
                            if Td1[clcr][y]<0 or Td1[clcr][y]>1 or Td1[clcr][y]*Md1[clcr] > Md1[y] or sum_y_giv_clcr \
                                    + Td1[clcr][y]>1:
                                ok = False
                            # cond = Td1[cr][y]*(Md[cr] - p_clcr)
                        sum_y_giv_cr = sum_y_giv_cr + Td1[cr][y]
                        sum_y_giv_clcr = sum_y_giv_clcr + Td1[clcr][y]

                        # Td1[clcr][y] = (Td[cr][y]*Md[cr] - cond)/p_clcr
                        # Td1[clcr][y]*p_clcr = Td[cr][y]*Md[cr] - cond
                        if Td1[clcr][y]*Md1[clcr] > Md1[y] or Td1[clcr][y]<=0:
                            valid = False
                        print('Td1[cr][y]', Td1[cr][y])
                        print('Td1[clcr][y]', Td1[clcr][y])
                        print('sum_y_giv_cr', sum_y_giv_cr)
                        print('sum_y_giv_clcr', sum_y_giv_clcr)
                    # Gray Constraints y == cr:
                    # Green constraints: y == cl:
                Td1[cr][clcr] = 1 - sum_y_giv_cr
                if Td1[cr][clcr] * Md1[cr] > Md1[clcr] or Td1[cr][clcr]<=0:
                    valid = False
                print('Td1[cr][clcr]', Td1[cr][clcr])
                if Td1[clcr][clcr] * Md1[clcr] >Md1[clcr] or Td1[clcr][clcr]<=0:
                    valid = False
                Td1[clcr][clcr] = 1 - sum_y_giv_clcr
                print('Td1[clcr][clcr]', Td1[clcr][clcr])

        elif x == clcr:
            pass

        else:# x is neither cl nor cr nor clcr
            for y in list(Md1.keys()):
                if y == cl:
                    # Orange constraints

                    valid = False
                    n = 0
                    while not valid:
                        valid = True
                        Td1[x][cl] = np.random.uniform(low=0, high=min([Td[x][cl], Md1[cl]/Md1[x],1]))
                        Td1[x][clcr] = Td[x][cl] - Td1[x][cl]
                        if Td1[x][clcr]>Md1[clcr]/Md1[x] or Td1[x][clcr] < 0 or Td1[x][clcr] > 1:
                            print('Td1[x][clcr]',Td1[x][clcr])
                            print('Td1[x][cl]',Td1[x][cl])
                            valid = False
                        n = n + 1
                        if n>=20:
                            print('')
                    print('x-->', n)
                elif y == cr:
                    Td1[x][cr] = Td[x][cr]
                elif y == clcr:
                    pass
                else:
                    Td1[x][y] = Td[x][y]
    return Md1, Td1


def d_to_dplusone(M,T,p_clcr,clcr,cl,cr,zero):
    Md = M.copy()
    Td = T.copy()

    Md1 = {}
    Td1 = {}

    # formulating new marginal probability
    for x in list(Md.keys()):
        if x != cl and x!= cr:
            Md1[x] = Md[x]/(1 - p_clcr)
    Md1[cl] = (Md[cl] - p_clcr)/(1-p_clcr)
    Md1[cr] = (Md[cr] - p_clcr)/(1-p_clcr)
    Md1[clcr] = p_clcr/(1 - p_clcr)

    if not all(i >= 0 for i in list(Md1.values())):
        print('check')
    #print(np.sum(list(Md1.values())))
    # initialize transition
    for x in list(Md1.keys()):
        Td1[x] = {}
        for y in list(Md1.keys()):
            Td1[x][y] = 0
    # formulating new transitional probability

    for x in list(Md1.keys()):

        if x == cl:
            sum_clx = 0
            for y in list(Md1.keys()):
                if y!=cl and y!=cr and y!= clcr:
                    if np.isclose((1 - Td[cl][cr]), 0):
                        Td1[cl][y] = 0
                    else:
                        Td1[cl][y] = Td[cl][y]/(1 - Td[cl][cr])
                sum_clx = sum_clx + Td1[cl][y]
            print('sum_clx ',sum_clx)
            Td1[cl][cr] = 0
            # yellow constraints
            if np.isclose((1 - Td[cl][cr]), 0):
                Td1[cl][cl] = 0
                Td1[cl][clcr] = 0
            else:
                #HIGH = Td[cl][cl]/(1-Td[cl][cr])
                valid = False
                n = 0
                while not valid:
                    valid = True
                    Td1[cl][cl] = np.random.uniform(low= 0, high = 1 - sum_clx)
                    Td1[cl][clcr] = Td[cl][cl]/(1-Td[cl][cr]) - Td1[cl][cl]
                    n = 0
                    if Td1[cl][cl]>1.0 or Td1[cl][cl]< 0 or Td1[cl][clcr]> Td[cl][cl]/(2*(1-Td[cl][cr])) or Td1[cl][\
                            clcr]> Md1[clcr]/Md1[cl]:
                        print('Td[cl][cl]', Td[cl][cl])
                        print('Td[cl][cr]', Td[cl][cr])
                        print('Td1[cl][clcr]', Td1[cl][clcr])
                        print('Td1[cl][cl]', Td1[cl][cl])
                        print('Td[cl][cl]/(1-Td[cl][cr]) ', Td[cl][cl]/(1-Td[cl][cr]))
                        valid = False
                        n = n + 1
                print('cl-->', n)

        elif x == cr:
            valid = False
            while not valid:
                valid = True
                sum_y_giv_cr = 0
                sum_y_giv_clcr = 0
                for y in list(Md.keys()):
                    if y!= clcr:
                        # cr to y, clcr to y
                        # red constraints
                        print('Md[cr] ', Md[cr])
                        print('p_clcr', p_clcr)
                        print('(Md[cr] - p_clcr)', (Md[cr] - p_clcr))
                        ok = False
                        while not ok:
                            ok = True
                            #HIGH = Td[cr][y]*Md[cr]
                            #cond = np.random.uniform(low=0, high=min(HIGH))
                            #Td1[cr][y] = cond/(Md[cr] - p_clcr)
                            #Td1[clcr][y] = ( HIGH - Td1[cr][y]*(Md[cr] - p_clcr) )/p_clcr
                            HIGH = Td[cr][y]*Md[cr]/(Md[cr] - p_clcr)
                            Td1[cr][y] = Td[cr][y]
                            Td1[clcr][y] = (Td[cr][y]*Md[cr] - Td1[cr][y]*(Md[cr] - p_clcr))/p_clcr
                            print(' Td1[cr][y]', Td1[cr][y])
                            print(' Td1[clcr][y]', Td1[clcr][y])

                            if Td1[cr][y] <0 or Td1[cr][y] >1 or Td1[cr][y]*Md1[cr] > Md1[y] or sum_y_giv_cr + Td1[
                                cr][y]>1:
                                ok = False
                            if Td1[clcr][y]<0 or Td1[clcr][y]>1 or Td1[clcr][y]*Md1[clcr] > Md1[y] or sum_y_giv_clcr \
                                    + Td1[clcr][y]>1:
                                ok = False
                            # cond = Td1[cr][y]*(Md[cr] - p_clcr)
                        sum_y_giv_cr = sum_y_giv_cr + Td1[cr][y]
                        sum_y_giv_clcr = sum_y_giv_clcr + Td1[clcr][y]

                        # Td1[clcr][y] = (Td[cr][y]*Md[cr] - cond)/p_clcr
                        # Td1[clcr][y]*p_clcr = Td[cr][y]*Md[cr] - cond
                        if Td1[clcr][y]*Md1[clcr] > Md1[y] or Td1[clcr][y]<=0:
                            valid = False
                        print('Td1[cr][y]', Td1[cr][y])
                        print('Td1[clcr][y]', Td1[clcr][y])
                        print('sum_y_giv_cr', sum_y_giv_cr)
                        print('sum_y_giv_clcr', sum_y_giv_clcr)
                    # Gray Constraints y == cr:
                    # Green constraints: y == cl:
                Td1[cr][clcr] = 1 - sum_y_giv_cr
                if Td1[cr][clcr] * Md1[cr] > Md1[clcr] or Td1[cr][clcr]<=0:
                    valid = False
                print('Td1[cr][clcr]', Td1[cr][clcr])
                if Td1[clcr][clcr] * Md1[clcr] >Md1[clcr] or Td1[clcr][clcr]<=0:
                    valid = False
                Td1[clcr][clcr] = 1 - sum_y_giv_clcr
                print('Td1[clcr][clcr]', Td1[clcr][clcr])

        elif x == clcr:
            pass

        else:# x is neither cl nor cr nor clcr
            for y in list(Md1.keys()):
                if y == cl:
                    # Orange constraints

                    valid = False
                    n = 0
                    while not valid:
                        valid = True
                        Td1[x][cl] = Td[x][cl]
                        Td1[x][clcr] = Td[x][cl] - Td1[x][cl]
                        if Td1[x][clcr]>Md1[clcr]/Md1[x] or Td1[x][clcr] < 0 or Td1[x][clcr] > 1:
                            print('Td1[x][clcr]',Td1[x][clcr])
                            print('Td1[x][cl]',Td1[x][cl])
                            valid = False
                        n = n + 1
                        if n>=20:
                            print('')
                    print('x-->', n)
                elif y == cr:
                    Td1[x][cr] = Td[x][cr]
                elif y == clcr:
                    pass
                else:
                    Td1[x][y] = Td[x][y]
    return Md1, Td1

def generative_model_stc():
    # # the chunk combination process:
    # if it is a temporal chunk, simply pick two chunks and combine them together
    # if it is a spatial chunk, pick two chunks that occurs spatially proximal and put them together.
    # if it is a spatial temporal chunk, pick two chunks that is neighboring each other spatial temporally, and then put them together

    w = 2  # width of the observation
    h = 2  # height of the observation
    n = 5  # the number of alphabets
    D = 4
    # determine how the chunks are related to each other
    spatial_chunk = False
    temporal_chunk = False
    spatial_temporal_chunk = True
    cg = Chunking_Graph()
    dictionary = initialize_atomic_chunks(w, h, n)  # in each of the conditions, start with atomic chunks
    for chunk in list(dictionary.keys()):
        cg.add_chunk_to_vertex(chunk)
    dict_ohne_0 = dictionary.copy()
    del dict_ohne_0[arr_to_tuple(np.zeros([1, h, w]))]

    previous_prob = [1]
    for d in range(0, D):
        # each item is independently sampled
        c_i = max(dict_ohne_0, key=lambda k: dict_ohne_0[k])  # the maximal value of the current dictionary
        P_c_i = dict_ohne_0[c_i]
        list_keys = list(dict_ohne_0.keys())
        n_iter = 0
        max_iter = 10
        c_j_candidates = list_keys.copy()
        cicj = []
        while cicj in list(dict_ohne_0.keys()) or np.sum(np.abs(tuple_to_arr(
                cicj))) <= 0 or cicj == []:  # it must keep sampling until something valid, something new combes up
            n_iter = n_iter + 1
            if n_iter > max_iter or len(c_j_candidates) == 0:
                print('Maximal Iteration Reached, No new chunks can be formed')
                break
            c_j = random.choice(c_j_candidates)
            c_j_candidates.remove(c_j)

            # if they are spatial chunks, check whether they are spatially proximal
            if temporal_chunk:
                cicj = arr_to_tuple(np.concatenate((tuple_to_arr(c_i), tuple_to_arr(c_j)), axis=0))
            if spatial_chunk:
                end_point_prev = 1  # they are of the same size
                end_point_post = 1
                prev = c_i
                post = c_j
                cicj, dt = check_adjacency(prev, post, end_point_prev, end_point_post)
                cicj = arr_to_tuple(cicj)
                print('ci, ', tuple_to_arr(c_i))
                print('cj, ', tuple_to_arr(c_j))
                print('cjci ', tuple_to_arr(cicj))
            if spatial_temporal_chunk:
                delta_t_max = tuple_to_arr(c_i).shape[0] + 1
                # if two chunks are spatial-temporally proximal, then they are combined together as a whole.
                for delta_t in reversed(range(0, delta_t_max)):
                    end_point_prev = tuple_to_arr(c_j).shape[0]  # they are of the same size
                    end_point_post = tuple_to_arr(c_i).shape[0] + delta_t
                    prev = c_i
                    post = c_j
                    # check spatial or temporal adjacency, and combine the chunks together.

                    cicj, dt = check_adjacency(prev, post, end_point_prev, end_point_post)
                    cicj = arr_to_tuple(cicj)
                    print('t = ', delta_t)
                    print('ci, ', tuple_to_arr(c_i))
                    print('cj, ', tuple_to_arr(c_j))
                    print('cicj ', tuple_to_arr(cicj))
                    if np.sum(np.abs(tuple_to_arr(cicj))) > 0:
                        print('success')
                        break

        # generate the conditional probability of P(c_j|c_i), the probability of observing c_j after c_i
        conditional = np.random.uniform(low=dict_ohne_0[c_j], high=min(1, min(previous_prob) / P_c_i))  # c_j | c_i
        joint = conditional * P_c_i  # should be smaller than min previous_prob

        previous_prob.append(joint)
        dict_ohne_0[c_i] = dict_ohne_0[c_i] - joint  # maintain the normalization
        dict_ohne_0[cicj] = joint  # this joint could have existed before.

        # update graph configuration
        cg.add_chunk_to_vertex(cicj, left=c_i, right=c_j)
        # x_ci = vertex_location[vertex_list.index(c_i)][0]
        # x_cj = vertex_location[vertex_list.index(c_j)][0]
        # y_cicj = vertex_location[-1][1] - 1
        # x_cicj = (x_ci + x_cj) * 0.5
        # vertex_location.append([x_cicj, y_cicj])
        # edge_list.append((vertex_list.index(c_j), vertex_list.index(cicj)))
        # edge_list.append((vertex_list.index(c_i), vertex_list.index(cicj)))

    full_dict = dict_ohne_0.copy()
    full_dict[arr_to_tuple(np.zeros([1, h, w]))] = dictionary[arr_to_tuple(np.zeros([1, h, w]))]
    dictionary = full_dict
    print('full dictionary is ', full_dict)
    return cg


def generative_model_uncommittal(D = 4, w=1, h=1, n=5):
    """Produce Generative Model"""
    # generate generic temporal chunks
    # w = 2 # width of the observation
    # h = 2 # height of the observation
    # n = 5 # the number of alphabets
    # D = 4 # Depth of the generative model

    # determine how the chunks are related to each other
    cg = Chunking_Graph()
    #================== Initialization Process ==================
    M0,T0,zero = initialize_atomic_chunks(w, h, n) # in each of the conditions, start with atomic chunks
    M = M0
    T = T0
    if not all(i>=0 for i in list(M.values())):
        print('check')

    for key in list(T.keys()):
        if not all(i>=0 for i in list(T[key].values())):
            print('check')
    for chunk in list(M0.keys()):
        cg.add_chunk_to_vertex(chunk)
    max_p, max_ck, max_cl, max_cr = find_max_p_chunk(M0,T0,zero)
    cg.add_chunk_to_vertex(max_ck,max_cl,max_cr)
    for d in range(0,D):
        #d_to_dplusone(M,T,p_clcr, clcr,cl,cr,zero)
        M, T = d_to_dplusone(M,T,max_p, max_ck,max_cl,max_cr,zero)
        max_p, max_ck, max_cl, max_cr = find_max_p_chunk(M,T,zero)
        cg.add_chunk_to_vertex(max_ck,max_cl,max_cr)
        if not all(i>=0 for i in list(M.values())):
            print('check')

        for key in list(T.keys()):
            if not all(i>=0 for i in list(T[key].values())):
                print('check')

    cg.M = M
    cg.T = T

    # the last T at step T is thrown away.
    return cg


def check_overlap(array1, array2):
    # check whether two arrays contain common elements
    output = np.empty((0, array1.shape[1]))
    for i0, i1 in itertools.product(np.arange(array1.shape[0]), np.arange(array2.shape[0])):
        if np.all(np.isclose(array1[i0], array2[i1])):
            output = np.concatenate((output, [array2[i1]]), axis=0)
    return output


import itertools


def check_adjacency(prev, post, end_point_prev, end_point_post):
    '''returns empty matrix if not chunkable'''
    # in fact prev and post denotes ending time, but based on how long the chunks are, prev can happen after post.
    # update transitions between chunks with a spatial proximity
    temporal_len_prev = tuple_to_arr(prev).shape[0]
    temporal_len_post = tuple_to_arr(post).shape[0]
    start_point_prev = end_point_prev - temporal_len_prev  # the exclusive temporal length
    start_point_post = end_point_post - temporal_len_post  # start point is inclusive

    T, W, H = tuple_to_arr(prev).shape[0:]

    # the overlapping temporal length between the two chunks
    delta_t = end_point_prev - max(start_point_post, start_point_prev)
    # the stretching temporal length of the two chunks
    t_chunk = max(end_point_post, end_point_prev) - min(end_point_post, end_point_prev) + delta_t + max(
        start_point_post, start_point_prev) - min(start_point_post, start_point_prev)

    if t_chunk == temporal_len_prev and t_chunk == temporal_len_post and prev == post:  # do not chunk a chunk by itself.
        concat_chunk = np.zeros([t_chunk, W, H])
        return concat_chunk
    else:
        concat_chunk_prev = np.zeros([t_chunk, W, H])
        concat_chunk_prev[0:temporal_len_prev, :, :] = prev
        concat_chunk_post = np.zeros([t_chunk, W, H])
        concat_chunk_post[-temporal_len_post:, :, :] = post

        adjacent = False
        prev_index_t, prev_index_i, prev_index_j = np.nonzero(concat_chunk_prev)
        post_index_t, post_index_i, post_index_j = np.nonzero(concat_chunk_post)

        prev_indexes = list(zip(prev_index_t, prev_index_i, prev_index_j))
        post_indexes = list(zip(post_index_t, post_index_i, post_index_j))

        # pad the boundary of nonzero values
        for prev_t, prev_i, prev_j in zip(prev_index_t, prev_index_i, prev_index_j):
            indices = [(min(prev_t + 1, T), prev_i, prev_j),
                       (max(prev_t - 1, 0), prev_i, prev_j),
                       (prev_t, min(prev_i + 1, H), prev_j),
                       (prev_t, max(prev_i - 1, 0), prev_j),
                       (prev_t, prev_i, min(prev_j + 1, H)),
                       (prev_t, prev_i, max(prev_j - 1, 0)), ]
            prev_indexes = prev_indexes + indices

        print(np.array(prev_indexes))
        overlap = check_overlap(np.array(prev_indexes), np.array(post_indexes))
        if overlap.shape[0] > 0:
            adjacent = True
        print('-------------')
        print(concat_chunk_prev)
        print(concat_chunk_post)
        print(adjacent)
        print('-------------')

        if adjacent == True:
            concat_chunk = np.zeros([t_chunk, W, H])
            concat_chunk[concat_chunk_prev > 0] = concat_chunk_prev[concat_chunk_prev > 0]
            concat_chunk[concat_chunk_prev == concat_chunk_post] = 0  # remove overlapping components
            concat_chunk[concat_chunk_post > 0] = concat_chunk_post[concat_chunk_post > 0]
            return concat_chunk
        else:
            concat_chunk = np.zeros([t_chunk, W, H])  # nonadjacent, then returns a high dimneisonal zero array
            return concat_chunk



def generate_new_chunk(setofchunks):
    zero = arr_to_tuple(np.zeros([1,1,1]))
    a = list(setofchunks)[
        np.random.choice(np.arange(0, len(setofchunks), 1))]  # better to be to choose based on occurrance probability
    b = list(setofchunks)[np.random.choice(np.arange(0, len(setofchunks), 1))]  # should exclude 0
    va, vb = tuple_to_arr(a), tuple_to_arr(b)
    la, lb = va.shape[0], vb.shape[0]
    lab = la + lb
    vab = np.zeros([lab, 1, 1])
    vab[0:va.shape[0], :, :] = va
    vab[va.shape[0]:, :, :] = vb
    ab = arr_to_tuple(vab)
    if ab in setofchunks or np.array_equal(a, zero) or np.array_equal(b, zero):
        return generate_new_chunk(setofchunks)
    else:
        return ab, a, b


def generative_model_random_combination(D=3, n=5):
    """ randomly generate a set of hierarchical chunks
        D: number of recombinations
        n: number of atomic, elemetary chunks """
    def check_independence(constraints, M):
        for ab, a, b in constraints:
            if M[ab]<=M[a]*M[b]+ 0.003:
                return False # constraint is not satisified
        return True

    cg = Chunking_Graph()
    setofchunks = []
    for i in range(0, n):
        zero = np.zeros([1,1,1])
        zero[0,0,0] = i
        chunk = zero
        setofchunks.append(arr_to_tuple(chunk))

    setofchunkswithoutzero = setofchunks.copy()
    setofchunkswithoutzero.remove(arr_to_tuple(np.zeros([1,1,1])))
    constraints = []
    for d in range(0, D):
        # pick random, new combinations
        ab, a, b = generate_new_chunk(setofchunkswithoutzero)
        while ab in setofchunks:# keep generating new chunks that is new
            ab, a, b = generate_new_chunk(setofchunkswithoutzero)
        constraints.append([ab, a, b])
        setofchunks.append(ab)
        setofchunkswithoutzero = setofchunks.copy()
        setofchunkswithoutzero.remove(arr_to_tuple(np.zeros([1, 1, 1])))

    # calculate the chunk occurance probabilities, if the chunks are combined independently, given this
    # distribution assignment.
    satisfy_constraint = False
    while satisfy_constraint==False:
        # assign probabilities to this set of chunks:
        genp = dirichlet_flat(len(setofchunks), sort=False)
        p0 = max(genp)
        genp.remove(p0)
        p = [p0] + genp
        # normalize again, sometimes they don't sum up to 1.
        M = dict(zip(setofchunks, p))  # so that empty observation is always ranked the highest.
        cg.M = M
        satisfy_constraint = check_independence(constraints, M)

    # joint chunk probabilties needs to be higher than the independence criteria, otherwise items are not going to be
    # chunked.
    return cg


def to_chunking_graph(cg):
    M = cg.M

    # first, filter out the best joint probability from the marginals
    # then find out
    atomic_chunks = find_atomic_chunks(M)# initialize with atomic chunks
    for ac in atomic_chunks:
        cg.add_chunk_to_vertex(ac)

    chunks = set()
    for ck in list(atomic_chunks.keys()):
        chunks.add(ck)

    complete = False
    proposed_joints = set()
    while complete == False:
        # calculate the mother chunk and the father chunk of the joint chunks
        joints_and_freq = calculate_joints(chunks, M)# the expected number of joint observations
        new_chunk, cl, cr = pick_chunk_with_max_prob(joints_and_freq)
        while new_chunk in proposed_joints:
            joints_and_freq.pop(new_chunk)
            new_chunk, cl, cr = pick_chunk_with_max_prob(joints_and_freq)

        cg.add_chunk_to_vertex(new_chunk, left=cl, right=cr) #update cg graph with newchunk and its components
        chunks.add(new_chunk)
        proposed_joints.add(new_chunk)
        complete = check_completeness(chunks,M)
    return cg


def check_completeness(chunks, gtM):
    for ck in list(gtM.keys()):
        if ck not in chunks:
            return False
    return True

def pick_chunk_with_max_prob(joints_and_freq):
    chunk = max(joints_and_freq, key=lambda k: joints_and_freq[k][0])  # the maximal value of the current dictionary
    cl = joints_and_freq[chunk][1]
    cr = joints_and_freq[chunk][2]
    return chunk, cl, cr

def find_atomic_chunks(M):
    atomic_chunks ={}
    for chunk in list(M.keys()):
        if tuple_to_arr(chunk).shape[0]== 1:
            atomic_chunks[chunk] = eval_atom_chunk_in_M(chunk,M)

    # normalize occurrance count to a probability
    SUM_occur = sum(atomic_chunks.values())
    for key in list(atomic_chunks.keys()):
        atomic_chunks[key] = atomic_chunks[key]/SUM_occur
    return atomic_chunks

def eval_atom_chunk_in_M(chunk,M):
    Echunkn = 0 # expected chunk occurrance
    for key in list(M.keys()):
        Mchunk = tuple_to_arr(key)
        # count how many times chunk occurrs in M_chunk
        Mchunkp = get_n(chunk,Mchunk) * M[key]
        Echunkn = Echunkn + Mchunkp
    return Echunkn

def get_n(chunk,Mchunk):
    mck = tuple_to_arr(Mchunk)
    ck = tuple_to_arr(chunk)
    n = 0
    for i in range(0, mck.shape[0]):
        # find the maximally fitting chunk in bag of chunks to partition Mchunk
        if np.array_equal(mck[i:min(i+ck.shape[0],mck.shape[0]),:,:],ck):
            n = n + 1
    return n


def calculate_joints(chunks, M):
    """Calculate the best joints to combine with the pre-existing chunks"""
    ZERO = arr_to_tuple(np.zeros([1,1,1]))
    joints_and_freq = {}
    for chunk1 in chunks:
        for chunk2 in chunks:
            if chunk1 != ZERO and chunk2 != ZERO:
                chunk12 = combine_joints(chunk1, chunk2)
                ck_prob = calculate_expected_occurrance(chunk12,chunks,M) # need to use the entire chunk set to
                joints_and_freq[chunk12] = (ck_prob[chunk12],chunk1,chunk2)
    return joints_and_freq

def combine_joints(chunk1,chunk2):
    c1 = tuple_to_arr(chunk1)
    c2 = tuple_to_arr(chunk2)
    chunk12 = np.zeros([c1.shape[0] + c2.shape[0], 1, 1])
    chunk12[0:c1.shape[0], :, :] = c1
    chunk12[c1.shape[0]:, :, :] = c2
    chunk12 = arr_to_tuple(chunk12)
    return chunk12


def calculate_expected_occurrance(chunk12,bagofchunks,M):
    """In case when [1] [2] and [1,2] both exist, evaluation of [1] and [2] is included in [1,2]
     """
    # alternatively, partition M according to the new chunks.
    newchunks = bagofchunks.union({chunk12})# new bag of chunks when chunk12 is appended.
    ckfreq = {}
    for chunk in newchunks:
        ckfreq[chunk] = 0


    for gck in list(M.keys()):
        gcka = tuple_to_arr(gck)
        lgck = gcka.shape[0]

        ckupdate = {}
        for chunk in newchunks:
            ckupdate[chunk] = 0

        l =0

        while l<lgck:
            maxl = 0
            best_fit = None
            for cuk in newchunks:
                ck = tuple_to_arr(cuk)
                lck = ck.shape[0]
                if np.array_equal(gcka[l:min(l + lck, lgck), :, :], ck) and lck >= maxl:
                    best_fit = cuk
                    maxl = lck
            l = l + maxl
            ckupdate[best_fit] = ckupdate[best_fit] + 1

        for chunk in list(ckupdate.keys()):
            ckupdate[chunk] = ckupdate[chunk] * M[gck]
            ckfreq[chunk] = ckfreq[chunk] + ckupdate[chunk]

    # normalization
    SUM =np.sum(list(ckfreq.values()))
    for chunk in list(ckfreq.keys()):
        ckfreq[chunk] = ckfreq[chunk]/SUM

    return ckfreq
