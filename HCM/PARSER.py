from Chunking_Graph import sample_from_distribution
import numpy as np
import pickle
from Hand_made_generative import generateseq
def PARSER(speech, PS = {'1':1,'2':1,'3':1,'4':1}, return_history = False):
    '''
    Percept Shaper (PS)
    PS is composed of the internal representations of the displayed material and may be thought of as a memory store
    or a mental lexicon. A weight, which reflects the person's familiarity with the item, is assigned to each element in PS.
    '''

    #print('Initial Percept Shaper: ' + str(PS))
    #print 'Stimulus              : ' + str(speech)
    chunk_record = []
    percept_history = {}
    i=0
    while i < len(speech)-1: # while there is at least one stimulus left to process
        # print("----------------------------------------------")
        # print('i (start of segment): ' + str(i))
        # print('stimulus coming up: ' + speech[i:i+10])
        # print('PS: '  + str(PS))


        # Attention
        n_components = np.random.choice([1,2,3])
        while i + n_components > len(speech): # if there is less stimulus left than attention capacity, then `decrease the capacity
            n_components -= 1

        # print("n_components: " + str(n_components))

        # Perceive
        components=[]
        for c in range(n_components):
            component_start, component_end = i,i+1 # smallest possible unit
            #print 'smallest possible unit: ' + str(speech[component_start : component_end])

            # if there is more stimulus to be parsed than a single stimulus
            # look for the longest unit, that has a threshold above which a unit is able to shape perception (this is set to 1)
            while component_end < len(speech) \
            and speech[component_start : (component_end + 1)] in PS.keys() \
            and PS[ speech[component_start : (component_end + 1)] ] >= 1:
                component_end += 1


            components.append(speech[component_start : component_end])
            i = component_end
            component = speech[component_start : component_end]
            denum =sum([PS[comp] for comp in list(PS.keys())])
            if denum>0 and component in list(PS.keys()):
                comp_p = PS[component]/denum
            else:
                comp_p = 0
            percept_history[i] = (component, comp_p)# components, and their occurrence probabilities
            #print "component_end: " + str(component_end)


            if i == len(speech):
                break

        # print('components: ' + str(components))
        chunk_record = chunk_record + components


        for c in components:
            # Add 0.5 weight to old units in PS if they served to shape perception
            if c in PS: PS[c] += 0.5

            # Interference is simulated by decreasing the weights of the units (with 0.005) in which any of the stimuli
            # involved in the currently processed unit are embedded.
            # print("Interference part")

            if len(c)>2: # find any part of c in any other element or part of element that is present in PS
                # decompose c into primitives
                c_parts=[]
                for x in range(0,len(c)): # these are the primitve (syllable) starts
                    c_parts.append(c[ x : x+1])
                # print('component parts: ' + str(c_parts))

                for part in c_parts:
                    for element in PS.keys():
                        for y in range(0,len(element)): # parse the element by stimuli
                            if part == element[y:y+1] and c != element: # if any part of the element matches the part of the component (not taking into account the percept shaping unit itself)
                                PS[element] -= 0.00005
                                # print('PS element that interferes: ' + str(element))

        # Forgetting is simulated by decreasing all the units by a fixed value: 0.05
        for element in list(PS.keys()).copy():
            PS[element] -= 0.05
            # Any element is removed from PS when its weight becomes null.
            if PS[element] <= 0:
                del PS[element]

        # Creation; the percept is introduced to PS with a weight of 1 (this alone will not be affected by forgetting in this cycle)
        percept=''
        for c in components:
            percept=percept + c
        # print('percept: ' + percept)
        if percept not in PS.keys():
            PS[percept] = 1
        # If it is already in PS (but it was under the shaping threshold, hence it didn't shape perception), add weight
        else:
            PS[percept] += 0.5


    print("Final Percept Shaper (with chunk weights): ")
    if return_history:
        return percept_history
    else:
        return PS, chunk_record

def parser_imagination(PS, n):
    # n: length
    s = ''
    t = 0
    while t <= n:
        k,v = sample_from_distribution(list(PS.keys()), list(PS.values()))
        s = s + k
        t = t + len(k)

    seq = [int(c) for c in s]
    imagined_sequence = np.array(seq).reshape([-1,1,1])

    # convert back to np.array format
    return imagined_sequence[:n,:,:]



def c3_parser_learning():
    """Train HCM on c3 condition of SRT instruction sequences"""
    df = {}
    df['time'] = []
    df['chunksize'] = []
    df['ID'] = []


    for ID in range(0, 50): # across 30 runs
        seq = np.array(generateseq('c3', seql=600)).reshape((600, 1, 1))
        speech = list((seq.flatten().astype('int')))
        speech = ''.join(str(e) for e in speech)
        ps, chunkrecord = PARSER(speech, PS = {'1':1,'2':1,'3':1,'4':1})  # with the rational chunk models, rational_chunk_all_info(seq, cg)
        t = 0
        for chunk in list(chunkrecord):
            df['time'].append(int(t))
            df['chunksize'].append(len(chunk))
            df['ID'].append(ID)
            t = t + len(chunk)
    with open('../OutputData/PARSER_time_chunksize.pkl', 'wb') as f:
        pickle.dump(df, f)
    return



def PARSER_c3_probability(training_seq):
    ''' Evaluate prediction probability of HCM trained on SRT instruction sequences '''

    time_series = np.array(training_seq).reshape([-1,1,1])
    seq = time_series.astype(int)
    speech = list((seq.flatten().astype('int')))
    speech = ''.join(str(e) for e in speech)

    chunkrecord = PARSER(speech, PS={'1': 1, '2': 1, '3': 1,'4': 1}, return_history=True)  # with the rational chunk models, rational_chunk_all_info(seq, cg)
    p = []
    eps = 0.05


    for t in range(0, len(seq)):
        if t in list(chunkrecord.keys()):
            p.append(chunkrecord[t][1])
        else:# a within-chunk element
            p.append(1 - 4*eps)

    return p