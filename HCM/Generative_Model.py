import numpy as np
import random
from Learning import *
from Chunking_Graph import *
import json


def partition_seq(this_sequence, bag_of_chunks):
    """
    Partition a sequence according to bag of chunks
    Parameters:
             this_sequence (list)
             bag_of_chunks: list of tuple formatted chunks
    Returns:
             partitioned_sequence: list of maximum sized fitting chunks
     """
    i = 0
    end_of_sequence = False
    partitioned_sequence = []

    while end_of_sequence == False:
        max_chunk = None
        max_length = 0
        for chunk in bag_of_chunks:
            this_chunk = json.loads(chunk)
            if this_sequence[i : i + len(this_chunk)] == this_chunk:
                if len(this_chunk) > max_length:
                    max_chunk = this_chunk
                    max_length = len(max_chunk)

        if max_chunk == None:
            partitioned_sequence.append([this_sequence[i]])
            i = i + 1
        else:
            partitioned_sequence.append(list(max_chunk))
            i = i + len(max_chunk)

        if i >= len(this_sequence):
            end_of_sequence = True

    return partitioned_sequence


def sample_from_distribution(states, prob):
    """
    Parameters:
             states (list)
             prob (list)
    Returns:
             one sampled state with its assigned probability
    """
    prob = [k / sum(prob) for k in prob]
    cdf = [0.0]
    for s in range(0, len(states)):
        cdf.append(cdf[s] + prob[s])
    k = np.random.rand()
    for i in range(1, len(states) + 1):
        if k >= cdf[i - 1]:
            if k < cdf[i]:
                return states[i - 1], prob[i - 1]


def generate_hierarchical_sequence(marginals, s_length=10):
    """
    Generate sequence based on a set of chunks with their assigned probability.

    Parameters
    ----------
    marginals (dict)
    s_length (int)

    Returns
    -------
    sequence
    """
    H, W = tuple_to_arr(list(marginals.keys())[0]).shape[1:]
    sequence = np.zeros([s_length, H, W])
    i = 0
    while i < s_length:
        indices = np.arange(len(list(marginals.keys())))
        idx = np.random.choice(indices, p=list(marginals.values()))
        new_sample = list(marginals.keys())[idx]
        t_len = tuple_to_arr(new_sample).shape[0]
        sample_duration = min(s_length, i + t_len) - i
        sequence[i : min(s_length, i + t_len), :, :] = tuple_to_arr(new_sample)[
            0:sample_duration, :, :
        ]
        i = i + t_len
    return sequence


def generate_random_hierarchical_sequence(marginals, s_length=10):
    """
    A generated random sequence should be the same when it is being parsed. clcr should not be a chunk inside the marginals.
    Parameters.
    """
    clen = {}
    for chunk in list(marginals):
        clen[chunk] = tuple_to_arr(chunk).shape[0]

    def checkoverlap(sequence, i, clen, this_chunk, len_this_chunk):
        """
        Check the generated sequences up to this moment overlap with pre-existing chunks.
        """
        for chunk in clen:
            lenc = clen[chunk]
            if (
                chunk != (((0,),),)
                and chunk != this_chunk
                and this_chunk != (((0,),),)
                and clen[chunk] > len_this_chunk
            ):
                tuple_seqchunk = arr_to_tuple(
                    sequence[
                        max(0, i + len_this_chunk - lenc) : min(
                            s_length, i + len_this_chunk
                        ),
                        :,
                        :,
                    ]
                )
                if tuple_seqchunk == chunk:
                    return True
        return False

    H, W = tuple_to_arr(list(marginals.keys())[0]).shape[1:]
    sequence = np.zeros([s_length, H, W])

    i = 0
    while i < s_length:
        new_sample = np.random.choice(
            list(marginals.keys()), p=list(marginals.values())
        )
        t_len = tuple_to_arr(new_sample).shape[0]
        sample_duration = min(s_length, i + t_len) - i
        sequence[i : min(s_length, i + t_len), :, :] = tuple_to_arr(new_sample)[
            0:sample_duration, :, :
        ]

        while checkoverlap(
            sequence, i, clen, new_sample, sample_duration
        ):  # check chunk overlaps
            new_sample = np.random.choice(
                list(marginals.keys()), p=list(marginals.values())
            )
            t_len = tuple_to_arr(new_sample).shape[0]
            sample_duration = min(s_length, i + t_len) - i
            sequence[i : min(s_length, i + t_len), :, :] = tuple_to_arr(new_sample)[
                0:sample_duration, :, :
            ]
        i = i + t_len
    return sequence


def dirichlet_flat(N, sort=True):
    alpha = tuple(
        [1 for _ in range(0, N)]
    )  # coefficient for the flat dirichlet distribution
    if sort:
        return sorted(list(np.random.dirichlet(alpha, 1)[0]), reverse=True)
    else:
        return list(np.random.dirichlet(alpha, 1)[0])


def generate_new_chunk(setofchunks):
    a = list(setofchunks)[
        np.random.choice(np.arange(0, len(setofchunks), 1))
    ]  # better to be to choose based on occurrance probability
    b = list(setofchunks)[
        np.random.choice(np.arange(0, len(setofchunks), 1))
    ]  # should exclude 0
    va, vb = tuple_to_arr(a), tuple_to_arr(b)
    la, lb = va.shape[0], vb.shape[0]
    lab = la + lb
    vab = np.zeros([lab, 1, 1])
    vab[0 : va.shape[0], :, :] = va
    vab[va.shape[0] :, :, :] = vb
    ab = arr_to_tuple(vab)
    if ab in setofchunks:
        return generate_new_chunk(setofchunks)
    else:
        return ab, a, b


def generative_model_random_combination(D=3, n=5):
    """ randomly generate a set of hierarchical chunks.
        D: number of recombinations
        n: number of atomic, elemetary chunks """

    def check_independence(constraints, M):
        """check: p(ab) < p(a)p(b) for all [ab, a, b] in M."""
        for ab, a, b in constraints:
            if M[ab] <= M[a] * M[b] + 0.003:
                return False
        return True

    cg = Chunking_Graph()
    setofchunks = []
    for i in range(0, n):
        zero = np.zeros([1, 1, 1])
        zero[0, 0, 0] = i
        chunk = zero
        setofchunks.append(arr_to_tuple(chunk))

    setofchunkswithoutzero = setofchunks.copy()
    setofchunkswithoutzero.remove(arr_to_tuple(np.zeros([1, 1, 1])))
    constraints = []
    for d in range(0, D):
        # pick random new combinations
        ab, a, b = generate_new_chunk(setofchunkswithoutzero)
        constraints.append([ab, a, b])
        setofchunks.append(ab)
        setofchunkswithoutzero = setofchunks.copy()
        setofchunkswithoutzero.remove(arr_to_tuple(np.zeros([1, 1, 1])))

    satisfy_constraint = False
    while satisfy_constraint == False:
        genp = dirichlet_flat(
            len(setofchunks), sort=False
        )  # assign probabilities to this set of chunks
        p0 = max(genp)
        genp.remove(p0)
        p = [p0] + genp
        M = dict(
            zip(setofchunks, p)
        )  # so that empty observation is always ranked the highest.
        cg.M = M
        satisfy_constraint = check_independence(constraints, M)

    return cg


def to_chunking_graph(cg):
    """Specify the graph connectivity given cg with cg.M. """

    def check_completeness(chunks, gtM):
        for ck in list(gtM.keys()):
            if ck not in chunks:
                return False
        return True

    M = cg.M

    atomic_chunks = find_atomic_chunks(M)  # initialize with atomic chunks
    for ac in atomic_chunks:
        cg.add_chunk_to_vertex(ac)

    chunks = set()
    for ck in list(atomic_chunks.keys()):
        chunks.add(ck)

    complete = False
    proposed_joints = set()
    while complete == False:
        # calculate the mother chunk and the father chunk of the joint chunks
        joints_and_freq = calculate_joints(
            chunks, M
        )  # the expected number of joint observations
        new_chunk, cl, cr = pick_chunk_with_max_prob(joints_and_freq)
        while new_chunk in proposed_joints:
            joints_and_freq.pop(new_chunk)
            new_chunk, cl, cr = pick_chunk_with_max_prob(joints_and_freq)

        cg.add_chunk_to_vertex(
            new_chunk, left=cl, right=cr
        )  # update cg graph with newchunk and its components
        chunks.add(new_chunk)
        proposed_joints.add(new_chunk)
        complete = check_completeness(chunks, M)
    return cg


def pick_chunk_with_max_prob(joints_and_freq):
    chunk = max(
        joints_and_freq, key=lambda k: joints_and_freq[k][0]
    )  # the maximal value of the current dictionary
    cl = joints_and_freq[chunk][1]
    cr = joints_and_freq[chunk][2]
    return chunk, cl, cr


def find_atomic_chunks(M):
    def eval_atom_chunk_in_M(chunk, M):
        def get_n(chunk, Mchunk):
            mck = tuple_to_arr(Mchunk)
            ck = tuple_to_arr(chunk)
            n = 0
            for i in range(0, mck.shape[0]):
                # find the maximally fitting chunk in bag of chunks to partition Mchunk
                if np.array_equal(
                    mck[i : min(i + ck.shape[0], mck.shape[0]), :, :], ck
                ):
                    n = n + 1
            return n

        Echunkn = 0
        for key in list(M.keys()):
            Mchunk = tuple_to_arr(key)
            Mchunkp = (
                get_n(chunk, Mchunk) * M[key]
            )  # count how many times chunk occurrs in M_chunk
            Echunkn = Echunkn + Mchunkp
        return Echunkn

    atomic_chunks = {}
    for chunk in list(M.keys()):
        if tuple_to_arr(chunk).shape[0] == 1:
            atomic_chunks[chunk] = eval_atom_chunk_in_M(chunk, M)

    # normalize occurrance count to a probability
    SUM_occur = sum(atomic_chunks.values())
    for key in list(atomic_chunks.keys()):
        atomic_chunks[key] = atomic_chunks[key] / SUM_occur
    return atomic_chunks


def calculate_joints(chunks, M):
    """Calculate the best joints to combine with the pre-existing chunks"""

    def combine_joints(chunk1, chunk2):
        c1 = tuple_to_arr(chunk1)
        c2 = tuple_to_arr(chunk2)
        chunk12 = np.zeros([c1.shape[0] + c2.shape[0], 1, 1])
        chunk12[0 : c1.shape[0], :, :] = c1
        chunk12[c1.shape[0] :, :, :] = c2
        chunk12 = arr_to_tuple(chunk12)
        return chunk12

    ZERO = arr_to_tuple(np.zeros([1, 1, 1]))
    joints_and_freq = {}
    for chunk1 in chunks:
        for chunk2 in chunks:
            if chunk1 != ZERO and chunk2 != ZERO:
                chunk12 = combine_joints(chunk1, chunk2)
                ck_prob = calculate_expected_occurrance(
                    chunk12, chunks, M
                )  # need to use the entire chunk set to
                joints_and_freq[chunk12] = (ck_prob[chunk12], chunk1, chunk2)
    return joints_and_freq


def calculate_expected_occurrance(chunk12, bagofchunks, M):
    """In case when [1] [2] and [1,2] both exist, evaluation of [1] and [2] is included in [1,2]
     """
    # alternatively, partition M according to the new chunks.
    newchunks = bagofchunks.union(
        {chunk12}
    )  # new bag of chunks when chunk12 is appended.
    ckfreq = {}
    for chunk in newchunks:
        ckfreq[chunk] = 0

    for gck in list(M.keys()):
        gcka = tuple_to_arr(gck)
        lgck = gcka.shape[0]

        ckupdate = {}
        for chunk in newchunks:
            ckupdate[chunk] = 0

        l = 0

        while l < lgck:
            maxl = 0
            best_fit = None
            for cuk in newchunks:
                ck = tuple_to_arr(cuk)
                lck = ck.shape[0]
                if (
                    np.array_equal(gcka[l : min(l + lck, lgck), :, :], ck)
                    and lck >= maxl
                ):
                    best_fit = cuk
                    maxl = lck
            l = l + maxl
            ckupdate[best_fit] = ckupdate[best_fit] + 1

        for chunk in list(ckupdate.keys()):
            ckupdate[chunk] = ckupdate[chunk] * M[gck]
            ckfreq[chunk] = ckfreq[chunk] + ckupdate[chunk]

    # normalization
    SUM = np.sum(list(ckfreq.values()))
    for chunk in list(ckfreq.keys()):
        ckfreq[chunk] = ckfreq[chunk] / SUM

    return ckfreq
