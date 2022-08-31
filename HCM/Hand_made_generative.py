from Learning import *
from Chunking_Graph import *


def simple_seq():
    seq = np.array(
        [
            [[0]],
            [[1]],
            [[2]],
            [[3]],
            [[4]],
            [[5]],
            [[6]],
            [[7]],
            [[8]],
            [[9]],
            [[10]],
            [[11]],
            [[12]],
        ]
    )
    return seq


def hierarchy1d():
    cg = Chunking_Graph()
    # ================== Initialization Process ==================
    # level I
    # A = np.zeros([3, 1, 1])
    one = np.array([[[1]]])
    two = np.array([[[2]]])
    D = np.array([[[2]], [[1]]])
    C = np.array([[[1]], [[2]]])

    A = np.array([[[1]], [[2]], [[1]]])
    B = np.array([[[2]], [[1]], [[1]]])

    # level II
    CD = np.array([[[1]], [[2]], [[2]], [[1]]])  #
    BD = np.array([[[2]], [[1]], [[1]], [[2]], [[1]]])  #
    AB = np.array([[[1]], [[2]], [[1]], [[2]], [[1]], [[1]]])  #

    # level III
    ACD = np.array([[[1]], [[2]], [[1]], [[1]], [[2]], [[2]], [[1]]])  #
    ABBD = np.array(
        [[[1]], [[2]], [[1]], [[2]], [[1]], [[1]], [[2]], [[1]], [[1]], [[2]], [[1]]]
    )
    E = np.zeros([1, 1, 1])
    E[0, 0, 0] = 0
    stim_set = [
        arr_to_tuple(E),
        arr_to_tuple(one),
        arr_to_tuple(two),
        arr_to_tuple(C),
        arr_to_tuple(D),
        arr_to_tuple(A),
        arr_to_tuple(B),
        arr_to_tuple(CD),
        arr_to_tuple(AB),
        arr_to_tuple(BD),
        arr_to_tuple(ACD),
        arr_to_tuple(ABBD),
    ]

    """Produce a generic generative model"""
    alpha = tuple(
        [1 for i in range(0, len(stim_set))]
    )  # coefficient for the flat dirichlet distribution
    probs = sorted(list(np.random.dirichlet(alpha, 1)[0]), reverse=True)
    generative_marginals = {}
    # generative_marginals[arr_to_tuple(E)] = probs[0]
    for i in range(0, len(stim_set)):
        generative_marginals[stim_set[i]] = probs[i]
    cg.M = generative_marginals
    return cg


def compositional_imgs():
    """Each higher level is made of composing elements in more elementary levels
    In the generative model, images are iid sampled"""

    # each chunk contains some probability of occurrance
    # at each time point, a chunk is sampled and displayed on one of the dimensions.
    # level I, primitives
    cg = Chunking_Graph()

    # =========== Initialization Process ================

    zero = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ]
    )

    A = np.array(
        [
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        ]
    )

    B = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ]
    )

    C = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        ]
    )

    D = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 1],
            ]
        ]
    )

    E = np.array(
        [
            [
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ]
    )

    # level II
    AB = np.array(
        [
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        ]
    )

    BC = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        ]
    )

    CE = np.array(
        [
            [
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        ]
    )

    AE = np.array(
        [
            [
                [1, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        ]
    )

    AD = np.array(
        [
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 1, 1],
            ]
        ]
    )

    # level III

    ABE = np.array(
        [
            [
                [1, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        ]
    )

    AED = np.array(
        [
            [
                [1, 1, 1, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 1, 1],
            ]
        ]
    )

    ABCE = np.array(
        [
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ]
        ]
    )

    stim_set = [
        arr_to_tuple(zero),
        arr_to_tuple(AB),
        arr_to_tuple(BC),
        arr_to_tuple(AE),
        arr_to_tuple(AD),
        arr_to_tuple(ABE),
        arr_to_tuple(AED),
        arr_to_tuple(ABCE),
        arr_to_tuple(A),
        arr_to_tuple(B),
        arr_to_tuple(C),
        arr_to_tuple(D),
        arr_to_tuple(E),
        arr_to_tuple(CE),
    ]

    """Produce a generic generative model"""
    alpha = tuple(
        [1 for i in range(0, len(stim_set))]
    )  # coefficient for the flat dirichlet distribution
    probs = sorted(list(np.random.dirichlet(alpha, 1)[0]), reverse=True)
    generative_marginals = {}
    for i in range(0, len(stim_set)):
        generative_marginals[stim_set[i]] = probs[i]
    cg.M = generative_marginals

    return cg


def transfer_original():
    """"""
    cg = Chunking_Graph()
    # ================== Initialization Process ==================
    # level I

    one = np.array([[[1]]])
    two = np.array([[[2]]])
    C = np.array([[[1]], [[2]]])
    D = np.array([[[2]], [[1]]])
    A = np.array([[[1]], [[2]], [[1]]])
    B = np.array([[[2]], [[1]], [[1]]])

    # level II
    CD = np.array([[[1]], [[2]], [[2]], [[1]]])
    BD = np.array([[[2]], [[1]], [[1]], [[2]], [[1]]])
    AB = np.array([[[1]], [[2]], [[1]], [[2]], [[1]], [[1]]])

    E = np.zeros([1, 1, 1])
    E[0, 0, 0] = 0
    stim_set = [
        arr_to_tuple(one),
        arr_to_tuple(two),
        arr_to_tuple(AB),
        arr_to_tuple(BD),
        arr_to_tuple(CD),
        arr_to_tuple(A),
        arr_to_tuple(B),
        arr_to_tuple(C),
        arr_to_tuple(D),
        arr_to_tuple(E),
    ]
    probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    generative_marginals = {}
    generative_marginals[arr_to_tuple(E)] = probs[0]
    for i in range(0, len(stim_set) - 1):
        generative_marginals[stim_set[i]] = probs[i + 1]
    cg.M = generative_marginals
    return cg


def generateseq(groupcond, seql=600):
    seq = []
    if groupcond == "c2":
        while len(seq) < seql:
            seq = seq + np.random.choice([[1, 2], [3], [4]])
        return seq[0:seql]
    if groupcond == "c3":
        while len(seq) < seql:
            seq = seq + np.random.choice([[1, 2, 3], [4]])
        return seq[0:seql]
    if groupcond == "ind":
        while len(seq) < seql:
            seq = seq + [np.random.choice([1, 2, 3, 4])]
        return seq[0:seql]


def generate_interfering_env():
    int_env = Chunking_Graph()  # interfering environment

    E = np.zeros([1, 1, 1])
    A = np.array([[[1]]])
    B = np.array([[[2]]])
    C = np.array([[[3]]])
    D = np.array([[[4]]])
    F = np.array([[[5]]])

    FD = np.array([[[5]], [[4]]])  #
    FC = np.array([[[5]], [[3]]])  #
    BD = np.array([[[2]], [[4]]])  #
    BDFD = np.array([[[2]], [[4]], [[5]], [[4]]])

    stim_set = [
        arr_to_tuple(E),
        arr_to_tuple(A),
        arr_to_tuple(B),
        arr_to_tuple(C),
        arr_to_tuple(D),
        arr_to_tuple(F),
        arr_to_tuple(FD),
        arr_to_tuple(FC),
        arr_to_tuple(BD),
        arr_to_tuple(BDFD),
    ]

    """Produce a random generative model"""
    alpha = tuple(
        [1 for _ in range(0, len(stim_set))]
    )  # coefficient for the flat dirichlet distribution
    probs = sorted(list(np.random.dirichlet(alpha, 1)[0]), reverse=True)
    generative_marginals = {}

    j = 0
    for i in reversed(range(0, len(stim_set))):
        generative_marginals[stim_set[i]] = probs[j]
        j = j + 1

    int_env.M = generative_marginals
    int_env.vertex_list = stim_set
    int_env.edge_list = [(4, 6), (5, 6), (5, 7), (3, 7), (2, 8), (4, 8), (8, 9), (6, 9)]
    int_env.vertex_location = [
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
        [5, 0],
        [6, 0],
        [5.0, 1],
        [4.0, 1],
        [3.0, 1],
        [4.5, 2],
    ]
    return int_env


def generate_facilitative_env():
    fac_env = Chunking_Graph()
    E = np.zeros([1, 1, 1])
    A = np.array([[[1]]])
    B = np.array([[[2]]])
    C = np.array([[[3]]])
    D = np.array([[[4]]])
    AB = np.array([[[1]], [[2]]])  #
    CA = np.array([[[3]], [[1]]])  #
    ABCA = np.array([[[1]], [[2]], [[3]], [[1]]])  #
    CAD = np.array([[[3]], [[1]], [[4]]])  #
    ABCACAD = np.array([[[1]], [[2]], [[3]], [[1]], [[3]], [[1]], [[4]]])  #

    stim_set = [
        arr_to_tuple(E),
        arr_to_tuple(A),
        arr_to_tuple(B),
        arr_to_tuple(C),
        arr_to_tuple(D),
        arr_to_tuple(AB),
        arr_to_tuple(CA),
        arr_to_tuple(ABCA),
        arr_to_tuple(CAD),
        arr_to_tuple(ABCACAD),
    ]

    """Produce a random generative model"""
    alpha = tuple(
        [1 for _ in range(0, len(stim_set))]
    )  # coefficient for the flat dirichlet distribution
    probs = sorted(list(np.random.dirichlet(alpha, 1)[0]), reverse=True)
    generative_marginals = {}

    j = 0
    for i in reversed(range(0, len(stim_set))):
        generative_marginals[stim_set[i]] = probs[j]
        j = j + 1

    fac_env.M = generative_marginals
    fac_env.vertex_list = stim_set
    fac_env.edge_list = [
        (1, 5),
        (2, 5),
        (1, 6),
        (3, 6),
        (5, 7),
        (6, 7),
        (6, 8),
        (4, 8),
        (8, 9),
        (7, 9),
    ]
    fac_env.vertex_location = [
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
        [5, 0],
        [3.0, 1],
        [4.0, 1],
        [3.5, 3],
        [4.5, 2],
        [3.0, 4],
    ]
    return fac_env


def impose_representation():
    cg = Chunking_Graph()
    E = np.zeros([1, 1, 1])
    A = np.array([[[1]]])
    B = np.array([[[2]]])
    C = np.array([[[3]]])
    D = np.array([[[4]]])
    AB = np.array([[[1]], [[2]]])  #
    CA = np.array([[[3]], [[1]]])  #
    ABCA = np.array([[[1]], [[2]], [[3]], [[1]]])  #
    CAD = np.array([[[3]], [[1]], [[4]]])  #
    N = 200  # previous training experience

    stim_set = [
        arr_to_tuple(E),
        arr_to_tuple(A),
        arr_to_tuple(B),
        arr_to_tuple(C),
        arr_to_tuple(D),
        arr_to_tuple(AB),
        arr_to_tuple(CA),
        arr_to_tuple(ABCA),
        arr_to_tuple(CAD),
    ]

    """Produce a random generative model"""
    alpha = tuple(
        [1 for _ in range(0, len(stim_set))]
    )  # coefficient for the flat dirichlet distribution
    probs = sorted(list(np.random.dirichlet(alpha, 1)[0]), reverse=True)
    generative_marginals = {}

    j = 0
    for i in reversed(range(0, len(stim_set))):
        generative_marginals[stim_set[i]] = probs[j] * N
        j = j + 1

    # '''Produce a generic generative model'''
    # probs =[1/len(stim_set)]*len(stim_set)
    # generative_marginals = {}
    # # generative_marginals[arr_to_tuple(E)] = probs[0]
    # for i in range(0, len(stim_set)):
    #     generative_marginals[stim_set[i]] = probs[i]
    cg.M = generative_marginals
    cg.vertex_list = stim_set
    cg.edge_list = [(1, 5), (2, 5), (1, 6), (3, 6), (5, 7), (6, 7), (6, 8), (4, 8)]
    cg.vertex_location = [
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
        [5, 0],
        [3.0, 1],
        [4.0, 1],
        [3.5, 3],
        [4.5, 2],
    ]
    return cg
