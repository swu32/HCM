from Learning import *
import numpy as np
import copy

class CG1:
    """
    Chunking graph class with sparse observational data representation.

    Attributes
    ----------
    vertex_list : list
        list of the chunks
    vertex_location : list
        graph location of the corresponding chunk
    visible_chunk_list: list
        list of chunks with their content represented as tuples
    edge_list : list
        Edge information about which chunk combined with which in the model,list of vertex tuples
    y0: int
        Graph height
    x_max: int
        Graph width
    chunks:
        List of chunk object
    concrete_chunks:
        List of chunks with no variables
    abstract_chunks:
        List of chunks with variables
    theta:
        Forgetting rate
    deletion_threshold:
        DT
    H:
        Height
    W:
        Width
    zero:
    relational_graph:
    pad:
        Adjacency threshold between chunks

    Methods
    -------
    get_N:
        returns the number of parsed observations
    empty_counts:
        empty count entries and transition entries in each chunk
    eval_avg_encoding_len:
        Evaluate the expected average encoding resource per sequence length
    getmaxchunksize:
        return the chunk with the biggest volume
    observation_to_tuple:
        convert observations to tuple-like representations
    update_hw:
    get_nonzeroM:
    reinitialize:
        reset chunk count to 0
    convert_chunks_in_arrays:
        save and store learned chunks as arrays
    save_graph:
        save and store chunking graph
    add_chunk:
    check_ancenstry:
        check the ancestors of a specific chunk
    forget:
        decay the count frequency by a factor
    checkcontentoverlap:
        Check if two chunks have the same content
    chunking_reorganization:
        reorganize the transition when a new chunk is added to the representation
    independence_test:
    hypothesis_test:
    """

    def __init__(self, y0=0, x_max=0, DT=0.01, theta=0.95, pad=1):
        self.vertex_list = []
        self.vertex_location = []
        self.visible_chunk_list = []
        # the concrete and abstract chunk list together i
        self.edge_list = []
        self.y0 = y0  # the initial height of the graph, used for plotting
        self.x_max = x_max  # initial x location of the graph
        self.chunks = []  # a dictonary with chunk keys and chunk tuples,
        self.concrete_chunks = []  # list of chunks with no variables
        self.abstract_chunks = []  # list of chunks with variables
        self.theta = theta  # forgetting rate
        self.deletion_threshold = DT
        self.H = 1  # default
        self.W = 1
        self.zero = None
        self.relational_graph = False
        self.pad = pad  # 1: adjacency chunking.

    def get_N(self):
        """returns the number of parsed observations"""
        assert len(self.chunks) > 0
        N = 0
        for ck in self.chunks:
            N = N + ck.count
        return N

    def empty_counts(self):
        """empty count entries and transition entries in each chunk"""
        for ck in self.chunks:
            ck.count = 0
            ck.empty_counts()
        return

    def eval_avg_encoding_len(self):
        """Evaluate the expected average encoding resource per sequence length
            ER = E_{c\in B}(-log(p_c)/|c|)"""
        N = self.get_N()
        ER = 0
        for chunk in self.chunks:
            c_sz = chunk.volume
            c_p = chunk.count / N
            ER = ER + c_p * (- np.log(c_p) / c_sz)

        return ER

    def getmaxchunksize(self):  # TODO: alternatively, update this value upon every chunk creation
        maxchunksize = 0
        if len(self.chunks) > 0:
            for ck in self.chunks:
                if ck.volume > maxchunksize:
                    maxchunksize = ck.volume

        return maxchunksize

    def observation_to_tuple(self, relevant_observations):
        """relevant_observations: array like object"""
        index_t, index_i, index_j = np.nonzero(relevant_observations)  # observation indexes
        value = [relevant_observations[t, i, j] for t, i, j in zip(index_t, index_i, index_j) if
                 relevant_observations[t, i, j] > 0]
        content = set(zip(index_t, index_i, index_j, value))
        maxT = max(index_t)
        return (content, maxT)

    def update_hw(self, H, W):
        self.H = H
        self.W = W
        return

    def update_dimensions(self, dims):
        self.dimensions = dims
        return

    def get_nonzeroM(self):
        nzm = list(self.M.keys()).copy()
        nzmm = nzm.copy()
        nzmm.remove(self.zero)
        return nzmm

    def reinitialize(self):
        # use in the case of reparsing an old sequence using the learned chunks.
        for ck in self.chunks:
            ck.count = 0
        return

    def convert_chunks_in_arrays(self):
        """convert chunk representation to arrays"""
        for chunk in self.chunks:
            chunk.to_array()
        return

    def save_graph(self, name='', path='../OutputData/'):
        """save graph configuration for visualization"""
        import json
        chunklist = []
        for ck in self.chunks:
            ck.to_array()
            chunklist.append(ck.arraycontent)
        data = {}
        data['vertex_location'] = self.vertex_location
        data['edge_list'] = self.edge_list
        # chunklist and graph structure is stored separately
        Name = path + name + 'graphstructure.json'
        a_file = open(Name, "w")
        json.dump(data, a_file)
        a_file.close()
        return

    def add_chunk(self, newc, leftidx=None, rightidx=None):
        """Add a new chunk to the representation"""
        self.vertex_list.append(newc)
        self.chunks.append(newc)  # add observation
        self.visible_chunk_list.append(newc.content)
        newc.index = self.chunks.index(newc)
        newc.H = self.H  # initialize height and weight in chunk
        newc.W = self.W
        # compute the x and y location of the chunk based on pre-existing
        # graph configuration, when this chunk first emerges
        if leftidx is None and rightidx is None:
            x_new_c = self.x_max + 1
            y_new_c = self.y0
            self.x_max = x_new_c
            self.vertex_location.append([x_new_c, y_new_c])
        else:
            l_x, l_y = self.vertex_location[leftidx]
            r_x, r_y = self.vertex_location[rightidx]
            x_c = (l_x + r_x) * 0.5
            y_c = self.y0
            self.y0 = self.y0 + 1
            idx_c = len(self.vertex_list) - 1
            self.edge_list.append((leftidx, idx_c))
            self.edge_list.append((rightidx, idx_c))
            self.vertex_location.append([x_c, y_c])
        return

    def check_ancestry(self, chunk, content):
        """check if content belongs to ancestor"""
        if chunk.parents == []:
            return content != chunk.content
        else:
            return np.any([self.check_ancestry(parent, content) for parent in chunk.parents])


    def forget(self):
        """ discounting past observations if the number of frequencies is beyond deletion threshold"""
        for chunk in self.chunks:
            chunk.count = chunk.count * self.theta
            # if chunk.count < self.deletion_threshold: # for now, cancel deletion threshold, as index to chunk is still vital
            #     self.chunks.pop(chunk)
            for dt in list(chunk.adjacency.keys()):
                for adj in list(chunk.adjacency[dt].keys()):
                    chunk.adjacency[dt][adj] = chunk.adjacency[dt][adj] * self.theta
                    if chunk.adjacency[dt][adj] < self.deletion_threshold:
                        chunk.adjacency[dt].pop(adj)

        return

    def checkcontentoverlap(self, content):
        """check of the content is already contained in one of the chunks"""
        for chunk in self.chunks:
            if chunk.contentagreement(content):
                return chunk
        return None

    def chunking_reorganization(self, previdx, currentidx, cat, dt):
        """ Reorganize marginal and transitional frequencies when a new chunk cat is created
                Parameters:
                        previdx (int): idx of the previous chunk
                        currentidx (int): idx of the current chunk
                        cat (Chunk): concatinated chunk
                        dt: time difference of the end of the previous chunk to the start of the next chunk
                    """

        prev = self.chunks[previdx]
        current = self.chunks[currentidx]
        chunk = self.checkcontentoverlap(cat.content)
        if chunk is None:
            self.add_chunk(cat, leftidx=previdx, rightidx=currentidx)  # add concatinated chunk to the network
            cat.count = prev.adjacency[dt][
                currentidx]  # need to add estimates of how frequent the joint frequency occurred
            cat.adjacency = copy.deepcopy(current.adjacency)
            # iterate through chunk organization find alternative paths arriving at the same chunk
            for _prevck in self.chunks:
                _previdx = _prevck.index
                for _dt in list(_prevck.adjacency.keys()):
                    for _postidx in list(_prevck.adjacency[_dt].keys()):
                        _postck = self.chunks[_postidx]
                        if _previdx != previdx and _postidx != currentidx:
                            _cat = combinechunks(_previdx, _postidx, _dt, self)
                            if _cat != None:
                                if _cat.contentagreement(cat.content):  # the same chunk
                                    cat.count = cat.count + self.chunks[_previdx].adjacency[_dt][_postidx]
                                    self.chunks[_previdx].adjacency[_dt][_postidx] = 0
        else:
            chunk.count = chunk.count + prev.adjacency[dt][currentidx]

        prev.adjacency[dt][currentidx] = 0

        if prev.count - 1 > 0:
            prev.count = prev.count - 1
        if current.count - 1 > 0:
            current.count = current.count - 1
        return

    def independence_test(self, threshold=0.05):
        """Test if the current set of chunks are independent in sequences"""
        N = self.get_N()
        f_obs = []
        f_exp = []
        for clidx in range(0, len(self.chunks)):
            cl = self.chunks[clidx]
            pMcl = cl.count / N
            for cridx in range(0, len(self.chunks)):
                cr = self.chunks[cridx]
                pMcr = cr.count / N
                # the number of times cl transition to cr
                if cl.count == 0:
                    return True
                else:
                    pTclcr = cl.get_transition(cridx) / cl.count
                    pclcr = pMcl * pMcr * N  # expected number of observations
                    oclcr = pMcl * pTclcr * N  # number of observations

                    f_exp.append(pclcr)
                    f_obs.append(oclcr)
        df = (len(self.chunks) - 1) ** 2
        _, pvalue = stats.chisquare(f_obs, f_exp=f_exp, ddof=df)

        if pvalue < threshold:
            return False  # reject independence hypothesis, there is a correlation
        else:
            return True

    def hypothesis_test(self, clidx, cridx, dt, threshold=0.05):
        """independence test on a pair of indexed chunks separated with a time interval dt
            returns True when the chunks are independent or when there is not enough data"""
        cl = self.chunks[clidx]
        cr = self.chunks[cridx]
        assert len(cl.adjacency) > 0
        assert dt in list(cl.adjacency.keys())
        n = self.get_N()

        if cr.count == 0 or (n - cr.count) == 0:
            return True  # not enough data

        # # Expected
        ep1p1 = cl.count / n * cr.count
        ep1p0 = cl.count / n * (n - cr.count)
        ep0p1 = (n - cl.count) / n * cr.count
        ep0p0 = (n - cl.count) / n * (n - cr.count)

        # Observed
        op1p1 = cl.adjacency[dt][cridx]
        op1p0 = cl.get_N_transition(dt) - cl.adjacency[dt][cridx]
        op0p1 = 0
        op0p0 = 0
        for ncl in list(self.chunks):  # iterate over p0, which is the cases where cl is not observed
            if ncl != cl:
                if dt in list(ncl.adjacency.keys()):
                    if cridx in list(ncl.adjacency[dt].keys()):
                        op0p1 = op0p1 + ncl.adjacency[dt][cridx]
                        for ncridx in list(ncl.adjacency[dt].keys()):
                            if ncridx != cr.index:
                                op0p0 = op0p0 + ncl.adjacency[dt][ncridx]

        obs = [op1p1, op1p0, op0p1, op0p0]
        exp = [ep1p1, ep1p0, ep0p1, ep0p0]
        obs = [item/sum(obs) for item in obs]
        exp = [item/sum(exp) for item in exp]
        _, pvalue = stats.chisquare(obs, f_exp=exp, ddof=1)
        if pvalue < threshold:
            return False  # reject independence hypothesis, there is a correlation
        else:
            return True
