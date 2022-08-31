import numpy as np


class Chunk:
    """ Spatial Temporal Chunk
        At the moment, it lacks a unique identifier for which of the chunk is which, making the searching process
        diffidult, ideally, each chunk should have its own unique name, (ideally related to how it looks like)


    Attributes
    ----------
    content:
        chunk content, represented as set of tuples, with the location of signal specified and the observational values
    variable:
        a list of other chunks
    H:
    W:
    index:
        location in the list of chunks
    count:
        frequencies
    pad:
        boundary size for nonadjacency detection, set the pad to not 1 to enable this feature
    adjacency:
        transition frequencies
    birth:
        chunk creation time
    volume:
        the size of the chunk
    indexloc:
        index locations of the nonzero chunk entries
    arraycontent:
        content casted as np.array
    boundarycontent:
        set of boundaries
    matching_threshold:
        threshold of matching formulation
    abstraction:
        list of abstract chunks summarizing this chunk
    entailment:
        the chunk that the abstract chunk is pointing to
    h,w,v:
        discount coefficient when computing similarity between two chunks, relative to the temporal discount being 1

    Methods
    -------
    get_full_content:
    find_content_recursive:
        recursively finding the entailing chunks if the current chunk is abstract
    update_variable_count:
    update:
    to_array:
        convert single chunk from tuple into np.array representation
    get_N_transition:
        Calculate the number of times this chunk transition to another chunk
    get_index:
        returns the index location of the chunk observations.
    get_index_padded:
        returns the padded index location of the chunks, used for non-adjacent chunk detection
    concatinate:
        concatinate one chunk with another chunk
    average_content:
        average the stored content with the matching sequence
    variable_check_match:
        check the matching of a variable chunk
    check_match:
        check whether chunk matches with sequential components
    check_adjacency:
    checksimilarity:
    pointdistance:
    update_transition:
    empty_counts:
        clear the transition and marginal count of each chunk
    contentagreement:
    get_transition:
        """

    # A code name unique to each chunk
    def __init__(
        self,
        chunkcontent,
        variable=[],
        count=1,
        H=None,
        W=None,
        dims=[55, 55, 64],
        pad=1,
    ):
        """chunkcontent: a list of tuples describing the location and the value of observation"""
        self.content = set(
            chunkcontent
        )  # the first dimension is always time, the last is value
        self.variable = variable
        self.T = int(
            max(np.array(chunkcontent)[:, 0]) + 1
        )  # those should be specified when joining a chunking graph
        self.H = H
        self.W = W
        self.dimensions = dims  # middle dimensions, excluding the time dimension and the value dimension
        self.index = None
        self.count = count  #
        self.pad = pad
        self.adjacency = {}
        self.birth = None  # chunk creation time
        self.volume = len(self.content)  #
        self.indexloc = self.get_index()
        self.arraycontent = None
        self.boundarycontent = set()
        self.get_index_padded()
        self.D = 10
        self.matching_threshold = 0.8
        self.matching_seq = {}
        self.abstraction = []  # what are the variables summarizing this chunk
        self.entailment = []  # concrete chunks that the variable is pointing to

        # discount coefficient when computing similarity between two chunks, relative to the temporal discount being 1
        self.h = 1.0
        self.w = 1.0
        self.v = 1.0

    def get_full_content(self):
        """returns a list of all possible content that this chunk can take"""
        self.possible_path = []
        self.find_content_recursive(self, [])
        return self.possible_path

    def find_content_recursive(self, node, path):
        path = path + list(node.content)
        if len(list(node.variable)) == 0:
            self.possible_path.append(path)
            return
        else:
            for Var in node.variable:
                self.find_content_recursive(Var, path)

    def update_variable_count(self):
        for ck in self.variable:
            ck.update()
        return

    def update(self):
        self.count = self.count + 1
        if len(self.variable) > 0:
            self.update_variable_count()  # update the count of the subordinate chunks
        return

    def to_array(self):
        """convert the content into array"""
        arrep = np.zeros(
            (
                int(max(np.atleast_2d(np.array(list(self.content)))[:, 0]) + 1),
                *self.dimensions,
            )
        )
        for t, i, j, v in self.content:
            arrep[t, i, j] = v
        self.arraycontent = arrep
        return

    def get_N_transition(self, dt):
        assert dt in list(self.adjacency.keys())
        N = 0
        for item in self.adjacency[dt]:
            N = N + self.adjacency[dt][item]
        return N

    def get_index(self):
        """ Get index location the nonzero chunk locations in chunk content  """
        return set(map(tuple, np.array(list(self.content))[:, 0:3]))

    def get_index_padded(self):
        """ Get padded index arund the nonzero chunk locations """
        padded_index = self.indexloc.copy()
        chunkcontent = self.content
        for p in range(1, self.pad + 1):
            for c in chunkcontent:
                padded_boundary_set = set()
                for d in range(0, len(c) - 1):
                    contentloc = list(c)[:-1].copy()  # excluding value within content
                    contentloc[d] = min(contentloc[d] + p, self.dimensions[d])
                    padded_boundary_set.add(tuple(contentloc))
                    contentloc = list(c)[:-1].copy()
                    contentloc[d] = max(contentloc[d] - p, 0)
                    padded_boundary_set.add(tuple(contentloc))
                    if p == 1 and padded_boundary_set.issubset(self.indexloc) == False:
                        self.boundarycontent.add(
                            c
                        )  # add chunk content to the boundary content of this chunk
                padded_index = padded_index.union(padded_boundary_set)

        return padded_index

    def conflict(self, c_):

        return False

    def concatinate(self, cR):
        if self.check_adjacency(cR):
            clcrcontent = self.content | cR.content
            clcr = Chunk(
                list(clcrcontent), H=self.H, W=self.W, pad=cR.pad, dims=self.dimensions
            )
            return clcr
        else:
            return None

    def average_content(self):
        # average the stored content with the sequence
        # calculate average deviation
        averaged_content = set()
        assert len(self.matching_seq) > 0
        for m in list(self.matching_seq.keys()):  # iterate through content points
            thispt = list(m)
            n_pt = len(self.matching_seq[m])
            count = max(1, self.count)
            for i in range(0, len(thispt)):
                thispt[i] = int(
                    thispt[0]
                    + 1
                    / count
                    * sum(map(lambda pt: pt[i] - thispt[i], self.matching_seq[m]))
                    / n_pt
                )
            averaged_content.add(tuple(thispt))

        self.content = averaged_content
        self.T = int(
            np.atleast_2d(np.array(list(self.content)))[:, 0].max() + 1
        )  # those should be specified when joining a chunking graph
        self.get_index()
        self.get_index_padded()  # update boundary content
        return

    def variable_check_match(
        self, seq
    ):  # a sequence matches any of its instantiated variables
        """returns true if the sequence matches any of the variable instantiaions"""
        if len(self.variable) == 0:
            return self.check_match(seq)
        else:
            match = []
            for ck in self.variable:
                match.append(ck.variable_check_match(seq))
            return any(match)

    def check_match(self, seq):
        """ Check explicit content match"""
        self.matching_seq = {}  # free up memory
        # key: chunk content, value: matching points
        D = self.D

        def dist(m, pt):
            nD = len(m)
            return sum(map(lambda i: (pt[i] - m[i]) ** 2, range(nD)))

        def point_approx_seq(m, seq):  # sequence is ordered in time
            for pt in seq:
                if dist(m, pt) <= D:
                    if m in self.matching_seq.keys():
                        self.matching_seq[m].append(pt)
                    else:
                        self.matching_seq[m] = [pt]
                    return True
            return False

        n_match = 0
        for obs in list(
            self.content
        ):  # find the number of observations that are close to the point
            if point_approx_seq(
                obs, seq
            ):  # there exists something that is close to this observation in this sequence:
                n_match = n_match + 1

        if n_match / len(self.content) > self.matching_threshold:
            return True  # 80% correct
        else:
            return False

    def check_adjacency(self, cR):
        """Check if two chunks overlap/adjacent in their content and location"""
        cLidx = self.indexloc
        cRidx = cR.get_index_padded()
        intersect_location = cLidx.intersection(cRidx)
        if (
            len(intersect_location) > 0
        ):  # as far as the padded chunk and another is intersecting,
            return True
        else:
            return False

    def checksimilarity(self, chunk2):
        """returns the minimal moving distance from point cloud chunk1 to point cloud chunk2"""
        pointcloud1, pointcloud2 = self.content.copy(), chunk2.content.copy()
        lc1, lc2 = len(pointcloud1), len(pointcloud2)
        # smallercloud = [pointcloud1,pointcloud2][np.argmin([lc1,lc2])]
        # match by minimal distance
        match = []
        minD = 0
        for x1 in pointcloud1:
            mindist = 1000
            minmatch = None
            # search for the matching point with the minimal distance
            if len(match) == min(lc1, lc2):
                break
            for x2 in pointcloud2:
                D = self.pointdistance(x1, x2)
                if D < mindist:
                    minmatch = (x1, x2)
                    mindist = D
            match.append(minmatch)
            minD = minD + mindist
            pointcloud2.pop(minmatch[1])
        return minD

    def pointdistance(self, x1, x2):
        """ calculate the the distance between two points """
        nD = len(x1)
        D = 0
        for i in range(0, nD):
            D = D + (x1[i] - x2[i]) * (x1[i] - x2[i])
        return D

    def update_transition(self, chunkidx, dt):
        if dt in list(self.adjacency.keys()):
            if chunkidx in list(self.adjacency[dt].keys()):
                self.adjacency[dt][chunkidx] = self.adjacency[dt][chunkidx] + 1
            else:
                self.adjacency[dt][chunkidx] = 1
        else:
            self.adjacency[dt] = {}
            self.adjacency[dt][chunkidx] = 1
        return

    def empty_counts(self):
        self.count = 0
        self.birth = None  # chunk creation time
        # empty transitional counts
        for dt in list(self.adjacency.keys()):
            for chunkidx in list(self.adjacency[dt].keys()):
                self.adjacency[dt][chunkidx] = 0
        return

    def contentagreement(self, content):
        if len(self.content) != len(content):
            return False
        else:  # sizes are the same
            return len(self.content.intersection(content)) == len(content)

    def get_transition(self, cridx):
        """ number of occurrence transitioning from chunk to cridx """
        if self.adjacency == {}:
            return 0
        else:
            nt = 0
            for dt in list(self.adjacency.keys()):
                if cridx in list(self.adjacency[dt].keys()):
                    nt = nt + self.adjacency[dt][cridx]
            return nt
