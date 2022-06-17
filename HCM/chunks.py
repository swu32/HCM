import random
import string
import numpy as np

class Chunk:
    """ Spatial Temporal Chunk
        At the moment, it lacks a unique identifier for which of the chunk is which, making the searching process
        diffidult, ideally, each chunk should have its own unique name, (ideally related to how it looks like) """

    # A code name unique to each chunk
    def __init__(self, chunkcontent, variable = [], count = 1, H = None,W = None, pad=1):
        """chunkcontent: a list of tuples describing the location and the value of observation"""
        self.content = set(chunkcontent)
        self.variable = variable # a list of other chunks
        self.T = int(max(np.array(chunkcontent)[:, 0])+1) # those should be specified when joining a chunking graph
        self.H = H
        self.W = W
        self.index = None
        self.count = count #
        self.pad = pad # boundary size for nonadjacency detection, set the pad to not 1 to enable this feature.
        self.adjacency = {}
        self.birth = None # chunk creation time
        self.volume = len(self.content) #
        self.indexloc = self.get_index()
        self.arraycontent = None
        self.boundarycontent = set()
        T,H,W,cRidx = self.get_index_padded()
        self.D = 10
        self.matching_threshold = 0.8
        self.matching_seq = {}
        self.abstraction = [] # what are the variables summarizing this chunk
        self.entailment = [] # concrete chunks that the variable is pointing to

        # discount coefficient when computing similarity between two chunks, relative to the temporal discount being 1
        self.h = 1.
        self.w = 1.
        self.v = 1.


    def get_full_content(self):
        '''returns a list of all possible content that this chunk can take'''
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
        if len(self.variable)>0:
            self.update_variable_count() # update the count of the subordinate chunks
        return


    def to_array(self):
        '''convert the content into array'''
        arrep = np.zeros((int(max(np.atleast_2d(np.array(list(self.content)))[:, 0])+1),self.H,self.W))
        for t,i,j,v in self.content:
            arrep[t,i,j] = v
        self.arraycontent = arrep
        return

    def get_N_transition(self, dt):
        assert dt in list(self.adjacency.keys())
        N = 0
        for item in self.adjacency[dt]:
            N = N + self.adjacency[dt][item]
        return N

    def get_index(self):
        ''' Get index location the nonzero chunk locations in chunk content  '''
        return set(map(tuple, np.array(list(self.content))[:, 0:3]))

    def get_index_padded(self):
        ''' Get padded index arund the nonzero chunk locations '''
        padded_index = self.indexloc.copy()
        chunkcontent = self.content
        self.boundarycontent = set()
        T,H,W = self.T,self.H, self.W
        for t, i, j, v in chunkcontent:
            point_pad = {(t + 1, i, j), (t - 1, i, j), (t, min(i + 1, H), j), (t, max(i - 1, 0), j),
                                   (t, i, min(j + 1, W)), (t, i, max(j - 1, 0))}

            if point_pad.issubset(self.indexloc) == False: # the current content is a boundary element
                self.boundarycontent.add((t,i,j,v))
            padded_index = padded_index.union(point_pad)

        if self.pad > 1: # pad extra layers around the chunk observations
            # there is max height, and max width, but there is no max time.
            for p in range(2, self.pad+1):
                for t, i, j, v in chunkcontent:
                    padded_boundary_set = {(t + p, i, j), (t - p, i, j), (t, min(i + p, H), j),
                                           (t, max(i - p, 0), j), (t, i, min(j + p, W)), (t, i, max(j - p, 0))}
                    padded_index = padded_index.union(padded_boundary_set)
        return T, H, W, padded_index

    def conflict(self, c_):
        return False

    def concatinate(self, cR):
        if self.check_adjacency(cR):
            clcrcontent = self.content | cR.content
            clcr = Chunk(list(clcrcontent), H=self.H, W=self.W, pad = cR.pad)
            return clcr
        else:
            return None

    def average_content(self):
        # average the stored content with the sequence
        # calculate average deviation
        averaged_content = set()
        assert(len(self.matching_seq) > 0)
        for m in list(self.matching_seq.keys()): # iterate through content points
            thispt = list(m)
            n_pt = len(self.matching_seq[m])
            otherpt0 = 0
            otherpt1 = 0
            otherpt2 = 0
            otherpt3 = 0
            for pt in self.matching_seq[m]:
                otherpt0 += pt[0]-thispt[0]
                otherpt1 += pt[1]-thispt[1]
                otherpt2 += pt[2]-thispt[2]
                otherpt3 += pt[3]-thispt[3]
            count = max(1, self.count)
            thispt[0] = int(thispt[0]+1/count * otherpt0/n_pt)
            thispt[1] = int(thispt[1]+1/count * otherpt1/n_pt)
            thispt[2] = int(thispt[2]+1/count * otherpt2/n_pt)
            thispt[3] = int(thispt[3]+1/count * otherpt3/n_pt)
            if np.any(thispt) < 0:
                print("")
            averaged_content.add(tuple(thispt))

        self.content = averaged_content
        self.T = int(np.atleast_2d(np.array(list(self.content)))[:, 0].max()+1) # those should be specified when joining a chunking graph
        self.get_index()
        self.get_index_padded() # update boundary content
        return

    def variable_check_match(self, seq): # a sequence matches any of its instantiated variables
        '''returns true if the sequence matches any of the variable instantiaions'''
        if len(self.variable) == 0:
            return self.check_match(seq)
        else:
            match = []
            for ck in self.variable:
                match.append(ck.variable_check_match(seq))
            return any(match)

    def check_match(self, seq):
        ''' Check explicit content match'''
        self.matching_seq = {} # free up memory
        # key: chunk content, value: matching points
        D = self.D
        def dist(m,pt):
            return (pt[0] - m[0])**2 + (pt[1] - m[1])**2 + (pt[2] - m[2])**2 + (pt[3] - m[3])**2

        def point_approx_seq(m, seq):# sequence is ordered in time
            for pt in seq:
                if dist(m, pt)<= D:
                    if m in self.matching_seq.keys():
                        self.matching_seq[m].append(pt)
                    else:
                        self.matching_seq[m] = [pt]
                    return True
            return False

        n_match = 0
        for obs in list(self.content): # find the number of observations that are close to the point
            if point_approx_seq(obs, seq): # there exists something that is close to this observation in this sequence:
                n_match = n_match + 1

        if n_match/len(self.content)>self.matching_threshold:
            return True # 80% correct
        else:
            return False

    def check_adjacency(self, cR):
        """Check if two chunks overlap/adjacent in their content and location"""
        cLidx = self.indexloc
        T,H,W,cRidx = cR.get_index_padded()
        intersect_location = cLidx.intersection(cRidx)
        if len(intersect_location) > 0: # as far as the padded chunk and another is intersecting,
            return True
        else:
            return False

    def checksimilarity(self, chunk2):
        '''returns the minimal moving distance from point cloud chunk1 to point cloud chunk2'''
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

    def pointdistance(self,x1,x2):
        ''' calculate the the distance between two points '''
        D = (x1[0]-x2[0])*(x1[0]-x2[0]) + self.h*(x1[1]-x2[1])*(x1[1]-x2[1]) + self.w*(x1[2]-x2[2])*(x1[2]-x2[2]) + self.v*(x1[0]-x2[0])*(x1[0]-x2[0])
        return D

    def update_transition(self, chunkidx, dt): #_c_
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
        # empty
        self.count = 0
        self.birth = None # chunk creation time
        # empty transitional counts
        for dt in list(self.adjacency.keys()):
            for chunkidx in list(self.adjacency[dt].keys()):
                self.adjacency[dt][chunkidx] = 0
        return


    def contentagreement(self, content):
        if len(self.content) != len(content):
            return False
        else: # sizes are the same
            return len(self.content.intersection(content)) == len(content)



