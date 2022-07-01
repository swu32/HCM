class buffer():
    """Buffer class to load the sequence in parts."""
    def __init__(self,t, seq, seql, arrayl, reloadsize = 20):
        self.t = t
        self.seq = seq
        self.seql = seql
        self.reloadsize = reloadsize
        self.arrayl = arrayl

    def __len__(self):
        return self.seql

    def print(self):
        print('t  = ', self.t)
        print('seq = ', self.seq)
        print('seql = ', self.seql)
        if self.seql<0:
            print('...')

    def refactor(self, seq, dt):
        # first, remove the identified current chunks from the sequence of observations
        seqcopy = []
        if seq != []:
            mintime = dt #seq[0][0]
            for item in seq:
                listobs = list(item)
                listobs[0] = int(listobs[0] - mintime) # this value can be negative, which means there are unexplained sequence before.
                if listobs[0]<0:
                    print('')
                seqcopy.append(tuple(listobs))  # there should not be negative ts, otherwise something is not explained properly
            self.seq = seqcopy
        else:
            self.seq = []
        self.t = self.t + dt # the current time
        self.seql = self.seql - dt # the current sequence length in relevance to the current time.
        return self.seq

    def checkreload(self, arayseq):
        if self.seql < self.reloadsize: # parsing queue is too small
            self.reload(arayseq)

    def checkseqover(self):
        seq_over = False
        if self.t > self.arrayl:
            seq_over = True
            return seq_over
        return seq_over

    def reload(self, arraysequence, Print = False):
        seq, seql, t = self.seq, self.seql, self.t
        max_chunksize = self.reloadsize
        # reload arraysequence starting from time point t to the set sequence representations
        # t: the current time point
        time = t + self.seql
        if Print:
            print('time ', time, ' max chunksize ', max_chunksize)
        relevantarray = arraysequence[time:time + max_chunksize, :, :]
        _, H, W = arraysequence.shape
        if len(seq)>2:
            last = seq[-1]
        else:
            last = (0,0)
        for tt in range(0, min(max_chunksize, relevantarray.shape[0])):
            for h in range(0, H):
                for w in range(0, W):
                    v = relevantarray[tt, h, w]
                    if v>0:# none zero observations goes inside the sequence.
                        if self.seql+tt < last[0]:
                            print('...')
                        seq.append((self.seql + tt, h, w, v))
        seql = seql + min(max_chunksize, relevantarray.shape[0])
        self.seq, self.seql = seq, seql

    def update(self, dt):
        self.t = self.t + dt
        self.seql = self.seql - dt
        return
