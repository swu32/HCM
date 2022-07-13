from scipy import stats

from Chunking_Graph import *
from buffer import *
from chunks import *


def learning_and_update_mtx(current_chunks, t, previous_chunk_boundary_record, chunk_termination_time, cg):
    """
    Update transitions and marginals and decide to chunk for matrix representations

    Parameters
    ----------
    t: the time point where the current chunks have ended.
    current_chunks: the chunks ending at the current time point
    previous_chunk_boundary_record: boundary record of when the previous chunks have ended.
    chunk_termination_time

    Returns
    -------
    chunk_termination_time
    cg: an updated chunking graph
    """


    for chunk in current_chunks:
        chunk_termination_time[
            np.array(chunk)[-1, :, :] > 0] = t  # update chunk termination time for the current chunks
        temporal_length_of_current_chunk = tuple_to_arr(chunk).shape[0]
        # current_chunk_starting_time = t - temporal_length_of_current_chunk + 1  # the "current chunk" starts at:
        for i in range(0, min(temporal_length_of_current_chunk + 1,
                              len(previous_chunk_boundary_record))):  # iterate through adjacency in
            candidate_adjacent_chunks = previous_chunk_boundary_record[-i - 1]  # start from the closest
            # to the furtherest.
            for candidate_chunk in candidate_adjacent_chunks:
                combined_chunk, dt = check_adjacency_mtx(candidate_chunk, chunk, t - i + 1, t + 1)
                if np.sum(np.abs(combined_chunk)) > 0:  # if the two chunks are adjacent to each other
                    cg.update_transition_between_chunks(candidate_chunk, chunk, dt)
                    combined_chunk = arr_to_tuple(combined_chunk)
                    cg, chunked = threshold_chunking_mtx(candidate_chunk, chunk, combined_chunk, dt, cg)

    return chunk_termination_time, cg


def learning_and_update(current_chunks_idx, chunk_record, cg, t, learn = True):
    '''
    Update transitions and marginals and decide to chunk

            Parameters:
                    current_chunks_idx (int): the chunks ending at the current time point
                    t (int): finishing parsing at time t
                    chunk_record (dict): boundary record of when the previous chunks have ended.
            Returns:
                    cg (CG1): Chunking Graph
    '''
    n_t = cg.pad
    if len(chunk_record) > 1:
        for chunk_idx in current_chunks_idx:
            delta_t = 0  # the difference between the end of the current chunk and the end of the previous chunks
            temporal_length_chunk = cg.chunks[chunk_idx].T
            while delta_t <= temporal_length_chunk + n_t and len(
                    chunk_record) > 1:  # looking backward to find, padding length
                # adjacent chunks
                chunkendtime = t - delta_t
                if chunkendtime in list(chunk_record.keys()):
                    previous_chunks = chunk_record[chunkendtime]
                    # current_chunk_starting_time = t - temporal_length_of_current_chunk + 1  # the "current chunk" starts at:
                    for prev_idx, _ in previous_chunks:
                        samechunk = prev_idx == chunk_idx and delta_t == 0
                        if not samechunk and cg.chunks[prev_idx].variable == []:  # do not check variable chunks
                            combined_chunk, dt = check_adjacency(prev_idx, chunk_idx, delta_t, t, cg)
                            if combined_chunk is not None:
                                cg.chunks[prev_idx].update_transition(chunk_idx, dt)
                                if learn == True:  # online learning
                                    cg, chunked = threshold_chunking(prev_idx, chunk_idx, combined_chunk, dt, cg)
                delta_t = delta_t + 1
    return cg


def rational_update(current_chunks_idx, chunk_record, cg, t):
    """
    check chunk adjacency in cg, and find the most urgent chunk to combine together

    Parameters
    ----------
    current_chunks_idx
    chunk_record
    cg
    t

    Returns
    -------

    """
    n_t = cg.pad
    for chunk_idx in cg.chunks:
        delta_t = 0  # the difference between the end of the current chunk and the end of the previous chunks
        temporal_length_chunk = cg.chunks[chunk_idx].T
        while delta_t <= temporal_length_chunk + n_t and len(chunk_record) > 1:
            chunkendtime = t - delta_t
            if chunkendtime in list(chunk_record.keys()):
                previous_chunks = chunk_record[chunkendtime]
                for prev_idx, _ in previous_chunks:
                    samechunk = prev_idx == chunk_idx and delta_t == 0
                    if not samechunk and cg.chunks[prev_idx].variable == []:  # do not check variable chunks
                        combined_chunk, dt = check_adjacency(prev_idx, chunk_idx, delta_t, t, cg)
                        if combined_chunk is not None:
                            cg.chunks[prev_idx].update_transition(chunk_idx, dt)
            delta_t = delta_t + 1
    return cg


def rational_learning(cg, n_update=10):
    """ given a learned representation, update chunks based on rank of joint occurrence frequency and hypothesis tests
            Parameters:
                n_update: the number of concatinations made based on the pre-existing cg records
            Returns:
                cg: chunking graph with empty chunks
    """
    candidancy_pairs = []  # iterate through all chunk pairs and find the chunks with the biggest candidancy
    for _prevck in cg.chunks:
        _previdx = _prevck.index
        for _dt in list(_prevck.adjacency.keys()):
            for _postidx in list(_prevck.adjacency[_dt].keys()):
                _postck = cg.chunks[_postidx]
                _cat = combinechunks(_previdx, _postidx, _dt, cg)
                # hypothesis test
                if cg.hypothesis_test(_previdx, _postidx, _dt) == False:  # reject null hypothesis
                    candidancy_pairs.append([(_previdx, _postidx, _cat, _dt), _prevck.adjacency[_dt][_postidx]])

    candidancy_pairs.sort(key=lambda tup: tup[1], reverse=True)
    print(candidancy_pairs)

    # number of chunk combinations allowed.
    for i in range(0, n_update):
        prev_idx, current_idx, cat, dt = candidancy_pairs[i][0]
        cg.chunking_reorganization(prev_idx, current_idx, cat, dt)

        if i > len(candidancy_pairs):
            break
    return cg


def threshold_chunking_mtx(prev, current, combined_chunk, dt, cg):
    """

    Parameters
    ----------
    prev
    current
    combined_chunk
    dt
    cg

    Returns
    -------
    cg: chunking graph
    chunked:

    """
    def hypothesis_test(cl, cr, Marg, T, dt, zero):
        M = Marg.copy()
        M.pop(zero)  # delete zero since it is not useful anyways
        N = np.sum(list(M.values()))  # still applicable to spatial and spatial temporal chunks?
        # Expected
        ep1p1 = M[cl] / N * M[cr]
        ep1p0 = M[cl] / N * (N - M[cr])
        ep0p1 = (N - M[cl]) / N * M[cr]
        ep0p0 = (N - M[cl]) / N * (N - M[cr])
        # Observed
        op1p1 = T[cl][dt][cr]
        op1p0 = np.sum(list(T[cl][dt].values())) - T[cl][dt][cr]
        op0p1 = 0
        op0p0 = 0

        for ncl in list(T.keys()):  # iterate over p0, which is the cases where cl is not observed
            if ncl != cl:
                if dt in list(T[ncl].keys()):
                    if cr in list(T[ncl][dt].keys()):
                        op0p1 = op0p1 + T[ncl][dt][cr]
                        for ncr in list(T[ncl][dt].keys()):
                            if ncr != cr:
                                op0p0 = op0p0 + T[ncl][dt][ncr]
        # if any cell contains less than 5 observations, need to
        if ep0p0 <= 3 or ep1p1 <= 3 or ep1p0 <= 3 or ep0p1 <= 3:
            print('not enough observations')
            return True  # cannot continute the test because of lack of sample size

        _, pvalue = stats.chisquare([op1p1, op1p0, op0p1, op0p0], f_exp=[ep1p1, ep1p0, ep0p1, ep0p0], ddof=1)
        if pvalue == 0.0:
            print('check')
        if pvalue < 0.05:
            return False  # reject independence hypothesis, there is a correlation
        else:
            return True

    """cg: chunking graph
    learning function manipulates when do the chunk update happen
    chunks when some combination goes beyond a threshold"""
    chunked = False  # unless later it was verified to fit into the chunking criteria.
    cat = combined_chunk
    satisfy_criteria = False  # the criteria for chunking
    chunk_f = cg.M
    chunk_pair_f = cg.T
    N = 10  # one needs three observations of one item in addition to three observations from one item to another item
    # to form a chunk
    if prev in list(chunk_f.keys()):
        try:
            if str(dt) in list(chunk_pair_f[prev].keys()):
                if current in list(chunk_pair_f[prev][str(dt)].keys()):
                    if chunk_f[prev] > N:
                        if chunk_pair_f[prev][str(dt)][current] > N:
                            if hypothesis_test(prev, current, cg.M, cg.T, str(dt), cg.zero) == False:
                                satisfy_criteria = True

        except(KeyError):
            print('check what is going on here')
            print('dt ', str(dt))
            print('chunk_pair_f ', chunk_pair_f)
            print('chunk_f ', chunk_f)
            print('current ', current, ' prev ', prev)

    if satisfy_criteria:
        # decide if the number of exposure is significant enough to be concatinated together as a chunk
        chunked = True
        cg.chunking_reorganization(prev, current, cat, dt)
    return cg, chunked


def threshold_chunking(prev_idx, current_idx, combined_chunk, dt, cg):
    """

    Parameters
    ----------
    prev_idx
    current_idx
    combined_chunk
    dt
    cg

    Returns
    -------
    cg:
    chunked:
    """

    def hypothesis_test(clidx, cridx, cg, dt):
        cl = cg.chunks[clidx]
        cr = cg.chunks[cridx]
        assert len(cl.adjacency) > 0
        assert dt in list(cl.adjacency.keys())
        n = cg.get_N()

        # Expected
        ep1p1 = cl.count / n * cr.count
        ep1p0 = cl.count / n * (n - cr.count)
        ep0p1 = (n - cl.count) / n * cr.count
        ep0p0 = (n - cl.count) / n * (n - cr.count)

        # Observed
        op1p1 = cl.adjacency[dt][cridx]
        op1p0 = cl.get_N_transition(dt) - cl.adjacency[dt][cridx]
        op0p1 = 0
        op0p0 = 0
        for ncl in list(cg.chunks):  # iterate over p0, which is the cases where cl is not observed
            if ncl != cl:
                if dt in list(ncl.adjacency.keys()):
                    if cridx in list(ncl.adjacency[dt].keys()):
                        op0p1 = op0p1 + ncl.adjacency[dt][cridx]
                        for ncridx in list(ncl.adjacency[dt].keys()):
                            if ncridx != cr:
                                op0p0 = op0p0 + ncl.adjacency[dt][ncridx]

        _, pvalue = stats.chisquare([op1p1, op1p0, op0p1, op0p0], f_exp=[ep1p1, ep1p0, ep0p1, ep0p0], ddof=1)
        if pvalue < 0.05:
            return False  # reject independence hypothesis, there is a correlation
        else:
            return True

    """cg: chunking graph
    learning function manipulates when do the chunk update happen
    chunks when some combination goes beyond a threshold"""
    chunked = False  # unless later it was verified to fit into the chunking criteria.
    cat = combined_chunk
    chunk_f = cg.chunks
    N = 3
    # the prevchunkindex needs to be in the list of all of the chunks
    prev = cg.chunks[prev_idx]
    if dt in list(prev.adjacency.keys()):
        if current_idx in list(prev.adjacency[dt].keys()):
            if prev.count > N:
                if prev.adjacency[dt][current_idx] > N:
                    if hypothesis_test(prev_idx, current_idx, cg, dt) == False:  # reject null hypothesis
                        chunked = True
                        cg.chunking_reorganization(prev_idx, current_idx, cat, dt)

    return cg, chunked


import itertools


def check_overlap(array1, array2):
    output = np.empty((0, array1.shape[1]))
    for i0, i1 in itertools.product(np.arange(array1.shape[0]),
                                    np.arange(array2.shape[0])):
        if np.all(np.isclose(array1[i0], array2[i1])):
            output = np.concatenate((output, [array2[i1]]), axis=0)
    return output


def checkequal(chunk1, chunk2):
    if len(chunk1.content.intersection(chunk2.content)) == max(len(chunk1.content), len(chunk2.content)):
        return True
    else:
        return False


def check_adjacency(prev_idx, post_idx, time_diff, t, cg):
    """
    returns empty matrix if not chunkable
    Parameters
    ----------
    prev_idx
    post_idx
    time_diff: difference between end of the post chunk and the end of the previous chunk
    t
    cg

    Returns
    -------

    """
    # check adjacency should only check chunks with no variables, in other words, concrete chunks,
    # and their ancestors are tagged with this variable relationship
    # update transitions between chunks with a temporal proximity
    # chunk ends at the point of the end_point_chunk
    # candidate chunk ends at the point of the end_point_candidate_chunk
    if prev_idx == post_idx:
        return None, -100  # do not chunk a chunk by itself.
    elif cg.chunks[prev_idx].conflict(cg.chunks[post_idx]):  # chunks have conflicting content
        return None, -100
    else:
        prev = cg.chunks[prev_idx]
        post = cg.chunks[post_idx]
        e_post = t  # inclusive
        e_prev = t - time_diff  # inclusive
        s_prev = e_prev - prev.T  # inclusive
        s_post = e_post - post.T  # inclusive
        dt = e_prev - s_post
        delta_t = e_prev - max(s_post, s_prev)  # the overlapping temporal length between the two chunks
        # initiate a new chunk.
        prevcontent = prev.content.copy()
        postcontent = post.content.copy()

        if s_prev > s_post:
            # shift content in time:
            prev_shift = s_prev - s_post
            newprevcontent = set()
            for msrm in prevcontent:
                lmsrm = list(msrm)
                lmsrm[0] = lmsrm[0] + prev_shift  # pad the time dimension
                newprevcontent.add(tuple(lmsrm))
            prevcontent = newprevcontent

        prevchunk = Chunk(list(prevcontent), H=prev.H, W=prev.W, pad=cg.pad)
        prevchunk.T = prev.T
        prevchunk.H = prev.H
        prevchunk.W = prev.W

        if s_prev < s_post:
            post_shift = s_post - s_prev
            newpostcontent = set()
            for msrm in postcontent:
                lmsrm = list(msrm)
                lmsrm[0] = lmsrm[0] + post_shift
                newpostcontent.add(tuple(lmsrm))
            postcontent = newpostcontent

        postchunk = Chunk(list(postcontent), H=post.H, W=post.W, pad=cg.pad)
        postchunk.T = post.T
        postchunk.H = post.H
        postchunk.W = post.W
        concat_chunk = prevchunk.concatinate(postchunk)
    return concat_chunk, dt


def combinechunks(prev_idx, post_idx, dt, cg):
    """ update transitions between chunks with a temporal proximity
     chunk ends at the point of the end_point_chunk
     candidate chunk ends at the point of the end_point_candidate_chunk
     time_diff: difference between end of the post chunk and the end of the previous chunk"""
    prev = cg.chunks[prev_idx]
    post = cg.chunks[post_idx]
    assert post_idx in prev.adjacency[dt].keys()
    e_prev = 0
    l_t_prev = prev.T
    l_t_post = post.T
    s_prev = e_prev - l_t_prev  # the exclusive temporal length
    s_post = e_prev - dt  # start point is inclusive
    e_post = s_post + l_t_post
    # dt = e_prev - s_post
    delta_t = e_prev - max(s_post, s_prev)  # the overlapping temporal length between the two chunks
    t_chunk = max(e_post, e_prev) - min(e_post, e_prev) + delta_t + max(s_post, s_prev) - min(s_post,
                                                                                              s_prev)  # the stretching temporal length of the two chunks
    if t_chunk == l_t_prev and t_chunk == l_t_post and checkequal(prev, post):
        return None  # do not chunk a chunk by itself.
    else:
        # initiate a new chunk.
        prevcontent = prev.content.copy()
        postcontent = post.content.copy()
        if s_prev > s_post:  # post start first
            # shift content in time:
            prev_shift = s_prev - s_post
            newprevcontent = set()
            for msrm in prevcontent:
                lmsrm = list(msrm)
                lmsrm[0] = lmsrm[0] + prev_shift  # pad the time dimension
                newprevcontent.add(tuple(lmsrm))
        prevchunk = Chunk(list(prevcontent), H=prev.H, W=prev.W, pad=cg.pad)
        prevchunk.T = prev.T
        prevchunk.H = prev.H
        prevchunk.W = prev.W

        if s_prev < s_post:  # prev start first
            post_shift = s_post - s_prev
            newpostcontent = set()
            for msrm in postcontent:
                lmsrm = list(msrm)
                lmsrm[0] = lmsrm[0] + post_shift
                newpostcontent.add(tuple(lmsrm))
            postcontent = newpostcontent
        postchunk = Chunk(list(postcontent), H=post.H, W=post.W, pad=cg.pad)
        postchunk.T = post.T
        postchunk.H = post.H
        postchunk.W = post.W
    return prevchunk.concatinate(postchunk)


def check_adjacency_mtx(prev, post, end_point_prev, end_point_post):
    """returns empty matrix if not chunkable"""
    # prev and post denotes ending time, but based on how long the chunks are, prev can end after post.
    # update transitions between chunks with a temporal proximity
    # chunk ends at the point of the end_point_chunk
    # candidate chunk ends at the point of the end_point_candidate_chunk
    temporal_len_prev = tuple_to_arr(prev).shape[0]
    temporal_len_post = tuple_to_arr(post).shape[0]
    start_point_prev = end_point_prev - temporal_len_prev  # the exclusive temporal length
    start_point_post = end_point_post - temporal_len_post  # start point is inclusive
    # dt = start_point_post - start_point_prev
    dt = end_point_prev - start_point_post
    T, W, H = tuple_to_arr(prev).shape[0:]

    # the overlapping temporal length between the two chunks
    delta_t = end_point_prev - max(start_point_post, start_point_prev)
    # the stretching temporal length of the two chunks
    t_chunk = max(end_point_post, end_point_prev) - min(end_point_post,
                                                        end_point_prev) + delta_t + max(
        start_point_post, start_point_prev) - min(start_point_post, start_point_prev)

    if t_chunk == temporal_len_prev and t_chunk == temporal_len_post and prev == post:  # do not chunk a chunk by itself.
        concat_chunk = np.zeros([t_chunk, W, H])
        return concat_chunk, dt
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
        for prev_t, prev_i, prev_j in zip(prev_index_t, prev_index_i,
                                          prev_index_j):
            indices = [(min(prev_t + 1, T), prev_i, prev_j),
                       (max(prev_t - 1, 0), prev_i, prev_j),
                       (prev_t, min(prev_i + 1, W), prev_j),
                       (prev_t, max(prev_i - 1, 0), prev_j),
                       (prev_t, prev_i, min(prev_j + 1, H)),
                       (prev_t, prev_i, max(prev_j - 1, 0)), ]
            prev_indexes = prev_indexes + indices
        for post_t, post_i, post_j in zip(post_index_t, post_index_i,
                                          post_index_j):
            indices = [(min(post_t + 1, T), post_i, post_j),
                       (max(post_t - 1, 0), post_i, post_j),
                       (post_t, min(post_i + 1, W), post_j),
                       (post_t, max(post_i - 1, 0), post_j),
                       (post_t, post_i, min(post_j + 1, H)),
                       (post_t, post_i, max(post_j - 1, 0)), ]
            post_indexes = post_indexes + indices
        # print(np.array(prev_indexes))
        overlap = check_overlap(np.array(prev_indexes), np.array(post_indexes))
        if overlap.shape[0] > 0:
            adjacent = True

        if adjacent == True:
            # how to check if two chunks are spatial temporally adjacent to each other.
            # observation_to_explain[best_matching_subchunk[0:l,:,:]>0] = observation_to_explain[best_matching_subchunk[0:l,:,:]>0] - best_matching_subchunk[0:l,:,:][best_matching_subchunk[0:l,:,:]>0]
            concat_chunk = np.zeros([t_chunk, W, H])
            concat_chunk[concat_chunk_prev > 0] = concat_chunk_prev[
                concat_chunk_prev > 0]
            concat_chunk[
                concat_chunk_prev == concat_chunk_post] = 0  # remove overlapping components
            concat_chunk[concat_chunk_post > 0] = concat_chunk_post[
                concat_chunk_post > 0]
            return concat_chunk, dt
        else:
            concat_chunk = np.zeros([t_chunk, W,
                                     H])  # nonadjacent, then returns a high dimneisonal zero array
            return concat_chunk, dt


def add_singleton_chunk_to_M(seq, t, cg):
    """Add the singleton chunk observed at the temporal point t of the seq to the set of chunks M"""
    H, W = seq.shape[1:]
    current_chunks = set()  # the set of chunks that end right before this moment t.
    for i in range(0, H):
        for j in range(0, W):
            chunk = np.zeros([1, H, W])
            chunk[0, i, j] = seq[t, i, j]
            if np.sum(np.abs(chunk)) > 0:  # not an empty chunks
                matchingchunk = cg.add_chunk_to_cg_class(chunk)
                current_chunks.add(matchingchunk)  # the chunk that ends right at the moment t
    return cg, current_chunks


def check_chunk_exist(chunk, M):
    for ck in M:
        if ck.profile == chunk:
            return True
    return False


def add_chunk_to_M(chunk, M):
    if chunk in list(M.keys()):
        M[chunk] = M[chunk] + 1
    else:
        M[chunk] = 1
    return M


def possible_best_matching_chunks(cg, observation_to_explain):
    """use chunks in M to explain observations
        chunks do not have to end at observations, it could end before, or after observations
        returns:
            possible_matching_chunks: a list containing the matching chunks that could end before or after the
            observation_to_explain
            best_matching_subchunk: matching chunk that ends right at the end of or after observation to explain
                                    if a zero aarray, it means no such chunk is matching
            """
    best_matching_subchunk = np.zeros(observation_to_explain.shape)
    observation_len = observation_to_explain.shape[0]
    possible_matching_chunks = []
    best_match_size = 0
    if len(cg.get_nonzeroM()) > 0:
        for chunk in cg.get_nonzeroM():  # can also go from the biggest chunk to the smallest
            # this is checking whether and which chunk has started at the first observation of the termination time
            subchunk_match, match_size = check_subchunk_match_mtx(observation_to_explain, chunk, size=True)
            if match_size >= best_match_size:
                best_match_size = match_size
            if subchunk_match:  # indicator that the subchunk is a match
                possible_matching_chunks.append(
                    chunk)  # can be that chunk ends earier or later than the current time point
                if np.array(chunk).shape[0] >= observation_len and match_size >= best_match_size:
                    best_matching_subchunk = np.array(chunk)

    return possible_matching_chunks, best_matching_subchunk


def find_relevant_observations(seq, termination_time, t):
    H, W = seq.shape[1:]
    min_termin = min(termination_time.flatten())
    relevant_observations = np.zeros([int(t - min(termination_time.flatten())), H, W])
    for i in range(0, H):
        for j in range(0, W):
            relevant_t = (termination_time[i, j] - min_termin)
            relevant_observations[relevant_t:, i, j] = seq[termination_time[i, j] + 1:, i, j]
    relevant_observations, n_zero = refactor_observation(relevant_observations, return_max_k=True)  # cut such that
    # there is an event
    # happening
    # at the beginning.
    return relevant_observations, n_zero


# no chunks ends at this moment
def update_termination_time(current_chunks, chunk_termination_time, t):
    """For each spatial dimension, update the time that the latest chunk terminated
            :returns
                chunk_termination_time: shape: np.array(H,W), the height and width of the spatial dimension, in
                each dimension, the latest time that a chunk has ended. """
    for chunk in current_chunks:  # update chunk termination time for the current chunks
        chunk_termination_time[np.array(chunk)[-1, :, :] > 0] = t
    return chunk_termination_time


def identify_chunks(cg, seq, termination_time, t, previous_chunk_boundary_record):
    """Identify chunks to explain the current observation, and
    if the pre-existing chunk does not explain the current observation
    create new chunks
    # which one of them is predicted, which one of them is surprising.
    cg: chunking graph object
    seq: sequence of observation
    termination time: the time in each dimension where the last chunk has terminated.
    """
    "Starting from the location where the last chunk terminates, " \
    "needs explanation for what comes thereafter "
    relevant_observations, nzero = find_relevant_observations(seq, termination_time, t)
    # cut such that there is an event

    # happening at the beginning.
    if relevant_observations.shape[0] > 0:  # there are some observations to explain:
        # use the chunks to explain the current relevant observations.
        observation_to_explain = relevant_observations.copy()  # the sequence the same size as chunks:
        if len(list(cg.M.keys())) > 0:
            # check chunk match, start from the biggest chunk/the most likely
            # in the order of temporal slice, explain the observation one by one, until the entire observation is
            # cancelled
            # load things into current chunks
            current_chunks, termination_time, cg = identify_chunk_ending_with_M(observation_to_explain, cg,
                                                                                termination_time, t,
                                                                                previous_chunk_boundary_record)
            previous_chunk_boundary_record.append(current_chunks)
        else:  # nothing in the marginals, find the singleton chunks and load the marginals, relevant obsertvation size
            # = 1
            cg, current_chunks = add_singleton_chunk_to_M(seq, t, cg)
            delta_t = (tuple_to_arr(observation_to_explain).shape[0] - 1)
            t_end_best_matching_subchunk = t - delta_t
            termination_time = update_termination_time(current_chunks, termination_time, t_end_best_matching_subchunk)
            previous_chunk_boundary_record.append(current_chunks)
        # moved transition update to learning and update
        return current_chunks, cg, termination_time, previous_chunk_boundary_record
    else:
        return {}, cg, termination_time, previous_chunk_boundary_record


def identify_chunk_ending_with_M(observation_to_explain, cg, termination_time, t, previous_chunk_boundary_record):
    current_chunks = set()
    debug = 1

    while np.sum(np.abs(observation_to_explain)) > 0:
        debug = debug + 1
        possible_matching_chunks, best_matching_subchunk = possible_best_matching_chunks(cg,
                                                                                         observation_to_explain)
        if np.sum(np.abs(best_matching_subchunk)) <= 0:  # no currentmatching chunk
            # check for matching chunks at previous time points.
            if possible_matching_chunks != []:  # chunks end before the termination point
                max_chunk_size = 0
                best_matching_subchunk = None
                for chunk in possible_matching_chunks:
                    chunkvolume = np.sum(tuple_to_arr(chunk) > 0)
                    if chunkvolume >= max_chunk_size:
                        max_chunk_size = chunkvolume
                        best_matching_subchunk = chunk
                delta_t = (tuple_to_arr(observation_to_explain).shape[0] -
                           tuple_to_arr(best_matching_subchunk).shape[0])
                t_end_best_matching_subchunk = t - delta_t

                previous_chunk_boundary_record[t_end_best_matching_subchunk].add(best_matching_subchunk)

                cg.add_chunk_to_cg_class(best_matching_subchunk)
                # M = add_chunk_to_M(arr_to_tuple(best_matching_subchunk), M)
                termination_time, cg = learning_and_update_mtx({best_matching_subchunk}, t_end_best_matching_subchunk,
                                                               previous_chunk_boundary_record, termination_time, cg)
                termination_time = update_termination_time({best_matching_subchunk}, termination_time,
                                                           t_end_best_matching_subchunk)
            else:  # possible_matching_chunks == []:
                # no chunk is matching starting at the initilaization point.
                chunk = find_atomic_chunk(observation_to_explain)
                best_matching_subchunk = arr_to_tuple(chunk)
                delta_t = (tuple_to_arr(observation_to_explain).shape[0] -
                           tuple_to_arr(best_matching_subchunk).shape[0])
                t_end_best_matching_subchunk = t - delta_t
                termination_time = update_termination_time({best_matching_subchunk}, termination_time,
                                                           t_end_best_matching_subchunk)

        sz = min(len(observation_to_explain), len(best_matching_subchunk))  # problematic when the
        # best_matching_subchunk is smaller than the
        best_matching_subchunk = tuple_to_arr(best_matching_subchunk)
        observation_to_explain[0:sz, :, :][best_matching_subchunk[0:sz, :, :] > 0] = 0
        if check_chunk_ending(best_matching_subchunk, observation_to_explain):
            current_chunks.add(arr_to_tuple(best_matching_subchunk))  # best ending subchunk ends right at t
            cg.add_chunk_to_cg_class(arr_to_tuple(best_matching_subchunk))
            # M = add_chunk_to_M(arr_to_tuple(best_matching_subchunk), M)
        if debug >= 1000: exit()
        observation_to_explain, max_k = refactor_observation(observation_to_explain, return_max_k=True)
    return current_chunks, termination_time, cg


def pop_chunk_in_seq_full(chunk_idx, seqc, cg):  # _c_
    chunk = cg.chunks[chunk_idx]
    content = chunk.content
    for tup in content:  # remove chunk content one by one from the sequence
        seqc.remove(tup)
    return seqc


def check_chunk_in_seq(chunk, seq):  # _c_
    """ returns the size of match, how big the matching content is, and how big the chunk is, to be matched"""
    # returns the volume and the temporal length of the chunk content
    content = chunk.content
    if content.issubset(seq):
        return chunk.volume, chunk.T
    else:
        return 0, 0


def check_chunk_in_seq_boundary(chunk, seq):
    ''' Use chunk boundaries to identify chunks in the sequence
        Discrepencies of none-boundary contents are being ignored '''
    boundary = chunk.boundarycontent
    assert boundary is not None
    if boundary.issubset(seq):
        return chunk.volume, chunk.T
    else:
        return 0, 0


def check_chunk_in_seq_approximate(chunk, seq):  # more approximate
    if chunk.check_match(seq):
        return chunk.volume, chunk.T
    else:
        return 0, 0


def pop_chunk_in_seq_boundary(chunk_idx, seqc, cg):
    # pop everything within the boundary of this chunk in seqc
    chunk = cg.chunks[chunk_idx]
    content = list(chunk.boundarycontent)
    T = sorted(content, key=lambda tup: tup[0])
    X = sorted(content, key=lambda tup: tup[1])
    Y = sorted(content, key=lambda tup: tup[2])

    tmax, tmin = T[-1][0], T[0][0]
    xmax, xmin = X[-1][1], X[0][1]
    ymax, ymin = Y[-1][2], Y[0][2]
    seqcc = seqc.copy()
    for tup in seqc:  # remove chunk content one by one from the sequence
        if tup[0] >= tmin and tup[0] <= tmax and tup[1] >= xmin and tup[1] <= xmax and tup[2] >= ymin and tup[
            2] <= ymax:
            seqcc.remove(tup)
            if tup not in chunk.content:
                chunk.content.add(tup)  # add sequential content within a chunk
    return seqcc


def identify_biggest_chunk(cg, seqc, checktype='full'):
    '''Chunk with a bigger size is priorized to explain the sequence'''
    # check the occation when the seqc start at above 0, in which case it implies that
    # there are empty observations in certain time slices.
    if checktype == 'full':
        chunkidentification = check_chunk_in_seq
        pop_chunk_in_seq = pop_chunk_in_seq_full
    elif checktype == 'boundary':
        chunkidentification = check_chunk_in_seq_boundary
        pop_chunk_in_seq = pop_chunk_in_seq_boundary
    else:
        chunkidentification = check_chunk_in_seq
        pop_chunk_in_seq = pop_chunk_in_seq_full
    maxsizematch = 0
    maxchunk = None
    for chunk in reversed(cg.chunks):  # identify one chunk in sequence at one time...
        matchsize, match_len = chunkidentification(chunk, seqc)  # do the specific match by default
        if matchsize > maxsizematch:
            maxsizematch = matchsize
            maxchunk = chunk.index
    # remove chunk from seq
    if maxchunk is None:
        maxchunk, cg = identify_singleton_chunk(cg, seqc)
        seqc = pop_chunk_in_seq(maxchunk, seqc, cg)  # pop identified chunks in sequence
    else:
        seqc = pop_chunk_in_seq(maxchunk, seqc, cg)  # pop identified chunks in sequence

    return maxchunk, seqc  # strange, maxchunk is indeed an instance inside cg, a chunk inside cg


def identify_singleton_chunk(cg, seqc):
    chunkcontent = [seqc[0]]
    chunk = Chunk(chunkcontent, H=cg.H, W=cg.W, pad=cg.pad)
    cg.add_chunk(chunk)
    return chunk.index, cg


def updatechunk(chunk, explainchunk, chunk_record, cg, max_chunk_idx, t):
    if chunk.abstraction == []:
        explainchunk.append((chunk.index, int(chunk.T), list(cg.visible_chunk_list[chunk.index])))
        cg.chunks[chunk.index].update()  # update chunk count among the currently identified chunks
        chunk_record = updatechunkrecord(chunk_record, chunk.index, int(cg.chunks[chunk.index].T) + t, cg)
        return explainchunk, cg, chunk_record
    else:
        for ck in chunk.abstraction:
            updatechunk(ck, explainchunk, chunk_record, cg, max_chunk_idx, t)

    chunk = cg.chunks[max_chunk_idx]

    while chunk.abstraction != []:
        for ck in chunk.abstraction:
            explainchunk.append((ck.index, int(ck.T), list(cg.visible_chunk_list[ck.index])))
            chunk_record = updatechunkrecord(chunk_record, ck.index, int(cg.chunks[ck.index].T) + t, cg)
            cg.chunks[ck.index].update()  # update chunk count among the currently identified chunks
    # trace back to the abstraction chunks

    return


def identify_one_chunk(cg, seqc, explainchunk, chunk_record, t):  # _c_
    max_chunk_idx, seqc = identify_biggest_chunk(cg, seqc)  # identify and pop chunks in sequence
    explainchunk, cg, chunk_record = updatechunk(cg.chunks[max_chunk_idx], explainchunk, chunk_record, cg,
                                                 max_chunk_idx, t)
    # explainchunk.append((max_chunk_idx, int(cg.chunks[max_chunk_idx].T), list(cg.visible_chunk_list[max_chunk_idx])))
    # chunk_record = updatechunkrecord(chunk_record, max_chunk_idx, int(cg.chunks[max_chunk_idx].T) + t, cg)
    # cg.chunks[max_chunk_idx].update()  # update chunk count among the currently identified chunks
    return cg, seqc, chunk_record, explainchunk


def pop_chunk_in_seq_approximate(chunk_idx, seqc, cg):
    # pop everything within the boundary of this chunk in seqc
    chunk = cg.chunks[chunk_idx]
    print(' chunk content before ', chunk.content)
    print(' matching seq ', chunk.matching_seq)

    chunk.average_content()  # average content
    print(' chunk content after ', chunk.content)

    cg.visible_chunk_list[chunk_idx] = chunk.content
    seqcc = seqc.copy()
    for m in list(chunk.matching_seq.keys()):
        for pt in chunk.matching_seq[m]:
            # can be the case that the same point in sequence matches multiple keys
            try:
                seqcc.remove(pt)
            except(ValueError):
                print(' one point matches two chunk points ')

    return seqcc


def identify_one_chunk_approximate(cg, seqc, explainchunk, chunk_record, t):
    maxsizematch = 0
    maxchunk = None
    for chunk in reversed(cg.chunks):  # identify one chunk in sequence at one time...
        matchsize, match_len = check_chunk_in_seq_approximate(chunk, seqc)
        if matchsize > maxsizematch:
            maxsizematch = matchsize
            maxchunk = chunk.index

    # remove chunk from seq
    if maxchunk is None:
        maxchunk, cg = identify_singleton_chunk(cg, seqc)
        seqc = pop_chunk_in_seq_full(maxchunk, seqc, cg)
    else:
        seqc = pop_chunk_in_seq_approximate(maxchunk, seqc, cg)

    max_chunk_idx = maxchunk
    explainchunk.append((max_chunk_idx, int(cg.chunks[max_chunk_idx].T), list(cg.visible_chunk_list[max_chunk_idx])))
    chunk_record = updatechunkrecord(chunk_record, max_chunk_idx, int(cg.chunks[max_chunk_idx].T) + t, cg)
    cg.chunks[max_chunk_idx].update()  # update chunk count among the currently identified chunks
    return cg, seqc, chunk_record, explainchunk


def check_seq_explained(seqc):
    # check whether there is a t = 0 in the seqc:
    if seqc == []:
        return True
    else:
        return seqc[0][0] != 0  # finished explaining the current time point


def updatechunkrecord(chunk_record, ckidx, endtime, cg, freq=True):
    if freq == True:
        if endtime not in list(chunk_record.keys()):
            chunk_record[endtime] = [(ckidx, cg.chunks[ckidx].count)]
        else:
            if ckidx not in chunk_record[endtime]:  # one chunk can only be identified at one time point. when there are
                # multiple points that correspond to the same chunk, this chunk is identified as occurring once.
                chunk_record[endtime].append((ckidx, cg.chunks[ckidx].count))
    else:
        p = cg.chunks[ckidx].count / endtime
        if endtime not in list(chunk_record.keys()):
            chunk_record[endtime] = [(ckidx, p)]
        else:
            if ckidx not in chunk_record[endtime]:  # one chunk can only be identified at one time point. when there are
                # multiple points that correspond to the same chunk, this chunk is identified as occurring once.
                chunk_record[endtime].append((ckidx, p))
    return chunk_record


def identify_latest_chunks(cg, seq, chunk_record, t):
    '''parse and store chunks in the chunkrecord '''

    def check_obs(s):  # there are observations at the current time point
        if s == []:
            return True  # in the case of empty sequence
        else:
            return s[0][0] != 0  # nothing happens at the current time point
        # identify biggest chunks that finish at the earliest time point.

    seq_explained = False
    seqc = seq.copy()
    explainchunk = []
    # find candidate chunks explaining the upcoming sequence, out of these candidate chunks, find the ones that
    # terminates at the earliest time point.
    no_observation = check_obs(seqc)  # no observation in this particular time slice
    if no_observation:
        current_chunks_idx = []
        dt = int(1.0)
    else:
        while seq_explained == False:
            # find explanations for the upcoming sequence
            # record termination time of each biggest chunk used for explanation
            cg, seqc, chunk_record, explainchunk = identify_one_chunk(cg, seqc, explainchunk, chunk_record, t)
            # cg, seqc, chunk_record, explainchunk = identify_one_chunk_approximate(cg, seqc, explainchunk, chunk_record, t)
            seq_explained = check_seq_explained(seqc)
        # chunkrecord: {time: [chunkindex]} stores the finishing time (exclusive) of each chunk
        # explainchunk[(maxchunkindex, chunktimelength, chunkcontent)]
        # decide which are current chunks so that time is appropriately updated
        explainchunk.sort(key=lambda tup: tup[1])  # sort according to finishing time
        if len(seqc) >= 1:
            dt = min(min(explainchunk, key=lambda tup: tup[1])[1], seqc[0][0])
        else:
            dt = min(explainchunk, key=lambda tup: tup[1])[1]
        current_chunks_idx = [item[0] for item in explainchunk if
                              item[1] == dt]  # move at the pace of the smallest identified chunk
        seq = seqc
    return current_chunks_idx, cg, dt, seq, chunk_record


def refactor(seq, current_chunk_idx, cg):
    # first, remove the identified current chunks from the sequence of observations
    if len(current_chunk_idx) > 0:
        for idx, _ in current_chunk_idx:
            content = cg.chunks[idx].content
            for tup in content:  # remove chunk content one by one from the sequence
                seq.remove(tup)

    seqcopy = []
    if seq != []:
        mintime = seq[0][0]
        for item in seq:
            listobs = list(item)
            listobs[0] = int(listobs[0] - mintime)
            # if listobs[0]>=0:
            seqcopy.append(
                tuple(listobs))  # there should not be negative ts, otherwise something is not explained properly
        return seqcopy
    else:
        return []


def check_subchunk_match_mtx(sequence_snapshot, chunk, size=False):
    """ check whether the subchunk of chunk matches with the sequence snapshot
        :returns True if there is a chunk match, false otherwise, when size is specified, the size of chunk match
        size: returns the size of the chunk, which is the number of observations that the chunk explains
        and also whether the sequence snapshot agrees with the chunk"""
    # whether the chunk could explain a subpart of the sequence snapshot
    l_seq = sequence_snapshot.shape[0]
    l_chunk = tuple_to_arr(chunk).shape[0]
    sz = min(l_seq, l_chunk)
    chunk = np.array(chunk)[0:sz, :, :]  # chunk can be smaller than sequence
    # snapshot, should decide what to do then.
    seq_part = sequence_snapshot[0:sz, :, :]
    # align them together
    diff = seq_part - chunk
    if size:
        return np.sum(np.abs(diff[chunk > 0])) == 0, np.sum(np.abs(chunk) > 0)  # return also the explanation covered by
    # the chunk
    else:
        return np.sum(np.abs(diff[chunk > 0])) == 0  # location where the template matching matters
    # if returns 0, then there is a complete matching.


def find_atomic_chunk(observation_to_explain):
    """:returns the first nonzero atomic unit"""
    H, W = observation_to_explain.shape[1:]
    for i in range(0, H):
        for j in range(0, W):
            chunk = np.zeros([1, H, W])
            if observation_to_explain[0, i, j] > 0:
                chunk[0, i, j] = observation_to_explain[0, i, j]  # group them by
                return chunk
    return np.zeros([1, H, W])


def rational_chunking(prev, current, combined_chunk, cg):
    """Updates chunking based on whether the new representation improves the evaluated loss function
        Updates the loss function associated with the new chunk, because the update of other chunks does not affect
        the loss function value estimation at the moment"""
    w = 0.5  # tradeoff between accuracy and reaction time.
    cat = combined_chunk
    chunked = False
    P_s_giv_s_last = cg.get_transitional_p(prev, current)
    if P_s_giv_s_last is not None:
        Expected_reward_cat = P_s_giv_s_last * len(current) + (1 - P_s_giv_s_last) * (-1) * w * len(current)
        if Expected_reward_cat > 0:
            chunked = True
            cg.chunking_reorganization(prev, current, cat)
    return cg, chunked


def arr_to_tuple(arr):
    test_list = arr.tolist()
    res = tuple([tuple([tuple(ele) for ele in sub]) for sub in test_list])
    return res


def tuple_to_arr(tup):
    return np.array(tup)


def refactor_observation(observation, return_max_k=False):
    ''' observation: matrix to refactor
    max_k: the steps of reduction does the observation reduces.'''
    max_k = 0
    sum_obs = np.sum(np.abs(observation), axis=(1, 2))

    if observation.shape[0] > 1:
        for k in range(1, observation.shape[0]):
            # if np.sum(np.abs(observation[0:k, :, :])) == 0:
            if np.sum(sum_obs[0:k]) == 0:
                if k > max_k:
                    max_k = k
    observation = observation[max_k:, :, :]
    if return_max_k:
        return observation, max_k
    else:
        return observation


def check_chunk_ending(chunk, observation):
    '''Given that a chunk could explain the observation, check on whether it is ending at the last column of the observation'''
    if len(chunk) == len(observation):
        return True
    else:
        return False


def transition_into_probabilities(chunk_f, chunk_pair_f, prev, current):
    # decide if the number of exposure is significant enough to be concatinated together as a chunk
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


def convert_sequence(seq):
    Observations = []
    T, H, W = seq.shape
    for t in range(0, T):
        for h in range(0, H):
            for w in range(0, W):
                v = seq[t, h, w]
                if v != 0:
                    Observations.append((t, h, w, seq[t, h, w]))
    return Observations, T


def save_chunk_record(chunkrecord, cg):
    import pickle
    ''' save chunk record for HCM learned on behaviorial data '''
    df = {}
    df['time'] = []
    df['chunksize'] = []
    for time in list(chunkrecord.keys()):
        df['time'].append(int(time))
        ckidx = chunkrecord[time][0][0]
        df['chunksize'].append(cg.chunks[ckidx].volume)
    with open('HCM_time_chunksize.pkl', 'wb') as f:
        pickle.dump(df, f)
    return


def hcm_learning(arayseq, cg, learn=True):
    '''Sequence is an n-d array
        Note that sequence should not be too long, otherwise subtraction would take a while for the algorithm
        when the learning handle is false, then only parsing, and no chunking will take place '''
    seql, H, W = arayseq.shape
    cg.update_hw(H, W)
    seq_over = False
    chunk_record = {}  # chunk ending time, and which chunk is it.

    seq, seql = convert_sequence(arayseq[0:1, :, :])  # loaded with the 0th observation
    t = 0
    Buffer = buffer(t, seq, seql, arayseq.shape[0])
    # seql: length of the current parsing buffer
    maxchunksize = 0

    while seq_over == False:
        # identify latest ending chunks
        current_chunks_idx, cg, dt, seq, chunk_record = identify_latest_chunks(cg, seq, chunk_record,Buffer.t)  # chunks
        seq = Buffer.refactor(seq, dt)

        # that ends right before t. the last
        if len(current_chunks_idx) > 0 and learn == True:
            cg = learning_and_update(current_chunks_idx, chunk_record, cg, Buffer.t)
        # previous and current chunk
        if learn == True:
            cg.forget()
        Buffer.reloadsize = maxchunksize + 1
        Buffer.checkreload(arayseq)
        seq_over = Buffer.checkseqover()
    return cg, chunk_record


def hcm_rational(arayseq, cg, maxIter = 20):
    """
    returns chunking graph based on rational chunk learning

            Parameters:
                    arayseq(ndarray): Observational Sequences
                    cg (CG1): Chunking Graph

            Returns:
                    cg (CG1): Learned Representation from Data
    """
    seql, H, W = arayseq.shape
    cg.update_hw(H, W)
    seq_over = False
    seqO, seql = convert_sequence(arayseq[0:1, :, :])  # loaded with the 0th observation

    Iter = 0
    independence = False
    while independence == False and Iter <= maxIter:
        seq = seqO # reset the sequence
        cg.empty_counts()  # empty out all the counts before the next sequence parse
        chunk_record = {}  # chunk ending time, and which chunk is it.
        t = 0
        Buffer = buffer(t, seqO, seql, arayseq.shape[0])  # seql: length of the current parsing buffer
        maxchunksize = cg.getmaxchunksize()
        print('============ empty out sequence ========== ')
        while seq_over == False:
            # identify latest ending chunks
            current_chunks_idx, cg, dt, seq, chunk_record = identify_latest_chunks(cg, seq, chunk_record,
                                                                                   Buffer.t)  # chunks

            cg = learning_and_update(current_chunks_idx, chunk_record, cg, Buffer.t, learn=False)

            maxchunksize = cg.getmaxchunksize()
            seq = Buffer.refactor(seq, dt)
            Buffer.reloadsize = maxchunksize + 1
            Buffer.checkreload(arayseq)
            seq_over = Buffer.checkseqover()
        print([ck.count for ck in cg.chunks])

        independence = cg.independence_test()
        cg = rational_learning(cg, n_update=10)
        print('Average Encoding Length is            ER = ', cg.eval_avg_encoding_len())
        seq_over = False
        Iter = Iter + 1

    return cg, chunk_record


def reload(seq, seql, arraysequence, t, max_chunksize):
    # reload arraysequence starting from time point t to the set sequence representations
    # t: the current time point
    T = seql  # current buffer length
    time = t + T
    relevantarray = arraysequence[time:time + max_chunksize, :, :]
    _, H, W = arraysequence.shape
    for tt in range(0, min(max_chunksize, relevantarray.shape[0])):
        for h in range(0, H):
            for w in range(0, W):
                v = relevantarray[tt, h, w]
                seq.append((T + tt, h, w, v))
    seql = seql + min(max_chunksize, relevantarray.shape[0])
    return seq, seql


def learn_stc_classes(sequence, cg):
    """
     Parameters:
             sequence: high dimensional spatial temporal sequence for learning
             cg: a chunking graph
     Returns:
             the hierarhcy of chunks learned
     """
    previous_chunk_boundary_record = []
    H, W = sequence.shape[1:]
    cg.update_hw(H, W)
    cg.generate_empty()

    chunk_termination_time = -1 * np.ones([H, W])
    total_time = sequence.shape[0]
    for t in range(0, total_time):  # goes with the order of time.
        chunk_termination_time = chunk_termination_time.astype(int)

        if np.sum(np.abs(sequence[t, :, :])) > 0:
            current_chunks, cg, chunk_termination_time, \
            previous_chunk_boundary_record = identify_chunks(cg, sequence[0:t + 1, :, :], \
                                                             chunk_termination_time, t,
                                                             previous_chunk_boundary_record)  # chunks that ends right before t. the last

        else:
            cg.update_empty(1)  # update empty observation througout the iteration process
            current_chunks = set()  # no chunks ending at the current point
            previous_chunk_boundary_record.append(current_chunks)

        chunk_termination_time, cg = learning_and_update_mtx(current_chunks, t, previous_chunk_boundary_record,
                                                             chunk_termination_time, cg)
        # previous and current chunk
        cg.forget()
    return cg


def independence_test(pM, pT, N):
    # f_obs, f_exp = None, ddof = 0
    f_obs = []
    f_exp = []
    B = list(pM.keys())
    for cl in B:
        for cr in B:
            pclcr = pM[cl] * pM[cr] * N  # expected number of observations
            oclcr = pM[cl] * pT[cl][cr] * N  # number of observations
            f_exp.append(pclcr)
            f_obs.append(oclcr)
    df = (len(B) - 1) ** 2
    _, pvalue = stats.chisquare(f_obs, f_exp=f_exp, ddof=df)

    if pvalue < 0.05:
        return False  # reject independence hypothesis, there is a correlation
    else:
        return True


def get_minimal_complete_cks(seq, cg):
    for i in range(0, seq.shape[0]):
        atomic_chunk = np.zeros([1, 1, 1])
        atomic_chunk[0, 0, 0] = seq[i, 0, 0]
        if arr_to_tuple(atomic_chunk) not in list(cg.M.keys()):
            cg.M[arr_to_tuple(atomic_chunk)] = 1
            cg.T[arr_to_tuple(atomic_chunk)] = {}
            cg.add_chunk_to_vertex(arr_to_tuple(atomic_chunk))
        else:
            cg.M[arr_to_tuple(atomic_chunk)] = cg.M[arr_to_tuple(atomic_chunk)] + 1

    return cg


def rational_chunking_all_info(seq, cg, maxit=4):
    """" Rational chunk learner with the access to information about all the sequence
    Implemented in the hierarchical spatial temporal sequence
    """

    def hypothesis_test(cl, cr, pM, pT, N):
        p1p1 = pM[cl] * pM[cr]
        p1p0 = pM[cl] * (1 - pM[cr])
        p0p1 = (1 - pM[cl]) * pM[cr]
        p0p0 = (1 - pM[cl]) * (1 - pM[cr])

        op1p1 = pM[cl] * pT[cl][cr]
        op1p0 = pM[cl] * (1 - pT[cl][cr])
        op0p1 = 0
        op0p0 = 0

        for ncl in list(pT.keys()):
            if ncl != cl:
                op0p1 = op0p1 + pT[ncl][cr] * pM[ncl]
                op0p0 = op0p0 + (1 - pT[ncl][cr]) * pM[ncl]
        # if any cell contains less than 5 observations, need to
        if op0p0 * N <= 5 or op1p1 * N <= 5 or op1p0 * N <= 5 or op0p1 * N <= 5:
            return True  # cannot continute the test because of lack of sample size

        chisquare = N * (p1p1 * ((op1p1 - p1p1) / p1p1) ** 2 + p1p0 * ((op1p0 - p1p0) / p1p0) ** 2 + p0p1 * (
                (op0p1 - p0p1) / p0p1) ** 2
                         + p0p0 * ((op0p0 - p0p0) / p0p0) ** 2)
        pvalue = stats.chi2.pdf(chisquare, 1)
        if pvalue < 0.0005:
            return False  # reject independence hypothesis, there is a correlation
        else:
            return True

    # get minimally complete chunk sets
    if len(cg.M) == 0:
        cg = get_minimal_complete_cks(seq, cg)

    if len(cg.M) > 0:
        pM, pT, _, N = partition_seq_hastily(seq, list(cg.M.keys()))

    proposed_chunk_p = {}
    for cl in list(pT.keys()):
        p_cl = pM[cl]
        for cr in list(pT[cl].keys()):
            cr_giv_cl = pT[cl][cr]
            p = p_cl * cr_giv_cl
            # check independence:
            if hypothesis_test(cl, cr, pM, pT, N) == False:  # there is a correlation
                if np.sum(tuple_to_arr(cl)) > 0 and np.sum(tuple_to_arr(cr)) > 0:
                    clcr = arr_to_tuple(np.concatenate((tuple_to_arr(cl), tuple_to_arr(cr)), axis=0))
                    proposed_chunk_p[clcr] = (cl, cr, p)
    sorted_proposal = {k: v for k, v in sorted(proposed_chunk_p.items(), key=lambda item: item[1][2])}

    # find the maximum transition probability, try chunking it
    it = 0  # number of iteration
    while independence_test(pM, pT, N) == False and len(list(sorted_proposal.keys())) > 0:
        new_chunk = list(sorted_proposal.keys())[-1]
        cl = sorted_proposal[new_chunk][0]
        cr = sorted_proposal[new_chunk][1]

        bag_of_chunks = list(pM.keys())
        while new_chunk in bag_of_chunks:
            # print('the new chunk is already existing in the bag of chunks')
            sorted_proposal.pop(new_chunk)
            proposed_chunk_p.pop(new_chunk)
            new_chunk = list(sorted_proposal.keys())[-1]
            cl = sorted_proposal[new_chunk][0]
            cr = sorted_proposal[new_chunk][1]

        cg.add_chunk_to_vertex(new_chunk, left=cl, right=cr)
        new_bag_of_chunks = bag_of_chunks.copy()
        new_bag_of_chunks.append(new_chunk)
        new_pM, new_pT, _, N = partition_seq_hastily(seq, new_bag_of_chunks)
        pM = new_pM
        pT = new_pT
        cg.M = new_pM
        cg.T = new_pT

        proposed_chunk_p = {}
        for cl in list(new_pM.keys()):
            p_cl = new_pM[cl]
            for cr in list(new_pT[cl].keys()):
                cr_giv_cl = new_pT[cl][cr]
                p = p_cl * cr_giv_cl
                if hypothesis_test(cl, cr, pM, pT, N) == False:  # there is a correlation
                    if np.sum(tuple_to_arr(cl)) > 0 and np.sum(tuple_to_arr(cr)) > 0:
                        clcr = arr_to_tuple(np.concatenate((tuple_to_arr(cl), tuple_to_arr(cr)), axis=0))
                        proposed_chunk_p[clcr] = (cl, cr, p)
        sorted_proposal = {k: v for k, v in sorted(proposed_chunk_p.items(), key=lambda item: item[1][2])}

        it = it + 1
        if it > maxit or len(list(sorted_proposal.keys())) == 0:  # in case there is no proposal, exit.
            break

    return cg


# helper functions
def eval_pM_pT(bag_of_chunks, partitioned_sequence, freq=True):
    """checked"""
    M = {}
    T = {}
    for chunk in bag_of_chunks:
        parsed_chunk = chunk
        if freq == True:
            M[chunk] = get_pM_from_partitioned_sequence(parsed_chunk, partitioned_sequence, freq=True)
        else:
            M[chunk] = get_pM_from_partitioned_sequence(parsed_chunk, partitioned_sequence, freq=False)
    for cl in bag_of_chunks:
        T[cl] = {}
        for cr in bag_of_chunks:
            T[cl][cr] = get_pT_from_partitioned_sequence(cl, cr, partitioned_sequence, freq=freq)
    return M, T


def get_pT_from_partitioned_sequence(cl, cr, partitioned_sequence, freq=True):
    '''checked'''
    '''Get the estimated empirical probability of P(chunk2|chunk1),the probability of chunk2 followed by chunk1, 
    in the generated sequence
    In the case when chunk1 never occurs, output the probability of 0'''
    chunk1_count = 0
    chunk1_chunk2 = 0
    # the transition probability from chunk1 to chunk2
    # get P(chunk2|chunk1)
    not_over = True
    i = 0
    for candidate_chunk in partitioned_sequence:
        if candidate_chunk == cl:
            chunk1_count += 1
            if i + 1 < len(partitioned_sequence):
                candidate2 = partitioned_sequence[i + 1]
                if candidate2 == cr:
                    chunk1_chunk2 += 1
        i = i + 1
    if freq:
        return chunk1_count
    else:
        if chunk1_count > 0:
            return chunk1_chunk2 / chunk1_count
        else:
            return 0.0  # returns 0 if there is no occurrance for the first probabilility


def get_pM_from_partitioned_sequence(chunk, partitioned_sequence, freq=True):
    '''Get the estimated empirical probability of P(chunk),the probability of chunk2 followed by chunk1, 
    in the generated sequence
    In the case when chunk1 never occurs, output the probability of 0 '''
    chunk1_count = 0
    for candidate_chunk in partitioned_sequence:
        if candidate_chunk == chunk:
            chunk1_count += 1
    if freq:
        return chunk1_count
    else:
        if chunk1_count > 0:
            return chunk1_count / len(partitioned_sequence)
        else:
            return 0.0


def partition_seq_hastily(this_sequence, bag_of_chunks, freq=False):
    """Parse the sequence using the learned set of chunks, use the maximal chunk that fits the upcoming sequence,
    as described by the rational chunking model in the paper,
    only works for one dimensional chunks"""
    # find the maximal chunk that fits the sequence
    i = 0
    N = 0
    lsq = this_sequence.shape[0]
    end_of_sequence = False
    partitioned_sequence = []
    while end_of_sequence == False:
        N = N + 1
        mxl = 0
        mxck = None
        for chunk in bag_of_chunks:
            this_chunk = tuple_to_arr(chunk)
            len_this_chunk = this_chunk.shape[0]
            if i + len_this_chunk < lsq:
                if np.isclose(this_sequence[i:i + len_this_chunk, :, :], this_chunk).all():
                    if len_this_chunk >= mxl:
                        mxl = len_this_chunk
                        mxck = this_chunk
        if mxck is None:
            break
        partitioned_sequence.append(arr_to_tuple(mxck))
        i = i + mxl
        if i >= lsq: end_of_sequence = True

    this_pM, this_pT = eval_pM_pT(bag_of_chunks, partitioned_sequence, freq=freq)

    return this_pM, this_pT, partitioned_sequence, N


def evaluate_KL_compared_to_ground_truth(reproduced_sequence, generative_marginals, cg):
    """generative_marginal: marginals used in generating ground truth """
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    ground_truth_set_of_chunks = set(generative_marginals.keys())
    for chunk in generative_marginals.keys():
        cg.M[chunk] = 0

    learned_M, _, _, N = partition_seq_hastily(reproduced_sequence, list(cg.M.keys()))
    # learned_M = partition_seq_STC(reproduced_sequence, cg) # in this

    # compare the learned M with the generative marginals
    # Iterate over dictionary keys, and add key values to the np.array to be compared
    # based on the assumption that the generative marginals should have the same key as the probability ground truth.
    probability_learned = []
    probability_ground_truth = []
    for key in list(learned_M.keys()):
        probability_learned.append(learned_M[key])
        probability_ground_truth.append(generative_marginals[key])
    probability_learned = np.array(probability_learned)
    probability_ground_truth = np.array(probability_ground_truth)
    eps = 0.000000001
    EPS = np.ones(probability_ground_truth.shape) * eps
    # return divergence

    v_M1 = probability_ground_truth  # input, q
    v_M1 = EPS + v_M1  # take out the protection to see what happens
    v_M2 = probability_learned  # output p
    v_M2 = EPS + v_M2

    # calculate the kl divergence
    def kl_divergence(p, q):  # usually, p is the output, and q is the input.
        return sum(p[i] * np.log2(p[i] / q[i]) for i in range(len(p)))  # KL divergence in units of bits.

    # p_log_p_div_q = np.multiply(v_M1,np.log(v_M1/v_M2)) # element wise multiplication
    KL = kl_divergence(v_M2, v_M1)
    # div = np.sum(np.matmul(v_M1.transpose(),p_log_p_div_q))
    return KL
