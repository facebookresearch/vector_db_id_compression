
import sys 
import argparse
import pdb
import time 

import numpy as np
from rpc.codecs import ClusterCodec, UniformScalarCodec, ROCSortedListCodec, \
   BigUniformScalarCodec, UniformScalarCodec, UniformCodec, BigUniform, BigUniformCodec
from rpc.rans import initialize_ans_state, compute_ans_state_size_in_bytes
from craystack.rans import flatten, unflatten
from sortedcontainers import SortedList

from rpc.rans import uniform_ans_decode, uniform_ans_encode
from craystack.rans import (
    push_with_finer_prec_uniform,
    pop_with_finer_prec_uniform)

from craystack import vrans, Uniform 
from warnings import warn

def run_version_0(message, codec): 
    ans_state = initialize_ans_state()
    ans_state = codec.encode(message, ans_state)
    # print(flatten(ans_state))

    _, new = codec.decode(ans_state)
    assert set(new) == set(message)


#####################################################
# version 1
#####################################################

class ClusterCodec2: # (Codec[NDArray[uint]]):

    def __init__(self, cluster_size, log_prec=64):
        self.roc_codec = ROCSortedListCodec(
            set_size=cluster_size,
            symbol_codec=BigUniformScalarCodec(log_prec=np.uint32(log_prec)),
            copy_input=False,
        )

    def encode(self, cluster_ids, ans_state):
        sorted_seq = SortedList(cluster_ids)
        ans_state = self.roc_codec.encode(sorted_seq, ans_state)
        return ans_state

    def decode(self, ans_state):
        ans_state, sorted_seq = self.roc_codec.decode(ans_state)
        return ans_state, np.array(sorted_seq)

rans_l = np.uint32(1 << 31)  # the lower bound of the normalisation interval

def run_version_1(message, codec, log_prec):
    ans_state = (np.full(1, rans_l, "uint64"), ())
    roc_codec = ROCSortedListCodec(
        set_size=len(message),
        symbol_codec=BigUniformScalarCodec(log_prec=np.uint32(log_prec)),
        copy_input=False,
    )
    sorted_seq = SortedList(message)
    ans_state = roc_codec.encode(sorted_seq, ans_state)
    # codec = ClusterCodec(cluster_size=len(message))
    # ans_state = codec.encode(message, ans_state)
    # print(flatten(ans_state))
    _, new = codec.decode(ans_state)
    assert set(new) == set(message)

#####################################################
# version 8
#####################################################

atleast_1d_func = lambda x: np.atleast_1d(x).astype("uint64")

def stack_extend(stack, arr):
    return arr, stack


def stack_slice(stack, n):
    slc = []
    while n > 0:
        if stack:
            arr, stack = stack
        else:
            warn('Popping from empty message. Generating random data.')
            rng = np.random.default_rng(0)
            arr, stack = rng.integers(1 << 32, size=n, dtype='uint32'), ()
        if n >= len(arr):
            slc.append(arr)
            n -= len(arr)
        else:
            slc.append(arr[:n])
            stack = arr[n:], stack
            break
    return stack, np.concatenate(slc)

def pop_with_finer_prec_uniform_3(ans_state, precisions, atleast_1d: bool = True):
    if atleast_1d:
        precisions = atleast_1d_func(precisions)

    assert len(precisions) == 1
    precision = precisions[0]
    
    head_, tail_ = ans_state
    # head_ in [2 ^ 32, 2 ^ 64)
    # idxs = head_ >= precisions * ((rans_l // precisions) << np.uint8(32))
    # if np.any(idxs):
    head_0 = head_[0]
    if head_0 >= precision * ((rans_l // precision) << np.uint8(32)):
        tail_ = stack_extend(tail_, np.array([head_0], dtype='uint32'))
        head_ = np.copy(head_)  # Ensure no side-effects
        head_0 >>= np.uint8(32)

    # head in [precisions * (2 ^ 32 // precisions), precisions * ((2 ^ 32 // precisions) << 32))
    # s' mod 2^r
    # TODO(dsevero): might be better to make the ANS head a scalar

    cfs = head_0 % precision 
    
    def pop(symbols):
        if atleast_1d:
            symbols = atleast_1d_func(symbols)

        # calculate previous state  s = p*(s' // 2^r) + (s' % 2^r) - c
        head = (head_0 // precision) + cfs - symbols

        # check which entries need renormalizing
        if head_0 < rans_l:
            # new_head = 32*n bits from the tail
            # tail = previous tail, with 32*n less bits
            tail, new_head = stack_slice(tail_, 1)

            # update LSBs of head, where needed
            # head[idxs] = (head[idxs] << 32) | new_head
            head = (head << np.uint8(32)) | new_head[0]
        else:
            tail = tail_
        return head, tail

    return cfs, pop

def run_version_8(message, codec, log_prec): 

    ans_state = (np.full(1, rans_l, "uint64"), ())
    symbol_codec = BigUniformScalarCodec(log_prec=np.uint32(log_prec))
    sorted_seq = SortedList(message)
    set_size = len(sorted_seq)
    for i in range(set_size):
        # Sample/Decode, without replacement, an index using ANS.
        # Initialize a uniform codec for the indices.
        prec = np.uint32(set_size - i)

        # ans_state, symbols = uniform_ans_decode(ans_state, [prec])
        symbols, pop = pop_with_finer_prec_uniform_3(ans_state, [prec])
        
        ans_state = pop(symbols)
        index = symbols

        # `index` is NDArray[uint], need to cast to int to pick the element.
        symbol = sorted_seq.pop(int(index))

        # Encode the element into the ans state.
        ans_state = symbol_codec.encode(symbol, ans_state)

    # print(flatten(ans_state))
    _, new = codec.decode(ans_state)
    assert set(new) == set(message)



#####################################################
# version 11
#####################################################

def codec_push_1(message, symbol, precision): 
    _uniform_enc_statfun = lambda s: (s, 1)
    start, freq = _uniform_enc_statfun(symbol)
    return vrans.push(message, start, freq, precision),

def codec_push_11(message, symbol, precision):
    for lower in [0, 16, 32, 48]:
        s = (symbol >> lower) & ((1 << 16) - 1)
        diff = np.where(precision >= lower, precision - lower, 0)
        p = np.minimum(diff, 16)
        
        # message, = Uniform(p).push(message, s)
        message, = codec_push_1(message, s, p)
    return message,    


def run_version_11(message, codec, log_prec): 
    ans_state = (np.full(1, rans_l, "uint64"), ())

    sorted_seq = SortedList(message)
    set_size = len(sorted_seq)
    for i in range(set_size):
        # Sample/Decode, without replacement, an index using ANS.
        # Initialize a uniform codec for the indices.
        prec = np.uint32(set_size - i)

        # ans_state, symbols = uniform_ans_decode(ans_state, [prec])
        symbols, pop = pop_with_finer_prec_uniform_3(ans_state, [prec])
        
        ans_state = pop(symbols)
        index = symbols

        # `index` is NDArray[uint], need to cast to int to pick the element.
        symbol = sorted_seq.pop(int(index))

        # Encode the element into the ans state.
        # ans_state = symbol_codec.encode(symbol, ans_state)
        # ans_state = symbol_codec_vectorized_codec.encode(np.array([symbol]), ans_state)
        (ans_state,) = codec_push_11(ans_state, np.array([symbol]), precision=np.uint32(log_prec))    

    # print(flatten(ans_state))
    _, new = codec.decode(ans_state)
    assert set(new.ravel()) == set(message)


#####################################################
# version 13
#####################################################

def stack_slice_13(stack):
    slc = []
    print("AAAA")
    # assert n == 1
    if stack:
        arr, stack = stack
    else:
        warn("Popping from empty message. Generating random data.")
        arr, stack = rng.integers(1 << 32, size=n, dtype="uint32"), ()
    return stack, arr

def stack_extend(stack, arr):
    return arr, stack


def pop_with_finer_prec_uniform_13(ans_state, precision):
    precision = np.uint64(precision)
    
    head_, tail_ = ans_state
    # head_ in [2 ^ 32, 2 ^ 64)
    head_0 = head_[0]
    if head_0 >= precision * ((rans_l // precision) << np.uint8(32)):
        tail_ = stack_extend(tail_, np.array([head_0], dtype='uint32'))
        head_0 >>= np.uint8(32)

    # head in [precisions * (2 ^ 32 // precisions), precisions * ((2 ^ 32 // precisions) << 32))
    # s' mod 2^r
    cfs = head_0 % precision 

    symbol = cfs
        
    # calculate previous state  s = p*(s' // 2^r) + (s' % 2^r) - c
    head = (head_0 // precision) # + cfs - symbol
    # check which entries need renormalizing
    if head_0 < rans_l:
        # new_head = 32*n bits from the tail
        # tail = previous tail, with 32*n less bits
        tail, new_head = stack_slice_13(tail_)

        # update LSBs of head, where needed
        # head[idxs] = (head[idxs] << 32) | new_head
        head = (head << np.uint8(32)) | new_head[0]
    else:
        tail = tail_
    # return np.array([head], dtype='uint64'), tail

    return cfs, (np.array([head], dtype='uint64'), tail)


def vrans_push_ref(x, starts, freqs, precisions):
    starts, freqs, precisions = map(atleast_1d_func, (starts, freqs, precisions))
    head, tail = x
    # assert head.shape == starts.shape == freqs.shape
    #print("REF", head, tail, freqs, ((rans_l >> precisions) << 32) * freqs)
    idxs = head >= ((rans_l >> precisions) << 32) * freqs
    if np.any(idxs):
        tail = stack_extend(tail, np.uint32(head[idxs]))
        head = np.copy(head)  # Ensure no side-effects
        head[idxs] >>= 32
    #print("REF2", head, tail)
    head_div_freqs, head_mod_freqs = np.divmod(head, freqs)
    return (head_div_freqs << precisions) + head_mod_freqs + starts, tail


def vrans_push_13(x, start, freq, precision):
    start = int(start)
    freq = int(freq)
    precision = int(precision)
    head, tail = x
    head = int(head[0])
    # print("NEW", head, tail, freq, ((int(rans_l) >> precision) << 32) * freq)
    assert ((int(rans_l) >> precision) << 32) * freq > 0, pdb.set_trace()
    # CAVEAT rans_l should be in 64 bits
    if head >= ((int(rans_l) >> precision) << 32) * freq:
        tail = stack_extend(tail, np.array([head & ((1<<32) - 1)], dtype=np.uint32))
        head >>= 32
    #print("NEW2", head, tail)
    assert freq == 1
    head_div_freq, head_mod_freq = head, 0
    head_2 = (head_div_freq << precision) + head_mod_freq + start
    return np.array([head_2], dtype='uint64'), tail


def codec_push_13(message, symbol, precision):
    symbol = int(symbol) 
    precision = int(precision)
    for lower in [0, 16, 32, 48]:
        s = (symbol >> lower) & ((1 << 16) - 1)
        p = min(max(precision - lower, 0), 16)        
        ref_message = vrans_push_ref(message, s, 1, p)
        message = vrans_push_13(message, s, 1, p)
        # message = vrans.push(message, s, 1, p)
        assert message == ref_message  # , pdb.set_trace()
        #message = ref_message
    return message    

def run_version_13(message, codec, log_prec):

    ans_state = (np.full(1, rans_l, "uint64"), ())

    sorted_seq = SortedList(message)
    set_size = len(sorted_seq)
    for i in range(set_size):
        # Sample/Decode, without replacement, an index using ANS.
        # Initialize a uniform codec for the indices.
        prec = np.uint32(set_size - i)

        # ans_state, symbols = uniform_ans_decode(ans_state, [prec])
        symbols, ans_state = pop_with_finer_prec_uniform_13(ans_state, prec)
        
        # ans_state = pop(symbols)
        index = symbols

        # `index` is NDArray[uint], need to cast to int to pick the element.
        symbol = sorted_seq.pop(int(index))

        # Encode the element into the ans state.
        ans_state = codec_push_13(ans_state, symbol, precision=log_prec)    

    # print(flatten(ans_state))
    _, new = codec.decode(ans_state)
    assert set(new.ravel()) == set(message)

#####################################################
# version 14
#####################################################


def stack_slice_14(stack):
    slc = []
    if stack:
        arr, stack = stack
    else:
        warn("Popping from empty message. Generating random data.")
        arr, stack = rng.integers(1 << 32, size=n, dtype="uint32"), ()
    return stack, arr

def stack_extend(stack, arr):
    return arr, stack


def pop_with_finer_prec_uniform_14(ans_state, precision):
    precision = np.uint64(precision)
    
    head_, tail_ = ans_state
    # head_ in [2 ^ 32, 2 ^ 64)
    head_0 = head_[0]
    if head_0 >= precision * ((rans_l // precision) << np.uint8(32)):
        tail_ = stack_extend(tail_, np.array([head_0], dtype='uint32'))
        head_0 >>= np.uint8(32)

    # head in [precisions * (2 ^ 32 // precisions), precisions * ((2 ^ 32 // precisions) << 32))
    # s' mod 2^r

    cfs = head_0 % precision 

    symbol = cfs
        
    # calculate previous state  s = p*(s' // 2^r) + (s' % 2^r) - c
    head = (head_0 // precision) # + cfs - symbol
    # check which entries need renormalizing
    if head_0 < rans_l:
        # new_head = 32*n bits from the tail
        # tail = previous tail, with 32*n less bits
        tail, new_head = stack_slice_14(tail_)

        # update LSBs of head, where needed
        # head[idxs] = (head[idxs] << 32) | new_head
        head = (head << np.uint8(32)) | new_head[0]
    else:
        tail = tail_
    # return np.array([head], dtype='uint64'), tail

    return cfs, (np.array([head], dtype='uint64'), tail)


def vrans_push_14(x, start, freq, precision):
    start = int(start)
    freq = int(freq)
    precision = int(precision)
    head, tail = x
    head = int(head[0])
    if head >= ((int(rans_l) >> precision) << 32) * freq:
        tail = stack_extend(tail, np.array([head & ((1<<32) - 1)], dtype=np.uint32))
        head >>= 32

    assert freq == 1
    head_div_freq, head_mod_freq = head, 0
    head_2 = (head_div_freq << precision) + head_mod_freq + start
    return np.array([head_2], dtype='uint64'), tail
   
def codec_push_14(message, symbol, precision):
    symbol = int(symbol) 
    precision = int(precision)
    for lower in [0, 16, 32, 48]:
        s = (symbol >> lower) & ((1 << 16) - 1)
        p = min(max(precision - lower, 0), 16)        
        message = vrans_push_14(message, s, 1, p)
    return message    

def run_version_14(message, codec, log_prec):

    ans_state = (np.full(1, rans_l, "uint64"), ())

    sorted_seq = SortedList(message)
    set_size = len(sorted_seq)
    for i in range(set_size):
        # Sample/Decode, without replacement, an index using ANS.
        # Initialize a uniform codec for the indices.
        nmax = np.uint32(set_size - i)

        # ans_state, symbols = uniform_ans_decode(ans_state, [prec])
        index, ans_state = pop_with_finer_prec_uniform_14(ans_state, nmax)
        
        # `index` is NDArray[uint], need to cast to int to pick the element.
        symbol = sorted_seq.pop(int(index))

        # Encode the element into the ans state.
        ans_state = codec_push_14(ans_state, symbol, precision=log_prec)    

    # print(flatten(ans_state))
    _, new = codec.decode(ans_state)
    assert set(new.ravel()) == set(message)


#####################################################
# version 16
#####################################################



class ANSState: 

    def __init__(self): 
        self.head = np.array([rans_l], dtype='uint64')
        self.stack = []


    def extend_stack(self, x): 
        x = np.array([x], dtype=np.uint32)
        self.stack.append(x)

    def set_head(self, x): 
        assert type(x) in (int, np.uint64, np.int64), pdb.set_trace()
        self.head = np.array([x], dtype='uint64')

    def stack_slice(self): 
        if self.stack:
            arr2 = self.stack.pop() 
            return arr2
        else: 
            rng = np.random.default_rng(0)
            warn("Popping from empty message. Generating random data.")
            return rng.integers(1 << 32, size=1, dtype="uint32")

    def get_head(self): 
        return self.head[0]

    def get_head_tail(self): 
        tail = ()
        for e in self.stack: 
            tail = e, tail 
        return self.head, tail 


def pop_with_finer_prec_uniform_16(ans_state, nmax):
    nmax = np.uint64(nmax)

    head_0 = ans_state.get_head()
    if head_0 >= nmax * ((int(rans_l) // int(nmax)) << 32):
        ans_state.extend_stack(head_0)
        head_0 >>= np.uint8(32)

    # head in [precisions * (2 ^ 32 // precisions), precisions * ((2 ^ 32 // precisions) << 32))
    # s' mod 2^r
    cfs = head_0 % nmax 

    # symbol = cfs
        
    # calculate previous state  s = p*(s' // 2^r) + (s' % 2^r) - c
    head = head_0 // nmax
    # check which entries need renormalizing
    if head_0 < rans_l:
        # tail = previous tail, with 32*n less bits
        new_head = ans_state.stack_slice()

        # update LSBs of head, where needed
        head = (head << np.uint8(32)) | new_head[0]
        
    ans_state.set_head(head)
    return cfs


def vrans_push_16(ans_state, start, precision):
    start = int(start)
    precision = int(precision)
    head = int(ans_state.get_head())
    if head >= ((int(rans_l) >> precision) << 32):
        ans_state.extend_stack(head & ((1<<32) - 1))
        head >>= 32

    head_2 = (head << precision) + start
    ans_state.set_head(head_2)
       
def codec_push_16(ans_state, symbol, precision):
    symbol = int(symbol) 
    precision = int(precision)
    for lower in [0, 16, 32, 48]:
        s = (symbol >> lower) & ((1 << 16) - 1)
        p = min(max(precision - lower, 0), 16)        
        vrans_push_16(ans_state, s, p)




def run_version_16(message, codec, precision):
    ans_state = ANSState()
    sorted_seq = SortedList(message)
    set_size = len(sorted_seq)
    for i in range(set_size):
        # Sample/Decode, without replacement, an index using ANS.
        # Initialize a uniform codec for the indices.
        nmax = np.uint32(set_size - i)

        index = pop_with_finer_prec_uniform_16(ans_state, nmax)
        # `index` is NDArray[uint], need to cast to int to pick the element.
        symbol = sorted_seq.pop(int(index))

        # Encode the element into the ans state.
        codec_push_16(ans_state, symbol, precision=precision)    

    # print(flatten(ans_state))
    _, new = codec.decode(ans_state.get_head_tail())
    assert set(new.ravel()) == set(message)


#####################################################
# With decoding 
#####################################################



def push_with_finer_prec_uniform_17(ans_state, symbol, precision):
    # head in [2 ^ 32, 2 ^ 64)
    head_0 = ans_state.get_head()
    if head_0 >= ((np.uint64(rans_l) // precision) << np.uint8(32)):
        ans_state.extend_stack(int(head_0) & 0xffffffff)
        head_0 >>= np.uint8(32)
    # head in [((rans_l // precisions)) * freqs), ((rans_l // precisions)) * freqs) << 32)

    # calculate next state s' = 2^r * (s // p) + (s % p) + c
    head_0 = head_0 * precision + symbol

    # head in [precisions * (2 ^ 32 // precisions), precisions * ((2 ^ 32 // precisions) << 32))
    # check which entries need renormalizing
    if head_0 < rans_l:
        # new_head = 32*n bits from the tail
        # tail = previous tail, with 32*n less bits
        new_head = ans_state.stack_slice()

        # update LSBs of head, where needed
        head_0 = (head_0 << np.uint8(32)) | new_head
    # head in [2 ^ 32, 2 ^ 64)
    ans_state.set_head(head_0)


def vrans_pop_17(ans_state, precision):
    head_0 = ans_state.get_head()
    precision = int(precision)    
    cfs = int(head_0) & ((1 << precision) - 1)
    head = (int(head_0) >> precision) #  + int(cfs) - start  
    if head < rans_l:
        new_head = ans_state.stack_slice()[0]
        head = (head << 32) | new_head

    ans_state.set_head(head)
    return cfs


def codec_pop_17(ans_state, precision): 
    symbol = 0
    for lower in [48, 32, 16, 0]:
        p = min(max(precision - lower, 0), 16)        
        s = vrans_pop_17(ans_state, p)
        symbol = (symbol << 16) | s
    return symbol


def codec_decode_17(ans_state, set_size, log_prec):
    sorted_seq = SortedList([])

    for i in range(set_size): 
        symbol = codec_pop_17(ans_state, np.uint32(log_prec))

        # Add it to the sorted list and recover `index`.
        sorted_seq.add(symbol)
        index = np.uint32(sorted_seq.index(symbol))

        # Encode `index` back into the state to reverse sampling.
        nmax = np.uint32(i + 1)
        push_with_finer_prec_uniform_17(ans_state, index, nmax)

    return np.array(sorted_seq)

def run_version_17(message, codec, precision):
    ans_state = ANSState()
    t0 = time.time()
    sorted_seq = SortedList(message)
    set_size = len(sorted_seq)
    for i in range(set_size):
        # Sample/Decode, without replacement, an index using ANS.
        # Initialize a uniform codec for the indices.
        nmax = np.uint32(set_size - i)

        index = pop_with_finer_prec_uniform_16(ans_state, nmax)
        # `index` is NDArray[uint], need to cast to int to pick the element.
        symbol = sorted_seq.pop(int(index))

        # Encode the element into the ans state.
        codec_push_16(ans_state, symbol, precision=precision)    

    # print("ans_state=", ans_state.head, ans_state.stack)
    t1 = time.time()
    new = codec_decode_17(ans_state, set_size, precision)
    t2 = time.time()
    print(f"encode {(t1-t0)*1000:.3f} ms,  decode {(t2-t1)*1000:.3f} ms")
    assert set(new.ravel()) == set(message)


#####################################################
# Main 
#####################################################

def main(): 
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('message options')
    aa('--size', type=int, default=200)
    aa('--nbits', type=int, default=27)
    aa('--seed', type=int, default=123)
    aa('--testmsg', type=int, default=0)

    group = parser.add_argument_group('codec options')
    aa('--version', type=int, default=0)
    aa('--nbits2', type=int, default=64)

    args = parser.parse_args()
    print("args:", args)

    #### make message

    if args.testmsg == 0: 
        rs = np.random.RandomState(args.seed)
        message = rs.choice(1<<args.nbits, size=args.size, replace=False)
    elif args.testmsg == 1: 
        message = np.array([12351235, 49024902, 17781778, 36663666]).astype('uint64')
    elif args.testmsg == 2: 
        message = np.array([12351235]).astype('uint64')
    elif args.testmsg == 3: 
        message = np.array([1235]).astype('uint64')

    # reference decoder
    codec = ClusterCodec2(cluster_size=len(message), log_prec=args.nbits2)

    if args.version == 0: 
        run_version_0(message, codec)
    elif args.version == 1: 
        run_version_1(message, codec, args.nbits2)
    elif args.version == 8: 
        run_version_8(message, codec, args.nbits2)
    elif args.version == 11: 
        run_version_11(message, codec, args.nbits2)
    elif args.version == 13: 
        run_version_13(message, codec, args.nbits2)
    elif args.version == 14: 
        run_version_14(message, codec, args.nbits2)
    elif args.version == 16: 
        run_version_16(message, codec, args.nbits2)
    elif args.version == 17: 
        run_version_17(message, codec, args.nbits2)

    else: 
        raise NotImplemented()

if __name__ == "__main__": 
    main()