#this is the code of private hierarchical heavy hitters.
#program.use_split(3)
from Compiler import types, library, instructions
from Compiler.library import break_loop, start_timer, stop_timer
from Compiler.types import Array, MemValue
import itertools
from Compiler.types import sint, cint, regint
from Compiler.circuit import sha3_256
from Compiler.GC.types import sbitvec, sbits
from Compiler.instructions import time,crash
import numpy as np
#program.use_split(3)

def gen_bit_perm(b):
    """
    input:b=[0,1,0,1], output:[0,2,1,3]
    leak: null
    """
    B = types.sint.Matrix(len(b), 2)
    B.set_column(0, 1 - b.get_vector())
    B.set_column(1, b.get_vector())
    Bt = B.transpose()  #Bt=[[1,0,1,0],[0,1,0,1]]
    Bt_flat = Bt.get_vector()
    St_flat = Bt.value_type.Array(len(Bt_flat))
    St_flat.assign(Bt_flat)  #St_flat=[1,0,1,0,0,1,0,1]
    @library.for_range(len(St_flat) - 1)
    def _(i):
        St_flat[i + 1] = St_flat[i + 1] + St_flat[i]
    Tt_flat = Bt.get_vector() * St_flat.get_vector()  # Tt_flat[[1],[0],[2],[0],[0],[3],[0],[4]]
    Tt = types.Matrix(*Bt.sizes, B.value_type)
    Tt.assign_vector(Tt_flat)  #Tt_flat  Tt=[[1,0,2,0], [0,3,0,4]]
      
    return sum(Tt) - 1  #[0,2,1,3]

def inverse_permutation(k):
    """
    get inverse permutation
    """
    shuffle = types.sint.get_secure_shuffle(len(k))  #shuffle
    k_prime = k.get_vector().secure_permute(shuffle).reveal()  #shuffle k,k_prime
    idx = Array.create_from(k_prime)
    res = Array.create_from(types.sint(types.regint.inc(len(k))))  #sint Array [0,1,2,...,len(k)-1]
    res.secure_permute(shuffle, reverse=False)  # shuffle res
    res.assign_slice_vector(idx, res.get_vector())
    library.break_point()
    instructions.delshuffle(shuffle)  #shuffle
    return res



def apply_perm(k, D, reverse=False):
    """
    apply k to D
    leak: null
    """
    assert len(k) == len(D)
    library.break_point()
    shuffle = types.sint.get_secure_shuffle(len(k))
    k_prime = k.get_vector().secure_permute(shuffle).reveal()
    idx = types.Array.create_from(k_prime)
    if reverse:
        D.assign_vector(D.get_slice_vector(idx))
        library.break_point()
        D.secure_permute(shuffle, reverse=True)
    else:
        D.secure_permute(shuffle)
        library.break_point()
        v = D.get_vector()
        D.assign_slice_vector(idx, v)
    library.break_point()
    instructions.delshuffle(shuffle)



def radix_sort(k0, D0, n_bits=16, get_D=True, signed=True):
    """
    this is same as the MP-SPDZ
    leak: only leak len(k)
    """
    k = k0.same_shape()
    k.assign(k0)
    D = D0.same_shape()
    D.assign(D0)
    #D.print_reveal_nested(end='\n')
    assert len(k) == len(D)  # k=[2,5,0,1]
    bs = types.Matrix.create_from(k.get_vector().bit_decompose(n_bits))  #bit_decompose bs=[[0,1,0,1],[1,0,0,0],[0,1,0,0],[0,0,0,0,],...[0,0,0,0]]
    if signed and len(bs) > 1:
        bs[-1][:] = bs[-1][:].bit_not()  # bs
    h = types.Array.create_from(types.sint(types.regint.inc(len(k))))  #sint Array [0,1,2,...,len(k)-1]
    @library.for_range(len(bs))  #
    def _(i):
        b = bs[i]
        c = gen_bit_perm(b)  #
        apply_perm(c, h, reverse=False)  # h1=c0(h0),h=cn...c3c2c1c0(h0),h0=[0,1,2,3,...]
        @library.if_e(i < len(bs) - 1)
        def _():
            apply_perm(h, bs[i + 1], reverse=True)
        @library.else_
        def _():
            #D.print_reveal_nested(end='\n')
            apply_perm(h, D, reverse=True)
            #D.print_reveal_nested(end='\n')
    D_order = inverse_permutation(h)  #
    #D.print_reveal_nested(end='\n')
    if get_D:
        return D
    else:
        return D_order

def phhh_1(k0,n_bits=16, t=1):
    # my first scheme for phhh, this scheme is secure and efficient. k0 is the data array, n_bits is the bit length. t is threshold.
    # leak: bit length n_bits, data number len(k0)
    k = k0.same_shape()
    k.assign(k0)
  
    sorted_data = radix_sort(k,k,n_bits,signed=False)
    #sorted_data.print_reveal_nested(end='\n')
    frequency = types.Array.create_from(types.sint(types.regint.inc(size=len(k),base=1,step=0)))#sint array [1,1,1...,1]
    equ = types.Matrix(n_bits,len(sorted_data)-1,sint)  #restore the equal information of k 

    bs = types.Matrix.create_from(sorted_data.get_vector().bit_decompose(n_bits))  #bit_decompose
    bsh2l = bs.same_shape()
    #bs.print_reveal_nested(end='\n')
    bsh2l_t = types.Matrix(len(k0),n_bits,sint)
    @library.for_range(len(bs))
    def _(i):
        bsh2l[i] = bs[n_bits-1-i]
    @library.for_range(len(bsh2l))
    def _(i):
        @library.for_range(len(bsh2l[i]))
        def _(j):
            bsh2l_t[j][i] = bsh2l[i][j]
    #bsh2l_t.print_reveal_nested(end='\n')
    b = bsh2l[0]
    @library.for_range(len(b)-1)
    def _(j):
        equ[0][j] = b[j].equal(b[j+1],1)
    @library.for_range(len(bsh2l)-1)
    def _(i):
        b = bsh2l[i+1] #bs[x], x must be a natural number
        @library.for_range(len(b)-1)
        def _(j):
            equ[i+1][j] = equ[i][j]*b[j].equal(b[j+1],1)
    hdata = sint.Tensor([n_bits,len(k0),n_bits])
    #equ.print_reveal_nested(end='\n')

    @library.for_range(len(equ))
    def _(i1):
        i = len(equ)-i1-1    
        @library.for_range(len(k0)-1)
        def _(j1):
            j = len(k0) - j1 -1
            frequency[j-1] = frequency[j-1] + frequency[j]*equ[i][j-1]
            frequency[j] = frequency[j]*(1-equ[i][j-1])  # operations where one of the operands is an sint either result in an sint or an sinbit, the latter for comparisons
        @library.for_range(len(k0))
        def _(j2):
            temp_equ = frequency[j2].greater_equal(t)
            temp_bs = bsh2l_t[0].same_shape()
            temp_bs.assign(bsh2l_t[j2])
            @library.for_range(len(temp_bs))
            def _(l1):
                temp_bs[l1] = temp_bs[l1] * temp_equ +(-2)*(temp_equ - 1) # 2 represent null
            @library.for_range(len(temp_bs)-i-1)
            def _(l2):
                temp_bs[l2+i+1] = sint(2)
            hdata[i][j2].assign(temp_bs)
            frequency[j2] = frequency[j2]*(1-temp_equ)
    
    #hdata.print_reveal_nested(end='\n')  #the true output without leaking
    @library.for_range(len(hdata))   #the output for observing
    def _(i):
        @library.for_range(len(hdata[i]))
        def _(j):
            @library.if_(sint(1).greater_equal(hdata[i][j][0]).reveal())
            def _():
                hdata[i][j].print_reveal_nested(end='; ')
    


   
def compact(t,p1,p2):
    # t is 0 or 1, p1 and p2 is payload. compact t=1 items to the head.
    #leak:len(t)
    #t.print_reveal_nested(end='\n')
    c0 = p1.same_shape()
    c1 = p1.same_shape()
    label = p1.same_shape()
    c1[0] = t[0]
    c0[0] = sint(1) - c1[0]
    @library.for_range(len(t)-1)
    def _(i):
        c1[i+1] = c1[i] + t[i+1]
        c0[i+1] = sint(i+2) - c1[i+1]
    @library.for_range(len(t))
    def _(i):
        temp_equ = t[i].equal(sint(1))
        label[i] = c1[i]*temp_equ + (c0[i]+c1[len(t)-1])*(1-temp_equ) - 1
    #t.print_reveal_nested(end='\n')
    #label.print_reveal_nested(end='\n')
    apply_perm(label,t)
    apply_perm(label,p1)
    apply_perm(label,p2)
    return c1[len(t)-1]

def bit_equal(a, b, n_bits, get_bits):
    '''
    secretly compare the highest get_bits bits, a b is sint with n_bits, return sint 0/1
    leak:null
    '''
    dec_a = types.Array.create_from(a.bit_decompose(n_bits))
    dec_b = types.Array.create_from(b.bit_decompose(n_bits))
    bits = sint.Array(1)
    bits[0] = sint(1)
    @library.for_range(get_bits)
    def _(i):
        bits[0] = bits[0] * dec_a[n_bits - i -1].equal(dec_b[n_bits - i - 1])
    return bits[0]


    '''
    dec_a =a.bit_decompose(n_bits)[n_bits-get_bits:]
    dec_b =b.bit_decompose(n_bits)[n_bits-get_bits:]
    bits = [1 - (bit_a - bit_b)%2 for bit_a,bit_b in zip(dec_a,dec_b)]
    while len(bits) >cint(1):
        bits.insert(cint(0), bits.pop()*bits.pop())
    return bits[0]
    '''

def get_frequency(sorted_k0):
    # compute frequency, sorted_k0 is the sorted data, n_bits represents the length of data, get_bits represents calculating the frequency of the previous get_bits layers
    # return compacted data and frequency, the number of deduplication data c(cint)
    # leak:len(sorted_k0), the number of deduplication data
    # this function is designed for PHHH2
    k = sorted_k0.same_shape()
    k.assign(sorted_k0)
    indices = types.Array.create_from(types.sint(types.regint.inc(len(k))))  #sint Array [0,1,2,...,len(k)-1]
    equ = k.same_shape()
    equ.assign_all(0)
    equ[0] = sint(1)
    start_timer(70)
    @library.for_range(len(k)-1)
    def _(i):
        equ[i+1] = sint(1) - k[i+1].equal(k[i])
    stop_timer(70)
    start_timer(71)
    c1 = compact(equ,k,indices).reveal()  #leaking c1, the number of deduplication data
    stop_timer(71)
    frequency = indices.same_shape()
    frequency.assign_all(sint(0))
    @library.for_range(c1-1)  #surprise, for_range(start,stop,step) :param start/stop/step: regint/cint/int
    def _(i):
        frequency[i] = indices[i+1]-indices[i]
    frequency[c1-1] = len(indices) - indices[c1-1]
    return k, frequency, c1



def get_frequency_secure(sorted_k0, n_bits, get_bits):
    # compute frequency, sorted_k0 is the sorted data, n_bits represents the length of data, get_bits represents calculating the frequency of the previous get_bits layers
    # return compacted data and frequency, the frequency of deleted data is 0
    # leak:len(sorted_k0)
    # reference: Vogue:Faster Computation of Private Heavy Hitters.
    # this function is designed for PHHH0(the trivial scheme)
    k = sorted_k0.same_shape()
    k.assign(sorted_k0)
    indices = types.Array.create_from(types.sint(types.regint.inc(len(k))))  #sint Array [0,1,2,...,len(k)-1]
    equ = k.same_shape()
    equ.assign_all(0)
    equ[0] = sint(1)
    @library.for_range(len(k)-1)
    def _(i):
        equ[i+1] = sint(1) - bit_equal(k[i+1],k[i],n_bits, get_bits)
            
    c1 = compact(equ,k,indices)
    
    frequency = indices.same_shape()
    frequency.assign_all(0)
    @library.for_range(len(sorted_k0)-1)
    def _(j):
        b21 = sint(2).equal(equ[j]+equ[j+1])
        b22 = indices[j+1] - indices[j]
        b31 = sint(1).equal(equ[j]+equ[j+1])
        b32 = len(indices) - indices[j]
        b2 = b21 * b22
        b3 = b31 * b32
        frequency[j] = b2 + b3
    return k,frequency




#I think mp-spdz do not support prefix tree
def phhh_2(k0,n_bits=16, t=1):
    # my second scheme for phhh, this scheme is insecure and more efficient than phhh_1. k0 is the data array, n_bits is the bit length. t is threshold.
    # leak: bit length n_bits, data number len(k0),the number of deduplication data c, b[s](except the node less than t)
    start_timer(90)
    k = k0.same_shape()
    k.assign(k0)
    
    sorted_data = radix_sort(k,k,n_bits,signed=False) 
    stop_timer(90)
    start_timer(91)
    data, frequency, c = get_frequency(sorted_data)   
    stop_timer(91)
    start_timer(92)
    @library.for_range(c, len(data))
    def _(i):
        data[i] *= sint(0)
      
    tags = cint.Array(len(k))
    tags.assign_all(0)   
    bs = types.Matrix.create_from(data.get_vector().bit_decompose(n_bits))  #bit_decompose
    bsh2l = bs.same_shape()
      
    @library.for_range(len(bs))
    def _(i):
        bsh2l[i] = bs[n_bits-1-i]

    start_timer(80)
    bsh2l_r = bsh2l.reveal()
    stop_timer(80)

    start_timer(81)
    @library.for_range(len(bsh2l))
    def _(i):
        bs_r = bsh2l[i].reveal()
    stop_timer(81)
   
    tags[0] = cint(1) #1 is boundary point, 0 is useful, 2 is deleted
    tags[c] = cint(2)
    fres = types.Matrix(n_bits, len(k), sint) #store the frequencys of all layers
    fres.assign(sint(0))
    fres_t = types.Matrix(n_bits, len(k), cint) # store the compare result, fres[i][j]>=t then fres_t[i][j]=1, else fres_t[i][j]=0, 2 represent do not know, 3 represent the origin is 1 and substract hhh items
    fres_t.assign_all(2)
    parent = types.Matrix(n_bits, len(k), cint) # restore the index of parent node, 0 represent null

    #create the highest layer of the prefixtree, prefixtree is stored as fres
    b = bsh2l[0]
    temp_end = cint(0)
    temp_begin = cint(0)
    temp_1 = cint(0)
       
    @library.for_range(c)
    def _(j):
        @library.if_((tags[j]==cint(1)).bit_and(j>=temp_end))
        def _():
            temp_begin.update(j)
            temp_end.update(c)
            @library.for_range(c-temp_begin-1)
            def _(s):
                @library.if_(tags[temp_begin + s+1] != cint(0))
                def _():
                    temp_end.update(temp_begin + s + 1) #not include end
                    break_loop()
            

            temp_1.update(temp_end)
            b_part = b.same_shape()
            b_part.assign_all(0)
            @library.for_range(temp_begin, temp_end)
            def _(s):
                b_part[s] = b[s]
            b_part_reveal = b_part.reveal()
            @library.for_range(temp_begin, temp_end)
            def _(s):
                #condition3 = b[s].reveal() == cint(1)
                condition3 = (b_part_reveal[s] == 1)
                @library.if_(condition3)
                def _():  # it is not must to reveal
                    temp_1.update(s)
                    break_loop()  

            @library.if_(temp_1==temp_end)
            def _():
                fres[0][temp_begin] = len(k)
            @library.if_(temp_1 == temp_begin)
            def _():
                fres[0][temp_begin] = len(k)
            @library.if_((temp_1>temp_begin).bit_and(temp_1<temp_end))
            def _():
                temp_frequency0 = sint(0)
                @library.for_range(temp_begin,temp_1)
                def _(w):
                    temp_frequency0.update(temp_frequency0 + frequency[w])
                fres[0][temp_begin] = temp_frequency0
                fres[0][temp_1] = len(k) - temp_frequency0
                tags[temp_1] = cint(1)
    stop_timer(92)
    start_timer(93)
    #create the other layers of the prefixtree
    @library.for_range(n_bits-1)
    def _(i):
        b = bsh2l[i+1]
        b_part = b.same_shape()
        b_part.assign_all(0)
        #b_reveal = b.reveal()
        temp_end.update(cint(0))
        @library.for_range(c)
        def _(j):
            condition = (tags[j]==cint(1)).bit_and(j>=temp_end).bit_and(fres_t[i][j]==2)
            @library.if_(condition)
            def _():
                fres_t[i][j] = fres[i][j].greater_equal(t).reveal()
                @library.if_(fres_t[i][j]==0)
                def _():
                    @library.for_range(n_bits -i -1)
                    def _(s):
                        fres_t[s+i+1][j] = 0
            condition1 = (tags[j]==cint(1)).bit_and(j>=temp_end).bit_and(fres_t[i][j])
            @library.if_(condition1)   #reveal()
            def _():
                temp_begin.update(j)
                temp_end.update(c)
                @library.for_range(c-temp_begin-1)
                def _(s):
                    condition2 = tags[temp_begin + s+1] != 0
                    @library.if_(condition2)
                    def _():
                        temp_end.update(temp_begin + s + 1)#not include end
                        break_loop()
                temp_1.update(temp_end)
                @library.for_range(temp_begin, temp_end)
                def _(s):
                    b_part[s] = b[s]
                b_part_reveal = b_part.reveal()
                @library.for_range(temp_begin, temp_end)
                def _(s):
                    condition3 = (b_part_reveal[s] == 1)
                    @library.if_(condition3)
                    def _():  # it is not must to reveal
                        temp_1.update(s)
                        break_loop()
                condition4 = temp_1==temp_end
                @library.if_(condition4)
                def _():
                    fres[i+1][temp_begin] = fres[i][temp_begin]
                    parent[i+1][temp_begin] = temp_begin
                condition5 = temp_1==temp_begin
                @library.if_(condition5)
                def _():
                    fres[i+1][temp_begin] = fres[i][temp_begin]
                    parent[i+1][temp_begin] = temp_begin
                condition6 = (temp_1>temp_begin).bit_and(temp_1<temp_end)
                @library.if_(condition6)
                def _():
                    temp_frequency0 = sint(0)
                    @library.for_range(temp_begin,temp_1)
                    def _(w):
                        temp_frequency0.update(temp_frequency0 + frequency[w])
                    fres[i+1][temp_begin] = temp_frequency0
                    fres[i+1][temp_1] = fres[i][temp_begin] - temp_frequency0
                    parent[i+1][temp_begin] = temp_begin
                    parent[i+1][temp_1] = temp_begin
                    tags[temp_1] = cint(1)              

    @library.for_range(c)
    def _(j):
        condition = (tags[j]==cint(1)).bit_and(fres_t[n_bits-1][j]==2)
        @library.if_(condition)
        def _():
            fres_t[n_bits-1][j] = fres[n_bits-1][j].greater_equal(t).reveal()
           
    stop_timer(93)
    start_timer(94)
    start_timer(87)
    hdata = sint.Tensor([n_bits,len(k0),n_bits])  # store the HHH items, 2 represent null
    hdata.assign_all(sint(2))   
    
    @library.for_range(n_bits)
    def _(i):
        fre = fres[n_bits - i - 1]
        fre_t = fres_t[n_bits -i -1]
        @library.for_range(c)
        def _(j):
            @library.if_(fre_t[j]==3)
            def _():
                fre_t[j] = fre[j].greater_equal(t).reveal()
            @library.if_((fre_t[j]==1))
            def _():
                start_timer(86)
                #condition = fre[j].greater_equal(t).reveal()
                stop_timer(86)
                @library.for_range(n_bits - i)
                def _(s): 
                    hdata[n_bits-i-1][j][s] = bsh2l[s][j]
                temp_i = cint(0)
                temp_j = cint(0)
                temp_j.update(j)
                @library.for_range(n_bits - i -1)  # delete HHH items
                def _(s):
                    temp_i.update(n_bits -i - 1 - s - 1)
                        #@library.while_do(lambda: fres[temp_i][temp_j].equal(0).reveal())
                    #@library.while_do(lambda: fres_t[temp_i][temp_j]==2)
                    #def _():
                    #    temp_j.update(temp_j - 1)
                    temp_j.update(parent[temp_i+1][temp_j])
                    fres[temp_i][temp_j] = fres[temp_i][temp_j] - fre[j]
                    fres_t[temp_i][temp_j] = 3
    
    #hdata.print_reveal_nested(end='\n')  #the true output without leaking
    stop_timer(87)
    hdata_reveal = hdata.reveal()  # without leaking
    @library.for_range(len(hdata_reveal))   #the output for observing
    def _(i):
        @library.for_range(len(hdata_reveal[i]))
        def _(j):
            @library.if_(1>=hdata_reveal[i][j][0])
            def _():
                hdata_reveal[i][j].print_reveal_nested(end='; ')
    stop_timer(94)
    #fres_t.print_reveal_nested(end='\n')




def phhh_0(k0,n_bits=16, t=1):
    # the trivial scheme for phhh, this scheme is secure and  inefficient. k0 is the data array, n_bits is the bit length. t is threshold.
    # leak: bit length n_bits, data number len(k0)
    k = k0.same_shape()
    k.assign(k0)
    
    sorted_data = radix_sort(k,k,n_bits,signed=False)  
    fres = types.Matrix(n_bits, len(k0), sint)
    hdata = sint.Tensor([n_bits, len(k0), n_bits])
    fres.assign_all(sint(0))
    hdata.assign_all(sint(2))
    
    @library.for_range(n_bits)
    def _(i_temp):
        i = n_bits - i_temp -1
        datas, fres[i] = get_frequency_secure(sorted_data, n_bits, n_bits - i_temp)
        bs = types.Matrix.create_from(datas.get_vector().bit_decompose(n_bits))  #bit_decompose
        @library.for_range(len(k0))
        def _(j):
            @library.for_range(i+1)
            def _(s):
                hdata[i][j][s] = bs[n_bits - s - 1][j]
    @library.for_range(n_bits)
    def _(i_temp):
        i = n_bits - i_temp -1
        @library.for_range(len(k))
        def _(j):
            tags = fres[i][j].greater_equal(t)
            fres[i][j] = fres[i][j] * tags
            @library.for_range(i+1)
            def _(s):
                hdata[i][j][s] = tags.if_else(hdata[i][j][s], sint(2))
            @library.for_range(i)
            def _(s):
                @library.for_range(len(k))
                def _(j1):
                    pre = sint(1)
                    @library.for_range(s+1)
                    def _(s1):
                        pre.update(pre * hdata[i][j][s1].equal(hdata[s][j1][s1]))
                    fres[s][j1] -= pre * fres[i][j]
    

    #hdata.print_reveal_nested(end='\n')  #the true output without leaking
    @library.for_range(len(hdata))   #the output for observing
    def _(i):
        @library.for_range(len(hdata[i]))
        def _(j):
            @library.if_(sint(1).greater_equal(hdata[i][j][0]).reveal())
            def _():
                hdata[i][j].print_reveal_nested(end='; ')

       


def generate_zipf_distribution(n_bits, num, zipf_exponent=1.03):
    zipf_data = np.random.zipf(zipf_exponent, size=num)
    max_value = 2**n_bits - 1
    zipf_data = np.clip(zipf_data, 1, max_value)
    return list(map(int, zipf_data.astype(np.uint64)))
    #return zipf_data








