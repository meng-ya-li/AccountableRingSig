from random import randint
import sympy 
import hashlib

n = 3
m = 2
l = randint(0,n**m-1)
int2byte_size = 64 # 64-bit integer
g = 14281528880334450723053870702714241313522996716555816136992203461280239035144
p = 72605870376593568031989838389601275874951467847143895321310129443024924195643


class global_param:
    def __init__(self, g=2, p=61):
        self.g = g
        self.p = p
        self.q = p
        self.s = []
        self.h = []
        for i in range(m*n):
            self.s.append(randint(1,self.q-1))
            self.h.append(pow(g,self.s[i],self.q))

    def show_param(self):
        print('g:', self.g, 'p:', self.p, 'q:', self.q, 's:', self.s, 'h:', self.h)


sprm = global_param(g,p)
sprm.show_param()

def generate_key_for_PKE():
    sk = randint(1,sprm.q-2)
    vk = pow(sprm.g, sk, sprm.q)
    return sk, vk
    '''
# algorithm as given in the paper
# this is not used because the decryption described in the paper does not work in coding.
# R4 zkp protocol has also been changed to adapt to the ElGamal actually coded in this implementation
def ElGamal_enc(pk,m,r):
    #r = randint(1,sprm.q)
    c1 = pow(pk, r, sprm.q)
    s = pow(sprm.g, r, sprm.q)
    c2 = s * m % sprm.q
    return c1, c2

def ElGamal_dec(dk, c1, c2):

    q1 = sprm.q-1
    dk_1 = pow(dk, q1-2, q1)
    invc1 = pow(c1, -1, sprm.q)
    invs = pow(invc1, dk_1, sprm.q)
    m = c2 * invs % sprm.q
    return m
'''
# algorithm as given on wikipedia

def ElGamal_enc(pk,m,r):
    #r = randint(1,sprm.q)
    c1 = pow(sprm.g, r, sprm.q)
    s = pow(pk, r, sprm.q)
    c2 = s * m % sprm.q
    return c1, c2

def ElGamal_dec(dk, c1, c2):
    invs = pow(c1, sprm.q - dk -1, sprm.q) # without the -1, g^q gives g instead of 1
    m = c2 * invs % sprm.q
    return m

# dlji is l encoded in n-ary format in a m row *n column matrix; delta_l_{i,j} in the paper
# use rowindex and columnindex to indicate value; value of each row is n^rowindex*columnindex; adding row values of the whole matrix and you get the value encoded in it. row/col index starts from 0.
# for example, encoding 3 in 2-ary form is [[0, 1], [0, 1], [1, 0]] => 1*(2^0)*1 + 1*(2^1)*1 + 1*(2^3)*0; 
# encoding 7 in 3-nary form is [[0, 1, 0], [0, 0, 1]] => 1*(3^0)*1 + 1*(3^1)*2
def nary_encoding(l): #n-ary encoding of an integer
    nary_l = []
    while l:
        lowest_bit = l % n
        l = int(l/n)
        nary_l.append(lowest_bit)

    if len(nary_l) < m:
        for _ in range(m-len(nary_l)):
            nary_l.append(0)

    return(nary_l)
# translate l into m*n bit encoding in n-ary format in dlji
def encoding_l(l):
    nary_l = []
    while l:
        lowest_bit = l % n
        l = int(l/n)
        nary_l.append(lowest_bit)

    if len(nary_l) < m:
        for _ in range(m-len(nary_l)):
            nary_l.append(0)
    #nary_l.reverse()
    #print(nary_l)
    dlji = []
    for j in range(m):
        row_bit = []
        for i in range(n):
            if nary_l[j]==i:
                row_bit.append(1)
            else:
                row_bit.append(0)
        dlji.append(row_bit)
    return(dlji)

def mn_matrix_to_list(mnmatrix):
    int_list = []
    for j in range(m):
        for i in range(n):
            int_list.append(mnmatrix[j][i])
    return int_list

def commit_int_2Dmnlist(intlist, r):
    commit = []
    for j in range(m):
        c = pow(sprm.g, r, sprm.q)
        for i in range(n):
            if (intlist[j][i] >= sprm.q) or (intlist[j][i] < 0):
                print('m to commit not in Zq')
            hm = pow(sprm.h[j*n+i], intlist[j][i], sprm.q)
            #print('j,i,h,m:',j,i,sprm.h[j*n+i], intlist[j][i])
            c = (c * hm) % sprm.q
        commit.append(c)
    return commit

def choose_A_matrix():
    Amatrix = []
    for j in range(m):
        ajirow=[]
        for i in range(n):
            ajirow.append(randint(1,sprm.q-1))
        sumrow = - sum(ajirow[1:])
        ajirow[0] = sumrow % sprm.q
        Amatrix.append(ajirow)
    return Amatrix

def compose_C_matrix(a, b):
    Cmatrix = []
    for j in range(m):
        cjirow = []
        for i in range(n):
            if b[j][i] == 0:
                cji = a[j][i]
            if b[j][i] == 1:
                cji = -a[j][i]
            #cji = cji % (sprm.q-1)
            cjirow.append(cji) #aji(1-2bji)
        Cmatrix.append(cjirow)
    return Cmatrix

def compose_D_matrix(a):
    Dmatrix = []
    for j in range(m):
        djirow = []
        for i in range(n):
            dji = - a[j][i] * a[j][i]
            #dji = dji % (sprm.q-1)
            djirow.append(dji)
        Dmatrix.append(djirow)
    return Dmatrix

def compose_f(Amatrix, Bmatrix, x):
    f = []
    for j in range(m):
        fjirow = []
        for i in range(n):
            if Bmatrix[j][i]  == 0:
                fji = Amatrix[j][i]
            if Bmatrix[j][i]  == 1:
                fji =  x + Amatrix[j][i] # f(j,i) = b(j,i)*x + a(j,i)
                #fji = fji % (sprm.q-1)
            fjirow.append(fji) 
        f.append(fjirow)
    return f

def commit_int_list(intlist, r):
    c = pow(sprm.g, r, sprm.q)
    for idx in range(len(intlist)):
        if (intlist[idx] >= sprm.q):
            print('m to commit not in Zq')
            ex = intlist[idx] % (sprm.q-1)
        else:
            ex = intlist[idx]
        hm = pow(sprm.h[idx], ex, sprm.q)
        c = (c * hm) % sprm.q
    return c

from sympy.abc import x as symx
# get coefficient p_{i,k} as in equation (1)
def coefficient_pik(dlji,aji):
    #symx = sympy.symbols('x')
    pi_coeff = []
    for i in range(n**m):
        ij = nary_encoding(i)
        item_x = []
        pi_poly = 1
        for j in range(m):
            i_j = ij[j]
            item = sympy.poly(dlji[j][i_j] * symx + aji[j][i_j], symx) # dlji[j][ij[j]] * x + a[j][ij[j]]
            #print(item, dlji[j][i_j], aji[j][i_j])
            pi_poly=pi_poly*item
        #print(pi_poly)
        coeff_pi = sympy.poly(pi_poly).all_coeffs()
        int_coeff = []
        for coeff in coeff_pi:
            int_coeff.append(int(coeff))
        #print(int_coeff)
        pad_zeros = []
        n_zeros = m + 1 - len(int_coeff)
        for _ in range(n_zeros):
            pad_zeros.append(0)
        for v in int_coeff:
            pad_zeros.append(v)
        pad_zeros.reverse()    
            
        pi_coeff.append(pad_zeros)
    print('pik',pi_coeff)
    return pi_coeff

def pik_correctness(Fmatrix, dlji, pik, aji, x):
    ''' this section checks pik is correctly calculated: product of f for each j equals to the evaluation of pi(x) at x.'''

    for i in range(n**m):
        pf = 1 # product of fiji
        ij = nary_encoding(i)
        for j in range(m):
            pf *= Fmatrix[j][ij[j]]
            #print('fjij',i,j,ij[j],f[j][ij[j]])
        pf = pf % sprm.q
        pix = 0
        for k in range(m+1):
            pix += pik[i][k] * pow(x, k, sprm.q)
            pix = pix % sprm.q
            #print(ij,k,pik[i][k])
        pif = 1
        for j in range(m):
            pif *= dlji[j][ij[j]] * x + aji[j][ij[j]]
            pif = pif % sprm.q
    print('correctness of pik:', pix, pf, pif)
    if not (pix == pf and pix == pif and pif == pf):
        raise ValueError('pik incorrect') # TODO it's always pf that's wrong; something wrong with Fmatrix then
        
def compose_Gk(c,pik,ek,rk):
    enc1, encek = ElGamal_enc(ek,1,rk)
    Gk = []
    for k in range(m):
        gk = encek
        for idx in range(n**m):
            gki = pow(c[idx], pik[idx][k], sprm.q) 
            gk = gk * gki 
            gk = gk % sprm.q
            #print('composing G',idx, k, pik[idx][k],gki)
        Gk.append(gk)
    return Gk

def FStransCha(msg):
    x = int(hashlib.sha256(msg).hexdigest() ,16)
    #print(x)
    x = x % sprm.q
    print('FS trans x:',x)
    return x


# start of protocol - Prover
def R1Prover(msg,dlji):
    Amatrix = choose_A_matrix()

    Cmatrix = compose_C_matrix(Amatrix, dlji)
    Dmatrix = compose_D_matrix(Amatrix)
    
    print('aji:',Amatrix)
    print('dlji:',dlji)
    print(Cmatrix)
    print(Dmatrix)

    rA = randint(1,sprm.q-1)
    rB = randint(1,sprm.q-1)
    rC = randint(1,sprm.q-1)
    rD = randint(1,sprm.q-1)
    print('r for commits R1:',rA,rB,rC,rD)

    CA = commit_int_2Dmnlist(Amatrix,rA)
    CB = commit_int_2Dmnlist(dlji,rB)
    CC = commit_int_2Dmnlist(Cmatrix,rC)
    CD = commit_int_2Dmnlist(Dmatrix,rD)
    print('CA', CA, 'CB', CB, 'CC', CC, 'CD', CD)

    AllCommitment = bytes()
    for com in [CA,CB,CC,CD]:
        for item in com:
            AllCommitment += item.to_bytes(int2byte_size, byteorder='big',signed=True)
    AllCommitment += msg
    x = FStransCha(AllCommitment)
    print('prover x:',x)
    Fmatrix = compose_f(Amatrix, dlji, x)
    print('fmatrix',Fmatrix)
    zA = (rB * x + rA) % (sprm.q-1)
    zC = (rC * x + rD) % (sprm.q-1)
    #print(zA, zC)

    return CA,CB,CC,CD, Fmatrix, zA, zC, Amatrix, x

def R1Verifier(CA,CB,CC,CD,Fmatrix,zA,zC,msg):
    AllCommitment = bytes()
    for com in [CA,CB,CC,CD]:
        for item in com:
            AllCommitment += item.to_bytes(int2byte_size, byteorder='big',signed=True)
    
    AllCommitment += msg
    x = FStransCha(AllCommitment)
 
    print('verifier x:',x)
    f = Fmatrix # it is provable that with fj0 = x - sum(fj[1:]), f = Fmatrix as calculated with fji = bji*x + aji

    com_fzA = commit_int_2Dmnlist(f,zA)

    f_for_CxD = []
    for j in range(m):
        fjirow = []
        for i in range(n):
            fji = f[j][i] * ((x - f[j][i])) # f(j,i)*(x-f(j,i))
            fji = fji % (sprm.q-1)
            fjirow.append(fji) 
        f_for_CxD.append(fjirow)
    print('f(x-f):',f_for_CxD)

    com_fzC = commit_int_2Dmnlist(f_for_CxD,zC)

    #print(zA, zC, com_fzA,com_fzC)
    R1flag = True
    for j in range(m):
        BxA = pow(CB[j], x, sprm.q) 
        BxA = (BxA * CA[j]) % sprm.q
        CxD = pow(CC[j], x, sprm.q)
        CxD = (CxD * CD[j]) % sprm.q
        print('R1: BxA ?=commit(f;zA), row:',j,BxA, com_fzA[j])
        print('R1: CxD ?=commit(f;zC), row:',j,CxD, com_fzC[j])
        if not (BxA == com_fzA[j]):
            R1flag = False
        if not (CxD == com_fzC[j]):
            R1flag = False
        if not R1flag:
            raise ValueError('R1 failed')
    return f
    
def fxf(a,b,x,zC):
    fxf = []
    for j in range(m):
        fjirow = []
        for i in range(n):

            if b[j][i] == 0:
                fji = a[j][i] * x - a[j][i] **2
            if b[j][i] == 1:
                fji = -a[j][i] **2 -  a[j][i] * x
            #fji = fji % (sprm.q -1)
            fjirow.append(fji)
        fxf.append(fjirow)
    print('f(x-f) alt:', fxf)
    com_fzC = commit_int_2Dmnlist(fxf,zC)
    print(com_fzC)
    return fxf


def form_ring():
    ring_member_keys =[]
    for k in range(n**m):
        rmsk, rmpk = generate_key_for_PKE()
        ring_member_keys.append([rmsk,rmpk])
    return ring_member_keys

def signing(msg,ek):
    s = randint(1,sprm.q-1)
    t = randint(1,sprm.q-1)
    ra = randint(1,sprm.q-1)
    rb = randint(1,sprm.q-1)
    print('l for this round:', l)
    dlji = encoding_l(l)
    ringkeys = form_ring()
    print('ring member sk/pk:',ringkeys)
    vkl = ringkeys[l][1]
    skl = ringkeys[l][0]
    d1, d = ElGamal_enc(ek, vkl,t) # d = Enc_ek(vk = g^sk,t)
    #print(vkl, d, ElGamal_dec(opener_sk,d1,d))
    ciph = []
    enc11, enc1ek = ElGamal_enc(ek,1,t)
    #print('enc of', ElGamal_dec(opener_sk,enc11,enc1ek), 'with ek', ek,' and t',t,':',enc1ek)
    for i in range(n**m):
        vki = ringkeys[i][1]
        invvk = pow(vki, -1, sprm.q) # vki^(-1) to be encrypted in ciph
        ciphc1, enc_invvki = ElGamal_enc(ek,invvk,0) # Enc_ek(vki^(-1),0) 
        c = d * enc_invvki % sprm.q
  
        ciph.append(c)
        #print('c1',ciphc1)
        #print(vki,invvk, ElGamal_dec(opener_sk,ciphc1,enc_invvki))
        #print(i, vki,invvk, c, enc_invvki)#, pow(sprm.g, vki, sprm.q), pow(sprm.g, invvk, sprm.q), pow(sprm.g, vki, sprm.q)*pow(sprm.g, invvk, sprm.q) % sprm.q)
    print('cipher:',ciph, 'ciph[l]?=enc1ek', ciph[l]==enc1ek)

    cvkr = randint(1,sprm.q-1)
    gcvkr,c_vk = ElGamal_enc(ek,vkl,cvkr) # this is necessary
    print('c=Enc(ek,vk;r):',gcvkr,c_vk)
    
    rouk = randint(1,sprm.q-1)
    CA,CB,CC,CD, Fmatrix, zA, zC, aji, x = R1Prover(msg,dlji)

    #fxf_alt = fxf(aji,dlji,x,zC)
    pik = coefficient_pik(dlji,aji)
    G = compose_Gk(ciph,pik, ek, rouk)
    z = t * pow(x, m)
    for k in range(m):
        z = z - rouk * pow(x, k)
    z = z % (sprm.q-1)
    A1, A = ElGamal_enc(ek, pow(sprm.g, s, sprm.q), ra) # A = enc_pk(g^s; ra)
    B1, B= ElGamal_enc(ek, pow(sprm.g, s, sprm.q), rb) # B = enc_pk(g^s; rb)
    zs = (skl * x + s) % (sprm.q-1) # zs =sk*x+s
    za = (cvkr * x + ra) % (sprm.q-1) # za = r*x + ra; 
    zb = (t*x + rb) % (sprm.q-1) # zb = t*x+rb
    print('prover z',z,G)
    pik_correctness(Fmatrix, dlji, pik, aji, x)
    return CA,CB,CC,CD, Fmatrix, zA, zC, G, z, ciph, zs, za, zb, d, A, B, gcvkr, c_vk, vkl
    
def SigVerifier(msg, ek, CA,CB,CC,CD, Fmatrix, zA, zC, G, z, ciph, zs, za, zb, d, A, B, c_vk):
    AllCommitment = bytes()
    for com in [CA,CB,CC,CD]:
        for item in com:
            AllCommitment += item.to_bytes(int2byte_size, byteorder='big',signed=True)
    
    AllCommitment += msg
    x = FStransCha(AllCommitment)
    cxA = pow(c_vk, x, sprm.q) 
    cxA = cxA * A % sprm.q
    dxB = pow(d, x, sprm.q)
    dxB = dxB * B % sprm.q
    gza, enc_pk_za = ElGamal_enc(ek, pow(sprm.g, zs, sprm.q), za)
    gzb, enc_pk_zb = ElGamal_enc(ek, pow(sprm.g, zs, sprm.q), zb)
    print('cxA ?= enc(pk,g^zs,za)',cxA, enc_pk_za)
    print('dxB ?= enc(ek,g^zs,zb)',dxB, enc_pk_zb)
    if not (cxA == enc_pk_za):
        raise ValueError('R3 failed')
    if not (dxB == enc_pk_zb):
        raise ValueError('R3 failed')
    f = R1Verifier(CA,CB,CC,CD,Fmatrix, zA, zC,msg)
    print('verifier z', z, G)

    sumci = 1
    for i in range(n**m): # multiply ciph[0,...,N-1]
        mf = 1
        ij = nary_encoding(i)
        for j in range(m): # multiple f[j][ij[j]] j = 1, ..., m            
            mf *= f[j][ij[j]]
            #print('fjij',i,j,ij[j],f[j][ij[j]])
        mf = mf % (sprm.q-1) # because mf is the exponent, it has the same pow(ciph, mf, q) with mod q-1 or without.
        sumci = sumci * pow(ciph[i], mf, sprm.q)
        #print('ci^f', i, mf, ciph[i], sumci)
        #sumci =  sumci % sprm.q
        
    sumgk = 1
    for k in range(m): #multiple Gk from 0 to m-1
        invxk = pow(x, k, sprm.q)
        gk = pow(G[k], -invxk, sprm.q)
        
        #gk1 = pow(G[k], -pow(x,k,sprm.q), sprm.q)
        sumgk *= gk
        #print('prod Gk^(-x^k)',k, invxk, G[k], gk, sumgk)
    
    leftitem = sumci * sumgk % sprm.q
    gz, rightitem = ElGamal_enc(ek,1,z)
    print('R2: left ?= right',leftitem, rightitem) # this is correct when x is small
    if not (leftitem == rightitem):
        raise ValueError('R2 failed')




def opener_open(dk, ek, u, v):
    # first, we find which ciph is vk
    # this can be found by comparing (ciph^dk)==ek^t, the ciph that makes this equation true is the signer
    # (u,v) which is the encypted signer vk is included in the signature as c, d is another encryption of it. c is not d. In signing, c and d is generated with different r.
    a = randint(1,sprm.q-1)
    A = pow(g, a, sprm.q)
    vk = ElGamal_dec(dk, u, v)
    B = pow(u, a, sprm.q)

    AllCommitment = bytes()

    for item in [A,B,vk]:
        AllCommitment += item.to_bytes(int2byte_size, byteorder='big',signed=True)
    
    AllCommitment += msg
    x = FStransCha(AllCommitment)
    z = (dk*x + a) % (sprm.q-1)
    return vk, A, B, z

def verify_opening(vk, A, B, z, ek, u, v):
    AllCommitment = bytes()
    for item in [A,B,vk]:
        AllCommitment += item.to_bytes(int2byte_size, byteorder='big',signed=True)
    
    AllCommitment += msg
    x = FStransCha(AllCommitment)
    for item in [ek, cvk1, cvk2, vk]:
        if item > sprm.q:
            raise ValueError('opening verification: invalid key or ciphertext: not a group element')
    pkxA = pow(ek, x, sprm.q) * A % sprm.q
    uz = pow(u, z, sprm.q)
    gz= pow(sprm.g, z, sprm.q)
    vvk = v * pow(vk,-1,sprm.q) 
    vvkxB = pow(vvk, x, sprm.q) * B % sprm.q
    print('vk is',vk, 'decrypted from:', u, v)
    print('pkxA ?= gz',pkxA == gz, pkxA, gz)
    print('uz ?= vvkxB',uz == vvkxB, uz, vvkxB)
    if not (pkxA == gz and uz == vvkxB):
        raise ValueError('opening not proved correct.')


msg = b'This is a byte message of length 35'
#print(len(msg))
opener_sk, ek = generate_key_for_PKE()
#print('ek',ek)
CA,CB,CC,CD, Fmatrix, zA, zC, G, z, ciph, zs, za, zb, d, A, B, cvk1, cvk2, vkl = signing(msg,ek)
#print('ek',ek)

SigVerifier(msg, ek, CA,CB,CC,CD, Fmatrix, zA, zC, G, z, ciph, zs, za, zb, d, A, B, cvk2)
print(opener_sk, ek,cvk1, cvk2, 'vk (from signing) is', vkl)
vk = ElGamal_dec(opener_sk, cvk1, cvk2)
print('authority try decrypting:', vk, vk == vkl)
vk, R4A, R4B, R4z = opener_open(opener_sk, ek, cvk1, cvk2)
verify_opening(vk, R4A, R4B, R4z, ek, cvk1, cvk2)