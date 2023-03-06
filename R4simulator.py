from random import randint

p = 61
g = 2
#g = 14281528880334450723053870702714241313522996716555816136992203461280239035144
#p = 72605870376593568031989838389601275874951467847143895321310129443024924195643
def generate_key_for_PKE():
    sk = randint(1,p-1)
    vk = pow(g, sk, p)
    return sk, vk

def ElGamal_enc(pk,m,r):
    #r = randint(1,p)
    c1 = pow(g, r, p)
    s = pow(pk, r, p)
    c2 = s * m % p
    return c1, c2

def ElGamal_dec(dk, c1, c2):
    
    invs = pow(c1, -dk, p)
    m = c2 * invs % p
    return m

def R4verifier(A,B,x,z,ek,vk,u,v):
    pkxA = pow(ek, x, p) * A % p
    uz = pow(u, z, p)
    gz= pow(g, z, p)
    vvk = v * pow(vk,-1,p) 
    vvkxB = pow(vvk, x, p) * B % p
    print('vk is',vk, 'decrypted from:', u, v)
    print('pkxA ?= gz',pkxA == gz, pkxA, gz)
    print('uz ?= vvkxB',uz == vvkxB, uz, vvkxB)
    if not (pkxA == gz and uz == vvkxB):
        raise ValueError('opening not proved correct.')

def R4simulator():
    x = randint(1, p-1)
    z = randint(1, p-1)
    a = z
    if x == 0: 
        A = pow(g, a, p)
        B = pow(u, a, p)
    
    if x > 0:
        x = randint(1,p-1)
        A = pow(ek, -x, p)  * pow(g, z, p) % p
        vvk = v * pow(dc,-1,p) 
        B = pow(vvk, -x, p)  * pow(u, z, p) % p
    return A,B,x,z

dk, ek = generate_key_for_PKE()
m = randint(1,p)
r = randint(1,p-1)
u,v = ElGamal_enc(ek, m, r)
dc = ElGamal_dec(dk, u, v)
print('decryption correct?',m, dc)
A,B,x,z = R4simulator()
print('simulator A,B,x,z:',A,B,x,z)
R4verifier(A,B,x,z,ek,dc,u,v)