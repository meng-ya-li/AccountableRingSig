# AccRingSignatureDDH

The correctly working py now are:
- R1Protocol.py (but haven't changed a,b,c,d in Zq)
- R1Protocol.py (but mod is applied in a,b,c,d,f; it is better to apply it in commitment function; see in anotherR3 commitment function)
- anotherR3.py
- correctR3.py
- timingAccRS.py (working with correctR3 for timing)

proof of correctness of the zkp protocols

### R1
Com is the author-defined pedersen commitment scheme to be applied on a sequence, $m_i$, and commitment key ck is a sequence as long as the sequence to be committed.
$Com(ck, {m_i};r) = g^r \cdot (h_i^{m_i})$  where $ck = \{h_i\}$   

the 2-D matrix a, b, c, d, f needs to be commited row-by-row; in the proof, the subscript is treated with only 1 dimension, and that means a row as a one-dimensional sequence.
$A = Com(ck, \{a_i\};r_A)$, also: $\{a_i\}$ is constructed so that $\sum(a_i) = 0$ 
$B = Com(ck, \{b_i\};r_B)$
$C = Com(ck, \{c_i\};r_C)$, $c_i = a_i$ when $b_i = 0$, and  $c_i = -a_i$  when $b_i = 1$ 
*coding: in exp modulo, g^(-a) mod p = g^(-a mod (p-1)) mod p*
$D = Com(ck, \{d_i\};r_D)$, $d_i = -a_i^2$ 
$f_i =b_ix+a_i$,which is $f_i= x+a_i$ wyhen when $b_i = 1$, and  $f_i = a_i$  when $b_i = 0$ 
$z_A = r_B x + r_A$
$z_C = r_C x + r_D$

Verify: 
$B^xA == Com(ck,f_i;z_A)$   
$C^xD == Com(ck,fxf_i;z_C)$    

$B^xA = (g^{r_B} \cdot h_i^{b_i}) ^x \cdot (g^{r_A} \cdot h_i^{a_i}) = g^{({r_B} \cdot x + {r_A})} \cdot h_i^{b_ix+a_i}$  
$Com(ck,f_i;z_A) = g^{z_A} \cdot h_i^{f_i} = g^{(r_B \cdot x + r_A)} \cdot h_i^{b_ix+a_i}$ 

$C^xD = (g^{r_C} \cdot h_i^{c_i}) ^x \cdot (g^{r_D} \cdot h_i^{d_i}) = (g^{r_C x + r_D})  \cdot ( h_i^{{c_i} x + {d_i}} )$  
		= $(g^{r_C x + r_D})  \cdot ( h_i ^ {(a_i - 2 a_i b_i)x - a_i^2} )$  = $(g^{r_C x + r_D})  \cdot ( h_i ^ {(a_i x) - ( 2 a_i b_i x - a_i^2)} )$ 

$Com(ck,fxf_i;z_C) = g^{z_C} \cdot (h_i ^ {f_i (x - f_i )}) = g^{z_C} \cdot (h_i ^ {(b_i x + a_i) (x - (b_i x + a_i) )})$  = $(g^{r_C x + r_D})  \cdot ( h_i ^ {b_i(1-b_i)x^2 + (a_i - 2a_i b_i)x - (a_i)^2} )$ 
because $b_i$ is either 0 or 1, the $x^2$ coefficient is always 0. the rest of the order $(a_i - 2a_i b_i)x - (a_i)^2$ is $a_i(1-2b_i) x + (-a_i^2)$, which is directly $c_i x + d_i$ 
*coding: if anything is to be an order in the commitment, it should be summed without any modulo, then after summing everything, do modulo p-1*
*coding: just do the mod p-1 in the commitment function*

### R2
To prove there is an encryption of 1 in a sequence of ciphertext. Encryption is based on ElGamal: 
$c = Enc(ek, m) = (ek^r, g^r m)$ = ${g^{dk \cdot r}, g^r m}$
$m = Dec(dk, c = (u,v)) = v \cdot u ^{-(dk^{-1})}$  = $g^r m (g^{dk \cdot r}) ^{(-1)\cdot{(1/dk)}}$   = $g^r m g^{-r}$  = m
This is not the standard ElGamal.



$\{c_i\}$ is a sequence of ciphertext, where $c_l$ is the encryption of 1: $c_l = Enc(ek, 1;r)$ = $(ek^r, g^r)$
$\delta_{l,i}$ is used to index the ciphertext of 1. $\delta_{l,i} = \prod_{j=0}^{m-1} \delta_{l_j,i_j}$  = 1 when $i=l$ and 0 when $i \neq l$ 
$l$ is represented as a n-ary matrix consisting of element $l_j$, $l = \sum_{j=0}^{m-1} l_j n^j$ ,  and each $i$ in the subscript of the ciphertext sequence $c_i$ can also be represented in n-nary form, with $i_j$ calculated in the same way as $l_j$ 
$\delta_{l_j,i_j}$ is the b-matrix to be proved with R1.

In addition to f in R1, the new thing here is $G_k$, constructed with a polynomial, $p_i(x) = \prod_{j=0}^{m-1} \delta_{l_j, i_j} x + a_{l_j, i_j}$  (1)
note the subscript was changed from what is written in the paper. The mapping is done by interpreting the design: R1 is used to prove B/or delta here is a number correctly encoded in the 2-D matrix. The $f_i$ is generated in the same way as in R1, so we map $\delta$ to $b$ in R1 and $a$ is still $a$, and $(l_j, i_j)$ is $(j,i)$ in R1. 
$p_i(x) = \delta_{l,i} x^m + \prod_{k=0}^{m-1} p_{i,k} x^k$ (2)
(1) = (2), so we can expand (1) and calculate the $coefficient p_{i,k}$, using $\delta_{l_j, i_j}$ and $a_{l_j, i_j}$.

$G_k$ is computed per order of x / per $x^k$, over all $c_i$ (there are in total N $c_i$)
$G_k = \prod_{i=0}^{N-1} c_i^{p_{i,k}} \cdot Enc(ek, 1; r_k)$  

#### Correctness
note that
$\prod_{i=0}^{N-1} c_i^{\prod_{j=0}^{m-1} f_{l_j,i_j}} \cdot \prod_{k=0}^{m-1} G_k^{-x^k}$ = $(\prod_{i=0}^{N-1} c_i^{\delta_{l,i}}) ^ {x^m}$ = $(c_l)^{x^m}$   (3)

Verifier verifies by:
$\prod_{i=0}^{N-1} c_i^{p_i(x)} \cdot \prod_{k=0}^{m-1} G_k^{-x^k}$ = $Enc(ek, 1;z)$ 

because $\prod_{j=0}^{m-1} f_{l_j,i_j} = p_i(x)$ , the left side is 
$\prod_{i=0}^{N-1} c_i^{p_i(x)} \cdot \prod_{k=0}^{m-1} G_k^{-x^k}$ = $\prod_{i=0}^{N-1} c_i^{p_i(x)} \cdot (\prod_{k=0}^{m-1} \prod_{i=0}^{N-1}c_i^{p_{i,k}} \cdot Enc(ek, 1; r_k))^{-x^k}$ 
= $\prod_{i=0}^{N-1} c_i^{p_i(x)}  \cdot \prod_{k=0}^{m-1} \prod_{i=0}^{N-1}c_i^{-p_{i,k}x^k} \cdot \prod_{k=0}^{m-1} Enc(ek, 1; r_k)^{-x^k}$  
=$(\prod_{i=0}^{N-1} c_i^{p_i(x)} \cdot \prod_{i=0}^{N-1} c_i^{-\sum_{k=0}^{m-1}  p_{i,k}x^k}) \cdot Enc(ek, 1; -\sum_{k=0}^{m-1} (x^kr_k))$ 
= $(c_l)^{x^m} \cdot Enc(ek, 1; -\sum_{k=0}^{m-1} (x^kr_k))$  (using (3))
=$Enc(ek, 1; r\cdot x^m) Enc(ek, 1; -\sum_{k=0}^{m-1} (x^kr_k))$
=$Enc(ek, 1;z)$
where z = $r x^m - \sum_{k=0}^{m-1} (x^kr_k)$ 

*coding: do not do any modulo before final step:
in z: do not do modulo in $x^k$ or the multiplication, do mod q-1 after the subtration
do not do mod when calculating the polynomial coefficient
do mod in Gk is fine.*
