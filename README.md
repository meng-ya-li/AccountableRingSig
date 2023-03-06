# Accountable Ring Signature based on DDH

a package of python code implementing the signature scheme proposed in the [Bootle 2015](https://eprint.iacr.org/2015/643) paper.

## Introduction

Accountable Ring Signature (ARS) is a signature scheme that:
- achieves k-anonymity through aggregation: signer is hidden in the members forming a ring of size k. Unpriviledged entities are unable to distinguish the signer from the other ring members with a probability higher than 1/k.
- achieves conditional anonymity: signer can be identified by an opener, based on the signature. The opener is designated at the time of signing by the signer.
- provides flexibility: the ring is formed at the time of the signing, and formed by the signer. This allows the signer to control the ring size and the ring members, hence the anonymity level it has.
- is zero-knowledge: signature is produced with zero-knowledge proof. The interactive zkp in the original paper is transformed into non-interactive zkp for the purpose of producing signatures, using Fiat-Shamir transformation.

In this project, the four protocols given in the original paper are implemented with Python. Certain modifications are made on the original design to make the coding feasible. The modified design has been analysed and proved so that it achieves the same security properties as in the original design.


## Architecture
There are six main functions in ARS scheme:

- system parameter initialisation: this function chooses system parameters, such as key length, prime ```p``` for the group of prime order, geneartor ```g``` of the group.
- key generation, ```keygen(sk,vk)```: this function generates a pair of public key and private key for any entities in the network.
- signing, ```ARSsign(sk, m)```: this function generates the signature on message ```m``` with the signer's private key ```sk```. It also forms the ring and chooses the opener. Its outputs are the ring ```R```, which is represented with the public keys of the ring members, the opener (represented by the opener's public key, ```dk```), and the signature ```sig```.
- verification, ```ARSverify(R,m,sig,dk)```: this function verifies the signature. It should verify that: the signature is generated on the message by one of the ring memebers; the signature can be opened by the opener. It outputs true or false.
- opening, ```ARSopen(ek, dk, sig)```: this function reveals the identity of the signer with the opener's private key ```ek```. It outputs the signer's public key ```vk``` and gives a NIZK proof of the opening results, ```P```, so that anyone can verify whether the given ```vk``` is actually the public key of the signer of this signature.
- opening verification, ```ARSopenverify(vk, sig, P)```: this function verifies the opening is correct, using the opened signer's public key ```vk```, the signature, and the proof ```P```. It outputs true or false.

To clarify, the notations of keys in this document are:
- ```sk, vk```: private key and public key of the signer. As the scheme is based on DDH, we have: ```vk = g^sk mod p```.
- ```ek, dk```: private key and pubic key of the opener. ```ek = g^dk mod p```.
- Ring ```R``` represented with a list of public keys: ```R = {vk_0, vk_1, ..., vk_r}``` for a ring of *r* members.


## Running the Code
Two Python scripts are provided. 

### ```R4protocol.py``` 
```R4protocol.py``` contains all the code needed for using the ARS scheme. At first, some global parameters are initialised, including prime ```p```, generator ```g```, ring size ```n^m```, and signer index ```l```.
The main code that calls the functions for key generation, signing, verification, opening, and opening verification is at the end of the script.

If a message is modified after signing, the code will raise a value error, saying the relation failed to be proven correctly. If ```R4Protocol.py``` runs without error, the signature verification is successful.

The whole signature, using the variable names in the code, is ```ek, CA,CB,CC,CD, Fmatrix, zA, zC, G, z, ciph, zs, za, zb, d, A, B, cvk2```. 
### ```R4simulator.py``` 
This is the simulator to prove the modified zkp protocol for R4 in this code is zero-knowledge. It is not needed during the signing and verification.

## The Main Ideas
The main ideas of how to faciliate a signature scheme that achieves the properties as said in Intro are:
- To enable opening, signer should send its public key encrypted with the opener's public key. This is for the opener to be able to reveal the signer's public key. This encryption ```c``` will be included in the signature. ```c = Enc(ek, vk)```.
- To prove the signature contains the encryption ```c```, three relations need to be proved.
  - R1: there is a commitment of 1 and exactly one commitment of 1 in a list of commitments of 0s and 1s.
  - R2: there is an encryption of an 1 and exactly one 1 in a list of ciphertext.
  - R3: a ciphertext ```c``` is an encryption of a public key in ```R```.
  
  Each of the three relations is proved based on the previous one.
- The opener can open the signer's public key easily by decrypting the encryption of ```vk``` in the signature. 
- To prove the opening is correct, the opener generates a zkp to prove R4: the ```vk``` it gives is indeed encrypted in ```c```.
- To prove R1, R2, R3 and R4, a interactive zkp protocol is given for each of them. By using Fiat-Shamir transformation, the zkp protocol for R3 can be transformed into a pair of signature signing and verification algorithms. The message to sign, ```m```, is used to generate the challenge ```x```, together with other items needed for the zkp. If the verifier can verify the signature by generating the challenge ```x``` on their side, using the message ```m``` and other items given by the prover, then the message is authentic and indeed from one of the ring members.

## Implementation Notes
The details of the four zkp protocols can be seen in the Bootle paper. However, there are a few modifications made during the implementation that are worth noticing.
- The ring has a size of N = n^m. The index of the signer in the ring is coded in a m*n matrix, in its n-ary format. In the matrix, each row should have one and exactly one 1, and the rest of the elements should be 0. The commitment scheme, a modified Pedersen commitment, commits a list of numbers into one commitment value, i.e. it processes 1-D matrix. When committing the 2-D matrix, and when doing the verification, the matrix should be processed one row at a time, so that the commitment scheme can work correctly.
- The encryption is done with ElGamal encryption scheme. The paper provided a modified version of ElGamal as its primitive, where the encryption is ```(u,v)=Enc(ek,m;r)=(ek^r, g^r*m)``` and the decryption is ```m = v*u^(-1/dk)```. However, this version has proved to be difficult to implement in Python, because of group exponentiation and modulus operations. In the code, the common version of ElGamal is used, where the encryption is ```(u,v)=Enc(ek,m;r)=(g^r, ek^r*m)```, and the decryption is ```m=(u^(-dk))*v```. This change does not affect R1-R3, but R4 zkp protocol has been changed accordingly. The modified R4 zkp protocol is as zero-knowledge as the original R4 protocol, and a simulator is built to verify that.
- The polynomial $p_i(x)$ does not have to be calculated. What is needed is the coefficients, and the coefficients can be calculated through iteration. A helper function, ```pik_correctness```, is provdied in the code, to check that coefficients are correctly calculated, by confirming the polymonial is evaluated correctly at a given x in all three ways.
- Some minor errors and modifications: 
  - It is proved that ```f``` does not need to be recalculated on the verifier side. Directly using what is given by the verifier has the same effect.
  - Notations used in the paper are somewhat mixed between protocols. To clarify, 
    - every subscript i in R2, such as in $a_{j,i}$, should be $i_j$, where $i_j$ is as the paper defined, the element in the 2-D matric encoding the n-ary form of i. 
    - every time there is a j=0 to m-1 sum or product operation, it should be 0 to m-1, not 0 to m.
    - ```A``` and ```B``` in R3 is not the same as in R1 and R2; however, ```A``` and ```B``` refer to the same items in R1 and R2.
    - ```d``` in R3 is not the same as ```c```. ```d``` is only used for generating the list of ciphertext which contains an encryption of 1, the $\{c_i:i \in [0,N-1]\}$ list. Both the list and ```c``` need to be given to the verifier.
  - The m not in Zq warning that shows up in running the code is because the exponent should be in Zq, and the Pedersen commitment function has a check on that. When the function finds the exponent is not in Zq, it does modulo on the exponent to put it in Zq before it does group exponentiation.

