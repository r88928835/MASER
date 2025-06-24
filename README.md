# MASER

This repository is for the implementation of the paper "MASER: Efficient Privacy-Preserving Cross-Silo Federated Learning with Multi-Key Homomorphic Encryption".

## How to Use
The first step is to generate a dll for our Python-based xMKCKKS library on your machine. Please follow the steps described in the *Instructions for xMKCKKS_Go2Py Library* part.

`config_FL.py`: Place to set experimental parameters.

`datasetsplit.py`: Script for dividing the dataset (MNIST/CIFAR-10) with IID/non-IID distribution (Dirichlet).

* Example: `python3 datasetsplit.py --num_clients 5 --dataset mnist --distribution non_iid --alpha 1.0`

`fl_enc_server.sh`: Script for launching the Federated Learning server using MASER. Always start the server first before launching the clients.
* Example: `sh fl_enc_server.sh`

`fl_enc_client.sh`: Script for launching all Federated Learning clients using MASER.
* Example: `sh fl_enc_client.sh`

## Dependencies:
* python==3.10
* flwr==1.6.0
* pympler == 1.0.1
* torch
* torchvision
* torchaudio
* scikit-learn
* pandas
* grpcio==1.52

## Instructions for xMKCKKS_Go2Py Library
The xMKCKKS_Go2Py folder contains our Python implementation for xMKCKKS, based on the Golang Multi-key CKKS Homomorphic Encryption library. We provide step-by-step instructions for using this library in the Jupyter Notebook file named `tutorial.ipynb`.

### Introduction

This Python code implements the xMKCKKS scheme proposed in "Privacy-preserving Federated Learning based on Multi-key Homomorphic Encryption" (https://arxiv.org/abs/2104.06824).

The Golang code base for MKCKKS is MKHE-KKLSS (https://github.com/SNUCP/MKHE-KKLSS), which is from the paper "Asymptotically Faster Multi-Key Homomorphic Encryption from Homomorphic Gadget Decomposition" (https://eprint.iacr.org/2022/347), published in ACM CCS'23.

This code builds a C-style dynamic link library (dll) from the Golang code using cgo, allowing calling go functions from Python programs using ctypes.

We implement this library to perform MK-HE in the context of Federated Learning (FL), currently this library supports:
- Public & private key generation
- Public key aggregation (for xMKCKKS)
- Encoding & encryption on a list of double data using public key
- Homomorphic addition between 2 ciphertexts
- Homomorphic multiplication between a ciphertext and a constant value
- Partial decryption
- Aggregating the partial decryption results and decoding

Note: Our implementation considers encrypting double data type, you will need to make your own changes to encrypt compelx data type.

### Prerequisites
- Go and cgo
- Python and ctypes

### How to Use

The C-style dll needs to be generated for different platforms, to generate your own dll, open a terminal at the root directory of this project and enter:

`go build -o xmkckks.so -buildmode=c-shared export.go`

This generates a dll named xmkckks.so (for Linux & macOS, change into xmkckks.dll for Windows) from the export.go file. Please refer to the jupyter notebook tutorials.ipynb for details of how to import this dll in Python and perform HE operations. 

## Acknowledgement
MKHE-KKLSS: https://github.com/SNUCP/MKHE-KKLSS

Python Wrapper for Lattigo: https://github.com/chandra-gummaluru/FL-Development/tree/MPHE

xMKCKKS Python Implementation: https://github.com/MetisPrometheus/MSc-thesis-xmkckks
