from ctypes import *
_so = cdll.LoadLibrary('./xmkckks_go2py/xmkckks.so')

class _Ldouble(Structure):
    _fields_ = [
        ('data', POINTER(c_double)),
        ('size', c_size_t)
    ]

class _Luint64(Structure):
    _fields_ = [
        ('data', POINTER(c_ulonglong)),
        ('size', c_size_t)
    ]
    
class _ParametersLiteral(Structure):
    _fields_ = [
        ('qi', _Luint64),
        ('pi', _Luint64),

        ('logN', c_int),
        ('logSlots', c_int),

        ('scale', c_double),
        ('sigma', c_double)
    ]
    
class _Poly(Structure):
    _fields_ = [
        ('coeffs', POINTER(_Luint64)),
        ('IsNTT', c_bool),
        ('IsMForm', c_bool),
        ('size', c_size_t)
    ]
    
class _PolyQP(Structure):
    _fields_ = [
        ('Q', POINTER(_Poly)),
        ('P', POINTER(_Poly))
    ]

class _PolyQPPair(Structure):
    _fields_ = [
        ('qp0', _PolyQP),
        ('qp1', _PolyQP)
    ]
    
class _Share(Structure):
    _fields_ = [
        ('data', POINTER(_Poly)),
        ('size', c_size_t)
    ]
    
class _Ciphertext(Structure):
    _fields_ = [
        ('data', POINTER(_Poly)),
        ('size', c_size_t),
        ('idxs', POINTER(c_int)),
        ('scale', c_double),
        # ('isNTT', c_bool)
    ]

class _Data(Structure):
    _fields_ = [
        ('data', POINTER(_Ciphertext)),
        ('size', c_size_t)
    ]

class _MPHEServer(Structure):
    _fields_ = [
        # ('params', _Params),
        ('paramsLiteral', _ParametersLiteral),
        ('crs', _Poly),
        ('sk', _PolyQP),
        ('pk', _PolyQPPair),

        # ('secretKey', _Poly),
        ('data', _Data),
        ('idx', c_int),
    ]

_newMPHEServer = _so.newMPHEServer
_newMPHEServer.restype = POINTER(_MPHEServer)

_encryptFromPk = _so.encryptFromPk
_encryptFromPk.argtypes = [ POINTER(_PolyQPPair), POINTER(c_double), c_size_t, c_int]
_encryptFromPk.restype = POINTER(_Ciphertext)

_partialDecrypt = _so.partialDecrypt
_partialDecrypt.argtypes = [ POINTER(_PolyQP), POINTER(_Ciphertext), c_int]
_partialDecrypt.restype = POINTER(_Ciphertext)

_ringQAddLvl = _so.ringQAddLvl
_ringQAddLvl.argtypes = [ POINTER(_Ciphertext), c_int, POINTER(_Ciphertext), c_int]
_ringQAddLvl.restype = POINTER(_Ciphertext)

_decodeAfterPartialDecrypt = _so.decodeAfterPartialDecrypt
_decodeAfterPartialDecrypt.argtypes = [ POINTER(_Ciphertext) ]
_decodeAfterPartialDecrypt.restype = POINTER(_Ldouble)

_addCTs = _so.addCTs
_addCTs.argtypes = [ POINTER(_Ciphertext), POINTER(_Ciphertext)]
_addCTs.restype = POINTER(_Ciphertext)

_multiplyCTConst = _so.multiplyCTConst
_multiplyCTConst.argtypes = [ POINTER(_Ciphertext), c_double]
_multiplyCTConst.restype = POINTER(_Ciphertext)

_addRingPs = _so.addRingPs
_addRingPs.argtypes = [ POINTER(_Poly), POINTER(_Poly)]
_addRingPs.restype = POINTER(_Poly)

# ----------------

### Wrapper Classes (pickle-able) ###

class ParametersLiteral:
    def __init__(self, _paramsLiteral):
        self.qi = _Conversion.from_luint64(_paramsLiteral.qi)
        self.pi = _Conversion.from_luint64(_paramsLiteral.pi)
        self.logN = _paramsLiteral.logN
        self.logSlots = _paramsLiteral.logSlots
        self.scale = _paramsLiteral.scale
        self.sigma = _paramsLiteral.sigma
    
    # So we can send to Lattigo
    def make_structure(self):
        _paramsLiteral = _ParametersLiteral()
        
        _paramsLiteral.qi = _Conversion.to_luint64(self.qi)
        _paramsLiteral.pi = _Conversion.to_luint64(self.pi)
        _paramsLiteral.logN = self.logN
        _paramsLiteral.logSlots = self.logSlots
        _paramsLiteral.scale = self.scale
        _paramsLiteral.sigma = self.sigma

        return _paramsLiteral

# use self.data instead of value used in go to be compatible with helper func to_list_with_conv() 
class Ciphertext:
    def __init__(self, _ct):
        self.data = [ None ] * _ct.size
        self.idxs = [ None ] * _ct.size

        for i in range(_ct.size):
            # self.data[i] = _Conversion.from_poly(_ct.data[i])
            self.data[i] = Poly(_ct.data[i])
            self.idxs[i] = _ct.idxs[i]
        self.scale = _ct.scale
        # self.idx = _ct.idx
        # self.isNTT = _ct.isNTT
    
    # So we can send to Lattigo
    def make_structure(self):
        _ct = _Ciphertext()

        data = [ None ] * len(self.data)
        idxs = [ None ] * len(self.idxs)
        for i in range(len(self.data)):
            data[i] = self.data[i].make_structure()
            idxs[i] = self.idxs[i]

        _ct.size = len(data)
        _ct.data = (_Poly * _ct.size)(*data)
        _ct.scale = self.scale
        _ct.idxs = (c_int * _ct.size)(*idxs)
        # _ct.idx = self.idx
        # _ct.isNTT = self.isNTT

        return _ct

class Poly:
    def __init__(self, _poly):
        self.coeffs = [ None ] * _poly.size
        
        for i in range(_poly.size):
            self.coeffs[i] = _Conversion.from_luint64(_poly.coeffs[i])
        
        self.IsNTT = _poly.IsNTT
        self.IsMForm = _poly.IsMForm
    
    # So we can send to Lattigo
    def make_structure(self):
        _poly = _Poly()

        coeffs = [ None ] * len(self.coeffs)
        
        for i in range(len(self.coeffs)):
            coeffs[i] = _Conversion.to_luint64(self.coeffs[i])
        
        _poly.size = len(coeffs)
        _poly.coeffs = (_Luint64 * _poly.size)(*coeffs)
        _poly.IsNTT = self.IsNTT
        _poly.IsMForm = self.IsMForm

        return _poly

class PolyQP:
    def __init__(self, _polyQP):
        self.Q = Poly(_polyQP.Q.contents)
        self.P = Poly(_polyQP.P.contents)
    
    # So we can send to Lattigo
    def make_structure(self):
        _polyQP = _PolyQP()
        
        _polyQP.Q.contents = self.Q.make_structure()
        _polyQP.P.contents = self.P.make_structure()

        return _polyQP

# Server that has Multi-Party Homomorphic Encryption functionality
class MPHEServer:
    def __init__(self, server_id):
        _server_ptr = _newMPHEServer(server_id)
        _server = _server_ptr.contents

        self.paramsLiteral = ParametersLiteral(_server.paramsLiteral) # implemented but currently not used, security parameters hardcoded in export.go
        self.sk = PolyQP(_server.sk)
        self.pk = _Conversion.from_polyQPpair(_server.pk)
        # self.sk = _Conversion.from_polyQP(_server.sk)
        # self.pk = _Conversion.from_polyQPpair(_server.pk)
        # self.crs = _Conversion.from_poly(_server.crs)
        # self.secret_key = _Conversion.from_poly(_server.secretKey)
        # self.data = []  # NOTE: always have this as decryptable by secret_key
        self.idx = _server.idx
    
    def encryptFromPk(self, data):
        # params = self.params.make_structure()
        # sk = _Conversion.to_poly(self.secret_key)
        pk = _Conversion.to_polyQPpair(self.pk)

        data_ptr = (c_double * len(data))(*data)
        enc_ct = _encryptFromPk(byref(pk), data_ptr, len(data), self.idx)

        # self.data = _Conversion.from_data(enc_ct.contents)
        self.data = Ciphertext(enc_ct.contents)

        return self.data
    
    def partialDecrypt(self, ciphertext):
        # params = self.params.make_structure()
        sk = self.sk.make_structure()
        # ct = _Conversion.to_data(self.data)
        ct = ciphertext.make_structure()

        partial_dec_ct = _partialDecrypt(byref(sk), byref(ct), self.idx)
        # dec_data = _Conversion.to_list(dec_data.contents)

        return Ciphertext(partial_dec_ct.contents)

    def ringAddLvl(self, ct1, ct1_idx, ct2, ct2_idx):
        op1 = ct1.make_structure()
        op2 = ct2.make_structure()
        op1 = _ringQAddLvl(op1, ct1_idx, op2, ct2_idx)

        return Ciphertext(op1.contents)

    def aggregate_pds(self, ct_pd_list, client_ids):
        # Aggregate partially decrypted ciphertexts, requires to be called by the server.
        # pd_list: a list of partilally decrypted ciphertexts
        # ct_pd_list: a list of client ids involved in the encryption
        # Return: ct_pd_agg is the decrytped ciphertext, 
        # which can be sent for decryption and decoding by calling server.decodeAfterPartialDecrypt(ct_pd_agg)
    
        # size mismatch of partially decrypted ciphertexts and client ids involved in the encryption
        if len(ct_pd_list) != len(client_ids):
            raise Exception("aggregate_pds(): ct_pd_list has a length of " + str(len(ct_pd_list)) + 
                            ", but client_ids has a length of " + str(len(client_ids)))
        
        # empty list of ciphertexts
        if len(ct_pd_list) == 0:
            raise Exception("aggregate_pds(): len(ct_pd_list) is 0.")
        else:
            # add polynomial ring on Ciphertext["0"] and Ciphertext["client_id"]
            ct_pd_agg = self.ringAddLvl(ct_pd_list[0], 0, ct_pd_list[0], client_ids[0])
            if len(ct_pd_list) > 1:
                for ct_id in range(1, len(ct_pd_list)):
                    ct_pd_agg = self.ringAddLvl(ct_pd_agg, 0, ct_pd_list[ct_id], client_ids[ct_id])
            return ct_pd_agg
    
    def addRingPs(self, rP1, rP2):
        ringP1 = rP1.make_structure()
        ringP2 = rP2.make_structure()
        sumRingP = _addRingPs(ringP1, ringP2)
        return Poly(sumRingP.contents)
    
    def aggregate_ringPs(self, ringP_list):
        # Generate the aggregated public key (the P ring in the public key) based on xMKCKKS (https://arxiv.org/abs/2104.06824)
        # pring_list: a list of P rings in the public keys' first QPRings, access by client_1.pk[0].P
        # Return: sum_ringP, a Poly ring that to be shared by all clients
        if len(ringP_list) == 0 or len(ringP_list) == 1:
            raise Exception("aggregate_ringPs(): ringP_list has a length of " + str(len(ringP_list)))
        else:
            sum_ringP = self.addRingPs(ringP_list[0], ringP_list[1])
            if len(ringP_list) > 2:
                for pk_id in range(2, len(ringP_list)):
                    self.addRingPs(sum_ringP, ringP_list[pk_id])
            return sum_ringP

    def decodeAfterPartialDecrypt(self, ciphertext):
        ct = ciphertext.make_structure()
        res = _decodeAfterPartialDecrypt(ct)
        return _Conversion.from_ldouble(res.contents)

    def addCTs(self, ct1, ct2):
        op1 = ct1.make_structure()
        op2 = ct2.make_structure()
        res = _addCTs(op1, op2)
        return Ciphertext(res.contents)

    def multiplyCTConst(self, ct1, const):
        op1 = ct1.make_structure()
        res = _multiplyCTConst(op1, const)
        return Ciphertext(res.contents)

# Performs conversion between Structures (which contain pointers) to pickle-able classes
class _Conversion:
    # (FYI) Convert to numpy array: https://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy

    # Generic array type Structure to list

    def to_list(_l):
        l = [ None ] * _l.size

        for i in range(_l.size):
            l[i] = _l.data[i]
        
        return l

    def to_list_with_conv(_l, conv):
        l = [ None ] * _l.size

        for i in range(_l.size):
            l[i] = conv(_l.data[i])
        
        return l

    def to_ptr(l, conv, t):
        lt = [ None ] * len(l)

        for i in range(len(l)):
            lt[i] = conv(l[i])
        
        return (t * len(lt))(*lt)

    ### _Luint64 (list of uint64)

    def from_luint64(_luint64):
        return _Conversion.to_list(_luint64)

    def to_luint64(l):
        luint64 = _Luint64()

        luint64.size = len(l)
        luint64.data = (c_ulonglong * luint64.size)(*l)

        return luint64

    ### _Ldouble (list of double)

    def from_ldouble(_ldouble):
        return _Conversion.to_list(_ldouble)

    def to_ldouble(l):
        ldouble = _Ldouble()

        ldouble.size = len(l)
        ldouble.data = (c_ulonglong * ldouble.size)(*l)

        return _ldouble

    # _PolyPair (list[2] of Poly)
    
    def from_polyQPpair(_qpp):
        qpp = [ None ] * 2

        qpp[0] = PolyQP(_qpp.qp0)
        qpp[1] = PolyQP(_qpp.qp1)
        
        return qpp

    def to_polyQPpair(qpp):        
        _qpp = _PolyQPPair()

        if len(qpp) != 2:
            print('ERROR: Only a list of size 2 makes a pair (not {})'.format(len(qpp)))
            return None

        _qpp.qp0 = qpp[0].make_structure()
        _qpp.qp1 = qpp[1].make_structure()

        return _qpp

    ### _Share (list of Poly)

    def from_share(_share):        
        return _Conversion.to_list_with_conv(_share, _Conversion.from_poly)

    def to_share(share):
        list_poly = [ None ] * len(share)

        for i in range(len(share)):
            list_poly[i] = _Conversion.to_poly(share[i])
        
        _share = _Share()
        _share.size = len(list_poly)
        _share.data = (_Poly * _share.size)(*list_poly)

        return _share

    ### _Data (list of Ciphertext)

    def from_data(_data):
        return _Conversion.to_list_with_conv(_data, Ciphertext)
    
    def to_data(data):
        list_ciphertext = [ None ] * len(data)

        for i in range(len(data)):
            list_ciphertext[i] = data[i].make_structure()
        
        _data = _Data()
        _data.size = len(list_ciphertext)
        _data.data = (_Ciphertext * _data.size)(*list_ciphertext)

        return _data