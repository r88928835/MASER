package main

/*
#include <stdbool.h>
#include <stddef.h>
// #include <complex.h>

typedef struct {
	double* data;
	size_t size;
} Ldouble;

typedef struct {
	long long unsigned int* data;
	size_t size;
} Luint64;

// // Message
// typedef struct {
// 	double complex* data;
// 	size_t size;
// } Message;

// Params
typedef struct {
	Luint64 qi;
	Luint64 pi;

    int logN;
	int logSlots;
	int gamma;

	double scale;
	double sigma;
} Params;

// ParametersLiteral
typedef struct {
	Luint64 qi;
	Luint64 pi;

    int logN;
	int logSlots;

	double scale;
	double sigma;
} ParametersLiteral;

// Poly
typedef struct {
	Luint64* coeffs;
	bool IsNTT;
	bool IsMForm;
	size_t size;
} Poly;

// PolyPair
typedef struct {
	Poly p0;
	Poly p1;
} PolyPair;

// PolyQP
typedef struct {
	Poly* Q;
	Poly* P;
} PolyQP;

// PolyQPPair
typedef struct {
	PolyQP qp0;
	PolyQP qp1;
} PolyQPPair;

// Share
typedef struct {
	Poly* data;
	size_t size;
} Share;

// Ciphertext
typedef struct {
	Poly* data;
	size_t size;
	int* idxs;

	double scale;
	// bool isNTT;
} Ciphertext;

// Data
typedef struct {
	Ciphertext* data;
	size_t size;
} Data;

// MPHEServer
typedef struct {
	// Params params;
	ParametersLiteral paramsLiteral;
	Poly crs;
	PolyQP sk;
	PolyQPPair pk;
	Data data;
	int idx;
} MPHEServer;

*/
import "C"
import (
	"fmt"
	"mk-lattigo/mkckks"
	"mk-lattigo/mkrlwe"
	"strconv"
	"unsafe"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

var PN14QP439 = ckks.ParametersLiteral{
	LogN:     14,
	LogSlots: 13,
	Q: []uint64{
		// 59 + 5x52
		0x7ffffffffe70001,

		0xffffffff00001, 0xfffffffe40001,
		0xfffffffe20001, 0xfffffffbe0001,
		0xfffffffa60001,
	},
	P: []uint64{
		// 60 x 2
		0xffffffffffc0001, 0xfffffffff840001,
	},
	Scale: 1 << 52,
	Sigma: rlwe.DefaultSigma,

	// LogN:     15,
	// LogSlots: 14,
	// //60 + 13x54
	// Q: []uint64{
	// 	0xfffffffff6a0001,

	// 	0x3fffffffd60001, 0x3fffffffca0001,
	// 	0x3fffffff6d0001, 0x3fffffff5d0001,
	// 	0x3fffffff550001, 0x3fffffff390001,
	// 	0x3fffffff360001, 0x3fffffff2a0001,
	// 	0x3fffffff000001, 0x3ffffffefa0001,
	// 	0x3ffffffef40001, 0x3ffffffed70001,
	// 	0x3ffffffed30001,
	// },
	// P: []uint64{
	// 	//59 x 2
	// 	0x7ffffffffe70001, 0x7ffffffffe10001,
	// },
	// Scale: 1 << 54,
	// Sigma: rlwe.DefaultSigma,
}

// func Save(path string, object interface{}) error {
// 	file, err := os.Create(path)
// 	if err != nil {
// 		return err
// 	}
// 	defer file.Close()
// 	return gob.NewEncoder(file).Encode(object)
// }

// func Load(path string, object interface{}) error {
// 	file, err := os.Open(path)
// 	if err != nil {
// 		return err
// 	}
// 	defer file.Close()
// 	return gob.NewDecoder(file).Decode(object)
// }

func loadCompactParams() mkckks.Parameters {

	PARAMSLITERAL := &[]ckks.ParametersLiteral{PN14QP439}[0] // hardcoded, assuming using one parameters lietral
	// server.paramsLiteral = *convParamsLiteral(PARAMSLITERAL)

	ckksParams, err := ckks.NewParametersFromLiteral(*PARAMSLITERAL)
	if ckksParams.PCount() < 2 {
		fmt.Printf("ckks Params.PCount < 2")
		// continue
	}

	if err != nil {
		panic(err)
	}

	// PARAMS := mkckks.NewParameters(ckksParams)
	params := mkckks.NewCompactParameters(ckksParams)

	return params
}

// func loadParams() mkckks.Parameters {
// 	// Serialized secure parameters exist on disk
// 	if _, err := os.Stat(secure_params_path); errors.Is(err, os.ErrNotExist) {
// 		fmt.Println("file not exist")
// 		// Load the default ckks parameters
// 		PARAMSLITERAL := &[]ckks.ParametersLiteral{PN14QP439}[0] // hardcoded, assuming using one parameters lietral

// 		ckksParams, err := ckks.NewParametersFromLiteral(*PARAMSLITERAL)

// 		if ckksParams.PCount() < 2 {
// 			fmt.Printf("ckks Params.PCount < 2")
// 			// continue
// 		}

// 		if err != nil {
// 			panic(err)
// 		}

// 		params := mkckks.NewParameters(ckksParams)
// 		fmt.Println("params created")
// 		fmt.Println(params)
// 		params.CRS = nil
// 		// fmt.Println(params.CRS)

// 		// Serialize the secure parameters and save to disk
// 		// json_params, err := json.Marshal(params)
// 		// if err != nil {
// 		// 	fmt.Printf("Error: %s", err)
// 		// 	return params
// 		// }
// 		// fmt.Println(string(json_params))

// 		//----

// 		// var buf bytes.Buffer
// 		// enc := gob.NewEncoder(&buf)
// 		// err = enc.Encode(params)
// 		// if err != nil {
// 		// 	log.Fatal(err)
// 		// }

// 		// // Write to file
// 		// err = os.WriteFile(secure_params_path, buf.Bytes(), 0600)
// 		// if err != nil {
// 		// 	log.Fatal(err)
// 		// }

// 		if err := Save(secure_params_path, params); err != nil {
// 			panic(err)
// 		}

// 		// if err := Save("crs.json", params.CRS); err != nil {
// 		// 	panic(err)
// 		// }
// 		return params

// 	} else {
// 		fmt.Println("file exists")
// 		// Open the file and load the object back!
// 		// file, err := os.Open(secure_params_path)

// 		// data, err := os.ReadFile(secure_params_path)
// 		// if err != nil {
// 		// 	log.Fatal(err)
// 		// }
// 		// dec := json.NewDecoder(file)
// 		// mkckks.Parameters params
// 		// params := new(mkckks.Parameters)
// 		var params mkckks.Parameters
// 		params = *new(mkckks.Parameters)
// 		params.Parameters = *new(mkrlwe.Parameters)
// 		params.Parameters.Parameters = *new(rlwe.Parameters)

// 		params.CRS = make(map[int]*mkrlwe.SwitchingKey)

// 		// params.Parameters.Parameters.r ringQ = new(ring.Ring)

// 		// dec := gob.NewDecoder(bytes.NewReader(data))

// 		// for {
// 		// 	err = dec.Decode(&params)
// 		// 	if err == io.EOF {
// 		// 		break
// 		// 	}
// 		// 	if err != nil {
// 		// 		log.Fatal(err)
// 		// 	}
// 		// 	fmt.Println("decoding fine")
// 		// }

// 		if err := Load(secure_params_path, &params); err != nil {
// 			panic(err)
// 		}
// 		PARAMSLITERAL := &[]ckks.ParametersLiteral{PN14QP439}[0] // hardcoded, assuming using one parameters lietral

// 		ckksParams, err := ckks.NewParametersFromLiteral(*PARAMSLITERAL)

// 		if ckksParams.PCount() < 2 {
// 			fmt.Printf("ckks Params.PCount < 2")
// 			// continue
// 		}

// 		if err != nil {
// 			panic(err)
// 		}
// 		params.Parameters.Parameters = ckksParams.Parameters
// 		// if err := Load("crs.json", &params.CRS); err != nil {
// 		// 	panic(err)
// 		// }
// 		return params
// 	}

// }

//export newMPHEServer
func newMPHEServer(user_idx C.int) *C.MPHEServer {
	server := (*C.MPHEServer)(C.malloc(C.sizeof_MPHEServer))

	PARAMSLITERAL := &[]ckks.ParametersLiteral{PN14QP439}[0] // hardcoded, assuming using one parameters lietral
	server.paramsLiteral = *convParamsLiteral(PARAMSLITERAL)

	ckksParams, err := ckks.NewParametersFromLiteral(*PARAMSLITERAL)
	if ckksParams.PCount() < 2 {
		fmt.Printf("ckks Params.PCount < 2")
		// continue
	}

	if err != nil {
		panic(err)
	}

	PARAMS := mkckks.NewParameters(ckksParams)

	// loadParams()

	// server.params = *convParams(&PARAMS)

	kgen := mkckks.NewKeyGenerator(PARAMS)
	// fmt.Println(kgen)

	// gen sk, pk, rlk, rk
	server.idx = user_idx // user_idx is C.int
	// user_id := "user" + strconv.Itoa(int(server.idx))
	user_id := strconv.Itoa(int(server.idx)) // C.int -> go int -> go string
	fmt.Printf(user_id)

	sk, pk := kgen.GenKeyPair(user_id)
	server.sk = *convPolyQP(&sk.SecretKey.Value)
	server.pk = *convPolyQPPair(pk.PublicKey.Value)

	return server
}

//export encryptFromPk
func encryptFromPk(pk *C.PolyQPPair, array *C.double, arraySize C.size_t, user_idx C.int) *C.Ciphertext {
	// func encryptFromPk(paramsLiteral *C.ParametersLiteral, pk *C.PolyQPPair, array *C.complexdouble, arraySize C.size_t, user_idx C.int) *C.Message {
	// PARAMSLITERAL := &[]ckks.ParametersLiteral{PN14QP439}[0] // hardcoded, assuming using one parameters lietral
	// // server.paramsLiteral = *convParamsLiteral(PARAMSLITERAL)

	// ckksParams, err := ckks.NewParametersFromLiteral(*PARAMSLITERAL)

	// // ckksParams, err := ckks.NewParametersFromLiteral(*paramsLiteral)
	// if ckksParams.PCount() < 2 {
	// 	fmt.Printf("ckks Params.PCount < 2")
	// 	// continue
	// }

	// if err != nil {
	// 	panic(err)
	// }

	// PARAMS := mkckks.NewParameters(ckksParams)
	PARAMS := loadCompactParams()

	publicKey := mkrlwe.NewPublicKey(PARAMS.Parameters, strconv.Itoa(int(user_idx)))

	pkPolyQP := convS2RingPolyQP(pk)
	publicKey.Value[0] = pkPolyQP[0].CopyNew()
	publicKey.Value[1] = pkPolyQP[1].CopyNew()

	// Encrypt the array element-wise
	size := int(arraySize)
	list := (*[1 << 30]C.double)(unsafe.Pointer(array))[:size:size]

	cts := new(mkckks.Ciphertext)
	msg := mkckks.NewMessage(PARAMS)

	for i, elem := range list {
		val := complex(float64(elem), 0.0)
		msg.Value[i] = val
	}
	encryptor := mkckks.NewEncryptor(PARAMS)
	cts = encryptor.EncryptMsgNew(msg, publicKey)
	return convCiphertext(cts)
}

//export partialDecrypt
func partialDecrypt(sk *C.PolyQP, ciphertext *C.Ciphertext, user_idx C.int) *C.Ciphertext {
	// Perform partial decryption using a single sk instead of skSet, update decrypted result in ct.Value[id],
	// requires using ringQAddLvl() to aggregate ct.Value["0"] and other participants' ct.Value[id],
	// then call decodeAfterPartialDecrypt() once the decrypted ciphertext is obtained

	// PARAMSLITERAL := &[]ckks.ParametersLiteral{PN14QP439}[0] // hardcoded, assuming using one parameters lietral
	// ckksParams, err := ckks.NewParametersFromLiteral(*PARAMSLITERAL)
	// if ckksParams.PCount() < 2 {
	// 	fmt.Printf("ckks Params.PCount < 2")
	// 	// continue
	// }
	// if err != nil {
	// 	panic(err)
	// }
	// PARAMS := mkckks.NewParameters(ckksParams)
	PARAMS := loadCompactParams()

	secretKey := mkrlwe.NewSecretKey(PARAMS.Parameters, strconv.Itoa(int(user_idx)))

	skPolyQP := convRingPolyQP(sk)
	secretKey.Value = skPolyQP.CopyNew()

	decryptor := mkckks.NewDecryptor(PARAMS)

	ct := convMKCKKSCiphertext(ciphertext)
	ctTmp := ct.CopyNew()
	decryptor.MyPartialDecrypt(ctTmp, secretKey)

	return convCiphertext(ctTmp)
}

//export ringQAddLvl
func ringQAddLvl(op1 *C.Ciphertext, op1_id C.int, op2 *C.Ciphertext, op2_id C.int) *C.Ciphertext {
	// Add op1.Value[op1_id] and op2.Value[op2_id], write results in a copy of op1.Value[op1_id]

	// PARAMSLITERAL := &[]ckks.ParametersLiteral{PN14QP439}[0] // hardcoded, assuming using one parameters lietral
	// ckksParams, err := ckks.NewParametersFromLiteral(*PARAMSLITERAL)
	// if ckksParams.PCount() < 2 {
	// 	fmt.Printf("ckks Params.PCount < 2")
	// 	// continue
	// }
	// if err != nil {
	// 	panic(err)
	// }
	// PARAMS := mkckks.NewParameters(ckksParams)
	PARAMS := loadCompactParams()

	ct1 := convMKCKKSCiphertext(op1)
	ct2 := convMKCKKSCiphertext(op2)
	ct1_op_id := strconv.Itoa(int(op1_id)) // operator id for ct.Value[id]
	ct2_op_id := strconv.Itoa(int(op2_id))

	ret := ct1.CopyNew()

	ringQ := PARAMS.RingQ()

	level := ct1.Level()
	if ct1.Level() != ct2.Level() {
		fmt.Printf("ringQAddLvl(): ct1.Level() != ct2.Level()")
	}

	ringQ.AddLvl(level, ct1.Value[ct1_op_id], ct2.Value[ct2_op_id], ret.Value[ct1_op_id])
	if ct1_op_id != "0" {
		delete(ret.Value, ct1_op_id)
	}

	if ct2_op_id != "0" {
		delete(ret.Value, ct2_op_id)
	}

	return convCiphertext(ret)
}

//export addRingPs
func addRingPs(ringP1 *C.Poly, ringP2 *C.Poly) *C.Poly {
	// PARAMSLITERAL := &[]ckks.ParametersLiteral{PN14QP439}[0] // hardcoded, assuming using one parameters lietral
	// // server.paramsLiteral = *convParamsLiteral(PARAMSLITERAL)

	// ckksParams, err := ckks.NewParametersFromLiteral(*PARAMSLITERAL)

	// // ckksParams, err := ckks.NewParametersFromLiteral(*paramsLiteral)
	// if ckksParams.PCount() < 2 {
	// 	fmt.Printf("ckks Params.PCount < 2")
	// 	// continue
	// }

	// if err != nil {
	// 	panic(err)
	// }

	// PARAMS := mkckks.NewParameters(ckksParams)
	PARAMS := loadCompactParams()

	rP1 := convRingPoly(ringP1)
	rP2 := convRingPoly(ringP2)

	levelP := PARAMS.PCount() - 1

	sumRP := rP1.CopyNew()

	ringP := PARAMS.RingP()
	ringP.AddLvl(levelP, rP1, rP2, sumRP)

	return convPoly(sumRP)
}

//export decodeAfterPartialDecrypt
func decodeAfterPartialDecrypt(ciphertext *C.Ciphertext) *C.Ldouble {
	// Perform post processing after obtaining the decrypted ciphertext and perform decoding, return double plaintext

	// PARAMSLITERAL := &[]ckks.ParametersLiteral{PN14QP439}[0] // hardcoded, assuming using one parameters lietral
	// ckksParams, err := ckks.NewParametersFromLiteral(*PARAMSLITERAL)
	// if ckksParams.PCount() < 2 {
	// 	fmt.Printf("ckks Params.PCount < 2")
	// 	// continue
	// }
	// if err != nil {
	// 	panic(err)
	// }
	// PARAMS := mkckks.NewParameters(ckksParams)
	PARAMS := loadCompactParams()

	decryptor := mkckks.NewDecryptor(PARAMS)

	ct := convMKCKKSCiphertext(ciphertext)

	// Pre and Post processing in mkrlwe.decryptor.Decrypt()
	ringQ := decryptor.RingQ()
	plaintext := decryptor.PtxtPool()

	level := utils.MinInt(ct.Level(), plaintext.Level())
	plaintext.Value.Coeffs = plaintext.Value.Coeffs[:level+1]

	if len(ct.Value) > 1 {
		panic("Cannot Decrypt: there is a missing secretkey")
	}

	ringQ.ReduceLvl(level, ct.Value["0"], plaintext.Value)

	// Post processing in mkckks.decryptor.Decrypt()
	plaintext.Scale = ct.Scale
	msg := new(mkckks.Message)
	msg.Value = decryptor.Decode(plaintext)

	values := make([]C.double, len(msg.Value))

	for i, complexVal := range msg.Value {
		values[i] = C.double(real(complexVal))
	}

	// Populate C.Ldouble
	array := (*C.Ldouble)(C.malloc(C.sizeof_Ldouble))

	array.data = (*C.double)(&values[0])
	array.size = C.size_t(len(values))
	return array
}

//export addCTs
func addCTs(op1 *C.Ciphertext, op2 *C.Ciphertext) *C.Ciphertext {
	// homomorphic addition on op1 and op2, results are returned in a new ciphertext
	// PARAMSLITERAL := &[]ckks.ParametersLiteral{PN14QP439}[0] // hardcoded, assuming using one parameters lietral
	// ckksParams, err := ckks.NewParametersFromLiteral(*PARAMSLITERAL)
	// if ckksParams.PCount() < 2 {
	// 	fmt.Printf("ckks Params.PCount < 2")
	// 	// continue
	// }
	// if err != nil {
	// 	panic(err)
	// }
	// PARAMS := mkckks.NewParameters(ckksParams)
	PARAMS := loadCompactParams()

	ct1 := convMKCKKSCiphertext(op1)
	ct2 := convMKCKKSCiphertext(op2)

	evaluator := mkckks.NewEvaluator(PARAMS)
	ct3 := evaluator.AddNew(ct1, ct2)

	return convCiphertext(ct3)
}

//export multiplyCTConst
func multiplyCTConst(op1 *C.Ciphertext, op2 C.double) *C.Ciphertext {
	// homomorphic multiplication on op1 (ciphertext) and op2 (constant in double), results are updated in op1
	// PARAMSLITERAL := &[]ckks.ParametersLiteral{PN14QP439}[0] // hardcoded, assuming using one parameters lietral
	// ckksParams, err := ckks.NewParametersFromLiteral(*PARAMSLITERAL)
	// if ckksParams.PCount() < 2 {
	// 	fmt.Printf("ckks Params.PCount < 2")
	// 	// continue
	// }
	// if err != nil {
	// 	panic(err)
	// }
	// PARAMS := mkckks.NewParameters(ckksParams)
	PARAMS := loadCompactParams()

	ct := convMKCKKSCiphertext(op1)
	constant := float64(op2)
	evaluator := mkckks.NewEvaluator(PARAMS)

	evaluator.MultByConst(ct, constant, ct)
	// ct.Scale *= float64(constant)
	// evaluator.Rescale(ct, PARAMS.Scale(), ct)
	return convCiphertext(ct)
}

/* HELPER: Conversion between C and Go structs */
// *ckks.ParametersLiteral --> *C.ParametersLiteral
func convParamsLiteral(p *ckks.ParametersLiteral) *C.ParametersLiteral {
	params_literal := (*C.ParametersLiteral)(C.malloc(C.sizeof_ParametersLiteral))

	// Populate struct
	qi := make([]uint64, len(p.Q))
	copy(qi, p.Q)
	params_literal.qi = convLuint64(qi)

	pi := make([]uint64, len(p.P))
	copy(pi, p.P)
	params_literal.pi = convLuint64(pi)

	params_literal.logN = C.int(p.LogN)
	params_literal.logSlots = C.int(p.LogSlots)

	params_literal.scale = C.double(p.Scale)
	params_literal.sigma = C.double(p.Sigma)

	return params_literal
}

// // *mkckks.Parameters --> *C.Params
// func convParams(p *mkckks.Parameters) *C.Params {
// 	params := (*C.Params)(C.malloc(C.sizeof_Params))

// 	// Populate struct
// 	qi := make([]uint64, len(p.qi))
// 	copy(qi, p.qi)
// 	params.qi = convLuint64(qi)

// 	pi := make([]uint64, len(p.pi))
// 	copy(pi, p.pi)
// 	params.pi = convLuint64(pi)

// 	params.logN = C.int(p.LogN())
// 	params.logSlots = C.int(p.LogSlots())
// 	params.gamma = C.int(2) // TODO: gamma = 2, hardcoded from mkrlwe.NewParameters()

// 	params.scale = C.double(p.Scale())
// 	params.sigma = C.double(p.Sigma())

// 	return params
// }

// // *C.Params --> *ckks.Parameters
// func convCKKSParams(params *C.Params) *ckks.Parameters {
// 	// Create Moduli struct wrapping slices qi, pi
// 	m := ckks.Moduli{
// 		Qi: convSuint64(params.qi),
// 		Pi: convSuint64(params.pi),
// 	}

// 	// Create and populate Params
// 	p, err := ckks.NewParametersFromModuli(uint64(params.logN), &m)

// 	if err != nil {
// 		fmt.Printf("C.Params built wrong: %v\n", err)
// 		return nil
// 	}

// 	p.SetLogSlots(uint64(params.logSlots))
// 	p.SetScale(float64(params.scale))
// 	p.SetSigma(float64(params.sigma))

// 	return p
// }

// /// Message

// // mkckks.Message --> C.Message
// func convMessage(msg mkckks.Message) C.Message {
// 	list := (*C.Message)(C.malloc(C.sizeof_Message))

// 	// for i, comp_val := range msg.Value {
// 	// 	list.data

// 	// }

// 	list.data = (*C.complexdouble)(&msg.Value[0])
// 	list.size = C.size_t(len(msg.Value))

// 	return *list
// }

// // C.Message --> []complex128
// func convMKCKKSMessage(list C.Message) *mkckks.Message {
// 	ret := new(mkckks.Message)
// 	size := int(list.size)
// 	vals := (*[1 << 30]complex128)(unsafe.Pointer(list.data))[:size:size]
// 	ret.Value = vals
// 	return ret
// }

/// Luint64

// []uint64 --> Luint64
func convLuint64(vals []uint64) C.Luint64 {
	list := (*C.Luint64)(C.malloc(C.sizeof_Luint64))

	list.data = (*C.ulonglong)(&vals[0])
	list.size = C.size_t(len(vals))

	return *list
}

// Luint64 --> []uint64
func convSuint64(list C.Luint64) []uint64 {
	size := int(list.size)
	vals := (*[1 << 30]uint64)(unsafe.Pointer(list.data))[:size:size]

	return vals
}

/// Poly

// *ring.Poly --> *C.Poly
func convPoly(r *ring.Poly) *C.Poly {
	p := (*C.Poly)(C.malloc(C.sizeof_Poly))

	// Retrieve each coeff in a slice of C.Luint64
	coeffs := make([]C.Luint64, len(r.Coeffs))
	for i, coeff := range r.Coeffs {
		c := convLuint64(coeff)
		coeffs[i] = c
	}

	// Populate C.Poly
	p.coeffs = (*C.Luint64)(&coeffs[0])
	p.size = C.size_t(len(coeffs))
	p.IsNTT = C.bool(r.IsNTT)
	p.IsMForm = C.bool(r.IsMForm)

	return p
}

// *rlwe.PolyQP --> *C.PolyQP
func convPolyQP(r *rlwe.PolyQP) *C.PolyQP {
	qp := (*C.PolyQP)(C.malloc(C.sizeof_PolyQP))

	qp.Q = convPoly(r.Q)
	qp.P = convPoly(r.P)

	return qp
}

// TODO: reverse not finished
// *C.PolyQP --> *rlwe.PolyQP
func convRingPolyQP(qp *C.PolyQP) *rlwe.PolyQP {
	// // Extract coeffs as []Luint64
	// size := int(p.size)
	// list := (*[1 << 30]C.Luint64)(unsafe.Pointer(p.coeffs))[:size:size]

	// // Extract []uint64 from Luint64 to create [][]uint64
	// coeffs := make([][]uint64, size)
	// for i, coeff := range list {
	// 	c := convSuint64(coeff)
	// 	coeffs[i] = c
	// }

	// // Populate ring.Poly
	// r := new(ring.Poly)
	// r.Coeffs = coeffs

	ret := new(rlwe.PolyQP)

	ret.Q = convRingPoly(qp.Q)
	ret.P = convRingPoly(qp.P)

	return ret
}

// *C.Poly --> *ring.Poly
func convRingPoly(p *C.Poly) *ring.Poly {
	// Extract coeffs as []Luint64
	size := int(p.size)
	list := (*[1 << 30]C.Luint64)(unsafe.Pointer(p.coeffs))[:size:size]

	// Extract []uint64 from Luint64 to create [][]uint64
	coeffs := make([][]uint64, size)
	for i, coeff := range list {
		c := convSuint64(coeff)
		coeffs[i] = c
	}

	// Populate ring.Poly
	r := new(ring.Poly)
	r.Coeffs = coeffs
	r.IsNTT = bool(p.IsNTT)
	r.IsMForm = bool(p.IsMForm)

	return r
}

/// PolyPair

// [2]*ring.Poly --> *C.PolyPair
func convPolyPair(rpp [2]*ring.Poly) *C.PolyPair {
	pp := (*C.PolyPair)(C.malloc(C.sizeof_PolyPair))

	pp.p0 = *convPoly(rpp[0])
	pp.p1 = *convPoly(rpp[1])

	return pp
}

// *C.PolyPair --> [2]*ring.Poly
func convS2RingPoly(pp *C.PolyPair) [2]*ring.Poly {
	var rpp [2]*ring.Poly

	rpp[0] = convRingPoly(&pp.p0)
	rpp[1] = convRingPoly(&pp.p1)

	return rpp
}

/// PolyQPPair

// [2]*rlwe.PolyQP --> *C.PolyQPPair
func convPolyQPPair(rpp [2]rlwe.PolyQP) *C.PolyQPPair {
	qpp := (*C.PolyQPPair)(C.malloc(C.sizeof_PolyQPPair))

	qpp.qp0 = *convPolyQP(&rpp[0])
	qpp.qp1 = *convPolyQP(&rpp[1])

	return qpp
}

// *C.PolyQPPair --> [2]*rlwe.PolyQP
func convS2RingPolyQP(pp *C.PolyQPPair) [2]rlwe.PolyQP {
	var rpp [2]rlwe.PolyQP

	rpp[0] = *convRingPolyQP(&pp.qp0)
	rpp[1] = *convRingPolyQP(&pp.qp1)

	return rpp
}

/// Ciphertext

// *mkckks.Ciphertext --> *C.Ciphertext
func convCiphertext(cc *mkckks.Ciphertext) *C.Ciphertext {
	c := (*C.Ciphertext)(C.malloc(C.sizeof_Ciphertext))

	// Retrieve each polynomial making up the Ciphertext
	value := make([]C.Poly, len(cc.Value))
	user_idxs := make([]C.int, len(cc.Value))
	// if len(cc.Value) > 2 {
	// 	fmt.Printf("WARNING: mkrlwe.Ciphertext contains map length > 2!")
	// }
	counter := 0
	// user_id := 0
	for key, val := range cc.Value {
		int_key, err := strconv.Atoi(key)
		// if int_key != 0 {
		// 	user_id = int_key
		// 	if counter == 0 {
		// 		fmt.Printf("ERROR: key with user_id was the first element in the map of mkrlwe.Ciphertext!")
		// 	}
		// }
		if err != nil {
			// ... handle error
			fmt.Printf("ERROR: Key in the map of mkrlwe.Ciphertext not a valid integer!")
			panic(err)
		}
		value[counter] = *convPoly(val)
		user_idxs[counter] = C.int(int_key)
		counter = counter + 1
	}

	// Populate C.Ciphertext
	c.data = (*C.Poly)(&value[0])
	c.size = C.size_t(len(value))
	// c.idx = (C.int)(user_id)
	c.idxs = (*C.int)(&user_idxs[0])
	c.scale = C.double(cc.Scale)
	// c.isNTT = C.bool(cc.Element.IsNTT())

	return c
}

// // old
// func convCiphertext(cc *ckks.Ciphertext) *C.Ciphertext {
// 	c := (*C.Ciphertext)(C.malloc(C.sizeof_Ciphertext))

// 	// Retrieve each polynomial making up the Ciphertext
// 	value := make([]C.Poly, len(cc.Element.Value()))
// 	for i, val := range cc.Element.Value() {
// 		value[i] = *convPoly(val)
// 	}

// 	// Populate C.Ciphertext
// 	c.value = (*C.Poly)(&value[0])
// 	c.size = C.size_t(len(value))
// 	c.scale = C.double(cc.Element.Scale())
// 	c.isNTT = C.bool(cc.Element.IsNTT())

// 	return c
// }

// *C.Ciphertext --> *mkckks.Ciphertext
func convMKCKKSCiphertext(c *C.Ciphertext) *mkckks.Ciphertext {
	size := int(c.size)
	list := (*[1 << 30]C.Poly)(unsafe.Pointer(c.data))[:size:size]
	list_idxs := (*[1 << 30]C.int)(unsafe.Pointer(c.idxs))[:size:size]

	// Extract []*ringPoly from []C.Poly
	// value := make([]*ring.Poly, size)
	value := make(map[string]*ring.Poly)
	for i, poly := range list { // TODO: i is key, might not be user_idx
		v := convRingPoly(&poly)
		value[strconv.Itoa(int(list_idxs[i]))] = v
		// if i == 0 {
		// 	value["0"] = v
		// } else {
		// 	value[strconv.Itoa(int(c.idx))] = v
		// }
	}

	// Populate ckks.Ciphertext
	cc := new(mkckks.Ciphertext)
	cc.Ciphertext = new(mkrlwe.Ciphertext)
	cc.Ciphertext.Value = make(map[string]*ring.Poly)
	// cc.Value = make(map[string]*ring.Poly)

	cc.Value = value

	cc.Scale = float64(c.scale)
	// cc.SetValue(value)
	// cc.Element.SetScale(float64(c.scale))
	// cc.Element.SetIsNTT(bool(c.isNTT))

	return cc
}

// // Old
// // *C.Ciphertext --> *ckks.Ciphertext
// func convCKKSCiphertext(c *C.Ciphertext) *ckks.Ciphertext {
// 	size := int(c.size)
// 	list := (*[1 << 30]C.Poly)(unsafe.Pointer(c.value))[:size:size]

// 	// Extract []*ringPoly from []C.Poly
// 	value := make([]*ring.Poly, size)
// 	for i, poly := range list {
// 		v := convRingPoly(&poly)
// 		value[i] = v
// 	}

// 	// Populate ckks.Ciphertext
// 	cc := new(ckks.Ciphertext)
// 	cc.Element = new(ckks.Element)

// 	cc.Element.SetValue(value)
// 	cc.Element.SetScale(float64(c.scale))
// 	cc.Element.SetIsNTT(bool(c.isNTT))

// 	return cc
// }

// /// Data
// // []*ckks.Ciphertext --> *C.Data
// func convData(sct []*mkrlwe.Ciphertext) *C.Data {
// 	data := (*C.Data)(C.malloc(C.sizeof_Data))

// 	// Retrieve pointer to slice
// 	ciphertexts := make([]C.Ciphertext, len(sct))
// 	for i, ct := range sct {
// 		ciphertexts[i] = *convCiphertext(ct)
// 	}

// 	data.data = (*C.Ciphertext)(&ciphertexts[0])
// 	data.size = C.size_t(len(sct))

// 	return data
// }

// // *C.Data --> []*ckks.Ciphertext
// func convSckksCiphertext(data *C.Data) []*mkrlwe.Ciphertext {
// 	size := int(data.size)
// 	cts := (*[1 << 30]C.Ciphertext)(unsafe.Pointer(data.data))[:size:size]

// 	// Extract []*ckks.Ciphertext from []C.Ciphertext
// 	cct := make([]*ckks.Ciphertext, size)
// 	for i, ciphertext := range cts {
// 		c := convMKRLWECiphertext(&ciphertext)
// 		cct[i] = c
// 	}

// 	return cct
// }

// // (*C.Data, C.size_t) --> [][]*ckks.Ciphertext
// func convSSckksCiphertext(datas *C.Data, datasSize C.size_t) [][]*ckks.Ciphertext {
// 	size := int(datasSize)
// 	data := (*[1 << 30]C.Data)(unsafe.Pointer(datas))[:size:size]

// 	// Extract [][]*ckks from []C.Data
// 	ccts := make([][]*ckks.Ciphertext, size)
// 	for i, ct := range data {
// 		ccts[i] = convSckksCiphertext(&ct)
// 	}

// 	return ccts
// }

/// Share

// *C.Share --> []*ring.Poly
func convSRingPoly(share *C.Share) []*ring.Poly {
	size := int(share.size)
	list := (*[1 << 30]C.Poly)(unsafe.Pointer(share.data))[:size:size]

	// Extract []*ringPoly from []C.Poly
	polys := make([]*ring.Poly, size)
	for i, poly := range list {
		polys[i] = convRingPoly(&poly)
	}

	return polys
}

// []*ring.Poly --> *C.Share
func convShare(polys []*ring.Poly) *C.Share {
	share := (*C.Share)(C.malloc(C.sizeof_Share))

	rps := make([]C.Poly, len(polys))
	for i, poly := range polys {
		rps[i] = *convPoly(poly)
	}

	share.data = (*C.Poly)(&rps[0])
	share.size = C.size_t(len(rps))

	return share
}

// (*C.Share, N C.size_t) --> [][]*ring.Poly (N rows, D cols)
func convSSRingPoly(shares *C.Share, sharesSize C.size_t) [][]*ring.Poly {
	size := int(sharesSize)
	list := (*[1 << 30]C.Share)(unsafe.Pointer(shares))[:size:size]

	// Extract []([]*ring.Poly) from []C.Share
	ssring := make([][]*ring.Poly, size)
	for i, share := range list {
		ssring[i] = convSRingPoly(&share)
	}

	// TODO: Error-check that all shares have the same number of polynomials
	// NOTE: in theory, one share per ciphertext

	return ssring
}

func main() {
	// start := time.Now()

	// loadCompactParams()

	// elapsed := time.Since(start)
	// fmt.Printf("elapsed time: %s", elapsed)
	// // fmt.Println(params)
	// // fmt.Println(params.CRS)
	// params := loadCompactParams()
	// // fmt.Println(cmp.Equal(bparams, params)) // will result in true
	// // fmt.Println(bparams.Parameters.Parameters.RingP().Modulus)
	// // fmt.Println(params.Parameters.Parameters.RingP().Modulus)

	// fmt.Println(bparams.Parameters.Parameters)
	// fmt.Println(params.Parameters.Parameters)

	// // fmt.Println(params.Parameters.CRS[0].Value[0].Q)

	// // fmt.Println(cmp.Equal(bparams.Parameters.CRS[0].Value[0].Q, params.Parameters.CRS[0].Value[0].Q)) // will result in true
	// // fmt.Println(cmp.Equal(bparams.Parameters.Parameters.Gamma(), params.Parameters.Parameters)) // will result in true

	// // .CRS[0].Value[0].Q
	// newMPHEServer(1)

}
