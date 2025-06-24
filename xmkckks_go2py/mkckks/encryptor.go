package mkckks

import "github.com/ldsec/lattigo/v2/rlwe"
import "github.com/ldsec/lattigo/v2/ckks"
import "mk-lattigo/mkrlwe"

type Encryptor struct {
	*mkrlwe.Encryptor
	encoder    ckks.Encoder
	params     Parameters
	ckksParams ckks.Parameters
	ptxtPool   *ckks.Plaintext
}

// NewEncryptor instatiates a new Encryptor for the CKKS scheme. The key argument can
// be either a *rlwe.PublicKey or a *rlwe.SecretKey.
func NewEncryptor(params Parameters) *Encryptor {
	ckksParams, _ := ckks.NewParameters(params.Parameters.Parameters, params.LogSlots(), params.Scale())

	ret := new(Encryptor)
	ret.Encryptor = mkrlwe.NewEncryptor(params.Parameters)
	ret.encoder = ckks.NewEncoder(ckksParams)
	ret.params = params
	ret.ckksParams = ckksParams
	ret.ptxtPool = ckks.NewPlaintext(ckksParams, params.MaxLevel(), params.Scale())
	return ret
}

// Encrypt encrypts the input plaintext and write the result on ctOut. The encryption
// algorithm depends on how the receiver encryptor was initialized (see NewEncryptor
// and NewFastEncryptor).
// The level of the output ciphertext is min(plaintext.Level(), ciphertext.Level()).
func (enc *Encryptor) EncryptPtxt(plaintext *ckks.Plaintext, pk *mkrlwe.PublicKey, ctOut *Ciphertext) {
	enc.Encryptor.Encrypt(&rlwe.Plaintext{Value: plaintext.Value}, pk, &mkrlwe.Ciphertext{Value: ctOut.Value})
	ctOut.Scale = plaintext.Scale
}

// EncryptMsg encode message and then encrypts the input plaintext and write the result on ctOut. The encryption
// algorithm depends on how the receiver encryptor was initialized (see NewEncryptor
// and NewFastEncryptor).
// The level of the output ciphertext is min(plaintext.Level(), ciphertext.Level()).
func (enc *Encryptor) EncryptMsg(msg *Message, pk *mkrlwe.PublicKey, ctOut *Ciphertext) {
	enc.encoder.Encode(enc.ptxtPool, msg.Value, enc.params.LogSlots())
	enc.EncryptPtxt(enc.ptxtPool, pk, ctOut)
}

// EncryptMsg encode message and then encrypts the input plaintext and write the result on ctOut. The encryption
// algorithm depends on how the receiver encryptor was initialized (see NewEncryptor
// and NewFastEncryptor).
// The level of the output ciphertext is min(plaintext.Level(), ciphertext.Level()).
func (enc *Encryptor) EncryptMsgNew(msg *Message, pk *mkrlwe.PublicKey) (ctOut *Ciphertext) {
	idset := mkrlwe.NewIDSet()
	idset.Add(pk.ID)
	ctOut = NewCiphertext(enc.params, idset, enc.params.MaxLevel(), enc.params.Scale())
	enc.EncryptMsg(msg, pk, ctOut)

	return
}

func (enc *Encryptor) EncodeMsgNew(msg *Message) (ptxtOut *ckks.Plaintext) {
	ptxtOut = ckks.NewPlaintext(enc.ckksParams, enc.params.MaxLevel(), enc.params.Scale())
	enc.encoder.Encode(ptxtOut, msg.Value, enc.params.LogSlots())
	return
}
