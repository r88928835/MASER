package mkrlwe

import (
	"github.com/ldsec/lattigo/v2/ring"
	"math"
	"math/bits"
	"unsafe"
)

// FastBasisExtender stores the necessary parameters for RNS basis extension.
// The used algorithm is from https://eprint.iacr.org/2018/117.pdf.
type FastBasisExtender struct {
	ringQ             *ring.Ring
	ringP             *ring.Ring
	paramsQtoP        []modupParams
	paramsPtoQ        []modupParams
	modDownparamsPtoQ [][]uint64
	modDownparamsQtoP [][]uint64

	polypoolQ *ring.Poly
	polypoolP *ring.Poly
}

type modupParams struct {
	//Parameters for basis extension from Q to P
	// (Q/Qi)^-1) (mod each Qi) (in Montgomery form)
	qoverqiinvqi []uint64
	// Q/qi (mod each Pj) (in Montgomery form)
	qoverqimodp [][]uint64
	// Q*v (mod each Pj) for v in [1,...,k] where k is the number of Pj moduli
	vtimesqmodp [][]uint64
}

func genModDownParams(ringQ, ringP *ring.Ring) (params [][]uint64) {

	params = make([][]uint64, len(ringP.Modulus))

	bredParams := ringQ.BredParams
	mredParams := ringQ.MredParams

	for j := range ringP.Modulus {
		params[j] = make([]uint64, len(ringQ.Modulus))
		for i, qi := range ringQ.Modulus {
			params[j][i] = ring.ModExp(ringP.Modulus[j], qi-2, qi)
			params[j][i] = ring.MForm(params[j][i], qi, bredParams[i])

			if j > 0 {
				params[j][i] = ring.MRed(params[j][i], params[j-1][i], qi, mredParams[i])
			}
		}
	}

	return
}

// NewFastBasisExtender creates a new FastBasisExtender, enabling RNS basis extension from Q to P and P to Q.
func NewFastBasisExtender(ringQ, ringP *ring.Ring) *FastBasisExtender {

	newParams := new(FastBasisExtender)

	newParams.ringQ = ringQ
	newParams.ringP = ringP

	newParams.paramsQtoP = make([]modupParams, len(ringQ.Modulus))
	for i := range ringQ.Modulus {
		newParams.paramsQtoP[i] = basisextenderparameters(ringQ.Modulus[:i+1], ringP.Modulus)
	}

	newParams.paramsPtoQ = make([]modupParams, len(ringP.Modulus))
	for i := range ringP.Modulus {
		newParams.paramsPtoQ[i] = basisextenderparameters(ringP.Modulus[:i+1], ringQ.Modulus)
	}

	newParams.modDownparamsPtoQ = genModDownParams(ringQ, ringP)
	newParams.modDownparamsQtoP = genModDownParams(ringP, ringQ)

	newParams.polypoolQ = ringQ.NewPoly()
	newParams.polypoolP = ringP.NewPoly()

	return newParams
}

func basisextenderparameters(Q, P []uint64) modupParams {

	bredParamsQ := make([][]uint64, len(Q))
	mredParamsQ := make([]uint64, len(Q))
	bredParamsP := make([][]uint64, len(P))
	mredParamsP := make([]uint64, len(P))

	for i := range Q {
		bredParamsQ[i] = ring.BRedParams(Q[i])
		mredParamsQ[i] = ring.MRedParams(Q[i])
	}

	for i := range P {
		bredParamsP[i] = ring.BRedParams(P[i])
		mredParamsP[i] = ring.MRedParams(P[i])
	}

	qoverqiinvqi := make([]uint64, len(Q))
	qoverqimodp := make([][]uint64, len(P))

	for i := range P {
		qoverqimodp[i] = make([]uint64, len(Q))
	}

	var qiStar uint64
	for i, qi := range Q {

		qiStar = ring.MForm(1, qi, bredParamsQ[i])

		for j := 0; j < len(Q); j++ {
			if j != i {
				qiStar = ring.MRed(qiStar, ring.MForm(Q[j], qi, bredParamsQ[i]), qi, mredParamsQ[i])
			}
		}

		// (Q/Qi)^-1) * r (mod Qi) (in Montgomery form)
		qoverqiinvqi[i] = ring.ModexpMontgomery(qiStar, int(qi-2), qi, mredParamsQ[i], bredParamsQ[i])

		for j, pj := range P {
			// (Q/qi * r) (mod Pj) (in Montgomery form)
			qiStar = 1
			for u := 0; u < len(Q); u++ {
				if u != i {
					qiStar = ring.MRed(qiStar, ring.MForm(Q[u], pj, bredParamsP[j]), pj, mredParamsP[j])
				}
			}

			qoverqimodp[j][i] = ring.MForm(qiStar, pj, bredParamsP[j])
		}
	}

	vtimesqmodp := make([][]uint64, len(P))
	var QmodPi uint64
	for j, pj := range P {
		vtimesqmodp[j] = make([]uint64, len(Q)+1)
		// Correction Term (v*Q) mod each Pj

		QmodPi = 1
		for _, qi := range Q {
			QmodPi = ring.MRed(QmodPi, ring.MForm(qi, pj, bredParamsP[j]), pj, mredParamsP[j])
		}

		v := pj - QmodPi
		vtimesqmodp[j][0] = 0
		for i := 1; i < len(Q)+1; i++ {
			vtimesqmodp[j][i] = ring.CRed(vtimesqmodp[j][i-1]+v, pj)
		}
	}

	return modupParams{qoverqiinvqi: qoverqiinvqi, qoverqimodp: qoverqimodp, vtimesqmodp: vtimesqmodp}
}

// ShallowCopy creates a shallow copy of this basis extender in which the read-only data-structures are
// shared with the receiver.
func (basisextender *FastBasisExtender) ShallowCopy() *FastBasisExtender {
	if basisextender == nil {
		return nil
	}
	return &FastBasisExtender{
		ringQ:             basisextender.ringQ,
		ringP:             basisextender.ringP,
		paramsQtoP:        basisextender.paramsQtoP,
		paramsPtoQ:        basisextender.paramsPtoQ,
		modDownparamsQtoP: basisextender.modDownparamsQtoP,
		modDownparamsPtoQ: basisextender.modDownparamsPtoQ,

		polypoolQ: basisextender.ringQ.NewPoly(),
		polypoolP: basisextender.ringP.NewPoly(),
	}
}

// ModUpQtoP extends the RNS basis of a polynomial from Q to QP.
// Given a polynomial with coefficients in basis {Q0,Q1....Qlevel},
// it extends its basis from {Q0,Q1....Qlevel} to {Q0,Q1....Qlevel,P0,P1...Pj}
func (basisextender *FastBasisExtender) ModUpQtoP(levelQ, levelP int, polQ, polP *ring.Poly) {
	modUpExact(polQ.Coeffs[:levelQ+1], polP.Coeffs[:levelP+1], basisextender.ringQ, basisextender.ringP, basisextender.paramsQtoP[levelQ])
}

// ModUpPtoQ extends the RNS basis of a polynomial from P to PQ.
// Given a polynomial with coefficients in basis {P0,P1....Plevel},
// it extends its basis from {P0,P1....Plevel} to {Q0,Q1...Qj}
func (basisextender *FastBasisExtender) ModUpPtoQ(levelP, levelQ int, polP, polQ *ring.Poly) {
	modUpExact(polP.Coeffs[:levelP+1], polQ.Coeffs[:levelQ+1], basisextender.ringP, basisextender.ringQ, basisextender.paramsPtoQ[levelP])
}

// ModDownQPtoQ reduces the basis of a polynomial.
// Given a polynomial with coefficients in basis {Q0,Q1....Qlevel} and {P0,P1...Pj},
// it reduces its basis from {Q0,Q1....Qlevel} and {P0,P1...Pj} to {Q0,Q1....Qlevel}
// and does a rounded integer division of the result by P.
func (basisextender *FastBasisExtender) ModDownQPtoQ(levelQ, levelP int, p1Q, p1P, p2Q *ring.Poly) {

	ringQ := basisextender.ringQ
	modDownParams := basisextender.modDownparamsPtoQ
	polypool := basisextender.polypoolQ

	// Then we target this P basis of p1 and convert it to a Q basis (at the "level" of p1) and copy it on polypool
	// polypool is now the representation of the P basis of p1 but in basis Q (at the "level" of p1)
	basisextender.ModUpPtoQ(levelP, levelQ, p1P, polypool)

	// Finally, for each level of p1 (and polypool since they now share the same basis) we compute p2 = (P^-1) * (p1 - polypool) mod Q
	for i := 0; i < levelQ+1; i++ {

		qi := ringQ.Modulus[i]
		twoqi := qi << 1
		p1tmp := p1Q.Coeffs[i]
		p2tmp := p2Q.Coeffs[i]
		p3tmp := polypool.Coeffs[i]
		params := qi - modDownParams[levelP][i]
		mredParams := ringQ.MredParams[i]

		// Then for each coefficient we compute (P^-1) * (p1[i][j] - polypool[i][j]) mod qi
		for j := 0; j < ringQ.N; j = j + 8 {

			x := (*[8]uint64)(unsafe.Pointer(&p1tmp[j]))
			y := (*[8]uint64)(unsafe.Pointer(&p3tmp[j]))
			z := (*[8]uint64)(unsafe.Pointer(&p2tmp[j]))

			z[0] = ring.MRed(y[0]+twoqi-x[0], params, qi, mredParams)
			z[1] = ring.MRed(y[1]+twoqi-x[1], params, qi, mredParams)
			z[2] = ring.MRed(y[2]+twoqi-x[2], params, qi, mredParams)
			z[3] = ring.MRed(y[3]+twoqi-x[3], params, qi, mredParams)
			z[4] = ring.MRed(y[4]+twoqi-x[4], params, qi, mredParams)
			z[5] = ring.MRed(y[5]+twoqi-x[5], params, qi, mredParams)
			z[6] = ring.MRed(y[6]+twoqi-x[6], params, qi, mredParams)
			z[7] = ring.MRed(y[7]+twoqi-x[7], params, qi, mredParams)
		}
	}

	// In total we do len(P) + len(Q) NTT, which is optimal (linear in the number of moduli of P and Q)
}

// ModDownQPtoQNTT reduces the basis of a polynomial.
// Given a polynomial with coefficients in basis {Q0,Q1....Qi} and {P0,P1...Pj},
// it reduces its basis from {Q0,Q1....Qi} and {P0,P1...Pj} to {Q0,Q1....Qi}
// and does a rounded integer division of the result by P.
// Inputs must be in the NTT domain.
func (basisextender *FastBasisExtender) ModDownQPtoQNTT(levelQ, levelP int, p1Q, p1P, p2Q *ring.Poly) {

	ringQ := basisextender.ringQ
	ringP := basisextender.ringP
	modDownParams := basisextender.modDownparamsPtoQ
	polypool := basisextender.polypoolQ

	// First we get the P basis part of p1 out of the NTT domain
	ringP.InvNTTLazyLvl(levelP, p1P, p1P)

	// Then we target this P basis of p1 and convert it to a Q basis (at the "level" of p1) and copy it on polypool
	// polypool is now the representation of the P basis of p1 but in basis Q (at the "level" of p1)
	basisextender.ModUpPtoQ(levelP, levelQ, p1P, polypool)

	// Finally, for each level of p1 (and polypool since they now share the same basis) we compute p2 = (P^-1) * (p1 - polypool) mod Q
	for i := 0; i < levelQ+1; i++ {

		qi := ringQ.Modulus[i]
		twoqi := qi << 1
		p1tmp := p1Q.Coeffs[i]
		p2tmp := p2Q.Coeffs[i]
		p3tmp := polypool.Coeffs[i]
		params := qi - modDownParams[levelP][i]
		mredParams := ringQ.MredParams[i]
		bredParams := ringQ.BredParams[i]
		nttPsi := ringQ.NttPsi[i]

		// First we switch back the relevant polypool CRT array back to the NTT domain
		ring.NTTLazy(p3tmp, p3tmp, ringQ.N, nttPsi, qi, mredParams, bredParams)

		// Then for each coefficient we compute (P^-1) * (p1[i][j] - polypool[i][j]) mod qi
		for j := 0; j < ringQ.N; j = j + 8 {

			x := (*[8]uint64)(unsafe.Pointer(&p1tmp[j]))
			y := (*[8]uint64)(unsafe.Pointer(&p3tmp[j]))
			z := (*[8]uint64)(unsafe.Pointer(&p2tmp[j]))

			z[0] = ring.MRed(y[0]+twoqi-x[0], params, qi, mredParams)
			z[1] = ring.MRed(y[1]+twoqi-x[1], params, qi, mredParams)
			z[2] = ring.MRed(y[2]+twoqi-x[2], params, qi, mredParams)
			z[3] = ring.MRed(y[3]+twoqi-x[3], params, qi, mredParams)
			z[4] = ring.MRed(y[4]+twoqi-x[4], params, qi, mredParams)
			z[5] = ring.MRed(y[5]+twoqi-x[5], params, qi, mredParams)
			z[6] = ring.MRed(y[6]+twoqi-x[6], params, qi, mredParams)
			z[7] = ring.MRed(y[7]+twoqi-x[7], params, qi, mredParams)
		}
	}

	// In total we do len(P) + len(Q) NTT, which is optimal (linear in the number of moduli of P and Q)
}

// ModDownQPtoP reduces the basis of a polynomial.
// Given a polynomial with coefficients in basis {Q0,Q1....QlevelQ} and {P0,P1...PlevelP},
// it reduces its basis from {Q0,Q1....QlevelQ} and {P0,P1...PlevelP} to {P0,P1...PlevelP}
// and does a floored integer division of the result by Q.
func (basisextender *FastBasisExtender) ModDownQPtoP(levelQ, levelP int, p1Q, p1P, p2P *ring.Poly) {

	ringP := basisextender.ringP
	modDownParams := basisextender.modDownparamsQtoP
	polypool := basisextender.polypoolP

	// Then we target this P basis of p1 and convert it to a Q basis (at the "level" of p1) and copy it on polypool
	// polypool is now the representation of the P basis of p1 but in basis Q (at the "level" of p1)
	basisextender.ModUpQtoP(levelQ, levelP, p1Q, polypool)

	// Finally, for each level of p1 (and polypool since they now share the same basis) we compute p2 = (P^-1) * (p1 - polypool) mod Q
	for i := 0; i < levelP+1; i++ {

		qi := ringP.Modulus[i]
		twoqi := qi << 1
		p1tmp := p1P.Coeffs[i]
		p2tmp := p2P.Coeffs[i]
		p3tmp := polypool.Coeffs[i]
		params := qi - modDownParams[levelP][i]
		mredParams := ringP.MredParams[i]

		// Then for each coefficient we compute (P^-1) * (p1[i][j] - polypool[i][j]) mod qi
		for j := 0; j < ringP.N; j = j + 8 {

			x := (*[8]uint64)(unsafe.Pointer(&p1tmp[j]))
			y := (*[8]uint64)(unsafe.Pointer(&p3tmp[j]))
			z := (*[8]uint64)(unsafe.Pointer(&p2tmp[j]))

			z[0] = ring.MRed(y[0]+twoqi-x[0], params, qi, mredParams)
			z[1] = ring.MRed(y[1]+twoqi-x[1], params, qi, mredParams)
			z[2] = ring.MRed(y[2]+twoqi-x[2], params, qi, mredParams)
			z[3] = ring.MRed(y[3]+twoqi-x[3], params, qi, mredParams)
			z[4] = ring.MRed(y[4]+twoqi-x[4], params, qi, mredParams)
			z[5] = ring.MRed(y[5]+twoqi-x[5], params, qi, mredParams)
			z[6] = ring.MRed(y[6]+twoqi-x[6], params, qi, mredParams)
			z[7] = ring.MRed(y[7]+twoqi-x[7], params, qi, mredParams)
		}
	}

	// In total we do len(P) + len(Q) NTT, which is optimal (linear in the number of moduli of P and Q)
}

// Caution, returns the values in [0, 2q-1]
func modUpExact(p1, p2 [][]uint64, ringQ, ringP *ring.Ring, params modupParams) {

	var v [8]uint64
	var y0, y1, y2, y3, y4, y5, y6, y7 [32]uint64

	Q := ringQ.Modulus
	P := ringP.Modulus
	mredParamsQ := ringQ.MredParams
	mredParamsP := ringP.MredParams
	vtimesqmodp := params.vtimesqmodp
	qoverqiinvqi := params.qoverqiinvqi
	qoverqimodp := params.qoverqimodp

	// We loop over each coefficient and apply the basis extension
	for x := 0; x < len(p1[0]); x = x + 8 {
		reconstructRNS(len(p1), x, p1, &v, &y0, &y1, &y2, &y3, &y4, &y5, &y6, &y7, Q, mredParamsQ, qoverqiinvqi)
		for j := 0; j < len(p2); j++ {
			multSum((*[8]uint64)(unsafe.Pointer(&p2[j][x])), &v, &y0, &y1, &y2, &y3, &y4, &y5, &y6, &y7, len(p1), P[j], mredParamsP[j], vtimesqmodp[j], qoverqimodp[j])
		}
	}
}

// Decomposer is a structure that stores the parameters of the arbitrary decomposer.
// This decomposer takes a p(x)_Q (in basis Q) and returns p(x) mod qi in basis QP, where
// qi = prod(Q_i) for 0<=i<=L, where L is the number of factors in P.
type Decomposer struct {
	ringQ, ringP *ring.Ring
	modUpParams  [][][]modupParams
}

// NewDecomposer creates a new Decomposer.
func NewDecomposer(ringQ, ringP *ring.Ring, gamma int) (decomposer *Decomposer) {
	decomposer = new(Decomposer)

	decomposer.ringQ = ringQ
	decomposer.ringP = ringP

	Q := ringQ.Modulus

	decomposer.modUpParams = make([][][]modupParams, len(ringP.Modulus)-1)

	for lvlP := range ringP.Modulus[1:] {

		P := ringP.Modulus[:lvlP+2]

		alpha := int(math.Ceil(float64(len(P)) / float64(gamma)))
		beta := int(math.Ceil(float64(len(Q)) / float64(alpha)))

		xalpha := make([]int, beta)
		for i := range xalpha {
			xalpha[i] = alpha
		}

		if len(Q)%alpha != 0 {
			xalpha[beta-1] = len(Q) % alpha
		}

		decomposer.modUpParams[lvlP] = make([][]modupParams, beta)

		// Create modUpParams for each possible combination of [Qi,Pj] according to xalpha
		for i := 0; i < beta; i++ {

			decomposer.modUpParams[lvlP][i] = make([]modupParams, xalpha[i]-1)

			for j := 0; j < xalpha[i]-1; j++ {

				Qi := make([]uint64, j+2)
				Pi := make([]uint64, len(Q)+len(P))

				for k := 0; k < j+2; k++ {
					Qi[k] = Q[i*alpha+k]
				}

				for k := 0; k < len(Q); k++ {
					Pi[k] = Q[k]
				}

				for k := len(Q); k < len(Q)+len(P); k++ {
					Pi[k] = P[k-len(Q)]
				}

				decomposer.modUpParams[lvlP][i][j] = basisextenderparameters(Qi, Pi)
			}
		}
	}

	return
}

// DecomposeAndSplit decomposes a polynomial p(x) in basis Q, reduces it modulo qi, and returns
// the result in basis QP separately.
func (decomposer *Decomposer) DecomposeAndSplit(levelQ, levelP, alpha, beta, gamma int, p0Q, p1Q, p1P *ring.Poly) {

	ringQ := decomposer.ringQ
	ringP := decomposer.ringP

	lvlQStart := beta * alpha

	var decompLvl int
	if levelQ > alpha*(beta+1)-1 {
		decompLvl = alpha - 2
	} else {
		decompLvl = (levelQ % alpha) - 1
	}

	// First we check if the vector can simply by coping and rearranging elements (the case where no reconstruction is needed)
	if decompLvl == -1 {

		for j := 0; j < levelQ+1; j++ {
			copy(p1Q.Coeffs[j], p0Q.Coeffs[lvlQStart])
		}

		for j := 0; j < levelP+1; j++ {
			copy(p1P.Coeffs[j], p0Q.Coeffs[lvlQStart])
		}

		// Otherwise, we apply a fast exact base conversion for the reconstruction
	} else {

		params := decomposer.modUpParams[gamma*alpha-2][beta][decompLvl]

		var v [8]uint64
		var vi [8]float64
		var y0, y1, y2, y3, y4, y5, y6, y7 [32]uint64

		Q := ringQ.Modulus
		P := ringP.Modulus
		mredParamsQ := ringQ.MredParams
		mredParamsP := ringP.MredParams
		qoverqiinvqi := params.qoverqiinvqi
		vtimesqmodp := params.vtimesqmodp
		qoverqimodp := params.qoverqimodp

		// We loop over each coefficient and apply the basis extension
		for x := 0; x < len(p0Q.Coeffs[0]); x = x + 8 {

			vi[0], vi[1], vi[2], vi[3], vi[4], vi[5], vi[6], vi[7] = 0, 0, 0, 0, 0, 0, 0, 0

			// Coefficients to be decomposed
			for i, j := 0, lvlQStart; i < decompLvl+2; i, j = i+1, j+1 {

				qqiinv := qoverqiinvqi[i]
				qi := Q[j]
				mredParams := mredParamsQ[j]
				qif := float64(qi)

				px := (*[8]uint64)(unsafe.Pointer(&p0Q.Coeffs[j][x]))
				py := (*[8]uint64)(unsafe.Pointer(&p1Q.Coeffs[j][x]))

				// For the coefficients to be decomposed, we can simply copy them
				py[0], py[1], py[2], py[3], py[4], py[5], py[6], py[7] = px[0], px[1], px[2], px[3], px[4], px[5], px[6], px[7]

				y0[i] = ring.MRed(px[0], qqiinv, qi, mredParams)
				y1[i] = ring.MRed(px[1], qqiinv, qi, mredParams)
				y2[i] = ring.MRed(px[2], qqiinv, qi, mredParams)
				y3[i] = ring.MRed(px[3], qqiinv, qi, mredParams)
				y4[i] = ring.MRed(px[4], qqiinv, qi, mredParams)
				y5[i] = ring.MRed(px[5], qqiinv, qi, mredParams)
				y6[i] = ring.MRed(px[6], qqiinv, qi, mredParams)
				y7[i] = ring.MRed(px[7], qqiinv, qi, mredParams)

				// Computation of the correction term v * Q%pi
				vi[0] += float64(y0[i]) / qif
				vi[1] += float64(y1[i]) / qif
				vi[2] += float64(y2[i]) / qif
				vi[3] += float64(y3[i]) / qif
				vi[4] += float64(y4[i]) / qif
				vi[5] += float64(y5[i]) / qif
				vi[6] += float64(y6[i]) / qif
				vi[7] += float64(y7[i]) / qif
			}

			// Index of the correction term
			v[0] = uint64(vi[0])
			v[1] = uint64(vi[1])
			v[2] = uint64(vi[2])
			v[3] = uint64(vi[3])
			v[4] = uint64(vi[4])
			v[5] = uint64(vi[5])
			v[6] = uint64(vi[6])
			v[7] = uint64(vi[7])

			// Coefficients of index smaller than the ones to be decomposed
			for j := 0; j < lvlQStart; j++ {
				multSum((*[8]uint64)(unsafe.Pointer(&p1Q.Coeffs[j][x])), &v, &y0, &y1, &y2, &y3, &y4, &y5, &y6, &y7, decompLvl+2, Q[j], mredParamsQ[j], vtimesqmodp[j], qoverqimodp[j])
			}

			// Coefficients of index greater than the ones to be decomposed
			for j := alpha * beta; j < levelQ+1; j = j + 1 {
				multSum((*[8]uint64)(unsafe.Pointer(&p1Q.Coeffs[j][x])), &v, &y0, &y1, &y2, &y3, &y4, &y5, &y6, &y7, decompLvl+2, Q[j], mredParamsQ[j], vtimesqmodp[j], qoverqimodp[j])
			}

			// Coefficients of the special primes Pi
			for j, u := 0, len(ringQ.Modulus); j < levelP+1; j, u = j+1, u+1 {
				multSum((*[8]uint64)(unsafe.Pointer(&p1P.Coeffs[j][x])), &v, &y0, &y1, &y2, &y3, &y4, &y5, &y6, &y7, decompLvl+2, P[j], mredParamsP[j], vtimesqmodp[u], qoverqimodp[u])
			}
		}
	}
}

func reconstructRNS(index, x int, p [][]uint64, v *[8]uint64, y0, y1, y2, y3, y4, y5, y6, y7 *[32]uint64, Q, QInv, QbMont []uint64) {

	var vi [8]float64
	var qi, qiInv, qoverqiinvqi uint64
	var qif float64

	for i := 0; i < index; i++ {

		qoverqiinvqi = QbMont[i]
		qi = Q[i]
		qiInv = QInv[i]
		qif = float64(qi)
		pTmp := (*[8]uint64)(unsafe.Pointer(&p[i][x]))

		y0[i] = ring.MRed(pTmp[0], qoverqiinvqi, qi, qiInv)
		y1[i] = ring.MRed(pTmp[1], qoverqiinvqi, qi, qiInv)
		y2[i] = ring.MRed(pTmp[2], qoverqiinvqi, qi, qiInv)
		y3[i] = ring.MRed(pTmp[3], qoverqiinvqi, qi, qiInv)
		y4[i] = ring.MRed(pTmp[4], qoverqiinvqi, qi, qiInv)
		y5[i] = ring.MRed(pTmp[5], qoverqiinvqi, qi, qiInv)
		y6[i] = ring.MRed(pTmp[6], qoverqiinvqi, qi, qiInv)
		y7[i] = ring.MRed(pTmp[7], qoverqiinvqi, qi, qiInv)

		// Computation of the correction term v * Q%pi
		vi[0] += float64(y0[i]) / qif
		vi[1] += float64(y1[i]) / qif
		vi[2] += float64(y2[i]) / qif
		vi[3] += float64(y3[i]) / qif
		vi[4] += float64(y4[i]) / qif
		vi[5] += float64(y5[i]) / qif
		vi[6] += float64(y6[i]) / qif
		vi[7] += float64(y7[i]) / qif
	}

	v[0] = uint64(vi[0])
	v[1] = uint64(vi[1])
	v[2] = uint64(vi[2])
	v[3] = uint64(vi[3])
	v[4] = uint64(vi[4])
	v[5] = uint64(vi[5])
	v[6] = uint64(vi[6])
	v[7] = uint64(vi[7])
}

// Caution, returns the values in [0, 2q-1]
func multSum(res, v *[8]uint64, y0, y1, y2, y3, y4, y5, y6, y7 *[32]uint64, alpha int, pj, qInv uint64, vtimesqmodp, qoverqimodp []uint64) {

	var rlo, rhi [8]uint64
	var mhi, mlo, c, hhi uint64

	// Accumulates the sum on uint128 and does a lazy montgomery reduction at the end
	for i := 0; i < alpha; i++ {

		mhi, mlo = bits.Mul64(y0[i], qoverqimodp[i])
		rlo[0], c = bits.Add64(rlo[0], mlo, 0)
		rhi[0] += mhi + c

		mhi, mlo = bits.Mul64(y1[i], qoverqimodp[i])
		rlo[1], c = bits.Add64(rlo[1], mlo, 0)
		rhi[1] += mhi + c

		mhi, mlo = bits.Mul64(y2[i], qoverqimodp[i])
		rlo[2], c = bits.Add64(rlo[2], mlo, 0)
		rhi[2] += mhi + c

		mhi, mlo = bits.Mul64(y3[i], qoverqimodp[i])
		rlo[3], c = bits.Add64(rlo[3], mlo, 0)
		rhi[3] += mhi + c

		mhi, mlo = bits.Mul64(y4[i], qoverqimodp[i])
		rlo[4], c = bits.Add64(rlo[4], mlo, 0)
		rhi[4] += mhi + c

		mhi, mlo = bits.Mul64(y5[i], qoverqimodp[i])
		rlo[5], c = bits.Add64(rlo[5], mlo, 0)
		rhi[5] += mhi + c

		mhi, mlo = bits.Mul64(y6[i], qoverqimodp[i])
		rlo[6], c = bits.Add64(rlo[6], mlo, 0)
		rhi[6] += mhi + c

		mhi, mlo = bits.Mul64(y7[i], qoverqimodp[i])
		rlo[7], c = bits.Add64(rlo[7], mlo, 0)
		rhi[7] += mhi + c
	}

	hhi, _ = bits.Mul64(rlo[0]*qInv, pj)
	res[0] = rhi[0] - hhi + pj + vtimesqmodp[v[0]]

	hhi, _ = bits.Mul64(rlo[1]*qInv, pj)
	res[1] = rhi[1] - hhi + pj + vtimesqmodp[v[1]]

	hhi, _ = bits.Mul64(rlo[2]*qInv, pj)
	res[2] = rhi[2] - hhi + pj + vtimesqmodp[v[2]]

	hhi, _ = bits.Mul64(rlo[3]*qInv, pj)
	res[3] = rhi[3] - hhi + pj + vtimesqmodp[v[3]]

	hhi, _ = bits.Mul64(rlo[4]*qInv, pj)
	res[4] = rhi[4] - hhi + pj + vtimesqmodp[v[4]]

	hhi, _ = bits.Mul64(rlo[5]*qInv, pj)
	res[5] = rhi[5] - hhi + pj + vtimesqmodp[v[5]]

	hhi, _ = bits.Mul64(rlo[6]*qInv, pj)
	res[6] = rhi[6] - hhi + pj + vtimesqmodp[v[6]]

	hhi, _ = bits.Mul64(rlo[7]*qInv, pj)
	res[7] = rhi[7] - hhi + pj + vtimesqmodp[v[7]]
}
