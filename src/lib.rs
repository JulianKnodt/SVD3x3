#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub type F = f32;

const JACOBI_STEPS: usize = 12;

/// Compute the matrix multiplication of arrays a and b.
pub fn matmul<const I: usize, const J: usize, const K: usize>(
    a: [[F; I]; J],
    b: [[F; J]; K],
) -> [[F; I]; K] {
    let mut out = [[0.; I]; K];

    for i in 0..I {
        for j in 0..J {
            for k in 0..K {
                out[k][i] += a[j][i] * b[k][j];
            }
        }
    }

    out
}

pub fn diag_to_mat([d0, d1, d2]: [F; 3]) -> [[F; 3]; 3] {
    [[d0, 0., 0.], [0., d1, 0.], [0., 0., d2]]
}

pub fn transpose(mat: [[F; 3]; 3]) -> [[F; 3]; 3] {
    let mut out = [[0.; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = mat[j][i];
        }
    }
    out
}

fn conj_sym(
    SymMatrix {
        v00,
        v01,
        v11,
        v02,
        v12,
        v22,
    }: SymMatrix,
    a: F,
    b: F,
) -> SymMatrix {
    SymMatrix {
        v00: a * (a * v00 + b * v01) + b * (a * v01 + b * v11),
        v01: a * (-b * v00 + a * v01) + b * (-b * v01 + a * v11),
        v11: -b * (-b * v00 + a * v01) + a * (-b * v01 + a * v11),
        v02: a * v02 + b * v12,
        v12: -b * v02 + a * v12,
        v22,
    }
}

fn jacobi_conj<const X: usize, const Y: usize, const Z: usize>(
    s: SymMatrix,
    [x, y, z, w]: [F; 4],
) -> (SymMatrix, [F; 4]) {
    let [ch, sh] = approx_givens_quat(&s);
    let scale = ch * ch + sh * sh;
    let a = (ch * ch - sh * sh) / scale;
    let b = (2. * ch * sh) / scale;

    let s = conj_sym(s, a, b);

    let tmp = [x, y, z].map(|v| v * sh);
    let mut q = [x, y, z, w].map(|v| v * ch);

    q[Z] += sh * w;
    q[3] -= tmp[Z];
    q[X] += tmp[Y];
    q[Y] -= tmp[X];

    let SymMatrix {
        v00,
        v01,
        v11,
        v02,
        v12,
        v22,
    } = s;
    let new_s = SymMatrix {
        v00: v11,
        v01: v12,
        v11: v22,
        v02: v01,
        v12: v02,
        v22: v00,
    };
    (new_s, q)
}

fn jacobi_eigen(mut s: SymMatrix) -> [F; 4] {
    let mut q = [0., 0., 0., 1.];
    for _ in 0..JACOBI_STEPS {
        let (ns, nq) = jacobi_conj::<0, 1, 2>(s, q);
        let (ns, nq) = jacobi_conj::<1, 2, 0>(ns, nq);
        let (ns, nq) = jacobi_conj::<2, 0, 1>(ns, nq);
        s = ns;
        q = nq;
    }
    q
}

pub fn quat_to_mat([x, y, z, w]: [F; 4]) -> [[F; 3]; 3] {
    [
        [
            1. - 2. * (y * y + z * z),
            2. * (x * y - w * z),
            2. * (x * z + w * y),
        ],
        [
            2. * (x * y + w * z),
            1. - 2. * (x * x + z * z),
            2. * (y * z - w * x),
        ],
        [
            2. * (x * z - w * y),
            2. * (y * z + w * x),
            1. - 2. * (x * x + y * y),
        ],
    ]
}

/// Returns (u, singular values, d)
pub fn svd(mat: [[F; 3]; 3]) -> [/**/ [[F; 3]; 3]; 3] {
    let mut v = quat_to_mat(jacobi_eigen(SymMatrix::new(matmul(transpose(mat), mat))));
    let mut b = matmul(v, mat);
    sort_singular_values(&mut b, &mut v);
    let [q, r] = qr_decomp(&mut b);
    [q, r, v]
}

#[derive(Clone, Copy)]
struct SymMatrix {
    v00: F,
    v01: F,
    v11: F,
    v02: F,
    v12: F,
    v22: F,
}

impl SymMatrix {
    pub fn new([[v00, v01, v02], [_, v11, v12], [_, _, v22]]: [[F; 3]; 3]) -> Self {
        SymMatrix {
            v00,
            v01,
            v11,
            v02,
            v12,
            v22,
        }
    }
}

fn approx_givens_quat(s: &SymMatrix) -> [F; 2] {
    let ch = 2. * s.v00 - s.v11;
    let sh = s.v01;

    if sh == 0. && ch == 0. {
        return [1., 0.];
    }

    let gamma = (8. as F).sqrt() + 3.;

    if gamma * sh * sh < ch * ch {
        let w = 1. / (sh * sh + ch * ch).sqrt();
        [w * ch, w * sh]
    } else {
        let cstar = (std::f64::consts::FRAC_PI_8 as F).cos();
        let sstar = (std::f64::consts::FRAC_PI_8 as F).sin();
        [cstar, sstar]
    }
}

fn qr_givens_quat(a1: F, a2: F) -> [F; 2] {
    const EPS: F = 1e-6;
    let rho = (a1 * a1 + a2 * a2).sqrt();
    let mut g = [a1.abs() + rho.max(EPS), if rho > EPS { a2 } else { 0. }];
    if a1 < 0. {
        g.swap(0, 1);
    }
    let w = (g[0] * g[0] + g[1] * g[1]).sqrt();
    g.map(|v| v / w)
}

pub fn sort_singular_values(b: &mut [[F; 3]; 3], v: &mut [[F; 3]; 3]) {
    let [mut rho1, mut rho2, mut rho3] = std::array::from_fn(|i| dist_sq(b[i]));
    if rho1 < rho2 {
        for i in 0..3 {
            b[i].swap(0, 1);
            b[i][1] = -b[i][1];
            v[i].swap(0, 1);
            v[i][1] = -v[i][1];
        }
        std::mem::swap(&mut rho1, &mut rho2);
    }
    if rho1 < rho3 {
        for i in 0..3 {
            b[i].swap(0, 2);
            b[i][2] = -b[i][2];
            v[i].swap(0, 2);
            v[i][2] = -v[i][2];
        }
        std::mem::swap(&mut rho1, &mut rho3);
    }
    if rho2 < rho3 {
        for i in 0..3 {
            b[i].swap(1, 2);
            b[i][2] = -b[i][2];
            v[i].swap(1, 2);
            v[i][2] = -v[i][2];
        }
        std::mem::swap(&mut rho2, &mut rho3);
    }
    assert!(rho1 >= rho2);
    assert!(rho2 >= rho3);
}

#[inline]
fn dist_sq([a, b, c]: [F; 3]) -> F {
    a * a + b * b + c * c
}

#[inline]
fn fmaf(x: F, y: F, z: F) -> F {
    x * y + z
}

/// Note that this outputs [q,r], and to recover b, it is necessary to use matmul(r,q).
pub fn qr_decomp(#[allow(non_snake_case)] B: &mut [[F; 3]; 3]) -> [[[F; 3]; 3]; 2] {
    #[allow(non_snake_case)]
    let mut Q = [[0.; 3]; 3];
    #[allow(non_snake_case)]
    let mut R = [[0.; 3]; 3];
    let [g1_ch, g1_sh] = qr_givens_quat(B[0][0], B[1][0]);
    let a = -2. * g1_sh * g1_sh + 1.;
    let b = 2. * g1_ch * g1_sh;
    // apply B = Q' * B
    R[0][0] = fmaf(a, B[0][0], b * B[1][0]);
    R[0][1] = fmaf(a, B[0][1], b * B[1][1]);
    R[0][2] = fmaf(a, B[0][2], b * B[1][2]);
    R[1][0] = fmaf(-b, B[0][0], a * B[1][0]);
    R[1][1] = fmaf(-b, B[0][1], a * B[1][1]);
    R[1][2] = fmaf(-b, B[0][2], a * B[1][2]);
    R[2][0] = B[2][0];
    R[2][1] = B[2][1];
    R[2][2] = B[2][2];
    // second givens rotation (ch,0,-sh,0)
    let [g2_ch, g2_sh] = qr_givens_quat(R[0][0], R[2][0]);
    let a = fmaf(-2., g2_sh * g2_sh, 1.);
    let b = 2. * g2_ch * g2_sh;
    // apply B = Q' * B;
    B[0][0] = fmaf(a, R[0][0], b * R[2][0]);
    B[0][1] = fmaf(a, R[0][1], b * R[2][1]);
    B[0][2] = fmaf(a, R[0][2], b * R[2][2]);
    B[1][0] = R[1][0];
    B[1][1] = R[1][1];
    B[1][2] = R[1][2];
    B[2][0] = fmaf(-b, R[0][0], a * R[2][0]);
    B[2][1] = fmaf(-b, R[0][1], a * R[2][1]);
    B[2][2] = fmaf(-b, R[0][2], a * R[2][2]);
    // third givens rotation (ch,sh,0,0)
    let [g3_ch, g3_sh] = qr_givens_quat(B[1][1], B[2][1]);
    let a = fmaf(-2., g3_sh * g3_sh, 1.);
    let b = 2. * g3_ch * g3_sh;
    // R is now set to desired value
    R[0][0] = B[0][0];
    R[0][1] = B[0][1];
    R[0][2] = B[0][2];
    R[1][0] = fmaf(a, B[1][0], b * B[2][0]);
    R[1][1] = fmaf(a, B[1][1], b * B[2][1]);
    R[1][2] = fmaf(a, B[1][2], b * B[2][2]);
    R[2][0] = fmaf(-b, B[1][0], a * B[2][0]);
    R[2][1] = fmaf(-b, B[1][1], a * B[2][1]);
    R[2][2] = fmaf(-b, B[1][2], a * B[2][2]);
    // construct the cumulative rotation Q=Q1 * Q2 * Q3
    // the number of floating point operations for three quaternion multiplications
    // is more or less comparable to the explicit form of the joined matrix.
    // certainly more memory-efficient!
    let sh12 = 2. * fmaf(g1_sh, g1_sh, -0.5);
    let sh22 = 2. * fmaf(g2_sh, g2_sh, -0.5);
    let sh32 = 2. * fmaf(g3_sh, g3_sh, -0.5);
    Q[0][0] = sh12 * sh22;
    Q[0][1] = fmaf(
        4. * g2_ch * g3_ch,
        sh12 * g2_sh * g3_sh,
        2. * g1_ch * g1_sh * sh32,
    );
    Q[0][2] = fmaf(
        4. * g1_ch * g3_ch,
        g1_sh * g3_sh,
        -2. * g2_ch * sh12 * g2_sh * sh32,
    );

    Q[1][0] = -2. * g1_ch * g1_sh * sh22;
    Q[1][1] = fmaf(
        -8. * g1_ch * g2_ch * g3_ch,
        g1_sh * g2_sh * g3_sh,
        sh12 * sh32,
    );
    Q[1][2] = fmaf(
        -2. * g3_ch,
        g3_sh,
        4. * g1_sh * fmaf(g3_ch * g1_sh, g3_sh, g1_ch * g2_ch * g2_sh * sh32),
    );

    Q[2][0] = 2. * g2_ch * g2_sh;
    Q[2][1] = -2. * g3_ch * sh22 * g3_sh;
    Q[2][2] = sh22 * sh32;
    [Q, R]
}
