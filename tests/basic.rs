use svd::{F, diag_to_mat, matmul, qr_decomp, svd, transpose};

#[test]
fn qr_ident_test() {
    let mut b = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    let [q, r] = qr_decomp(&mut b);
    assert_eq!(b, q);
    assert_eq!(b, r);
}

#[test]
fn test_more_qr() {
    let mut v = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let og_v = v.clone();
    let [q, r] = qr_decomp(&mut v);
    // matmul r, q = v
    let out = matmul(r, q);
    for i in 0..3 {
        for j in 0..3 {
            assert!((out[i][j] - og_v[i][j]).abs() < 1e-5);
        }
    }
}

#[test]
fn svd_ident_test() {
    let b = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    let [s, v, d] = svd(b);
    assert_eq!(s, b);
    assert_eq!(d, b);
    assert_eq!(v, b);
}

#[test]
fn svd_more_test() {
    let v = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let [u, sigma, v] = svd(v);
    println!("{sigma:?}");
    for i in 0..3 {
      assert!(dot(v[i], v[(i+1)%3]).abs() < 1e-6);
    }

    let new_v = matmul(matmul(transpose(v), sigma), u);
    println!("{new_v:?}");

    todo!();
}

pub fn dot([a, b, c]: [F; 3], [d, e, f]: [F; 3]) -> F {
    a * d + b * e + c * f
}
