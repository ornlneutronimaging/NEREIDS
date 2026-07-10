use nereids_physics::resolution::TabulatedResolution;

fn main() {
    let text = "header\n-----\n1.0 0.0\n-1.0 0.2\n0.0 1.0\n1.0 0.2\n\n10.0 0.0\n-1.0 0.2\n0.0 1.0\n1.0 0.2\n";
    for fp in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        match TabulatedResolution::from_text(text, fp) {
            Ok(tab) => println!(
                "flight_path={fp:?}: constructor=Ok broaden={:?} support={:?}",
                tab.broaden(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.0])
                    .unwrap(),
                tab.kernel_support_ev(2.0)
            ),
            Err(e) => println!("flight_path={fp:?}: constructor=Err({e})"),
        }
    }
    let zero = TabulatedResolution::from_kernels(
        vec![1.0],
        vec![(vec![-1.0, 0.0, 1.0], vec![0.0, 0.0, 0.0])],
        25.0,
    )
    .unwrap();
    println!(
        "all_zero_kernel: constructor=Ok broaden={:?}",
        zero.broaden(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.0])
            .unwrap()
    );
}
