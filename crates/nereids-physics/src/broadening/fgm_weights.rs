//! SAMMY-inspired free-gas Doppler weight generation (`Modsmp` / `Modfpl`).
//!
//! This module ports the numerical weight construction strategy used in
//! SAMMY `mfgm2.f90` for convolution in velocity space.

const EXPMAX: f64 = 85.0;
const ONE: f64 = 1.0;
const TWO: f64 = 2.0;

#[inline]
fn safe_div(num: f64, den: f64) -> f64 {
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

#[inline]
fn exp_neg_sq(x: f64) -> f64 {
    let e = x * x;
    if e > EXPMAX {
        0.0
    } else {
        (-e).exp()
    }
}

fn abcexp(x: f64) -> (f64, f64, f64, f64, f64, i32) {
    let small = 0.99_f64;
    let half = 0.5_f64;
    let six = 6.0_f64;
    if x.abs() < small {
        let mut y = 0.0_f64;
        let mut z = 1.0 / 24.0;
        for i in 5..=60 {
            y += z;
            z = z * x / (i as f64);
            if y + z == y {
                break;
            }
        }
        let d = y;
        let c = ONE / six + d * x;
        let b = half + c * x;
        let a = ONE + b * x;
        let q = ONE + a * x;
        (q, a, b, c, d, 0)
    } else {
        let q = x.exp();
        let a = safe_div(q - ONE, x);
        let b = safe_div(a - ONE, x);
        let c = safe_div(b - 0.5, x);
        let d = safe_div(c - (ONE / 6.0), x);
        (q, a, b, c, d, 1)
    }
}

fn asympt(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let e = ONE / x;
    if e == 0.0 {
        return 0.0;
    }
    let mut a = ONE;
    let b = ONE / (x * x);
    let mut c = b * 0.5;
    for n in 1..=40 {
        a -= c;
        c = -((n as f64) + 0.5) * b * c;
        if a - c == a {
            break;
        }
        if (c / a).abs() < 1.0e-8 {
            break;
        }
    }
    a * e
}

fn exerfc(x: f64) -> f64 {
    // Generates exp(x^2) * erfc(x) * sqrt(pi)
    let sqrtpi = std::f64::consts::PI.sqrt();
    let tsqrpi = 2.0 * sqrtpi;
    let a1 = 8.584_076_57e-1;
    let a2 = 3.078_181_93e-1;
    let a3 = 6.383_238_91e-2;
    let a4 = 1.824_050_75e-4;
    let a5 = 6.509_742_65e-1;
    let a6 = 2.294_848_19e-1;
    let a7 = 3.403_018_23e-2;
    let xmax = 5.01_f64;

    if x < 0.0 {
        let xn = -x;
        if xn > xmax {
            let a = asympt(xn);
            tsqrpi - a
        } else {
            let a =
                (a1 + xn * (a2 + xn * (a3 - xn * a4))) / (ONE + xn * (a5 + xn * (a6 + xn * a7)));
            let b = sqrtpi + xn * (TWO - a);
            let a2v = safe_div(b, xn * b + ONE);
            tsqrpi * (x * x).exp() - a2v
        }
    } else if x > xmax {
        asympt(x)
    } else if x != 0.0 {
        let a = (a1 + x * (a2 + x * (a3 - x * a4))) / (ONE + x * (a5 + x * (a6 + x * a7)));
        let b = sqrtpi + x * (TWO - a);
        safe_div(b, x * b + ONE)
    } else {
        sqrtpi
    }
}

fn aaaerf(x_in: f64, y_in: f64, z: f64) -> (f64, f64, i32) {
    // Generates [Erfc(x-y)-Erfc(x)] * sqrt(pi)/2 * exp(z)
    let sqrtpi = std::f64::consts::PI.sqrt();
    let tsqrpi = 2.0 * sqrtpi;
    let yy_min = 0.001_f64;

    let mut x = x_in;
    let mut y = y_in;
    let mut sign = 1.0_f64;
    if y_in < 0.0 {
        x = -x_in;
        y = -y_in;
        sign = -1.0;
    }

    if y < yy_min {
        return (0.0, 0.0, 1);
    }

    let x2 = x * x;
    let d = if x > 0.0 {
        let exx = (-x2 + z).exp() * exerfc(x);
        let xy = x - y;
        let exxy = if xy > 0.0 {
            (-(xy * xy) + z).exp() * exerfc(xy)
        } else {
            let xyp = -xy;
            tsqrpi * z.exp() - (-(xyp * xyp) + z).exp() * exerfc(xyp)
        };
        (exxy - exx) / TWO
    } else if x < 0.0 {
        let xy = -x + y;
        let exx = (-(xy * xy) + z).exp() * exerfc(xy);
        let xm = -x;
        let exxy = (-(xm * xm) + z).exp() * exerfc(xm);
        (exxy - exx) / TWO
    } else if y < 0.0 {
        let xyp = -y;
        ((-(xyp * xyp) + z).exp() * exerfc(xyp) - sqrtpi * z.exp()) / TWO
    } else {
        let xyp = y;
        (sqrtpi * z.exp() - (-(xyp * xyp) + z).exp() * exerfc(xyp)) / TWO
    };

    let a = safe_div(d, y);
    (sign * d, a, 0)
}

fn abcerf(x: f64, y: f64) -> (f64, f64, f64, f64, i32) {
    // Generates [Erfc(x-y)-Erfc(x)] * sqrt(pi)/2 * exp(x^2)
    let xymax = 0.90_f64;
    let yy_min = 0.001_f64;

    if (x * y).abs() < xymax && y.abs() < ONE {
        let x2 = x * x;
        let mut c = (-ONE + TWO * x2) / 3.0;
        let mut d = y;
        let mut f2 = c;
        let f = x * (-0.5 + x2 / 3.0);
        if c + f * d == c {
            let b = x + y * c;
            let a = ONE + y * b;
            return (a * y, a, b, c, 0);
        }

        c += f * d;
        d *= y;
        let mut f1 = f;
        let mut a0 = 4.0;
        let mut a1 = 4.0;
        let mut a2 = 3.0;
        for _ in 5..=100 {
            a0 += ONE;
            let f0 = (TWO / a0) * (x * f1 - a2 * f2 / a1);
            if c + f0 * d == c {
                break;
            }
            c += f0 * d;
            f2 = f1;
            f1 = f0;
            a2 = a1;
            a1 = a0;
            d *= y;
        }

        let b = x + y * c;
        let a = ONE + y * b;
        (a * y, a, b, c, 0)
    } else if y.abs() < yy_min {
        (0.0, 0.0, 0.0, 0.0, 2)
    } else {
        let (q, a, _n) = aaaerf(x, y, x * x);
        let b = safe_div(a - ONE, y);
        let c = safe_div(b - x, y);
        (q, a, b, c, 1)
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct FgmState {
    xm1: f64,
    x00: f64,
    y00: f64,
    em1: f64,
    e00: f64,
    z: f64,
    w: f64,
    im1: isize,
    i00: isize,
    xm2: f64,
    xp1: f64,
    ym1: f64,
    yp1: f64,
    rm1: f64,
    r00: f64,
    rp1: f64,
    s00: f64,
    sp1: f64,
    im2: isize,
    ip1: isize,
    itype: i32,
}

struct FgmCalculator<'a> {
    velocities: &'a [f64],
    kc0: usize,
    ipnts: usize,
    naux: usize,
    vv: f64,
    ddo: f64,
    wts: Vec<f64>,
    st: FgmState,
}

impl<'a> FgmCalculator<'a> {
    fn new(velocities: &'a [f64], kc0: usize, ipnts: usize, vv: f64, ddo: f64) -> Self {
        Self {
            velocities,
            kc0,
            ipnts,
            naux: velocities.len(),
            vv,
            ddo,
            wts: vec![0.0; ipnts + 4],
            st: FgmState::default(),
        }
    }

    #[inline]
    fn kc1(&self) -> isize {
        (self.kc0 + 1) as isize
    }

    #[inline]
    fn vel_local(&self, local: isize) -> Option<f64> {
        let idx = self.kc0 as isize + local - 1;
        if idx < 0 || (idx as usize) >= self.naux {
            None
        } else {
            Some(self.velocities[idx as usize])
        }
    }

    #[inline]
    fn x_from_local(&self, local: isize) -> f64 {
        self.vel_local(local)
            .map(|v| safe_div(v - self.vv, self.ddo))
            .unwrap_or(0.0)
    }

    #[inline]
    fn w_add(&mut self, idx: isize, val: f64) {
        if idx >= 1 {
            let i = idx as usize;
            if i < self.wts.len() {
                self.wts[i] += val;
            }
        }
    }

    #[inline]
    fn w_set(&mut self, idx: isize, val: f64) {
        if idx >= 1 {
            let i = idx as usize;
            if i < self.wts.len() {
                self.wts[i] = val;
            }
        }
    }

    fn normalize(&mut self) -> bool {
        let mut s = 0.0;
        let mut iz = false;
        for i in 1..=self.ipnts {
            s += self.wts[i];
            if self.wts[i] != 0.0 {
                iz = true;
            }
        }
        if s == 0.0 || !iz {
            return false;
        }
        let inv_s = ONE / s;
        for i in 1..=self.ipnts {
            let v = self.vel_local(i as isize).unwrap_or(0.0);
            self.wts[i] = self.wts[i] * inv_s * v * v;
        }
        true
    }

    fn resets_modsmp(&mut self) {
        self.st.xm1 = self.st.x00;
        self.st.em1 = self.st.e00;
        self.st.im1 = self.st.i00;
        self.st.i00 += 1;
        self.st.x00 = self.x_from_local(self.st.i00);
        self.st.y00 = self.st.x00 - self.st.xm1;
        self.st.w = self.st.x00 + self.st.xm1;
        self.st.z = self.st.y00 * self.st.w;
        self.st.e00 = exp_neg_sq(self.st.x00);
    }

    fn modsmp(&mut self) -> bool {
        // For sorted source windows, this collapses to local index 1.
        self.st.i00 = 1;
        self.st.x00 = self.x_from_local(self.st.i00);
        self.st.e00 = exp_neg_sq(self.st.x00);
        self.w_set(self.st.i00, 0.0);
        self.resets_modsmp();

        self.st.itype = if self.st.x00 > 0.0 { 1 } else { 0 };

        loop {
            if self.st.x00 > 0.0 {
                self.st.itype = 1;
            }
            if self.st.itype == 0 {
                let (_q1, _a1, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                bb *= self.st.w * self.st.w;
                self.w_add(
                    self.st.im1,
                    self.st.e00 * self.st.y00 * (ONE + TWO * self.st.x00 * b - bb),
                );
                self.w_set(
                    self.st.i00,
                    self.st.e00 * self.st.y00 * (ONE - TWO * self.st.xm1 * b + bb),
                );
            } else if self.st.em1 != 0.0 {
                let (_q1, _a1, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
                bb *= self.st.w * self.st.w;
                self.w_add(
                    self.st.im1,
                    self.st.em1 * self.st.y00 * (ONE + TWO * self.st.x00 * b + bb),
                );
                self.w_set(
                    self.st.i00,
                    self.st.em1 * self.st.y00 * (ONE - TWO * self.st.xm1 * b - bb),
                );
            }

            self.resets_modsmp();
            if self.st.im1 == self.ipnts as isize {
                break;
            }
            let idx1 = self.st.i00 + self.kc1() - 1;
            if idx1 > self.naux as isize {
                break;
            }
        }

        self.normalize()
    }

    fn reset_modfpl(&mut self) {
        self.st.im2 = self.st.im1;
        self.st.im1 = self.st.i00;
        self.st.i00 = self.st.ip1;
        self.st.ip1 = self.st.i00 + 1;

        self.st.xm2 = self.st.xm1;
        self.st.xm1 = self.st.x00;
        self.st.x00 = self.st.xp1;
        self.st.xp1 = self.x_from_local(self.st.ip1);
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }

        self.st.ym1 = self.st.y00;
        self.st.y00 = self.st.yp1;
        self.st.yp1 = self.st.xp1 - self.st.x00;

        self.st.w = self.st.x00 + self.st.xm1;
        self.st.z = self.st.y00 * self.st.w;

        self.st.rm1 = self.st.r00;
        self.st.r00 = self.st.rp1;
        self.st.rp1 = ONE + TWO * self.st.xp1 * self.st.x00;

        self.st.s00 = self.st.sp1;
        self.st.sp1 = ONE + TWO * self.st.xp1 * self.st.xm1;

        self.st.em1 = self.st.e00;
        self.st.e00 = exp_neg_sq(self.st.x00);
    }

    fn run_modfpl(&mut self) -> bool {
        for i in 1..=self.ipnts {
            self.wts[i] = 0.0;
        }

        let kc1 = self.kc1();
        if kc1 == 1 {
            self.start1();
        } else if kc1 == 2 {
            self.start2();
        } else if kc1 == 3 {
            self.start3();
        } else {
            self.start4();
        }

        if self.st.ip1 <= self.ipnts as isize {
            loop {
                self.wtabcd();
                if self.st.im1 < self.ipnts as isize {
                    self.reset_modfpl();
                    if self.st.ip1 > self.ipnts as isize {
                        break;
                    }
                    continue;
                }
                break;
            }
        }

        if self.st.em1 != 0.0 {
            let iq = self.ipnts as isize + kc1 + 2;
            if iq <= self.naux as isize {
                self.quit4();
            } else if iq == self.naux as isize + 1 {
                self.quit3();
            } else if iq == self.naux as isize + 2 {
                self.quit2();
            } else if iq == self.naux as isize + 3 {
                self.quit1();
            }
        }

        self.normalize()
    }

    fn wtxxxd(&mut self) {
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }
        if self.st.itype == 0 {
            if self.st.e00 != 0.0 {
                let (_q1, _a1, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                bb *= self.st.w * self.st.w;
                let d = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                let term = d
                    * safe_div(self.st.y00, self.st.yp1)
                    * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                    * self.st.e00;
                self.w_set(self.st.ip1, term);
            }
        } else if self.st.em1 != 0.0 {
            let (_q1, _a1, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
            let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
            bb *= self.st.w * self.st.w;
            let d = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
            let term = d
                * safe_div(self.st.y00, self.st.yp1)
                * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                * self.st.em1;
            self.w_set(self.st.ip1, term);
        }
    }

    fn wtxxcd(&mut self) {
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }
        if self.st.itype == 0 {
            if self.st.e00 != 0.0 {
                let (_q1, _a1, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                bb *= self.st.w * self.st.w;
                let d = self.st.rm1 * b - self.st.xm2 - self.st.xm2 * bb;
                let e = -self.st.sp1 * b + self.st.xp1 + self.st.xp1 * bb;
                let term_i00 = (safe_div(d, self.st.y00 + self.st.ym1) + safe_div(e, self.st.yp1))
                    * self.st.y00
                    * self.st.e00;
                self.w_add(self.st.i00, term_i00);
                let d2 = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                let term_ip1 = d2
                    * safe_div(self.st.y00, self.st.yp1)
                    * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                    * self.st.e00;
                self.w_set(self.st.ip1, term_ip1);
            }
        } else if self.st.em1 != 0.0 {
            let (_q1, mut aa, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
            let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
            aa *= self.st.w;
            bb *= self.st.w * self.st.w;
            let d = self.st.rm1 * b - self.st.xm2 + self.st.xm2 * bb + aa;
            let e = -self.st.sp1 * b + self.st.xp1 - self.st.xp1 * bb - aa;
            let term_i00 = (safe_div(d, self.st.y00 + self.st.ym1) + safe_div(e, self.st.yp1))
                * self.st.y00
                * self.st.em1;
            self.w_add(self.st.i00, term_i00);
            let d2 = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
            let term_ip1 = d2
                * safe_div(self.st.y00, self.st.yp1)
                * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                * self.st.em1;
            self.w_set(self.st.ip1, term_ip1);
        }
    }

    fn wtxbcd(&mut self) {
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }
        if self.st.itype == 0 {
            if self.st.e00 != 0.0 {
                let (_q1, mut aa, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                aa *= self.st.w;
                bb *= self.st.w * self.st.w;
                let d = -self.st.s00 * b - self.st.xm2 + self.st.xm2 * bb + aa;
                let e = self.st.rp1 * b + self.st.xp1 - self.st.xp1 * bb - aa;
                let term_im1 = (safe_div(d, self.st.ym1) + safe_div(e, self.st.yp1 + self.st.y00))
                    * self.st.y00
                    * self.st.e00;
                self.w_add(self.st.im1, term_im1);
                let d2 = self.st.rm1 * b - self.st.xm2 - self.st.xm2 * bb;
                let e2 = -self.st.sp1 * b + self.st.xp1 + self.st.xp1 * bb;
                let term_i00 = (safe_div(d2, self.st.y00 + self.st.ym1)
                    + safe_div(e2, self.st.yp1))
                    * self.st.y00
                    * self.st.e00;
                self.w_add(self.st.i00, term_i00);
                let d3 = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                let term_ip1 = d3
                    * safe_div(self.st.y00, self.st.yp1)
                    * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                    * self.st.e00;
                self.w_set(self.st.ip1, term_ip1);
            }
        } else if self.st.em1 != 0.0 {
            let (_q1, mut aa, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
            let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
            aa *= self.st.w;
            bb *= self.st.w * self.st.w;
            let d = -self.st.s00 * b - self.st.xm2 - self.st.xm2 * bb;
            let e = self.st.rp1 * b + self.st.xp1 + self.st.xp1 * bb;
            let term_im1 = (safe_div(d, self.st.ym1) + safe_div(e, self.st.y00 + self.st.yp1))
                * self.st.y00
                * self.st.em1;
            self.w_add(self.st.im1, term_im1);
            let d2 = self.st.rm1 * b - self.st.xm2 + self.st.xm2 * bb + aa;
            let e2 = -self.st.sp1 * b + self.st.xp1 - self.st.xp1 * bb - aa;
            let term_i00 = (safe_div(d2, self.st.y00 + self.st.ym1) + safe_div(e2, self.st.yp1))
                * self.st.y00
                * self.st.em1;
            self.w_add(self.st.i00, term_i00);
            let d3 = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
            let term_ip1 = d3
                * safe_div(self.st.y00, self.st.yp1)
                * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                * self.st.em1;
            self.w_set(self.st.ip1, term_ip1);
        }
    }

    fn wtabcd(&mut self) {
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }
        if self.st.itype == 0 {
            if self.st.e00 != 0.0 {
                let (_q1, mut aa, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                aa *= self.st.w;
                bb *= self.st.w * self.st.w;
                let d_im2 = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                self.w_add(
                    self.st.im2,
                    d_im2
                        * safe_div(self.st.y00, self.st.ym1)
                        * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                        * self.st.e00,
                );
                let d = -self.st.s00 * b - self.st.xm2 + self.st.xm2 * bb + aa;
                let e = self.st.rp1 * b + self.st.xp1 - self.st.xp1 * bb - aa;
                self.w_add(
                    self.st.im1,
                    (safe_div(d, self.st.ym1) + safe_div(e, self.st.yp1 + self.st.y00))
                        * self.st.y00
                        * self.st.e00,
                );
                let d2 = self.st.rm1 * b - self.st.xm2 - self.st.xm2 * bb;
                let e2 = -self.st.sp1 * b + self.st.xp1 + self.st.xp1 * bb;
                self.w_add(
                    self.st.i00,
                    (safe_div(d2, self.st.y00 + self.st.ym1) + safe_div(e2, self.st.yp1))
                        * self.st.y00
                        * self.st.e00,
                );
                let d3 = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                self.w_set(
                    self.st.ip1,
                    d3 * safe_div(self.st.y00, self.st.yp1)
                        * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                        * self.st.e00,
                );
            }
        } else if self.st.em1 != 0.0 {
            let (_q1, mut aa, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
            let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
            aa *= self.st.w;
            bb *= self.st.w * self.st.w;
            let d_im2 = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
            self.w_add(
                self.st.im2,
                d_im2
                    * safe_div(self.st.y00, self.st.ym1)
                    * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                    * self.st.em1,
            );
            let d = -self.st.s00 * b - self.st.xm2 - self.st.xm2 * bb;
            let e = self.st.rp1 * b + self.st.xp1 + self.st.xp1 * bb;
            self.w_add(
                self.st.im1,
                (safe_div(d, self.st.ym1) + safe_div(e, self.st.y00 + self.st.yp1))
                    * self.st.y00
                    * self.st.em1,
            );
            let d2 = self.st.rm1 * b - self.st.xm2 + self.st.xm2 * bb + aa;
            let e2 = -self.st.sp1 * b + self.st.xp1 - self.st.xp1 * bb - aa;
            self.w_add(
                self.st.i00,
                (safe_div(d2, self.st.y00 + self.st.ym1) + safe_div(e2, self.st.yp1))
                    * self.st.y00
                    * self.st.em1,
            );
            let d3 = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
            self.w_set(
                self.st.ip1,
                d3 * safe_div(self.st.y00, self.st.yp1)
                    * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                    * self.st.em1,
            );
        }
    }

    fn wtabcx(&mut self) {
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }
        if self.st.itype == 0 {
            if self.st.e00 != 0.0 {
                let (_q1, mut aa, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                aa *= self.st.w;
                bb *= self.st.w * self.st.w;
                let d_im2 = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                self.w_add(
                    self.st.im2,
                    d_im2
                        * safe_div(self.st.y00, self.st.ym1)
                        * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                        * self.st.e00,
                );
                let d = -self.st.s00 * b - self.st.xm2 + self.st.xm2 * bb + aa;
                let e = self.st.rp1 * b + self.st.xp1 - self.st.xp1 * bb - aa;
                self.w_add(
                    self.st.im1,
                    (safe_div(d, self.st.ym1) + safe_div(e, self.st.yp1 + self.st.y00))
                        * self.st.y00
                        * self.st.e00,
                );
                let d2 = self.st.rm1 * b - self.st.xm2 - self.st.xm2 * bb;
                let e2 = -self.st.sp1 * b + self.st.xp1 + self.st.xp1 * bb;
                self.w_add(
                    self.st.i00,
                    (safe_div(d2, self.st.y00 + self.st.ym1) + safe_div(e2, self.st.yp1))
                        * self.st.y00
                        * self.st.e00,
                );
            }
        } else if self.st.em1 != 0.0 {
            let (_q1, mut aa, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
            let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
            aa *= self.st.w;
            bb *= self.st.w * self.st.w;
            let d_im2 = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
            self.w_add(
                self.st.im2,
                d_im2
                    * safe_div(self.st.y00, self.st.ym1)
                    * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                    * self.st.em1,
            );
            let d = -self.st.s00 * b - self.st.xm2 - self.st.xm2 * bb;
            let e = self.st.rp1 * b + self.st.xp1 + self.st.xp1 * bb;
            self.w_add(
                self.st.im1,
                (safe_div(d, self.st.ym1) + safe_div(e, self.st.y00 + self.st.yp1))
                    * self.st.y00
                    * self.st.em1,
            );
            let d2 = self.st.rm1 * b - self.st.xm2 + self.st.xm2 * bb + aa;
            let e2 = -self.st.sp1 * b + self.st.xp1 - self.st.xp1 * bb - aa;
            self.w_add(
                self.st.i00,
                (safe_div(d2, self.st.y00 + self.st.ym1) + safe_div(e2, self.st.yp1))
                    * self.st.y00
                    * self.st.em1,
            );
        }
    }

    fn wtabxx(&mut self) {
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }
        if self.st.itype == 0 {
            if self.st.e00 != 0.0 {
                let (_q1, mut aa, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                aa *= self.st.w;
                bb *= self.st.w * self.st.w;
                let d_im2 = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                self.w_add(
                    self.st.im2,
                    d_im2
                        * safe_div(self.st.y00, self.st.ym1)
                        * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                        * self.st.e00,
                );
                let d = -self.st.s00 * b - self.st.xm2 + self.st.xm2 * bb + aa;
                let e = self.st.rp1 * b + self.st.xp1 - self.st.xp1 * bb - aa;
                self.w_add(
                    self.st.im1,
                    (safe_div(d, self.st.ym1) + safe_div(e, self.st.yp1 + self.st.y00))
                        * self.st.y00
                        * self.st.e00,
                );
            }
        } else if self.st.em1 != 0.0 {
            let (_q1, _aa, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
            let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
            bb *= self.st.w * self.st.w;
            let d_im2 = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
            self.w_add(
                self.st.im2,
                d_im2
                    * safe_div(self.st.y00, self.st.ym1)
                    * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                    * self.st.em1,
            );
            let d = -self.st.s00 * b - self.st.xm2 - self.st.xm2 * bb;
            let e = self.st.rp1 * b + self.st.xp1 + self.st.xp1 * bb;
            self.w_add(
                self.st.im1,
                (safe_div(d, self.st.ym1) + safe_div(e, self.st.y00 + self.st.yp1))
                    * self.st.y00
                    * self.st.em1,
            );
        }
    }

    fn wtaxxx(&mut self) {
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }
        if self.st.itype == 0 {
            if self.st.e00 != 0.0 {
                let (_q1, _a1, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                bb *= self.st.w * self.st.w;
                let d = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                self.w_add(
                    self.st.im2,
                    d * safe_div(self.st.y00, self.st.ym1)
                        * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                        * self.st.e00,
                );
            }
        } else if self.st.em1 != 0.0 {
            let (_q1, _a1, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
            let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
            bb *= self.st.w * self.st.w;
            let d = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
            self.w_add(
                self.st.im2,
                d * safe_div(self.st.y00, self.st.ym1)
                    * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                    * self.st.em1,
            );
        }
    }

    fn start4(&mut self) {
        self.st.im2 = -2;
        self.st.im1 = self.st.im2 + 1;
        self.st.i00 = self.st.im1 + 1;
        self.st.ip1 = self.st.i00 + 1;
        self.st.xm2 = self.x_from_local(self.st.im2);
        self.st.xm1 = self.x_from_local(self.st.im1);
        self.st.x00 = self.x_from_local(self.st.i00);
        self.st.xp1 = self.x_from_local(self.st.ip1);
        self.st.ym1 = self.st.xm1 - self.st.xm2;
        self.st.y00 = self.st.x00 - self.st.xm1;
        self.st.yp1 = self.st.xp1 - self.st.x00;
        self.st.w = self.st.x00 + self.st.xm1;
        self.st.z = self.st.y00 * self.st.w;
        self.st.rm1 = ONE + TWO * self.st.xm1 * self.st.xm2;
        self.st.r00 = ONE + TWO * self.st.x00 * self.st.xm1;
        self.st.rp1 = ONE + TWO * self.st.xp1 * self.st.x00;
        self.st.s00 = ONE + TWO * self.st.x00 * self.st.xm2;
        self.st.sp1 = ONE + TWO * self.st.xp1 * self.st.xm1;
        self.st.em1 = exp_neg_sq(self.st.xm1);
        self.st.e00 = exp_neg_sq(self.st.x00);
        self.st.itype = 0;
        self.wtxxxd();
        self.reset_modfpl();
        self.wtxxcd();
        self.reset_modfpl();
        self.wtxbcd();
        self.reset_modfpl();
    }

    fn start3(&mut self) {
        self.st.im2 = -2;
        self.st.im1 = self.st.im2 + 1;
        self.st.i00 = self.st.im1 + 1;
        self.st.ip1 = self.st.i00 + 1;
        self.st.xm1 = self.x_from_local(self.st.im1);
        self.st.x00 = self.x_from_local(self.st.i00);
        self.st.xp1 = self.x_from_local(self.st.ip1);
        self.st.y00 = self.st.x00 - self.st.xm1;
        self.st.yp1 = self.st.xp1 - self.st.x00;
        self.st.w = self.st.x00 + self.st.xm1;
        self.st.z = self.st.y00 * self.st.w;
        self.st.r00 = ONE + TWO * self.st.x00 * self.st.xm1;
        self.st.rp1 = ONE + TWO * self.st.xp1 * self.st.x00;
        self.st.sp1 = ONE + TWO * self.st.xp1 * self.st.xm1;
        self.st.em1 = exp_neg_sq(self.st.xm1);
        self.st.e00 = exp_neg_sq(self.st.x00);
        self.st.itype = 0;
        self.wtzxxd();
        self.reset_modfpl();
        self.wtxxcd();
        self.reset_modfpl();
        self.wtxbcd();
        self.reset_modfpl();
    }

    fn start2(&mut self) {
        self.st.im2 = -2;
        self.st.im1 = self.st.im2 + 1;
        self.st.i00 = self.st.im1 + 1;
        self.st.ip1 = self.st.i00 + 1;
        self.st.x00 = self.x_from_local(self.st.i00);
        self.st.xp1 = self.x_from_local(self.st.ip1);
        self.st.yp1 = self.st.xp1 - self.st.x00;
        self.st.rp1 = ONE + TWO * self.st.xp1 * self.st.x00;
        self.st.e00 = exp_neg_sq(self.st.x00);
        self.st.itype = 0;
        self.reset_modfpl();
        self.wtzxcd();
        self.reset_modfpl();
        self.wtxbcd();
        self.reset_modfpl();
    }

    fn start1(&mut self) {
        self.st.im2 = -2;
        self.st.im1 = self.st.im2 + 1;
        self.st.i00 = self.st.im1 + 1;
        self.st.ip1 = self.st.i00 + 1;
        self.st.xp1 = self.x_from_local(self.st.ip1);
        self.st.itype = 0;
        self.reset_modfpl();
        self.reset_modfpl();
        self.wtzbcd();
        self.reset_modfpl();
    }

    fn wtzxxd(&mut self) {
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }
        if self.st.itype == 0 {
            if self.st.e00 != 0.0 {
                let (_q1, _a1, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                bb *= self.st.w * self.st.w;
                let d = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                self.w_set(
                    self.st.ip1,
                    d * safe_div(self.st.y00, self.st.yp1)
                        * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                        * self.st.e00
                        * TWO,
                );
            }
        } else if self.st.em1 != 0.0 {
            let (_q1, _a1, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
            let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
            bb *= self.st.w * self.st.w;
            let d = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
            self.w_set(
                self.st.ip1,
                d * safe_div(self.st.y00, self.st.yp1)
                    * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                    * self.st.em1
                    * TWO,
            );
        }
    }

    fn wtzxcd(&mut self) {
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }
        if self.st.itype == 0 {
            if self.st.e00 != 0.0 {
                let (_q1, _a1, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                bb *= self.st.w * self.st.w;
                let e = -self.st.sp1 * b + self.st.xp1 + self.st.xp1 * bb;
                self.w_set(
                    self.st.i00,
                    safe_div(e, self.st.yp1) * self.st.y00 * self.st.e00 * TWO,
                );
                if self.st.ip1 <= self.ipnts as isize {
                    let d = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                    self.w_set(
                        self.st.ip1,
                        d * safe_div(self.st.y00, self.st.yp1)
                            * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                            * self.st.e00
                            * TWO,
                    );
                }
            }
        } else if self.st.em1 != 0.0 {
            let (_q1, mut aa, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
            let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
            aa *= self.st.w;
            bb *= self.st.w * self.st.w;
            let e = -self.st.sp1 * b + self.st.xp1 - self.st.xp1 * bb - aa;
            self.w_set(
                self.st.i00,
                safe_div(e, self.st.yp1) * self.st.y00 * self.st.em1 * TWO,
            );
            if self.st.ip1 <= self.ipnts as isize {
                let d = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
                self.w_set(
                    self.st.ip1,
                    d * safe_div(self.st.y00, self.st.yp1)
                        * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                        * self.st.em1
                        * TWO,
                );
            }
        }
    }

    fn wtzbcd(&mut self) {
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }
        if self.st.itype == 0 {
            if self.st.e00 != 0.0 {
                let (_q1, mut aa, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                aa *= self.st.w;
                bb *= self.st.w * self.st.w;
                let e1 = self.st.rp1 * b + self.st.xp1 - self.st.xp1 * bb - aa;
                self.w_set(
                    self.st.im1,
                    safe_div(e1, self.st.yp1 + self.st.y00) * self.st.y00 * self.st.e00 * TWO,
                );
                let e2 = -self.st.sp1 * b + self.st.xp1 + self.st.xp1 * bb;
                self.w_set(
                    self.st.i00,
                    safe_div(e2, self.st.yp1) * self.st.y00 * self.st.e00 * TWO,
                );
                if self.st.ip1 <= self.ipnts as isize {
                    let d = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                    self.w_set(
                        self.st.ip1,
                        d * safe_div(self.st.y00, self.st.yp1)
                            * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                            * self.st.e00
                            * TWO,
                    );
                }
            }
        } else if self.st.em1 != 0.0 {
            let (_q1, mut aa, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
            let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
            aa *= self.st.w;
            bb *= self.st.w * self.st.w;
            let e1 = self.st.rp1 * b + self.st.xp1 + self.st.xp1 * bb;
            self.w_set(
                self.st.im1,
                safe_div(e1, self.st.y00 + self.st.yp1) * self.st.y00 * self.st.em1 * TWO,
            );
            let e2 = -self.st.sp1 * b + self.st.xp1 - self.st.xp1 * bb - aa;
            self.w_set(
                self.st.i00,
                safe_div(e2, self.st.yp1) * self.st.y00 * self.st.em1 * TWO,
            );
            if self.st.ip1 <= self.ipnts as isize {
                let d = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
                self.w_set(
                    self.st.ip1,
                    d * safe_div(self.st.y00, self.st.yp1)
                        * safe_div(self.st.y00, self.st.yp1 + self.st.y00)
                        * self.st.em1
                        * TWO,
                );
            }
        }
    }

    fn wtabcz(&mut self) {
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }
        if self.st.itype == 0 {
            if self.st.e00 != 0.0 {
                let (_q1, mut aa, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                aa *= self.st.w;
                bb *= self.st.w * self.st.w;
                let d_im2 = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                self.w_add(
                    self.st.im2,
                    d_im2
                        * safe_div(self.st.y00, self.st.ym1)
                        * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                        * self.st.e00
                        * TWO,
                );
                let d_im1 = -self.st.s00 * b - self.st.xm2 + self.st.xm2 * bb + aa;
                self.w_add(
                    self.st.im1,
                    safe_div(d_im1, self.st.ym1) * self.st.y00 * self.st.e00,
                );
                let d_i00 = self.st.rm1 * b - self.st.xm2 - self.st.xm2 * bb;
                self.w_add(
                    self.st.i00,
                    safe_div(d_i00, self.st.y00 + self.st.ym1) * self.st.y00 * self.st.e00,
                );
            }
        } else if self.st.em1 != 0.0 {
            let (_q1, mut aa, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
            let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
            aa *= self.st.w;
            bb *= self.st.w * self.st.w;
            let d_im2 = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
            self.w_add(
                self.st.im2,
                d_im2
                    * safe_div(self.st.y00, self.st.ym1)
                    * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                    * self.st.em1
                    * TWO,
            );
            let d_im1 = -self.st.s00 * b - self.st.xm2 - self.st.xm2 * bb;
            self.w_add(
                self.st.im1,
                safe_div(d_im1, self.st.ym1) * self.st.y00 * self.st.em1 * TWO,
            );
            let d_i00 = self.st.rm1 * b - self.st.xm2 + self.st.xm2 * bb + aa;
            self.w_add(
                self.st.i00,
                safe_div(d_i00, self.st.y00 + self.st.ym1) * self.st.y00 * self.st.em1,
            );
        }
    }

    fn wtabxz(&mut self) {
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }
        if self.st.itype == 0 {
            if self.st.e00 != 0.0 {
                let (_q1, mut aa, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                aa *= self.st.w;
                bb *= self.st.w * self.st.w;
                let d_im2 = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                self.w_add(
                    self.st.im2,
                    d_im2
                        * safe_div(self.st.y00, self.st.ym1)
                        * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                        * self.st.e00
                        * TWO,
                );
                let d_im1 = -self.st.s00 * b - self.st.xm2 + self.st.xm2 * bb + aa;
                self.w_add(
                    self.st.im1,
                    safe_div(d_im1, self.st.ym1) * self.st.y00 * self.st.e00,
                );
            }
        } else if self.st.em1 != 0.0 {
            let (_q1, _aa, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
            let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
            bb *= self.st.w * self.st.w;
            let d_im2 = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
            self.w_add(
                self.st.im2,
                d_im2
                    * safe_div(self.st.y00, self.st.ym1)
                    * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                    * self.st.em1
                    * TWO,
            );
            let d_im1 = -self.st.s00 * b - self.st.xm2 - self.st.xm2 * bb;
            self.w_add(
                self.st.im1,
                safe_div(d_im1, self.st.ym1) * self.st.y00 * self.st.em1 * TWO,
            );
        }
    }

    fn wtaxxz(&mut self) {
        if self.st.x00 > 0.0 {
            self.st.itype = 1;
        }
        if self.st.itype == 0 {
            if self.st.e00 != 0.0 {
                let (_q1, _a1, b, _c1, _n1) = abcerf(self.st.x00, self.st.y00);
                let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(self.st.z);
                bb *= self.st.w * self.st.w;
                let d_im2 = self.st.r00 * b - self.st.x00 - self.st.x00 * bb;
                self.w_add(
                    self.st.im2,
                    d_im2
                        * safe_div(self.st.y00, self.st.ym1)
                        * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                        * self.st.e00
                        * TWO,
                );
            }
        } else if self.st.em1 != 0.0 {
            let (_q1, _a1, b, _c1, _n1) = abcerf(-self.st.xm1, self.st.y00);
            let (_q2, _a2, mut bb, _c2, _d2, _n2) = abcexp(-self.st.z);
            bb *= self.st.w * self.st.w;
            let d_im2 = self.st.r00 * b + self.st.xm1 + self.st.xm1 * bb;
            self.w_add(
                self.st.im2,
                d_im2
                    * safe_div(self.st.y00, self.st.ym1)
                    * safe_div(self.st.y00, self.st.y00 + self.st.ym1)
                    * self.st.em1
                    * TWO,
            );
        }
    }

    fn quit4(&mut self) {
        self.wtabcx();
        self.reset_modfpl();
        if self.st.em1 != 0.0 {
            self.wtabxx();
            self.reset_modfpl();
            if self.st.em1 != 0.0 {
                self.wtaxxx();
            }
        }
    }

    fn quit3(&mut self) {
        self.wtabcx();
        self.reset_modfpl();
        if self.st.em1 != 0.0 {
            self.wtabxx();
            self.reset_modfpl();
            if self.st.em1 != 0.0 {
                self.wtaxxz();
            }
        }
    }

    fn quit2(&mut self) {
        self.wtabcx();
        self.reset_modfpl();
        if self.st.em1 != 0.0 {
            self.wtabxz();
        }
    }

    fn quit1(&mut self) {
        self.wtabcz();
    }
}

/// Compute `Σ_i w_i * xs_i` where `w_i` are SAMMY-style free-gas Doppler
/// weights on a local source-grid window.
pub fn fgm_weighted_sum(
    velocities: &[f64],
    xs: &[f64],
    kc0: usize,
    ipnts: usize,
    vv: f64,
    ddo: f64,
) -> Option<f64> {
    if ipnts <= 2 || ddo <= 0.0 {
        return None;
    }
    if velocities.len() != xs.len() {
        return None;
    }
    if kc0 >= velocities.len() || kc0 + ipnts > velocities.len() {
        return None;
    }

    let mut calc = FgmCalculator::new(velocities, kc0, ipnts, vv, ddo);
    let ok = if ipnts <= 3 {
        calc.modsmp()
    } else {
        calc.run_modfpl()
    };
    if !ok {
        return None;
    }

    let mut sum = 0.0;
    for i in 1..=ipnts {
        let gi = kc0 + i - 1;
        sum += calc.wts[i] * xs[gi];
    }
    Some(sum)
}
