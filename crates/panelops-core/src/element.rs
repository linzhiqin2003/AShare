//! Fused elementwise operations with scalar broadcasts and missing propagation.
use rayon::prelude::*;
#[derive(Clone, Copy)]
pub struct Inputs<'a> {
    pub panels: [&'a [f64]; 3],
    pub scalars: [f64; 3],
    pub scalar: [bool; 3],
    pub mask: &'a [bool],
    pub length: usize,
    pub lower: f64,
    pub upper: f64,
}
impl Inputs<'_> {
    #[inline(always)]
    fn at(&self, k: usize, i: usize) -> f64 {
        if self.scalar[k] {
            self.scalars[k]
        } else {
            self.panels[k][i]
        }
    }
}
pub fn valid(op: &str) -> bool {
    matches!(
        op,
        "gt" | "ge"
            | "lt"
            | "le"
            | "eq"
            | "ne"
            | "and_"
            | "or_"
            | "not_"
            | "where"
            | "maximum"
            | "minimum"
            | "sign"
            | "power"
            | "signed_power"
            | "sqrt"
            | "exp"
            | "clip"
            | "is_nan"
            | "fill_nan"
            | "add"
            | "subtract"
            | "multiply"
            | "divide"
            | "log"
            | "abs"
            | "negate"
    )
}
pub fn run(inputs: Inputs<'_>, op: &str) -> Vec<f64> {
    match op {
        "gt" => kernel::<0>(inputs),
        "ge" => kernel::<1>(inputs),
        "lt" => kernel::<2>(inputs),
        "le" => kernel::<3>(inputs),
        "eq" => kernel::<4>(inputs),
        "ne" => kernel::<5>(inputs),
        "and_" => kernel::<6>(inputs),
        "or_" => kernel::<7>(inputs),
        "not_" => kernel::<8>(inputs),
        "where" => kernel::<9>(inputs),
        "maximum" => kernel::<10>(inputs),
        "minimum" => kernel::<11>(inputs),
        "sign" => kernel::<12>(inputs),
        "power" => kernel::<13>(inputs),
        "signed_power" => kernel::<14>(inputs),
        "sqrt" => kernel::<15>(inputs),
        "exp" => kernel::<16>(inputs),
        "clip" => kernel::<17>(inputs),
        "is_nan" => kernel::<18>(inputs),
        "fill_nan" => kernel::<19>(inputs),
        "add" => kernel::<20>(inputs),
        "subtract" => kernel::<21>(inputs),
        "multiply" => kernel::<22>(inputs),
        "divide" => kernel::<23>(inputs),
        "log" => kernel::<24>(inputs),
        "abs" => kernel::<25>(inputs),
        "negate" => kernel::<26>(inputs),
        _ => unreachable!(),
    }
}
fn kernel<const O: usize>(inputs: Inputs<'_>) -> Vec<f64> {
    let mut out = vec![0.; inputs.length];
    if (O == 18 || O == 19)
        && !inputs.scalar[0]
        && (O == 18 || !inputs.scalar[1])
        && inputs.mask.is_empty()
    {
        // Validate each input slice once per contiguous block, allowing a vectorized zip.
        out.par_chunks_mut(65536)
            .enumerate()
            .for_each(|(block, row)| {
                let start = block * 65536;
                let end = start + row.len();
                let left = &inputs.panels[0][start..end];
                if O == 18 {
                    for (dest, &a) in row.iter_mut().zip(left) {
                        *dest = f64::from(!a.is_finite());
                    }
                    return;
                }
                let right = &inputs.panels[1][start..end];
                for ((dest, &a), &b) in row.iter_mut().zip(left).zip(right) {
                    let av = a.is_finite();
                    let take = 0u64.wrapping_sub(u64::from(av));
                    let selected = f64::from_bits((a.to_bits() & take) | (b.to_bits() & !take));
                    *dest = if av | b.is_finite() {
                        selected
                    } else {
                        f64::NAN
                    };
                }
            });
        return out;
    }
    out.par_iter_mut().enumerate().for_each(|(i, dest)| {
        if !inputs.mask.is_empty() && !inputs.mask[i] {
            *dest = f64::NAN;
            return;
        }
        let a = inputs.at(0, i);
        let b = inputs.at(1, i);
        let av = a.is_finite();
        let bv = b.is_finite();
        let z = match O {
            0 => {
                if av && bv {
                    f64::from(a > b)
                } else {
                    f64::NAN
                }
            }
            1 => {
                if av && bv {
                    f64::from(a >= b)
                } else {
                    f64::NAN
                }
            }
            2 => {
                if av && bv {
                    f64::from(a < b)
                } else {
                    f64::NAN
                }
            }
            3 => {
                if av && bv {
                    f64::from(a <= b)
                } else {
                    f64::NAN
                }
            }
            4 => {
                if av && bv {
                    f64::from(a == b)
                } else {
                    f64::NAN
                }
            }
            5 => {
                if av && bv {
                    f64::from(a != b)
                } else {
                    f64::NAN
                }
            }
            6 => {
                if av && bv {
                    f64::from(a != 0. && b != 0.)
                } else {
                    f64::NAN
                }
            }
            7 => {
                if av && bv {
                    f64::from(a != 0. || b != 0.)
                } else {
                    f64::NAN
                }
            }
            8 => {
                if av {
                    f64::from(a == 0.)
                } else {
                    f64::NAN
                }
            }
            9 => {
                let c = inputs.at(2, i);
                // A bit selection avoids unpredictable branches for balanced conditions.
                // Both operands have validated shapes; only the selected bits affect validity.
                let take = 0u64.wrapping_sub(u64::from(a != 0.));
                let selected = f64::from_bits((b.to_bits() & take) | (c.to_bits() & !take));
                if av { selected } else { f64::NAN }
            }
            10 => {
                if av && bv {
                    a.max(b)
                } else {
                    f64::NAN
                }
            }
            11 => {
                if av && bv {
                    a.min(b)
                } else {
                    f64::NAN
                }
            }
            12 => {
                if av {
                    f64::from(a > 0.) - f64::from(a < 0.)
                } else {
                    f64::NAN
                }
            }
            13 => {
                if av && bv {
                    a.powf(b)
                } else {
                    f64::NAN
                }
            }
            14 => {
                if av && bv {
                    (f64::from(a > 0.) - f64::from(a < 0.)) * a.abs().powf(b)
                } else {
                    f64::NAN
                }
            }
            15 => {
                if av {
                    a.sqrt()
                } else {
                    f64::NAN
                }
            }
            16 => {
                if av {
                    a.exp()
                } else {
                    f64::NAN
                }
            }
            17 => {
                if av {
                    a.clamp(inputs.lower, inputs.upper)
                } else {
                    f64::NAN
                }
            }
            18 => f64::from(!av),
            19 => {
                let take = 0u64.wrapping_sub(u64::from(av));
                f64::from_bits((a.to_bits() & take) | (b.to_bits() & !take))
            }
            20 => {
                if av && bv {
                    a + b
                } else {
                    f64::NAN
                }
            }
            21 => {
                if av && bv {
                    a - b
                } else {
                    f64::NAN
                }
            }
            22 => {
                if av && bv {
                    a * b
                } else {
                    f64::NAN
                }
            }
            23 => {
                if av && bv {
                    a / b
                } else {
                    f64::NAN
                }
            }
            24 => {
                if av {
                    a.ln()
                } else {
                    f64::NAN
                }
            }
            25 => {
                if av {
                    a.abs()
                } else {
                    f64::NAN
                }
            }
            26 => {
                if av {
                    -a
                } else {
                    f64::NAN
                }
            }
            _ => unreachable!(),
        };
        *dest = if z.is_finite() { z } else { f64::NAN };
    });
    out
}
