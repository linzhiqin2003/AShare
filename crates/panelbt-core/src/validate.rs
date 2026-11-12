//! Content validation and market preparation, parallel over fixed chunks.
//!
//! Every check reports only whether *some* value is invalid; messages carry no
//! position, so the scan may stop at the first bad chunk any thread finds.
//! Structural checks that protect memory safety live in the binding and run
//! whether or not content validation is requested.
use rayon::prelude::*;

/// Elements per parallel scan chunk. Tests place bad values on both sides of
/// these boundaries.
pub const CHUNK: usize = 1 << 14;

#[derive(Clone, Copy, Debug)]
pub enum Domain {
    /// Finite positive or NaN (prices).
    PositiveOrNan,
    /// Finite nonnegative or NaN (volume).
    NonnegativeOrNan,
    /// Finite positive (split multipliers).
    Positive,
    /// Finite nonnegative (dividends, rates).
    Nonnegative,
    /// Any finite value (target weights).
    Finite,
}

#[inline]
fn bad(x: f64, d: Domain) -> bool {
    match d {
        Domain::PositiveOrNan => x.is_infinite() || x <= 0.0,
        Domain::NonnegativeOrNan => x.is_infinite() || x < 0.0,
        Domain::Positive => !x.is_finite() || x <= 0.0,
        Domain::Nonnegative => !x.is_finite() || x < 0.0,
        Domain::Finite => !x.is_finite(),
    }
}

/// Inputs of at most one chunk are scanned on the calling thread: for small
/// plans the thread fan-out costs more than the scan.
pub const SERIAL: usize = CHUNK;

/// Scan `a` in `CHUNK`-sized pieces, in parallel when there is more than one.
fn any_chunk<T: Sync>(a: &[T], test: impl Fn(&[T]) -> bool + Sync + Send) -> bool {
    if a.len() <= SERIAL {
        a.chunks(CHUNK).any(test)
    } else {
        a.par_chunks(CHUNK).any(test)
    }
}

/// True when any value lies outside the domain.
pub fn any_bad(a: &[f64], d: Domain) -> bool {
    any_chunk(a, |c| c.iter().any(|&x| bad(x, d)))
}

/// Boolean panels are read as bytes; anything other than 0 or 1 is invalid.
pub fn any_nonbool(a: &[u8]) -> bool {
    any_chunk(a, |c| c.iter().any(|&x| x > 1))
}

/// True when any cell of the listed columns fails `test`.
pub fn any_in_columns<T: Copy + Sync>(
    a: &[T],
    n: usize,
    columns: &[usize],
    test: impl Fn(T) -> bool + Sync,
) -> bool {
    if columns.is_empty() || n == 0 {
        return false;
    }
    let rows = (CHUNK / n).max(1);
    let scan = |block: &[T]| {
        block
            .chunks(n)
            .any(|row| columns.iter().any(|&i| test(row[i])))
    };
    if a.len() <= SERIAL {
        a.chunks(rows * n).any(scan)
    } else {
        a.par_chunks(rows * n).any(scan)
    }
}

/// First row with a nonzero delist byte per column; `t` when never delisted.
pub fn first_delist(delist: &[u8], t: usize, n: usize) -> Vec<i64> {
    if delist.is_empty() {
        return vec![t as i64; n];
    }
    let rows = (CHUNK / n.max(1)).max(1);
    delist
        .par_chunks(rows * n)
        .enumerate()
        .fold(
            || vec![t as i64; n],
            |mut first, (block, cells)| {
                for (r, row) in cells.chunks(n).enumerate() {
                    let day = (block * rows + r) as i64;
                    for (i, &x) in row.iter().enumerate() {
                        if x != 0 && first[i] > day {
                            first[i] = day;
                        }
                    }
                }
                first
            },
        )
        .reduce(
            || vec![t as i64; n],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x = (*x).min(y);
                }
                a
            },
        )
}

/// Row-major CSR of cells with a split, a dividend or the first delist row.
pub fn action_index(
    split: &[f64],
    dividend: &[f64],
    first: &[i64],
    t: usize,
    n: usize,
) -> (Vec<i64>, Vec<i64>) {
    let delisted = first.iter().any(|&d| (d as usize) < t);
    if split.is_empty() && dividend.is_empty() && !delisted {
        return (vec![0; t + 1], Vec::new());
    }
    let active = |day: usize, i: usize| {
        let k = day * n + i;
        (!split.is_empty() && split[k] != 1.0)
            || (!dividend.is_empty() && dividend[k] != 0.0)
            || first[i] == day as i64
    };
    let counts: Vec<usize> = (0..t)
        .into_par_iter()
        .map(|day| (0..n).filter(|&i| active(day, i)).count())
        .collect();
    let mut indptr = Vec::with_capacity(t + 1);
    indptr.push(0i64);
    let mut total = 0usize;
    for &c in &counts {
        total += c;
        indptr.push(total as i64);
    }
    let mut cols = vec![0i64; total];
    let mut rest = cols.as_mut_slice();
    let mut segments = Vec::with_capacity(t);
    for &c in &counts {
        let (head, tail) = rest.split_at_mut(c);
        segments.push(head);
        rest = tail;
    }
    segments.into_par_iter().enumerate().for_each(|(day, out)| {
        let mut k = 0;
        for i in 0..n {
            if active(day, i) {
                out[k] = i as i64;
                k += 1;
            }
        }
    });
    (indptr, cols)
}

/// Facts about a sparse plan that the Python layer caches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanSummary {
    /// Columns strictly increase within every row (sorted and unique).
    pub sorted: bool,
    pub has_negative: bool,
    pub max_row_sum: f64,
}

/// Plan checks in the original NumPy order. `validate=false` keeps only the
/// pointer and index checks that slicing and indexing depend on.
pub fn check_plan(
    days: &[i64],
    indptr: &[i64],
    cols: &[i64],
    weights: &[f64],
    validate: bool,
) -> Result<PlanSummary, String> {
    if validate && (days.iter().any(|&d| d < 0) || days.windows(2).any(|w| w[1] <= w[0])) {
        return Err("days: expected nonnegative strictly increasing rebalance indices".into());
    }
    if indptr.len() != days.len() + 1
        || indptr[0] != 0
        || indptr[indptr.len() - 1] != cols.len() as i64
        || indptr.windows(2).any(|w| w[1] < w[0])
    {
        return Err("indptr: expected length R+1, nondecreasing from 0 to nnz".into());
    }
    if cols.len() != weights.len() || any_chunk(cols, |c| c.iter().any(|&x| x < 0)) {
        return Err("cols/weights: expected equal lengths and nonnegative cols".into());
    }
    if validate && any_bad(weights, Domain::Finite) {
        return Err("weights: expected finite targets".into());
    }
    let row = |w: &[i64]| {
        let (a, b) = (w[0] as usize, w[1] as usize);
        let sorted = cols[a..b].windows(2).all(|x| x[0] < x[1]);
        let mut sum = 0.0;
        let mut negative = false;
        for &x in &weights[a..b] {
            sum += x;
            negative |= x < 0.0;
        }
        (sorted, negative, sum)
    };
    let rows: Vec<(bool, bool, f64)> = if cols.len() <= SERIAL {
        indptr.windows(2).map(row).collect()
    } else {
        indptr.par_windows(2).map(row).collect()
    };
    Ok(PlanSummary {
        sorted: rows.iter().all(|r| r.0),
        has_negative: rows.iter().any(|r| r.1),
        max_row_sum: rows.iter().fold(0.0, |m: f64, r| m.max(r.2)),
    })
}

/// Account context for plan checks applied when a plan meets a market.
#[derive(Clone, Copy, Debug)]
pub struct Context<'a> {
    pub t: usize,
    pub n: usize,
    pub future: &'a [bool],
    pub allow_short: bool,
    /// Long-only unlevered weights: spot row sums must not exceed 1+1e-9.
    pub cash_weights: bool,
}

/// Context checks for one plan, in the original NumPy order. Bounds are always
/// checked; sign, integer and row-sum checks only for validated plans.
pub fn check_context(
    days: &[i64],
    indptr: &[i64],
    cols: &[i64],
    weights: &[f64],
    shares: bool,
    validate: bool,
    c: &Context<'_>,
) -> Result<(), String> {
    if days.iter().any(|&d| d < 0 || d as usize >= c.t) {
        return Err(format!("days: expected index < T={}", c.t));
    }
    if cols.iter().any(|&x| x < 0 || x as usize >= c.n) {
        return Err(format!("cols: expected index < N={}", c.n));
    }
    if !validate {
        return Ok(());
    }
    let future = |x: i64| !c.future.is_empty() && c.future[x as usize];
    if !c.allow_short
        && cols
            .iter()
            .zip(weights)
            .any(|(&x, &w)| w < 0.0 && !future(x))
    {
        return Err("weights: negative targets require allow_short".into());
    }
    if shares
        && cols
            .iter()
            .zip(weights)
            .any(|(&x, &w)| future(x) && w != w.trunc())
    {
        return Err("futures shares: target contracts must be integers".into());
    }
    if !shares && c.cash_weights {
        for w in indptr.windows(2) {
            let (a, b) = (w[0] as usize, w[1] as usize);
            let mut sum = 0.0;
            for k in a..b {
                sum += if future(cols[k]) { 0.0 } else { weights[k] };
            }
            if sum > 1.0 + 1e-9 {
                return Err("weights: cash account requires row sum <= 1 + 1e-9".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domains_match_numpy_rules() {
        let nan = f64::NAN;
        assert!(!any_bad(&[1.0, nan], Domain::PositiveOrNan));
        assert!(any_bad(&[0.0], Domain::PositiveOrNan));
        assert!(any_bad(&[f64::NEG_INFINITY], Domain::PositiveOrNan));
        assert!(!any_bad(&[0.0, nan], Domain::NonnegativeOrNan));
        assert!(any_bad(&[nan], Domain::Positive));
        assert!(any_bad(&[nan], Domain::Nonnegative));
        assert!(!any_bad(&[-3.0], Domain::Finite));
    }

    #[test]
    fn every_chunk_edge_is_scanned() {
        for k in [0, CHUNK - 1, CHUNK, 2 * CHUNK - 1, 2 * CHUNK, 2 * CHUNK + 6] {
            let mut a = vec![1.0; 2 * CHUNK + 7];
            a[k] = -1.0;
            assert!(any_bad(&a, Domain::Positive), "{k}");
            let mut b = vec![0u8; 2 * CHUNK + 7];
            b[k] = 2;
            assert!(any_nonbool(&b), "{k}");
        }
    }

    #[test]
    fn delist_and_actions_follow_row_major_order() {
        let (t, n) = (3, 2);
        let first = first_delist(&[0, 0, 1, 0, 1, 1], t, n);
        assert_eq!(first, vec![1, 2]);
        let split = [1.0, 2.0, 1.0, 1.0, 1.0, 1.0];
        let (p, c) = action_index(&split, &[], &first, t, n);
        assert_eq!(p, vec![0, 1, 2, 3]);
        assert_eq!(c, vec![1, 0, 1]);
    }
}
