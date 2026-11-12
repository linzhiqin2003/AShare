"""Review mutation IDs mapped onto the streaming implementation, plus new contracts."""

T = "crates/panelops-core/src/time.rs"
R = "crates/panelops-core/src/lib.rs"
V = "crates/panelops-core/src/evaluation.rs"
E = "crates/panelops-core/src/element.rs"
A = "packages/panelops/python/panelops/api.py"
G = "src/ashare/factor/engine.py"
X = "src/ashare/factor/expression.py"
L = "src/ashare/factor/library.py"
CASES = []


def case(name, path, old, new, scope=None, tests=None):
    CASES.append(
        dict(
            name=name,
            edits=[dict(path=path, old=old, new=new, scope=scope)],
            tests=tests
            or (["test/factor"] if path in (G, X, L) else ["test/panelops"]),
        )
    )


def extra(path, old, new, scope=None):
    CASES[-1]["edits"].append(dict(path=path, old=old, new=new, scope=scope))


case(
    "R01 sum outgoing value",
    T,
    "add(&mut main[k], &mut correction[k], -a);",
    "add(&mut main[k], &mut correction[k], 0.);",
    "fn sum_rows",
)
case(
    "R02 sum never rebuild (periodic and event)",
    T,
    "if t % 512 == 0 {",
    "if t == 0 {",
    "fn sum_rows",
)
extra(
    T,
    "if !value.is_finite() || 8. * f64::EPSILON * peak[k] > 1e-11 * (value.abs() + 1.) {",
    "if false {",
    "fn sum_rows",
)
case(
    "R03 paired moments never rebuild",
    T,
    "if t % 128 == 0 || risk {",
    "if M != 5 && (t % 128 == 0 || risk) {",
    "fn moment_block",
)
extra(T, "if t % 512 == 0 || risk {", "if false {", "fn std_rows")
case(
    "R04 moment outgoing coefficient",
    T,
    "let sign = if positive { 1. } else { -1. };",
    "let sign = if positive { 1. } else { -0.999999 };",
)
case(
    "R05 ordered min_periods",
    T,
    "counts[k] < p.min_periods",
    "counts[k] + 1 < p.min_periods",
    "fn ordered",
)
case(
    "R06 rank missing today",
    T,
    "if !ok {\n                    continue;\n                }",
    "if false {\n                    continue;\n                }",
    "fn ordered",
)
case(
    "R07 time rank minimum ties",
    T,
    "let rank = less as f64 + (equal + 1) as f64 / 2.;",
    "let rank = less as f64 + 1.;",
    "fn ordered",
)
case(
    "R08 cross rank normalization",
    R,
    "(rank - 1.) / (ids.len() - 1) as f64",
    "(rank - 1.) / ids.len() as f64",
    "fn rank_in_place",
)
case("R09 extrema earliest ties", T, "x >= prefix[k]", "x > prefix[k]", "fn extrema")
extra(T, "x <= prefix[k]", "x < prefix[k]", "fn extrema")
case(
    "R10 argmax position instead of age",
    T,
    "(t - index) as f64",
    "(p.window - (t - index)) as f64",
    "fn extrema",
)
case(
    "R11 decay rebuilt weights reversed",
    T,
    "let w = (p.window - (t - r)) as f64;",
    "let w = (1 + t - r) as f64;",
    "fn sums",
)
case(
    "R12 decay incoming weight",
    T,
    "weighted[k].add(x * p.window as f64);",
    "weighted[k].add(x * (p.window - 1) as f64);",
    "fn sums",
)
case(
    "R13 EMA swapped alpha",
    T,
    "p.alpha * x + (1. - p.alpha) * state[k]",
    "(1. - p.alpha) * x + p.alpha * state[k]",
)
case(
    "R14 regression predictor variance",
    T,
    "let slope = cross / vy;",
    "let slope = cross / vx;",
    "fn moment_result",
)
case(
    "R15 regression residual sign",
    T,
    "x - s.ox - a[0] / c - slope * (y - s.oy - a[1] / c)",
    "x - s.ox - a[0] / c + slope * (y - s.oy - a[1] / c)",
    "fn moment_result",
)
case(
    "R16 winsor lower bound",
    R,
    "v.iter().map(|v| v.clamp(lo, hi)).collect()",
    "v.iter().map(|v| v.min(hi)).collect()",
    "pub fn cs",
)
case(
    "R17 MAD consistency multiplier",
    R,
    "(med - p.scale * mad, med + p.scale * mad)",
    "(med - p.scale * mad * 1.4826, med + p.scale * mad * 1.4826)",
    "pub fn cs",
)
case(
    "R18 negative group valid", R, "p.groups[i] >= 0", "p.groups[i] >= -1", "pub fn cs"
)
case("R19 unweighted design", R, "*v *= w.sqrt();", "*v *= 1. + 0. * w;", "fn residual")
case(
    "R20 rank IC returns unranked",
    R,
    "rank_in_place(y, order, false);",
    "// Removed return ranking.",
    "pub fn ic",
)
case(
    "R21 group boundaries",
    R,
    "out[j] = i * q / ids.len();",
    "out[j] = i * q / (ids.len() + 1);",
    "fn labels",
)
case(
    "R22 turnover wrong retained group",
    V,
    "usize::from(old[j] == g)",
    "usize::from(old[j] != usize::MAX)",
    "pub fn turnover",
)
case(
    "R23 targets k+1", R, "k.min(ids.len())", "(k + 1).min(ids.len())", "pub fn targets"
)
case(
    "R24 targets fraction floor",
    R,
    "(ids.len() as f64 * fraction).ceil() as usize",
    "(ids.len() as f64 * fraction).floor() as usize",
    "pub fn targets",
)
case(
    "R25 targets reversed tie break",
    R,
    ".then(a.cmp(&b))",
    ".then(b.cmp(&a))",
    "pub fn targets",
)
case(
    "R26 omitted final streamed row",
    T,
    "for (t, row) in rows.into_iter().enumerate() {",
    "for (t, row) in rows.into_iter().enumerate() {\n        if t + 1 == pan.t { continue; }",
    "fn sum_rows",
)
case("R27 delta reversed", T, "now[j] - old[j]", "old[j] - now[j]", "fn dispatch")
case(
    "R28 ignore historical mask",
    T,
    "D || (pan.valid(i) && x.is_finite() && (!paired || y.is_finite()))",
    "D || (x.is_finite() && (!paired || y.is_finite()))",
    "fn values",
)
case(
    "R29 kurtosis excess omitted",
    T,
    "m4 / denominator - 3.",
    "m4 / denominator",
    "fn moment_result",
)
case(
    "R30 skew central moment",
    T,
    "+ 2. * m * m * m;",
    "+ 3. * m * m * m;",
    "fn moment_result",
)
case(
    "R31 covariance ddof",
    T,
    "cross / (s.count - p.ddof) as f64",
    "cross / s.count as f64",
    "fn moment_result",
)
case(
    "R32 quantile interpolation",
    R,
    "sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)",
    "sorted[lo]",
    "fn quantile",
)
case(
    "R33 product sign ignored",
    T,
    ".filter(|v| v.is_finite())\n                        .product()",
    ".filter(|v| v.is_finite())\n                        .map(|v| v.abs())\n                        .product()",
    "fn ordered",
)
case(
    "R34 variance error guard removed",
    T,
    "error > 1e-11 * variance.abs()",
    "error > f64::INFINITY * variance.abs()",
    "fn std_rows",
)
case(
    "R35 cross zscore ddof",
    R,
    "(variance / (count - p.ddof) as f64).sqrt()",
    "(variance / count as f64).sqrt()",
    "pub fn cs",
)
case(
    "R36 higher moments never rebuild",
    T,
    "if t % 128 == 0 || risk {",
    "if M != 4 && (t % 128 == 0 || risk) {",
    "fn moment_block",
)
case("P01 EMA default alpha", A, "2 / (window + 1)", "1 / window")
case(
    "P02 autocorrelation lag",
    A,
    "_core.autocorr(x, mm, periods, rank, _THREADS)",
    "_core.autocorr(x, mm, periods + 1, rank, _THREADS)",
)
case("P03 cache data version", G, "            source.version,\n", "")
case("P04 cache build identity", G, "            panelops.build_id,\n", "")
case("P05 cache axes", G, "            axes,\n", "")
case(
    "P06 cache mask",
    G,
    "None if mask is None else hashlib.sha256(mask.tobytes()).hexdigest(),",
    "None,",
)
case("P07 cache universe", G, "            universe_id,\n", "")
case(
    "P08 arithmetic nonfinite output",
    E,
    "*dest = if z.is_finite() { z } else { f64::NAN };",
    "*dest = z;",
    tests=["test/factor", "test/panelops/test_streaming.py"],
)
case(
    "P09 unordered canonical parameters",
    X,
    "tuple(sorted(params.items()))",
    "tuple(params.items())",
)
case(
    "P10 nested lookback max not addition",
    X,
    'return base + params["window"] - 1',
    'return max(base, params["window"] - 1)',
)
case(
    "P11 delay lookback",
    X,
    'return base + params["periods"]',
    'return base + params["periods"] - 1',
)
case("P12 reversal sign", L, "-returns(window),", "returns(window),")
case(
    "P13 unadjusted prices",
    L,
    'return field("close") * field("adj_factor")',
    'return field("close") * 1',
)
case(
    "P14 momentum skipped month",
    L,
    'call("ts_delay", price, periods=21)',
    'call("ts_delay", price, periods=20)',
)
case(
    "P15 Amihud signed return",
    L,
    'call("abs", daily) / field("amount")',
    'daily / field("amount")',
)
case("P16 bare field mask", G, "valid &= mask", "valid &= True")
case(
    "P17 constant hash type",
    X,
    "value = [2, self.op, [a.digest for a in self.args], list(self.params)]",
    "value = [2, self.op, [a.digest for a in self.args], [(k, float(v) if type(v) is int else v) for k, v in self.params]]",
)

case(
    "N01 sum incoming update",
    T,
    "add(&mut main[k], &mut correction[k], a);",
    "add(&mut main[k], &mut correction[k], 0.);",
    "fn sum_rows",
)
case("N02 std outgoing square", T, "s2[k] -= d2;", "s2[k] -= 0. * d2;", "fn std_rows")
case("N03 std incoming square", T, "s2[k] += d2;", "s2[k] += 0. * d2;", "fn std_rows")
case(
    "N04 paired incoming update",
    T,
    "s.update::<O>(x, y, true);",
    "s.update::<O>(0., y, true);",
    "fn moment_block",
)
case(
    "N05 paired outgoing update",
    T,
    "s.update::<O>(x, y, false);",
    "s.update::<O>(0., y, false);",
    "fn moment_block",
)
case(
    "N06 decay outgoing update",
    T,
    "total[k].add(-if O == COUNT",
    "total[k].add(if O == COUNT",
    "fn sums",
)
case(
    "N07 std constant proof",
    T,
    "if run[k] >= c {",
    "if run[k] + 1 >= c {",
    "fn std_rows",
)
case(
    "N08 constant predictor proof",
    T,
    "self.count > 0 && self.run_y >= self.count",
    "self.count > 0 && self.run_y + 1 >= self.count",
)
case(
    "N09 cross error guard",
    T,
    "ec > 1e-11 * (cross.abs() + 1.)",
    "ec > f64::INFINITY * (cross.abs() + 1.)",
    "fn moment_result",
)
case(
    "N10 column block omitted",
    T,
    "let first = b * width;",
    "let first = b * width;\n        if b > 0 { return; }",
    "fn dispatch",
)
case(
    "N11 extrema outgoing boundary",
    T,
    "let i = (offset + 1) * n + k;",
    "let i = offset * n + k;",
    "fn extrema",
)
case(
    "N12 rank equal count",
    T,
    "let equal = values.iter().filter(|&&v| v == x).count();",
    "let equal = values.iter().filter(|&&v| v >= x).count();",
    "fn ordered",
)
case(
    "N13 long quantile order statistic",
    T,
    "let at = q * (self.size(self.root) - 1) as f64;",
    "let at = 0. * q * (self.size(self.root) - 1) as f64;",
)
case(
    "N14 MAD deviations",
    T,
    "*v = (*v - median).abs();",
    "*v = *v - median;",
    "fn ordered",
)
case("N15 count nonzero", T, "f64::from(x != 0.)", "f64::from(x > 0.)", "fn sum_rows")
case(
    "N16 sumif missing condition",
    T,
    "(O != SUMIF || y.is_finite())",
    "true",
    "fn sum_rows",
)
case(
    "N17 ffill limit boundary", T, "t - last[k] <= p.periods", "t - last[k] < p.periods"
)
case(
    "N18 rsquare square",
    T,
    "if O == RSQUARE { r * r } else { r }",
    "if O == RSQUARE { r } else { r }",
    "fn moment_result",
)
case(
    "N19 idx position origin",
    T,
    "(p.window - (t - index)) as f64",
    "(p.window - (t - index) + 1) as f64",
    "fn extrema",
)
case(
    "N20 variance output",
    T,
    "if O == VAR {\n                    v",
    "if O == VAR {\n                    v.sqrt()",
    "fn std_rows",
)
case(
    "N21 zscore missing today",
    T,
    "if ok {\n                        (x - origin[k] - mean) / v.sqrt()",
    "if true {\n                        (last[k] - origin[k] - mean) / v.sqrt()",
    "fn std_rows",
)
case(
    "N22 scale absolute norm",
    R,
    "if v.is_finite() { v.abs() } else { 0. }",
    "if v.is_finite() { v } else { 0. }",
    "pub fn cs",
)
case(
    "N23 demean centering",
    R,
    "let centered = (value - origin) - m;",
    "let centered = (value - origin) + m;",
    "pub fn cs",
)
case(
    "N24 median quantile",
    T,
    "if O == MEDIAN { 0.5 } else { p.q }",
    "if O == MEDIAN { 0.25 } else { p.q }",
    "fn ordered",
)
for i, op in enumerate(
    (
        "gt ge lt le eq ne and_ or_ not_ where maximum minimum sign power signed_power sqrt exp clip is_nan fill_nan"
    ).split()
):
    case(
        f"E{i + 1:02} {op} dispatch",
        E,
        f'"{op}" => kernel::<{i}>(inputs)',
        f'"{op}" => kernel::<26>(inputs)',
        tests=[
            f"test/panelops/test_streaming.py::test_elementwise_independent_reference[{op}]"
        ],
    )

case(
    "N25 unmasked std outgoing update",
    T,
    "*ss -= d * d;",
    "*ss -= 0. * d * d;",
    "fn std_rows",
)
case(
    "N26 unmasked std incoming update",
    T,
    "*ss += d * d;",
    "*ss += 0. * d * d;",
    "fn std_rows",
)

case(
    "N27 rank ring outgoing count",
    T,
    "counts[k] -= usize::from(old.is_finite());",
    "counts[k] -= 0;",
    "fn ordered",
    tests=["test/panelops/test_streaming.py::test_random_long_windows_rebuilds"],
)
case(
    "N28 long quantile tree expiration",
    T,
    "tr.advance(t, value);",
    "if t < p.window {tr.advance(t, value);}",
    "fn ordered",
)
case(
    "N29 sum output mask",
    T,
    "if counts[k] >= p.min_periods && (D || pan.valid(t * pan.n + first + k)) {",
    "if counts[k] >= p.min_periods {",
    "fn sum_rows",
)
case(
    "N30 neutralization intercept",
    R,
    "let dependent = demean(v);",
    "let dependent = v.clone();",
    "pub fn cs",
)
case(
    "N31 IC date alignment",
    R,
    "y.push(pan.y[i]);",
    "y.push(pan.y[(i + pan.n) % pan.y.len()]);",
    "pub fn ic",
)
case(
    "N32 CSR row pointer",
    R,
    "ptr.push(cols.len() as i64);",
    "ptr.push(cols.len() as i64 + 1);",
    "pub fn targets",
)
case(
    "N33 sum rebuild excludes today",
    T,
    "for r in (t + 1).saturating_sub(p.window)..=t {",
    "for r in (t + 1).saturating_sub(p.window)..t {",
    "fn sum_rows",
)
case(
    "N34 cancellation expansion disabled",
    T,
    "value = accurate_sum(contributions.into_iter());",
    "value = contributions.into_iter().sum();",
    "fn sum_rows",
    tests=[
        "test/panelops/test_adversarial.py::test_multiple_scales_cancel_without_losing_small_sum"
    ],
)

case(
    "P18 fingerprint omits Python",
    "packages/panelops/build.rs",
    "('binding/python',python_root,['**/*.py'])",
    "('binding/python',python_root,[])",
    tests=["test/panelops/test_build_id.py"],
)
case(
    "P19 fingerprint omits core",
    "packages/panelops/build.rs",
    "('core',core,['src/**/*.rs'])",
    "('core',core,[])",
    tests=["test/panelops/test_build_id.py"],
)

case(
    "N35 where bit selection",
    E,
    "u64::from(a != 0.)",
    "u64::from(a == 0.)",
    tests=[
        "test/panelops/test_streaming.py::test_elementwise_independent_reference[where]"
    ],
)

case(
    "N36 product singleton signed zero",
    T,
    "Op::Product if p.window == 1",
    "Op::Product if p.window == 0",
    tests=[
        "test/panelops/test_streaming.py::test_single_observation_product_preserves_finite_bits"
    ],
)
case(
    "N37 fill_nan bit selection",
    E,
    "u64::from(av)",
    "u64::from(!av)",
    tests=[
        "test/panelops/test_streaming.py::test_elementwise_independent_reference[fill_nan]"
    ],
)

case(
    "N38 fill_nan block boundary",
    E,
    "let start = block * 65536;",
    "let start = block * 65535;",
    tests=[
        "test/panelops/test_streaming.py::test_fill_nan_contiguous_blocks_and_threads"
    ],
)

case(
    "N39 is_nan contiguous validity",
    E,
    "*dest = f64::from(!a.is_finite());",
    "*dest = f64::from(a.is_finite());",
    tests=[
        "test/panelops/test_streaming.py::test_elementwise_independent_reference[is_nan]"
    ],
)
