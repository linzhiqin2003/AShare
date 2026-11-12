//! Preserve every assertion from the original panelbt lot-rounding test.
use tradecore::round_open;

#[test]
fn original_lots() {
    assert_eq!(round_open(199.0, 200.0, 1.0), 0.0);
    assert_eq!(round_open(201.9, 200.0, 1.0), 201.0);
    assert_eq!(round_open(75.5, 100.0, 0.0), 75.5);
    assert_eq!(round_open(299.9, 100.0, 100.0), 200.0);
}
