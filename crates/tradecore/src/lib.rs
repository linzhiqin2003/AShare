//! Small, allocation-free spot execution primitives shared by trading engines.
//! Inputs must be finite, nonnegative and validated by the caller. Money is in
//! one currency. No arrays, clock, I/O, Python, or exchange-specific rules.
#![forbid(unsafe_code)]

/// Broker commission plus directional tax and other fees.
#[derive(Clone, Copy, Debug, Default)]
pub struct FeeTerms {
    pub commission_rate: f64,
    pub commission_per_share: f64,
    pub commission_min: f64,
    pub commission_max_rate: Option<f64>,
    pub tax_rate: f64,
    pub fee_rate: f64,
    pub fee_per_share: f64,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Fees {
    pub commission: f64,
    pub tax: f64,
    pub other: f64,
}
impl FeeTerms {
    /// Charge a nonzero fill. The commission cap takes precedence over its floor.
    #[inline]
    pub fn charge(self, quantity: f64, price: f64) -> Fees {
        let amount = quantity * price;
        if amount <= 0.0 || quantity <= 0.0 {
            return Fees::default();
        }
        let mut commission = (self.commission_rate * amount + self.commission_per_share * quantity)
            .max(self.commission_min);
        if let Some(cap) = self.commission_max_rate {
            commission = commission.min(cap * amount);
        }
        Fees {
            commission,
            tax: self.tax_rate * amount,
            other: self.fee_rate * amount + self.fee_per_share * quantity,
        }
    }
}
/// Round a positive opening quantity down. step=0 permits fractional shares.
#[inline]
pub fn round_open(quantity: f64, minimum: f64, step: f64) -> f64 {
    if step == 0.0 {
        quantity.max(0.0)
    } else if quantity < minimum {
        0.0
    } else {
        minimum + ((quantity - minimum) / step).floor() * step
    }
}
/// A complete close may include odd lots; partial closes use the quantity step.
#[inline]
pub fn round_close(quantity: f64, position: f64, minimum: f64, step: f64) -> f64 {
    if quantity >= position || step == 0.0 {
        quantity.min(position).max(0.0)
    } else if quantity < minimum {
        0.0
    } else {
        (quantity / step).floor() * step
    }
}
/// Balance-sheet values. Future variation margin can change cash independently
/// of spot principal; derivative margin belongs in `additional_margin` below.
#[derive(Clone, Copy, Debug, Default)]
pub struct Account {
    pub cash: f64,
    pub long: f64,
    pub short: f64,
}
impl Account {
    #[inline]
    pub fn equity(self) -> f64 {
        self.cash + self.long - self.short
    }
    #[inline]
    pub fn own_cash(self, short_proceeds_usable: bool) -> f64 {
        self.cash
            - if short_proceeds_usable {
                0.0
            } else {
                self.short
            }
    }
    #[inline]
    pub fn margin(self, long_rate: f64, short_rate: f64, additional_margin: f64) -> f64 {
        long_rate * self.long + short_rate * self.short + additional_margin
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cap_wins() {
        let x = FeeTerms {
            commission_per_share: 0.01,
            commission_min: 5.0,
            commission_max_rate: Some(0.01),
            ..Default::default()
        }
        .charge(10.0, 2.0);
        assert_eq!(x.commission, 0.2);
    }
    #[test]
    fn lots_and_account() {
        assert_eq!(round_open(201.9, 200.0, 1.0), 201.0);
        assert_eq!(round_close(51.5, 51.5, 100.0, 100.0), 51.5);
        let a = Account {
            cash: 100.0,
            long: 200.0,
            short: 50.0,
        };
        assert_eq!(a.equity(), 250.0);
        assert_eq!(a.own_cash(false), 50.0);
        assert_eq!(a.margin(0.5, 1.0, 10.0), 160.0);
    }
}
