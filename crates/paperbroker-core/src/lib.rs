//! Deterministic, IO-free cash account and snapshot matching engine.
#![forbid(unsafe_code)]
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const WIDTH: usize = 18;
pub const SCALE: i64 = 10_000;
const MAX_PRICE: i64 = 1_000_000_000;
const MAX_QTY: i64 = 1_000_000_000;
const MAX_CASH: i64 = 1_000_000_000_000_000_000;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Buy,
    Sell,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Preopen,
    OpenAuction,
    Continuous,
    Break,
    CloseAuction,
    Closed,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Limit,
    Market,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Reject {
    Session,
    UnknownSymbol,
    NoQuote,
    Suspended,
    Quantity,
    Tick,
    PriceBand,
    Cash,
    Sellable,
    UnknownOrder,
    Time,
    AlreadySettled,
    InvalidAction,
}

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Fees {
    pub commission: i64,
    pub tax: i64,
    pub other: i64,
}
impl Fees {
    pub fn total(&self) -> i64 {
        self.commission + self.tax + self.other
    }
    fn difference(&self, old: &Self) -> Self {
        Self {
            commission: self.commission - old.commission,
            tax: self.tax - old.tax,
            other: self.other - old.other,
        }
    }
    fn add(&mut self, x: &Self) {
        self.commission += x.commission;
        self.tax += x.tax;
        self.other += x.other;
    }
}
pub trait CostModel: Clone {
    fn cumulative(
        &self,
        symbol: usize,
        profile: Option<&Instrument>,
        side: Side,
        notional: i64,
    ) -> Fees;
}
pub trait LotRule: Clone {
    fn valid(
        &self,
        symbol: usize,
        profile: Option<&Instrument>,
        qty: i64,
        side: Side,
        available: i64,
    ) -> bool;
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleCosts {
    pub commission_ppm: i64,
    pub minimum: i64,
    pub sell_tax_ppm: i64,
}
fn rate(value: i64, ppm: i64) -> i64 {
    ((value as i128 * ppm as i128 + 999_999) / 1_000_000) as i64
}
impl CostModel for SimpleCosts {
    fn cumulative(
        &self,
        _symbol: usize,
        profile: Option<&Instrument>,
        side: Side,
        value: i64,
    ) -> Fees {
        let model = profile.map_or(self, |p| &p.costs);
        if value == 0 {
            return Fees::default();
        }
        Fees {
            commission: rate(value, model.commission_ppm).max(model.minimum),
            tax: if side == Side::Sell {
                rate(value, model.sell_tax_ppm)
            } else {
                0
            },
            other: 0,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleLots {
    pub step: i64,
    #[serde(default)]
    pub minimum: i64,
}
impl LotRule for SimpleLots {
    fn valid(
        &self,
        _symbol: usize,
        profile: Option<&Instrument>,
        qty: i64,
        side: Side,
        available: i64,
    ) -> bool {
        let model = profile.map_or(self, |p| &p.lots);
        qty > 0
            && qty <= MAX_QTY
            && ((qty >= model.minimum.max(model.step) && qty % model.step == 0)
                || (side == Side::Sell
                    && if model.step == 1 {
                        qty == available
                    } else {
                        qty % model.step == available % model.step
                    }))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instrument {
    pub name: String,
    pub profile_id: String,
    pub cost_model: String,
    pub lot_model: String,
    pub tick: i64,
    pub t_plus_one: bool,
    pub costs: SimpleCosts,
    pub lots: SimpleLots,
}
impl Instrument {
    fn valid(&self) -> bool {
        !self.name.is_empty()
            && !self.profile_id.is_empty()
            && self.cost_model == "simple-v1"
            && self.lot_model == "minimum-step-v1"
            && (1..=MAX_PRICE).contains(&self.tick)
            && (1..=MAX_QTY).contains(&self.lots.step)
            && (0..=MAX_QTY).contains(&self.lots.minimum)
            && (0..=1_000_000).contains(&self.costs.commission_ppm)
            && (0..=1_000_000).contains(&self.costs.sell_tax_ppm)
            && (0..=MAX_CASH / 4).contains(&self.costs.minimum)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub symbols: usize,
    #[serde(default)]
    pub instruments: Vec<Instrument>,
    #[serde(default)]
    pub utc_offset_seconds: i64,
    pub initial_cash: i64,
    pub tick: i64,
    pub participation_ppm: i64,
    pub t_plus_one: bool,
    pub enforce_bands: bool,
    pub slippage_ticks: i64,
    pub slippage_ppm: i64,
    pub costs: SimpleCosts,
    pub lots: SimpleLots,
    pub margin_model: String,
}
impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if (!self.instruments.is_empty()
            && (self.instruments.len() != self.symbols
                || self.instruments.iter().any(|p| !p.valid())))
            || self.utc_offset_seconds.unsigned_abs() > 86_400
            || self.symbols == 0
            || self.symbols > 100_000
            || !(0..=MAX_CASH).contains(&self.initial_cash)
            || self.tick <= 0
            || self.tick > MAX_PRICE
            || !(0..=1_000_000).contains(&self.participation_ppm)
            || self.slippage_ticks < 0
            || self.slippage_ticks > 10_000
            || !(0..=1_000_000).contains(&self.slippage_ppm)
            || !(1..=MAX_QTY).contains(&self.lots.step)
            || !(0..=1_000_000).contains(&self.costs.commission_ppm)
            || !(0..=1_000_000).contains(&self.costs.sell_tax_ppm)
            || !(0..=MAX_CASH / 4).contains(&self.costs.minimum)
            || self.margin_model != "cash"
        {
            return Err("invalid configuration or unsupported margin model".into());
        }
        Ok(())
    }
}
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Position {
    pub total: i64,
    pub sellable: i64,
    pub frozen: i64,
    pub cost: i64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Order {
    pub id: u64,
    pub symbol: usize,
    pub side: Side,
    pub kind: Kind,
    pub qty: i64,
    pub remaining: i64,
    pub price: i64,
    pub time: i64,
    pub reserve: i64,
    pub notional: i64,
    pub fees: Fees,
    pub status: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Action {
    pub symbol: usize,
    pub cash_per_share: i64,
    pub split_num: i64,
    pub split_den: i64,
    pub reference_price: Option<i64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Input {
    Session {
        time: i64,
        day: i32,
        phase: Phase,
    },
    Order {
        time: i64,
        symbol: usize,
        side: Side,
        kind: Kind,
        qty: i64,
        price: i64,
    },
    Cancel {
        time: i64,
        id: u64,
    },
    Settle {
        time: i64,
        day: i32,
        actions: Vec<Action>,
        interest: i64,
    },
    Corporate {
        time: i64,
        #[serde(default)]
        day: Option<i32>,
        actions: Vec<Action>,
    },
    Transfer {
        time: i64,
        amount: i64,
    },
    Append {
        time: i64,
        instruments: Vec<Instrument>,
    },
}
impl Input {
    fn time(&self) -> i64 {
        match self {
            Self::Session { time, .. }
            | Self::Order { time, .. }
            | Self::Cancel { time, .. }
            | Self::Settle { time, .. }
            | Self::Corporate { time, .. }
            | Self::Transfer { time, .. }
            | Self::Append { time, .. } => *time,
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct State {
    pub version: u32,
    pub config: Config,
    pub day: i32,
    pub time: i64,
    pub phase: Phase,
    pub cash: i64,
    pub frozen: i64,
    pub positions: Vec<Position>,
    pub quotes: Vec<Option<[i64; WIDTH]>>,
    pub corporate_done: Vec<bool>,
    pub corporate_dates: Vec<i32>,
    pub actions_through: i32,
    pub event_sequence: u64,
    pub order_offset: u64,
    pub auction_done: Vec<[bool; 2]>,
    pub marks: Vec<i64>,
    pub orders: Vec<Order>,
    pub fees: Fees,
    pub day_fees: Fees,
    pub flows: i64,
    pub settled_day: i32,
    pub quote_received: Vec<i64>,
}
#[derive(Clone)]
struct OutputQueue {
    day: i32,
    pending: Vec<Value>,
}
impl OutputQueue {
    fn push(&mut self, mut event: Value, sequence: u64) {
        event["seq"] = json!(sequence);
        event["day"] = json!(self.day);
        self.pending.push(event);
    }
}
#[derive(Clone)]
pub struct Engine<C: CostModel = SimpleCosts, L: LotRule = SimpleLots> {
    pub state: State,
    costs: C,
    lots: L,
    active: Vec<Vec<usize>>,
    events: OutputQueue,
}
impl Engine<SimpleCosts, SimpleLots> {
    pub fn new(config: Config) -> Result<Self, String> {
        config.validate()?;
        let n = config.symbols;
        let state = State {
            version: 3,
            cash: config.initial_cash,
            flows: config.initial_cash,
            config: config.clone(),
            day: 0,
            time: 0,
            phase: Phase::Closed,
            frozen: 0,
            positions: vec![Position::default(); n],
            quotes: vec![None; n],
            quote_received: vec![0; n],
            corporate_done: vec![false; n],
            corporate_dates: vec![0; n],
            actions_through: 0,
            event_sequence: 0,
            order_offset: 0,
            auction_done: vec![[false; 2]; n],
            marks: vec![0; n],
            orders: vec![],
            fees: Fees::default(),
            day_fees: Fees::default(),
            settled_day: 0,
        };
        Ok(Self::with_models(state, config.costs, config.lots))
    }
    pub fn restore(s: &str) -> Result<Self, String> {
        let state: State = serde_json::from_str(s).map_err(|_| "invalid state".to_string())?;
        state.config.validate()?;
        if state.version != 3
            || state.positions.len() != state.config.symbols
            || state.quotes.len() != state.config.symbols
            || state.marks.len() != state.config.symbols
            || state.quote_received.len() != state.config.symbols
            || state.auction_done.len() != state.config.symbols
            || state.corporate_done.len() != state.config.symbols
            || state.corporate_dates.len() != state.config.symbols
            || state.orders.iter().enumerate().any(|(i, o)| {
                o.symbol >= state.config.symbols || o.id != state.order_offset + i as u64 + 1
            })
        {
            return Err("invalid state layout".into());
        }
        Ok(Self::with_models(
            state.clone(),
            state.config.costs,
            state.config.lots,
        ))
    }
}
impl<C: CostModel, L: LotRule> Engine<C, L> {
    pub fn with_models(state: State, costs: C, lots: L) -> Self {
        let mut active = vec![vec![]; state.config.symbols];
        for (i, o) in state.orders.iter().enumerate() {
            if o.remaining > 0 && o.status == "open" {
                active[o.symbol].push(i);
            }
        }
        let day = state.day;
        Self {
            state,
            costs,
            lots,
            active,
            events: OutputQueue {
                day,
                pending: vec![],
            },
        }
    }
    pub fn dump(&self) -> String {
        serde_json::to_string(&self.state).expect("integer state")
    }
    pub fn drain(&mut self) -> Vec<Value> {
        std::mem::take(&mut self.events.pending)
    }
    pub fn account(&self) -> Value {
        let market_value: i128 = self
            .state
            .positions
            .iter()
            .zip(&self.state.marks)
            .map(|(p, price)| p.total as i128 * (*price) as i128)
            .sum();
        json!({"cash":self.state.cash,"frozen":self.state.frozen,"market_value":market_value,"equity":self.state.cash as i128+self.state.frozen as i128+market_value,"positions":self.state.positions,"fees":self.state.fees,"day_fees":self.state.day_fees,"flows":self.state.flows,"day":self.state.day,"phase":self.state.phase})
    }
    fn emit(&mut self, event: Value) {
        self.state.event_sequence += 1;
        self.events.push(event, self.state.event_sequence);
    }
    fn changed(&mut self, symbol: Option<usize>) {
        self.emit(json!({"type":"account","time":self.state.time,"cash":self.state.cash,"frozen":self.state.frozen,"symbol":symbol,"position":symbol.map(|s|self.state.positions[s].clone())}));
    }
    fn reject(&mut self, reason: Reject, id: Option<u64>) {
        self.emit(json!({"type":"rejected","time":self.state.time,"reason":reason,"id":id}));
    }
    fn reserve(&self, o: &Order) -> i64 {
        if o.side == Side::Sell || o.remaining == 0 {
            return 0;
        }
        o.remaining * o.price
            + self
                .costs
                .cumulative(
                    o.symbol,
                    self.state.config.instruments.get(o.symbol),
                    o.side,
                    o.notional + o.remaining * o.price,
                )
                .total()
            - o.fees.total()
    }
    pub fn apply(&mut self, input: Input) {
        if input.time() < self.state.time {
            self.reject(Reject::Time, None);
            return;
        }
        self.state.time = input.time();
        match input {
            Input::Append { instruments, .. } => {
                let existing: std::collections::BTreeSet<_> = self
                    .state
                    .config
                    .instruments
                    .iter()
                    .map(|p| p.name.clone())
                    .collect();
                let added: std::collections::BTreeSet<_> =
                    instruments.iter().map(|p| p.name.clone()).collect();
                if self.state.config.instruments.len() != self.state.config.symbols
                    || instruments
                        .iter()
                        .any(|p| !p.valid() || existing.contains(&p.name))
                    || added.len() != instruments.len()
                    || self.state.config.symbols + instruments.len() > 100_000
                {
                    self.reject(Reject::InvalidAction, None);
                    return;
                }
                for instrument in instruments {
                    self.state.config.instruments.push(instrument);
                    self.state.positions.push(Position::default());
                    self.state.quotes.push(None);
                    self.state.quote_received.push(0);
                    self.state.marks.push(0);
                    self.state.corporate_done.push(false);
                    self.state.corporate_dates.push(0);
                    self.state.auction_done.push([false; 2]);
                    self.active.push(vec![]);
                    self.state.config.symbols += 1;
                }
                self.emit(json!({"type":"appended", "time":self.state.time, "symbols":self.state.config.symbols}));
            }
            Input::Session { day, phase, .. } => {
                if day < self.state.day
                    || (day != self.state.day && self.state.day != self.state.settled_day)
                {
                    self.reject(Reject::Session, None);
                    return;
                }
                if day != self.state.day {
                    self.state.order_offset += self.state.orders.len() as u64;
                    self.state.orders.clear();
                    for orders in &mut self.active {
                        orders.clear();
                    }
                    self.state.quotes.fill(None);
                    self.state.quote_received.fill(0);
                    self.state.auction_done.fill([false; 2]);
                    self.state.corporate_done.fill(false);
                    self.state.day_fees = Fees::default();
                    self.state.day = day;
                    self.events.day = day;
                }
                if phase != self.state.phase
                    && matches!(self.state.phase, Phase::OpenAuction | Phase::CloseAuction)
                {
                    let kind = if self.state.phase == Phase::OpenAuction {
                        1
                    } else {
                        2
                    };
                    for s in 0..self.state.config.symbols {
                        if let Some(q) = self.state.quotes[s]
                            && q[17] == kind
                            && !self.state.auction_done[s][kind as usize - 1]
                        {
                            self.match_quote(s, &q, q[16], true);
                            self.state.auction_done[s][kind as usize - 1] = true;
                        }
                    }
                }
                self.state.phase = phase;
                self.emit(json!({"type":"session","time":self.state.time,"day":day,"phase":phase}));
            }
            Input::Order {
                symbol,
                side,
                kind,
                qty,
                price,
                ..
            } => self.order(symbol, side, kind, qty, price),
            Input::Cancel { id, .. } => {
                if id <= self.state.order_offset
                    || id > self.state.order_offset + self.state.orders.len() as u64
                    || self.state.orders[(id - self.state.order_offset - 1) as usize].status
                        != "open"
                {
                    self.reject(Reject::UnknownOrder, Some(id));
                    return;
                }
                self.finish((id - self.state.order_offset - 1) as usize, "cancelled");
            }
            Input::Transfer { amount, .. } => {
                let cash = self.state.cash as i128 + amount as i128;
                if !(0..=MAX_CASH as i128).contains(&cash) {
                    self.reject(Reject::Cash, None);
                    return;
                }
                self.state.cash = cash as i64;
                self.state.flows += amount;
                self.changed(None);
            }
            Input::Corporate { actions, day, .. } => {
                let action_day = day.unwrap_or(self.state.day);
                if self.state.phase != Phase::Preopen
                    || self.state.orders.iter().any(|o| o.status == "open")
                    || action_day < self.state.actions_through
                    || action_day > self.state.day
                    || !self.valid_actions(&actions, action_day)
                {
                    self.reject(Reject::InvalidAction, None);
                    return;
                }
                self.actions(&actions, action_day);
                self.state.actions_through = action_day;
            }
            Input::Settle {
                day,
                actions,
                interest,
                ..
            } => {
                if day == self.state.settled_day {
                    self.reject(Reject::AlreadySettled, None);
                    return;
                }
                if day != self.state.day || self.state.phase != Phase::Closed {
                    self.reject(Reject::Session, None);
                    return;
                }
                if !self.valid_actions(&actions, day)
                    || interest.unsigned_abs() > (MAX_CASH / 4) as u64
                    || self.state.cash as i128 + self.state.frozen as i128 + (interest as i128) < 0
                {
                    self.reject(Reject::InvalidAction, None);
                    return;
                }
                for i in 0..self.state.orders.len() {
                    if self.state.orders[i].status == "open" {
                        self.finish(i, "expired");
                    }
                }
                self.actions(&actions, day);
                self.state.actions_through = self.state.actions_through.max(day);
                self.state.cash += interest;
                for p in &mut self.state.positions {
                    p.sellable = p.total;
                }
                self.state.settled_day = day;
                self.changed(None);
                self.emit(json!({"type":"settled","time":self.state.time,"day":day,"account":self.account()}));
            }
        }
    }
    fn valid_actions(&self, actions: &[Action], day: i32) -> bool {
        let mut seen = std::collections::BTreeSet::new();
        actions.iter().all(|a| {
            a.symbol < self.state.config.symbols
                && self.state.corporate_dates[a.symbol] < day
                && seen.insert(a.symbol)
                && (0..=MAX_PRICE).contains(&a.cash_per_share)
                && (1..=1000).contains(&a.split_num)
                && (1..=1000).contains(&a.split_den)
                && a.reference_price
                    .is_none_or(|p| (1..=MAX_PRICE).contains(&p))
                && self.state.positions[a.symbol].total as i128 * a.split_num as i128
                    / a.split_den as i128
                    <= MAX_QTY as i128
        })
    }
    fn actions(&mut self, actions: &[Action], day: i32) {
        for a in actions {
            self.state.corporate_done[a.symbol] = day == self.state.day;
            self.state.corporate_dates[a.symbol] = day;
            let p = &mut self.state.positions[a.symbol];
            let old = p.total;
            let dividend = old * a.cash_per_share;
            self.state.cash = self
                .state
                .cash
                .checked_add(dividend)
                .expect("accounting overflow");
            p.total = old * a.split_num / a.split_den;
            p.sellable = p.sellable * a.split_num / a.split_den;
            p.cost = (p.cost - dividend).max(0);
            self.state.marks[a.symbol] = a.reference_price.unwrap_or(
                (self.state.marks[a.symbol] - a.cash_per_share).max(0) * a.split_den / a.split_num,
            );
            if let Some(q) = &mut self.state.quotes[a.symbol] {
                q[2] = self.state.marks[a.symbol];
            }
            let after = p.total;
            self.emit(json!({"type":"corporate","time":self.state.time,"ex_day":day,"symbol":a.symbol,"dividend":dividend,"before":old,"after":after}));
            self.changed(Some(a.symbol));
        }
    }
    fn order(&mut self, symbol: usize, side: Side, kind: Kind, qty: i64, price: i64) {
        let reason = if !matches!(
            self.state.phase,
            Phase::Continuous | Phase::OpenAuction | Phase::CloseAuction
        ) || self.state.day == self.state.settled_day
        {
            Some(Reject::Session)
        } else if symbol >= self.state.config.symbols {
            Some(Reject::UnknownSymbol)
        } else if let Some(q) = self.state.quotes[symbol] {
            let p = &self.state.positions[symbol];
            if q[15] != 0 {
                Some(Reject::Suspended)
            } else if !self.lots.valid(
                symbol,
                self.state.config.instruments.get(symbol),
                qty,
                side,
                p.sellable,
            ) {
                Some(Reject::Quantity)
            } else if price <= 0 || price > MAX_PRICE || price % self.tick(symbol) != 0 {
                Some(Reject::Tick)
            } else if self.state.config.enforce_bands
                && ((q[13] > 0 && price > q[13]) || (q[14] > 0 && price < q[14]))
            {
                Some(Reject::PriceBand)
            } else if side == Side::Sell && qty > p.sellable {
                Some(Reject::Sellable)
            } else {
                None
            }
        } else {
            Some(Reject::NoQuote)
        };
        if let Some(r) = reason {
            self.reject(r, None);
            return;
        }
        let id = self.state.order_offset + self.state.orders.len() as u64 + 1;
        let mut o = Order {
            id,
            symbol,
            side,
            kind,
            qty,
            remaining: qty,
            price,
            time: self.state.time,
            reserve: 0,
            notional: 0,
            fees: Fees::default(),
            status: "open".into(),
        };
        o.reserve = self.reserve(&o);
        if o.reserve > self.state.cash {
            self.reject(Reject::Cash, None);
            return;
        }
        self.state.cash -= o.reserve;
        self.state.frozen += o.reserve;
        if side == Side::Sell {
            self.state.positions[symbol].sellable -= qty;
            self.state.positions[symbol].frozen += qty;
        }
        self.active[symbol].push(self.state.orders.len());
        self.state.orders.push(o);
        self.emit(json!({"type":"accepted","time":self.state.time,"id":id}));
        self.changed(Some(symbol));
    }
    fn finish(&mut self, i: usize, status: &str) {
        let o = &mut self.state.orders[i];
        self.state.cash += o.reserve;
        self.state.frozen -= o.reserve;
        o.reserve = 0;
        o.status = status.into();
        let s = o.symbol;
        if o.side == Side::Sell {
            self.state.positions[s].sellable += o.remaining;
            self.state.positions[s].frozen -= o.remaining;
        }
        let (id, remaining) = (o.id, o.remaining);
        self.emit(json!({"type":"ended","time":self.state.time,"id":id,"status":status,"remaining":remaining}));
        self.active[s].retain(|&j| j != i);
        self.changed(Some(s));
    }
    pub fn validate_quotes(rows: &[i64], symbols: usize) -> Result<(), String> {
        if !rows.len().is_multiple_of(WIDTH) {
            return Err("quotes must have 18 columns".into());
        }
        for q in rows.chunks_exact(WIDTH) {
            if q[0] < 0
                || q[0] as usize >= symbols
                || q[1] < 0
                || q[2] <= 0
                || q[2] > MAX_PRICE
                || q[3..7].iter().any(|&v| !(0..=MAX_PRICE).contains(&v))
                || q[7] < 0
                || q[7] > 1_000_000_000_000
                || q[8] < 0
                || q[9..15]
                    .iter()
                    .any(|&v| !(-1..=1_000_000_000_000).contains(&v))
                || q[9] > MAX_PRICE
                || q[11] > MAX_PRICE
                || q[13] > MAX_PRICE
                || q[14] > MAX_PRICE
                || !(0..=1).contains(&q[15])
                || !(0..=1_000_000_000_000).contains(&q[16])
                || !(0..=2).contains(&q[17])
            {
                return Err("invalid quote".into());
            }
        }
        Ok(())
    }
    fn tick(&self, symbol: usize) -> i64 {
        self.state
            .config
            .instruments
            .get(symbol)
            .map_or(self.state.config.tick, |p| p.tick)
    }
    pub fn feed(&mut self, rows: &[i64]) -> Result<(), String> {
        self.feed_at(rows, None)
    }
    /// Receipt time orders inputs; exchange timestamps only describe quote freshness.
    pub fn feed_at(&mut self, rows: &[i64], received_at: Option<i64>) -> Result<(), String> {
        Self::validate_quotes(rows, self.state.config.symbols)?;
        if received_at.is_some_and(|t| t < self.state.time) {
            return Err("receipt time precedes engine time".into());
        }
        for row in rows.chunks_exact(WIDTH) {
            let q: [i64; WIDTH] = row.try_into().expect("validated width");
            let s = q[0] as usize;
            // Synthetic clocks (< year 2000) have no civil date. Production feeds do.
            if q[1] >= 946_684_800_000
                && civil_day(q[1], self.state.config.utc_offset_seconds) != self.state.day
            {
                continue;
            }
            let old = self.state.quotes[s];
            if old.is_some_and(|v| {
                q[1] < v[1]
                    || (q[1] == v[1]
                        && !(q[17] > 0
                            && v[17] == 0
                            && q[2..16] == v[2..16]
                            && !self.state.auction_done[s][q[17] as usize - 1]
                            && matches!(
                                self.state.phase,
                                Phase::OpenAuction | Phase::CloseAuction
                            )))
                    || (q[7] < v[7] && self.state.phase != Phase::Closed)
            }) {
                continue;
            }
            let delta = old.map_or(0, |v| q[7] - v[7]);
            self.state.time = received_at.unwrap_or(self.state.time);
            self.state.quotes[s] = Some(q);
            self.state.marks[s] = q[2];
            if self.state.phase == Phase::Continuous && q[17] == 0 {
                self.match_quote(s, &q, delta, false);
            }
            self.state.quote_received[s] = self.state.time;
        }
        if let Some(t) = received_at {
            self.state.time = t;
        }
        Ok(())
    }
    fn match_quote(&mut self, s: usize, q: &[i64; WIDTH], volume: i64, auction: bool) {
        if q[15] != 0 || self.active[s].is_empty() {
            return;
        }
        let mut capacity =
            ((volume as i128 * self.state.config.participation_ppm as i128) / 1_000_000) as i64;
        let mut ask = q[10].max(0);
        let mut bid = q[12].max(0);
        for i in self.active[s].clone() {
            let o = self.state.orders[i].clone();
            let buy = o.side == Side::Buy;
            if q[1] <= o.time || (!auction && self.state.quote_received[s] < o.time) {
                continue;
            }
            let book = if buy { q[9] } else { q[11] };
            let auction_price = if auction && q[17] == 1 { q[3] } else { q[2] };
            if (buy && q[13] > 0 && auction_price >= q[13] && (book <= 0 || ask == 0))
                || (!buy && q[14] > 0 && auction_price <= q[14] && (book <= 0 || bid == 0))
            {
                continue;
            }
            if !auction && book == 0 {
                continue;
            }
            let base = if auction {
                if q[17] == 1 { q[3] } else { q[2] }
            } else if book > 0 {
                book
            } else {
                q[2]
            };
            let slip = self.state.config.slippage_ticks * self.tick(s)
                + rate(base, self.state.config.slippage_ppm);
            let raw = if buy { base + slip } else { base - slip };
            let tick = self.tick(s);
            let price = if buy {
                (raw + tick - 1) / tick * tick
            } else {
                raw / tick * tick
            };
            if price <= 0
                || (buy && price > o.price)
                || (!buy && price < o.price)
                || (self.state.config.enforce_bands
                    && ((q[13] > 0 && price > q[13]) || (q[14] > 0 && price < q[14])))
            {
                continue;
            }
            let depth = if auction || book < 0 {
                i64::MAX
            } else if buy {
                ask
            } else {
                bid
            };
            let qty = o.remaining.min(capacity).min(depth);
            if qty <= 0 {
                continue;
            }
            let value = price * qty;
            let cumulative = self.costs.cumulative(
                s,
                self.state.config.instruments.get(s),
                o.side,
                o.notional + value,
            );
            let fees = cumulative.difference(&o.fees);
            if !buy && self.state.cash as i128 + value as i128 - (fees.total() as i128) < 0 {
                self.reject(Reject::Cash, Some(o.id));
                self.finish(i, "unaffordable_fees");
                continue;
            }
            let mut next = o.clone();
            next.remaining -= qty;
            next.notional += value;
            next.fees = cumulative;
            next.reserve = self.reserve(&next);
            if buy {
                self.state.frozen += next.reserve - o.reserve;
                self.state.cash += o.reserve - next.reserve - value - fees.total();
                let p = &mut self.state.positions[s];
                p.total += qty;
                p.cost += value + fees.total();
                if !self
                    .state
                    .config
                    .instruments
                    .get(s)
                    .map_or(self.state.config.t_plus_one, |p| p.t_plus_one)
                {
                    p.sellable += qty;
                }
            } else {
                self.state.cash += value - fees.total();
                let p = &mut self.state.positions[s];
                p.cost = if p.total == qty {
                    0
                } else {
                    (p.cost as i128 * (p.total - qty) as i128 / p.total as i128) as i64
                };
                p.total -= qty;
                p.frozen -= qty;
            }
            self.state.fees.add(&fees);
            self.state.day_fees.add(&fees);
            self.state.orders[i] = next;
            capacity -= qty;
            if !auction && book > 0 {
                if buy {
                    ask -= qty;
                } else {
                    bid -= qty;
                }
            }
            self.emit(json!({"type":"fill","time":self.state.time,"quote_time":q[1],"id":o.id,"symbol":s,"side":o.side,"qty":qty,"price":price,"fees":fees,"auction":auction,"fallback":!auction && book<0}));
            self.changed(Some(s));
            if self.state.orders[i].remaining == 0 {
                self.finish(i, "filled");
            }
        }
    }
}

// Gregorian civil date from epoch days; no wall clock or timezone database IO.
fn civil_day(ms: i64, offset: i64) -> i32 {
    let z = (ms / 1000 + offset).div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    ((y + i64::from(m <= 2)) * 10_000 + m * 100 + d) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fees_and_lots_are_replaceable() {
        let cost = SimpleCosts {
            commission_ppm: 250,
            minimum: 50_000,
            sell_tax_ppm: 500,
        };
        assert_eq!(
            cost.cumulative(0, None, Side::Buy, 1_000_000).total(),
            50_000
        );
        assert_eq!(
            cost.cumulative(0, None, Side::Sell, 1_000_000).total(),
            50_500
        );
        let lot = SimpleLots {
            step: 100,
            minimum: 100,
        };
        assert!(lot.valid(0, None, 100, Side::Buy, 0));
        assert!(!lot.valid(0, None, 99, Side::Buy, 0));
        assert!(lot.valid(0, None, 99, Side::Sell, 99));
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    fn broker() -> Engine {
        Engine::new(
            serde_json::from_value(json!({
                "symbols":1,"initial_cash":10000000000_i64,"tick":100,
                "participation_ppm":100000,"t_plus_one":true,"enforce_bands":true,
                "slippage_ticks":0,"slippage_ppm":0,"margin_model":"cash",
                "costs":{"commission_ppm":250,"minimum":50000,"sell_tax_ppm":500},
                "lots":{"step":100,"minimum":100}
            }))
            .unwrap(),
        )
        .unwrap()
    }
    fn quote(time: i64, volume: i64) -> [i64; WIDTH] {
        [
            0, time, 100000, 100000, 100000, 100000, 100000, volume, 0, 100000, 1000, 100000, 1000,
            110000, 90000, 0, 0, 0,
        ]
    }
    #[test]
    fn civil_dates_and_injected_offset() {
        assert_eq!(civil_day(1794447000000, 28800), 20261112);
        assert_eq!(civil_day(0, 0), 19700101);
    }
    #[test]
    fn baseline_precedes_order_capacity() {
        let mut b = broker();
        b.apply(Input::Session {
            time: 0,
            day: 1,
            phase: Phase::Continuous,
        });
        b.feed(&quote(1, 0)).unwrap();
        b.apply(Input::Order {
            time: 2,
            symbol: 0,
            side: Side::Buy,
            kind: Kind::Limit,
            qty: 1000,
            price: 110000,
        });
        b.feed(&quote(3, 10000)).unwrap();
        assert_eq!(b.state.positions[0].total, 0);
        b.feed(&quote(4, 11000)).unwrap();
        assert_eq!(b.state.positions[0].total, 100);
        assert_eq!(b.state.positions[0].sellable, 0);
        let frozen = b.state.frozen;
        assert!(frozen > 0);
        b.apply(Input::Cancel { time: 5, id: 1 });
        assert_eq!(b.state.frozen, 0);
        assert_eq!(b.state.cash, 10000000000 - 10000000 - 50000);
    }
    #[test]
    fn locked_open_auction_never_fills() {
        let mut b = broker();
        b.apply(Input::Session {
            time: 0,
            day: 1,
            phase: Phase::OpenAuction,
        });
        b.feed(&quote(1, 0)).unwrap();
        b.apply(Input::Order {
            time: 2,
            symbol: 0,
            side: Side::Buy,
            kind: Kind::Limit,
            qty: 1000,
            price: 110000,
        });
        let mut q = quote(3, 10000);
        q[3] = 110000;
        q[9] = 0;
        q[10] = 0;
        q[16] = 10000;
        q[17] = 1;
        b.feed(&q).unwrap();
        b.apply(Input::Session {
            time: 4,
            day: 1,
            phase: Phase::Continuous,
        });
        assert_eq!(b.state.positions[0].total, 0);
    }
    #[test]
    fn malformed_batch_is_atomic() {
        let mut b = broker();
        let before = b.dump();
        let mut q = quote(3, 1);
        q[0] = 1;
        assert!(b.feed(&q).is_err());
        assert_eq!(b.dump(), before);
    }
    #[test]
    #[should_panic(expected = "overflow")]
    fn accounting_overflow_traps_in_release_too() {
        std::hint::black_box(Fees {
            commission: i64::MAX,
            tax: 1,
            other: 0,
        })
        .total();
    }
}
