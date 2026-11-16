//! Saved pending buys: a snapshot taken when buys are cancelled, and a one-key restore.
//!
//! Every Cancel Buy path (the chart button, the per-market hotkey, "cancel all buys", "cancel buys
//! in all bots") goes through [`Backend::cancel_buy_orders`], which calls
//! [`Backend::save_pending_buys`] first. The snapshot is per `(core, market)`: cancelling a market
//! replaces only that market's saved set, and a cancel that finds nothing to cancel never wipes a
//! set that is already saved.
//!
//! [`Backend::restore_canceled_buys_for_core`] (the "Restore canceled buys" hotkey) places the
//! saved orders of the window's ACTIVE core again, and of no other core. Each order is re-anchored to the CURRENT price at the
//! same distance it had when it was cancelled, because the price has usually moved since, and an
//! absolute price on the wrong side of the market would fill at once as a taker. An order that
//! cannot be placed safely (no current price, or it would cross the market) is skipped and stays
//! saved, so a second press retries it. An order that was sent is removed from the saved set, so a
//! repeated press cannot double the position.
//!
//! A "cancel all" (one core or all cores) is a fresh picture: it first drops what the user's earlier
//! cancels saved on that core, so a coin the user has since left is not brought back with the new
//! batch. A single-market cancel replaces only that market.
//!
//! Buys that go away WITHOUT the user's cancel (a strategy or the core cancels them, the exchange
//! drops them) are kept too. [`Backend::track_pending_buys`] watches the live pending buys, and
//! one that the core closes as cancelled with no fill at all joins the same saved set, so the same
//! key brings it back. Three guards keep that safe around a strategy that re-places its orders:
//!
//! - an order that FILLED (fully or partly) is never saved, so an auto-repeat after a profit is not
//!   mistaken for a cancel;
//! - an order the strategy re-placed (a live "twin" with the same side, strategy, distance and
//!   notional) is not restored, and neither is a saved duplicate of another saved order;
//! - an automatically cancelled order is forgotten after [`AUTO_TTL_MS`].
//!
//! The saved set lives in `cfg/saved_limits.json`, so it survives a terminal restart.

use super::selectors::ready_cores;
use crate::Backend;
use moon_core::config::GroupExitSettings;
use moon_core::feed::OrderRow;
use moon_core::session::CoreId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// File name inside the `cfg/` directory.
const SAVED_FILE_NAME: &str = "saved_limits.json";

/// Restore at the same distance from the current price (`true`), or at the saved absolute price
/// (`false`). The distance mode is the safe one; see the module docs.
const RESTORE_BY_DISTANCE: bool = true;

/// Keep the order's USD notional (size x price) when it is placed at a new price, instead of the
/// coin amount. Right for linear (USDT-margined) markets, whose sizes are set in USD.
const KEEP_USD_NOTIONAL: bool = true;

/// An automatically cancelled order stays restorable this long (48 hours: a trader may be away
/// all day, and the order comes back at the distance it had from the price, not at a stale price).
const AUTO_TTL_MS: i64 = 48 * 60 * 60 * 1000;

/// Minimum gap between two scans of the live orders.
const TRACK_INTERVAL_MS: i64 = 500;

/// A tracked order that is no longer a plain pending buy but has not been closed yet (the core is
/// between two states) is followed this long before it is dropped as unknown.
const TRACKED_LINGER_MS: i64 = 120_000;

/// Two orders are "twins" when their distances from the price differ by no more than this
/// (0.003 = 0.3 percentage points) ...
const TWIN_DISTANCE_TOL: f64 = 0.003;

/// ... and their USD notionals by no more than this fraction.
const TWIN_NOTIONAL_TOL: f64 = 0.10;

/// One cancelled pending buy, with everything needed to place it again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SavedLimit {
    /// Canonical market name, the key every order command uses.
    market: String,
    /// Position side: `true` for a short entry.
    short: bool,
    /// Entry price at the moment of cancellation. For a pending (conditional) order this is the
    /// TRIGGER price.
    price: f64,
    /// `(price - last) / last` at the moment of cancellation, or `None` when the last price was
    /// not known.
    distance: Option<f64>,
    /// Size in the base unit the order commands use.
    size: f64,
    /// Strategy the order ran under; `0` means none.
    strat_id: u64,
    /// Planned sell price of a manual (strategy-less) order, `0.0` for none.
    planned_sell: f64,
    /// `true` when the saved order was a pending (conditional) order, restored as one.
    pending: bool,
    /// The core's order uid, `0` for a record from an older file.
    #[serde(default)]
    uid: u64,
    /// `true` when the order went away without the user's cancel; such records expire.
    #[serde(default)]
    auto: bool,
    /// When the record was saved, in Unix milliseconds; `0` for a record from an older file.
    #[serde(default)]
    saved_ms: i64,
}

/// The whole saved set: core -> market -> orders.
#[derive(Debug, Default, Serialize, Deserialize)]
struct SavedFile {
    #[serde(default)]
    cores: BTreeMap<CoreId, BTreeMap<String, Vec<SavedLimit>>>,
}

/// Process-wide saved set, loaded from disk on first use.
fn store() -> MutexGuard<'static, SavedFile> {
    static STORE: OnceLock<Mutex<SavedFile>> = OnceLock::new();
    STORE
        .get_or_init(|| Mutex::new(load_file()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn file_path() -> std::path::PathBuf {
    moon_core::config::paths::cfg_dir().join(SAVED_FILE_NAME)
}

fn load_file() -> SavedFile {
    let path = file_path();
    match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|error| {
            log::warn!("saved limits: {} is unreadable ({error}); starting empty", path.display());
            SavedFile::default()
        }),
        Err(_) => SavedFile::default(),
    }
}

/// Write the saved set through a temp file and a rename, so a crash never leaves half a file.
fn persist(file: &SavedFile) {
    let path = file_path();
    let tmp = path.with_extension("json.tmp");
    let result = serde_json::to_string_pretty(file)
        .map_err(|error| error.to_string())
        .and_then(|text| std::fs::write(&tmp, text).map_err(|error| error.to_string()))
        .and_then(|()| std::fs::rename(&tmp, &path).map_err(|error| error.to_string()));
    if let Err(error) = result {
        log::warn!("saved limits: could not write {}: {error}", path.display());
    }
}

/// Whether an order row is a pending buy the Cancel Buy command would cancel.
///
/// The same phase test `feed::trade::cancel_market_buys` applies on the wire (`None` and `BuySet`),
/// minus emulator orders, which cannot be placed again as emulator orders.
fn is_cancellable_buy(row: &OrderRow) -> bool {
    let status = row.status.as_str();
    !row.job_is_done
        && !row.emulator
        && (status.eq_ignore_ascii_case("None") || status.eq_ignore_ascii_case("BuySet"))
}

/// Build the saved record for one row, or `None` when the row carries no usable price or size.
///
/// A row whose entry is already partly filled is not saved: cancelling it leaves a real position
/// behind, and placing the full original size again would buy that part twice.
fn saved_from_row(row: &OrderRow) -> Option<SavedLimit> {
    if row.fill_pct > 0.0 {
        log::info!(
            "saved limits: market={} order uid={} is partly filled ({}%); not saved",
            row.market,
            row.uid,
            row.fill_pct
        );
        return None;
    }
    let price = if row.pending {
        row.pending_cond.unwrap_or(0.0)
    } else {
        row.buy_price
    };
    if !(price.is_finite() && price > 0.0 && row.size.is_finite() && row.size > 0.0) {
        return None;
    }
    let last = f64::from(row.price);
    let distance = (last.is_finite() && last > 0.0).then(|| (price - last) / last);
    Some(SavedLimit {
        market: row.market.clone(),
        short: row.is_short,
        price,
        distance,
        size: row.size,
        strat_id: row.strat_id,
        planned_sell: if row.sell_price.is_finite() && row.sell_price > 0.0 {
            row.sell_price
        } else {
            0.0
        },
        pending: row.pending,
        uid: row.uid,
        auto: false,
        saved_ms: now_ms(),
    })
}

/// Round a price to the exchange's tick grid and strip float noise.
fn snap_to_tick(price: f64, tick: Option<f64>) -> f64 {
    let Some(tick) = tick.filter(|tick| tick.is_finite() && *tick > 0.0) else {
        return price;
    };
    let snapped = (price / tick).round() * tick;
    let decimals = (-tick.log10()).ceil().clamp(0.0, 12.0) as i32;
    let scale = 10f64.powi(decimals);
    let clean = (snapped * scale).round() / scale;
    if clean.is_finite() && clean > 0.0 { clean } else { price }
}

/// The price a saved order is placed at now: the saved distance applied to the current price, or
/// the saved absolute price when the distance mode is off or either figure is missing.
fn target_price(saved: &SavedLimit, last: Option<f64>, tick: Option<f64>) -> f64 {
    let raw = match (RESTORE_BY_DISTANCE, saved.distance, last) {
        (true, Some(distance), Some(last)) => last * (1.0 + distance),
        _ => saved.price,
    };
    snap_to_tick(raw, tick)
}

/// Whether a LIMIT entry at `price` would fill at once against the market at `last`: a long entry
/// at or above the market, or a short entry at or below it.
fn crosses_market(short: bool, price: f64, last: f64) -> bool {
    if short { price <= last } else { price >= last }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or(0)
}

/// Whether two orders are the same logical order: same side, kind and strategy, about the same
/// distance from the price and about the same USD notional.
///
/// This is what tells a strategy's re-placed order (or an auto-repeat) from a different one.
fn is_twin(a: &SavedLimit, b: &SavedLimit) -> bool {
    if a.short != b.short || a.pending != b.pending || a.strat_id != b.strat_id {
        return false;
    }
    let (notional_a, notional_b) = (a.size * a.price, b.size * b.price);
    if !(notional_a > 0.0 && notional_b > 0.0) {
        return false;
    }
    if (notional_a - notional_b).abs() / notional_a.max(notional_b) > TWIN_NOTIONAL_TOL {
        return false;
    }
    match (a.distance, b.distance) {
        (Some(x), Some(y)) => (x - y).abs() <= TWIN_DISTANCE_TOL,
        _ => (a.price - b.price).abs() / a.price.max(b.price) <= TWIN_DISTANCE_TOL,
    }
}

/// Drop automatically saved records older than [`AUTO_TTL_MS`], and empty markets and cores.
fn prune_expired(file: &mut SavedFile, now: i64) {
    for markets in file.cores.values_mut() {
        for orders in markets.values_mut() {
            orders.retain(|saved| !(saved.auto && now - saved.saved_ms > AUTO_TTL_MS));
        }
        markets.retain(|_, orders| !orders.is_empty());
    }
    file.cores.retain(|_, markets| !markets.is_empty());
}

/// Forget every record the user's own cancels saved on one core, keeping the automatic ones.
///
/// A "cancel all" is a fresh picture of what the user wants back, so what an earlier cancel saved
/// (another coin the user has since left) must not ride along. Records of orders that went away on
/// their own are kept: the user did not cancel those, so a cancel-all does not speak for them.
fn drop_user_saved(file: &mut SavedFile, core: CoreId) {
    let Some(markets) = file.cores.get_mut(&core) else {
        return;
    };
    for orders in markets.values_mut() {
        orders.retain(|saved| saved.auto);
    }
    markets.retain(|_, orders| !orders.is_empty());
    let empty = markets.is_empty();
    if empty {
        file.cores.remove(&core);
    }
}

/// Add an order that went away without the user's cancel.
///
/// Skipped when the saved set already holds it (the user's own cancel saved it first) or holds a
/// user-saved twin. An older automatic twin is replaced, so repeated re-placing of one order keeps
/// one record.
fn add_auto(file: &mut SavedFile, core: CoreId, mut saved: SavedLimit, now: i64) {
    let list = file
        .cores
        .entry(core)
        .or_default()
        .entry(saved.market.clone())
        .or_default();
    if list
        .iter()
        .any(|kept| kept.uid == saved.uid || (!kept.auto && is_twin(kept, &saved)))
    {
        return;
    }
    list.retain(|kept| !(kept.auto && is_twin(kept, &saved)));
    saved.auto = true;
    saved.saved_ms = now;
    list.push(saved);
}

/// A pending buy followed since it was seen open, until its fate is clear.
struct Tracked {
    saved: SavedLimit,
    last_open_ms: i64,
}

/// Process-wide set of followed orders, never written to disk.
fn tracked() -> MutexGuard<'static, HashMap<(CoreId, u64), Tracked>> {
    static TRACKED: OnceLock<Mutex<HashMap<(CoreId, u64), Tracked>>> = OnceLock::new();
    TRACKED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// One scan of a core's orders: follow the open pending buys, and return those that were closed
/// as cancelled with no fill since the last scan.
///
/// An order that vanishes without the core's closing flag is dropped, not saved: it may have been
/// filled, and a saved fill would buy again.
fn track_core<'a>(
    core: CoreId,
    rows: impl Iterator<Item = &'a OrderRow>,
    now: i64,
) -> Vec<SavedLimit> {
    let rows: Vec<&OrderRow> = rows.collect();
    let by_uid: HashMap<u64, &OrderRow> = rows.iter().map(|row| (row.uid, *row)).collect();
    let mut tracked = tracked();
    for row in &rows {
        if is_cancellable_buy(row)
            && row.fill_pct <= 0.0
            && let Some(saved) = saved_from_row(row)
        {
            tracked.insert(
                (core, row.uid),
                Tracked {
                    saved,
                    last_open_ms: now,
                },
            );
        }
    }
    let mut cancelled = Vec::new();
    tracked.retain(|key, followed| {
        if key.0 != core {
            return true;
        }
        match by_uid.get(&key.1) {
            None => false,
            Some(row) if row.filled || row.fill_pct > 0.0 => false,
            Some(row) if row.job_is_done => {
                cancelled.push(followed.saved.clone());
                false
            }
            Some(_) => now - followed.last_open_ms <= TRACKED_LINGER_MS,
        }
    });
    cancelled
}

impl Backend {
    /// Follow the live pending buys, and save the ones the core cancels without a fill.
    ///
    /// Cheap and called often: it returns at once when the last scan was under
    /// [`TRACK_INTERVAL_MS`] ago.
    pub(crate) fn track_pending_buys(&self) {
        static LAST_SCAN_MS: AtomicI64 = AtomicI64::new(0);
        let now = now_ms();
        if now - LAST_SCAN_MS.load(Ordering::Relaxed) < TRACK_INTERVAL_MS {
            return;
        }
        LAST_SCAN_MS.store(now, Ordering::Relaxed);
        let (ready, _offline) = ready_cores(self.session.store().statuses());
        let mut cancelled: Vec<(CoreId, SavedLimit)> = Vec::new();
        for core in ready {
            let found = self
                .session
                .store()
                .core(core)
                .map(|data| track_core(core, data.orders.iter(), now))
                .unwrap_or_default();
            cancelled.extend(found.into_iter().map(|saved| (core, saved)));
        }
        if cancelled.is_empty() {
            return;
        }
        let mut file = store();
        for (core, saved) in cancelled {
            log::info!(
                "saved limits: core={} market={} order uid={} was cancelled without a fill; kept",
                moon_core::feed::core_label(core),
                saved.market,
                saved.uid
            );
            add_auto(&mut file, core, saved, now);
        }
        prune_expired(&mut file, now);
        persist(&file);
    }

    /// Start a "cancel all" on one core: drop what the user's earlier cancels saved there, so the
    /// batch about to be saved replaces it. See [`drop_user_saved`].
    pub(crate) fn begin_cancel_all_saving(&self, core: CoreId) {
        let mut file = store();
        drop_user_saved(&mut file, core);
        persist(&file);
    }

    /// Save the pending buys of one market before they are cancelled.
    ///
    /// Replaces that market's saved set only when at least one buy is found, so pressing Cancel
    /// Buy a second time (nothing left to cancel) keeps the first press's snapshot.
    pub(crate) fn save_pending_buys(&self, core: CoreId, market: &str) {
        let saved: Vec<SavedLimit> = self
            .session
            .store()
            .core(core)
            .map(|data| {
                data.orders
                    .iter()
                    .filter(|row| row.market == market && is_cancellable_buy(row))
                    .filter_map(saved_from_row)
                    .collect()
            })
            .unwrap_or_default();
        if saved.is_empty() {
            return;
        }
        let count = saved.len();
        let mut file = store();
        file.cores
            .entry(core)
            .or_default()
            .insert(market.to_string(), saved);
        persist(&file);
        log::info!(
            "saved limits: kept {count} buy order(s) of core={} market={market} before cancel",
            moon_core::feed::core_label(core)
        );
    }

    /// Place the saved buys of ONE core again: the window's active core, never another. Returns how
    /// many orders were sent.
    ///
    /// Orders that cannot be placed safely stay saved; see the module docs. A core that is not
    /// online is left alone, and so are the saved orders of every other core.
    pub(crate) fn restore_canceled_buys_for_core(&self, core: CoreId) -> usize {
        let (ready, _offline) = ready_cores(self.session.store().statuses());
        let label = moon_core::feed::core_label(core);
        if !ready.contains(&core) {
            log::info!("restore canceled buys: core={label} is not online; nothing sent");
            return 0;
        }
        let mut sent_total = 0;
        let mut kept_total = 0;
        let mut twin_total = 0;
        let mut file = store();
        prune_expired(&mut file, now_ms());
        if let Some(markets) = file.cores.get_mut(&core) {
            for (market, orders) in markets.iter_mut() {
                let source = self.session.market_source();
                let last = source
                    .latest_price(core, market)
                    .ok()
                    .map(f64::from)
                    .filter(|last| last.is_finite() && *last > 0.0);
                let tick = source.price_tick(core, market);
                // Orders already live again (a strategy re-placed them, or an auto-repeat did) are
                // not placed a second time.
                let live: Vec<SavedLimit> = self
                    .session
                    .store()
                    .core(core)
                    .map(|data| {
                        data.orders
                            .iter()
                            .filter(|row| {
                                row.market == *market
                                    && is_cancellable_buy(row)
                                    && row.fill_pct <= 0.0
                            })
                            .filter_map(saved_from_row)
                            .collect()
                    })
                    .unwrap_or_default();
                orders.retain(|saved| {
                    if live.iter().any(|twin| is_twin(twin, saved)) {
                        twin_total += 1;
                        // The old order may still be followed; it must not come back as a cancel.
                        tracked().remove(&(core, saved.uid));
                        return false;
                    }
                    let sent = self.place_saved_limit(core, saved, last, tick);
                    if sent {
                        sent_total += 1;
                        tracked().remove(&(core, saved.uid));
                    } else {
                        kept_total += 1;
                    }
                    !sent
                });
            }
            markets.retain(|_, orders| !orders.is_empty());
        }
        file.cores.retain(|_, markets| !markets.is_empty());
        persist(&file);
        log::info!(
            "restore canceled buys: core={label}: {sent_total} order(s) sent, {twin_total} skipped \
             as already live, {kept_total} kept for retry"
        );
        sent_total
    }

    /// Place one saved order. Returns whether it was SENT (and so must leave the saved set).
    fn place_saved_limit(
        &self,
        core: CoreId,
        saved: &SavedLimit,
        last: Option<f64>,
        tick: Option<f64>,
    ) -> bool {
        let label = moon_core::feed::core_label(core);
        let price = target_price(saved, last, tick);
        if !(price.is_finite() && price > 0.0) {
            log::warn!("restore buy: core={label} market={} has no usable price; kept", saved.market);
            return false;
        }
        let strategy_id = (saved.strat_id != 0).then_some(saved.strat_id);
        // The same USD notional at the new price, not the same coin amount.
        let size = if KEEP_USD_NOTIONAL && saved.price > 0.0 {
            saved.size * saved.price / price
        } else {
            saved.size
        };
        let result = if saved.pending {
            // A pending order's price is a trigger, so it is not compared with the market.
            self.session.place_pending_order(
                core,
                saved.market.clone(),
                saved.short,
                price,
                size,
                strategy_id,
                GroupExitSettings::default(),
                false,
            )
        } else {
            if let Some(last) = last {
                if crosses_market(saved.short, price, last) {
                    log::warn!(
                        "restore buy: core={label} market={} price={price} would fill at once \
                         against last={last}; kept",
                        saved.market
                    );
                    return false;
                }
            } else if RESTORE_BY_DISTANCE {
                // No current price, so there is nothing to check the saved absolute price against.
                log::warn!("restore buy: core={label} market={} has no current price; kept", saved.market);
                return false;
            }
            // The strategy, when there is one, decides the sell; the saved target only rides with
            // a strategy-less order, scaled with the entry so its take-profit percent is kept.
            let planned_sell = if strategy_id.is_none() && saved.planned_sell > 0.0 {
                saved.planned_sell * price / saved.price
            } else {
                0.0
            };
            self.session.place_order(
                core,
                saved.market.clone(),
                saved.short,
                price,
                size,
                strategy_id,
                GroupExitSettings::default(),
                planned_sell,
                false,
            )
        };
        match result {
            Ok(()) => {
                log::info!(
                    "restore buy: core={label} market={} short={} price={price} size={} strat={strategy_id:?} pending={}",
                    saved.market, saved.short, size, saved.pending
                );
                true
            }
            Err(error) => {
                log::warn!("restore buy: core={label} market={} failed: {error:#}; kept", saved.market);
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn saved(price: f64, distance: Option<f64>) -> SavedLimit {
        SavedLimit {
            market: "WLFIUSDT".into(),
            short: false,
            price,
            distance,
            size: 100.0,
            strat_id: 0,
            planned_sell: 0.0,
            pending: false,
            uid: 1,
            auto: false,
            saved_ms: 0,
        }
    }

    #[test]
    fn distance_is_reapplied_to_the_current_price() {
        let s = saved(0.0477, Some(-0.20));
        let p = target_price(&s, Some(0.0500), None);
        assert!((p - 0.0400).abs() < 1e-12);
    }

    #[test]
    fn absolute_price_is_the_fallback_without_a_current_price() {
        let s = saved(0.0477, Some(-0.20));
        assert_eq!(target_price(&s, None, None), 0.0477);
    }

    #[test]
    fn snap_rounds_to_the_tick_grid() {
        assert_eq!(snap_to_tick(0.047_712_9, Some(0.000_01)), 0.047_71);
        assert_eq!(snap_to_tick(1.234, None), 1.234);
    }

    #[test]
    fn a_replaced_order_is_a_twin_and_a_different_one_is_not() {
        let a = saved(0.0400, Some(-0.20));
        // Same distance, same notional at a moved price: the strategy re-placed it.
        let mut moved = saved(0.0440, Some(-0.2005));
        moved.size = 100.0 * 0.0400 / 0.0440;
        assert!(is_twin(&a, &moved));
        // A different distance is another order.
        assert!(!is_twin(&a, &saved(0.0400, Some(-0.30))));
        // A different strategy is another order.
        let mut other = saved(0.0400, Some(-0.20));
        other.strat_id = 7;
        assert!(!is_twin(&a, &other));
    }

    #[test]
    fn auto_records_expire_and_user_records_do_not() {
        let mut file = SavedFile::default();
        let mut old_auto = saved(0.04, Some(-0.2));
        old_auto.auto = true;
        old_auto.saved_ms = 0;
        let user = saved(0.05, Some(-0.1));
        file.cores
            .entry(1)
            .or_default()
            .insert("WLFIUSDT".into(), vec![old_auto, user.clone()]);
        prune_expired(&mut file, AUTO_TTL_MS + 1);
        assert_eq!(file.cores[&1]["WLFIUSDT"], vec![user]);
    }

    #[test]
    fn an_auto_record_never_duplicates_a_saved_order() {
        let mut file = SavedFile::default();
        let user = saved(0.04, Some(-0.2));
        file.cores
            .entry(1)
            .or_default()
            .insert("WLFIUSDT".into(), vec![user.clone()]);
        // Same uid: the user's cancel saved it first.
        add_auto(&mut file, 1, user.clone(), 10);
        // A twin under another uid: the user's record already stands for it.
        let mut twin = user.clone();
        twin.uid = 2;
        add_auto(&mut file, 1, twin, 10);
        assert_eq!(file.cores[&1]["WLFIUSDT"].len(), 1);
    }

    #[test]
    fn cancel_all_drops_user_records_but_keeps_automatic_ones() {
        let mut file = SavedFile::default();
        let old_coin = saved(0.04, Some(-0.2));
        let mut auto = saved(0.05, Some(-0.1));
        auto.auto = true;
        auto.saved_ms = 5;
        let core = file.cores.entry(1).or_default();
        core.insert("OLDUSDT".into(), vec![old_coin]);
        core.insert("AUTOUSDT".into(), vec![auto.clone()]);
        drop_user_saved(&mut file, 1);
        assert!(!file.cores[&1].contains_key("OLDUSDT"));
        assert_eq!(file.cores[&1]["AUTOUSDT"], vec![auto]);
        // Another core is untouched, and a core with nothing left disappears.
        file.cores.entry(2).or_default().insert("XUSDT".into(), vec![saved(1.0, None)]);
        drop_user_saved(&mut file, 2);
        assert!(!file.cores.contains_key(&2));
    }

    #[test]
    fn crossing_is_detected_per_side() {
        assert!(crosses_market(false, 1.0, 1.0));
        assert!(!crosses_market(false, 0.9, 1.0));
        assert!(crosses_market(true, 1.0, 1.0));
        assert!(!crosses_market(true, 1.1, 1.0));
    }
}
