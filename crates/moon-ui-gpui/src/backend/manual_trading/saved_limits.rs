//! Saved pending buys: a snapshot taken when buys are cancelled, and a one-key restore.
//!
//! Every Cancel Buy path (the chart button, the per-market hotkey, "cancel all buys", "cancel buys
//! in all bots") goes through [`Backend::cancel_buy_orders`], which calls
//! [`Backend::save_pending_buys`] first. The snapshot is per `(core, market)`: cancelling a market
//! replaces only that market's saved set, and a cancel that finds nothing to cancel never wipes a
//! set that is already saved.
//!
//! [`Backend::restore_canceled_buys_all_cores`] (the "Restore canceled buys" hotkey) places the
//! saved orders again on every online core. Each order is re-anchored to the CURRENT price at the
//! same distance it had when it was cancelled, because the price has usually moved since, and an
//! absolute price on the wrong side of the market would fill at once as a taker. An order that
//! cannot be placed safely (no current price, or it would cross the market) is skipped and stays
//! saved, so a second press retries it. An order that was sent is removed from the saved set, so a
//! repeated press cannot double the position.
//!
//! The saved set lives in `cfg/saved_limits.json`, so it survives a terminal restart.

use super::selectors::ready_cores;
use crate::Backend;
use moon_core::config::GroupExitSettings;
use moon_core::feed::OrderRow;
use moon_core::session::CoreId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// File name inside the `cfg/` directory.
const SAVED_FILE_NAME: &str = "saved_limits.json";

/// Restore at the same distance from the current price (`true`), or at the saved absolute price
/// (`false`). The distance mode is the safe one; see the module docs.
const RESTORE_BY_DISTANCE: bool = true;

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

impl Backend {
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

    /// Place the saved buys again on every online core. Returns how many orders were sent.
    ///
    /// Orders that cannot be placed safely stay saved; see the module docs.
    pub(crate) fn restore_canceled_buys_all_cores(&self) -> usize {
        let (ready, offline) = ready_cores(self.session.store().statuses());
        let mut sent_total = 0;
        let mut kept_total = 0;
        let mut file = store();
        for core in ready {
            let Some(markets) = file.cores.get_mut(&core) else {
                continue;
            };
            for (market, orders) in markets.iter_mut() {
                let source = self.session.market_source();
                let last = source
                    .latest_price(core, market)
                    .ok()
                    .map(f64::from)
                    .filter(|last| last.is_finite() && *last > 0.0);
                let tick = source.price_tick(core, market);
                orders.retain(|saved| {
                    let sent = self.place_saved_limit(core, saved, last, tick);
                    if sent {
                        sent_total += 1;
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
            "restore canceled buys: {sent_total} order(s) sent, {kept_total} kept for retry, \
             {offline} offline core(s) skipped"
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
        let result = if saved.pending {
            // A pending order's price is a trigger, so it is not compared with the market.
            self.session.place_pending_order(
                core,
                saved.market.clone(),
                saved.short,
                price,
                saved.size,
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
                saved.size,
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
                    saved.market, saved.short, saved.size, saved.pending
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
    fn crossing_is_detected_per_side() {
        assert!(crosses_market(false, 1.0, 1.0));
        assert!(!crosses_market(false, 0.9, 1.0));
        assert!(crosses_market(true, 1.0, 1.0));
        assert!(!crosses_market(true, 1.1, 1.0));
    }
}
