use core::fmt;
use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use base64::write;
use hashbrown::HashMap;
use siphasher::sip::SipHasher;
extern crate siphasher;

use crate::event::MonotonicTimestamp;
use crate::feedback::producers::{SimilarityConfig, SymmetryReductionScheme};
use crate::feedback::reward::{RewardFunction, SummaryTask};
#[cfg(feature = "selfcheck")]
use crate::nemesis::schedules::ScheduleId;
use crate::{
    event::{LamportEvent, ProcessId},
    feedback::producers::{EventKind, MMConfig, MMRewardConfig, MMState, MMStateRecordConfig},
};

const MAX_NUM_TO_DISPLAY: usize = 8;

#[derive(Clone)]
pub struct MMEventHistoryStat {
    pub overall_total_events: usize,
    pub overall_total_message_events: usize,
    pub overall_per_node_events: Vec<usize>,
    pub overall_per_node_message_events: Vec<usize>,
    pub overall_total_global_mm_states: usize,
    pub overall_total_per_event_mm_states: usize,
    pub overall_global_mm_states_sum: usize,
    pub overall_per_event_mm_states_sum: usize,
    pub total_distinct_global_mm_states: usize,
    pub top_hit_global_mm_states: HashMap<u64, usize>, // hash of MMState -> count
    pub total_distinct_per_event_mm_states: usize,
    pub top_hit_per_event_mm_states: HashMap<u64, usize>, // hash of MMState -> count
    pub global_mm_states: HashMap<MMState, usize>,
    pub per_event_mm_states: HashMap<EventKind, HashMap<MMState, usize>>,
    pub num_global_mm_states_since_last_reward: usize,
    pub num_per_event_mm_states_since_last_reward: usize,
    pub num_global_mm_states_diff_skipped: usize,
    pub num_per_event_mm_states_diff_skipped: usize,
    pub global_mm_states_diff_sum: usize,
    pub per_event_mm_states_diff_sum: usize,
    pub global_mm_states_diff_count: usize,
    pub per_event_mm_states_diff_count: usize,
    pub global_mm_states_hash_hit_count: usize,
    pub per_event_mm_states_hash_hit_count: usize,
    pub global_mm_states_diff_percent_sum: usize,
    pub per_event_mm_states_diff_percent_sum: usize,
    pub global_mm_config: Arc<MMConfig>,
    pub per_event_mm_config: Arc<MMConfig>,
    pub similarity_config: Arc<SimilarityConfig>,
    pub mm_state_record_config: Arc<MMStateRecordConfig>,
    pub mm_reward_config: Arc<MMRewardConfig>,
}

impl MMEventHistoryStat {
    pub fn new(
        global_mm_config: Arc<MMConfig>,
        per_event_mm_config: Arc<MMConfig>,
        similarity_config: Arc<SimilarityConfig>,
        mm_state_record_config: Arc<MMStateRecordConfig>,
        mm_reward_config: Arc<MMRewardConfig>,
    ) -> Self {
        MMEventHistoryStat {
            overall_total_events: 0,
            overall_total_message_events: 0,
            overall_per_node_events: vec![0; global_mm_config.num_nodes],
            overall_per_node_message_events: vec![0; global_mm_config.num_nodes],
            overall_total_global_mm_states: 0,
            overall_total_per_event_mm_states: 0,
            overall_global_mm_states_sum: 0,
            overall_per_event_mm_states_sum: 0,
            total_distinct_global_mm_states: 0,
            top_hit_global_mm_states: HashMap::new(),
            total_distinct_per_event_mm_states: 0,
            top_hit_per_event_mm_states: HashMap::new(),
            global_mm_states: HashMap::new(),
            per_event_mm_states: HashMap::new(),
            num_global_mm_states_diff_skipped: 0,
            num_per_event_mm_states_diff_skipped: 0,
            global_mm_states_diff_sum: 0,
            per_event_mm_states_diff_sum: 0,
            global_mm_states_diff_count: 0,
            per_event_mm_states_diff_count: 0,
            global_mm_states_hash_hit_count: 0,
            per_event_mm_states_hash_hit_count: 0,
            global_mm_states_diff_percent_sum: 0,
            per_event_mm_states_diff_percent_sum: 0,
            num_global_mm_states_since_last_reward: 0,
            num_per_event_mm_states_since_last_reward: 0,
            global_mm_config: global_mm_config.clone(),
            per_event_mm_config: per_event_mm_config.clone(),
            similarity_config: similarity_config.clone(),
            mm_state_record_config: mm_state_record_config.clone(),
            mm_reward_config: mm_reward_config.clone(),
        }
    }

    pub fn reset(&mut self) {
        self.num_global_mm_states_since_last_reward = 0;
        self.num_per_event_mm_states_since_last_reward = 0;
        // Do not need to clear the remaining fields, as they will be recorded across schedules.
    }

    pub fn update_event(&mut self, new_event: &LamportEvent, mm_event_history: &MMEventHistory) {
        let proc = new_event.proc();
        if let Some(event) = EventKind::from_event(new_event) {
            self.overall_total_events += 1;
            self.overall_per_node_events[proc as usize] += 1;
            if event.is_message_event() {
                self.overall_total_message_events += 1;
                self.overall_per_node_message_events[proc as usize] += 1;
            }

            if (!self
                .mm_state_record_config
                .record_global_mm_state_only_on_reward_reporting
                && self.overall_total_events
                    % self.mm_state_record_config.global_mm_state_record_interval
                    == 0)
            {
                self.update_state(&mm_event_history, true, None);
            }

            let record_interval = self
                .mm_state_record_config
                .per_event_mm_state_record_interval_normal;
            if self.mm_state_record_config.record_per_event_mm_state
                && event.is_execution()
                && ((self.overall_total_events - self.overall_total_message_events)
                    % record_interval
                    == record_interval - 1)
            {
                self.update_state(&mm_event_history, false, Some(&event));
            }
        }
    }

    pub fn update_state(
        &mut self,
        mm_event_history: &MMEventHistory,
        is_global_state: bool,
        event_kind: Option<&EventKind>,
    ) -> bool {
        let result = self._update_state(
            &mm_event_history.per_node_events,
            if is_global_state {
                &mm_event_history.current_global_state
            } else {
                &mm_event_history.current_per_event_state
            },
            is_global_state,
            event_kind,
            1,
        );
        if result {
            log::debug!(
                "[MMEventHistoryStat] New distinct {} MM state recorded: {}",
                if is_global_state {
                    "global"
                } else {
                    "per-event"
                },
                if is_global_state {
                    &mm_event_history.current_global_state
                } else {
                    &mm_event_history.current_per_event_state
                },
            )
        }
        return result;
    }

    fn _update_state(
        &mut self,
        per_node_events: &Vec<usize>,
        mm_state: &MMState,
        is_global_state: bool,
        event_kind: Option<&EventKind>,
        count_to_add: usize,
    ) -> bool {
        // returns true if a new distinct state is added

        if is_global_state {
            self.overall_total_global_mm_states += 1;
            self.overall_global_mm_states_sum += (mm_state.mm_sum + mm_state.vc_sum) as usize;
        } else {
            self.overall_total_per_event_mm_states += 1;
            self.overall_per_event_mm_states_sum += (mm_state.mm_sum + mm_state.vc_sum) as usize;
        }

        let mm_config = if is_global_state {
            self.global_mm_config.clone()
        } else {
            self.per_event_mm_config.clone()
        };
        let similarity_config = &self.similarity_config.clone();

        // 1. Create the Cow (Clone-on-Write)
        // `state_key` will be either a borrowed reference or a new, owned, sorted state.
        let state_key: Cow<MMState> =
            if mm_config.symmetry_reduction_scheme == SymmetryReductionScheme::Sort {
                Cow::Owned(mm_state.sort_and_copy(per_node_events))
            } else {
                Cow::Borrowed(mm_state)
            };

        let mut hasher = SipHasher::new_with_keys(0, 0);
        state_key.as_ref().hash(&mut hasher);
        let state_hash = hasher.finish();

        let (store, total_distinct_count, top_hit_counts, count_since_last_reward) =
            if is_global_state {
                (
                    &mut self.global_mm_states,
                    &mut self.total_distinct_global_mm_states,
                    &mut self.top_hit_global_mm_states,
                    &mut self.num_global_mm_states_since_last_reward,
                )
            } else {
                (
                    {
                        if let Some(sub_map) = self.per_event_mm_states.get_mut(event_kind.unwrap())
                        {
                            sub_map
                        } else {
                            self.per_event_mm_states
                                .insert(*event_kind.unwrap(), HashMap::new());
                            self.per_event_mm_states
                                .get_mut(event_kind.unwrap())
                                .unwrap()
                        }
                    },
                    &mut self.total_distinct_per_event_mm_states,
                    &mut self.top_hit_per_event_mm_states,
                    &mut self.num_per_event_mm_states_since_last_reward,
                )
            };

        // Use `state_key.as_ref()` for lookups.
        // `state_key.as_ref()` gives us &MMState, which works with HashMap.
        if let Some(existing_count) = store.get_mut(state_key.as_ref()) {
            *existing_count += count_to_add;

            Self::insert_record(state_hash, *existing_count, top_hit_counts);
            if is_global_state {
                self.global_mm_states_hash_hit_count += 1;
            } else {
                self.per_event_mm_states_hash_hit_count += 1;
            }
            false
        } else {
            let result = store.iter_mut().find(|(other_state, _)| {
                let comparision_result = Self::states_are_similar(
                    other_state,
                    state_key.as_ref(),
                    is_global_state,
                    mm_config.clone(),
                    similarity_config.clone(),
                );
                if comparision_result.1 {
                    if is_global_state {
                        self.num_global_mm_states_diff_skipped += 1;
                    } else {
                        self.num_per_event_mm_states_diff_skipped += 1;
                    }
                } else {
                    if is_global_state {
                        self.global_mm_states_diff_sum += comparision_result.2 as usize;
                        self.global_mm_states_diff_count += 1;
                        self.global_mm_states_diff_percent_sum += comparision_result.3 as usize;
                    } else {
                        self.per_event_mm_states_diff_sum += comparision_result.2 as usize;
                        self.per_event_mm_states_diff_count += 1;
                        self.per_event_mm_states_diff_percent_sum += comparision_result.3 as usize;
                    }
                };
                comparision_result.0
            });

            if let Some((_, existing_count)) = result {
                *existing_count += count_to_add;
                Self::insert_record(state_hash, *existing_count, top_hit_counts);
                false
            } else {
                // This is the magic:
                // - If it was Cow::Owned, it moves the value (no cost).
                // - If it was Cow::Borrowed, it clones it now (as desired).
                store.insert(state_key.into_owned(), count_to_add);
                *total_distinct_count += 1;
                *count_since_last_reward += 1;
                Self::insert_record(state_hash, count_to_add, top_hit_counts);
                true
            }
        }
    }

    fn states_are_similar(
        state1: &MMState,
        state2: &MMState,
        is_global_state: bool,
        mm_config: Arc<MMConfig>,
        similarity_config: Arc<SimilarityConfig>,
    ) -> (bool, bool, u64, u64) {
        // (is_similar, skipped_diff, diff, diff_percent)
        let use_absolute_threshold = similarity_config.mm_use_absolute_similarity_threshold;
        let (vc_sum1, mm_sum1) = (state1.vc_sum as u64, state1.mm_sum as u64);
        let (vc_sum2, mm_sum2) = (state2.vc_sum as u64, state2.mm_sum as u64);
        let vc_sum_max = std::cmp::max(vc_sum1, vc_sum2);
        let mm_sum_max = std::cmp::max(mm_sum1, mm_sum2);
        let vc_sum_min = std::cmp::min(vc_sum1, vc_sum2);
        let mm_sum_min = std::cmp::min(mm_sum1, mm_sum2);

        if mm_config.split_vc_mm {
            let (thres_vc, thres_mm) = if use_absolute_threshold {
                if is_global_state {
                    (
                        similarity_config
                            .mm_event_history_global_similarity_threshold_vc
                            .round() as u64,
                        similarity_config
                            .mm_event_history_global_similarity_threshold_mm
                            .round() as u64,
                    )
                } else {
                    (
                        similarity_config
                            .mm_event_history_per_event_similarity_threshold_vc
                            .round() as u64,
                        similarity_config
                            .mm_event_history_per_event_similarity_threshold_mm
                            .round() as u64,
                    )
                }
            } else {
                if is_global_state {
                    (
                        (similarity_config.mm_event_history_global_similarity_threshold_vc
                            * vc_sum_max as f64)
                            .round() as u64,
                        (similarity_config.mm_event_history_global_similarity_threshold_mm
                            * mm_sum_max as f64)
                            .round() as u64,
                    )
                } else {
                    (
                        (similarity_config.mm_event_history_per_event_similarity_threshold_vc
                            * vc_sum_max as f64)
                            .round() as u64,
                        (similarity_config.mm_event_history_per_event_similarity_threshold_mm
                            * mm_sum_max as f64)
                            .round() as u64,
                    )
                }
            };
            if vc_sum_min < (vc_sum_max - thres_vc) || mm_sum_min < (mm_sum_max - thres_mm) {
                return (false, true, 0, 0);
            }
            let (diff_vc, diff_mm) = state1.diff_split(state2);
            let diff_percent = if vc_sum_max + mm_sum_max > 0 {
                ((diff_vc + diff_mm) as f64 / (vc_sum_max + mm_sum_max) as f64 * 100.0).round()
                    as u64
            } else {
                0
            };
            return (
                diff_vc <= thres_vc && diff_mm <= thres_mm,
                false,
                diff_vc + diff_mm,
                diff_percent as u64,
            ); // HONGYI TODO: return both diffs?
        }

        // non-split case
        let sum1 = mm_sum1; // in non-split case, we only care about mm_sum and vc_sum is 0
        let sum2 = mm_sum2;
        let sum_max = sum1.max(sum2);
        let sum_min = sum1.min(sum2);
        let thres = if use_absolute_threshold {
            if is_global_state {
                similarity_config
                    .mm_event_history_global_similarity_threshold
                    .round() as u64
            } else {
                similarity_config
                    .mm_event_history_per_event_similarity_threshold
                    .round() as u64
            }
        } else {
            if is_global_state {
                (similarity_config.mm_event_history_global_similarity_threshold * sum_max as f64)
                    .round() as u64
            } else {
                (similarity_config.mm_event_history_per_event_similarity_threshold * sum_max as f64)
                    .round() as u64
            }
        };
        if sum_min < (sum_max - thres) {
            return (false, true, 0, 0);
        }
        let diff_percent = if sum_max > 0 {
            ((sum_max - sum_min) as f64 / sum_max as f64 * 100.0).round() as u64
        } else {
            0
        };
        let diff = state1.diff(state2);
        (diff <= thres, false, diff, diff_percent)
    }

    fn insert_record(hash: u64, count: usize, top_records: &mut HashMap<u64, usize>) {
        if top_records.len() < MAX_NUM_TO_DISPLAY {
            top_records.insert(hash, count);
            return;
        }

        if count <= *top_records.values().min().unwrap() {
            return;
        }

        let lowest_key: u64 = top_records
            .iter()
            .min_by_key(|&(_, &v)| v)
            .map(|(&k, _)| k)
            .unwrap();
        top_records.insert(hash, count);
        top_records.remove(&lowest_key);
    }

    pub fn print(&self) {
        log::info!("{}", self);
    }

    // HONGYI TODO: rethink this
    pub fn is_special_event(&self, event: &EventKind) -> bool {
        if let Some(per_event_map) = self.per_event_mm_states.get(event) {
            let total_count: usize = per_event_map.len();
            return total_count < self.mm_state_record_config.special_event_threshold;
        }
        true
    }

    pub fn report_reward(&mut self) -> f64 {
        // HONGYI TODO: customize reward function

        let cool_down_factor: f64 = if (self.mm_reward_config.use_cool_down_factor
            && self.mm_reward_config.cool_down_threshold > self.overall_total_global_mm_states)
        {
            (self.overall_total_global_mm_states as f64)
                .min(self.mm_reward_config.cool_down_threshold as f64)
                / self.mm_reward_config.cool_down_threshold as f64 // Not to favour early rewards too much
        } else {
            1.0
        };

        let mut reward: f64 = 0.0;
        if self.mm_state_record_config.record_per_event_mm_state {
            reward += self.num_per_event_mm_states_since_last_reward as f64
                * self.mm_reward_config.reward_on_per_event_state_change;
        }

        reward += self.num_global_mm_states_since_last_reward as f64
            * self.mm_reward_config.reward_on_global_state_change;

        self.num_global_mm_states_since_last_reward = 0;
        self.num_per_event_mm_states_since_last_reward = 0;

        cool_down_factor * (reward - 1.0) // penalty for not discovering new states
    }
}

impl RewardFunction for MMEventHistoryStat {
    fn reward_function(
        task: SummaryTask,
        _overall_cumulative: &Self,
        _schedule_cumulative: &Self,
        _schedule_window_summaries: &HashMap<crate::nemesis::schedules::StepId, Self>,
        _current: &mut Self,
        _state_similarity_threshold: f64,
    ) -> crate::feedback::reward::RewardEntry {
        let reward_value = _current.report_reward();
        crate::feedback::reward::RewardEntry::new(
            crate::feedback::producers::SummaryProducerIdentifier::EventHistory,
            task,
            reward_value,
        )
    }
}

impl fmt::Display for MMEventHistoryStat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "MMEventHistory[")?;

        // Global State Statistics
        writeln!(f, "GLOBAL_STATE_STATISTICS:")?;

        writeln!(
            f,
            "overall_total_message_events / overall_total_events: {} / {}",
            self.overall_total_message_events, self.overall_total_events,
        )?;

        writeln!(
            f,
            "overall_per_node_message_events / overall_per_node_events: {:?} / {:?}",
            self.overall_per_node_message_events, self.overall_per_node_events,
        )?;

        writeln!(
            f,
            "overall_global_mm_states_sum_avg: {:.4}",
            if self.overall_total_global_mm_states > 0 {
                self.overall_global_mm_states_sum as f64
                    / self.overall_total_global_mm_states as f64
            } else {
                0.0
            }
        )?;

        writeln!(
            f,
            "total_distinct_global_mm_states / overall_total_global_mm_states: {} / {}",
            self.total_distinct_global_mm_states, self.overall_total_global_mm_states,
        )?;

        let top_global_mm_states = &self
            .top_hit_global_mm_states
            .values()
            .collect::<Vec<&usize>>();
        writeln!(f, "top_global_mm_states: {:?}", top_global_mm_states)?;

        writeln!(
            f,
            "global_mm_states_hash_hit_count: {}",
            self.global_mm_states_hash_hit_count
        )?;

        writeln!(
            f,
            "global_mm_states_diff_skipped: {}",
            self.num_global_mm_states_diff_skipped
        )?;

        writeln!(
            f,
            "global_mm_states_diff_count: {}",
            self.global_mm_states_diff_count
        )?;

        writeln!(
            f,
            "global_mm_states_diff_avg: {:.2}",
            if self.global_mm_states_diff_count > 0 {
                self.global_mm_states_diff_sum as f64 / self.global_mm_states_diff_count as f64
            } else {
                0.0
            }
        )?;

        writeln!(
            f,
            "global_mm_states_diff_percent_avg: {:.4}%",
            if self.global_mm_states_diff_count > 0 {
                self.global_mm_states_diff_percent_sum as f64
                    / self.global_mm_states_diff_count as f64
            } else {
                0.0
            }
        )?;

        // Per-Event State Statistics
        writeln!(f, "PER_EVENT_STATE_STATISTICS:")?;

        writeln!(
            f,
            "overall_per_event_mm_states_sum_avg: {:.4}",
            if self.overall_total_per_event_mm_states > 0 {
                self.overall_per_event_mm_states_sum as f64
                    / self.overall_total_per_event_mm_states as f64
            } else {
                0.0
            },
        )?;

        writeln!(
            f,
            "total_distinct_per_event_mm_states / overall_total_per_event_mm_states: {} / {}",
            self.total_distinct_per_event_mm_states, self.overall_total_per_event_mm_states,
        )?;

        let top_per_event_mm_states = &self
            .top_hit_per_event_mm_states
            .values()
            .collect::<Vec<&usize>>();
        writeln!(f, "top_per_event_mm_states: {:?}", top_per_event_mm_states)?;

        writeln!(
            f,
            "per_event_mm_states_hash_hit_count: {}",
            self.per_event_mm_states_hash_hit_count
        )?;

        writeln!(
            f,
            "per_event_mm_states_diff_skipped: {}",
            self.num_per_event_mm_states_diff_skipped
        )?;

        writeln!(
            f,
            "per_event_mm_states_diff_count: {}",
            self.per_event_mm_states_diff_count
        )?;

        writeln!(
            f,
            "per_event_mm_states_diff_avg: {:.2}",
            if self.per_event_mm_states_diff_count > 0 {
                self.per_event_mm_states_diff_sum as f64
                    / self.per_event_mm_states_diff_count as f64
            } else {
                0.0
            }
        )?;

        writeln!(
            f,
            "per_event_mm_states_diff_percent_avg: {:.4}%",
            if self.per_event_mm_states_diff_count > 0 {
                self.per_event_mm_states_diff_percent_sum as f64
                    / self.per_event_mm_states_diff_count as f64
            } else {
                0.0
            }
        )?;

        writeln!(f, "END_MMEventHistory]")?;

        Ok(())
    }
}

#[derive(Clone)]
pub struct MMEventHistoryPerfStat {
    pub mm_event_history_update_time: u128,
    pub old_event_history_update_time: u128,
}

impl MMEventHistoryPerfStat {
    pub fn new() -> Self {
        MMEventHistoryPerfStat {
            mm_event_history_update_time: 0,
            old_event_history_update_time: 0,
        }
    }

    pub fn print(&self) {
        log::info!(
            "[MMEventHistoryPerfStat] mm_event_history_update_time (ms): {}",
            self.mm_event_history_update_time
        );
        log::info!(
            "[MMEventHistoryPerfStat] old_event_history_update_time (ms): {}",
            self.old_event_history_update_time
        );
    }

    pub fn increment_mm_event_history_update_time(&mut self, time_ms: u128) {
        self.mm_event_history_update_time += time_ms;
    }

    pub fn increment_old_event_history_update_time(&mut self, time_ms: u128) {
        self.old_event_history_update_time += time_ms;
    }
}

impl Default for MMEventHistoryPerfStat {
    fn default() -> Self {
        MMEventHistoryPerfStat {
            mm_event_history_update_time: 0,
            old_event_history_update_time: 0,
        }
    }
}

impl fmt::Display for MMEventHistoryPerfStat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "MMEventHistoryPerfStat[")?;
        writeln!(
            f,
            "mm_event_history_update_time (ms): {}",
            self.mm_event_history_update_time,
        )?;
        writeln!(
            f,
            "old_event_history_update_time (ms): {}",
            self.old_event_history_update_time
        )?;
        writeln!(f, "END_MMEventHistoryPerfStat]")?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct MMEventHistory {
    num_nodes: usize,

    pub current_global_state: MMState,

    pub current_per_event_state: MMState,

    pub per_node_events: Vec<usize>,

    v_ts: HashMap<ProcessId, MonotonicTimestamp>,

    mm_state_record_config: Arc<MMStateRecordConfig>,

    #[cfg(feature = "selfcheck")]
    /// For debugging purposes.
    pub last_reset: HashMap<ProcessId, (ScheduleId, StepId)>,
}

impl std::hash::Hash for MMEventHistory {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.current_global_state.hash(state);
        if self.mm_state_record_config.record_per_event_mm_state {
            self.current_per_event_state.hash(state);
        }
    }
}

impl fmt::Display for MMEventHistory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            "Current event sum: {}",
            self.per_node_events.iter().sum::<usize>()
        )?;
        writeln!(f, "Current per_node_events: {:?}", self.per_node_events)?;
        Ok(())
    }
}

impl MMEventHistory {
    fn get_ts(&self, proc: ProcessId) -> MonotonicTimestamp {
        *self
            .v_ts
            .get(&proc)
            .unwrap_or(&MonotonicTimestamp::from(0, proc))
    }

    fn _merge_update_ts(&mut self, proc: ProcessId, ts: MonotonicTimestamp) -> bool {
        let old_ts = self.get_ts(proc);
        if ts > old_ts {
            self.v_ts.insert(proc, ts);
            return true;
        }
        false
    }

    pub fn new(
        mm_state_record_config: Arc<MMStateRecordConfig>,
        global_mm_config: Arc<MMConfig>,
        per_event_mm_config: Arc<MMConfig>,
    ) -> Self {
        MMEventHistory {
            num_nodes: global_mm_config.num_nodes,
            current_global_state: MMState::from_config(global_mm_config.clone()),
            current_per_event_state: MMState::from_config(per_event_mm_config.clone()),
            per_node_events: vec![0; global_mm_config.num_nodes],
            v_ts: HashMap::new(),
            mm_state_record_config: mm_state_record_config.clone(),
            #[cfg(feature = "selfcheck")]
            last_reset: HashMap::new(),
        }
    }

    pub fn reset(&mut self) {
        self.current_global_state.reset();
        if self.mm_state_record_config.record_per_event_mm_state {
            self.current_per_event_state.reset();
        }
        self.per_node_events = vec![0; self.num_nodes];
        self.v_ts.clear();
        log::debug!("[MMEventHistory] ResetSummary.")
    }

    // pub fn can_update(&self, new_event: &LamportEvent) -> bool {
    //     let proc = new_event.proc();
    //     let ts = new_event.ts();
    //     let old_ts = self.get_ts(proc);
    //     return ts > old_ts;
    // }

    pub fn update(&mut self, new_event: &LamportEvent) {
        if let Some(_) = EventKind::from_event(new_event) {
            let proc = new_event.proc();
            let ts = new_event.ts();
            self._merge_update_ts(proc, ts);

            let local_logical_clock = self.per_node_events[proc as usize];
            self.current_global_state
                .update(new_event, local_logical_clock);
            if self.mm_state_record_config.record_per_event_mm_state {
                self.current_per_event_state
                    .update(new_event, local_logical_clock);
            }

            self.per_node_events[proc as usize] += 1;
        }
    }

    pub fn merge(&mut self, this_event: &LamportEvent, other: &Self, other_event: &LamportEvent) {
        self.current_global_state.merge(&other.current_global_state);
        if self.mm_state_record_config.record_per_event_mm_state {
            self.current_per_event_state
                .merge(&other.current_per_event_state);
        }
        self._merge_update_ts(other_event.proc(), other_event.ts());
        for (i, count) in self.per_node_events.iter_mut().enumerate() {
            *count = (*count).max(other.per_node_events[i]);
        }
    }

    pub fn union(&mut self, other: &Self) {
        // No op
    }
}
