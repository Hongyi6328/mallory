pub mod afl;
pub mod eventcount;
pub mod eventhistory;
pub mod mmeventhistory;
pub mod vectorclock;

use core::fmt;
use ndarray::{Array, ArrayD, Axis, Ix4, Ix5, ShapeBuilder, Zip};
use std::cell::Cell;
use std::sync::{Arc, RwLock};

use enum_dispatch::enum_dispatch;
use hashbrown::{HashMap, HashSet};
use siphasher::sip::SipHasher;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use crate::event::{AdministrativeEvent, BlockId, Event, FunctionId, LamportEvent, NodeId};

pub use self::{
    afl::AFLBranchFeedback, eventcount::EventCount, eventhistory::EventHistory,
    vectorclock::VectorClock,
};

use super::summary::SummaryWrapper;

#[enum_dispatch(SummaryControl)]
pub enum SummaryKind {
    VectorClock(SummaryWrapper<VectorClock>),
    EventCount(SummaryWrapper<EventCount>),
    EventHistory(SummaryWrapper<EventHistory>),
    AFLBranchFeedback(SummaryWrapper<AFLBranchFeedback>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum SummaryProducerIdentifier {
    EventHistory,
    AFLBranchFeedback,
    #[default]
    Unspecified,
}

pub trait SummaryProducer {
    fn new() -> Self;

    /// Restore the summary to a pristine state.
    fn reset(&mut self);

    /// Update the summary with a new event.
    fn update(&mut self, new_event: &LamportEvent, state_similarity_threshold: f64);

    /// Merge two summaries from different processes.
    fn merge(&mut self, this_event: &LamportEvent, other: &Self, other_event: &LamportEvent);

    /// Union two summaries to obtain a cumulative summary.
    /// This is different from merging in that it does not
    /// take pairwise-max, but treats the two as disjoint and
    /// adds the components together.
    fn union(&mut self, other: &Self);
}

/// A wrapper around HashSet that that has a 'hash' field that can
/// be used to detect changes.
#[derive(Debug, Clone)]
pub struct ChangeAwareSet<T> {
    hash: u64,
    set: HashSet<T>,

    pub _num_skipped: usize,
}

impl<T: Eq + Hash + Clone> ChangeAwareSet<T> {
    pub fn new() -> ChangeAwareSet<T> {
        ChangeAwareSet {
            hash: 0,
            set: HashSet::new(),
            _num_skipped: 0,
        }
    }

    pub fn insert(&mut self, value: T) -> bool {
        let new_hash = {
            let mut hasher = SipHasher::new_with_keys(0, 0);
            value.hash(&mut hasher);
            self.hash ^ hasher.finish()
        };
        let changed = self.set.insert(value);
        if changed {
            self.hash = new_hash;
        }
        changed
    }

    pub fn contains(&self, value: &T) -> bool {
        self.set.contains(value)
    }

    pub fn len(&self) -> usize {
        self.set.len()
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn different_from(&self, other: &ChangeAwareSet<T>) -> bool {
        self.hash != other.hash
    }

    /// First checks that the hash is different, and if so, extends.
    pub fn extend(&mut self, other: &ChangeAwareSet<T>) {
        if self.different_from(other) {
            for value in &other.set {
                self.insert(value.clone());
            }
        } else {
            self._num_skipped += other.len();
        }
    }

    pub fn inner_set(&self) -> &HashSet<T> {
        &self.set
    }

    pub fn iter(&self) -> hashbrown::hash_set::Iter<T> {
        self.set.iter()
    }

    pub fn clear(&mut self) {
        self.set.clear();
        self.hash = 0;
    }
}

impl<T: Eq + Hash + Clone> Default for ChangeAwareSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> IntoIterator for &'a ChangeAwareSet<T> {
    type Item = &'a T;
    type IntoIter = hashbrown::hash_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.set.iter()
    }
}

/// A change-aware set for pairs and triplets of values.
#[derive(Debug, Clone)]
pub struct LayeredChangeAwareSet<T> {
    _id: u32,

    single: ChangeAwareSet<T>,
    // Pairs are obtained my adding a T2 after every T1 in the single set.
    pairs: ChangeAwareSet<(T, T)>,
    hash_of_single_when_last_added: HashMap<T, u64>,
    _num_skipped_pairs: usize,

    // Triplets are obtained by adding a T3 after every (T1, T2) in the pairs set.
    triplets: ChangeAwareSet<(T, T, T)>,
    hash_of_pairs_when_last_added: HashMap<T, u64>,
    _num_skipped_triplets: usize,
}

impl<T: Eq + Hash + Clone + Debug> LayeredChangeAwareSet<T> {
    pub fn new() -> LayeredChangeAwareSet<T> {
        LayeredChangeAwareSet {
            _id: rand::random(),
            single: ChangeAwareSet::new(),
            pairs: ChangeAwareSet::new(),
            hash_of_single_when_last_added: HashMap::new(),
            _num_skipped_pairs: 0,
            triplets: ChangeAwareSet::new(),
            hash_of_pairs_when_last_added: HashMap::new(),
            _num_skipped_triplets: 0,
        }
    }

    fn insert_single(&mut self, value: T) -> bool {
        self.single.insert(value)
    }

    fn insert_pairs_with(&mut self, value: T) {
        let last_hash = self
            .hash_of_single_when_last_added
            .entry(value.clone())
            .or_default();
        // If single hasn't changed since we last added `value` to pairs,
        // then there is nothing to do.
        if *last_hash == self.single.hash() {
            self._num_skipped_pairs += self.single.len();
            return;
        }
        // let _prev_hash = *last_hash;
        // let _curr_hash = self.single.hash();
        *last_hash = self.single.hash();

        // let prev_pairs_hash = self.triplets.hash();
        // Add all pairs to the pairs set.
        let mut _num_added: usize = 0;
        for ev_a in &self.single {
            let pair = (ev_a.clone(), value.clone());
            _num_added += if self.pairs.insert(pair) { 1 } else { 0 };
        }
        // log::info!(
        //     "[LAYERS {}] Added {} pairs with {:?} (on top of hash {} | last seen {}); pairs: {} => {}",
        //     self._id,
        //     _num_added,
        //     value,
        //     _curr_hash,
        //     _prev_hash,
        //     prev_pairs_hash,
        //     self.pairs.hash()
        // );
    }

    fn insert_triplets_with(&mut self, value: T) {
        let last_hash = self
            .hash_of_pairs_when_last_added
            .entry(value.clone())
            .or_default();
        // If pairs hasn't changed since we last added `value` to triplets,
        // then there is nothing to do.
        if *last_hash == self.pairs.hash() {
            self._num_skipped_triplets += self.pairs.len();
            return;
        }
        // let _prev_hash = *last_hash;
        // let _curr_hash = self.pairs.hash();
        *last_hash = self.pairs.hash();

        // let prev_triplets_hash = self.triplets.hash();
        // Add all triplets to the triplets set.
        let mut _num_added: usize = 0;
        for (ev_a, ev_b) in &self.pairs {
            let triplet = (ev_a.clone(), ev_b.clone(), value.clone());
            _num_added += if self.triplets.insert(triplet) { 1 } else { 0 };
        }
        // log::info!(
        //     "[LAYERS {}] Added {} triplets with {:?} (on top of hash {} | last seen {}); triplets: {} => {}",
        //     self._id,
        //     _num_added,
        //     value,
        //     _curr_hash,
        //     _prev_hash,
        //     prev_triplets_hash,
        //     self.triplets.hash()
        // );
    }

    pub fn insert(&mut self, value: T) {
        // log::info!("[LAYERS {}] Inserting {:?}", self._id, value);
        // self.insert_triplets_with(value.clone());
        self.insert_pairs_with(value.clone());
        self.insert_single(value);
    }

    pub fn get_single(&self) -> &ChangeAwareSet<T> {
        &self.single
    }

    pub fn get_pairs(&self) -> &ChangeAwareSet<(T, T)> {
        &self.pairs
    }

    pub fn get_triplets(&self) -> &ChangeAwareSet<(T, T, T)> {
        &self.triplets
    }

    /// TODO: can this be improved?
    pub fn extend(&mut self, other: &LayeredChangeAwareSet<T>) {
        self.single.extend(&other.single);
        self.pairs.extend(&other.pairs);
        self.triplets.extend(&other.triplets);
        self._num_skipped_pairs += other._num_skipped_pairs;
        self._num_skipped_triplets += other._num_skipped_triplets;
    }

    pub fn get_stats(&self) -> (usize, usize, usize) {
        (
            self.single._num_skipped,
            self._num_skipped_pairs,
            self._num_skipped_triplets,
        )
    }
}

impl<T: Eq + Hash + Clone + Debug> Default for LayeredChangeAwareSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Coalesces concrete events into their "kind". This only collects "real"
/// events, i.e. of the SUT, and not "environment" events, like `ClientRequest`
/// `ClientResponse` or `Fault`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub enum EventKind {
    BlockExecute { block_id: BlockId },
    FunctionExecute { function_id: FunctionId },
    PacketSend { data: u32, from: NodeId, to: NodeId },
    PacketReceive { data: u32, from: NodeId, to: NodeId },
    ResetSummary,
}

impl fmt::Display for EventKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BlockExecute { block_id } => write!(f, "BB({})", block_id),
            Self::FunctionExecute { function_id } => write!(f, "F({})", function_id),
            Self::PacketSend { data, from, to } => {
                write!(f, "Send({} from {} to {})", data, from, to)
            }
            Self::PacketReceive { data, from, to } => {
                write!(f, "Receive({} from {} at {})", data, from, to)
            }
            Self::ResetSummary => write!(f, "Reset"),
        }
    }
}

impl EventKind {
    fn from_event(event: &LamportEvent) -> Option<EventKind> {
        match event.bare_event() {
            Event::BlockExecute { block_id, .. } => Some(EventKind::BlockExecute {
                block_id: *block_id,
            }),
            Event::FunctionExecute { function_id, .. } => Some(EventKind::FunctionExecute {
                function_id: *function_id,
            }),
            Event::PacketSend { data, to, .. } => Some(EventKind::PacketSend {
                data: *data,
                from: event.proc(),
                to: *to,
            }),
            Event::PacketReceive { data, from, .. } => Some(EventKind::PacketReceive {
                data: *data,
                from: *from,
                to: event.proc(),
            }),
            Event::TimelineEvent(AdministrativeEvent::StartWindow { .. }) => {
                Some(EventKind::ResetSummary)
            }
            _ => None,
        }
    }

    fn is_packet_recv(&self) -> bool {
        matches!(self, Self::PacketReceive { .. })
    }

    fn is_message_event(&self) -> bool {
        matches!(self, Self::PacketReceive { .. } | Self::PacketSend { .. })
    }

    fn is_execution(&self) -> bool {
        matches!(
            self,
            Self::BlockExecute { .. } | Self::FunctionExecute { .. }
        )
    }
}

#[derive(Clone)]
pub enum LocalEventKind {
    BlockExecute { block_id: BlockId },
    FunctionExecute { function_id: FunctionId },
}

#[derive(Clone, PartialEq, Eq)]
pub struct PartitionScheme {
    pub num_local_event_partitions: usize, // > 0; if 1, no partitioning
    pub local_event_partition_len: usize,  // > 0
    pub num_message_events_partitions: usize,
    pub message_event_partition_len: usize,
}

#[derive(Clone, PartialEq, Eq)]
pub enum SymmetryReductionScheme {
    None,
    Sort,
    Merge, // HONGYI TODO: implement Merge
}

#[derive(Clone, PartialEq, Eq)]
pub struct EventKindConfig {
    pub num_local_event_kinds: usize, // can be 0, which means no local event kinds are recorded
    pub num_message_event_kinds: usize, // can be 0, which means no message event kinds are recorded
    pub categorize_message_by_data: bool,
}

impl Default for EventKindConfig {
    fn default() -> Self {
        EventKindConfig {
            num_local_event_kinds: 1,
            num_message_event_kinds: 1,
            categorize_message_by_data: false,
        }
    }
}

#[derive(Clone)]
pub struct MMEventKindMapper {
    exec_map: HashMap<EventKind, usize>,
    exec_map_range: usize,
    exec_map_choice: Vec<usize>,
    message_map: HashMap<usize, usize>, // Map by length
    message_map_range: usize,
    message_map_choice: Vec<usize>,
    categorize_message_by_data: bool,
}

impl MMEventKindMapper {
    pub fn from_config(config: &EventKindConfig) -> Self {
        let exec_map: HashMap<EventKind, usize> = HashMap::new();
        let mut exec_map_choice: Vec<usize> = Vec::new();
        let message_map: HashMap<usize, usize> = HashMap::new(); // Map by length
        let mut message_map_choice: Vec<usize> = Vec::new();

        let num_local_event_kinds = config.num_local_event_kinds;
        let num_message_event_kinds = config.num_message_event_kinds;

        for i in 0..num_local_event_kinds {
            exec_map_choice.push(i);
        }

        for i in 0..num_message_event_kinds {
            message_map_choice.push(i);
        }

        MMEventKindMapper {
            exec_map,
            exec_map_range: num_local_event_kinds,
            exec_map_choice,
            message_map,
            message_map_range: num_message_event_kinds,
            message_map_choice,
            categorize_message_by_data: config.categorize_message_by_data,
        }
    }

    fn get_exec_kind(&mut self, event_kind: &EventKind) -> usize {
        assert!(
            event_kind.is_execution(),
            "EventKind {:?} is not an execution kind.",
            event_kind
        );

        if let Some(kind) = self.exec_map.get(event_kind) {
            *kind
        } else {
            if self.exec_map_choice.is_empty() {
                for i in 0..self.exec_map_range {
                    self.exec_map_choice.push(i);
                }
            }

            // HONGYI TODO: fix non-determinism
            // --- FIX: Use a deterministic hash ---
            let mut hasher = SipHasher::new_with_keys(0, 0);
            event_kind.hash(&mut hasher);
            let kind = (hasher.finish() as usize) % self.exec_map_range;
            self.exec_map.insert(event_kind.clone(), kind);
            kind
            // --- END FIX ---

            // let idx_to_remove = rand::random::<usize>() % self.exec_map_choice.len();
            // let choice = self.exec_map_choice.remove(idx_to_remove);
            // self.exec_map.insert(event_kind.clone(), choice);
            // choice
        }
    }

    fn get_message_kind(&self, event_kind: &EventKind) -> usize {
        match event_kind {
            EventKind::PacketSend { data: d, .. } | EventKind::PacketReceive { data: d, .. } => {
                let data = *d;
                if self.categorize_message_by_data {
                    return (data as usize) % self.message_map_range;
                } else {
                    let len = data;
                    return (data as usize) % self.message_map_range; // HONGYI TODO: use len
                }
            }
            _ => {
                panic!("EventKind {:?} is not a message kind.", event_kind)
            }
        }
    }
}

#[derive(Clone)]
pub struct MMConfig {
    pub num_nodes: usize,
    pub split_vc_mm: bool,
    pub record_sender_partition: bool,
    pub partition_scheme: PartitionScheme,
    pub symmetry_reduction_scheme: SymmetryReductionScheme,
    pub event_kind_config: EventKindConfig,
    pub event_mapper: Arc<RwLock<MMEventKindMapper>>,
}

impl MMConfig {
    pub fn new(
        num_nodes: usize,
        split_vc_mm: bool,
        record_sender_partition: bool,
        partition_scheme: PartitionScheme,
        symmetry_reduction_scheme: SymmetryReductionScheme,
        event_kind_config: EventKindConfig,
        event_mapper: Arc<RwLock<MMEventKindMapper>>,
    ) -> Self {
        assert! {num_nodes > 0, "N must be greater than 0."};
        assert! {partition_scheme.num_local_event_partitions > 0, "partition_scheme.num_local_event_partitions must be greater than 0."};
        assert! {partition_scheme.num_message_events_partitions > 0, "partition_scheme.num_message_events_partitionas must be greater than 0."};
        assert! {partition_scheme.local_event_partition_len > 0, "partition_scheme.local_event_partition_len must be greater than 0."};
        assert! {partition_scheme.message_event_partition_len > 0, "partition_scheme.message_event_partition_len must be greater than 0."};
        assert! {split_vc_mm || !record_sender_partition, "If split_vc_mm is false, record_sender_partition must also be false."};
        assert! {split_vc_mm || (event_kind_config.num_local_event_kinds == event_kind_config.num_message_event_kinds), "If split_vc_mm is false, event_kind_config must have equal number of local event kinds and message event kind."};
        assert! {split_vc_mm || (partition_scheme.num_local_event_partitions == partition_scheme.num_message_events_partitions), "If split_vc_mm is false, partition_scheme must have equal number of local event partitions and message event partitions."};
        // assert! {event_kind_config.num_local_event_kinds > 0 || !split_vc_mm, "If split_vc_mm is true, event_kind_config must have at least one local event kind."};
        // assert! {event_kind_config.num_message_event_kinds > 0, "event_kind_config must have at least one message event kind."};

        MMConfig {
            num_nodes,
            split_vc_mm,
            record_sender_partition,
            partition_scheme,
            symmetry_reduction_scheme,
            event_kind_config,
            event_mapper,
        }
    }
}

pub trait AbstractMMState: Clone + Eq {}

#[derive(Clone)]
pub struct MMState {
    pub vc_sum: usize,
    pub mm_sum: usize,
    pub vc: ArrayD<usize>,
    pub mm: ArrayD<usize>,
    config: Arc<MMConfig>,
    has_changed: Cell<bool>,
    hash: Cell<u64>,
}

impl PartialEq for MMState {
    fn eq(&self, other: &Self) -> bool {
        if self.vc_sum != other.vc_sum || self.mm_sum != other.mm_sum {
            return false;
        }
        return self.vc == other.vc && self.mm == other.mm;
    }
}

impl Eq for MMState {}

impl Hash for MMState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if self.has_changed.get() {
            let mut hasher = SipHasher::new_with_keys(0, 0);
            self.vc.hash(&mut hasher);
            self.mm.hash(&mut hasher);
            let new_hash = hasher.finish();
            self.hash.set(new_hash);
            self.has_changed.set(false);
            state.write_u64(new_hash);
        } else {
            // use cached hash
            state.write_u64(self.hash.get());
        }
    }
}

impl MMState {
    pub fn from_config(mm_config: Arc<MMConfig>) -> Self {
        let num_nodes = mm_config.num_nodes;
        let partition_scheme = &mm_config.partition_scheme;
        let event_kind_config = &mm_config.event_kind_config;

        let vc = if mm_config.split_vc_mm && event_kind_config.num_local_event_kinds > 0 {
            Array::zeros((
                num_nodes,
                event_kind_config.num_local_event_kinds,
                partition_scheme.num_local_event_partitions,
            ))
            .into_dyn()
        } else {
            Array::zeros((1).f()).into_dyn()
        };

        let mm = if event_kind_config.num_message_event_kinds > 0 {
            if mm_config.record_sender_partition {
                Array::<usize, Ix5>::zeros(
                    (
                        num_nodes,
                        num_nodes,
                        event_kind_config.num_message_event_kinds,
                        partition_scheme.num_message_events_partitions, // NOTE: receiver partition
                        partition_scheme.num_message_events_partitions, // NOTE: sender partition
                    )
                        .f(),
                )
                .into_dyn()
            } else {
                Array::<usize, Ix4>::zeros(
                    (
                        num_nodes,
                        num_nodes,
                        event_kind_config.num_message_event_kinds,
                        partition_scheme.num_message_events_partitions,
                    )
                        .f(),
                )
                .into_dyn()
            }
        } else {
            Array::zeros((1).f()).into_dyn()
        };

        MMState {
            vc_sum: 0,
            mm_sum: 0,
            vc,
            mm,
            config: mm_config.clone(),
            has_changed: Cell::new(true),
            hash: Cell::new(0),
        }
    }

    fn get_partition_id(&self, local_logical_clock: usize, is_local_event: bool) -> usize {
        let partition_scheme = &self.config.partition_scheme;

        let (num_partitions, partition_len) = if is_local_event {
            (
                partition_scheme.num_local_event_partitions,
                partition_scheme.local_event_partition_len,
            )
        } else {
            (
                partition_scheme.num_message_events_partitions,
                partition_scheme.message_event_partition_len,
            )
        };
        if num_partitions == 1 {
            0
        } else {
            (local_logical_clock % (num_partitions * partition_len)) / partition_len
        }
    }

    pub fn has_same_config(&self, other: &MMState) -> bool {
        Arc::ptr_eq(&self.config, &other.config)
    }

    pub fn update(&mut self, event: &LamportEvent, local_logical_clock: usize) -> bool {
        let changed = self._update(event, local_logical_clock);
        if changed {
            self.has_changed.set(true);
        }
        changed
    }

    fn _update(&mut self, event: &LamportEvent, local_logical_clock: usize) -> bool {
        // returns true if the MM state is updated
        let proc = event.proc() as usize;
        if let Some(ev_kind) = EventKind::from_event(event) {
            if ev_kind.is_execution() {
                if self.config.event_kind_config.num_local_event_kinds == 0 {
                    return false;
                }
                let mapped_kind = self
                    .config
                    .event_mapper
                    .write()
                    .unwrap()
                    .get_exec_kind(&ev_kind);
                let partition_id = self.get_partition_id(local_logical_clock, true);
                if self.config.split_vc_mm {
                    self.vc[[proc, mapped_kind, partition_id]] += 1;
                    self.vc_sum += 1;
                } else {
                    if self.config.record_sender_partition {
                        self.mm[[proc, proc, mapped_kind, partition_id, 0]] += 1;
                        self.mm_sum += 1;
                    // HONGYI TODO: do it more elegantly
                    } else {
                        self.mm[[proc, proc, mapped_kind, partition_id]] += 1;
                        self.mm_sum += 1;
                    }
                }
                true
            } else if ev_kind.is_message_event() {
                if self.config.event_kind_config.num_message_event_kinds == 0 {
                    return false;
                }
                let kind = self
                    .config
                    .event_mapper
                    .read()
                    .unwrap()
                    .get_message_kind(&ev_kind);
                let partition_id = self.get_partition_id(local_logical_clock, false);
                match ev_kind {
                    EventKind::PacketSend { to, .. } => {
                        // let to_proc = to as usize;
                        // if self.config.record_sender_partition {
                        //     let to_partition_id = self.get_partition_id(local_logical_clock, false); // HONGYI TODO: get the real sender partition id
                        //     self.mm[[proc, to_proc, kind, to_partition_id, partition_id]] += 1;
                        // } else {
                        //     self.mm[[proc, to_proc, kind, partition_id]] += 1; // HONGYI TODO: get the real sender partition id and set up a receive matrix instead
                        // }
                        false
                    }
                    EventKind::PacketReceive { from, .. } => {
                        let from_proc = from as usize;
                        if self.config.record_sender_partition {
                            let from_partition_id =
                                self.get_partition_id(local_logical_clock, false); // HONGYI TODO: get the real sender partition id
                            self.mm[[from_proc, proc, kind, from_partition_id, partition_id]] += 1;
                            self.mm_sum += 1;
                        } else {
                            self.mm[[from_proc, proc, kind, partition_id]] += 1;
                            self.mm_sum += 1;
                        }
                        true
                    }
                    _ => false,
                }
            } else {
                false
            }
        } else {
            false
        }
    }

    // after sorting, return a new MMState
    // such that mm is sorted along the first axis according to args
    // and each of mm[i] is also sorted according to args
    // args.len() is guaranteed to be equal to mm.shape()[0] and mm.shape()[1]
    pub fn sort_and_copy(&self, num_events: &Vec<usize>) -> MMState {
        assert!(
            self.config.symmetry_reduction_scheme == SymmetryReductionScheme::Sort,
            "sort_and_copy can only be used with SymmetryReductionScheme::Sort."
        );

        let mut permutation: Vec<usize> = (0..num_events.len()).collect();
        permutation.sort_unstable_by_key(|&i| num_events[i]);

        let sorted_vc = if self.config.split_vc_mm {
            self.vc.select(Axis(0), &permutation) // Sorts along Axis 0 (rows)
        } else {
            self.vc.clone()
        };

        let sorted_mm = self
            .mm
            .select(Axis(0), &permutation) // Sorts along Axis 0 (rows)
            .select(Axis(1), &permutation); // Sorts along Axis 1 (cols)

        MMState {
            vc_sum: self.vc_sum.clone(),
            mm_sum: self.mm_sum.clone(),
            vc: sorted_vc,
            mm: sorted_mm,
            config: self.config.clone(),
            has_changed: Cell::new(true),
            hash: Cell::new(0),
        }
    }

    pub fn diff(&self, other: &MMState) -> u64 {
        assert!(
            self.has_same_config(other),
            "Cannot diff MMStates with different configurations."
        );
        assert!(
            !self.config.split_vc_mm,
            "diff can only be used when split_vc_mm is false."
        );

        let mut diff: u64 = 0;

        Zip::from(&self.mm)
            .and(&other.mm)
            .for_each(|&elem1, &elem2| {
                diff += elem1.abs_diff(elem2) as u64;
            });
        diff
    }

    pub fn diff_split(&self, other: &MMState) -> (u64, u64) {
        assert!(
            self.has_same_config(other),
            "Cannot diff_split MMStates with different configurations."
        );
        assert!(
            self.config.split_vc_mm,
            "diff_split can only be used when split_vc_mm is true."
        );

        let mut diff_vc: u64 = 0;
        let mut diff_mm: u64 = 0;

        Zip::from(&self.vc)
            .and(&other.vc)
            .for_each(|&elem1, &elem2| {
                diff_vc += elem1.abs_diff(elem2) as u64;
            });

        Zip::from(&self.mm)
            .and(&other.mm)
            .for_each(|&elem1, &elem2| {
                diff_mm += elem1.abs_diff(elem2) as u64;
            });
        (diff_vc, diff_mm)
    }

    pub fn merge(&mut self, other: &MMState) {
        assert!(
            self.has_same_config(other),
            "Cannot merge MMStates with different configurations."
        );

        if self.config.split_vc_mm {
            Zip::from(&mut self.vc)
                .and(&other.vc)
                .for_each(|elem_a, elem_b| {
                    *elem_a = std::cmp::max(*elem_a, *elem_b);
                });
        }
        Zip::from(&mut self.mm)
            .and(&other.mm)
            .for_each(|elem_a, elem_b| {
                *elem_a = std::cmp::max(*elem_a, *elem_b);
            });
        self.has_changed.set(true);
    }

    pub fn reset(&mut self) {
        self.vc.fill(0);
        self.mm.fill(0);
        self.vc_sum = 0;
        self.mm_sum = 0;
        self.has_changed.set(true);
    }
}

#[derive(Clone)]
pub struct SimilarityConfig {
    // if absolute, then use the thresholds below directly as u64 counts
    // if not absolute, then use them as fractions of the total counts
    //      in general, dissimilar if min_sum < (1 - threshold) * max_sum
    //      similar if diff <= threshold * max_sum
    //      by generalized triangular inequality, max_sum - min_sum <= pair-wise-abs-diff
    pub mm_use_absolute_similarity_threshold: bool,
    pub mm_event_history_global_similarity_threshold: f64,
    pub mm_event_history_per_event_similarity_threshold: f64,
    pub mm_event_history_global_similarity_threshold_vc: f64,
    pub mm_event_history_global_similarity_threshold_mm: f64,
    pub mm_event_history_per_event_similarity_threshold_vc: f64,
    pub mm_event_history_per_event_similarity_threshold_mm: f64,
}

#[derive(Clone)]
pub struct MMStateRecordConfig {
    pub record_global_mm_state_only_on_reward_reporting: bool,
    pub global_mm_state_record_interval: usize,
    pub record_per_event_mm_state: bool,
    pub per_event_mm_state_record_interval_normal: usize, // for example, record a per_event_mm_state every 100 events
    pub per_event_mm_state_record_interval_special: usize, // if an event is marked "special", record its per_event_mm_state on its every occurrence // HONGYI TODO
    pub special_event_threshold: usize, // if an event occurs less than this threshold, it is considered special // HONGYI TODO
}

impl MMStateRecordConfig {
    pub fn new(
        record_global_mm_state_only_on_reward_reporting: bool,
        global_mm_state_record_interval: usize,
        record_per_event_mm_state: bool,
        per_event_mm_state_record_interval_normal: usize,
        per_event_mm_state_record_interval_special: usize,
        special_event_threshold: usize,
    ) -> Self {
        assert!(
            global_mm_state_record_interval > 0,
            "global_mm_state_record_interval must be greater than 0"
        );
        assert!(
            per_event_mm_state_record_interval_normal > 0,
            "per_event_mm_state_record_interval_normal must be greater than 0"
        );
        assert!(
            per_event_mm_state_record_interval_special > 0,
            "per_event_mm_state_record_interval_special must be greater than 0"
        );
        MMStateRecordConfig {
            record_global_mm_state_only_on_reward_reporting,
            global_mm_state_record_interval,
            record_per_event_mm_state,
            per_event_mm_state_record_interval_normal,
            per_event_mm_state_record_interval_special,
            special_event_threshold: special_event_threshold,
        }
    }
}

pub struct MMRewardConfig {
    pub reward_on_global_state_change: f64,
    pub reward_on_per_event_state_change: f64,
    pub use_cool_down_factor: bool,
    pub cool_down_threshold: usize,
}

pub struct ProcessIdMapper {
    num_procs: usize,
    current_id: usize,
    proc_id_map: HashMap<NodeId, usize>,
}

impl ProcessIdMapper {
    pub fn new(num_procs: usize) -> Self {
        ProcessIdMapper {
            num_procs,
            current_id: 0,
            proc_id_map: HashMap::new(),
        }
    }

    pub fn get_or_assign_id(&mut self, node_id: NodeId) -> usize {
        if let Some(id) = self.proc_id_map.get(&node_id) {
            *id
        } else {
            assert!(
                self.current_id < self.num_procs,
                "Exceeded the maximum number of processes."
            );
            let assigned_id = self.current_id;
            self.proc_id_map.insert(node_id, assigned_id);
            self.current_id += 1;
            assigned_id
        }
    }
}
