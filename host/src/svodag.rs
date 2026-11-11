use glam::{uvec3, vec2, vec3, UVec3, Vec3};
use shader::svodag::PackedNode;
use shader::{SvoIndex, SvodagView};
use std::collections::HashMap;

pub use shader::svodag::SvodagQueryResult;
pub type SvoNodeRef = SvoIndex;

/// Owns the same packed words read by the shader. Roots are explicit: callers
/// must pass every root they intend to keep to `collect`.
#[derive(Debug)]
pub struct Svodag {
    pub depth: u32,
    data: Vec<u32>,
    live: Vec<bool>,
    free: Vec<usize>,
    interned: HashMap<PackedNode, SvoIndex>,
    dirty: Vec<bool>,
}

impl Svodag {
    fn intern(&mut self, node: PackedNode) -> SvoIndex {
        if let Some(&index) = self.interned.get(&node) {
            return index;
        }
        let slot = self.free.last().copied().unwrap_or(self.live.len());
        let offset = slot
            .checked_mul(9)
            .filter(|&n| n.checked_add(8).is_some_and(|end| end <= u32::MAX as usize))
            .expect("SVoDAG exceeds u32 address space");
        if self.free.last().is_some() {
            self.free.pop();
            self.live[slot] = true;
            self.dirty[slot] = true;
        } else {
            self.data.extend_from_slice(&[0; 9]);
            self.live.push(true);
            self.dirty.push(true);
        }
        self.data[offset..offset + 9].copy_from_slice(&node.words);
        let index = unsafe { SvoIndex::from_raw(offset as u32) };
        self.interned.insert(node, index);
        index
    }

    /// Returns a canonical empty root suitable for rendering an empty scene.
    pub fn empty_root(&mut self) -> SvoNodeRef {
        self.intern(PackedNode::branch())
    }
    fn edit(
        &mut self,
        root: Option<SvoIndex>,
        position: UVec3,
        depth: u32,
        occupied: bool,
    ) -> Option<SvoIndex> {
        let mut path = [None; 32];
        let mut current = root;
        let mut level = 0;
        while level < depth {
            path[level as usize] = current;
            current = match current {
                Some(index) if self.view().is_leaf(index) => {
                    if occupied {
                        return root;
                    }
                    Some(index)
                }
                Some(index) => self
                    .view()
                    .node(index)
                    .child(morton(position, level, depth)),
                None => {
                    if !occupied {
                        return root;
                    }
                    None
                }
            };
            level += 1;
        }
        let present = current.is_some_and(|index| self.view().is_leaf(index));
        if occupied == present {
            return root;
        }
        let mut result = if occupied {
            Some(self.intern(PackedNode::leaf()))
        } else {
            None
        };
        while level > 0 {
            level -= 1;
            let m = morton(position, level, depth);
            let mut node = match path[level as usize] {
                Some(index) if self.view().is_leaf(index) => {
                    let mut node = PackedNode::branch();
                    for i in 0..8 {
                        node.set_child(i, Some(index));
                    }
                    node
                }
                Some(index) => self.view().node(index),
                None => PackedNode::branch(),
            };
            node.set_child(m, result);
            if node.mask() == 0 {
                result = None;
            } else if node.mask() == 255
                && (0..8).all(|i| self.view().is_leaf(node.child(i).unwrap()))
            {
                result = Some(self.intern(PackedNode::leaf()));
            } else {
                result = Some(self.intern(node));
            }
        }
        result
    }

    pub fn new(depth: u32) -> Self {
        assert!(depth <= 32);
        Self {
            depth,
            data: Vec::new(),
            live: Vec::new(),
            free: Vec::new(),
            interned: HashMap::new(),
            dirty: Vec::new(),
        }
    }
    pub fn view(&self) -> SvodagView<'_> {
        SvodagView::from_slice(&self.data)
    }
    pub fn data(&self) -> &[u32] {
        &self.data
    }
    /// Sorted, coalesced ranges of modified packed words since the last clear.
    pub fn dirty_ranges(&self) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::new();
        let mut start = None;
        for (slot, &dirty) in self.dirty.iter().enumerate() {
            if dirty {
                start.get_or_insert(slot * 9);
            } else if let Some(begin) = start.take() {
                ranges.push(begin..slot * 9);
            }
        }
        if let Some(begin) = start {
            ranges.push(begin..self.data.len());
        }
        ranges
    }

    /// Call only after all dirty ranges have been successfully uploaded.
    pub fn clear_dirty(&mut self) {
        self.dirty.fill(false);
    }
    pub fn len(&self) -> usize {
        self.interned.len()
    }
    pub fn is_empty(&self) -> bool {
        self.interned.is_empty()
    }
    pub fn get(&self, root: &SvoNodeRef, position: UVec3) -> SvodagQueryResult {
        self.view().get(*root, position, self.depth)
    }
    pub fn insert(&mut self, root: Option<&SvoNodeRef>, position: UVec3, depth: u32) -> SvoNodeRef {
        assert_eq!(depth, self.depth);
        self.edit(root.copied(), position, depth, true).unwrap()
    }
    pub fn insert_mutate(
        &mut self,
        root: Option<SvoNodeRef>,
        position: UVec3,
        depth: u32,
    ) -> SvoNodeRef {
        self.insert(root.as_ref(), position, depth)
    }
    pub fn remove(
        &mut self,
        root: Option<&SvoNodeRef>,
        position: UVec3,
        depth: u32,
    ) -> Option<SvoNodeRef> {
        assert_eq!(depth, self.depth);
        self.edit(root.copied(), position, depth, false)
    }
    pub fn remove_mutate(
        &mut self,
        root: Option<SvoNodeRef>,
        position: UVec3,
        depth: u32,
    ) -> Option<SvoNodeRef> {
        self.remove(root.as_ref(), position, depth)
    }

    /// Trace all supplied roots, reclaim unreachable slots, and remove their
    /// entries from the intern table before any reclaimed slot can be reused.
    pub fn collect(&mut self, roots: &[SvoNodeRef]) {
        let mut marked = vec![false; self.live.len()];
        let mut stack = roots.to_vec();
        while let Some(index) = stack.pop() {
            let offset = index.to_raw() as usize;
            assert!(
                offset % 9 == 0 && offset / 9 < self.live.len() && self.live[offset / 9],
                "invalid DAG root or child"
            );
            let slot = offset / 9;
            if marked[slot] {
                continue;
            }
            marked[slot] = true;
            let view = self.view();
            if !view.is_leaf(index) {
                for i in 0..view.get_childmask(index).count_ones() {
                    stack.push(view.get_child(index, i));
                }
            }
        }
        for slot in 0..self.live.len() {
            if self.live[slot] && !marked[slot] {
                let index = unsafe { SvoIndex::from_raw((slot * 9) as u32) };
                let node = self.view().node(index);
                self.interned.remove(&node);
                self.live[slot] = false;
                self.free.push(slot);
            }
        }
    }
}

fn morton(position: UVec3, level: u32, depth: u32) -> u8 {
    let shift = depth - level - 1;
    ((((position.x >> shift) & 1) << 2)
        | (((position.y >> shift) & 1) << 1)
        | ((position.z >> shift) & 1)) as u8
}

fn is_example_voxel(position: UVec3, width: u32) -> bool {
    let voxel_size = 2.0 / width as f32;
    let point = (position.as_vec3() + Vec3::splat(0.5)) * voxel_size - Vec3::ONE;
    let shell_thickness = voxel_size * 1.25;

    let sphere = point - vec3(-0.34, -0.02, 0.02);
    let sphere_cage = (sphere.length() - 0.43).abs() <= shell_thickness
        && (sphere.x.abs() <= 0.1 || sphere.y.abs() <= 0.1 || sphere.z.abs() <= 0.1);

    let torus = point - vec3(0.34, -0.02, 0.04);
    let torus_ring = vec2(torus.x, torus.y).length() - 0.3;
    let torus_shell = (vec2(torus_ring, torus.z).length() - 0.1).abs() <= shell_thickness;

    let tile_size = (width / 16).max(1);
    let checker = (position.x / tile_size + position.z / tile_size) % 2 == 0;
    let platform_bounds = point.x.abs() <= 0.92 && point.z.abs() <= 0.72;
    let platform_border = point.x.abs() >= 0.82 || point.z.abs() >= 0.62;
    let platform = point.y >= -0.72
        && point.y < -0.72 + voxel_size
        && platform_bounds
        && (checker || platform_border);

    let left_support = vec2(point.x + 0.34, point.z - 0.02).length() <= 0.045;
    let right_support = vec2(point.x - 0.34, point.z - 0.04).length() <= 0.045;
    let support_height = point.y >= -0.72 && point.y <= -0.43;

    sphere_cage || torus_shell || platform || (support_height && (left_support || right_support))
}

pub fn get_example_svodag(depth: u32) -> (Svodag, SvoNodeRef) {
    let width = 1 << depth;
    let mut dag = Svodag::new(depth);

    let mut root: Option<SvoNodeRef> = None;
    for i in 0..width {
        for j in 0..width {
            for k in 0..width {
                if is_example_voxel(uvec3(i, j, k), width) {
                    root = Some(dag.insert_mutate(root, uvec3(i, j, k), depth));
                }
            }
        }
        if let Some(root) = root {
            dag.collect(&[root]);
        }
    }

    dag.collect(&[root.unwrap()]);

    (dag, root.unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;

    #[test]
    fn host_editor_uses_shared_view() {
        let mut storage = Svodag::new(2);
        let mut root = None;
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    root = Some(storage.insert(root.as_ref(), uvec3(x, y, z), 2));
                }
            }
        }
        let full = root.unwrap();
        assert!(storage.view().is_leaf(full));
        let hole = uvec3(1, 2, 3);
        let partial = storage.remove(root.as_ref(), hole, 2).unwrap();
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    let p = uvec3(x, y, z);
                    assert_eq!(storage.view().get(partial, p, 2).is_some(), p != hole);
                }
            }
        }
        assert_eq!(storage.insert(Some(&partial), hole, 2), full);
    }

    #[test]
    fn dirty_ranges_cover_growth_reuse_and_only_actual_writes() {
        let mut dag = Svodag::new(1);
        assert!(dag.dirty_ranges().is_empty());
        let leaf = dag.intern(PackedNode::leaf());
        let mut branches = Vec::new();
        for i in 0..3 {
            let mut node = PackedNode::branch();
            node.set_child(i, Some(leaf));
            branches.push(dag.intern(node));
        }
        assert_eq!(dag.dirty_ranges(), vec![0..36]);
        dag.clear_dirty();
        assert_eq!(dag.intern(PackedNode::leaf()), leaf);
        dag.collect(&[branches[1]]);
        assert!(dag.dirty_ranges().is_empty());
        for i in 3..5 {
            let mut node = PackedNode::branch();
            node.set_child(i, Some(leaf));
            dag.intern(node);
        }
        assert_eq!(dag.data().len(), 36);
        assert_eq!(dag.dirty_ranges(), vec![9..18, 27..36]);
        let mut node = PackedNode::branch();
        node.set_child(5, Some(leaf));
        dag.intern(node);
        assert_eq!(dag.dirty_ranges(), vec![9..18, 27..45]);
        dag.clear_dirty();
        assert!(dag.dirty_ranges().is_empty());
    }

    #[test]
    fn collection_does_not_discard_pending_writes() {
        let mut dag = Svodag::new(1);
        dag.insert(None, UVec3::ZERO, 1);
        let pending = dag.dirty_ranges();
        dag.collect(&[]);
        assert_eq!(dag.dirty_ranges(), pending);
        let replacement = dag.insert(None, uvec3(1, 1, 1), 1);
        assert!(dag.get(&replacement, uvec3(1, 1, 1)).is_some());
        assert_eq!(dag.dirty_ranges(), pending);
    }

    #[test]
    fn empty_root_survives_collection_and_supports_edits() {
        for depth in [0, 2] {
            let mut dag = Svodag::new(depth);
            let empty = dag.empty_root();
            assert_eq!(empty, dag.empty_root());
            assert!(dag.get(&empty, UVec3::ZERO).is_none());
            let filled = dag.insert(Some(&empty), UVec3::ZERO, depth);
            assert!(dag.get(&filled, UVec3::ZERO).is_some());
            assert!(dag.remove(Some(&filled), UVec3::ZERO, depth).is_none());
            dag.collect(&[empty]);
            assert!(dag.get(&empty, UVec3::ZERO).is_none());
            assert_eq!(dag.empty_root(), empty);
        }
    }

    #[test]
    fn deep_hole_in_full_tree_preserves_every_other_voxel_and_recollapses() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let mut root = None;
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    root = Some(dag.insert_mutate(root, uvec3(x, y, z), depth));
                }
            }
        }
        let full = root.unwrap();
        assert!(dag.view().is_leaf(full));
        assert_eq!(dag.insert(Some(&full), uvec3(2, 3, 6), depth), full);
        let hole = uvec3(2, 3, 6);
        let partial = dag.remove(Some(&full), hole, depth).unwrap();
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    let p = uvec3(x, y, z);
                    assert_eq!(dag.get(&partial, p).is_some(), p != hole);
                }
            }
        }
        assert_eq!(dag.insert(Some(&partial), hole, depth), full);
    }

    #[test]
    fn tracing_keeps_all_roots_and_reuses_slots_without_stale_interning() {
        let mut dag = Svodag::new(3);
        let a = uvec3(0, 0, 0);
        let b = uvec3(7, 7, 7);
        let first = dag.insert(None, a, 3);
        let second = dag.insert(Some(&first), b, 3);
        let discarded = dag.insert(Some(&second), uvec3(3, 4, 5), 3);
        let before = dag.len();
        dag.collect(&[first, second]);
        assert!(dag.len() < before);
        assert!(dag.get(&first, a).is_some());
        assert!(dag.get(&first, b).is_none());
        assert!(dag.get(&second, a).is_some());
        assert!(dag.get(&second, b).is_some());
        let capacity = dag.data().len();
        let other = dag.insert(Some(&first), uvec3(4, 3, 2), 3);
        assert_eq!(dag.data().len(), capacity);
        assert!(dag.get(&other, uvec3(4, 3, 2)).is_some());
        let rebuilt = dag.insert(Some(&second), uvec3(3, 4, 5), 3);
        assert!(dag.get(&rebuilt, uvec3(3, 4, 5)).is_some());
        assert_eq!(dag.insert(Some(&second), uvec3(3, 4, 5), 3), rebuilt);
        let _ = discarded;
        dag.collect(&[first, other, rebuilt]);
        assert!(dag.get(&first, a).is_some());
        assert!(dag.get(&other, uvec3(4, 3, 2)).is_some());
        assert!(dag.get(&rebuilt, b).is_some());
        dag.collect(&[]);
        assert!(dag.is_empty());
    }

    #[test]
    fn mixed_edits_and_collection_match_dense_voxels() {
        let mut dag = Svodag::new(2);
        let mut root = None;
        let mut voxels = [false; 64];
        let mut state = 0x8273_1a5du32;
        let mut retained = None;
        let mut retained_voxels = voxels;
        for step in 0..300 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let index = ((state >> 8) % 64) as usize;
            let position = uvec3(
                (index >> 4) as u32,
                ((index >> 2) & 3) as u32,
                (index & 3) as u32,
            );
            let occupied = (state >> 24) & 1 != 0;
            root = if occupied {
                Some(dag.insert(root.as_ref(), position, 2))
            } else {
                dag.remove(root.as_ref(), position, 2)
            };
            voxels[index] = occupied;
            if step == 100 {
                retained = root;
                retained_voxels = voxels;
            }
            if step % 11 == 0 {
                let roots: Vec<_> = [root, retained].into_iter().flatten().collect();
                dag.collect(&roots);
            }
            for i in 0..64 {
                let p = uvec3((i >> 4) as u32, ((i >> 2) & 3) as u32, (i & 3) as u32);
                assert_eq!(
                    root.is_some_and(|r| dag.get(&r, p).is_some()),
                    voxels[i],
                    "step {step}, voxel {i}"
                );
                if let Some(old) = retained {
                    assert_eq!(
                        dag.get(&old, p).is_some(),
                        retained_voxels[i],
                        "retained root at step {step}, voxel {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn identical_subtrees_are_hash_consed() {
        let mut dag = Svodag::new(2);
        let first = dag.insert(None, uvec3(0, 0, 0), 2);
        let root = dag.insert(Some(&first), uvec3(2, 0, 0), 2);
        let view = dag.view();
        assert_eq!(view.node(root).child(0), view.node(root).child(4));
        dag.collect(&[root]);
        assert!(dag.get(&root, uvec3(0, 0, 0)).is_some());
        assert!(dag.get(&root, uvec3(2, 0, 0)).is_some());
    }

    #[test]
    fn zero_depth_edits_work() {
        let mut dag = Svodag::new(0);
        let root = dag.insert(None, UVec3::ZERO, 0);
        assert_eq!(
            dag.get(&root, UVec3::ZERO),
            SvodagQueryResult::Some { depth: 0 }
        );
        assert_eq!(dag.remove(Some(&root), UVec3::ZERO, 0), None);
    }

    #[test]
    fn duplicate_insert_mutate_reuses_existing_nodes() {
        let depth = 3;
        let position = uvec3(5, 2, 7);
        let mut dag = Svodag::new(depth);

        let root = dag.insert_mutate(None, position, depth);
        let initial_len = dag.len();
        let root_after_duplicate = dag.insert_mutate(Some(root), position, depth);

        assert_eq!(root_after_duplicate, root);
        assert_eq!(dag.len(), initial_len);
        assert_eq!(
            dag.get(&root_after_duplicate, position),
            SvodagQueryResult::Some { depth }
        );
    }

    #[test]
    fn removing_absent_point_preserves_existing_contents() {
        let depth = 2;
        let present = uvec3(0, 3, 1);
        let absent = uvec3(3, 0, 2);
        let mut dag = Svodag::new(depth);

        let root = dag.insert_mutate(None, present, depth);
        let root_after_remove = dag.remove_mutate(Some(root), absent, depth).unwrap();

        assert_eq!(
            dag.get(&root_after_remove, present),
            SvodagQueryResult::Some { depth }
        );
        assert!(dag.get(&root_after_remove, absent).is_none());
    }

    #[test]
    fn removing_from_empty_tree_returns_none() {
        let mut dag = Svodag::new(2);

        assert_eq!(dag.remove(None, uvec3(0, 0, 0), 2), None);
    }

    #[test]
    fn removing_one_of_two_branches_preserves_the_other() {
        let depth = 2;
        let removed = uvec3(0, 0, 0);
        let kept = uvec3(3, 3, 3);
        let mut dag = Svodag::new(depth);

        let root = dag.insert_mutate(None, removed, depth);
        let root = dag.insert_mutate(Some(root), kept, depth);
        let root_after_remove = dag.remove_mutate(Some(root), removed, depth).unwrap();

        assert_eq!(
            dag.get(&root_after_remove, kept),
            SvodagQueryResult::Some { depth }
        );
        assert!(dag.get(&root_after_remove, removed).is_none());
    }

    #[test]
    fn removing_inside_collapsed_child_replaces_that_child() {
        let depth = 2;
        let removed = uvec3(1, 1, 1);
        let kept_inside_child = uvec3(0, 0, 0);
        let kept_outside_child = uvec3(3, 3, 3);
        let mut dag = Svodag::new(depth);
        let mut root = None;

        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    root = Some(dag.insert_mutate(root, uvec3(x, y, z), depth));
                }
            }
        }
        let root = dag.insert_mutate(root, kept_outside_child, depth);

        let root_after_remove = dag.remove_mutate(Some(root), removed, depth).unwrap();

        assert!(dag.get(&root_after_remove, removed).is_none());
        assert_eq!(
            dag.get(&root_after_remove, kept_inside_child),
            SvodagQueryResult::Some { depth }
        );
        assert_eq!(
            dag.get(&root_after_remove, kept_outside_child),
            SvodagQueryResult::Some { depth }
        );
    }

    #[test]
    fn removing_from_collapsed_leaf_expands_to_missing_child() {
        let depth = 1;
        let removed = uvec3(1, 0, 1);
        let mut dag = Svodag::new(depth);
        let mut root = None;

        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    root = Some(dag.insert_mutate(root, uvec3(x, y, z), depth));
                }
            }
        }

        let full_root = root.unwrap();
        assert!(dag.view().is_leaf(full_root));

        let root_after_remove = dag.remove_mutate(Some(full_root), removed, depth).unwrap();
        let root_node = dag.view().node(root_after_remove);
        let removed_morton = ((removed.x << 2) | (removed.y << 1) | removed.z) as usize;

        assert!(!root_node.is_leaf());
        assert_eq!(root_node.mask(), 0b1111_1111 ^ (1 << removed_morton));
        assert_eq!(
            dag.get(&root_after_remove, removed),
            SvodagQueryResult::None { depth }
        );
        assert_eq!(
            dag.get(&root_after_remove, uvec3(0, 0, 0)),
            SvodagQueryResult::Some { depth }
        );
    }

    #[test]
    fn serialize_preserves_multi_level_child_order() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let mut root = None;

        for position in [uvec3(6, 6, 3), uvec3(3, 7, 0), uvec3(1, 6, 0)] {
            root = Some(dag.insert_mutate(root, position, depth));
        }

        let view = dag.view();
        let root = root.unwrap();
        let root_index = root;

        assert_eq!(view.get_childmask(root_index), 0b0100_0100);

        let morton_6_branch = view.get_child(root_index, 0);
        let morton_2_branch = view.get_child(root_index, 1);

        assert_eq!(view.get_childmask(morton_6_branch), 0b1000_0000);
        assert_eq!(view.get_childmask(morton_2_branch), 0b0100_0100);

        let nested_morton_6_branch = view.get_child(morton_2_branch, 0);
        let nested_morton_2_branch = view.get_child(morton_2_branch, 1);

        assert_eq!(view.get_childmask(nested_morton_6_branch), 0b0100_0000);
        assert_eq!(view.get_childmask(nested_morton_2_branch), 0b0001_0000);
    }

    #[test]
    fn example_svodag_contains_each_procedural_feature() {
        let depth = 5;
        let (mut dag, root) = get_example_svodag(depth);

        for position in [uvec3(16, 4, 16), uvec3(3, 15, 15), uvec3(27, 15, 16)] {
            assert!(
                dag.get(&root, position).is_some(),
                "expected occupied voxel at {position}"
            );
        }
        assert!(dag.get(&root, uvec3(31, 31, 31)).is_none());
    }

    #[test]
    fn example_svodag_is_complex_and_serializes_valid_indices() {
        let (mut dag, root) = get_example_svodag(7);

        assert!(dag.len() > 1_000);
        assert!((root.to_raw() as usize) < dag.data().len());
    }

    proptest! {
        #[test]
        fn test_insert_get(
            depth in 1u32..5,
            x in 0u32..100,
            y in 0u32..100,
            z in 0u32..100,
        ) {
            let size = 1 << depth;
            let pos = uvec3(x % size, y % size, z % size);

            let mut dag = Svodag::new(depth);
            let root = dag.insert_mutate(None, pos, depth);

            // Verify get returns the depth (meaning found)
            prop_assert_eq!(dag.get(&root, pos), SvodagQueryResult::Some{depth});


        }

        #[test]
        fn test_insert_remove(
            depth in 1u32..5,
            x in 0u32..100,
            y in 0u32..100,
            z in 0u32..100,
        ) {
            let size = 1 << depth;
            let pos = uvec3(x % size, y % size, z % size);

            let mut dag = Svodag::new(depth);
            let root = dag.insert_mutate(None, pos, depth);

            prop_assert!(dag.get(&root, pos).is_some());

            let root_after = dag.remove_mutate(Some(root), pos, depth);

            // After removing the only node, root should be None
            prop_assert!(root_after.is_none());
        }

        #[test]
        fn test_multiple_insert_consistency(
            depth in 1u32..4,
            points in proptest::collection::vec((any::<u32>(), any::<u32>(), any::<u32>()), 1..20)
        ) {
            let size = 1 << depth;
            let mut unique_points = HashSet::new();
            for (x, y, z) in &points {
                 unique_points.insert(uvec3(x % size, y % size, z % size));
            }

            let mut dag = Svodag::new(depth);
            let mut root = None;

            for &p in &unique_points {
                root = Some(dag.insert_mutate(root, p, depth));
            }

            let root_ref = root.expect("Root should exist");

            for &p in &unique_points {
                prop_assert!(dag.get(&root_ref, p).is_some(), "Point {:?} missing", p);
            }

            for x in 0..size {
                for y in 0..size {
                    for z in 0..size {
                        let p = uvec3(x, y, z);
                        prop_assert_eq!(dag.get(&root_ref, p).is_some(), unique_points.contains(&p));
                    }
                }
            }

        }

        #[test]
        fn test_fill_collapse(
             depth in 1u32..3 // keep small for perf
        ) {
             let size = 1 << depth;
             let mut dag = Svodag::new(depth);
             let mut root = None;

             for x in 0..size {
                 for y in 0..size {
                     for z in 0..size {
                         root = Some(dag.insert_mutate(root, uvec3(x, y, z), depth));
                     }
                 }
             }

             let root_ref = root.unwrap();
             let root_node = dag.view().node(root_ref);

             // If full, the root node should be a leaf
             prop_assert!(root_node.is_leaf(), "Root should be a leaf when fully filled");
             prop_assert_eq!(root_node.mask(), 0, "Leaf node should have empty child mask");

        }
    }
}
