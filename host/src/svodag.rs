use glam::{uvec3, vec2, vec3, UVec3, Vec3};
use shader::{SerializedSvodagView, SvoIndex};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use std::vec::Vec;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
#[must_use = "Please call Svodag.release to release the reference"]
pub struct SvoNodeRef(pub usize);

impl fmt::Display for SvoNodeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

fn morton_at(
    position: UVec3,
    level: u32, /*level 0 is root, level 1 is first child, etc*/
    max_level: u32,
) -> usize {
    let shift_amount = max_level - level - 1;

    let x = (position.x >> shift_amount) & 1;
    let y = (position.y >> shift_amount) & 1;
    let z = (position.z >> shift_amount) & 1;
    ((x << 2) + (y << 1) + (z)) as usize
}

#[derive(Hash, Eq, PartialEq, Debug, Clone)]
pub struct SvoNode {
    pub child_mask: u8,
    pub leaf: bool,
    pub children: Vec<SvoNodeRef>,
}

impl fmt::Display for SvoNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "leaf: {} child_mask: 0b{:08b}",
            self.leaf, self.child_mask
        )?;

        for i in 0..8 {
            write!(f, "x: {} y: {} z: {} : ", i >> 2, (i >> 1) & 1, i & 1)?;
            if (self.child_mask >> i) & 1 == 1 {
                writeln!(f, "{}", self.children[self.index_of_morton(i)])?;
            } else {
                writeln!(f, "Empty")?;
            }
        }

        Ok(())
    }
}

impl SvoNode {
    pub fn new() -> Self {
        Self {
            child_mask: 0,
            leaf: true,
            children: Vec::new(),
        }
    }

    pub fn index_of(&self, position: UVec3, level: u32, max_level: u32) -> usize {
        let morton = morton_at(position, level, max_level);

        self.index_of_morton(morton)
    }

    pub fn index_of_morton(&self, morton: usize) -> usize {
        ((self.child_mask >> morton).count_ones().max(1) - 1) as usize
    }

    pub fn child_at(&self, position: UVec3, level: u32, max_level: u32) -> Option<&SvoNodeRef> {
        let morton = morton_at(position, level, max_level);
        if (self.child_mask >> morton) & 1 == 1 {
            Some(&self.children[self.index_of_morton(morton)])
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum SvodagQueryResult {
    Some { depth: u32 },
    None { depth: u32 },
}

impl SvodagQueryResult {
    pub fn is_some(&self) -> bool {
        matches!(self, SvodagQueryResult::Some { depth })
    }

    pub fn is_none(&self) -> bool {
        matches!(self, SvodagQueryResult::None { depth })
    }

    pub fn depth(&self) -> u32 {
        match self {
            SvodagQueryResult::None { depth } => *depth,
            SvodagQueryResult::Some { depth } => *depth,
        }
    }
}

impl fmt::Display for SvodagQueryResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug)]
pub struct Svodag {
    pub depth: u32,
    nodes: Vec<SvoNode>,
    reference_counts: RefCell<Vec<u32>>,
    indices: Vec<usize>,
    free_indices: Vec<SvoNodeRef>,
    hash_map: HashMap<SvoNode, SvoNodeRef>,
    counter: usize,
}

impl Svodag {
    pub fn new(depth: u32) -> Self {
        Self {
            depth,
            nodes: Vec::new(),
            reference_counts: RefCell::new(Vec::new()),
            indices: Vec::new(),
            free_indices: Vec::new(),
            hash_map: HashMap::new(),
            counter: 0,
        }
    }

    pub fn to_index(&self, node_ref: &SvoNodeRef) -> usize {
        self.indices[node_ref.0]
    }

    pub fn deref(&self, node_ref: &SvoNodeRef) -> &SvoNode {
        &self.nodes[self.to_index(node_ref)]
    }

    fn deref_mut(&mut self, node_ref: &SvoNodeRef) -> &mut SvoNode {
        let index = self.to_index(node_ref);
        &mut self.nodes[index]
    }

    fn new_ref(&mut self) -> SvoNodeRef {
        let result = self.free_indices.pop().unwrap_or_else(|| {
            self.counter += 1;
            self.indices.push(0);
            SvoNodeRef(self.counter - 1)
        });

        self.indices[result.0] = self.nodes.len();

        result
    }

    fn increase_refcount(&self, node: &SvoNodeRef) {
        self.reference_counts.borrow_mut()[self.to_index(node)] += 1;
    }

    fn acquire(&mut self, node: SvoNode) -> SvoNodeRef {
        let result;
        if let Some(value) = self.hash_map.get(&node) {
            self.increase_refcount(value);
            result = *value;
        } else {
            result = self.new_ref();
            self.hash_map.insert(node.clone(), result);
            self.reference_counts.borrow_mut().push(1);
            self.nodes.push(node);
        }

        result
    }

    pub fn clone_node_ref(&self, node_ref: &SvoNodeRef) -> SvoNodeRef {
        self.increase_refcount(node_ref);

        *node_ref
    }

    pub fn release(&mut self, node_ref: SvoNodeRef) {
        let a_index = self.to_index(&node_ref);
        self.reference_counts.borrow_mut()[a_index] -= 1;

        if self.reference_counts.borrow_mut()[a_index] == 0 {
            self.free_indices.push(node_ref);
            self.hash_map.remove(&self.deref(&node_ref).clone());
            let b_index = self.reference_counts.borrow().len() - 1;
            self.reference_counts.borrow_mut().swap(a_index, b_index);
            self.reference_counts.borrow_mut().pop();
            self.nodes.swap(a_index, b_index);
            let mut node = self.nodes.pop().unwrap();

            if a_index != b_index {
                let b_ref = self.indices.iter().position(|&x| x == b_index).unwrap();
                self.indices[node_ref.0] = usize::MAX;
                self.indices[b_ref] = a_index;
            } else {
                self.indices[node_ref.0] = usize::MAX;
            }

            node.child_mask = 0;
            let children = std::mem::take(&mut node.children);

            for child in children {
                self.release(child);
            }
        }
    }

    pub fn checkout(&self, node_ref: &SvoNodeRef) -> SvoNode {
        let cloned = self.deref(node_ref).clone();

        cloned
    }

    pub fn commit(&mut self, node: SvoNode) -> SvoNodeRef {
        let result;
        if let Some(value) = self.hash_map.get(&node) {
            self.increase_refcount(value);
            result = *value;
        } else {
            result = self.new_ref();
            self.hash_map.insert(node.clone(), result);
            for child in &node.children {
                self.increase_refcount(child);
            }
            self.reference_counts.borrow_mut().push(1);
            self.nodes.push(node);
        }

        result
    }

    fn insert_recursive(
        &mut self,
        node_ref_maybe: Option<&SvoNodeRef>,
        position: UVec3,
        level: u32,
        max_level: u32,
    ) -> SvoNodeRef {
        if let Some(node_ref) = node_ref_maybe {
            if level == max_level {
                self.acquire(SvoNode::new())
            } else {
                let mut node = self.checkout(node_ref);

                node.leaf = false;

                let morton = morton_at(position, level, max_level);
                let new_child = self.insert_recursive(
                    node.child_at(position, level, max_level),
                    position,
                    level + 1,
                    max_level,
                );

                if (node.child_mask >> morton) & 1 == 1 {
                    let index = node.index_of_morton(morton);
                    node.children[index] = new_child;
                } else {
                    node.children
                        .insert((node.child_mask >> morton).count_ones() as usize, new_child);
                    node.child_mask |= 1 << morton;
                }

                if node.child_mask == 0b11111111 && node.children.iter().all(|x| self.deref(x).leaf)
                {
                    node.child_mask = 0;
                    node.children.clear();
                    node.leaf = true;
                }

                let result = self.commit(node);
                self.release(new_child);
                result
            }
        } else {
            if level == max_level {
                self.acquire(SvoNode::new())
            } else {
                let mut node = SvoNode::new();
                node.leaf = false;

                let morton = morton_at(position, level, max_level);
                node.child_mask = 1 << morton;
                node.children
                    .push(self.insert_recursive(None, position, level + 1, max_level));

                self.acquire(node)
            }
        }
    }

    pub fn insert(&mut self, root: Option<&SvoNodeRef>, position: UVec3, level: u32) -> SvoNodeRef {
        self.insert_recursive(root, position, 0, level)
    }

    pub fn insert_mutate(
        &mut self,
        root: Option<SvoNodeRef>,
        position: UVec3,
        level: u32,
    ) -> SvoNodeRef {
        let result = self.insert(root.as_ref(), position, level);
        if let Some(root) = root {
            self.release(root);
        }

        result
    }

    pub fn remove_mutate(
        &mut self,
        root: Option<SvoNodeRef>,
        position: UVec3,
        level: u32,
    ) -> Option<SvoNodeRef> {
        let result = self.remove(root.as_ref(), position, level);
        if let Some(root) = root {
            self.release(root);
        }

        result
    }

    pub fn remove(
        &mut self,
        root: Option<&SvoNodeRef>,
        position: UVec3,
        level: u32,
    ) -> Option<SvoNodeRef> {
        self.remove_recurse(root, position, 0, level)
    }

    fn remove_recurse(
        &mut self,
        node_root_maybe: Option<&SvoNodeRef>,
        position: UVec3,
        level: u32,
        max_level: u32,
    ) -> Option<SvoNodeRef> {
        if let Some(node_ref) = node_root_maybe {
            if level == max_level {
                None
            } else {
                if self.deref(node_ref).leaf {
                    let mut node = SvoNode::new();

                    node.leaf = false;
                    let morton = morton_at(position, level, max_level);

                    let empty_leaf_node = self.acquire(SvoNode::new());
                    for _ in 0..7 {
                        node.children.push(empty_leaf_node);
                    }
                    let index = self.to_index(&empty_leaf_node);
                    self.reference_counts.borrow_mut()[index] += 7;

                    node.child_mask = 0b11111111 - (1 << morton);

                    Some(self.acquire(node))
                } else {
                    let mut node = self.checkout(node_ref);
                    let morton = morton_at(position, level, max_level);
                    let child_maybe = node.child_at(position, level, max_level);

                    if child_maybe.is_some() {
                        let new_child_maybe =
                            self.remove_recurse(child_maybe, position, level + 1, max_level);

                        if let Some(new_child) = new_child_maybe {
                            let index = node.index_of_morton(morton);
                            node.children[index] = new_child;

                            let result = self.commit(node);
                            self.release(new_child);
                            Some(result)
                        } else {
                            // SAFETY: We shouldn't do any memory management on checkout'd node
                            let _ = node.children.remove(node.index_of_morton(morton));
                            node.child_mask ^= 1 << morton;

                            if node.child_mask == 0 {
                                None
                            } else {
                                Some(self.commit(node))
                            }
                        }
                    } else {
                        Some(self.commit(node))
                    }
                }
            }
        } else {
            None
        }
    }

    fn get_recurse(&self, node_ref: &SvoNodeRef, position: UVec3, level: u32) -> SvodagQueryResult {
        let node = self.deref(node_ref);
        if node.leaf {
            SvodagQueryResult::Some { depth: level }
        } else {
            if let Some(child) = node.child_at(position, level, self.depth) {
                self.get_recurse(child, position, level + 1)
            } else {
                SvodagQueryResult::None { depth: level + 1 }
            }
        }
    }

    pub fn get(&self, root: &SvoNodeRef, position: UVec3) -> SvodagQueryResult {
        self.get_recurse(root, position, 0)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn serialize(&self) -> (Vec<u16>, Vec<SvoIndex>) {
        let mut actual_indices: Vec<SvoIndex> = Vec::new();
        let mut result: Vec<u16> = Vec::new();

        for node in &self.nodes {
            actual_indices.push(unsafe { SvoIndex::from_raw(result.len() as u16) });
            result.push((node.child_mask as u16) << 8 | (node.leaf as u16));

            for child in &node.children {
                result.push(self.indices[child.0] as u16);
            }
        }

        let mut index = 0;
        for (i, node) in self.nodes.iter().enumerate() {
            index += 1;
            for _ in &node.children {
                result[index] = actual_indices[result[index as usize] as usize].to_raw();

                index += 1;
            }
        }

        (result, actual_indices)
    }
}

impl fmt::Display for Svodag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "depth: {}", self.depth)?;
        writeln!(f, "indices: {:?}", self.indices)?;
        writeln!(f, "node count: {:?}", self.nodes.len())?;
        writeln!(f, "free indices: {:?}", self.free_indices)?;
        writeln!(f, " nodes: ")?;
        for i in 0..self.indices.len() {
            if self.indices[i] != usize::MAX {
                writeln!(
                    f,
                    "reference: {} refcount: {}\n{}",
                    i,
                    self.reference_counts.borrow()[self.indices[i]],
                    self.nodes[self.indices[i]]
                )?;
            }
        }

        Ok(())
    }
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
    }

    println!("{}", dag.len());

    (dag, root.unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::uvec3;
    use proptest::prelude::*;
    use std::collections::HashSet;

    #[test]
    fn query_result_helpers_report_state_and_depth() {
        let found = SvodagQueryResult::Some { depth: 3 };
        let missing = SvodagQueryResult::None { depth: 2 };

        assert!(found.is_some());
        assert!(!found.is_none());
        assert_eq!(found.depth(), 3);

        assert!(missing.is_none());
        assert!(!missing.is_some());
        assert_eq!(missing.depth(), 2);
    }

    #[test]
    fn node_child_lookup_uses_packed_child_mask_order() {
        let node = SvoNode {
            child_mask: 0b1000_1001,
            leaf: false,
            children: vec![SvoNodeRef(12), SvoNodeRef(11), SvoNodeRef(10)],
        };

        assert_eq!(node.index_of_morton(0), 2);
        assert_eq!(node.index_of_morton(3), 1);
        assert_eq!(node.index_of_morton(7), 0);
        assert_eq!(node.index_of(uvec3(1, 1, 1), 0, 1), 0);

        assert_eq!(node.child_at(uvec3(0, 0, 0), 0, 1), Some(&SvoNodeRef(10)));
        assert_eq!(node.child_at(uvec3(0, 1, 1), 0, 1), Some(&SvoNodeRef(11)));
        assert_eq!(node.child_at(uvec3(1, 1, 1), 0, 1), Some(&SvoNodeRef(12)));
        assert_eq!(node.child_at(uvec3(1, 0, 0), 0, 1), None);
    }

    #[test]
    fn display_impls_include_node_and_reference_details() {
        let node_ref = SvoNodeRef(42);
        assert_eq!(node_ref.to_string(), "42");
        assert_eq!(
            SvodagQueryResult::Some { depth: 4 }.to_string(),
            "Some { depth: 4 }"
        );

        let node = SvoNode {
            child_mask: 0b1000_0001,
            leaf: false,
            children: vec![SvoNodeRef(7), SvoNodeRef(3)],
        };
        let node_display = node.to_string();
        assert!(node_display.contains("leaf: false child_mask: 0b10000001"));
        assert!(node_display.contains("x: 0 y: 0 z: 0 : 3"));
        assert!(node_display.contains("x: 1 y: 1 z: 1 : 7"));
        assert!(node_display.contains("Empty"));

        let depth = 1;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(0, 0, 0), depth);
        let dag_display = dag.to_string();
        assert!(dag_display.contains("depth: 1"));
        assert!(dag_display.contains("node count:"));
        assert!(dag_display.contains("reference:"));

        dag.release(root);
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

        dag.release(root_after_duplicate);
    }

    #[test]
    fn cloned_node_ref_keeps_nodes_alive_until_released() {
        let depth = 2;
        let position = uvec3(1, 2, 3);
        let mut dag = Svodag::new(depth);

        let root = dag.insert_mutate(None, position, depth);
        let cloned = dag.clone_node_ref(&root);
        let initial_len = dag.len();

        dag.release(root);

        assert_eq!(dag.len(), initial_len);
        assert_eq!(
            dag.get(&cloned, position),
            SvodagQueryResult::Some { depth }
        );

        dag.release(cloned);
        assert_eq!(dag.len(), 0);
    }

    #[test]
    fn deref_mut_returns_mutable_node_without_changing_mapping() {
        let depth = 1;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(0, 0, 0), depth);

        let root_index = dag.to_index(&root);
        dag.deref_mut(&root).leaf = dag.deref(&root).leaf;

        assert_eq!(dag.to_index(&root), root_index);

        dag.release(root);
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

        dag.release(root_after_remove);
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

        dag.release(root_after_remove);
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

        dag.release(root_after_remove);
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
        assert!(dag.deref(&full_root).leaf);

        let root_after_remove = dag.remove_mutate(Some(full_root), removed, depth).unwrap();
        let root_node = dag.deref(&root_after_remove);
        let removed_morton = morton_at(removed, 0, depth);

        assert!(!root_node.leaf);
        assert_eq!(root_node.child_mask, 0b1111_1111 ^ (1 << removed_morton));
        assert_eq!(
            dag.get(&root_after_remove, removed),
            SvodagQueryResult::None { depth }
        );
        assert_eq!(
            dag.get(&root_after_remove, uvec3(0, 0, 0)),
            SvodagQueryResult::Some { depth }
        );

        dag.release(root_after_remove);
    }

    #[test]
    fn serialize_preserves_multi_level_child_order() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let mut root = None;

        for position in [uvec3(6, 6, 3), uvec3(3, 7, 0), uvec3(1, 6, 0)] {
            root = Some(dag.insert_mutate(root, position, depth));
        }

        let (serialized, mapping) = dag.serialize();
        let view = SerializedSvodagView::from_slice(&serialized);
        let root = root.unwrap();
        let root_index = mapping[dag.to_index(&root)];

        assert_eq!(view.get_childmask(root_index), 0b0100_0100);

        let morton_6_branch = view.get_child(root_index, 0);
        let morton_2_branch = view.get_child(root_index, 1);

        assert_eq!(view.get_childmask(morton_6_branch), 0b1000_0000);
        assert_eq!(view.get_childmask(morton_2_branch), 0b0100_0100);

        let nested_morton_6_branch = view.get_child(morton_2_branch, 0);
        let nested_morton_2_branch = view.get_child(morton_2_branch, 1);

        assert_eq!(view.get_childmask(nested_morton_6_branch), 0b0100_0000);
        assert_eq!(view.get_childmask(nested_morton_2_branch), 0b0001_0000);

        dag.release(root);
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

        dag.release(root);
    }

    #[test]
    fn example_svodag_is_complex_and_fits_serialized_index_space() {
        let (mut dag, root) = get_example_svodag(7);

        assert!(dag.len() > 1_000);
        let (serialized, _) = dag.serialize();
        assert!(serialized.len() <= u16::MAX as usize);

        dag.release(root);
    }

    #[test]
    fn release_removes_unreferenced_nodes() {
        let depth = 2;
        let mut dag = Svodag::new(depth);

        let root = dag.insert_mutate(None, uvec3(1, 2, 3), depth);
        assert!(dag.len() > 0);

        dag.release(root);

        assert_eq!(dag.len(), 0);
        assert!(dag.indices.iter().all(|index| *index == usize::MAX));
        assert_eq!(dag.reference_counts.borrow().len(), 0);
    }

    #[test]
    fn test_regression_free_last_node() {
        let depth = 2;
        let width = 1 << depth;

        let mut dag = Svodag::new(depth);

        let mut root: Option<SvoNodeRef> = None;
        for i in 0..width {
            for j in 0..width {
                for k in 0..width {
                    if (i | j | k) == i + j + k {
                        root = Some(dag.insert_mutate(root, uvec3(i, j, k), depth));
                        assert!(dag
                            .indices
                            .iter()
                            .all(|i| *i == usize::MAX || *i < dag.nodes.len()));
                    }
                }
            }
        }
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

            // Clean up not strictly needed for test correctness but good for sanity
            dag.release(root);
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

             // Verify points NOT in the set return None (probabilistic check)
            for _ in 0..10 {
                let x = rand::random::<u32>() % size;
                let y = rand::random::<u32>() % size;
                let z = rand::random::<u32>() % size;
                let p = uvec3(x, y, z);
                if !unique_points.contains(&p) {
                    prop_assert!(dag.get(&root_ref, p).is_none(), "Point {:?} should not be present", p);
                }
            }

            dag.release(root_ref);
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
             let root_node = dag.deref(&root_ref);

             // If full, the root node should be a leaf
             prop_assert!(root_node.leaf, "Root should be a leaf when fully filled");
             prop_assert_eq!(root_node.child_mask, 0, "Leaf node should have empty child mask");

             dag.release(root_ref);
        }
    }
}
