use glam::UVec3;
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
    child_mask: u8,
    leaf: bool,
    children: Vec<SvoNodeRef>,
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

#[derive(Debug)]
pub struct Svodag {
    pub depth: u32,
    nodes: Vec<SvoNode>,
    reference_counts: Vec<u32>,
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
            reference_counts: Vec::new(),
            indices: Vec::new(),
            free_indices: Vec::new(),
            hash_map: HashMap::new(),
            counter: 0,
        }
    }

    fn to_index(&self, node_ref: &SvoNodeRef) -> usize {
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

    fn acquire(&mut self, node: SvoNode) -> SvoNodeRef {
        let result;
        if let Some(value) = self.hash_map.get(&node) {
            let index = self.to_index(value);
            self.reference_counts[index] += 1;
            result = *value;
        } else {
            result = self.new_ref();
            self.hash_map.insert(node.clone(), result);
            self.reference_counts.push(1);
            self.nodes.push(node);
        }

        result
    }

    pub fn clone_node_ref(&mut self, node_ref: &SvoNodeRef) -> SvoNodeRef {
        let index = self.to_index(node_ref);
        self.reference_counts[index] += 1;

        *node_ref
    }

    pub fn release(&mut self, node_ref: SvoNodeRef) {
        let a_index = self.to_index(&node_ref);
        self.reference_counts[a_index] -= 1;

        if self.reference_counts[a_index] == 0 {
            self.free_indices.push(node_ref);
            self.hash_map.remove(&self.deref(&node_ref).clone());
            let b_index = self.reference_counts.len() - 1;
            self.reference_counts.swap(a_index, b_index);
            self.reference_counts.pop();
            self.nodes.swap(a_index, b_index);
            let mut node = self.nodes.pop().unwrap();

            let b_ref = self.indices.iter().position(|&x| x == b_index).unwrap();
            self.indices[node_ref.0] = usize::MAX;
            self.indices[b_ref] = a_index;

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
            let index = self.to_index(value);
            self.reference_counts[index] += 1;
            result = *value;
        } else {
            result = self.new_ref();
            self.hash_map.insert(node.clone(), result);
            for child in &node.children {
                let index = self.to_index(child);
                self.reference_counts[index] += 1;
            }
            self.reference_counts.push(1);
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
                    self.reference_counts[index] += 7;

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

    fn get_recurse(&self, node_ref: &SvoNodeRef, position: UVec3, level: u32) -> Option<u32> {
        let node = self.deref(node_ref);
        if node.leaf {
            Some(level)
        } else {
            if let Some(child) = node.child_at(position, level, self.depth) {
                self.get_recurse(child, position, level + 1)
            } else {
                None
            }
        }
    }

    pub fn get(&self, root: &SvoNodeRef, position: UVec3) -> Option<u32> {
        self.get_recurse(root, position, 0)
    }
}

impl fmt::Display for Svodag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "depth: {}", self.depth)?;
        writeln!(f, "{:?}", self.indices)?;
        writeln!(f, "{:?}", self.free_indices)?;
        writeln!(f, " nodes: ")?;
        for i in 0..self.indices.len() {
            if self.indices[i] != usize::MAX {
                writeln!(
                    f,
                    "reference: {} refcount: {}\n{}",
                    i, self.reference_counts[self.indices[i]], self.nodes[self.indices[i]]
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::uvec3;
    use proptest::prelude::*;
    use std::collections::HashSet;

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
            prop_assert_eq!(dag.get(&root, pos), Some(depth));

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
