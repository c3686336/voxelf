use spirv_std::glam::UVec3;

use crate::{SvoIndex, SvodagView};

/// Canonical node contents. Children are stored in descending Morton order.
#[derive(Clone, Copy, Debug, Eq, Hash)]
pub struct PackedNode {
    pub words: [u32; 9],
}

impl PartialEq for PackedNode {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..9 {
            if self.words[i] != other.words[i] {
                return false;
            }
        }
        true
    }
}

impl PackedNode {
    pub const fn leaf() -> Self {
        Self {
            words: [1, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    }

    pub const fn branch() -> Self {
        Self { words: [0; 9] }
    }

    pub fn mask(&self) -> u8 {
        (self.words[0] >> 8) as u8
    }
    pub fn is_leaf(&self) -> bool {
        self.words[0] & 1 != 0
    }
    pub fn child(&self, morton: u8) -> Option<SvoIndex> {
        let mask = self.mask();
        if mask & (1 << morton) == 0 {
            return None;
        }
        let index = (mask >> morton).count_ones() as usize;
        Some(unsafe { SvoIndex::from_raw(self.words[index]) })
    }
    pub fn set_child(&mut self, morton: u8, child: Option<SvoIndex>) {
        let old = *self;
        *self = Self::branch();
        let mut mask = old.mask();
        if child.is_some() {
            mask |= 1 << morton;
        } else {
            mask &= !(1 << morton);
        }
        self.words[0] = (mask as u32) << 8;
        let mut offset = 1;
        for i in (0..8).rev() {
            let value = if i == morton { child } else { old.child(i) };
            if let Some(value) = value {
                self.words[offset] = value.to_raw();
                offset += 1;
            }
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
        matches!(self, Self::Some { .. })
    }
    pub fn is_none(&self) -> bool {
        !self.is_some()
    }
    pub fn depth(&self) -> u32 {
        match *self {
            Self::Some { depth } | Self::None { depth } => depth,
        }
    }
}

fn morton(position: UVec3, level: u32, depth: u32) -> u8 {
    let shift = depth - level - 1;
    ((((position.x >> shift) & 1) << 2)
        | (((position.y >> shift) & 1) << 1)
        | ((position.z >> shift) & 1)) as u8
}

impl SvodagView<'_> {
    pub fn node(&self, index: SvoIndex) -> PackedNode {
        let mut node = PackedNode::branch();
        node.words[0] = self.get_data()[index.to_raw() as usize];
        let count = self.get_childmask(index).count_ones() as usize;
        for i in 0..count {
            node.words[i + 1] = self.get_child(index, i as u32).to_raw();
        }
        node
    }

    pub fn get(&self, root: SvoIndex, position: UVec3, depth: u32) -> SvodagQueryResult {
        assert!(depth <= 32);
        let mut current = root;
        let mut level = 0;
        loop {
            if self.is_leaf(current) {
                return SvodagQueryResult::Some { depth: level };
            }
            if level == depth {
                return SvodagQueryResult::None { depth: level };
            }
            match self.node(current).child(morton(position, level, depth)) {
                Some(child) => current = child,
                None => return SvodagQueryResult::None { depth: level + 1 },
            }
            level += 1;
        }
    }
}
