pub mod svodag;

use anyhow;
use glam::uvec3;
use svodag::Svodag;

fn main() -> anyhow::Result<()> {
    println!("Hello, world!");

    let mut dag = Svodag::new(2);

    println!("{}", dag);
    let mut root1 = dag.insert_mutate(None, uvec3(0, 0, 0), 2);
    println!("{} {}", dag, root1);
    root1 = dag.insert_mutate(Some(root1), uvec3(1, 0, 0), 2);
    root1 = dag.insert_mutate(Some(root1), uvec3(0, 1, 0), 2);
    root1 = dag.insert_mutate(Some(root1), uvec3(0, 0, 1), 2);
    root1 = dag.insert_mutate(Some(root1), uvec3(1, 1, 0), 2);
    root1 = dag.insert_mutate(Some(root1), uvec3(1, 0, 1), 2);
    root1 = dag.insert_mutate(Some(root1), uvec3(0, 1, 1), 2);
    root1 = dag.insert_mutate(Some(root1), uvec3(1, 1, 1), 2);
    println!("{} {root1}", dag);
    root1 = dag.remove_mutate(Some(root1), uvec3(1, 1, 1), 2).unwrap();
    root1 = dag.remove_mutate(Some(root1), uvec3(1, 1, 1), 2).unwrap();
    root1 = dag.remove_mutate(Some(root1), uvec3(0, 1, 1), 2).unwrap();
    println!("{} {root1}", dag);

    Ok(())
}
