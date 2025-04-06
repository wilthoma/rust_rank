use std::fs::File;
use std::io::{BufRead, BufReader};
use graph6_rs::Graph as G6Graph;
use petgraph::graph::UnGraph;

pub fn count_triangles_in_file(file_path: &str) -> std::io::Result<Vec<usize>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut triangle_counts = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        // Parse graph6 line using graph6-rs
        let g6 = match G6Graph::from_g6(line) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Failed to parse graph6 line '{}': {:?}", line, e);
                // triangle_counts.push(0);
                continue;
            }
        };

        // Convert to petgraph::UnGraph
        let n = g6.n;
        let mut graph = UnGraph::<(), ()>::default();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
        let bit_vec = g6.bit_vec;
        for i in 0..n{
            for j in 0..n{
                if i<j && bit_vec[  i*n+j]>0 {
                    graph.add_edge(nodes[i], nodes[j], ());
                }
            }
        }
        // for (u, v) in g6.edges() {
        //     graph.add_edge(nodes[u], nodes[v], ());
        // }

        let count = count_valences(&graph)+ 10000000000000*count_triangles(&graph);
        // let count = count_triangles(&graph);
        triangle_counts.push(count);
    }

    Ok(triangle_counts)
}

fn count_triangles(graph: &UnGraph<(), ()>) -> usize {
    let mut count = 0;

    for u in graph.node_indices() {
        let neighbors: Vec<_> = graph.neighbors(u).collect();
        for i in 0..neighbors.len() {
            for j in i + 1..neighbors.len() {
                if graph.contains_edge(neighbors[i], neighbors[j]) {
                    count += 1;
                }
            }
        }
    }

    count / 3 // Each triangle is counted three times (once per vertex)
}


fn count_valences(graph: &UnGraph<(), ()>) -> usize {
    let mut v : Vec<usize> = Vec::new(); 
    let mut ret:usize = 0;

    for u in graph.node_indices() {
        let neighbors: Vec<_> = graph.neighbors(u).collect();
        let n_neighbors = neighbors.len();
        v.push( n_neighbors);
    }
    v.sort_by(|a, b| b.cmp(a));
    for i in 0..v.len() {
        ret = (v[i]-2) + 10 * ret;
    }
    ret
}
