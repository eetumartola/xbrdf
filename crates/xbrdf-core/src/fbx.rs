use fbx::{Node, Property};

pub fn geometry_nodes(node: &Node) -> Vec<&Node> {
    let mut nodes = Vec::new();
    collect_geometry_nodes(node, &mut nodes);
    nodes
}

fn collect_geometry_nodes<'a>(node: &'a Node, nodes: &mut Vec<&'a Node>) {
    if node.name == "Geometry" {
        nodes.push(node);
    }
    for child in &node.children {
        collect_geometry_nodes(child, nodes);
    }
}

pub fn child_f64_array(node: &Node, name: &str) -> Option<Vec<f64>> {
    node.children
        .iter()
        .find(|child| child.name == name)
        .and_then(|child| match child.properties.first()? {
            Property::F64Array(values) => Some(values.clone()),
            Property::F32Array(values) => Some(values.iter().map(|value| *value as f64).collect()),
            _ => None,
        })
}

pub fn child_i32_array(node: &Node, name: &str) -> Option<Vec<i32>> {
    node.children
        .iter()
        .find(|child| child.name == name)
        .and_then(|child| match child.properties.first()? {
            Property::I32Array(values) => Some(values.clone()),
            _ => None,
        })
}

pub fn child_string(node: &Node, name: &str) -> Option<String> {
    node.children
        .iter()
        .find(|child| child.name == name)
        .and_then(|child| match child.properties.first()? {
            Property::String(value) => Some(value.clone()),
            _ => None,
        })
}

pub fn decode_polygon_index(raw: i32) -> Option<(usize, bool)> {
    let end = raw < 0;
    let index = if end {
        raw.checked_neg()?.checked_sub(1)?
    } else {
        raw
    };
    usize::try_from(index).ok().map(|index| (index, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_indices_decode_without_overflow() {
        assert_eq!(decode_polygon_index(3), Some((3, false)));
        assert_eq!(decode_polygon_index(-4), Some((3, true)));
        assert_eq!(decode_polygon_index(i32::MIN), None);
    }
}
