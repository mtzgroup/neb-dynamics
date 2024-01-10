"""
This will be a bunch of ugly functions in Javascript and HTML Big time
This module will make the DOM access to the jupyter notebook display.
It will draw molecules using D3
"""

import json
import random
from string import Template

import numpy as np
from IPython.core.display import HTML
from rdkit import Chem

from neb_dynamics.rdkit_draw import moldrawsvg

# this initialization is needed because I want to randomize the name of
# the animation element in the SVG that is appended in the jupyer page.
# Otherwise I get the new draw appended to the last one that is opened.
random.seed()


def get_index_from_dictionary_indexx_list(list_of_dicts, number):
    """
    I need consecutive numbers to make d3 work, but reaction templates can return me graphs with "various numbers"
    so I created this function to remake the edges with the position in list number, not the INDEXX.

    I call the thing INDEXX because INDEX is a special function in d3.

    """
    return [x for x, element in enumerate(list_of_dicts) if element["indexx"] == "{}".format(number)][0]


def molecule_to_d3json(graph_molecule, node_index=False, charges=False, neighbors=False):
    if node_index:  # this is the drawing without indexes
        nodes = [{"indexx": str(i), "atom_name": graph_molecule.nodes[i]["element"].strip(), "labelZ": f'{graph_molecule.nodes[i]["element"].strip()}_{i}'} for i in graph_molecule.nodes()]
    else:
        nodes = [{"indexx": str(i), "atom_name": graph_molecule.nodes[i]["element"].strip(), "labelZ": graph_molecule.nodes[i]["element"].strip()} for i in graph_molecule.nodes()]

    if charges:  # I am appending the charge number to the string.
        for node in nodes:
            node["labelZ"] = f'{node["labelZ"]} {graph_molecule.nodes[int(node["indexx"])]["charge"]}'

    if neighbors:  # I am appending the charge number to the string.
        for node in nodes:
            node["labelZ"] = f'{node["labelZ"]} {graph_molecule.nodes[int(node["indexx"])]["neighbors"]}'

    for node in nodes:
        if node["atom_name"] == "A":
            node["labelZ"] = f'{graph_molecule.nodes[int(node["indexx"])]["element_matching"]}'

    links = [{"source": get_index_from_dictionary_indexx_list(nodes, s), "target": get_index_from_dictionary_indexx_list(nodes, t), "order": d["bond_order"]} for s, t, d in graph_molecule.edges(data=True)]

    return nodes, links


def get_name(x, node_index):
    name = ""
    if "name" in x:
        name = x["name"]
    else:
        if x["type"] == "molecule" and node_index:
            name = x["molecule"][0].smiles
    return name


def get_a_shade_of_grey(x):
    """
    we do this little function because of how digitize works.
    Zero is a special case here.
    From a score (which is a float between 0 and 1), we get one of the rxn shades
    for the d3 graph.
    """
    if x == 0:
        inds = 0
    else:
        bins = np.linspace(0.0, 1.0, 11)
        inds = np.digitize(x, bins)
    return f"rxn{inds}"


def get_type(x, weighted=True):
    label = "mol" if x["type"] == "molecule" else "rxn"
    if label == "mol":
        if "root" in x:
            label = "root"
        if x["grow"]["duplicate"]:
            label = "duplicate"
        if x["grow"]["no_hits"]:
            label = "nohits"
        if x["grow"]["purchasable"]:
            label = "purchasable"
    if weighted and label == "rxn":
        label = get_a_shade_of_grey(x["score"])
    return label


def retrotree_to_d3json(tree, node_index=False, draw_index=True):
    """
    This is how we can do
    """

    if draw_index:
        nodes = [{"indexx": str(i), "atom_name": get_type(tree.nodes[i]), "labelZ": str(i)} for i in tree.nodes()]
    else:
        nodes = [{"indexx": str(i), "atom_name": get_type(tree.nodes[i]), "labelZ": get_name(tree.nodes[i], node_index)} for i in tree.nodes()]

    links = [{"source": get_index_from_dictionary_indexx_list(nodes, s), "target": get_index_from_dictionary_indexx_list(nodes, t), "order": "single"} for s, t, d in tree.edges(data=True)]

    return nodes, links


# forward madness
def get_type_forward(x, i, leaves):
    # print(f'garbonotos {x} {leaves}')
    if "root" in x:
        label = "root"
    elif i in leaves:
        # print(f'{x} is in {leaves}')
        label = "forward"
    else:
        label = "target"
    return label


def forward_to_d3json(tree, leaves):
    """
    This is how we can do
    """
    nodes = [{"indexx": str(i), "atom_name": get_type_forward(tree.nodes[i], i, leaves), "labelZ": str(i)} for i in tree.nodes()]

    links = [{"source": get_index_from_dictionary_indexx_list(nodes, s), "target": get_index_from_dictionary_indexx_list(nodes, t), "order": "single"} for s, t, d in tree.edges(data=True)]

    return nodes, links


#############################


def draw_d3(nodes, links, smile=None, size=None, string_mode=False, percentage=None, force_layout_charge=None, force_field_size=None, draw_index=True):
    """
    string_mode :: Bool <- returns a string if true, a HTML canvas if false.
    """
    width, height = size or (960, 960)
    force_layout_charge = force_layout_charge or 200
    force_field_size = force_field_size or 960

    python_data = json.dumps({"nodes": nodes, "links": links})

    css_text_template = Template(
        """
    .link$crazy_number line {
      stroke: #696969;
    }

    .link$crazy_number line.separator {
      stroke: #fff;
      stroke-width: 3px;
    }

    .link$crazy_number line.separatorT1 {
      stroke: #fff;
      stroke-width: 8px;
    }

    .link$crazy_number line.separatorT2 {
      stroke:#696969;
      stroke-width: 3px;
    }

    .node$crazy_number circle {
      stroke: #000;
      stroke-width: 3px;
    }

    .node$crazy_number text {
      font: bold 15px sans-serif;
      pointer-events: none;
    }
    """
    )

    js_text_template = Template(
        """
    function color(data) {
                  var d = data.split("_")[0];
                  console.log(data);
                  console.log(d);
                  if (d == "C") {return "grey"}
                  else if (d == "H") {return "lightgray"}
                  else if (d == "S") {return "yellow"}
                  else if (d == "N") {return "blue"}
                  else if (d == "Cl") {return "green"}
                  else if (d == "Br") {return "darkred"}
                  else if (d == "F") {return "cyan"}
                  else if (d == "I") {return "indigo"}
                  else if (d == "B") {return "orange"}
                  else if (d == "Rx") {return "Aquamarine"}
                  else if (d == "Li") {return "DarkCyan" }
                  else if (d == "O") {return "red"}
                  else if (d == "Na") {return "Aquamarine"}
                  else if (d == "P") {return "DeepPink"}
                  else if (d == "Al") {return "DarkOrchid"}
                  else if (d == "Pd") {return "FloralWhite"}
                  else if (d == "Ph") {return "AliceBlue"}
                  else if (d[0] == "R") {return "white"}
                  else if (d == "rxn") {return "rgb(255,255,255)"}
                  else if (d == "rxn0") {return "rgb(0.00,0.00,0.00)"}
                  else if (d == "rxn1") {return "rgb(23.18,23.18,23.18)"}
                  else if (d == "rxn2") {return "rgb(46.36,46.36,46.36)"}
                  else if (d == "rxn3") {return "rgb(69.55,69.55,69.55)"}
                  else if (d == "rxn4") {return "rgb(92.73,92.73,92.73)"}
                  else if (d == "rxn5") {return "rgb(115.91,115.91,115.91)"}
                  else if (d == "rxn6") {return "rgb(139.09,139.09,139.09)"}
                  else if (d == "rxn7") {return "rgb(162.27,162.27,162.27)"}
                  else if (d == "rxn8") {return "rgb(185.45,185.45,185.45)"}
                  else if (d == "rxn9") {return "rgb(208.64,208.64,208.64)"}
                  else if (d == "rxn10") {return "rgb(231.82,231.82,231.82)"}
                  else if (d == "rxn11") {return "rgb(255.00,255.00,255.00)"}
                  else if (d == "mol") {return "darkred"}
                  else if (d == "root") {return "orange"}
                  else if (d == "duplicate") {return "CornflowerBlue"}
                  else if (d == "nohits") {return "Aqua"}
                  else if (d == "purchasable") {return "blue"}
                  else if (d == "converged") {return "pink"}
                  else if (d == "stopped") {return "darkgreen"}
                  else if (d == "forward") {return "darkgreen"}
                  else if (d == "target") {return "red"}
                  else if (d == "A") {return "AliceBlue"}
                  else if (d == "L") {return "AliceBlue"}
                  else {return "black"}
                  };
     function radii(data) {
                   var d = data.split("_")[0];
                   if (d == "C") {return 10;}
                   else if (d == "H") {return 5;}
                   else if (d == "B") {return 7;}
                   else if (d == "S") {return 14;}
                   else if (d == "Cl") {return 14;}
                   else if (d == "Br") {return 18;}
                   else if (d == "Li") {return 8;}
                   else if (d == "N") {return 9;}
                   else if (d == "Al") {return 10}
                   else if (d == "I") {return 22;}
                   else if (d == "Rx") {return 12;}
                   else if (d == "F") {return 10;}
                   else if (d == "O") {return 13;}
                   else if (d == "Na") {return 21;}
                   else if (d == "P") {return 18;}
                   else if (d == "Pd") {return 25;}
                   else if (d == "Ph") {return 10;}
                   else if (d[0] == "R") {return 10;}
                   else if (d == "rxn") {return 18}
                   else if (d == "rxn0") {return 10}
                   else if (d == "rxn1") {return 11}
                   else if (d == "rxn2") {return 12}
                   else if (d == "rxn3") {return 13}
                   else if (d == "rxn4") {return 14}
                   else if (d == "rxn5") {return 15}
                   else if (d == "rxn6") {return 16}
                   else if (d == "rxn7") {return 17}
                   else if (d == "rxn8") {return 18}
                   else if (d == "rxn9") {return 19}
                   else if (d == "rxn10") {return 20}
                   else if (d == "rxn11") {return 21}
                   else if (d == "mol") {return 12}
                   else if (d == "root") {return 22}
                   else if (d == "purchasable") {return 12}
                   else if (d == "converged") {return 12}
                   else if (d == "duplicate") {return 12}
                   else if (d == "nohits") {return 12}
                   else if (d == "stopped") {return 12}
                   else if (d == "forward") {return 12}
                   else if (d == "target") {return 14}
                   else if (d == "A") {return 18}
                   else if (d == "L") {return 13}
                   else {return 5;}
                   };

    function order(d) {if (d =="single") {return 1;}
                  else if (d =="")       {return 1;}
                  else if (d == "double") {return 2;}
                  else if (d == "triple") {return 3;}
                  else {return 2}
                  };


    var width = $width,
        height = $height;

    var svg = d3.select("#animation$crazy_number").append("svg")
        .attr({
          "width": "100%",
          "height": "100%"
        })
        .attr("viewBox", "0 0 " + width + " " + height )
        .attr("preserveAspectRatio", "xMidYMid meet")
        .append("g")
        .html('$python_rdkit_svg');

    var force = d3.layout.force()
        .size([width, height])
        .charge(-$force_layout_charge)
        .linkDistance(function(d) { return radii(d.source.atom_name) + radii(d.target.atom_name) + 15; });

    var graph  = $python_data ;
      force
          .nodes(graph.nodes)
          .links(graph.links)
          .on("tick", tick)
          .start();

      var link$crazy_number = svg.selectAll(".link")
          .data(graph.links)
        .enter().append("g")
          .attr("class", "link$crazy_number");

      link$crazy_number.append("line")
          .style("stroke-width", function(d) { return (order(d.order) * 3 - 1) * 2 + "px"; });

      link$crazy_number.filter(function(d) { return d.order == "double"; }).append("line")
          .attr("class", "separator");

      link$crazy_number.filter(function(d) { return d.order == "triple"; })
          .append("line")
          .attr("class", "separatorT1")

      link$crazy_number.filter(function(d) { return d.order == "triple"; })
          .append("line")
          .attr("class", "separatorT2")

      var node$crazy_number = svg.selectAll(".node")
          .data(graph.nodes)
          .enter().append("g")
          .attr("class", "node$crazy_number")
          .call(force.drag);

      node$crazy_number.append("circle")
          .attr("r", function(d) { return radii(d.atom_name); })
          .style("fill", function(d) { return color(d.atom_name); });

      node$crazy_number.append("text")
          .attr("dy", function(d) { return -radii(d.atom_name) -3;})
          .attr("dx", function(d) { return -radii(d.atom_name)/2;})
          .attr("font-size", "10")
          .attr("fill", "red")
          .text(function(d) { return d.labelZ; });

      function tick() {
        link$crazy_number.selectAll("line")
            .attr("x1", function(d) { return d.source.x; })
            .attr("y1", function(d) { return d.source.y; })
            .attr("x2", function(d) { return d.target.x; })
            .attr("y2", function(d) { return d.target.y; });

        node$crazy_number.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
      };

    """
    )

    crazy_number = random.randint(0, 10000000)

    percentage = percentage or 0.35

    html_template = Template(
        """
    <style> $css_text </style>
    <div id="animation{}" style="width: {}%;"></div>
    <script> $js_text </script>
    """.format(
            crazy_number, int(percentage * 100)
        )
    )

    if smile is not None:
        svg_code = moldrawsvg(Chem.MolFromSmiles(smile, fixed_bond_length=30)).replace("\n", " ").replace("'", '"')
    else:
        svg_code = ""

    js_text = js_text_template.substitute(
        {
            "python_data": python_data,
            "crazy_number": crazy_number,
            "width": width,
            "height": height,
            # 'force_field_size': force_field_size,
            "force_layout_charge": force_layout_charge,
            "python_rdkit_svg": svg_code,
        }
    )
    css_text = css_text_template.substitute({"crazy_number": crazy_number})

    if string_mode:
        return html_template.substitute({"css_text": css_text, "js_text": js_text})
    else:
        return HTML(html_template.substitute({"css_text": css_text, "js_text": js_text}))
