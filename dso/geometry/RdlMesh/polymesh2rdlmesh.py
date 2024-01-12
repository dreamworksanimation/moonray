#!/usr/local/bin/python
# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

import sys
import os

try:
    from arnold import *
except:
    print "* " * 40
    print "You need to be in an Arnold environment to run this... try:" 
    print "\t\trez-env arnold"
    print "* " * 40
    raise

usage = """
*******************************************************************************
polymesh2rdlmesh.py Usage
    
    polymesh2rdlmesh.py <ass_file.ass> <mesh_name_in_ass_file> <output.rdla>
*******************************************************************************    
"""
if len(sys.argv) != 4:
    raise RuntimeError(usage)

ass_file = sys.argv[1]
mesh_name = sys.argv[2]
out_file = sys.argv[3]

def dump_array(rdl_attr, array, type_name, num_per_line, outlines, motion_step=0):
    outlines.append("\t[\"%s\"] = {" % rdl_attr)
    n = 0
    line = "\t\t"
    print rdl_attr, array.contents.nelements
    start = array.contents.nelements * motion_step
    end = array.contents.nelements * (motion_step + 1)
    for i in xrange(start, end):
        if type_name == "uint":
            index = AiArrayGetUInt(array, i)
            line += "%d, " % index
        elif type_name == "pnt":
            index = AiArrayGetPnt(array, i)
            line += "Vec3(%f, %f, %f), " % (index.x, index.y, index.z)
            
        n += 1
        if n % num_per_line == 0:
            outlines.append(line)
            line = "\t\t"
    
    if line != "\t\t":
        outlines.append(line)
    
    outlines.append("\t},")
    
    
def dump_indexed_array(rdl_attr, array, index_array, type_name, num_per_line, outlines):
    outlines.append("\t[\"%s\"] = {" % rdl_attr)
    n = 0
    line = "\t\t"
    print rdl_attr, index_array.contents.nelements
    
    for i in xrange(index_array.contents.nelements):
        index = AiArrayGetUInt(index_array, i)
        if type_name == "pnt2":
            pnt = AiArrayGetPnt2(array, index)
            line += "Vec2(%f, %f), " % (pnt.x, pnt.y)
        elif type_name == "pnt":
            pnt = AiArrayGetPnt(array, index)
            line += "Vec3(%f, %f, %f), " % (pnt.x, pnt.y, pnt.z)
            
        n += 1
        if n % num_per_line == 0:
            outlines.append(line)
            line = "\t\t"
    
    if line != "\t\t":
        outlines.append(line)
    
    outlines.append("\t},")

AiBegin()
AiASSLoad(ass_file)

mesh_node = AiNodeLookUpByName(mesh_name)
if mesh_node is None:
    AiEnd()
    raise RuntimeError(mesh_name + " does not exist in the scene")

if not AiNodeIs(mesh_node, "polymesh"):
    AiEnd()
    raise RuntimeError(mesh_name + " is not a polymesh object")

outlines = []
# declaration
outlines.append("RdlMeshGeometry(\"%s\") {" % mesh_name)

# get vert indices
vidxs = AiNodeGetArray(mesh_node, "vidxs")
dump_array("vertices by index", vidxs, "uint", 10, outlines)

# get sides
nsides = AiNodeGetArray(mesh_node, "nsides")
dump_array("face vertex count", nsides, "uint", 20, outlines)

# get vert list
vlist = AiNodeGetArray(mesh_node, "vlist")
dump_array("vertex list", vlist, "pnt", 3, outlines)
num_verts = vlist.contents.nelements

if vlist.contents.nkeys > 1:
    # using motion blur, grab last key
    dump_array("vertex list mb", vlist, "pnt", 3, outlines, vlist.contents.nkeys - 1)

# get uv list - need to generate these per-vertex
uvidxs = AiNodeGetArray(mesh_node, "uvidxs")
uvlist = AiNodeGetArray(mesh_node, "uvlist")
if uvidxs.contents.nelements > 0:
    dump_indexed_array("uv list", uvlist, uvidxs, "pnt2", 5, outlines)
    
# get the normal list
nidxs = AiNodeGetArray(mesh_node, "nidxs")
nlist = AiNodeGetArray(mesh_node, "nlist")
if nidxs.contents.nelements > 0:
    dump_indexed_array("normal list", nlist, nidxs, "pnt", 3, outlines)
    
# TODO creases once Moonray supports them

# set other attributes
subd_type = AiNodeGetStr(mesh_node, "subdiv_type")
if subd_type == "None":
    outlines.append("\t[\"is subd\"] = false,")
else:
    outlines.append("\t[\"is subd\"] = true,")
    if subd_type == "catclark":
        outlines.append("\t[\"subd scheme\"] = 1,")
    else:
        outlines.append("\t[\"subd scheme\"] = 0,")
        
subd_itr = AiNodeGetByte(mesh_node, "subdiv_iterations")
outlines.append("\t[\"subd resolution\"] = %d," % pow(2, subd_itr))

outlines.append("}") # end of RdlMeshGeometry object

AiEnd()

with open(out_file, 'w') as f:
    f.write("\n".join(outlines))

