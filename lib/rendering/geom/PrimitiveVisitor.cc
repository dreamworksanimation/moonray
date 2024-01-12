// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PrimitiveVisitor.cc
/// $Id$
///

#include "PrimitiveVisitor.h"

#include <moonray/rendering/geom/Box.h>
#include <moonray/rendering/geom/Curves.h>
#include <moonray/rendering/geom/Instance.h>
#include <moonray/rendering/geom/Points.h>
#include <moonray/rendering/geom/PolygonMesh.h>
#include <moonray/rendering/geom/Primitive.h>
#include <moonray/rendering/geom/PrimitiveGroup.h>
#include <moonray/rendering/geom/Sphere.h>
#include <moonray/rendering/geom/SubdivisionMesh.h>
#include <moonray/rendering/geom/TransformedPrimitive.h>
#include <moonray/rendering/geom/VdbVolume.h>

namespace moonray {
namespace geom {

void
PrimitiveVisitor::visitPrimitive(Primitive&)
{
}

void
PrimitiveVisitor::visitCurves(Curves& c)
{
    visitPrimitive(static_cast<Primitive&>(c));
}

void
PrimitiveVisitor::visitInstance(Instance& i)
{
    visitPrimitive(static_cast<Primitive&>(i));
}

void
PrimitiveVisitor::visitPoints(Points& p)
{
    visitPrimitive(static_cast<Primitive&>(p));
}

void
PrimitiveVisitor::visitPolygonMesh(PolygonMesh& p)
{
    visitPrimitive(static_cast<Primitive&>(p));
}

void
PrimitiveVisitor::visitPrimitiveGroup(PrimitiveGroup& p)
{
    visitPrimitive(static_cast<Primitive&>(p));
}

void
PrimitiveVisitor::visitSphere(Sphere& s)
{
    visitPrimitive(static_cast<Primitive&>(s));
}
void
PrimitiveVisitor::visitBox(Box& b)
{
    visitPrimitive(static_cast<Primitive&>(b));
}

void
PrimitiveVisitor::visitSubdivisionMesh(SubdivisionMesh& s)
{
    visitPrimitive(static_cast<Primitive&>(s));
}

void
PrimitiveVisitor::visitTransformedPrimitive(TransformedPrimitive& t)
{
    visitPrimitive(static_cast<Primitive&>(t));
}

void
PrimitiveVisitor::visitVdbVolume(VdbVolume& v)
{
    visitPrimitive(static_cast<Primitive&>(v));
}

} // namespace geom
} // namespace moonray

