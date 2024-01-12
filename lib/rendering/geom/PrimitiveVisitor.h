// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file PrimitiveVisitor.h
/// $Id$
///

#pragma once

namespace moonray {
namespace geom {

class Box;
class Curves;
class Instance;
class Points;
class PolygonMesh;
class Primitive;
class PrimitiveGroup;
class SubdivisionMesh;
class Sphere;
class TransformedPrimitive;
class VdbVolume;

/// @class PrimitiveVisitor
/// @remark For realtime rendering use. Feature film shader development
///     does not require this functionality.
/// @brief Base class for performing operation on Primitives.
///
/// To extend the operation performed on Primitives, extend this base class
/// and supply your own functionality for each type. If you want to apply
/// operation in base class (Primitive) level, just override visitPrimitive
class PrimitiveVisitor
{
public:
    virtual ~PrimitiveVisitor() = default;
    /// override this method to apply an opeartion to the
    /// primitives Procedural stores
    virtual void visitPrimitive(Primitive& p);
    /// override this method to apply an opeartion to the
    /// Curves primitives Procedural stores
    virtual void visitCurves(Curves& c);
    /// override this method to apply an opeartion to the
    /// Instance primitives Procedural stores
    virtual void visitInstance(Instance& i);
    /// override this method to apply an opeartion to the
    /// Points primitives Procedural stores
    virtual void visitPoints(Points& p);
    /// override this method to apply an opeartion to the
    /// PolyMesh primitives Procedural stores
    virtual void visitPolygonMesh(PolygonMesh& p);
    /// override this method to apply an opeartion to the
    /// PrimitiveGroup primitives Procedural stores
    virtual void visitPrimitiveGroup(PrimitiveGroup& s);
    /// override this method to apply an opeartion to the
    /// Sphere primitives Procedural stores
    virtual void visitSphere(Sphere& s);
    /// override this method to apply an opeartion to the
    /// Box primitives Procedural stores
    virtual void visitBox(Box& b);
    /// override this method to apply an opeartion to the
    /// SubdivisionMesh primitives Procedural stores
    virtual void visitSubdivisionMesh(SubdivisionMesh& s);
    /// override this method to apply an opeartion to the
    /// TransformedPrimitive primitives Procedural stores
    virtual void visitTransformedPrimitive(TransformedPrimitive& t);
    /// override this method to apply an operation to the
    /// VdbVolume primitives Procedural stores
    virtual void visitVdbVolume(VdbVolume& v);
};

} // namespace geom
} // namespace moonray


