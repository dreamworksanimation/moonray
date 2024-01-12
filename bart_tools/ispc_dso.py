# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0


'''
Methods for creating rdl2 dsos from ispc code

The shader writer is expected to author:
   dsoName.json
   dsoName.ispc
   dsoName.cc

From dsoName.json, we produce the intermediate targets
   attributes.cc - expected to be included by dsoName.cc
   attributesISPC.cc - built as dso source
   attributes.isph - expected to be included by dsoName.ispc

and then pass this source to build the rdl dso and proxy dso
'''

import dwa_install

import json
import sys
import os.path
from collections import OrderedDict


def IspcKeyType(rdl2Type):
    '''
    From the rdl2::Type, determine the corresponding ISPC key type.
    Not all rdl2 attributes types are supported in ISPC code.
    '''
    dataType = ''
    if rdl2Type == 'Bool':
        dataType += 'BoolAttrKeyISPC'
    elif rdl2Type == 'Int':
        dataType += 'IntAttrKeyISPC'
    elif rdl2Type == 'Float':
        dataType += 'FloatAttrKeyISPC'
    elif rdl2Type == 'Vec2f':
        dataType += 'Float2AttrKeyISPC'
    elif rdl2Type == 'Vec3f' or rdl2Type == 'Rgb':
        dataType += 'Float3AttrKeyISPC'
    elif rdl2Type == 'Vec4f' or rdl2Type == 'Rgba':
        dataType += 'Float4AttrKeyISPC'
    else:
        dataType += 'UNKNOWN_ATTR_TYPE'
    return dataType

def IspcType(rdl2Type):
    '''
    From the rdl2::Type, determine the corresponding ISPC shader type.
    Not all rdl2 attributes types are supported in ISPC shader code.
    '''
    shaderType = ''
    if rdl2Type == 'Bool':
        shaderType += 'bool'
    elif rdl2Type == 'Int':
        shaderType += 'int'
    elif rdl2Type == 'Float':
        shaderType += 'float'
    elif rdl2Type == 'Vec2f':
        shaderType += 'Vec2f'
    elif rdl2Type == 'Vec3f':
        shaderType += 'Vec3f'
    elif rdl2Type == 'Rgb':
        shaderType = 'Color'
    return shaderType

def cap(name):
    return name[0].upper() + name[1:]

def getAttrFn(name):
    return 'get' + cap(name)

def evalAttrFn(name):
    return 'eval' + cap(name)

def evalAttrBoundFn(name):
    return 'eval' + cap(name) + 'Bound'

def evalAttrUnBoundFn(name):
    return 'eval' + cap(name) + 'UnBound'

def evalCompFn(name):
    return 'evalComp' + cap(name)

def evalNormalFn(name):
    return 'evalNormal' + cap(name)

def evalNormalBoundFn(name):
    return 'evalNormal' + cap(name) + 'Bound'

def evalNormalUnBoundFn(name):
    return 'evalNormal' + cap(name) + 'UnBound'

def defineGetAttrFn(shaderType, retType, attrName):
    text = ('inline uniform ' + retType +
            '\n' + getAttrFn(attrName) + 
            '(const uniform ' + shaderType + ' * uniform obj) {\n' +
            '    return ' + getAttrFn(retType) + '(obj, ' + attrName + ');\n}\n')
    return text

def defineEvalAttrFn(shaderType, retType, attrName):
    # evalAttrName(obj, tls, state);
    text = ('inline varying ' + retType +
            '\n' + evalAttrFn(attrName) +
            '(const uniform ' + shaderType + ' * uniform obj, ' +
            'uniform ShadingTLState *uniform tls, ' +
            'const varying State &state) {\n' +
            '    return ' + evalAttrFn(retType) +
            '(obj, tls, state, ' + attrName + ');\n}\n')

    return text

def defineEvalCompFn(shaderType, compName, attrColor, attrFactor, attrShow):
    # evalCompName
    text = 'inline varying Color\n'
    text +=(evalCompFn(compName) +
            '(const uniform ' + shaderType + ' * uniform obj, ' +
            'uniform ShadingTLState *uniform tls, ' +
            'const varying State &state)\n')
    text += '{\n'
    text += '    Color result = Color_ctor(0.f);\n'
    text += '    if (' + getAttrFn(attrShow) + '(obj)) {\n'
    text += '        const uniform float factor = ' + getAttrFn(attrFactor) + '(obj);\n'
    text += '        if (!isZero(factor)) {\n'
    text += '            result = ' + evalAttrFn(attrColor) + '(obj, tls, state) * factor;\n'
    text += '        }\n'
    text += '    }\n'
    text += '    return result;\n'
    text += '}\n'

    return text

def defineEvalNormalFn(shaderType, normal, attrNameNormal, attrNameBlend, attrNameSpace):
    # evalNormalAttrName
    text = 'inline Vec3f\n'
    text +=(evalNormalFn(normal) + '(const uniform ' + shaderType +
            ' * uniform obj, uniform ShadingTLState *uniform tls, ' + 
            'const varying State &state)\n')
    text += '{\n'
    text +=('    return evalNormal(obj, tls, state, ' + attrNameNormal +
            ', ' + attrNameBlend + ', ' + attrNameSpace + ');\n')
    text += '}\n'

    return text

def mergeJson(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, OrderedDict())
            mergeJson(value, node)
        else:
            destination[key] = value

def declareAttribute(data, text, attr, ns, dataKeywords, aliases, name=''):
    if not name == '':
        text += ('%s::%s = sceneClass.declareAttribute<%s>("%s"' %
                 (ns, attr, data['type'], name))
    else:
        text += ('%s::%s = sceneClass.declareAttribute<%s>("%s"' %
                 (ns, attr, data['type'], data['name']))
    if 'default' in data:
        text += ', %s' % data['default']

    flags = 'FLAGS_NONE'
    if 'flags' in data:
        flags = data['flags']
    interface = 'INTERFACE_GENERIC'
    if 'interface' in data:
        interface = data['interface']
    aliasStr = '{' + ', '.join('"' + x + '"' for x in aliases) + '}'
    text += ', %s, %s, %s' % (flags, interface, aliasStr)
    text += ');\n'

    if 'group' in data:
        text += ('sceneClass.setGroup("%s", %s::%s);\n' %
                 (data['group'], ns, attr))
    if 'enum' in data:
        for enum, value in data['enum'].iteritems():
            text += ('sceneClass.setEnumValue(%s::%s, %s, "%s");\n' %
                     (ns, attr, value, enum))
    if 'metadata' in data:
        for metaKey, metaStr in data['metadata'].iteritems():
            text += ('sceneClass.setMetadata(%s::%s, "%s", "%s");\n' %
                     (ns, attr, metaKey, metaStr))

    # Add rest as metadata, including "comment"
    for metaKey, metaStr in data.iteritems():
        if not metaKey in dataKeywords :
            text += ('sceneClass.setMetadata(%s::%s, "%s", "%s");\n' %
                     (ns, attr, metaKey, metaStr))
    return text

def declareISPCAttributeKey(data, text, attr, ns):
    # if the keytype is unknown, then very likely this attribute is
    # not supported in ispc and exists only for c++ code
    keyType = IspcKeyType(data['type'])
    if keyType != 'UNKNOWN_ATTR_TYPE':
        text += ('namespace ' + ns + ' { extern AttributeKey<%s> %s;' %
                 (data['type'], attr) + ' }\n')
        text += ('%s *%s = (%s *) &%s::%s;\n' %
                 (keyType, attr, keyType, ns, attr))
    return text

def declareISPCAttributeFunctions(data, text, attr, shader):
    # if the keytype is unknown, then very likely this attribute is
    # not supported in ispc and exists only for c++ code
    keyType = IspcKeyType(data['type'])
    if keyType != 'UNKNOWN_ATTR_TYPE':
        text += '//-------------------------------------------------\n'
        text += '// ' + attr + '\n'
        text += '//-------------------------------------------------\n'
        text += 'extern uniform ' + keyType + ' * uniform ' + attr + ';\n'
        ispcType = IspcType(data['type'])
        shaderType = shader['type']
        needsEval = False
        if ispcType == 'Color' or ispcType == 'float' or ispcType == 'Vec2f' or ispcType == 'Vec3f':
            if 'flags' in data and 'FLAGS_BINDABLE' in data['flags']:
                needsEval = True
        text += defineGetAttrFn(shaderType, ispcType, attr)
        if needsEval:
            text += defineEvalAttrFn(shaderType, ispcType, attr)
    return text

def addIncludeDependencies(target, source, env):
    '''
    Add directive includes as target dependencies.
    '''
    shader = json.loads(env.File(source[0]).get_contents(), object_pairs_hook=OrderedDict)
    if 'directives' in shader.keys():
        for attr, data in shader['directives'].iteritems():
            if attr == 'include' and type(data) == list:
                for includeFile in data:
                    includeFile = env.File('#%s' % includeFile)
                    env.Depends(target, includeFile)

def BuildIspcDsoSource(target, source, env):
    '''
    Create attributes.cc (target[0]), attributesISPC.cc (target[1]),
    attributes.isph (target[2]), labels.h (target[3]),
    and labels.isph (target[4]) from dsoName.json (source[0])
    '''

    # load the shader attribute definition from the json file
    shader = json.loads(env.File(source[0]).get_contents(), object_pairs_hook=OrderedDict)

    # Include other json files from a directives 
    # block which looks like the following:
    #{
    #    "directives": {
    #        "include": [
    #            "lib/shaders/dwabase/json/specular.json",
    #            "lib/shaders/dwabase/json/metallic.json"
    #        ]
    #    }
    #}
    if 'directives' in shader.keys():
        for attr, data in shader['directives'].iteritems():
            if attr == 'include' and type(data) == list:
                for includeFile in data:
                    includeFile = env.File('#%s' % includeFile)
                    try:
                        includeJson = json.loads(includeFile.get_contents(), object_pairs_hook=OrderedDict)
                        mergeJson(includeJson, shader)
                    except ValueError as err:
                        print 'Error with file: %s' % includeFile
                        print 'Error: %s' % err

        shader.pop('directives')

    # Process group ordering block which looks like this:
    #{
    #    "groups": {
    #    "order": [
    #        "Specular",
    #        "Diffuse",
    #        "Clearcoat",
    #        "Iridescence",
    #        "Transmission",
    #        "Emission",
    #        "Normal",
    #        "Common"
    #    ]
    #   }
    #}
    if 'groups' in shader.keys():
        for groupsAttr, groupsData in shader['groups'].iteritems():
            if groupsAttr == 'order' and type(groupsData) == list:
                attributesSorted = OrderedDict()
                for group in groupsData:
                    for attr, attrData in shader['attributes'].iteritems():
                        if attrData['group'] == group:
                            attributesSorted[attr] = shader['attributes'].pop(attr)
                for attr, attrData in shader['attributes'].iteritems():
                    attributesSorted[attr] = shader['attributes'].pop(attr)
                shader['attributes'] = attributesSorted
        shader.pop('groups')

    # unlike standard rdl2 dsos, we place the shader attributes into
    # a non-anonymous namespace so we can set the (global) ispc
    # attribute keys to the same names.  at the end of attributes.cc
    # we set 'using namespace ns' which should have a similar effect
    # to using an anonymous namespace.
    ns = shader['name'] + '_attr'

    # create attributes.cc (target[0])
    # included by dsoName.cc
    text  = '#include <scene_rdl2/scene/rdl2/rdl2.h>\n'
    text += 'using namespace scene_rdl2::rdl2;\n'
    text += 'RDL2_DSO_ATTR_DECLARE_NS(' + ns + ')\n'
    for attr, data in shader['attributes'].iteritems():
        if 'multi' in data:
            for i in range(int(data['multi'])):
                multiAttr = attr + str(i)
                text += 'AttributeKey<%s> %s;\n' % (data['type'], multiAttr)
        else:
            text += 'AttributeKey<%s> %s;\n' % (data['type'], attr)
    text += 'RDL2_DSO_ATTR_DEFINE(%s)\n' % (shader['type'])
    
    # Establish set of attribute keywords that have specific meaning.
    # Any data who's key is not in this list will be added as metadata
    # Obviously, if 'metadata' exists it will also be added as metadata
    dataKeywords = ['type', 'name', 'default', 'flags', 'interface', 'aliases',
                    'group', 'enum', 'multi', 'metadata']
    for attr, data in shader['attributes'].iteritems():

        # attributes can have aliases
        aliases = list()
        if 'aliases' in data:
            aliases = list(data['aliases'])

        if 'multi' in data:
            for i in range(int(data['multi'])):
                name = data['name'] + str(i)
                attrName = attr + str(i)
                attrAliases = aliases
                for j in range(len(attrAliases)):
                    attrAliases[j] = attrAliases[j] + str(i)
                text = declareAttribute(data, text, attrName, ns, dataKeywords, attrAliases, name)
            continue;
        else:
            text = declareAttribute(data, text, attr, ns, dataKeywords, aliases)

    if 'labels' in shader:
        text += 'static const char *labels[] = {\n'
        for variable, label in shader['labels'].iteritems():
            text += '    "%s",\n' % label
        text += '    nullptr\n};\n'
        text += 'sceneClass.declareDataPtr("labels", labels);\n'

    if 'interface_flags' in shader:
        text += 'rdl2_dso_interface |= %s;\n' % shader['interface_flags']
            
    text += 'RDL2_DSO_ATTR_END\n'
    text += 'using namespace ' + ns + ';\n'
    f = open(str(target[0]), "wb")
    f.write(text)
    f.close()

    # create attributesISPC.cc (target[1])
    # this file creates the global ispc attribute keys
    # namespace ns { extern AttributeKey<Type> attrName; }
    # TypeAttrKeyISPC *attrName = (TypeAttrKeyISPC *) &ns::attrName;
    text  = '#include <scene_rdl2/scene/rdl2/rdl2.h>\n'
    text += '#include <scene_rdl2/scene/rdl2/ISPCSupport.h>\n'
    text += 'using namespace scene_rdl2::rdl2;\n'
    for attr, data in shader['attributes'].iteritems():
        if 'multi' in data:
            for i in range(int(data['multi'])):
                attrName = attr + str(i)
                text = declareISPCAttributeKey(data, text, attrName, ns)
            continue;
        else:
            text = declareISPCAttributeKey(data, text, attr, ns)
    f = open(str(target[1]), "wb")
    f.write(text)
    f.close()

    # create attributes.isph (target[2])
    # included by dsoName.ispc
    # extern uniform <Type>AttrKeyISPC * uniform attrKey;
    assert str(target[2]).endswith('.isph')
    text = '#pragma once\n'
    text += '#include <moonray/rendering/shading/ispc/Shading.isph>\n'
    text += '#include <scene_rdl2/scene/rdl2/rdl2.isph>\n'
    text += '#include <scene_rdl2/scene/rdl2/ISPCSupport.h>\n'
    for attr, data in shader['attributes'].iteritems():
        if 'multi' in data:
            for i in range(int(data['multi'])):
                attrName = attr + str(i)
                text = declareISPCAttributeFunctions(data, text, attrName, shader)
            continue;
        else:
            text = declareISPCAttributeFunctions(data, text, attr, shader)

    if 'components' in shader:
        for comp, data in shader['components'].iteritems():
            shaderType = shader['type']
            text += '//-------------------------------------------------\n'
            text += '// Component ' + comp + '\n'
            text += '//-------------------------------------------------\n'
            text += defineEvalCompFn(shaderType, comp, data['color'],
                                     data['factor'], data['show'])
    if 'normals' in shader:
        for normal, data in shader['normals'].iteritems():
            text += '//-----------------------------------------------------\n'
            text += '// Normal ' + normal + '\n'
            text += '//-----------------------------------------------------\n'
            shaderType = shader['type']
            text += defineEvalNormalFn(shaderType, normal, data['value'], data['dial'], data['space'])
    f = open(str(target[2]), "wb")
    f.write(text)
    f.close()

    # create labels.h (target[3])
    # optionally included by shaders that assign labels to lobes
    assert str(target[3]).endswith('.h')
    text = '#pragma once\n'
    if 'labels' in shader:
        val = 1
        for variable, label in shader['labels'].iteritems():
            text += 'static const int %s = %d;\n' % (variable, val)
            val = val + 1
    f = open(str(target[3]), 'wb')
    f.write(text)
    f.close()

    # create labels.isph
    # optionally included by shaders that assign labels to lobes
    assert str(target[4]).endswith('.isph')
    text = '#pragma once\n'
    if 'labels' in shader:
        val = 1
        for variable, label in shader['labels'].iteritems():
            text += 'static const uniform int %s = %d;\n' % (variable, val)
            val = val + 1
    f = open(str(target[4]), 'wb')
    f.write(text)
    f.close()

def DWAIspcDso(parentEnv, target, ccSource, ispcSource, jsonSource, **kwargs):
    '''
    return a dso based on source (.cc, .ispc files) and attribute_source (.json)
    several source files are generated from the .json
    '''

    # autogenerate attributes.cc, attributesISPC.cc, attributes.isph,
    # labels.h, and labels.isph
    # from the json source in the directory of the target
    targetDir = os.path.dirname(target.abspath)
    autogen = [targetDir + '/attributes.cc',
               targetDir + '/attributesISPC.cc',
               targetDir + '/attributes.isph',
               targetDir + '/labels.h',
               targetDir + '/labels.isph']
    builder = parentEnv.Command(autogen, jsonSource, BuildIspcDsoSource)
    # If I add the include dependencies to the autogen file they will get
    # re-generated each build.  This way they only re-regenerate when an
    # include json file is changed.
    addIncludeDependencies(builder, jsonSource, parentEnv)
    # Scons is smart enough to know that changes to the program
    # code of the BuildIspcDsoSource() function should trigger a rebuild of
    # the builder's targets, but it is not smart enough to know that
    # changes to the program text of functions that BuildIspcDsoSource()
    # calls should also trigger a change.  For example, adding a
    # print 'foo' to BuildIspcDsoSource() will trigger a rebuild of
    # the targets as expected, but adding a print 'foo' to
    # IspcKeyType() (a function called by BuildIspcDsoSource()) will not.
    # The solution to this problem that I've come up with is to depend the builder
    # on this file itself.  I use BuildIspcDsoSource.func_code.co_filename
    # instead of __file__ because __file__ can be either the .py or compiled
    # .pyc file while the former is always and unambiguously the .py file.
    # The other, less desirable solution would be to remove all the function
    # calls from BuildIspcDsoSource(), but that would substantially reduce
    # the readability of an already fairly complicated function.
    parentEnv.Depends(builder, BuildIspcDsoSource.func_code.co_filename)

    # now build the dso with the correct paths
    env = parentEnv.Clone();

    env.AppendUnique(CPPPATH=[env.Dir('.')])

    sources  = [ccSource]
    sources += [autogen[1]]
    ispc_output, ispc_header = env.IspcShared([ispcSource], **kwargs)
    sources += ispc_output

    dso = env.DWARdl2Dso(target, sources, RDL2_ATTRIBUTES_SOURCE = autogen[0],
                         RPATH_AUTO_ORIGIN_RT_DIR='$INSTALL_DIR/rdl2dso',
                         **kwargs)
    bc = env.SharedIspc2Bc(ispcSource, **kwargs)
    dso.update({'bc' : bc})
    return dso

def DWAInstallIspcDso(env, dsoinfo):
    env.DWAInstallRdl2Dso(dsoinfo)
    if 'bc' in dsoinfo:
        # Install the llvm bitcode file into rdl2dso
        dwa_install.installTargetIntoDir(env, '@install_dso', 'rdl2dso', dsoinfo['bc'])

def generate(env):
    env.AddMethod(DWAIspcDso)
    env.AddMethod(DWAInstallIspcDso)

def exists(env):
    return True

