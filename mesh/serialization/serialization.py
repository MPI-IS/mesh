#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2013 Max Planck Society. All rights reserved.
# Created by Matthew Loper on 2013-02-20.

import re
import os
import sys
import numpy as np

from ..errors import SerializationError

"""
serialization.py


"""

__all__ = ['load_from_obj', 'load_from_obj_cpp', 'write_obj', 'write_mtl',
           'write_json', 'write_three_json',
           'set_landmark_indices_from_ppfile', 'set_landmark_indices_from_lmrkfile',
           'load_from_ply', 'load_from_file']

# import os.path


def load_from_obj(self, filename):
    v = []
    f = []
    ft = []
    fn = []
    vt = []
    vn = []
    vc = []
    segm = dict()
    landm_raw_xyz = dict()
    currSegm = ''
    currLandm = ''
    with open(filename, 'r', buffering=2 ** 10) as fp:
        for line in fp:
            line = line.split()
            if len(line) > 0:
                if line[0] == 'v':
                    v.append([float(x) for x in line[1:4]])
                    if len(line) == 7:
                        vc.append([float(x) for x in line[4:]])
                    if currLandm:
                        landm_raw_xyz[currLandm] = v[-1]
                        currLandm = ''
                elif line[0] == 'vt':
                    vt.append([float(x) for x in line[1:]])
                elif line[0] == 'vn':
                    vn.append([float(x) for x in line[1:]])
                elif line[0] == 'f':
                    faces = [x.split('/') for x in line[1:]]
                    for iV in range(1, len(faces) - 1):  # trivially triangulate faces
                        f.append([int(faces[0][0]), int(faces[iV][0]), int(faces[iV + 1][0])])
                        if (len(faces[0]) > 1) and faces[0][1]:
                            ft.append([int(faces[0][1]), int(faces[iV][1]), int(faces[iV + 1][1])])
                        if (len(faces[0]) > 2) and faces[0][2]:
                            fn.append([int(faces[0][2]), int(faces[iV][2]), int(faces[iV + 1][2])])
                        if currSegm:
                            segm[currSegm].append(len(f) - 1)
                elif line[0] == 'g':
                    currSegm = line[1]
                    if currSegm not in segm.keys():
                        segm[currSegm] = []
                elif line[0] == '#landmark':
                    currLandm = line[1]
                elif line[0] == 'mtllib':
                    self.materials_filepath = os.path.join(os.path.dirname(filename), line[1])
                    self.materials_file = open(self.materials_filepath, 'r').readlines()

    self.v = np.array(v)
    self.f = np.array(f) - 1
    if vt:
        self.vt = np.array(vt)
    if vn:
        self.vn = np.array(vn)
    if vc:
        self.vc = np.array(vc)
    if ft:
        self.ft = np.array(ft) - 1
    if fn:
        self.fn = np.array(fn) - 1
    self.segm = segm
    self.landm_raw_xyz = landm_raw_xyz
    self.recompute_landmark_indices()

    if hasattr(self, 'materials_file'):
        for line in self.materials_file:
            if line and line.split() and line.split()[0] == 'map_Ka':
                self.texture_filepath = os.path.abspath(os.path.join(os.path.dirname(filename), line.split()[1]))


def load_from_obj_cpp(self, filename):
    from .loadobj import loadobj
    if sys.version_info[:2] == (2, 6):
        from OrderedDict import OrderedDict
    else:
        from collections import OrderedDict

    v, vt, vn, f, ft, fn, mtl_path, landm, segm = loadobj(filename)
    if v.size != 0:
        self.v = v
    if f.size != 0:
        self.f = f
    if vn.size != 0:
        self.vn = vn
    if vt.size != 0:
        self.vt = vt
    if fn.size != 0:
        self.fn = fn
    if ft.size != 0:
        self.ft = ft
    if segm:
        self.segm = OrderedDict([(k, v if type(v) is list else v.tolist()) for k, v in segm.items()])
    if mtl_path:
        try:
            self.materials_filepath = os.path.join(os.path.dirname(filename), mtl_path.strip())
            self.materials_file = file(self.materials_filepath, 'r').readlines()
        except:
            self.materials_filepath = None
    if hasattr(self, 'materials_file'):
        for line in self.materials_file:
            if line and line.split() and line.split()[0] == 'map_Ka':
                self.texture_filepath = os.path.abspath(os.path.join(os.path.dirname(filename), line.split()[1]))
    if landm:
        self.landm = landm
        self.recompute_landmark_xyz()


def write_obj(self, filename, flip_faces=False, group=False, comments=None):
    if os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    ff = -1 if flip_faces else 1

    def write_face_to_obj_file(face_index, obj_file):
        vertex_indices = self.f[face_index][::ff] + 1

        if hasattr(self, 'ft'):
            texture_indices = self.ft[face_index][::ff] + 1
            if not hasattr(self, 'fn'):
                self.reset_face_normals()
            normal_indices = self.fn[face_index][::ff] + 1
            obj_file.write('f %d/%d/%d %d/%d/%d  %d/%d/%d\n' % tuple(
                np.array([vertex_indices, texture_indices, normal_indices]).T.flatten()))
        elif hasattr(self, 'fn'):
            normal_indices = self.fn[face_index][::ff] + 1
            obj_file.write('f %d//%d %d//%d  %d//%d\n' % tuple(np.array([vertex_indices, normal_indices]).T.flatten()))
        else:
            obj_file.write('f %d %d %d\n' % tuple(vertex_indices))

    with open(filename, 'w') as fi:
        if comments is not None:
            if isinstance(comments, str):
                comments = [comments]
            for comment in comments:
                for line in comment.split("\n"):
                    fi.write("# %s\n" % line)

        if hasattr(self, 'texture_filepath'):
            outfolder = os.path.dirname(filename)
            outbase = os.path.splitext(os.path.basename(filename))[0]
            mtlpath = outbase + '.mtl'
            fi.write('mtllib %s\n' % mtlpath)
            from shutil import copyfile
            texture_name = outbase + os.path.splitext(self.texture_filepath)[1]
            if os.path.abspath(self.texture_filepath) != os.path.abspath(os.path.join(outfolder, texture_name)):
                copyfile(self.texture_filepath, os.path.join(outfolder, texture_name))
            self.write_mtl(os.path.join(outfolder, mtlpath), outbase, texture_name)

        for r in self.v:
            fi.write('v %f %f %f\n' % (r[0], r[1], r[2]))

        if hasattr(self, 'fn') and hasattr(self, 'vn'):
            for r in self.vn:
                fi.write('vn %f %f %f\n' % (r[0], r[1], r[2]))

        if hasattr(self, 'ft'):
            for r in self.vt:
                if len(r) == 3:
                    fi.write('vt %f %f %f\n' % (r[0], r[1], r[2]))
                else:
                    fi.write('vt %f %f\n' % (r[0], r[1]))
        if hasattr(self, 'segm') and self.segm and not group:
            for p in self.segm.keys():
                fi.write('g %s\n' % p)
                for face_index in self.segm[p]:
                    write_face_to_obj_file(face_index, fi)
        else:
            if hasattr(self, 'f'):
                for face_index in range(len(self.f)):
                    write_face_to_obj_file(face_index, fi)


def write_mtl(self, path, material_name, texture_name):
    """Material attribute file serialization"""
    with open(path, 'w') as f:
        f.write('newmtl %s\n' % material_name)
        # copied from another obj, no idea about what it does
        f.write('ka 0.329412 0.223529 0.027451\n')
        f.write('kd 0.780392 0.568627 0.113725\n')
        f.write('ks 0.992157 0.941176 0.807843\n')
        f.write('illum 0\n')
        f.write('map_Ka %s\n' % texture_name)
        f.write('map_Kd %s\n' % texture_name)
        f.write('map_Ks %s\n' % texture_name)


def write_ply(self, filename, flip_faces=False, ascii=False, little_endian=True, comments=[]):
    from psbody.mesh.serialization import plyutils

    if os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    ff = -1 if flip_faces else 1

    if isinstance(comments, str):
        comments = [comments]
    comments = filter(lambda c: len(c) > 0, sum(map(lambda c: c.split("\n"), comments), []))

    plyutils.write(list([list(x) for x in self.v]),
                   list([list(x[::ff]) for x in self.f] if hasattr(self, 'f') else []),
                   list([list((x * 255).astype(int)) for x in ([] if not hasattr(self, 'vc') else self.vc)]),
                   filename, ascii, little_endian, list(comments),
                   list([list(x) for x in ([] if not hasattr(self, 'vn') else self.vn)]))


def write_three_json(self, filename, name=""):
    import json

    if os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    name = name if name else self.basename
    name = name if name else os.path.splitext(os.path.basename(filename))[0]

    metadata = {"formatVersion": 3.1,
                "sourceFile": "%s.obj" % name,
                "generatedBy": "korper",
                "vertices": len(self.v),
                "faces": len(self.f),
                "normals": len(self.vn),
                "colors": 0,
                "uvs": len(self.vt),
                "materials": 1
                }
    materials = [{"DbgColor": 15658734,
                  "DbgIndex": 0,
                  "DbgName": "defaultMat",
                  "colorAmbient": [0.0, 0.0, 0.0],
                  "colorDiffuse": [0.64, 0.64, 0.64],
                  "colorSpecular": [0.5, 0.5, 0.5],
                  "illumination": 2,
                  "opticalDensity": 1.0,
                  "specularCoef": 96.078431,
                  "transparency": 1.0
                  }]

    mesh_data = {"metadata": metadata,
                 'scale': 0.35,
                 "materials": materials,
                 "morphTargets": [],
                 "morphColors": [],
                 "colors": []}
    mesh_data["vertices"] = self.v.flatten().tolist()
    mesh_data["normals"] = self.vn.flatten().tolist()
    mesh_data["uvs"] = [np.array([[vt[0], vt[1]] for vt in self.vt]).flatten().tolist()]
    mesh_data["faces"] = np.array([[42, self.f[i][0], self.f[i][1], self.f[i][2], 0, self.ft[i][0], self.ft[i][1],
                                    self.ft[i][2], self.fn[i][0], self.fn[i][1], self.fn[i][2]] for i in
                                   range(len(self.f))]).flatten().tolist()

    json_or_js_file = open(filename, 'w')
    json_or_js_file.write(json.dumps(mesh_data, indent=4))
    json_or_js_file.close()


def write_json(self, filename, header="", footer="", name="", include_faces=True, texture_mode=True):
    import json

    if os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    name = name if name else self.basename
    name = name if name else os.path.splitext(os.path.basename(filename))[0]

    if texture_mode:
        vertex_texture_pairs = {}
        for face_index in range(len(self.f)):
            for i in [0, 1, 2]:
                v_index = self.f[face_index][i]
                t_index = self.ft[face_index][i]
                vertex_texture_pairs[(v_index, t_index)] = []
        for face_index in range(len(self.f)):
            for i in [0, 1, 2]:
                v_index = self.f[face_index][i]
                t_index = self.ft[face_index][i]
                vertex_texture_pairs[(v_index, t_index)].append((face_index, i))
        mesh_data = {'name': name,
                     'vertices': [],
                     'textures': []
                     }
        for v_index, t_index, faces_entries in vertex_texture_pairs.items():
            mesh_data['vertices'].append()

            if include_faces:
                mesh_data['faces'] = list([[int(np.asscalar(i)) for i in list(x)] for x in self.f])

    else:
        mesh_data = {'name': name,
                     'vertices': list([list(x) for x in self.v])
                     }
        if include_faces:
            mesh_data['faces'] = list([[int(np.asscalar(i)) for i in list(x)] for x in self.f])

    json_or_js_file = open(filename, 'w')
    if os.path.basename(filename).endswith('js'):
        json_or_js_file.write(header + '\nmesh = ') if header else json_or_js_file.write('var mesh = ')
        json_or_js_file.write(json.dumps(mesh_data, indent=4))
        json_or_js_file.write(footer)
    else:
        json_or_js_file.write(json.dumps(mesh_data, indent=4))
    json_or_js_file.close()


def set_landmark_indices_from_ppfile(self, ppfilename):
    from xml.etree import ElementTree
    tree = ElementTree.parse(ppfilename)

    def get_xyz(e):
        try:
            return [float(e.attrib['x']), float(e.attrib['y']), float(e.attrib['z'])]
        except:  # may happen if landmarks are just spaces
            return [0, 0, 0]

    self.landm_raw_xyz = dict((e.attrib['name'], get_xyz(e)) for e in tree.iter() if e.tag == 'point')
    self.recompute_landmark_indices(ppfilename)


def set_landmark_indices_from_lmrkfile(self, lmrkfilename):
    with open(lmrkfilename, 'r') as lmrkfile:
        self.landm_raw_xyz = {}

        for line in lmrkfile.readlines():
            if not line.strip():
                continue
            command = line.split()[0]
            data = [float(x) for x in line.split()[1:]]

            if command == '_scale':
                selfscale_factor = np.matrix(data)
            elif command == '_translate':
                self.caesar_translation_vector = np.matrix(data)
            elif command == '_rotation':
                self.caesar_rotation_matrix = np.matrix(data).reshape(3, 3)
            else:
                self.landm_raw_xyz[command] = [data[1], data[2], data[0]]
        self.recompute_landmark_indices(lmrkfilename)


def _is_lmrkfile(filename):
    is_lmrk = re.compile('^_scale\s[-\d\.]+\s+_translate(\s[-\d\.]+){3}\s+_rotation(\s[-\d\.]+){9}\s+')
    with open(filename) as f:
        data = f.read()
        res = is_lmrk.match(data)
    return res


def set_landmark_indices_from_any(self, landmarks):
    '''
    Sets landmarks given any of:
     - ppfile
     - ldmk file
     - dict of {name:inds} (i.e. mesh.landm)
     - dict of {name:xyz} (i.e. mesh.landm_xyz)
     - pkl, json, yaml file containing either of the above dicts
    '''
    import json
    import pickle

    try:
        path_exists = os.path.exists(landmarks)
    except:
        path_exists = False
    if path_exists:
        if re.search(".ya{0,1}ml$", landmarks):
            import yaml
            with open(landmarks) as f:
                self.set_landmarks_from_raw(yaml.load(f, Loader=yaml.FullLoader))
        elif re.search(".json$", landmarks):
            with open(landmarks) as f:
                self.set_landmarks_from_raw(json.load(f))
        elif re.search(".pkl$", landmarks):
            with open(landmarks, "rb") as f:
                self.set_landmarks_from_raw(pickle.load(f))
        elif _is_lmrkfile(landmarks):
            self.set_landmark_indices_from_lmrkfile(landmarks)
        else:
            try:
                self.set_landmark_indices_from_ppfile(landmarks)
            except:
                raise Exception("Landmark file %s is of unknown format" % landmarks)
    else:
        self.set_landmarks_from_raw(landmarks)


def load_from_file(self, filename, use_cpp=True):
    if re.search(".ply$", filename):
        self.load_from_ply(filename)
    elif re.search(".obj$", filename):
        # XXX experimental cpp obj loader, if problems, switch back to
        if use_cpp:
            self.load_from_obj_cpp(filename)
        else:
            self.load_from_obj(filename)

    elif re.search(".bsf$", filename):
        self.load_from_bsf(filename)
    else:
        raise NotImplementedError("Unknown mesh file format.")


def load_from_ply(self, filename):
    from os.path import abspath, dirname, join

    test_data_folder = abspath(join(dirname(__file__), '..', 'data', 'unittest'))

    from psbody.mesh.serialization import plyutils
    try:
        res = plyutils.read(filename)
    except plyutils.error as e:
        raise SerializationError(e)

    self.v = np.array(res['pts']).T.copy()
    self.f = np.array(res['tri']).T.copy()

    if 'color' in res:
        self.set_vertex_colors(np.array(res['color']).T.copy() / 255)
    if 'normals' in res:
        self.vn = np.array(res['normals']).T.copy()
