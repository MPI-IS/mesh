#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2012 Max Planck Society. All rights reserved.
# Created by Matthew Loper on 2012-05-11.

"""
Mesh visualization and related classes
--------------------------------------

This module contains the core visualization tools for meshes.  The
backend used for visualization is OpenGL.

The module itself can be run like the following

..  code::

    python -m psbody.mesh.meshviewer arguments

The following commands are used

* ``arguments=TEST_FOR_OPENGL`` a basic OpenGL support is run.  This
  is usually performed on a forked python process.  In case OpenGL is
  not supported, a `DummyClass`` mesh viewer is returned.

* ``arguments=title nb_x_axis nb_y_axis width height`` a new window is
  created

..  autosummary::

    MeshViewer
    MeshViewers
    MeshViewerLocal
    test_for_opengl
"""

import copy
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
from OpenGL import GL, GLU, GLUT
from OpenGL.arrays.vbo import VBO
from PIL import Image
import zmq

# if this file is processed/run as a python script/standalone, especially from the
# internal command
if __package__ is not None:
    from .arcball import (
        ArcBallT, Matrix3fT, Matrix4fT, Point2fT,
        Matrix3fMulMatrix3f, Matrix3fSetRotationFromQuat4f,
        Matrix4fSetRotationFromMatrix3f)
    from .geometry.tri_normals import TriNormals
    from .fonts import get_textureid_with_text
    from .mesh import Mesh


# this block is below the previous one to make my linter happy
if __package__ is None:
    print("this file cannot be executed as a standalone python module")
    print("python -m psbody.mesh.%s arguments" % (os.path.splitext(os.path.basename(__file__))[0]))
    sys.exit(-1)


# Default transport and host are such that we can listen for incoming
# network connections.
ZMQ_TRANSPORT = "tcp"
ZMQ_HOST = "0.0.0.0"
# The dynamic port range.
ZMQ_PORT_MIN = 49152
ZMQ_PORT_MAX = 65535

MESH_VIEWER_DEFAULT_TITLE = "Mesh Viewer"
MESH_VIEWER_DEFAULT_SHAPE = (1, 1)
MESH_VIEWER_DEFAULT_WIDTH = 1280
MESH_VIEWER_DEFAULT_HEIGHT = 960


def _run_self(args, stdin=None, stdout=None, stderr=None):
    """Executes this same script module with the given arguments (forking without subprocess dependencies)"""
    return subprocess.Popen([sys.executable] +
                            ['-m'] + ['%s.%s' % (__package__, os.path.splitext(os.path.basename(__file__))[0])] +
                            args,
                            stdin=stdin,
                            stdout=stdout,  # if stdout is not None else subprocess.PIPE,
                            stderr=stderr)


def _test_for_opengl():
    try:
        # from OpenGL.GLUT import glutInit
        GLUT.glutInit()
    except Exception as e:
        print(e, file=sys.stderr)
        print('failure')
    else:
        print('success')


test_for_opengl_cached = None


def test_for_opengl():
    """Tests if opengl is supported.

    .. note:: the result of the test is cached

    """

    global test_for_opengl_cached
    if test_for_opengl_cached is None:

        with open(os.devnull) as dev_null, \
            tempfile.TemporaryFile() as out, \
            tempfile.TemporaryFile() as err:

                p = _run_self(["TEST_FOR_OPENGL"],
                              stdin=dev_null,
                              stdout=out,
                              stderr=err)
                p.wait()

                out.seek(0)
                err.seek(0)

                line = ''.join(out.read().decode())
                test_for_opengl_cached = 'success' in line
                if not test_for_opengl_cached:
                    print('OpenGL test failed: ')
                    print('\tstdout:', line)
                    print('\tstderr:', '\n'.join(err.read().decode()))

    return test_for_opengl_cached


class Dummy:

    def __getattr__(self, name):
        return Dummy()

    def __call__(self, *args, **kwargs):
        return Dummy()

    def __getitem__(self, key):
        return Dummy()

    def __setitem__(self, key, value):
        pass


def MeshViewer(titlebar='Mesh Viewer',
               static_meshes=None,
               static_lines=None,
               uid=None,
               autorecenter=True,
               shape=(1, 1),
               keepalive=True,
               window_width=1280,
               window_height=960,
               snapshot_camera=None):
    """Allows visual inspection of geometric primitives.

    Write-only Attributes:

    :param titlebar: string printed in the window titlebar
    :param static_meshes: list of Mesh objects to be displayed
    :param static_lines: list of Lines objects to be displayed

    .. note:: `static_meshes` is meant for Meshes that are updated infrequently,
              `and dynamic_meshes` is for Meshes that are updated frequently
              (same for `dynamic_lines` vs. `static_lines`).
              They may be treated differently for performance reasons.

    """

    if not test_for_opengl():
        return Dummy()

    mv = MeshViewerLocal(shape=(1, 1),
                         uid=uid,
                         titlebar=titlebar,
                         keepalive=keepalive,
                         window_width=window_width,
                         window_height=window_height)
    result = mv.get_subwindows()[0][0]
    result.snapshot_camera = snapshot_camera
    if static_meshes:
        result.static_meshes = static_meshes
    if static_lines:
        result.static_lines = static_lines
    result.autorecenter = autorecenter

    return result


def MeshViewers(shape=(1, 1),
                titlebar="Mesh Viewers",
                keepalive=True,
                window_width=1280,
                window_height=960):
    """Allows subplot-style inspection of primitives in multiple subwindows.

    :param shape: a tuple indicating the number of vertical and horizontal windows requested
    :param titlebar: the title appearing on the created window


    Returns: a list of lists of MeshViewer objects: one per window requested.
    """

    if not test_for_opengl():
        return Dummy()

    mv = MeshViewerLocal(shape=shape,
                         titlebar=titlebar,
                         uid=None,
                         keepalive=keepalive,
                         window_width=window_width,
                         window_height=window_height)
    return mv.get_subwindows()


class MeshSubwindow:

    def __init__(self, parent_window, which_window):
        self.parent_window = parent_window
        self.which_window = which_window

    def set_dynamic_meshes(self, list_of_meshes, blocking=False):
        self.parent_window.set_dynamic_meshes(list_of_meshes, blocking, self.which_window)

    def set_static_meshes(self, list_of_meshes, blocking=False):
        self.parent_window.set_static_meshes(list_of_meshes, blocking, self.which_window)

    # list_of_model_names_and_parameters should be of form [{'name': scape_model_name, 'parameters': scape_model_parameters}]
    # here scape_model_name is the filepath of the scape model.
    def set_dynamic_models(self, list_of_model_names_and_parameters, blocking=False):
        self.parent_window.set_dynamic_models(list_of_model_names_and_parameters, blocking, self.which_window)

    def set_dynamic_lines(self, list_of_lines, blocking=False):
        self.parent_window.set_dynamic_lines(list_of_lines, blocking, self.which_window)

    def set_static_lines(self, list_of_lines, blocking=False):
        self.parent_window.set_static_lines(list_of_lines, blocking=blocking, which_window=self.which_window)

    def set_titlebar(self, titlebar, blocking=False):
        self.parent_window.set_titlebar(titlebar, blocking, which_window=self.which_window)

    def set_lighting_on(self, lighting_on, blocking=True):
        self.parent_window.set_lighting_on(lighting_on, blocking=blocking, which_window=self.which_window)

    def set_autorecenter(self, autorecenter, blocking=False):
        self.parent_window.set_autorecenter(autorecenter, blocking=blocking, which_window=self.which_window)

    def set_background_color(self, background_color, blocking=False):
        self.parent_window.set_background_color(background_color, blocking=blocking, which_window=self.which_window)

    def save_snapshot(self, path, blocking=False):
        self.parent_window.save_snapshot(
            path, blocking=blocking, which_window=self.which_window)

    def get_event(self):
        return self.parent_window.get_event()

    def get_keypress(self):
        return self.parent_window.get_keypress()['key']

    def get_mouseclick(self):
        return self.parent_window.get_mouseclick()

    def close(self):
        self.parent_window.p.terminate()

    background_color = property(fset=set_background_color, doc="Background color, as 3-element numpy array where 0 <= color <= 1.0.")
    dynamic_meshes = property(fset=set_dynamic_meshes, doc="List of meshes for dynamic display.")
    static_meshes = property(fset=set_static_meshes, doc="List of meshes for static display.")
    dynamic_models = property(fset=set_dynamic_models, doc="List of model names and parameters for dynamic display.")
    dynamic_lines = property(fset=set_dynamic_lines, doc="List of Lines for dynamic display.")
    static_lines = property(fset=set_static_lines, doc="List of Lines for static display.")
    titlebar = property(fset=set_titlebar, doc="Titlebar string.")
    lighting_on = property(fset=set_lighting_on, doc="Titlebar string.")


class MeshViewerSingle:

    def __init__(self, x1_pct, y1_pct, width_pct, height_pct):
        assert(width_pct <= 1)
        assert(height_pct <= 1)
        self.dynamic_meshes = []
        self.static_meshes = []
        self.dynamic_models = []
        self.dynamic_lines = []
        self.static_lines = []
        self.lighting_on = True
        self.scape_models = {}
        self.x1_pct = x1_pct
        self.y1_pct = y1_pct
        self.width_pct = width_pct
        self.height_pct = height_pct
        self.autorecenter = True

    def get_dimensions(self):
        d = {}
        d['window_width'] = GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH)
        d['window_height'] = GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)
        d['subwindow_width'] = self.width_pct * d['window_width']
        d['subwindow_height'] = self.height_pct * d['window_height']
        d['subwindow_origin_x'] = self.x1_pct * d['window_width']
        d['subwindow_origin_y'] = self.y1_pct * d['window_height']
        return d

    def on_draw(self, transform, want_camera=False):

        d = self.get_dimensions()

        GL.glViewport(
            int(d['subwindow_origin_x']),
            int(d['subwindow_origin_y']),
            int(d['subwindow_width']),
            int(d['subwindow_height']))

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()

        fov_degrees = 45.
        near = 1.0
        far = 100.
        ratio = float(d['subwindow_width']) / float(d['subwindow_height'])
        if d['subwindow_width'] < d['subwindow_height']:
            xt = np.tan(fov_degrees * np.pi / 180. / 2.0) * near
            yt = xt / ratio
            GL.glFrustum(-xt, xt, -yt, yt, near, far)
        else:
            GLU.gluPerspective(fov_degrees, ratio, near, far)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, GL.GL_TRUE)

        GL.glTranslatef(0.0, 0.0, -6.0)
#        GL.glTranslatef(0.0,0.0,-3.5)

        GL.glPushMatrix()
        GL.glMultMatrixf(transform)
        GL.glColor3f(1.0, 0.75, 0.75)

        if self.autorecenter:
            camera = self.draw_primitives_recentered(want_camera=want_camera)
        else:
            if hasattr(self, 'current_center') and hasattr(self, 'current_scalefactor'):
                camera = self.draw_primitives(scalefactor=self.current_scalefactor, center=self.current_center)
            else:
                camera = self.draw_primitives(want_camera=want_camera)

        GL.glPopMatrix()

        if want_camera:
            return camera

    def draw_primitives_recentered(self, want_camera=False):
        return self.draw_primitives(recenter=True, want_camera=want_camera)

    @staticmethod
    def set_shaders(m):
        VERTEX_SHADER = GL.shaders.compileShader("""void main() {
                    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                }""", GL.GL_VERTEX_SHADER)
        FRAGMENT_SHADER = GL.shaders.compileShader("""void main() {
                    gl_FragColor = vec4( 0, 1, 0, 1 );
                }""", GL.GL_FRAGMENT_SHADER)
        m.shaders = GL.shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)

    @staticmethod
    def set_texture(m):
        texture_data = np.array(m.texture_image, dtype='int8')
        m.textureID = GL.glGenTextures(1)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, m.textureID)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, texture_data.shape[1], texture_data.shape[0], 0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE, texture_data.flatten())
        GL.glHint(GL.GL_GENERATE_MIPMAP_HINT, GL.GL_NICEST)  # must be GL_FASTEST, GL.GL_NICEST or GL_DONT_CARE
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

    @staticmethod
    def draw_mesh(m, lighting_on):

        # Supply vertices
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        m.vbo['v'].bind()
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, m.vbo['v'])
        m.vbo['v'].unbind()

        # Supply normals
        if 'vn' in m.vbo.keys():
            GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
            m.vbo['vn'].bind()
            GL.glNormalPointer(GL.GL_FLOAT, 0, m.vbo['vn'])
            m.vbo['vn'].unbind()
        else:
            GL.glDisableClientState(GL.GL_NORMAL_ARRAY)

        # Supply colors
        if 'vc' in m.vbo.keys():
            GL.glEnableClientState(GL.GL_COLOR_ARRAY)
            m.vbo['vc'].bind()
            GL.glColorPointer(3, GL.GL_FLOAT, 0, m.vbo['vc'])
            m.vbo['vc'].unbind()
        else:
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)

        if ('vt' in m.vbo.keys()) and hasattr(m, 'textureID'):
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_MODULATE)
            GL.glBindTexture(GL.GL_TEXTURE_2D, m.textureID)

            GL.glEnableClientState(GL.GL_TEXTURE_COORD_ARRAY)
            m.vbo['vt'].bind()
            GL.glTexCoordPointer(2, GL.GL_FLOAT, 0, m.vbo['vt'])
            m.vbo['vt'].unbind()
        else:
            GL.glDisable(GL.GL_TEXTURE_2D)
            GL.glDisableClientState(GL.GL_TEXTURE_COORD_ARRAY)

        # Draw
        if len(m.f) > 0:
            # ie if it is triangulated
            if lighting_on:
                GL.glEnable(GL.GL_LIGHTING)
            else:
                GL.glDisable(GL.GL_LIGHTING)
            GL.glDrawElementsui(GL.GL_TRIANGLES, np.arange(m.f.size, dtype=np.uint32))
        else:
            # not triangulated, so disable lighting
            GL.glDisable(GL.GL_LIGHTING)
            GL.glPointSize(2)
            GL.glDrawElementsui(GL.GL_POINTS, np.arange(len(m.v), dtype=np.uint32))
        if hasattr(m, 'v_to_text'):

            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_DECAL)

            bgcolor = np.array(GL.glGetDoublev(GL.GL_COLOR_CLEAR_VALUE))
            fgcolor = 1. - bgcolor

            from .lines import Lines
            sc = float(np.max(np.max(m.v, axis=0) - np.min(m.v, axis=0))) / 10.

            cur_mtx = np.linalg.pinv(GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX).T)
            xdir = cur_mtx[:3, 0]
            ydir = cur_mtx[:3, 1]

            GL.glEnable(GL.GL_LINE_SMOOTH)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

            for vidx, text in m.v_to_text.items():
                pos0 = m.v[vidx].copy()
                pos1 = m.v[vidx].copy()
                if hasattr(m, 'vn'):
                    pos1 += m.vn[vidx] * sc
                GL.glLineWidth(5.0)
                ln = Lines(v=np.vstack((pos0, pos1)), e=np.array([[0, 1]]))
                GL.glEnable(GL.GL_LIGHTING)
                GL.glColor3f(1. - 0.8, 1. - 0.8, 1. - 1.00)
                MeshViewerSingle.draw_lines(ln)

                GL.glDisable(GL.GL_LIGHTING)

                texture_id = get_textureid_with_text(text, bgcolor, fgcolor)
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)

                GL.glPushMatrix()
                GL.glTranslatef(pos1[0], pos1[1], pos1[2])

                dx = xdir * .10
                dy = ydir * .10
                if False:
                    GL.glBegin(GL.GL_QUADS)

                    GL.glTexCoord2f(1., 0.)
                    GL.glVertex3f(*(+dx + dy))

                    GL.glTexCoord2f(1., 1.)
                    GL.glVertex3f(*(+dx - dy))

                    GL.glTexCoord2f(0., 1.)
                    GL.glVertex3f(*(-dx - dy))

                    GL.glTexCoord2f(0., 0.)
                    GL.glVertex3f(*(-dx + dy))

                    # gluSphere(quadratic,0.05,32,32)
                    GL.glEnd()
                else:
                    GL.glBegin(GL.GL_POLYGON)

                    for r in np.arange(0, np.pi * 2., .01):
                        GL.glTexCoord2f(np.cos(r) / 2. + .5, np.sin(r) / 2. + .5)
                        GL.glVertex3f(*(dx * np.cos(r) + -dy * np.sin(r)))

                    GL.glEnd()
                GL.glPopMatrix()


    @staticmethod
    def draw_lines(ls):
        GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glLineWidth(3.0)
        allpts = ls.v[ls.e.flatten()].astype(np.float32)
        GL.glVertexPointerf(allpts)
        if hasattr(ls, 'vc') or hasattr(ls, 'ec'):
            GL.glEnableClientState(GL.GL_COLOR_ARRAY)
            if hasattr(ls, 'vc'):
                GL.glColorPointerf(ls.vc[ls.e.flatten()].astype(np.float32))
            else:
                clrs = np.ones((ls.e.shape[0] * 2, 3)) * np.repeat(ls.ec, 2, axis=0)
                GL.glColorPointerf(clrs)
        else:
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)

        GL.glDisable(GL.GL_LIGHTING)
        GL.glDrawElementsui(GL.GL_LINES, np.arange(len(allpts), dtype=np.uint32))

    def draw_primitives(self,
                        scalefactor=1.0,
                        center=[0.0, 0.0, 0.0],
                        recenter=False,
                        want_camera=False):

        # measure the bounding box of all our primitives, so that we can
        # recenter them in our field of view
        if recenter:
            all_meshes = self.static_meshes + self.dynamic_meshes
            all_lines = self.static_lines + self.dynamic_lines

            if (len(all_meshes) + len(all_lines)) == 0:
                if want_camera:
                    return {'modelview_matrix': GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX),
                            'projection_matrix': GL.glGetDoublev(GL.GL_PROJECTION_MATRIX),
                            'viewport': GL.glGetIntegerv(GL.GL_VIEWPORT)
                            }
                else:
                    return None

            for m in all_meshes:
                m.v = m.v.reshape((-1, 3))

            all_verts = np.concatenate(
                [m.v[m.f.flatten() if len(m.f) > 0 else np.arange(len(m.v))] for m in all_meshes] +
                [l.v[l.e.flatten()] for l in all_lines],
                axis=0)

            maximum = np.max(all_verts, axis=0)
            minimum = np.min(all_verts, axis=0)
            center = (maximum + minimum) / 2.
            scalefactor = (maximum - minimum) / 4.
            scalefactor = np.max(scalefactor)
        else:
            center = np.array(center)
#            for mesh in self.dynamic_meshes :
#                if mesh.f : mesh.reset_normals()
            all_meshes = self.static_meshes + self.dynamic_meshes
            all_lines = self.static_lines + self.dynamic_lines
        self.current_center = center
        self.current_scalefactor = scalefactor

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        # uncomment to add a default rotation (useful when automatically snapshoting kinect data
        # glRotate(220, 0.0, 1.0, 0.0)

        tf = np.identity(4, 'f') / scalefactor
        tf[:3, 3] = -center / scalefactor
        tf[3, 3] = 1
        cur_mtx = GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX).T

        GL.glLoadMatrixf(cur_mtx.dot(tf).T)

        if want_camera:
            result = {'modelview_matrix': GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX),
                      'projection_matrix': GL.glGetDoublev(GL.GL_PROJECTION_MATRIX),
                      'viewport': GL.glGetIntegerv(GL.GL_VIEWPORT)
                      }
        else:
            result = None

        for m in all_meshes:
            if not hasattr(m, 'vbo'):
                # Precompute vertex vbo
                fidxs = m.f.flatten() if len(m.f) > 0 else np.arange(len(m.v))
                allpts = m.v[fidxs].astype(np.float32).flatten()
                vbo = VBO(allpts)
                m.vbo = {'v': vbo}

                # Precompute normals vbo
                if hasattr(m, 'vn'):
                    ns = m.vn.astype(np.float32)
                    ns = ns[m.f.flatten(), :]
                    m.vbo['vn'] = VBO(ns.flatten())
                elif hasattr(m, 'f') and m.f.size > 0:
                    ns = TriNormals(m.v, m.f).reshape(-1, 3)
                    ns = np.tile(ns, (1, 3)).reshape(-1, 3).astype(np.float32)
                    m.vbo['vn'] = VBO(ns.flatten())

                # Precompute texture vbo
                if hasattr(m, 'ft') and (m.ft.size > 0):
                    ftidxs = m.ft.flatten()
                    data = m.vt[ftidxs].astype(np.float32)[:, 0:2]
                    data[:, 1] = 1.0 - 1.0 * data[:, 1]
                    m.vbo['vt'] = VBO(data)

                # Precompute color vbo
                if hasattr(m, 'vc'):
                    data = m.vc[fidxs].astype(np.float32)
                    m.vbo['vc'] = VBO(data)
                elif hasattr(m, 'fc'):
                    data = np.tile(m.fc, (1, 3)).reshape(-1, 3).astype(np.float32)
                    m.vbo['vc'] = VBO(data)

        for e in all_lines:
            self.draw_lines(e)

        for m in all_meshes:
            if hasattr(m, 'texture_image') and not hasattr(m, 'textureID'):
                self.set_texture(m)
            self.draw_mesh(m, self.lighting_on)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()

        return result


class MeshViewerLocal:
    """Proxy viewer instance for visual inspection of geometric primitives.

    The class forks another python process holding the display. It communicates
    the commands with the remote instance seamlessly.

    Write-only attributes:

    :param titlebar: string printed in the window titlebar
    :param dynamic_meshes: list of Mesh objects to be displayed
    :param static_meshes: list of Mesh objects to be displayed
    :param dynamic_lines: list of Lines objects to be displayed
    :param static_lines: list of Lines objects to be displayed

    .. note::

      `static_meshes` is meant for Meshes that are
      updated infrequently, and dynamic_meshes is for Meshes
      that are updated frequently (same for dynamic_lines vs
      static_lines). They may be treated differently for
      performance reasons.

    """

    managed = {}

    def __new__(
        cls,
        titlebar=MESH_VIEWER_DEFAULT_TITLE,
        uid=None,
        host=ZMQ_HOST,
        port=None,
        shape=MESH_VIEWER_DEFAULT_SHAPE,
        keepalive=False,
        window_width=MESH_VIEWER_DEFAULT_WIDTH,
        window_height=MESH_VIEWER_DEFAULT_HEIGHT
    ):

        assert(uid is None or isinstance(uid, str))

        if uid == 'stack':
            uid = ''.join(traceback.format_list(traceback.extract_stack()))
        if uid and uid in MeshViewer.managed.keys():
            return MeshViewer.managed[uid]

        viewer = super(MeshViewerLocal, cls).__new__(cls)

        viewer.remote_host = host
        viewer.remote_port = port

        viewer.client = zmq.Context.instance().socket(zmq.PUSH)
        viewer.client.linger = 0

        if viewer.remote_port:
            addr = "{}://{}:{}".format(
                ZMQ_TRANSPORT,
                viewer.remote_host,
                viewer.remote_port)
            viewer.client.connect(addr)

            # XXX: Proper shape querying over the network is not
            # possible as of now. This should be tackled during
            # refactoring in the future.

            # viewer.shape = viewer.get_window_shape()
            viewer.shape = (1, 1)

            return viewer

        with open(os.devnull) as dev_null, \
            tempfile.TemporaryFile() as err:

                viewer.p = _run_self([titlebar, str(shape[0]), str(shape[1]), str(window_width), str(window_height)],
                                     stdin=dev_null,
                                     stdout=subprocess.PIPE,
                                     stderr=err)

                line = viewer.p.stdout.readline().decode()
                viewer.p.stdout.close()
                current_port = re.match('<PORT>(.*?)</PORT>', line)
                if not current_port:
                    raise Exception("MeshViewer remote appears to have failed to launch")
                current_port = int(current_port.group(1))
                viewer.client.connect('{}://{}:{}'.format(ZMQ_TRANSPORT, ZMQ_HOST, current_port))

                logging.info(
                    "started remote viewer on port {}".format(current_port))

                if uid:
                    MeshViewerLocal.managed[uid] = viewer
                viewer.shape = shape
                viewer.keepalive = keepalive
                return viewer

    def get_subwindows(self):
        return [[MeshSubwindow(parent_window=self, which_window=(r, c)) for c in range(self.shape[1])] for r in range(self.shape[0])]

    @staticmethod
    def _sanitize_meshes(list_of_meshes):
        lm = []

        # have to copy the meshes for now, because some contain CPython members,
        # before pushing them on the queue
        for m in list_of_meshes:
            if hasattr(m, 'fc'):
                lm.append(Mesh(v=m.v, f=m.f, fc=m.fc))
            elif hasattr(m, 'vc'):
                lm.append(Mesh(v=m.v, f=m.f, vc=m.vc))
            else:
                lm.append(Mesh(v=m.v, f=m.f if hasattr(m, 'f') else []))

            if hasattr(m, 'vn'):
                lm[-1].vn = m.vn
            if hasattr(m, 'fn'):
                lm[-1].fn = m.fn

            if hasattr(m, 'v_to_text'):
                lm[-1].v_to_text = m.v_to_text
            if hasattr(m, 'texture_filepath') and hasattr(m, 'vt') and hasattr(m, 'ft'):
                lm[-1].texture_filepath = m.texture_filepath
                lm[-1].vt = m.vt
                lm[-1].ft = m.ft

        return lm

    def _send_pyobj(self, label, obj, blocking, which_window):
        logging.debug("sending a request:")
        logging.debug("\tlabel = {!r}".format(label))
        logging.debug("\tobj = {!r}".format(obj))
        logging.debug("\tblocking = {!r}".format(blocking))
        logging.debug("\twhich_window = {!r}".format(which_window))

        if blocking:
            context = zmq.Context.instance()
            server = context.socket(zmq.PULL)
            server.linger = 0
            port = server.bind_to_random_port(
                "{}://{}".format(ZMQ_TRANSPORT, ZMQ_HOST),
                min_port=ZMQ_PORT_MIN,
                max_port=ZMQ_PORT_MAX,
                max_tries=100000)

            # sending with blocking'
            self.client.send_pyobj({
                'label': label,
                'obj': obj,
                'port': port,
                'which_window': which_window
            })

            task_completion_time = server.recv_pyobj()
            # task completion time was %.2fs in other process' % (task_completion_time,)
            server.close()
        else:
            # sending nonblocking
            res = self.client.send_pyobj({
                'label': label,
                'obj': obj,
                'which_window': which_window
            })

    def _recv_pyobj(self, label, port=None):
        context = zmq.Context.instance()
        server = context.socket(zmq.PULL)
        server.linger = 0

        if not port:
            port = server.bind_to_random_port(
                "{}://{}".format(ZMQ_TRANSPORT, ZMQ_HOST),
                min_port=ZMQ_PORT_MIN,
                max_port=ZMQ_PORT_MAX,
                max_tries=100000)

        self._send_pyobj(label, port, blocking=True, which_window=(0, 0))
        result = server.recv_pyobj()
        server.close()

        return result

    def set_dynamic_meshes(self, list_of_meshes, blocking=False, which_window=(0, 0)):
        self._send_pyobj('dynamic_meshes', self._sanitize_meshes(list_of_meshes), blocking, which_window)

    def set_static_meshes(self, list_of_meshes, blocking=False, which_window=(0, 0)):
        self._send_pyobj('static_meshes', self._sanitize_meshes(list_of_meshes), blocking, which_window)

    # list_of_model_names_and_parameters should be of form [{'name': scape_model_name, 'parameters': scape_model_parameters}]
    # here scape_model_name is the filepath of the scape model.
    def set_dynamic_models(self, list_of_model_names_and_parameters, blocking=False, which_window=(0, 0)):
        self._send_pyobj('dynamic_models', list_of_model_names_and_parameters, blocking, which_window)

    def set_dynamic_lines(self, list_of_lines, blocking=False, which_window=(0, 0)):
        self._send_pyobj('dynamic_lines', list_of_lines, blocking, which_window)

    def set_static_lines(self, list_of_lines, blocking=False, which_window=(0, 0)):
        self._send_pyobj('static_lines', list_of_lines, blocking, which_window)

    def set_titlebar(self, titlebar, blocking=False, which_window=(0, 0)):
        self._send_pyobj('titlebar', titlebar, blocking, which_window)

    def set_lighting_on(self, lighting_on, blocking=False, which_window=(0, 0)):
        self._send_pyobj('lighting_on', lighting_on, blocking, which_window)

    def set_autorecenter(self, autorecenter, blocking=False, which_window=(0, 0)):
        self._send_pyobj('autorecenter', autorecenter, blocking, which_window)

    def set_background_color(self, background_color, blocking=False, which_window=(0, 0)):
        assert(isinstance(background_color, np.ndarray))
        assert(background_color.size == 3)
        self._send_pyobj('background_color', background_color.flatten(), blocking, which_window)

    def get_keypress(self):
        return self._recv_pyobj('get_keypress')

    def get_mouseclick(self):
        """Returns a mouse click event.

        .. note::

          the call is blocking the caller until an event is received
        """
        return self._recv_pyobj('get_mouseclick')

    def get_event(self):
        return self._recv_pyobj('get_event')

    def get_window_shape(self):
        response = self._recv_pyobj("get_window_shape")
        return response["shape"]

    background_color = property(fset=set_background_color,
                                doc="Background color, as 3-element numpy array where 0 <= color <= 1.0.")

    dynamic_meshes = property(fset=set_dynamic_meshes,
                              doc="List of meshes for dynamic display.")
    static_meshes = property(fset=set_static_meshes,
                             doc="List of meshes for static display.")
    dynamic_models = property(fset=set_dynamic_models,
                              doc="List of model names and parameters for dynamic display.")

    dynamic_lines = property(fset=set_dynamic_lines,
                             doc="List of Lines for dynamic display.")
    static_lines = property(fset=set_static_lines,
                            doc="List of Lines for static display.")

    titlebar = property(fset=set_titlebar,
                        doc="Titlebar string.")

    def save_snapshot(self, path, blocking=False, which_window=(0, 0), wait_time=1):
        """Saves a snapshot of the current window into the specified file

        :param path: filename to which the current window content will be saved
        :param wait_time: waiting time to save snapshot. Increase it if the image is incomplete
        """
        print('Saving snapshot to %s, please wait...' % path)
        self._send_pyobj('save_snapshot', path, blocking, which_window)
        time.sleep(wait_time)

    def __del__(self):
        if hasattr(self, "p") and not self.keepalive:
            self.p.terminate()


class MeshViewerRemote:
    def __init__(
        self,
        titlebar=MESH_VIEWER_DEFAULT_TITLE,
        subwins_vert=MESH_VIEWER_DEFAULT_SHAPE[1],
        subwins_horz=MESH_VIEWER_DEFAULT_SHAPE[0],
        width=MESH_VIEWER_DEFAULT_WIDTH,
        height=MESH_VIEWER_DEFAULT_HEIGHT,
        port=None
    ):

        context = zmq.Context.instance()
        self.server = context.socket(zmq.PULL)
        self.server.linger = 0

        if not port:
            self.port = self.server.bind_to_random_port(
                "{}://{}".format(ZMQ_TRANSPORT, ZMQ_HOST),
                min_port=ZMQ_PORT_MIN,
                max_port=ZMQ_PORT_MAX,
                max_tries=100000)
        else:
            self.server.bind(
                "{}://{}:{}".format(ZMQ_TRANSPORT, ZMQ_HOST, port))
            self.port = port

        logging.debug(
            "listening for incoming messages on port {}"
            .format(self.port))

        # Print out our port so that our client can connect to us with it. Flush stdout immediately; otherwise
        # our client could wait forever.
        print('<PORT>%d</PORT>\n' % (self.port,))
        sys.stdout.flush()

        self.arcball = ArcBallT(width, height)
        self.transform = Matrix4fT()
        self.lastrot = Matrix3fT()
        self.thisrot = Matrix3fT()
        self.isdragging = False
        self.need_redraw = True

        self.shape = (subwins_vert, subwins_horz)
        self.mesh_viewers = [
            [
                MeshViewerSingle(
                    float(c) / (subwins_horz),
                    float(r) / (subwins_vert),
                    1. / subwins_horz,
                    1. / subwins_vert)
                for c in range(subwins_horz)
            ]
            for r in range(subwins_vert)
        ]

        self.tm_for_fps = 0.
        self.titlebar = titlebar
        self.activate(width, height)

    def snapshot(self, path, wait_time=0.1):
        """
        Takes a snapshot of the meshviewer window and saves it to disc.

        :param path: path to save the snapshot at.
        :param wait_time: waiting time to save snapshot. Increase it if the image is incomplete

        .. note:: Requires the Pillow package to be installed.
        """
        self.on_draw()
        time.sleep(wait_time)

        width = GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH)
        height = GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)

        data = (GLU.GLubyte * (3 * width * height))(0)
        GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, data)
        image = Image.frombytes(mode="RGB", size=(width, height), data=data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # Save image to disk
        image.save(path)

    def activate(self, width, height):
        GLUT.glutInit(['mesh_viewer'])
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_ALPHA | GLUT.GLUT_DEPTH)
        GLUT.glutInitWindowSize(width, height)
        GLUT.glutInitWindowPosition(0, 0)
        self.root_window_id = GLUT.glutCreateWindow(self.titlebar)

        GLUT.glutTimerFunc(100, self.checkQueue, 0)
        GLUT.glutReshapeFunc(self.on_resize_window)

        GLUT.glutKeyboardFunc(self.on_keypress)
        GLUT.glutMouseFunc(self.on_click)
        GLUT.glutMotionFunc(self.on_drag)
        GLUT.glutDisplayFunc(self.on_draw)

        self.init_opengl()

        GLUT.glutMainLoop()  # won't return until process is killed

    def on_drag(self, cursor_x, cursor_y):
        """ Mouse cursor is moving
            Glut calls this function (when mouse button is down)
            and pases the mouse cursor postion in window coords as the mouse moves.
        """
        from .geometry.rodrigues import rodrigues
        if (self.isdragging):
            mouse_pt = Point2fT(cursor_x, cursor_y)
            ThisQuat = self.arcball.drag(mouse_pt)  # // Update End Vector And Get Rotation As Quaternion
            self.thisrot = Matrix3fSetRotationFromQuat4f(ThisQuat)  # // Convert Quaternion Into Matrix3fT
            # Use correct Linear Algebra matrix multiplication C = A * B
            self.thisrot = Matrix3fMulMatrix3f(self.lastrot, self.thisrot)  # // Accumulate Last Rotation Into This One

            # make sure it is a rotation
            self.thisrot = rodrigues(rodrigues(self.thisrot)[0])[0]
            self.transform = Matrix4fSetRotationFromMatrix3f(self.transform, self.thisrot)  # // Set Our Final Transform's Rotation From This One
            GLUT.glutPostRedisplay()
        return

    # The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)
    def on_keypress(self, *args):
        key = args[0]
        if hasattr(self, 'event_port'):
            self.keypress_port = self.event_port
            del self.event_port
        if hasattr(self, 'keypress_port'):
            client = zmq.Context.instance().socket(zmq.PUSH)
            client.connect('{}://{}:{}'.format(ZMQ_TRANSPORT, ZMQ_HOST, self.keypress_port))
            client.send_pyobj({'event_type': 'keyboard', 'key': key})
            del self.keypress_port

    def on_click(self, button, button_state, cursor_x, cursor_y):
        """ Mouse button clicked.
            Glut calls this function when a mouse button is
            clicked or released.
        """

        self.isdragging = False

        if (button == GLUT.GLUT_LEFT_BUTTON and button_state == GLUT.GLUT_UP):
            # Left button released
            self.lastrot = copy.copy(self.thisrot)  # Set Last Static Rotation To Last Dynamic One

        elif (button == GLUT.GLUT_LEFT_BUTTON and button_state == GLUT.GLUT_DOWN):
            # Left button clicked down
            self.lastrot = copy.copy(self.thisrot)  # Set Last Static Rotation To Last Dynamic One
            self.isdragging = True  # // Prepare For Dragging
            mouse_pt = Point2fT(cursor_x, cursor_y)
            self.arcball.click(mouse_pt)  # Update Start Vector And Prepare For Dragging

        elif (button == GLUT.GLUT_RIGHT_BUTTON and button_state == GLUT.GLUT_DOWN):
            # If a mouse click location was requested, return it to caller
            if hasattr(self, 'event_port'):
                self.mouseclick_port = self.event_port
                del self.event_port
            if hasattr(self, 'mouseclick_port'):
                self.send_mouseclick_to_caller(cursor_x, cursor_y)

        elif (button == GLUT.GLUT_MIDDLE_BUTTON and button_state == GLUT.GLUT_DOWN):
            # If a mouse click location was requested, return it to caller
            if hasattr(self, 'event_port'):
                self.mouseclick_port = self.event_port
                del self.event_port
            if hasattr(self, 'mouseclick_port'):
                self.send_mouseclick_to_caller(cursor_x, cursor_y, button='middle')

        GLUT.glutPostRedisplay()

    def send_mouseclick_to_caller(self, cursor_x, cursor_y, button='right'):

        client = zmq.Context.instance().socket(zmq.PUSH)
        client.connect('{}://{}:{}'.format(ZMQ_TRANSPORT, ZMQ_HOST, self.mouseclick_port))
        cameras = self.on_draw(want_cameras=True)

        window_height = GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)
        depth_value = GL.glReadPixels(cursor_x, window_height - cursor_y, 1, 1, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)

        pyobj = {
            'event_type': 'mouse_click_%sbutton' % button,
            'u': None, 'v': None,
            'x': None, 'y': None, 'z': None,
            'subwindow_row': None,
            'subwindow_col': None
        }

        for subwin_row, camera_list in enumerate(cameras):
            for subwin_col, camera in enumerate(camera_list):

                # test for out-of-bounds
                if cursor_x < camera['viewport'][0]:
                    continue
                if cursor_x > (camera['viewport'][0] + camera['viewport'][2]):
                    continue
                if window_height - cursor_y < camera['viewport'][1]:
                    continue
                if window_height - cursor_y > (camera['viewport'][1] + camera['viewport'][3]):
                    continue

                xx, yy, zz = GLU.gluUnProject(
                    cursor_x, window_height - cursor_y, depth_value,
                    camera['modelview_matrix'],
                    camera['projection_matrix'],
                    camera['viewport'])

                pyobj = {
                    'event_type': 'mouse_click_%sbutton' % button,
                    'u': cursor_x - camera['viewport'][0], 'v': window_height - cursor_y - camera['viewport'][1],
                    'x': xx, 'y': yy, 'z': zz,
                    'which_subwindow': (subwin_row, subwin_col)
                }

        client.send_pyobj(pyobj)
        del self.mouseclick_port

    def on_draw(self, want_cameras=False):
        # sys.stderr.write('fps: %.2e\n' % (1. / (time.time() - self.tm_for_fps)))
        self.tm_for_fps = time.time()
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        cameras = []
        for mvl in self.mesh_viewers:
            cameras.append([])
            for mv in mvl:
                cameras[-1].append(mv.on_draw(self.transform, want_cameras))
        GL.glFlush()  # Flush The GL Rendering Pipeline
        GLUT.glutSwapBuffers()
        self.need_redraw = False
        if want_cameras:
            return cameras

    def on_resize_window(self, Width, Height):
        """Reshape The Window When It's Moved Or Resized"""
        self.arcball.setBounds(Width, Height)  # //*NEW* Update mouse bounds for arcball
        return

    def send_window_shape(self, port):
        client = zmq.Context.instance().socket(zmq.PUSH)
        client.connect('{}://{}:{}'.format(ZMQ_TRANSPORT, ZMQ_HOST, port))
        client.send_pyobj({
            'event_type': 'window_shape',
            'shape': self.shape
        })

    def handle_request(self, request):
        label = request['label']
        obj = request['obj']
        w = request['which_window']
        mv = self.mesh_viewers[w[0]][w[1]]

        logging.debug("received a request: {}".format(request))

        # Handle each type of request.
        # Some requests require a redraw, and
        # some don't.
        if label == 'dynamic_meshes':
            mv.dynamic_meshes = obj
            self.need_redraw = True
        elif label == 'dynamic_models':
            mv.dynamic_models = obj
            self.need_redraw = True
        elif label == 'static_meshes':
            mv.static_meshes = obj
            self.need_redraw = True
        elif label == 'dynamic_lines':
            mv.dynamic_lines = obj
            self.need_redraw = True
        elif label == 'static_lines':
            mv.static_lines = obj
            self.need_redraw = True
        elif label == 'autorecenter':
            mv.autorecenter = obj
            self.need_redraw = True
        elif label == 'titlebar':
            assert(isinstance(obj, str))
            self.titlebar = obj
            GLUT.glutSetWindowTitle(obj)
        elif label == 'lighting_on':
            mv.lighting_on = obj
            self.need_redraw = True
        elif label == 'background_color':
            GL.glClearColor(obj[0], obj[1], obj[2], 1.0)
            self.need_redraw = True
        elif label == 'save_snapshot':  # redraws for itself
            assert(isinstance(obj, str))
            self.snapshot(obj)
        elif label == 'get_keypress':
            self.keypress_port = obj
        elif label == 'get_mouseclick':
            self.mouseclick_port = obj
        elif label == 'get_event':
            self.event_port = obj
        elif label == 'get_window_shape':
            self.send_window_shape(obj)
        else:
            return False  # can't handle this request string

        return True  # handled the request string

    def checkQueue(self, unused_timer_id):
        GLUT.glutTimerFunc(20, self.checkQueue, 0)

        try:
            request = self.server.recv_pyobj(zmq.NOBLOCK)
        except zmq.ZMQError as e:
            if e.errno != zmq.EAGAIN:
                raise  # something wrong besides empty queue
            return  # empty queue, no problem

        if not request:
            return

        while (request):
            task_completion_time = time.time()
            if not self.handle_request(request):
                raise Exception('Unknown command string: %s' % (request['label']))
            task_completion_time = time.time() - task_completion_time

            if 'port' in request:  # caller wants confirmation
                port = request['port']
                client = zmq.Context.instance().socket(zmq.PUSH)
                client.connect('{}://{}:{}'.format(ZMQ_TRANSPORT, ZMQ_HOST, port))
                client.send_pyobj(task_completion_time)
            try:
                request = self.server.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError as e:
                if e.errno != zmq.EAGAIN:
                    raise
                request = None

        if self.need_redraw:
            GLUT.glutPostRedisplay()

    def init_opengl(self):
        """A general OpenGL initialization function.  Sets all of the initial parameters.

        We call this right after our OpenGL window is created.
        """

        GL.glClearColor(0.0, 0.0, 0.0, 1.0)  # This Will Clear The Background Color To Black
        GL.glClearDepth(1.0)  # Enables Clearing Of The Depth Buffer
        GL.glDepthFunc(GL.GL_LEQUAL)  # The Type Of Depth Test To Do
        GL.glEnable(GL.GL_DEPTH_TEST)  # Enables Depth Testing
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glHint(GL.GL_PERSPECTIVE_CORRECTION_HINT, GL.GL_NICEST)  # Really Nice Perspective Calculations

        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_LIGHTING)

        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glEnable(GL.GL_NORMALIZE)  # important since we rescale the modelview matrix

        return True


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'TEST_FOR_OPENGL':
        _test_for_opengl()

    elif len(sys.argv) > 2:
        m = MeshViewerRemote(titlebar=sys.argv[1],
                             subwins_vert=int(sys.argv[2]),
                             subwins_horz=int(sys.argv[3]),
                             width=int(sys.argv[4]),
                             height=int(sys.argv[5]))
    else:
        print("#" * 10)
        print('Usage:')
        print("python -m %s.%s arguments" % (__package__, os.path.splitext(os.path.basename(__file__))[0]))
