#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2013 Max Planck Society. All rights reserved.

import os
import numpy as np
from OpenGL.GL import glPixelStorei, \
    glGenTextures, \
    glBindTexture, \
    glGenerateMipmap, \
    glHint, \
    glTexImage2D
from OpenGL.GL import GL_UNPACK_ALIGNMENT, \
    GL_TEXTURE_2D, \
    GL_RGB, \
    GL_BGR, \
    GL_GENERATE_MIPMAP_HINT, \
    GL_NICEST, \
    GL_UNSIGNED_BYTE


def get_image_with_text(text, fgcolor, bgcolor):
    if not hasattr(get_image_with_text, 'cache'):
        get_image_with_text.cache = {}

    import zlib
    uid = str(zlib.crc32(text)) + str(zlib.crc32(np.array(fgcolor))) + str(zlib.crc32(np.array(bgcolor)))
    if uid not in get_image_with_text.cache:
        from PIL import ImageFont
        from PIL import Image
        from PIL import ImageDraw

        font = ImageFont.truetype("/Library/Fonts/Courier New.ttf", 30)

        imsize = (256, 256)

        bgarray = np.asarray(np.zeros((imsize[0], imsize[1], 3)), np.uint8)
        bgarray[:, :, 0] += bgcolor[0]
        bgarray[:, :, 1] += bgcolor[1]
        bgarray[:, :, 2] += bgcolor[2]
        img = Image.fromarray(bgarray)
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(text, font=font)
        text_pos = ((imsize[0] - w) / 2, (imsize[1] - h) / 2)
        draw.text(text_pos, text, fill=fgcolor, font=font)
        get_image_with_text.cache[uid] = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3) * 255
    return get_image_with_text.cache[uid]


def get_textureid_with_text(text, fgcolor, bgcolor):
    if not hasattr(get_textureid_with_text, 'cache'):
        get_textureid_with_text.cache = {}

    import zlib
    uid = str(zlib.crc32(text)) + str(zlib.crc32(np.array(fgcolor))) + str(zlib.crc32(np.array(bgcolor)))
    if uid not in get_textureid_with_text.cache:
        from PIL import ImageFont
        from PIL import Image
        from PIL import ImageDraw

        font = ImageFont.truetype(os.path.join(os.path.dirname(__file__),
                                               "ressources",
                                               "Arial.ttf"),
                                  100)

        imsize = (128, 128)

        bgarray = np.asarray(np.zeros((imsize[0], imsize[1], 3)), np.uint8)
        bgarray[:, :, 0] += bgcolor[0]
        bgarray[:, :, 1] += bgcolor[1]
        bgarray[:, :, 2] += bgcolor[2]
        img = Image.fromarray(bgarray)
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(text, font=font)
        text_pos = ((imsize[0] - w) / 2, (imsize[1] - h) / 2)
        draw.text(text_pos, text, fill=tuple(np.asarray(fgcolor, np.uint8)), font=font)
        texture_data = np.asarray(np.array(img.getdata()).reshape(img.size[0], img.size[1], 3) * 255, np.uint8)

        textureID = glGenTextures(1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, textureID)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_data.shape[1], texture_data.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, texture_data.flatten())
        glHint(GL_GENERATE_MIPMAP_HINT, GL_NICEST)  # must be GL_FASTEST, GL_NICEST or GL_DONT_CARE
        glGenerateMipmap(GL_TEXTURE_2D)
        get_textureid_with_text.cache[uid] = textureID

    return get_textureid_with_text.cache[uid]
