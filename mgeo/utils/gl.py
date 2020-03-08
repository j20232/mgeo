import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from pygame.locals import *


def setup(width, height, title):
    pygame.init()
    pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)
    pygame.display.set_caption(title)


def draw_background(bmp_name, width, height):
    bg_image = pygame.image.load(bmp_name).convert()
    bg_data = pygame.image.tostring(bg_image, "RGBX", 1)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Link texture
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, glGenTextures(1))
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_data)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    # Fill window
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-1.0, -1.0, -1.0)

    glTexCoord2f(1.0, 0.0)
    glVertex3f(1.0, -1.0, -1.0)

    glTexCoord2f(1.0, 1.0)
    glVertex3f(1.0, 1.0, -1.0)

    glTexCoord2f(0.0, 1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glEnd()

    glDeleteTextures(1)


def set_projection_from_intrinsic(K, width, height, near=0.1, far=100.0):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    fx = K[0, 0]
    fy = K[1, 1]
    fovy = 2 * np.arctan(0.5 * height / fy) * 180 / np.pi
    aspect = float(width * fy) / float(height * fx)
    gluPerspective(fovy, aspect, near, far)


def set_modelview_from_extrinsic(Rt):
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # adjust rotation
    R = Rt[:, :3]
    U, S, V = np.linalg.svd(R)
    R = U @ V
    R[0] = - R[0]

    # set translation
    t = Rt[:, 3]

    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # cv -> gl
    M = np.eye(4)
    M[:3, :3] = R @ Rx
    M[:3, 3] = t

    M = M.T
    m = M.flatten()
    glLoadMatrixf(m)


def draw_teapot(size):
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)

    # draw a red teapot
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0, 0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.0, 0.0, 0.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.7, 0.6, 0.6, 0.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 0.25 * 128.0)
    # glutInit()
    glutSolidTeapot(size)
    glFlush()
