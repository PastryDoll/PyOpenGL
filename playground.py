import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np 
import ctypes
import pyrr

pg.init()
pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
pg.display.set_mode((1080,720), pg.OPENGL|pg.DOUBLEBUF|pg.RESIZABLE, 32)
clock = pg.time.Clock()
glClearColor(0.1,0.2,0.0,1.0)
glEnable(GL_BLEND)
glEnable(GL_DEPTH_TEST)
glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

# x,y,z,r,g,b,s,t
triangle_mesh = (-0.5,-0.5,0.0,1.0,0.0,0.0,0.0,1.0,
            0.5,-0.5,0.0,0.0,1.0,0.0,1.0,1.0,
            0.0,0.5,0.0,0.0,0.0,1.0,0.5,0.0)

cube_mesh  = (
            -0.5, -0.5, -0.5, 0, 0,
             0.5, -0.5, -0.5, 1, 0,
             0.5,  0.5, -0.5, 1, 1,

             0.5,  0.5, -0.5, 1, 1,
            -0.5,  0.5, -0.5, 0, 1,
            -0.5, -0.5, -0.5, 0, 0,

            -0.5, -0.5,  0.5, 0, 0,
             0.5, -0.5,  0.5, 1, 0,
             0.5,  0.5,  0.5, 1, 1,

             0.5,  0.5,  0.5, 1, 1,
            -0.5,  0.5,  0.5, 0, 1,
            -0.5, -0.5,  0.5, 0, 0,

            -0.5,  0.5,  0.5, 1, 0,
            -0.5,  0.5, -0.5, 1, 1,
            -0.5, -0.5, -0.5, 0, 1,

            -0.5, -0.5, -0.5, 0, 1,
            -0.5, -0.5,  0.5, 0, 0,
            -0.5,  0.5,  0.5, 1, 0,

             0.5,  0.5,  0.5, 1, 0,
             0.5,  0.5, -0.5, 1, 1,
             0.5, -0.5, -0.5, 0, 1,

             0.5, -0.5, -0.5, 0, 1,
             0.5, -0.5,  0.5, 0, 0,
             0.5,  0.5,  0.5, 1, 0,

            -0.5, -0.5, -0.5, 0, 1,
             0.5, -0.5, -0.5, 1, 1,
             0.5, -0.5,  0.5, 1, 0,

             0.5, -0.5,  0.5, 1, 0,
            -0.5, -0.5,  0.5, 0, 0,
            -0.5, -0.5, -0.5, 0, 1,

            -0.5,  0.5, -0.5, 0, 1,
             0.5,  0.5, -0.5, 1, 1,
             0.5,  0.5,  0.5, 1, 0,

             0.5,  0.5,  0.5, 1, 0,
            -0.5,  0.5,  0.5, 0, 0,
            -0.5,  0.5, -0.5, 0, 1
        )            

class TriangleMesh:
    def __init__(self, vertices, vertex_count,texture,vertex_shader,fragment_shader) -> None:
        self.vertices = vertices
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.vertex_count = vertex_count
        self.texture = texture
        self.vao = 0
        self.vbo = 0
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader

def loadMesh(filename: str) -> list[float]:
    """
        Load a mesh from an obj file.

        Parameters:

            filename: the filename.
        
        Returns:

            The loaded data, in a flattened format.
    """

    v = []
    vt = []
    vn = []

    vertices = []

    with open(filename, "r") as file:

        line = file.readline()

        while line:

            words = line.split(" ")
            match words[0]:
            
                case "v":
                    v.append(read_vertex_data(words))

                case "vt":
                    vt.append(read_texcoord_data(words))
                
                case "vn":
                    vn.append(read_normal_data(words))
            
                case "f":
                    read_face_data(words, v, vt, vn, vertices)
            
            line = file.readline()

    return vertices
    
def read_vertex_data(words: list[str]) -> list[float]:
    """
        Returns a vertex description.
    """

    return [
        float(words[1]),
        float(words[2]),
        float(words[3])
    ]
    
def read_texcoord_data(words: list[str]) -> list[float]:
    """
        Returns a texture coordinate description.
    """

    return [
        float(words[1]),
        float(words[2])
    ]
    
def read_normal_data(words: list[str]) -> list[float]:
    """
        Returns a normal vector description.
    """

    return [
        float(words[1]),
        float(words[2]),
        float(words[3])
    ]

def read_face_data(
    words: list[str], 
    v: list[list[float]], vt: list[list[float]], 
    vn: list[list[float]], vertices: list[float]) -> None:
    """
        Reads an edgetable and makes a face from it.
    """

    triangleCount = len(words) - 3

    for i in range(triangleCount):

        make_corner(words[1], v, vt, vn, vertices)
        make_corner(words[2 + i], v, vt, vn, vertices)
        make_corner(words[3 + i], v, vt, vn, vertices)
    
def make_corner(corner_description: str, 
    v: list[list[float]], vt: list[list[float]], 
    vn: list[list[float]], vertices: list[float]) -> None:
    """
        Composes a flattened description of a vertex.
    """

    v_vt_vn = corner_description.split("/")
    
    for element in v[int(v_vt_vn[0]) - 1]:
        vertices.append(element)
    for element in vt[int(v_vt_vn[1]) - 1]:
        vertices.append(element)
    for element in vn[int(v_vt_vn[2]) - 1]:
        vertices.append(element)
        
def createShader(vertexFilePath, fragmentFilePath):
    with open(vertexFilePath,'r') as f:
        vertex_src = f.readlines()
    
    with open(fragmentFilePath, 'r') as f:
        fragment_src = f.readlines()

    shader = compileProgram(
        compileShader(vertex_src,GL_VERTEX_SHADER),
        compileShader(fragment_src, GL_FRAGMENT_SHADER)
    )

    return shader

def BindActivateTexture(filepath):
    image = pg.image.load(filepath).convert_alpha()
    image_w, image_h = image.get_rect().size
    image_data = pg.image.tostring(image,"RGBA")
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D,texture)
    glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,image_w,image_h,0,GL_RGBA,GL_UNSIGNED_BYTE,image_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glActiveTexture(GL_TEXTURE0)
    return texture

def drawMesh(mesh, pos, ori, scale):

    projection_transform = pyrr.matrix44.create_perspective_projection(
    fovy=45, aspect=1080/720,
    near=0.1, far=10, dtype=np.float32
    )

    model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
    model_transform = pyrr.matrix44.multiply(
            m1=model_transform, 
            m2=pyrr.matrix44.create_from_eulers(
                eulers=  np.radians(ori), 
                dtype = np.float32
            )
        )
    model_transform = pyrr.matrix44.multiply(
            m1=model_transform, 
            m2=pyrr.matrix44.create_from_translation(
                vec=np.array(pos),dtype=np.float32
            )
        )

    mesh.vao = glGenVertexArrays(1)
    glBindVertexArray(mesh.vao)
    mesh.vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo)
    glBufferData(GL_ARRAY_BUFFER, mesh.vertices.nbytes, mesh.vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,32,ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,32,ctypes.c_void_p(12))
    # glEnableVertexAttribArray(2)
    # glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,32,ctypes.c_void_p(24))
    BindActivateTexture(f"textures/{mesh.texture}")

    shader = createShader(f"shaders/{mesh.vertex_shader}", f"shaders/{mesh.fragment_shader}")
    glUseProgram(shader)

    glUniformMatrix4fv(
                glGetUniformLocation(shader,"model"), 1, GL_FALSE, 
                model_transform)
    
    glUniformMatrix4fv(
        glGetUniformLocation(shader, "projection"),
        1, GL_FALSE, projection_transform
    )
    glUniform1i(glGetUniformLocation(shader, "imageTexture"), 0)

    glDrawArrays(GL_TRIANGLES, 0, mesh.vertex_count)


running = True

# triangle = TriangleMesh(triangle_mesh,3, "cat.png", "vertex_triangle.txt", "fragment_triangle.txt")

cube = TriangleMesh(cube_mesh,len(cube_mesh)//5, "wood.jpeg", "vertex_cube.txt", "fragment_cube.txt")

counter = 0
position = [0,0,-8]
eulers = [0,0,0]
while(running):
    for event in pg.event.get():
        if (event.type ==  pg.WINDOWCLOSE):
            running = False
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    position = np.array(position, dtype=np.float32)
    eulers = np.array(eulers, dtype=np.float32)

    eulers[2] += 0.25*5
        
    if eulers[2] > 360:
        eulers[2] -= 360

    drawMesh(cube,position,eulers,0)
    print(eulers)

    pg.display.flip()

    clock.tick(60)

print("Freeing memory")
# glDeleteVertexArrays(1,(triangle.vao,))
# glDeleteBuffers(1,(triangle.vbo,))
print("Quitting...")
pg.quit()
print("Exit pygame...")