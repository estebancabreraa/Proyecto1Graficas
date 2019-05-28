################################################################################
#                             PROYECTO # 1 GRAFICAS                            #
################################################################################
#Nombre: Esteban Cabrera Arevalo.
#Carnet: 17781.
#Fecha: 07/05/2019.
import struct
from math import *
import random

################################################################################
#                                       BMP                                    #
################################################################################
#Nombre: Esteban Cabrera Arevalo.
#Carnet: 17781.
#Fecha: 07/05/2019.
class ClaseBMP(object):

        def write(self, filename):

                
                file = open(filename, "bw")
                
                ancho_t = self.padding(4,self.width)
                altura_t = self.padding(4,self.height)
                
                file.write(self.char("B"))
                file.write(self.char("M"))

                file.write(self.dword(14 + 40 + self.width * self.height*3))
                file.write(self.dword(0))
                file.write(self.dword(14+40))

                file.write(self.dword(40))
                file.write(self.dword(self.width))
                file.write(self.dword(self.height))
                file.write(self.word(1))
                file.write(self.word(24))
                file.write(self.dword(0))
                file.write(self.dword(self.width*self.height*3))
                file.write(self.dword(0))
                file.write(self.dword(0))
                file.write(self.dword(0))
                file.write(self.dword(0))
                for x in range(ancho_t):
                        for y in range(self.height):
                                if (x<self.width and y<self.height):
                                        file.write(self.framebuffer[y][x])
                                else:
                                        file.write(self.char("c"))
                file.close()
                
        def char(self,c):
                return struct.pack("=c", c.encode("ascii"))
        
        def word(self,c):
                return struct.pack("=h", c)
        
        def dword(self,c):
                return struct.pack("=l", c)

        def padding(self, base,c):
                if(c % base== 0):
                        return c
                else:
                        while (c%base):
                                c +=1
                        return c


        def __init__(self, width, height):
                self.width = abs(int(width))
                self.height = abs(int(height))
                self.framebuffer = []
                self.zbuffer=[]
                self.clear()


        def clear(self, r=0, b=0, g=130):
                self.framebuffer =[
                        [
                                self.color(r,g,b)
                                        for x in range(self.width)
                        ]
                        for y in range(self.height)
                ]
                self.zbuffer= [ [-float('inf') for x in range(self.width)] for y in range(self.height)]

        def color(self, r=0, g=0, b=0):
                if (r > 255 or g > 255 or b > 255 or r < 0 or g < 0 or b <0):
                        r = 0
                        g = 0
                        b = 0
                return bytes([b,g,r])

        def point(self,x,y,color):
                if(x < self.width and y < self.height):
                        self.framebuffer[x][y] =color

        def setZBValue(self, x,y,value):
                if x<self.width and y<self.height:
                        self.zbuffer[x][y]=value

        def getZBValue(self,x,y):
                if x<self.width and y<self.height:
                        return self.zbuffer[x][y]
                else:
                	return -float("inf")
                

################################################################################
#                                       OBJ                                    #
################################################################################
class OBJCTF(object):
        def __init__(self, filename):
                self.filename=filename
                self.faces=[]
                self.vertex=[]
                self.nvertex=[]
                self.materials=None
                self.tvert=[]
                self.materialF=[]

        def load(self):
                #Abrimos archivo
                doc = open(self.filename,"r")
                faces=[]
                contFaces=0
                mtlActual="default"
                mtlAnterior=mtlActual
                indice=[]
                lineas=doc.readlines()

                for line in  lineas:
                        
                        line=line.rstrip().split(" ")


                        if line[0] =="mtllib":
                                mtlFile = class_MTL(line[1])
                                if mtlFile.isFileO():
                                        mtlFile.load()
                                        self.materials=mtlFile.materials
                                else:
                                        self.faces=[]
              
                        elif line[0] == "v":
                                line.pop(0)
                                if line[0]=="":
                                        i=1
                                else:
                                        i=0
                                self.vertex.append((float(line[i]), float(line[i+1]), float(line[i+2])))
   
                        elif line[0] == "usemtl":
                                if self.materials:
                                        mtlAnterior=mtlActual
                                        mtlActual= line[1]
                                        indice.append(contFaces)
                                        if len(indice)==2:
                                                self.materialF.append((indice,mtlAnterior))
                                                indice=[indice[1]+1]

                        elif line[0] == "f":
                                line.pop(0)
                                face = []
                                for i in line:
                                        i = i.split("/")
                                        if i[1]=="":
                                                face.append((int(i[0]),int(i[-1])))
                                        else:
                                                face.append((int(i[0]),int(i[-1]),int(i[1])))
                                contFaces=contFaces+1
                                self.faces.append(face)
                                
                                
			
                        elif line[0] == "vn":
                                line.pop(0)
                                if line[0]=="":
                                        i=1
                                else:
                                        i=0
                                self.nvertex.append((float(line[i]), float(line[i+1]), float(line[i+2])))

                        elif line[0]=="vt":
                                line.pop(0)
                                self.tvert.append((float(line[0]), float(line[1])))
                if len(indice)<2 and self.materials:
                        indice.append(contFaces)
                        self.materialF.append((indice,mtlActual))
                        indice=[indice[1]+1]

                doc.close()
                

        def getFacesL(self):#Faces
                return self.faces
        def getVertexL(self):#Vertex
                return self.vertex
        def getVertexNormalL(self):
                return self.nvertex
        def getMaterials(self):
                return self.materials
        def getMaterialF(self):
                return self.materialF
        def getTVertex(self):
                return self.tvert


################################################################################
#                                       MTL                                    #
################################################################################
class class_MTL(object):

        def __init__(self,nombreArchivo):
                self.nombreArchivo=nombreArchivo
                self.readMTL()
                self.materials={}
                self.archivo=self.readMTL()

        def isFileO(self):
                return self.mtldoc
        
        def readMTL(self):
        	try:
        		file = open(self.nombreArchivo,"r")
        		self.mtldoc= True
        		return file
        	except Exception as e:
        		self.mtldoc=False


        def load(self):

                if self.isFileO():
                        materialActual= None
                        opticalD,ambientColor,emissiveC,shini,difuseColor,trans,ill,ds=0,0,0,0,0,0,0,0

                        for linea in self.archivo.readlines():
                                linea=linea.split(" ")

                                if linea[0]=="Ni":
                                        opticalD=float(linea[1])

                                elif linea[0]=="Ka":
                                        ambientColor=(float(linea[1]), float(linea[2]), float(linea[3]))

                                elif linea[0]=="Ke":
                                        emissiveC=(float(linea[1]), float(linea[2]), float(linea[3]))

                                elif linea[0]=="Ns":
                                        shini=float(linea[1])

                                elif linea[0]=="Kd":
                                        difuseColor= (float(linea[1]), float(linea[2]), float(linea[3]))

                                elif linea[0] == "d" or linea[0] == "Tr":
                                        trans=(float(linea[1]), linea[0])

                                elif linea[0]=="illum":
                                        ill=int(linea[1])

                                elif linea[0]=="newmtl":
                                        materialActual=linea[1].rstrip()
                                elif linea[0] == "ks":
                                        ds=(float(line[1]), float(line[2]), float(line[3]))
                                        
                                elif materialActual:
                                        self.materials[materialActual]=MaterialClase(materialActual,ambientColor,difuseColor,ds,emissiveC,trans,ill,opticalD)
                        if materialActual not in self.materials.keys():
                                self.materials[materialActual]=MaterialClase(materialActual,ambientColor,difuseColor,ds,emissiveC,trans,ill,opticalD)

################################################################################
#                                       BMP                                    #
################################################################################
class MaterialClase(object):

        """
        Constructor
        """
        def __init__(self,materialActual,ambientColor,difuseColor,ds,emissiveC,trans,ill,opticalD):
               self.name=materialActual.rstrip()
               self.ambientColor=ambientColor
               self.difuseColor=difuseColor
               self.ds=ds
               self.emissiveC=emissiveC
               self.trans=trans
               self.ill=ill
               self.opticalD=opticalD

################################################################################
#                                   TEXTURE                                    #
################################################################################        
class Texture(object):

        def __init__(self,nombreA):
                self.archivo=nombreA
                self.text=None
                self.load()

        def load(self):

                self.texto=ClaseBMP(0,0)
                try:
                        self.text.load(self.archivo)
                except:
                        self.text=None

        def write(self):
                self.text.write(self.archivo[:len(self.archivo)-4]+"text.bmp")

        def textured(self):
                return True if self.text else False


        def getColor(self, x,y, intensity=1):
                if y==1:
                        px= self.text.width-1
                else:
                        px=int (y*self.texto.width)
                if x==1:
                        py=self.text.height
                else:
                        py=int(x*self.text.height)
                return  bytes(map(lambda b: round(b*intensity) if b*intensity > 0 else 0, self.__text.framebuffer[py][px]))

################################################################################
#                                       MTX                                    #
################################################################################
class MTX(object):

	def __init__(self, data):
		self.data = data
		self.col = len(data[0])
		self.row = len(data)
		
	def __mul__(self, MTX2):
		resultado = []
		for i in range(self.row):
			resultado.append([])
			for j in range(MTX2.col):
				resultado[-1].append(0)
		for i in range(self.row):
			for j in range(MTX2.col):
				for k in range(MTX2.row):
					resultado[i][j] += self.data[i][k] * MTX2.data[k][j]
		return MTX(resultado)
	
	def getData(self):
		return self.data

################################################################################
#                                       SR                                     #
################################################################################
class SR(object):
    def glInit(self):
        self.drawing =ClaseBMP(0,0)
        self.Portsize=(0,0)
        self.PortStart=(0,0)
        self.name="dibujo.bmp"

        self.color=self.drawing.color(255,255,255)
        self.obj=None
        self.text=None
    """
    Crea imagen 
    """
    def glFinish(self):
        self.drawing.write(self.name)

    def glCreateWindow(self,width,height):
        self.drawing=ClaseBMP(width,height)
        self.Portsize=(width,height)

    def glColor(self,r,g,b):
        self.color=self.drawing.color(int(r*255),int(g*255),int(b*255))
        return self.color


    def glViewPort(self,x,y,width,height):
        self.PortStart=(x,y)
        self.Portsize=(width,height)

    def glClear(self):
        self.drawing.clear()


    def glClearColor(self,r,g,b):
        self.drawing.clear(r,g,b)


    def glVertex(self,x,y):
        coorx=int(self.Portsize[0]*(x+1)*(1/2)*self.PortStart[0])
        coory=int(self.Portsize[1]*(y+1)*(1/2)*self.PortStart[1])
        self.drawing.point(coorx,coory,self.color)
    

    def glVertexPro(self,x,y):
        coorx=int(self.Portsize[0]*(x+1)*(1/2)*self.PortStart[0])
        coory=int(self.Portsize[1]*(y+1)*(1/2)*self.PortStart[1])
        self.drawing.point(coorx,coory,self.color)
        self.drawing.point(coorx,coory+1,self.color)
        self.drawing.point(coorx+1,coory,self.color)
        self.drawing.point(coorx+1,coory+1,self.color)
                    

    def norm(self,v0):
        v=self.length(v0)
        if not v:
            return [0,0,0]
        return[v0[0]/v, v0[1]/v, v0[2]/v]


    def bar(self,a,b,c,x,y):
        vertice1=(c[0]-a[0], b[0]-a[0],a[0]-x)
        vertice2=(c[1]-a[1], b[1]-a[1],a[1]-y)
        bari=self.calculateCross(vertice1,vertice2)
        if abs(bari[2])<1:
            return -1,-1,-1
        return ( 1 - (bari[0] + bari[1]) / bari[2], bari[1] / bari[2], bari[0] / bari[2])

################################################################################
#                                TRIANGULOS                                    #
################################################################################
    def triangulo(self,a,b,c,color=None, texture=None,  txtcoor=(),intensity=1, normals=None, shader=None,baseColor=(1,1,1)):
        limitadormin,limitadormax=self.limitB(a,b,c)
        for x in range(limitadormin[0],limitadormax[0]+1):
            for y in range(limitadormin[1], limitadormax[1] + 1):
                m,n,o=self.bar(a,b,c,x,y)
                if m <0 or n<0 or o<0:
                    continue
                if texture:
                    Texturea=txtcoor
                    Textureb=txtcoor
                    Texturec=txtcoor
                    tx=Texturea[0] * m + Textureb[0] * n + Texturec[0] * o
                    ty=Texturea[1] * m + Textureb[1] * n + Texturec[1] * o
                    color=self.text.getColor(tx,ty,intensity)
                elif shader:
                    color = shader(self,bar(b,n,o),Vnormals=normals, baseColor=baseColor)
                q=a[2]*m+b[2]*n+c[2]*o
                if x<0 or y<0:
                    continue
                if q>self.drawing.getZBValue(x,y):
                	self.drawing.point(x,y,color)
                	self.drawing.setZBValue(x,y,q)

    def limitB(self,*listV):
        xs= [vertex[0] for vertex in listV] 
        ys= [vertex[1] for vertex in listV]
        xs.sort()
        ys.sort()
        return (xs[0],ys[0]),(xs[-1],ys[-1])

    def calculateDot(self,v0,v1):
        return v0[0]*v1[0] + v0[1] * v1[1] + v0[2] * v1[2]

    def calculateCross(self,v0,v1):
        return [v0[1] * v1[2] - v0[2] * v1[1], v0[2] * v1[0] - v0[0] * v1[2], v0[0] * v1[1] - v0[1] * v1[0]]

    def length(self,v0):
        return (v0[0]**2 + v0[1]**2 + v0[2]**2)**0.5
    def sub(self,v0,v1):
        return [v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]]
    

    def calculatePQ(self,p,q):
        return [q[0]-p[0],q[1]-p[1],q[2]-p[2]]
################################################################################
#                                 CARGA OBJ                                    #
################################################################################

    def loadOBJ(self, filename, translate=(0, 0, 0), scale=(1, 1, 1), fill=True, textured=None, rotate=(0, 0, 0), shader=None):
        self.obj= OBJCTF(filename)
        self.obj.load()
        self.modelm(translate,scale,rotate)
        light=(0,0,1)
        faces=self.obj.getFacesL()
        vertex=self.obj.getVertexL()
        vn=self.obj.getVertexNormalL()
        materials=self.obj.getMaterials()
        mtlFaces=self.obj.getMaterialF()
        tVertex=self.obj.getTVertex()
        self.text=Texture(textured)
        if materials:
            for mats in mtlFaces:
                inicio,final=mats[0]
                color=materials[mats[1]].difuseColor
                for index in range (inicio,final):
                    face =faces[index]
                    cont=len(face)
                    if cont==3:
                        c1=face[0][0]-1
                        c2=face[1][0]-1
                        c3=face[2][0]-1
                        a=self.trans(vertex[c1])
                        b=self.trans(vertex[c2])
                        c=self.trans(vertex[c3])
                        if shader:
                            sa=vn[f1]
                            sb=vn[f1]
                            sc=vn[f1]
                            self.triangulo(a, b, c, baseColor=color, shader=shader,normals=(sa, sb, sc))
                        else:
                            normal=self.norm(self.calculateCross(self.sub(b,a),self.sub(c,a)))
                            intensity=self.calculateDot(normal,light)
                            if not(self.text.textured()):
                                if intensity<0:
                                    continue
                                self.triangulo(a,b,c,color=self.glColor(color[0]*intensity, color[1]*intensity, color[2]*intensity))
        else:
            print("No existen materiales de: "+filename)
            for face in faces:
                cont=len(face)
                if cont==3:
                    c1=face[0][0]-1
                    c2=face[1][0]-1
                    c3=face[2][0]-1
                    a=self.trans(vertex[c1])
                    b=self.trans(vertex[c2])
                    c=self.trans(vertex[c3])
                    if shader:
                        sa=vn[f1]
                        sb=vn[f2]
                        sc=vn[f3]
                        self.triangulo(a, b, c, baseColor=color, shader=shader,normals=(sa, sb, sc))
                    else:
                        nl=self.norm(self.calculateCross(self.sub(b,a), self.sub(c,a)))
                        intensity=self.calculateDot(nl,light)

                        if not self.text.textured():
                            if intensity>0:
                                continue
                            self.triangulo(a, b, c,color=self.glColor(intensity, intensity, intensity))
                        else:
                            if self.text.textured():
                                txt1 = face[0][-1]-1
                                txt2 = face[1][-1]-1
                                txt3 = face[2][-1]-1
                                txta = tVertex[txt1]
                                txtb = tVertex[txt2]
                                txtc = tVertex[txt3]
                                self.triangulo(a, b, c, texture=self.text.isTextured(), texture_coords=(ta,tb,tc), intensity=intensity)
                else:
                    txt1 = face[0][-1]-1
                    txt2 = face[1][-1]-1
                    txt3 = face[2][-1]-1
                    txt4 = face[3][-1]-1
                    listV=[self.trans(vertex[f1]),self.trans(vertex[f2]),self.trans(vertex[f3]),self.trans(vertex[f4])]
                    nl=self.norm(self.calculateCross(self.sub(listV[0], listV[1]), self.sub(listV[1], listV[2])))
                    intensity=self.calculateDot(nl,light)
                    a=listv
                    b=listv
                    c=listv
                    d=listv
                    if not textured:
                        if intensity<0:
                            continue
                        self.triangulo(a,b,c,color=self.glColor(intensity, intensity, intensity))
                        self.triangulo(a,b,d,color=self.glColor(intensity, intensity, intensity))
                    if textured:
                        if self.text.textured():
                            t1=face[0][-1]-1
                            t2=face[1][-1]-1
                            t3=face[2][-1]-1
                            t4=face[3][-1]-1
                            ta=tVertex[t1]
                            tb=tVertex[t2]
                            tc=tVertex[t3]
                            td=tVertex[t4]
                            self.triangulo(a,b,c,texture=self.text.isTextured(), texture_coords=(ta, tb, tc), intensity=intensity)
                            self.triangulo(a,b,d,texture=self.text.isTextured(), texture_coords=(ta, tb, td), intensity=intensity)

    def modelm(self, translate=(0, 0, 0), scale=(1, 1, 1), rotate=(0, 0, 0)):
        translation_MTX = MTX([[1, 0, 0, translate[0]],[0, 1, 0, translate[1]],[0, 0, 1, translate[2]],[0, 0, 0, 1],])
        a = rotate[0]
        rotation_MTX_x = MTX([[1, 0, 0, 0],[0, cos(a), -sin(a), 0],[0, sin(a),  cos(a), 0],[0, 0, 0, 1]])
        a = rotate[1]
        rotation_MTX_y = MTX([[cos(a), 0,  sin(a), 0],[     0, 1,       0, 0],[-sin(a), 0,  cos(a), 0],[     0, 0,       0, 1]])
        a = rotate[2]
        rotation_MTX_z = MTX([[cos(a), -sin(a), 0, 0],[sin(a),  cos(a), 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
        rotation_MTX = rotation_MTX_x * rotation_MTX_y * rotation_MTX_z
        scale_MTX = MTX([[scale[0], 0, 0, 0],[0, scale[1], 0, 0],[0, 0, scale[2], 0],[0, 0, 0, 1],])
        self.Model = translation_MTX * rotation_MTX * scale_MTX

    def viewm(self, x, y, z, center):
        m = MTX([[x[0], x[1], x[2],  0],[y[0], y[1], y[2], 0],[z[0], z[1], z[2], 0],[0,0,0,1]])
        o = MTX([[1, 0, 0, -center[0]],[0, 1, 0, -center[1]],[0, 0, 1, -center[2]],[0, 0, 0, 1]])
        self.View = m * o

    def projectionm(self, coeff):
       	self.Projection = MTX([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, coeff, 1]])

    def viewportm(self, x=0, y =0):
       	self.Viewport =  MTX([[self.drawing.width/2, 0, 0, x + self.drawing.width/2],[0, self.drawing.height/2, 0, y + self.drawing.height/2],[0, 0, 128, 128],[0, 0, 0, 1]])
        
    def aim(self, eye, center, up):
       	z = self.norm(self.sub(eye, center))
       	x = self.norm(self.calculateCross(up, z))
       	y = self.norm(self.calculateCross(z,x))
       	self.viewm(x, y, z, center)
       	self.projectionm(-1/self.length(self.sub(eye, center)))
       	self.viewportm()

    def trans(self, vertex):
       	agv = MTX([[vertex[0]],[vertex[1]],[vertex[2]],[1]])
       	transformed_vertex = self.Viewport * self.Projection * self.View * self.Model * agv
       	transformed_vertex = transformed_vertex.getData()
       	tra = [round(transformed_vertex[0][0]/transformed_vertex[3][0]), round(transformed_vertex[1][0]/transformed_vertex[3][0]), round(transformed_vertex[2][0]/transformed_vertex[3][0])]
       	return tra


    def setfname(self,filename):
        self.name=filename


################################################################################
#                            INICIALIZACION                                    #
################################################################################
image = SR()
image.glInit()
image.glCreateWindow(800, 800)
image.aim((-1,3,5), (0,0,0), (0,1,0))
image.glViewPort(0,0,800,800)
image.setfname("proyecto.bmp")

################################################################################
#                CARGA DE MODELOS, POSICION, ESCALA Y ROTACION                 #
################################################################################
print("Cargando y renderizando calle.obj...")
image.loadOBJ("calle.obj", translate=(-0.5,-0.5,0), scale=(0.05,0.05,0.05), rotate=(0,1,-0.25), fill=True)
print("Cargando y renderizando mcqueen.obj...")
image.loadOBJ("mcqueen.obj", translate=(0,-0.5,-0.5), scale=(0.2,0.2,0.2), rotate=(0,-0.8,0), fill=True)
print("Cargando y renderizando farmhouse_obj.obj...")
image.loadOBJ("farmhouse_obj.obj", translate=(0.2,0.2,-0.5), scale=(0.02,0.02,0.02), rotate=(0,-0.8,0), fill=True)
print("Cargando y renderizando stop.obj...")
image.loadOBJ("stop.obj", translate=(-1,-0.6,-0.4), scale=(0.05,0.05,0.05), rotate=(0,1,-0.25), fill=True)


image.glFinish()
