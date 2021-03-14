import numpy as np
import pygame as pg

class Shaper:
    def __init__(self,dimen,vdimen):
        self.dimen = dimen  #640, 360
        self.vdimen = vdimen
        self.imgarr = np.zeros((dimen[1],dimen[0]))
        self.a = np.stack([np.arange(dimen[0])]*dimen[1],axis = 0)
        self.b = np.stack([np.arange(dimen[1])]*dimen[0],axis = 1)
    def drawCircleSmooth2(self,x,y,r):
        a=(self.a-x).astype(np.float)
        b=-(self.b-y).astype(np.float)
        r2=r**2
        
        #kdl,kdr,kul,kur
        k1 = [a*a+b*b,(a+1)**2+b*b,a*a+(b+1)**2,(a+1)**2+(b+1)**2]
        k2 = [[i<r2,i<=r2,i>=r2,i>r2] for i in k1]
        bdy=np.sign(a)*np.sqrt(np.absolute(r2-b**2))
        buy=np.sign(a+1)*np.sqrt(np.absolute(r2-(b+1)**2))
        alx=np.sign(b)*np.sqrt(np.absolute(r2-a**2))
        arx=np.sign(b+1)*np.sqrt(np.absolute(r2-(a+1)**2))
        
        return self.areaShader(a,b,k2,buy,bdy,alx,arx)
    def drawTriRegionLine(self,x,y,t1):
        a=(self.a-x).astype(np.float)
        b=-(self.b-y).astype(np.float)
        r2 = 0

        #kdl,kdr,kul,kur
        k1 = [\
            a*np.sin(t1)-b*np.cos(t1),\
            (a+1)*np.sin(t1)-b*np.cos(t1),\
            a*np.sin(t1)-(b+1)*np.cos(t1),\
            (a+1)*np.sin(t1)-(b+1)*np.cos(t1)]
        k2 = [[i<r2,i<=r2,i>=r2,i>r2] for i in k1]
        
        if np.absolute(t1) != np.pi and t1 != 0:
            bdy=(b)/np.tan(t1)
            buy=(b+1)/np.tan(t1)
        else:
            bdy=np.zeros_like(b)
            buy=np.zeros_like(b)
        if np.absolute(t1) != np.pi/2:
            alx=(a)*np.tan(t1)
            arx=(a+1)*np.tan(t1)
        else:
            alx=np.zeros_like(a)
            arx=np.zeros_like(a)
        return self.areaShader(a,b,k2,buy,bdy,alx,arx)
    def drawTriRegion(self,x,y,t1,t2):
        br = 125
        t2s = -1*(t2<0)+1*(t2>=0)
        img1 = self.drawTriRegionLine(x,y,t1)
        img2 = self.drawTriRegionLine(x,y,t2-(t2s*np.pi))
        if(t2<t1): img = np.clip(img1+img2,None,1.0)
        else: img = np.clip(img1+img2-1.0,0,None)
        return img
    def drawArc(self,x,y,r1,r2,t1,t2):
        br = 125
        img1 = self.drawCircleSmooth2(x,y,r1)
        img2 = self.drawCircleSmooth2(x,y,r2)
        img3 = self.drawTriRegion(x,y,t1,t2)

        if r1>r2: img = np.clip(img1-img2,0,None)
        else: img = np.clip(img2-img1,0,None)
        img = np.clip(img+img3-1,0,None)
        self.imgarr = self.stitchImg(self.imgarr,(img*br).astype(np.int8))
        #self.imgarr = np.clip(self.imgarr+(img*br).astype(np.int8),0,255)
    def drawPtRegionLine(self,x,y,x1,y1,x2,y2):
        a=(self.a-x).astype(np.float)
        b=-(self.b-y).astype(np.float)
        r2 = 0
        #kdl,kdr,kul,kur
        k1 = [\
            (a-x2)*(y1-y2)-(b-y2)*(x1-x2),\
            (a-x2+1)*(y1-y2)-(b-y2)*(x1-x2),\
            (a-x2)*(y1-y2)-(b-y2+1)*(x1-x2),\
            (a-x2+1)*(y1-y2)-(b-y2+1)*(x1-x2)]
        k2 = [[i<r2,i<=r2,i>=r2,i>r2] for i in k1]
        
        if y1 != y2:
            bdy=(b-y2)*(x1-x2)/(y1-y2)+x2
            buy=(b-y2+1)*(x1-x2)/(y1-y2)+x2
        else:
            bdy=np.zeros_like(b)
            buy=np.zeros_like(b)
        if x1 != x2:
            alx=(a-x2)*(y1-y2)/(x1-x2)+y2
            arx=(a-x2+1)*(y1-y2)/(x1-x2)+y2
        else:
            alx=np.zeros_like(a)
            arx=np.zeros_like(a)
        return self.areaShader(a,b,k2,buy,bdy,alx,arx)
    def drawRect(self,x,y,x1,y1,w,h):
        a=self.a-x
        b=-(self.b-y)
        br = 125
        img = ((a>=x1)*(a<x1+w)*(b>=y1)*(b<y1+h))*1
        self.imgarr = self.stitchImg(self.imgarr,(img*br).astype(np.int8))
    def areaShader(self,a,b,k2,buy,bdy,alx,arx):
        cn = [\
            [[1,1,1,1]],[[0,0,0,3]],[[0,0,3,0]],[[0,3,0,0]],[[3,0,0,0]],\
            [[1,1,2,2]],\
            [[0,2,0,2],[0,3,1,3],[1,3,0,3]],\
            [[2,2,1,1]],\
            [[2,0,2,0],[3,1,3,0],[3,0,3,1]],\
            [[0,3,3,3]],[[3,0,3,3]],[[3,3,0,3]],[[3,3,3,0]]]
        carr = []
        for i in cn:
            g1 = False
            for j in range(len(i)):
                g2 = True
                for k in range(len(i[j])):
                    g2 = g2*k2[k][i[j][k]]
                g1=g1+g2
            carr = carr + [g1]
        carr = [np.where(i) for i in carr]
        farr = [\
            lambda t:img[t]+1,\
            lambda t:(buy[t]-a[t]+(a[t]+1-buy[t])*(arx[t]-b[t]+1)/2),\
            lambda t:(a[t]+1-buy[t]+(buy[t]-a[t])*(alx[t]-b[t]+1)/2),\
            lambda t:(bdy[t]-a[t]+(a[t]+1-bdy[t])*(b[t]+2-arx[t])/2),\
            lambda t:(a[t]+1-bdy[t]+(bdy[t]-a[t])*(b[t]+2-alx[t])/2),\
            lambda t:((alx[t]+arx[t])/2-b[t]),\
            lambda t:((bdy[t]+buy[t])/2-a[t]),\
            lambda t:(b[t]+1-(alx[t]+arx[t])/2),\
            lambda t:(a[t]+1-(bdy[t]+buy[t])/2),\
            lambda t:((alx[t]-b[t])*(bdy[t]-a[t])/2),\
            lambda t:((arx[t]-b[t])*(a[t]+1-bdy[t])/2),\
            lambda t:((b[t]+1-alx[t])*(buy[t]-a[t])/2),\
            lambda t:((b[t]+1-arx[t])*(a[t]+1-buy[t])/2)]
        img = np.zeros_like(a)
        for i,j in zip(carr,farr):
            img[i] = j(i)
        return img
    def drawPoly(self,x,y,xyc):
        br = 125
        #xyc.shape = (n,2)
        xyc = xyc + [xyc[0]]
        img = self.drawPtRegionLine(x,y,xyc[0][0],xyc[0][1],xyc[1][0],xyc[1][1])
        for i in range(1,len(xyc)-1):
            img2 = self.drawPtRegionLine(x,y,xyc[i][0],xyc[i][1],xyc[i+1][0],xyc[i+1][1])
            img = np.clip(img+img2-1.0,0,1.0)
        self.imgarr = self.stitchImg(self.imgarr,(img*br).astype(np.int8))
        #self.imgarr = np.clip(self.imgarr+(img*br).astype(np.int8),0,255)
    def stitchImg(self,img,img1):
        g = (img<img1)
        img[g] = img1[g]
        return img
    def getImg(self):
        return self.gry1_RGB3(self.imgarr)
    def gry1_RGB3(self,img):
        return np.stack([img,np.zeros_like(img),np.zeros_like(img)],axis = 2)
    def getGrad(self):
        dimen = self.dimen


k = (640,360)

s = Shaper(k,k)

#s.drawArcSmooth2(200,150,50,100,0,np.pi/2)
#s.drawRectSmooth(10,10,[[1,1],[1,8],[4,8],[4,1]])
#s.drawRectSmooth(320,180,[[-100,100],[100,-100],[0,-100],[-100,0]])

s.drawRect(200,200,37,25,25,75)
s.drawRect(200,200,12,0,75,25)
s.drawRect(200,200,12,75,75,25)

s.drawRect(300,200,19,0,31,25)
s.drawArc(300+50,200-32,7,32,-np.pi/2,np.pi/2)
s.drawArc(300+50,200-70,6,31,np.pi/2,-np.pi/2)
s.drawRect(300,200,50,76,32,25)

pg.init()
disp = pg.display.set_mode(k)
clk = pg.time.Clock()

surf = pg.transform.scale(pg.surfarray.make_surface(np.transpose(s.getImg(), axes=[1,0,2])),k)

r = True
while r:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            r = False
    disp.fill((255,255,255))
    disp.blit(surf,(0,0))
    pg.display.flip()
    clk.tick(60)