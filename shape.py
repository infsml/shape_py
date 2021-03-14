import numpy as np
import pygame as pg
"""
/-----------------------
|viewport
|output_img
|x->
|y
||
|v
|
|
"""

"""
a = [[1,2,3,4,5],
     [1,2,3,4,5]]
"""

class Shaper:
    def __init__(self,dimen,vdimen):
        self.dimen = dimen  #640, 360
        self.vdimen = vdimen
        self.imgarr = np.zeros((dimen[1],dimen[0]))
        self.a = np.stack([np.arange(dimen[0])]*dimen[1],axis = 0)
        self.b = np.stack([np.arange(dimen[1])]*dimen[0],axis = 1)

    def ptInTri(self,tri,p):
        k1=lambda x:(p[0]-x[1][0])*(x[0][1]-x[1][1]) - (p[1]-x[1][1])*(x[0][0]-x[1][0])
        tri = np.array(tri)
        d1 = k1(tri[[0,1]])
        d2 = k1(tri[[1,2]])
        d3 = k1(tri[[2,0]])
        return ~(((d1<0)+(d2<0)+(d3<0))*((d1>0)+(d2>0)+(d3>0)))
    def drawRect(self,xyc):
        #xyc = np.array((4x2)) [0]=x [1]=y
        a=self.a
        b=self.b
        #x-x1  = y-y1
        #x2-x1   y2-y1
        k = self.ptInTri([xyc[0],xyc[1],xyc[2]],[a,b])+self.ptInTri([xyc[0],xyc[2],xyc[3]],[a,b])
        self.imgarr = k*255+(~k)*self.imgarr
    def drawArc(self,xyc):
        #xyc = np.array([x,y,r1,r2,t1,t1])
        a=self.a-xyc[0]
        b=self.b-xyc[1]
        k=np.sqrt(a*a+b*b)
        if(xyc[2]>xyc[3]):
            k = (k<xyc[2])*(k>xyc[3])
        else:
            k = (k>xyc[2])*(k<xyc[3])
        t = np.arctan(b.astype(np.float)/a)
        an = a<0
        bn = b<0
        t = (an*(~bn))*(np.pi+t)+(~(an*(~bn)))*t
        t = an*bn*(t-np.pi) + (~(an*bn))*t
        if(xyc[4]<xyc[5]):
            t = (t<xyc[4])+(t>xyc[5])
        else:
            t = (t<xyc[4])*(t>xyc[5])
        k = k*(~t)
        self.imgarr = k*125+(~k)*self.imgarr
    def drawArcSmooth(self,x,y,r):
        a=(self.a-x).astype(np.float)
        b=-(self.b-y).astype(np.float)
        r2=r**2
        
        kdl=a*a+b*b
        kdr=(a+1)**2+b*b
        kul=a*a+(b+1)**2
        kur=(a+1)**2+(b+1)**2
        
        bdy=np.sign(a)*np.sqrt(np.absolute(r2-b**2))
        buy=np.sign(a+1)*np.sqrt(np.absolute(r2-(b+1)**2))
        alx=np.sign(b)*np.sqrt(np.absolute(r2-a**2))
        arx=np.sign(b+1)*np.sqrt(np.absolute(r2-(a+1)**2))
        i=lambda ax:(ax*np.sqrt(r2-ax*ax)+(r2)*np.arcsin(ax/r))/2
        br = 125
        img = \
            (kdl<=r2)*(kdr<=r2)*(kul<=r2)*(kur<=r2)*br+\
            (kdl<r2)*(kdr<r2)*(kul<r2)*(kur>r2)*br*(buy-a+i(a+1)-i(buy)-(a+1-buy)*b)+\
            (kdl<r2)*(kdr<r2)*(kul>r2)*(kur<r2)*br*((a+1)-buy+i(buy)-i(a)-(buy-a)*b)+\
            (kdl<r2)*(kdr>r2)*(kul<r2)*(kur<r2)*br*(bdy-a+i(a+1)-i(bdy)-(a+1-bdy)*(-(b+1)))+\
            (kdl>r2)*(kdr<r2)*(kul<r2)*(kur<r2)*br*((a+1)-bdy+i(bdy)-i(a)-(bdy-a)*(-(b+1)))+\
            (kdl<=r2)*(kdr<=r2)*(kul>=r2)*(kur>=r2)*br*(i(a+1)-i(a)-b)+\
            ((kdl<r2)*(kdr>=r2)*(kul<r2)*(kur>=r2)+(kdl<r2)*(kdr>r2)*(kul<=r2)*(kur>r2)+(kdl<=r2)*(kdr>r2)*(kul<r2)*(kur>r2))*br*(i(b+1)-i(b)-a)+\
            (kdl>=r2)*(kdr>=r2)*(kul<=r2)*(kur<=r2)*br*(i(a+1)-i(a)+b+1)+\
            ((kdl>=r2)*(kdr<r2)*(kul>=r2)*(kur<r2)+(kdl>r2)*(kdr<=r2)*(kul>r2)*(kur<r2)+(kdl>r2)*(kdr<r2)*(kul>r2)*(kur<=r2))*br*(i(b+1)-i(b)+a+1)+\
            (kdl<r2)*(kdr>r2)*(kul>r2)*(kur>r2)*br*(i(bdy)-i(a)-(bdy-a)*b)+\
            (kdl>r2)*(kdr<r2)*(kul>r2)*(kur>r2)*br*(i(a+1)-i(bdy)-(a+1-bdy)*b)+\
            (kdl>r2)*(kdr>r2)*(kul<r2)*(kur>r2)*br*(i(buy)-i(a)-(buy-a)*(-(b+1)))+\
            (kdl>r2)*(kdr>r2)*(kul>r2)*(kur<r2)*br*(i(a+1)-i(buy)-(a+1-buy)*(-(b+1)))
        self.imgarr = img.astype(np.int8)
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
        
        br = 125
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
    def drawTriRegion(self,x,y,t1,t2):
        br = 125
        t2s = -1*(t2<0)+1*(t2>=0)
        img1 = self.drawTriRegionLine(x,y,t1)
        img2 = self.drawTriRegionLine(x,y,t2-(t2s*np.pi))
        if(t2<t1): img = np.clip(img1+img2,None,1.0)
        else: img = np.clip(img1+img2-1.0,0,None)
        return img
    def drawArcSmooth2(self,x,y,r1,r2,t1,t2):
        br = 125
        img1 = self.drawCircleSmooth2(x,y,r1)
        img2 = self.drawCircleSmooth2(x,y,r2)
        img3 = self.drawTriRegion(x,y,t1,t2)

        if r1>r2: img = np.clip(img1-img2,0,None)
        else: img = np.clip(img2-img1,0,None)
        img = np.clip(img+img3-1,0,None)
        self.imgarr = np.clip(self.imgarr+(img*br).astype(np.int8),0,255)
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
    def drawRectSmooth(self,x,y,xyc):
        br = 125
        #xyc.shape = (n,2)
        xyc = xyc + [xyc[0]]
        img = self.drawPtRegionLine(x,y,xyc[0][0],xyc[0][1],xyc[1][0],xyc[1][1])
        for i in range(1,len(xyc)-1):
            img2 = self.drawPtRegionLine(x,y,xyc[i][0],xyc[i][1],xyc[i+1][0],xyc[i+1][1])
            img = np.clip(img+img2-1.0,0,1.0)
        self.imgarr = np.clip(self.imgarr+(img*br).astype(np.int8),0,255)
    def smudger(self,mask):
        dimen = self.dimen
        l = np.concatenate((mask[:,1:],np.zeros((dimen[1],1),dtype = np.bool)),axis=1)
        r = np.concatenate((np.zeros((dimen[1],1),dtype = np.bool),mask[:,:-1]),axis=1)
        u = np.concatenate((mask[1:,:],np.zeros((1,dimen[0]),dtype = np.bool)),axis=0)
        d = np.concatenate((np.zeros((1,dimen[0]),dtype = np.bool),mask[:-1,:]),axis=0)
        lu = np.concatenate((l[1:,:],np.zeros((1,dimen[0]),dtype = np.bool)),axis=0)
        ld = np.concatenate((np.zeros((1,dimen[0]),dtype = np.bool),l[:-1,:]),axis=0)
        ru = np.concatenate((r[1:,:],np.zeros((1,dimen[0]),dtype = np.bool)),axis=0)
        rd = np.concatenate((np.zeros((1,dimen[0]),dtype = np.bool),r[:-1,:]),axis=0)
        return (mask+l+r+u+d+lu+ld+ru+rd)
    def getImg(self):
        return self.gry1_RGB3(self.imgarr)
    def gry1_RGB3(self,img):
        return np.stack([img,np.zeros_like(img),np.zeros_like(img)],axis = 2)
    def getGrad(self):
        dimen = self.dimen
"""
array([[ 0.5880026 ,  0.78539816,  1.10714872, -1.57079633, -1.10714872,-0.78539816, -0.5880026 ],
       [ 0.32175055,  0.46364761,  0.78539816, -1.57079633, -0.78539816,-0.46364761, -0.32175055],
       [-0.        , -0.        , -0.        ,         nan,  0.        ,0.        ,  0.        ],
       [-0.32175055, -0.46364761, -0.78539816,  1.57079633,  0.78539816,0.46364761,  0.32175055],
       [-0.5880026 , -0.78539816, -1.10714872,  1.57079633,  1.10714872,0.78539816,  0.5880026 ]])
"""

k = (640,360)
k2 = (20,20)
s = Shaper(k,k)

#s.drawRect([[100,100],[120,150],[130,140],[160,170]])

#s.drawArc([400,150,50,100,-np.pi/2,0])
#s.drawArc([300,150,0,130,-np.pi/3,-2*np.pi/3])
#s.drawArc([300,150,100,130,4*np.pi/3,5*np.pi/3])
#s.drawArc([300,150,100,130,-4*np.pi/3,-5*np.pi/3])

#s.drawArcSmooth2(300,150,100)
#s.drawArcSmooth2(10,10,5)
#s.drawTriRegion(310,110,1.2,0.8)

#s.drawArcSmooth2(200,150,50,100,0,np.pi/2)

#s.drawRectSmooth(10,10,[[1,1],[1,8],[4,8],[4,1]])
s.drawRectSmooth(320,180,[[-100,100],[100,-100],[0,-100],[-100,0]])
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