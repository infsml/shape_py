import moviepy.editor as mpe
import numpy as np
import pygame as pg


class MC:
    def __init__(self,img):
        self.imgarr = img
        cshp = np.array(img.shape)
        a = np.stack([np.arange(cshp[1])]*cshp[0],axis=0)
        b = np.stack([np.arange(cshp[0])]*cshp[1],axis=1)
        self.k = np.stack([np.sqrt((a-(cshp[1]/2))**2+(b-(cshp[0]/2))**2)]*3,axis=2)
        self.cshp = cshp/2
        self.mxd = np.sqrt(self.cshp[1]**2+self.cshp[1]**2)
    def fn(self,t):
        return self.imgarr*(self.k<(t/4)*self.mxd)
    def fn2(self,t):
        return np.exp(-self.k*np.exp(0.5*(4-t))/self.mxd)*255


clip = mpe.VideoFileClip("1.mp4")
k=(clip.duration*clip.fps-1)/clip.fps
last_frame = clip.get_frame(k)

last_frame = np.transpose(last_frame,(1,0,2))

print(last_frame.shape)
#last_frame = np.ones((720,1280,3))*255
c = MC(last_frame)

lastfx = mpe.VideoClip(c.fn, duration=4)

lastfx.write_videofile("o.mp4",fps = 30)



"""
pg.init();
screen = pg.display.set_mode((last_frame.shape[0],last_frame.shape[1]))
clock = pg.time.Clock()

surface = pg.surfarray.make_surface(lastfx.get_frame(0))


r = True
while r:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            r = False

    screen.fill((0,0,0))
    screen.blit(surface,(0,0))
    pg.display.flip()
    clock.tick(60)
"""