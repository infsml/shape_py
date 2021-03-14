import gizeh
import numpy as np
import pygame as pg

k = np.array((640,360))
surf = gizeh.Surface(k[0],k[1])

ar = gizeh.arc(100,np.pi/2,0,xy = k/2, fill = (1,0,0))
ar.draw(surf)


pg.init()
disp = pg.display.set_mode(k)
clk = pg.time.Clock()

surf2 = pg.surfarray.make_surface(np.transpose(surf.get_npimage(),axes=(1,0,2)))
r=True

while r:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            r = False
    disp.fill((255,255,255))
    disp.blit(surf2,(0,0))
    pg.display.flip()
    clk.tick(60)