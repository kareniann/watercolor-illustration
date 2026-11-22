import math, random, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from svgpathtools import parse_path

S = 2; W_, H_ = 720, 900
FONT = 'Caveat.ttf'; OUTDIR = 'out'
rnd = random.Random(11); nrs = np.random.RandomState(5)
PIG = {  # full-strength pigments, 0..1 RGB; swap for your palette
 'rose': (.86,.40,.36), 'ochre': (.90,.66,.20), 'sage': (.44,.62,.42), 'blue': (.36,.50,.78),
 'pale': (.60,.76,.92), 'skin': (.95,.66,.50), 'grey': (.56,.53,.51), 'brown': (.62,.42,.27),
 'dark': (.28,.25,.25), 'night': (.07,.15,.15),
}
GRAPHITE = np.array([.30,.28,.27], dtype=np.float32)

def _noise(sigma):
    a = gaussian_filter(nrs.rand(H_*S, W_*S).astype(np.float32), sigma, mode='wrap')
    a -= a.min(); a /= a.max() + 1e-9; return a
CLOUD = _noise(34*S)      # big wet blotches
CLOUD2 = _noise(9*S)      # edge wobble, hue drift
GRAIN = np.clip((_noise(.7*S) - .5) * 2.2 + .5, 0, 1)   # paper tooth
FIBER = _noise(2.2*S)

def circ(cx, cy, r): return f'M{cx-r},{cy} A{r},{r} 0 1 1 {cx+r},{cy} A{r},{r} 0 1 1 {cx-r},{cy} Z'
def rect(x, y, w, h): return f'M{x},{y} L{x+w},{y} L{x+w},{y+h} L{x},{y+h} Z'

def sample(d, step=4.0):
    out = []
    for sub in parse_path(d).continuous_subpaths():
        pts = []
        for seg in sub:
            n = max(1, int(math.ceil(seg.length() / step)))
            for i in range(n):
                p = seg.point(i / n); pts.append((p.real, p.imag))
        closed = sub.isclosed()
        if not closed:
            p = sub[-1].end; pts.append((p.real, p.imag))
        out.append((pts, closed))
    return out

def wobble(pts, amp, closed):
    """Smooth hand tremor: two sine bands along the arc length."""
    if len(pts) < 2: return pts
    s = [0.0]
    for i in range(1, len(pts)): s.append(s[-1] + math.dist(pts[i], pts[i-1]))
    L = s[-1] + (math.dist(pts[-1], pts[0]) if closed else 0) or 1.0
    k1 = max(1, round(L / 70)); k2 = max(2, round(L / 26))
    ph = [rnd.random() * 6.283 for _ in range(4)]; a = min(amp, L / 14)
    return [(x + a*(math.sin(k1*6.283*si/L+ph[0]) + .5*math.sin(k2*6.283*si/L+ph[1]))/1.3,
             y + a*(math.sin(k1*6.283*si/L+ph[2]) + .5*math.sin(k2*6.283*si/L+ph[3]))/1.3) for (x, y), si in zip(pts, s)]

class Canvas:
    """Transmittance canvas: 1 = bare paper. Painting multiplies it down."""
    def __init__(self):
        self.T = np.ones((H_*S, W_*S, 3), dtype=np.float32); self.box = None
    def _win(self, pts_list, pad):
        xs = [p[0] for pts in pts_list for p in pts]; ys = [p[1] for pts in pts_list for p in pts]
        x0 = max(0, int((min(xs)-pad)*S)); y0 = max(0, int((min(ys)-pad)*S))
        x1 = min(W_*S, int((max(xs)+pad)*S)+1); y1 = min(H_*S, int((max(ys)+pad)*S)+1)
        if x1 <= x0 or y1 <= y0: return None
        b = self.box
        self.box = (x0,y0,x1,y1) if b is None else (min(b[0],x0), min(b[1],y0), max(b[2],x1), max(b[3],y1))
        return x0, y0, x1, y1
    def _mask(self, win, polys=None, lines=None, width=1):
        x0, y0, x1, y1 = win
        im = Image.new('L', (x1-x0, y1-y0), 0); dr = ImageDraw.Draw(im)
        for pts in polys or []: dr.polygon([(x*S-x0, y*S-y0) for x, y in pts], fill=255)
        for pts in lines or []:
            q = [(x*S-x0, y*S-y0) for x, y in pts]
            dr.line(q, fill=255, width=max(1, int(round(width*S))), joint='curve')
            r = width*S/2
            for (x, y) in (q[0], q[-1]): dr.ellipse((x-r, y-r, x+r, y+r), fill=255)
        return np.asarray(im, dtype=np.float32) / 255.0
    def _paint(self, win, m, color, strength, soft, rimk=.6):
        x0, y0, x1, y1 = win; sl = (slice(y0, y1), slice(x0, x1))
        flat = color == 'night'                      # dark washes: keep them even
        mb = gaussian_filter(m, soft*S)               # 1. soften the mask
        v = mb + .17*(CLOUD2[sl]-.5)*2*min(1, soft/2.2) + .05*(GRAIN[sl]-.5)
        mm = np.clip((v-.40)/.2, 0, 1); mm = mm*mm*(3-2*mm)   # 2. re-threshold: organic edge
        rim = np.clip(mm - gaussian_filter(mm, 3.2*S), 0, 1) * 2.4   # 3. pigment pools at the edge
        dens = mm*((.86+.14*CLOUD[sl]) if flat else (.42+.5*CLOUD[sl])) + rimk*rim*mm   # 4. uneven wash
        dens *= (.93+.14*GRAIN[sl]) if flat else (.80+.40*GRAIN[sl])   # 5. granulation
        dens = np.clip(dens * (.9+.2*FIBER[sl]) * strength, 0, 1)
        c = np.clip(np.array(PIG[color], dtype=np.float32) + (CLOUD2[sl]-.5)[..., None]*np.array([.10,.02,-.08], dtype=np.float32), 0, 1)
        self.T[sl] *= 1 - dens[..., None] * (1 - c)   # 6. multiply onto the paper
    def wash(self, d, color, strength=.62, off=(0,0), soft=None, holes=None):
        subs = sample(d); allp = [p for pts, _ in subs for p in pts]
        sz = max(max(p[0] for p in allp)-min(p[0] for p in allp), max(p[1] for p in allp)-min(p[1] for p in allp))
        k = min(1.0, sz/70); jo = (rnd.uniform(-2.4, 2.4)*k, rnd.uniform(-2.4, 2.4)*k)   # misregistration
        pl = [[(x+off[0]+jo[0], y+off[1]+jo[1]) for x, y in wobble(pts, 2.4*k+.3, True)] for pts, c in subs]
        win = self._win(pl, 14 + 3*(soft or 0))
        if not win: return
        m = self._mask(win, polys=pl)
        if holes:
            hp = [[(x+off[0], y+off[1]) for x, y in wobble(pts, 1.2, True)] for hd in holes for pts, c in sample(hd)]
            m = m * (1 - self._mask(win, polys=hp))
        sf = soft if soft is not None else (.7 + 1.6*k)
        self._paint(win, m, color, strength, sf, rimk=(0 if sf >= 10 else .6))
    def brush(self, d, color, width=5, strength=.62, off=(0,0)):
        ll = [[(x+off[0], y+off[1]) for x, y in wobble(pts, 1.4, c)] for pts, c in sample(d)]
        win = self._win(ll, width+10)
        if win: self._paint(win, self._mask(win, lines=ll, width=width), color, strength, .6+min(1.2, width/8), rimk=.35)
    def pencil(self, d, off=(0,0), alpha=.62, width=1.15):
        subs = sample(d, 3.0)
        for i in range(2):                                   # two passes = sketchy double line
            ll = [[(x+off[0], y+off[1]) for x, y in wobble(pts+([pts[0]] if c else []), 1+.7*i, False)] for pts, c in subs]
            win = self._win(ll, 6)
            if not win: continue
            x0, y0, x1, y1 = win; sl = (slice(y0, y1), slice(x0, x1))
            m = gaussian_filter(self._mask(win, lines=ll, width=width), .45*S)
            tooth = np.clip((GRAIN[sl]-.28)*2.4, 0, 1)           # graphite catches the paper tooth
            a = np.clip(m*1.5, 0, 1) * (.25+.75*tooth) * (.6+.4*CLOUD2[sl]) * alpha * (1 if i == 0 else .55)
            self.T[sl] *= 1 - a[..., None] * (1 - GRAPHITE)
    def text(self, x, y, s, size, rot=0, off=(0,0), alpha=.8, color=None):
        f = ImageFont.truetype(FONT, int(size*S)); bb = f.getbbox(s)
        im = Image.new('L', (bb[2]+8, bb[3]+8), 0); ImageDraw.Draw(im).text((2, 2), s, font=f, fill=255)
        if rot: im = im.rotate(-rot, expand=True, resample=Image.BICUBIC)
        px, py = int((x+off[0])*S), int((y+off[1])*S)          # (x, y) is the top-left corner
        a = np.asarray(im, dtype=np.float32) / 255; h, w = a.shape
        x0, y0, x1, y1 = max(0, px), max(0, py), min(W_*S, px+w), min(H_*S, py+h)
        a = a[y0-py:y1-py, x0-px:x1-px]; sl = (slice(y0, y1), slice(x0, x1))
        a = a * (.45 + .55*np.clip((GRAIN[sl]-.2)*2, 0, 1)) * alpha
        ink = GRAPHITE if color is None else np.array(PIG[color], dtype=np.float32)
        self.T[sl] *= 1 - a[..., None] * (1 - ink)
        b = self.box
        self.box = (x0,y0,x1,y1) if b is None else (min(b[0],x0), min(b[1],y0), max(b[2],x1), max(b[3],y1))
    def save(self, name):
        if self.box is None: return None
        x0, y0, x1, y1 = self.box; x0 -= x0 % S; y0 -= y0 % S; x1 += (-x1) % S; y1 += (-y1) % S
        arr = (np.clip(self.T[y0:y1, x0:x1], 0, 1)*255 + .5).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(OUTDIR, name), 'WEBP', quality=90, method=5)
        return (x0//S, y0//S, (x1-x0)//S, (y1-y0)//S)       # left, top, width, height in stage px

def paper(name='paper.jpg', base=(.985, .978, .962)):
    t = 1 - .016*(GRAIN-.5)[..., None] - .02*(FIBER-.5)[..., None] - .02*(CLOUD-.5)[..., None]
    Image.fromarray((np.clip(np.array(base, dtype=np.float32)*t, 0, 1)*255+.5).astype(np.uint8)).save(os.path.join(OUTDIR, name), 'JPEG', quality=86)
