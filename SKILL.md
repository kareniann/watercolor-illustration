---
name: "watercolor-illustration"
description: "Paint hand-drawn watercolor illustrations (optionally animated) for web pages, posters or design canvases: washes with pooled edges, paper grain and loose pencil line, built as layered sprites."
---

# Watercolor illustration

Use this when someone wants an illustration that looks hand-painted in watercolor: soft uneven washes, darker pooled edges, paper tooth, a loose pencil line that doesn't quite register with the colour. It works for single pictures, sets of scenes, and animated scenes on a web page.

Flat translucent SVG fills do **not** read as watercolor, however much you wobble their outlines. The look comes from painting each shape as a raster with pigment physics, then layering the results with multiply blending. This skill is that pipeline.

## The method in one paragraph

Describe the picture as simple vector shapes. A Python script paints each group of shapes into a sprite on white: irregular edge, uneven pigment, pooled rim, paper grain, pencil on top. Sprites are stacked in HTML with `mix-blend-mode: multiply` over a paper texture, so white vanishes and overlapping washes darken like real glazes. Anything that should move gets its own layer and is animated with CSS transforms. Render a preview, look at it, fix, repeat.

## Art direction rules

- **Minimal.** One subject, two or three supporting things, generous empty paper. Cut one element before finishing.
- **Limited palette.** Six to nine pigments for a whole set; one dominant hue per picture; at most one vivid accent.
- **Pale.** Paint at 40-75% strength. Pigment values are the full-strength colour; strength thins them.
- **One soft underpainting** behind the subject: a large irregular blob, strength 0.15-0.25, very soft edge (`soft` 20-30), no pencil line.
- **Loose registration.** Washes are nudged a couple of pixels off their pencil outline on purpose.
- **Pencil is quiet.** Thin, grainy, about 50-60% strength. Use it for faces, small details and a light outline; never a heavy black contour.
- **Lettering is drawn**, in a handwriting font rendered into the sprite so it takes the same grain. Captions outside the picture can be normal HTML type.
- **Occlusion is manual.** Washes are transparent, so a figure in front of a window shows the window through it. Hold the figure out with `holes=` on the washes behind it, the way a painter leaves paper bare.
- **Light things on dark.** Multiply can only darken. For a night scene, paint the dark wash with holes where light objects go, and make glows from a second dark wash with a large, very soft hole.
- **Keep polygons off the stage edge.** Wobble reveals scalloped paper slivers; extend edge-touching shapes 30px beyond the stage.

## Motion rules (if the picture moves)

- Pick one motion language and keep it: three eases only. Sway/loop `cubic-bezier(.45,.05,.55,.95)`, landing `cubic-bezier(.22,1,.36,1)`, departure `cubic-bezier(.5,0,.9,.5)`.
- Each picture gets **one main action** with a start, a peak and a rest; one or two supporting movements that follow slightly late; at most one quiet ambient loop. Never six things at once.
- Squash before a jump or lift-off, stretch during it, squash on landing. Trailing parts (scarves, tails, flags) lag behind.
- Ambient loops are slow (3-5s) and small. Movement should have a cause: wind, current, a cable.
- Animate only `transform`, `opacity` and `clip-path`, on layer containers, never on painted pixels.
- Always honour `prefers-reduced-motion`, and offer a pause.

## Renderer

Needs Python with `numpy`, `scipy`, `Pillow`, `svgpathtools`, and a handwriting font file (e.g. Caveat) for lettering. Save as `rc.py`. Set `W_, H_` to the stage size and `S` to 2 for crisp output.

```python
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
```

## Describing and assembling a scene

Work in stage pixels. Paint back to front. Consecutive static shapes go into one `Canvas` and are saved as one sprite; every moving part gets its own `Canvas` and its own container.

```python
c = Canvas()
c.wash('M150,440 Q140,330 300,318 Q480,290 585,400 Q650,520 565,640 Q420,726 250,684 Q128,624 150,440 Z', 'ochre', .2, soft=26)  # underpainting
c.wash(circ(290, 520, 40), 'skin'); c.pencil(circ(290, 520, 40), alpha=.5)   # wash, then its outline
c.pencil('M272,522 q7,6 14,0 M300,520 q7,6 14,0', alpha=.8)                  # closed eyes
c.brush('M190,690 Q360,742 536,686', 'brown', 6)                             # a painted stroke
left, top, w, h = c.save('scene-00.webp')
```

HTML, back to front:

```html
<div class="scene" style="position:relative;width:720px;height:900px;overflow:hidden;background:#fbf9f5 url(paper.jpg) center/cover">
  <img src="scene-00.webp" alt="" style="position:absolute;left:Lpx;top:Tpx;width:Wpx;height:Hpx;mix-blend-mode:multiply">
  <div class="rock" style="position:absolute;left:0;top:0;width:720px;height:900px;mix-blend-mode:multiply">
    <img src="scene-01.webp" alt="" style="position:absolute;...;mix-blend-mode:multiply">
  </div>
</div>
```

```css
.rock{transform-origin:360px 716px;animation:rock 3.2s cubic-bezier(.45,.05,.55,.95) infinite alternate}
@keyframes rock{from{transform:rotate(-2.6deg)}to{transform:rotate(2.6deg)}}
.paused *{animation-play-state:paused!important}
@media (prefers-reduced-motion:reduce){.scene *{animation:none!important}}
```

Rules that bite:

- **Every** sprite and **every** container needs `mix-blend-mode: multiply`. An animated container is its own stacking context; without multiply on it, its children's white boxes show.
- If an ancestor of the scene has a `transform` (for scaling), put `mix-blend-mode: multiply` on that element too, or the paper behind it stops showing through.
- Containers are full-stage boxes, so `transform-origin` is written in stage pixels.
- To scale a fixed stage to any width: set `transform: scale(k)` with `transform-origin: 0 0`, where `k = containerWidth / stageWidth` (a `ResizeObserver` does it in three lines).
- For fewer requests, pack a scene's sprites into one sheet and show each with a `<div>` using `background-position` and `background-size`; leave 8px of white padding between sprites.
- A reveal (typing, drawing on) is `clip-path: inset(...)` animated on the container.

## Always look at it

Never ship unseen. Write a plain HTML preview with local file paths, open it headless, freeze the animations at a few moments and screenshot:

```python
await page.evaluate('(t)=>document.getAnimations().forEach(a=>{a.pause();a.currentTime=t*1000})', 1.6)
await page.screenshot(path='frame.png')
```

Check, in this order: does it read as paint (uneven, pooled edges, grain) rather than flat shapes; is anything showing through that should be hidden; is lettering on its object and inside the stage; are edges of the stage clean; does each frozen moment of the motion make sense. Then remove one element.

## Tuning knobs

| Want | Change |
|---|---|
| Paler / stronger colour | `strength` (0.15 underpainting, 0.5 normal, 0.8 accent) |
| Wetter, blurrier edge | `soft` up (2-6 soft, 20+ wet-in-wet; rim disappears at 10+) |
| Crisper small shapes | `soft` 0.5-0.8 |
| More granulation | widen the `.80+.40*GRAIN` term |
| Blotchier wash | widen the `.42+.5*CLOUD` term |
| Stronger pooled edge | `rimk` |
| Rougher hand | `wobble` amplitudes |
| Fainter pencil | `alpha` on `pencil` |
| Warmer or cooler paper | `base` in `paper()` |

## Common mistakes

- Drawing with outlines first and colouring in: start from washes, add pencil last and sparingly.
- Too many colours, too many objects, too many things moving.
- Perfect circles and straight edges left unwobbled, or pencil that sits exactly on the wash edge.
- Dots or texture marks on fruit and faces that read as buttons; leave them out.
- Forgetting that white objects (snow, paper, clouds) cannot be painted: leave bare paper, or tint them very pale blue.