#!/usr/bin/env python3
"""Icon for the 'Gemma 4 - Chat' launcher.

Family cues borrowed from the existing Gemma launcher (dark green-teal rounded
square, mint rim, ice-blue faceted gem) plus the launcher-set house style
(cyan accent, spaced small-caps label). The gem sits inside a speech bubble:
same brain, plain conversation.
"""
import os
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

S = 2048                      # supersampled canvas; downsampled to 1024 at the end
OUT = os.path.expanduser("~/.local/icons/gemma4_chat.png")

MARGIN = 96
R = int((S - 2 * MARGIN) * 0.225)
BOX = (MARGIN, MARGIN, S - MARGIN, S - MARGIN)

TOP = (26, 74, 68)            # #1a4a44
BOT = (8, 22, 26)             # #08161a
MINT = (126, 224, 176)
CYAN = (74, 214, 196)


def rounded_mask(size, box, radius, scale=1):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle(box, radius=radius, fill=255)
    return m


# ── background: vertical gradient clipped to the rounded square ──────────────
grad = Image.new("RGB", (1, S))
gp = grad.load()
for y in range(S):
    t = y / (S - 1)
    t = t ** 0.85
    gp[0, y] = tuple(int(TOP[i] + (BOT[i] - TOP[i]) * t) for i in range(3))
bg = grad.resize((S, S))

card = Image.new("RGBA", (S, S), (0, 0, 0, 0))
card.paste(bg, (0, 0), rounded_mask((S, S), BOX, R))

# soft radial glow behind the bubble
glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
cx, cy = S // 2, int(S * 0.455)
for i, a in ((620, 26), (470, 30), (330, 34), (210, 36)):
    gd.ellipse((cx - i, cy - i * 0.86, cx + i, cy + i * 0.86), fill=CYAN + (a,))
glow = glow.filter(ImageFilter.GaussianBlur(150))
card.alpha_composite(Image.composite(glow, Image.new("RGBA", (S, S), (0, 0, 0, 0)),
                                     rounded_mask((S, S), BOX, R)))

# top sheen
sheen = Image.new("RGBA", (S, S), (0, 0, 0, 0))
ImageDraw.Draw(sheen).ellipse((MARGIN - 260, MARGIN - 700, S - MARGIN + 260, MARGIN + 520),
                              fill=(200, 255, 240, 26))
sheen = sheen.filter(ImageFilter.GaussianBlur(120))
card.alpha_composite(Image.composite(sheen, Image.new("RGBA", (S, S), (0, 0, 0, 0)),
                                     rounded_mask((S, S), BOX, R)))

# ── speech bubble ───────────────────────────────────────────────────────────
# Body and tail are unioned into ONE mask, then the outline is taken as
# (mask − eroded mask). Stroking the two shapes separately leaves a seam where
# they meet; this way the stroke is a single continuous ring.
BW, BH = 1180, 790
bx0, by0 = (S - BW) // 2, int(S * 0.445 - BH / 2)
bx1, by1 = bx0 + BW, by0 + BH
BR = 235
tail = [(bx0 + 296, by1 - 150), (bx0 + 636, by1 - 150), (bx0 + 214, by1 + 176)]

shape = Image.new("L", (S, S), 0)
shd = ImageDraw.Draw(shape)
shd.rounded_rectangle((bx0, by0, bx1, by1), radius=BR, fill=255)
shd.polygon(tail, fill=255)

STROKE_W = 26
eroded = shape
for _ in range(STROKE_W // 4):
    eroded = eroded.filter(ImageFilter.MinFilter(9))
ring = ImageChops.subtract(shape, eroded)

stroke = Image.new("RGBA", (S, S), (0, 0, 0, 0))
stroke.paste((124, 235, 220, 255), (0, 0), ring)

bub = Image.new("RGBA", (S, S), (0, 0, 0, 0))
bub.paste((120, 240, 225, 26), (0, 0), eroded)          # glassy interior

inner = eroded
for _ in range(9):
    inner = inner.filter(ImageFilter.MinFilter(9))
bub.paste((180, 250, 240, 46), (0, 0), ImageChops.subtract(eroded, inner))

halo = stroke.filter(ImageFilter.GaussianBlur(40))
bub.alpha_composite(halo)
bub.alpha_composite(halo)
bub.alpha_composite(stroke)
card.alpha_composite(bub)

# ── faceted gem inside the bubble ───────────────────────────────────────────
gcx, gcy = S // 2, by0 + int(BH * 0.50)
W, H = 520, 440                      # gem bounding box
tw = W * 0.46                        # table width
ty = gcy - H * 0.42                  # table (top face) y
gy = gcy - H * 0.12                  # girdle y
by = gcy + H * 0.58                  # tip y

gem = Image.new("RGBA", (S, S), (0, 0, 0, 0))
gd = ImageDraw.Draw(gem)
L, Rt = gcx - W / 2, gcx + W / 2
tl, tr = gcx - tw / 2, gcx + tw / 2

gd.polygon([(tl, ty), (tr, ty), (Rt, gy), (L, gy)], fill=(176, 226, 250, 255))   # crown band
gd.polygon([(tl, ty), (L, gy), (gcx - W * 0.17, gy)], fill=(214, 241, 255, 255))  # crown facet L
gd.polygon([(tr, ty), (Rt, gy), (gcx + W * 0.17, gy)], fill=(150, 208, 242, 255))  # crown facet R
gd.polygon([(tl, ty), (tr, ty), (gcx + W * 0.17, gy), (gcx - W * 0.17, gy)],
           fill=(232, 249, 255, 255))                                            # table
gd.polygon([(L, gy), (gcx, by), (gcx - W * 0.17, gy)], fill=(86, 178, 232, 255))  # pavilion L
gd.polygon([(Rt, gy), (gcx, by), (gcx + W * 0.17, gy)], fill=(46, 132, 200, 255))  # pavilion R
gd.polygon([(gcx - W * 0.17, gy), (gcx + W * 0.17, gy), (gcx, by)],
           fill=(112, 200, 244, 255))                                            # pavilion center
gd.line([(gcx - W * 0.17, gy), (gcx, by)], fill=(226, 246, 255, 130), width=5)
gd.line([(gcx + W * 0.17, gy), (gcx, by)], fill=(226, 246, 255, 90), width=5)
gd.line([(L, gy), (Rt, gy)], fill=(240, 252, 255, 150), width=6)

gemglow = gem.filter(ImageFilter.GaussianBlur(46))
card.alpha_composite(gemglow)
card.alpha_composite(gem)

# ── label ───────────────────────────────────────────────────────────────────
def load_font(size):
    for path, idx in (("/System/Library/Fonts/Avenir Next.ttc", 2),
                      ("/System/Library/Fonts/Avenir Next.ttc", 0),
                      ("/System/Library/Fonts/Supplemental/Futura.ttc", 0)):
        try:
            return ImageFont.truetype(path, size, index=idx)
        except Exception:
            continue
    return ImageFont.load_default()


def spaced(draw, text, font, cx, y, fill, tracking):
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = cx - total / 2
    for ch, w in zip(text, widths):
        draw.text((x, y), ch, font=font, fill=fill)
        x += w + tracking


lab = Image.new("RGBA", (S, S), (0, 0, 0, 0))
ld = ImageDraw.Draw(lab)
f = load_font(132)
spaced(ld, "CHAT", f, S // 2, int(S * 0.755), (196, 246, 238, 255), 46)
card.alpha_composite(lab.filter(ImageFilter.GaussianBlur(24)))
card.alpha_composite(lab)

# ── rim: mint edge that matches the sibling Gemma launcher ──────────────────
rim = Image.new("RGBA", (S, S), (0, 0, 0, 0))
rd = ImageDraw.Draw(rim)
rd.rounded_rectangle(BOX, radius=R, outline=MINT + (235,), width=13)
card.alpha_composite(rim.filter(ImageFilter.GaussianBlur(22)))
card.alpha_composite(rim)

inner = Image.new("RGBA", (S, S), (0, 0, 0, 0))
ImageDraw.Draw(inner).rounded_rectangle(
    (BOX[0] + 20, BOX[1] + 20, BOX[2] - 20, BOX[3] - 20), radius=R - 20,
    outline=(255, 255, 255, 34), width=5)
card.alpha_composite(inner)

card.resize((1024, 1024), Image.LANCZOS).save(OUT)
print(OUT)
