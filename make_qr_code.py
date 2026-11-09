"""Make a styled, permanent (static) QR code.

Style: black square modules, gold square finder rings with a black dot in the
middle, white background.

"Permanent": the link is encoded directly in the image — no redirect or
QR-hosting service that can expire. It works as long as the link itself works.

Needs:  pip install qrcode pillow
Run:    .venv/Scripts/python make_qr_code.py
"""
import qrcode
from PIL import Image, ImageDraw

# ── Settings ──────────────────────────────────────────────────────────────────
URL         = 'https://utdallas.box.com/s/isz5rkeuuxfproyey9av272ecr8sdype'
OUT_PATH    = 'box_qr_code.png'
MODULE_PX   = 40              # size of one QR square in pixels (40 → ~1.3k px image)
BORDER      = 1               # white margin, in squares. Keep 4 (the official minimum):
                              # with the gold rings, 2 stopped scanning when the
                              # image was shown small (tested); 4 scanned at every size.
DARK        = (0, 0, 0)       # data squares + finder dots
GOLD        = (213, 160, 3)   # finder rings (#D5A003, sampled from the example)
BACKGROUND  = (255, 255, 255)
LOGO_PATH   = 'secondary-black.png'   # logo in the middle; None = no logo
LOGO_WIDTH  = 0.40            # logo width as a fraction of the QR code's width.
                              # Keep it ≤ ~0.30: the logo hides squares, and the
                              # code only survives ~30% of them missing.
# ──────────────────────────────────────────────────────────────────────────────


def make_qr(url=URL, out_path=OUT_PATH, module_px=MODULE_PX, border=BORDER,
            dark=DARK, gold=GOLD, background=BACKGROUND,
            logo_path=LOGO_PATH, logo_width=LOGO_WIDTH):
    # Error correction H (30%) keeps it scannable despite the restyled finders.
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, border=0)
    qr.add_data(url)
    qr.make(fit=True)
    m = qr.get_matrix()                      # list of rows of bools, no border
    n = len(m)

    size = (n + 2 * border) * module_px
    img = Image.new('RGB', (size, size), background)
    d = ImageDraw.Draw(img)

    def px(i):                               # module index → pixel coordinate
        return (i + border) * module_px

    # The three 7×7 finder patterns (top-left, top-right, bottom-left).
    finders = [(0, 0), (0, n - 7), (n - 7, 0)]

    def in_finder(r, c):
        return any(fr <= r < fr + 7 and fc <= c < fc + 7 for fr, fc in finders)

    # Data / timing / alignment squares.
    for r in range(n):
        for c in range(n):
            if m[r][c] and not in_finder(r, c):
                d.rectangle([px(c), px(r), px(c + 1) - 1, px(r + 1) - 1], fill=dark)

    # Finders: gold square ring (1 module thick) + black dot (3 modules wide).
    for fr, fc in finders:
        x0, y0 = px(fc), px(fr)
        x1, y1 = px(fc + 7) - 1, px(fr + 7) - 1
        d.rectangle([x0, y0, x1, y1], fill=gold)
        d.rectangle([x0 + module_px, y0 + module_px,
                     x1 - module_px, y1 - module_px], fill=background)
        d.ellipse([px(fc + 2), px(fr + 2), px(fc + 5) - 1, px(fr + 5) - 1], fill=dark)

    # Logo: a white box (snapped to whole squares, so no squares are cut in
    # half) in the center, with the logo pasted on it.
    if logo_path:
        logo = Image.open(logo_path).convert('RGBA')
        logo = logo.crop(logo.getchannel('A').getbbox() or logo.getbbox())  # trim empty edges
        w = int(n * module_px * logo_width)
        h = round(w * logo.height / logo.width)
        logo = logo.resize((w, h), Image.LANCZOS)
        pad = module_px // 2                                   # white space around the logo
        bw = -(-(w + 2 * pad) // module_px)                    # box size in squares (rounded up)
        bh = -(-(h + 2 * pad) // module_px)
        bw += (n - bw) % 2                                     # same parity as n → exactly centered
        bh += (n - bh) % 2
        c0, r0 = (n - bw) // 2, (n - bh) // 2
        d.rectangle([px(c0), px(r0), px(c0 + bw) - 1, px(r0 + bh) - 1], fill=background)
        img.paste(logo, (px(c0) + (bw * module_px - w) // 2,
                         px(r0) + (bh * module_px - h) // 2), logo)
        print(f'Logo: {bw}×{bh} squares covered ({100 * bw * bh / n ** 2:.1f}% of the code)')

    img.save(out_path, dpi=(300, 300))
    print(f'Saved {out_path}  ({size}×{size} px, QR version {qr.version}, {n}×{n} squares)')
    return out_path


if __name__ == '__main__':
    make_qr()
