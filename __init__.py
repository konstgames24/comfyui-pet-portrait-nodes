"""Nodes for pastel pet portraits.

Pet Portrait Layout: places a masked subject on a canvas with a fixed amount of empty space
above it, the chest running past the bottom edge, and the subject's bounding box centred.
Pet Head Portrait Layout: the same for an untouched photo, framed by the animal's head box.
Pet Restore Colors: brushwork from a painting, colours from the photo it was painted from.
Pet Flatten Background: exact flat background that keeps the painted fur along the outline.
Pet Pastel Background: solid pastel canvas plus its name for the prompt.
Experimental: Pet Portrait Layout (padded) - space below the bust and a brushy fade at its bottom;
Pet Color Palette - recolours the animal with a gradient-map palette;
Pet Pastel Palette - 24 pastel background tones, chosen by hand, at random or to suit the coat;
Pet Split Animals - one crop per animal, so a photo of several pets gives several portraits;
Pet Face Mask - weaker painterly pass on the face, so the eyes and nose keep their likeness;
Pet Group Layout - all animals from one photo side by side on one image, or separate portraits."""
import colorsys
import re

import torch
import torch.nn.functional as F


class PetPortraitLayout:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "canvas": ("IMAGE", {"tooltip": "Background the subject is placed on; sets the output size."}),
            "top_space": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.8, "step": 0.01,
                                    "tooltip": "Share of the canvas height left empty above the subject."}),
            "bottom_overflow": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.3, "step": 0.01,
                                          "tooltip": "How far the subject runs past the bottom edge, as a share of the canvas height."}),
            "max_width": ("FLOAT", {"default": 0.9, "min": 0.2, "max": 1.0, "step": 0.01,
                                    "tooltip": "Largest allowed subject width as a share of the canvas width."}),
            "edge_softness": ("INT", {"default": 4, "min": 0, "max": 32,
                                      "tooltip": "Soft edge around the subject, in canvas pixels; 0 keeps the hard cut-out."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "layout"
    CATEGORY = "image/transform"

    def layout(self, image, mask, canvas, top_space, bottom_overflow, max_width, edge_softness):
        out = []
        for b in range(image.shape[0]):
            img = image[b]
            m = mask[min(b, mask.shape[0] - 1)].to(img)
            if m.shape != img.shape[:2]:
                m = F.interpolate(m[None, None], size=img.shape[:2], mode="bilinear")[0, 0]
            cv = canvas[min(b, canvas.shape[0] - 1)].to(img).clone()
            ch, cw = cv.shape[:2]

            ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
            if len(ys) == 0:
                out.append(cv)
                continue
            y0, y1 = ys.min().item(), ys.max().item() + 1
            x0, x1 = xs.min().item(), xs.max().item() + 1

            top = top_space * ch
            bottom = ch * (1 + bottom_overflow)
            s = min((bottom - top) / (y1 - y0), max_width * cw / (x1 - x0))
            out.append(_paste(cv, img, m, s, round(cw / 2 - (x0 + x1) / 2 * s), round(bottom - y1 * s), edge_softness))
        return (torch.stack(out),)


class PetPortraitLayoutPadded:
    """Like Pet Portrait Layout, but the bust ends inside the frame: there is empty space below it too,
    and its bottom dissolves into the background in uneven, brush-like strokes instead of a straight cut."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "canvas": ("IMAGE", {"tooltip": "Background the subject is placed on; sets the output size."}),
            "top_space": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.8, "step": 0.01,
                                    "tooltip": "Share of the canvas height left empty above the subject."}),
            "bottom_padding": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.4, "step": 0.01,
                                         "tooltip": "Empty space below the subject, as a share of the canvas height."}),
            "bottom_fade": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.8, "step": 0.01,
                                      "tooltip": "Lower share of the subject that fades into the background."}),
            "max_width": ("FLOAT", {"default": 0.86, "min": 0.2, "max": 1.0, "step": 0.01,
                                    "tooltip": "Largest allowed subject width as a share of the canvas width."}),
            "edge_softness": ("INT", {"default": 4, "min": 0, "max": 32,
                                      "tooltip": "Soft edge around the subject, in canvas pixels; 0 keeps the hard cut-out."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "tooltip": "Shape of the brushy fade line."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "layout"
    CATEGORY = "image/transform"

    def layout(self, image, mask, canvas, top_space, bottom_padding, bottom_fade, max_width, edge_softness, seed):
        img = image[0]
        m = mask[0].to(img)
        if m.shape != img.shape[:2]:
            m = F.interpolate(m[None, None], size=img.shape[:2], mode="bilinear")[0, 0]
        cv = canvas[0].to(img).clone()
        ch, cw = cv.shape[:2]

        ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
        if len(ys) == 0:
            return (cv[None], torch.zeros(1, ch, cw))
        y0, y1 = ys.min().item(), ys.max().item() + 1
        x0, x1 = xs.min().item(), xs.max().item() + 1

        if bottom_fade > 0:
            m = m * _brush_fade(m.shape, y0, y1, bottom_fade, seed).to(m)

        top = top_space * ch
        bottom = ch * (1 - bottom_padding)
        s = min((bottom - top) / (y1 - y0), max_width * cw / (x1 - x0))
        ox, oy = round(cw / 2 - (x0 + x1) / 2 * s), round(bottom - y1 * s)
        out = _paste(cv, img, m, s, ox, oy, edge_softness)
        alpha = _paste(torch.zeros(ch, cw, 1).to(img), torch.ones_like(img[..., :1]), m, s, ox, oy, edge_softness)
        return (out[None], alpha[None, ..., 0])


GROUP = "group (all on one image)"
SEPARATE = "separate portraits"


class PetGroupLayout:
    """Takes the list of painted animals (one per Pet Split Animals crop). "separate" lays each out
    on its own canvas like Pet Portrait Layout (padded); "group" puts them all side by side on one
    wider canvas, equal height, the middle ones in front, each bust fading out at the bottom."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "canvas": ("IMAGE", {"tooltip": "Background; the group canvas takes the first one's colour and height."}),
            "mode": ([GROUP, SEPARATE],),
            "top_space": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.8, "step": 0.01}),
            "bottom_padding": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.4, "step": 0.01}),
            "bottom_fade": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.8, "step": 0.01}),
            "max_width": ("FLOAT", {"default": 0.86, "min": 0.2, "max": 1.0, "step": 0.01,
                                    "tooltip": "Share of the canvas width the animals may take."}),
            "overlap": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.6, "step": 0.01,
                                  "tooltip": "How much neighbours overlap, as a share of their width (group only)."}),
            "max_aspect": ("FLOAT", {"default": 1.6, "min": 0.5, "max": 3.0, "step": 0.05,
                                     "tooltip": "Widest group canvas, as width / height."}),
            "edge_softness": ("INT", {"default": 4, "min": 0, "max": 32}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "canvas")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "layout"
    CATEGORY = "image/transform"

    def layout(self, image, mask, canvas, mode, top_space, bottom_padding, bottom_fade, max_width, overlap,
               max_aspect, edge_softness, seed):
        mode, top_space, bottom_padding, bottom_fade = mode[0], top_space[0], bottom_padding[0], bottom_fade[0]
        max_width, overlap, max_aspect, edge_softness, seed = max_width[0], overlap[0], max_aspect[0], edge_softness[0], seed[0]
        n = len(image)
        canvases = [canvas[min(i, len(canvas) - 1)] for i in range(n)]
        if mode == SEPARATE or n == 1:
            single = PetPortraitLayoutPadded()
            outs = [single.layout(image[i], mask[min(i, len(mask) - 1)], canvases[i], top_space, bottom_padding,
                                  bottom_fade, max_width, edge_softness, seed + i) for i in range(n)]
            return ([o[0] for o in outs], [o[1] for o in outs], canvases)

        ch = canvas[0].shape[1]
        top, bottom = top_space * ch, ch * (1 - bottom_padding)
        parts = []
        for i in range(n):
            img = image[i][0]
            m = mask[min(i, len(mask) - 1)][0].to(img)
            if m.shape != img.shape[:2]:
                m = F.interpolate(m[None, None], size=img.shape[:2], mode="bilinear")[0, 0]
            ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
            if len(ys) == 0:
                continue
            y0, y1, x0, x1 = ys.min().item(), ys.max().item() + 1, xs.min().item(), xs.max().item() + 1
            if bottom_fade > 0:
                m = m * _brush_fade(m.shape, y0, y1, bottom_fade, seed + i).to(m)
            parts.append([img, m, y0, y1, x0, x1, (bottom - top) / (y1 - y0)])
        widths = [(p[5] - p[4]) * p[6] for p in parts]
        total = sum(widths) - overlap * sum(widths[1:])
        # wider canvas for a group, rounded to 64 px for the latent; shrink the animals if still too wide
        cw = min(max(canvas[0].shape[2], total / max_width), ch * max_aspect)
        cw = max(64, int(round(cw / 64)) * 64)
        k = min(1.0, max_width * cw / total)
        pixel = canvas[0][0, 0, 0].to(parts[0][0])
        cv = pixel.expand(ch, cw, 3).clone()
        alpha = torch.zeros(ch, cw, 1).to(cv)
        x = (cw - total * k) / 2
        placed = []
        for p, wd in zip(parts, widths):
            placed.append((p, x))
            x += wd * k * (1 - overlap)
        # outer animals first, so the middle ones overlap them
        mid = (len(placed) - 1) / 2
        order = sorted(range(len(placed)), key=lambda i: -abs(i - mid))
        for (img, m, y0, y1, x0, x1, s), px in (placed[i] for i in order):
            s *= k
            ox, oy = round(px - x0 * s), round(bottom - y1 * s)
            cv = _paste(cv, img, m, s, ox, oy, edge_softness)
            alpha = torch.maximum(alpha, _paste(torch.zeros(ch, cw, 1).to(cv), torch.ones_like(img[..., :1]), m, s, ox, oy, edge_softness))
        return ([cv[None]], [alpha[None, ..., 0]], [pixel.expand(1, ch, cw, 3).clone()])


def _brush_fade(shape, y0, y1, fade, seed):
    """Alpha that goes from 1 to 0 over the lowest `fade` share of rows y0..y1, with a ragged,
    stroke-like end line: each column fades at a slightly different height."""
    h, w = shape
    g = torch.Generator().manual_seed(seed)

    def noise(knots):
        v = torch.rand(1, 1, knots, generator=g) * 2 - 1
        return F.interpolate(v, size=w, mode="linear", align_corners=True)[0, 0]

    jitter = 0.3 * noise(9) + 0.15 * noise(70)
    length = max(1.0, fade * (y1 - y0))
    rows = torch.arange(h, dtype=torch.float32)[:, None]
    t = (rows - (y1 - length)) / length + jitter[None, :]
    t = t.clamp(0, 1)
    return 1 - t * t * (3 - 2 * t)


def _soften(m, r):
    """Shrink the mask by r px, then blur it by r px: a soft edge that stays inside the old outline,
    so no fringe of the original background is blended in."""
    k = 2 * r + 1
    m = -F.max_pool2d(-m[None, None], k, 1, r)
    for _ in range(2):
        m = F.avg_pool2d(m, k, 1, r, count_include_pad=False)
    return m[0, 0]


def _paste(cv, img, m, s, ox, oy, softness=0):
    """Scale img/mask by s and alpha-blend them onto canvas cv with the top-left corner at (ox, oy)."""
    ch, cw = cv.shape[:2]
    nh, nw = max(1, round(img.shape[0] * s)), max(1, round(img.shape[1] * s))
    simg = F.interpolate(img.movedim(-1, 0)[None], size=(nh, nw), mode="bicubic", antialias=True)[0].movedim(0, -1).clamp(0, 1)
    sm = F.interpolate(m[None, None], size=(nh, nw), mode="bilinear")[0, 0].clamp(0, 1)
    if softness > 0:
        sm = _soften(sm, softness)
    ty0, tx0 = max(oy, 0), max(ox, 0)
    ty1, tx1 = min(oy + nh, ch), min(ox + nw, cw)
    if ty1 > ty0 and tx1 > tx0:
        src = simg[ty0 - oy:ty1 - oy, tx0 - ox:tx1 - ox]
        a = sm[ty0 - oy:ty1 - oy, tx0 - ox:tx1 - ox, None]
        cv[ty0:ty1, tx0:tx1] = src * a + cv[ty0:ty1, tx0:tx1] * (1 - a)
    return cv


class PetHeadPortraitLayout:
    """Places the cut-out animal from a photo as a portrait, keeping its original pose: the head box
    (Qwen3-VL "bbox_2d" in 0-1000 coordinates) is centred horizontally, its top sits at top_space and
    the body runs past the bottom edge."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "head_box": ("STRING", {"forceInput": True, "tooltip": 'Text containing [x1, y1, x2, y2] in 0-1000 coordinates.'}),
            "canvas": ("IMAGE", {"tooltip": "Background the animal is placed on; sets the output size."}),
            "top_space": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.7, "step": 0.01,
                                    "tooltip": "Share of the canvas height left empty above the head."}),
            "head_height": ("FLOAT", {"default": 0.34, "min": 0.1, "max": 0.8, "step": 0.01,
                                      "tooltip": "Head height (ears to chin) as a share of the canvas height."}),
            "max_head_width": ("FLOAT", {"default": 0.7, "min": 0.2, "max": 1.0, "step": 0.01}),
            "bottom_overflow": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.3, "step": 0.01}),
            "edge_fade": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.2, "step": 0.01,
                                    "tooltip": "Fade the animal out where the photo frame cuts it on the left, right or top."}),
            "edge_softness": ("INT", {"default": 4, "min": 0, "max": 32,
                                      "tooltip": "Soft edge around the animal, in canvas pixels; 0 keeps the hard cut-out."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "layout"
    CATEGORY = "image/transform"

    def layout(self, image, mask, head_box, canvas, top_space, head_height, max_head_width, bottom_overflow, edge_fade, edge_softness):
        img = image[0]
        h, w = img.shape[:2]
        m = mask[0].to(img)
        if m.shape != img.shape[:2]:
            m = F.interpolate(m[None, None], size=(h, w), mode="bilinear")[0, 0]
        cv = canvas[0].to(img).clone()
        ch, cw = cv.shape[:2]

        ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
        if len(ys) == 0:
            return (cv[None],)
        body_bottom = ys.max().item() + 1

        box = re.search(r"\[([^\]]*)\]", head_box)  # "bbox_2d" itself contains a digit
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", box.group(1))[:4]] if box else []
        if len(nums) == 4 and nums[2] > nums[0] and nums[3] > nums[1]:
            hx0, hy0, hx1, hy1 = nums[0] * w / 1000, nums[1] * h / 1000, nums[2] * w / 1000, nums[3] * h / 1000
        else:
            # No usable head box: take the top of the animal, as tall as it is wide.
            x0, x1, y0 = xs.min().item(), xs.max().item() + 1, ys.min().item()
            hx0, hx1, hy0 = x0, x1, y0
            hy1 = min(y0 + (x1 - x0) * 0.8, body_bottom)

        if edge_fade > 0:
            r = max(1, round(edge_fade * min(h, w)))
            ramp = torch.linspace(0, 1, r, device=m.device, dtype=m.dtype)
            m = m.clone()
            m[:r] *= ramp[:, None]
            m[:, :r] *= ramp[None, :]
            m[:, -r:] *= ramp.flip(0)[None, :]

        top = top_space * ch
        bottom = ch * (1 + bottom_overflow)
        s = head_height * ch / (hy1 - hy0)
        # When the photo ends too high the chest would float: first grow the animal (up to the
        # head width limit), then move it down, which only adds empty space above.
        s = max(s, (bottom - top) / (body_bottom - hy0))
        s = min(s, max_head_width * cw / (hx1 - hx0))
        ox = round(cw / 2 - (hx0 + hx1) / 2 * s)
        oy = round(max(top - hy0 * s, bottom - body_bottom * s))
        return (_paste(cv, img, m, s, ox, oy, edge_softness)[None],)


class PetFlattenBackground:
    """Replaces the background with the exact flat colour, but only away from the animal: the mask is
    grown and feathered so the painted wisps of fur along the outline stay and no cut-out edge appears."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "canvas": ("IMAGE", {"tooltip": "Flat background of the same size as the image."}),
            # A wide band also keeps whatever the models painted next to the animal (e.g. a shadow).
            "protect": ("INT", {"default": 4, "min": 0, "max": 128,
                                "tooltip": "Pixels around the animal kept as painted (fur wisps)."}),
            "feather": ("INT", {"default": 8, "min": 1, "max": 128,
                                "tooltip": "Width of the soft transition into the flat background, in pixels."}),
        }, "optional": {
            "extra_mask": ("MASK", {"tooltip": "Also kept as painted, e.g. the faded bottom of the bust from the layout."}),
            # SDXL likes to sign the empty corner of a vignette; capping by the layout keeps the fade clean.
            "limit": ("INT", {"default": 0, "min": 0, "max": 256,
                              "tooltip": "If > 0, nothing further than this many pixels outside extra_mask is kept, "
                                         "and its faded part stays faded."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "flatten"
    CATEGORY = "image/postprocessing"

    def flatten(self, image, mask, canvas, protect, feather, extra_mask=None, limit=0):
        m = mask.to(image)[:, None]
        if m.shape[2:] != image.shape[1:3]:
            m = F.interpolate(m, size=image.shape[1:3], mode="bilinear")
        if extra_mask is not None:
            e = F.interpolate(extra_mask.to(image)[:, None], size=image.shape[1:3], mode="bilinear")
            m = torch.maximum(m, e)
            if limit > 0:
                m = torch.minimum(m, F.max_pool2d(e, 2 * limit + 1, 1, limit))
        if protect > 0:
            m = F.max_pool2d(m, 2 * protect + 1, 1, protect)
        for _ in range(2):
            m = F.avg_pool2d(m, 2 * feather + 1, 1, feather, count_include_pad=False)
        a = m.movedim(1, -1)
        cv = canvas.to(image)
        if cv.shape[1:3] != image.shape[1:3]:
            cv = F.interpolate(cv.movedim(-1, 1), size=image.shape[1:3], mode="nearest").movedim(1, -1)
        return (image * a + cv * (1 - a),)


_TO_YCBCR = torch.tensor([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])


class PetRestoreColors:
    """Keeps the brushwork (brightness) of a painting but takes the colours from the photo it was
    painted from: the edit model tends to invent coat colours, e.g. turning a silver cat brown."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "painting": ("IMAGE",),
            "photo": ("IMAGE", {"tooltip": "The framed photo with the same layout as the painting."}),
            "color_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "photo_brightness": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "Share of the photo's brightness mixed back in to recover fine detail."}),
            "color_blur": ("INT", {"default": 8, "min": 1, "max": 64,
                                   "tooltip": "Colours are taken from the photo shrunk by this factor, so small misalignments do not show."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore"
    CATEGORY = "image/postprocessing"

    def restore(self, painting, photo, color_strength, photo_brightness, color_blur):
        h, w = painting.shape[1:3]
        photo = photo.movedim(-1, 1).to(painting)
        if photo.shape[2:] != (h, w):
            photo = F.interpolate(photo, size=(h, w), mode="bilinear", antialias=True)
        soft = F.interpolate(F.interpolate(photo, size=(max(1, h // color_blur), max(1, w // color_blur)), mode="area"),
                             size=(h, w), mode="bilinear")
        m = _TO_YCBCR.to(painting)
        paint_ycc = painting @ m.T
        photo_ycc = photo.movedim(1, -1) @ m.T
        soft_ycc = soft.movedim(1, -1) @ m.T
        y = paint_ycc[..., :1] * (1 - photo_brightness) + photo_ycc[..., :1] * photo_brightness
        cbcr = paint_ycc[..., 1:] * (1 - color_strength) + soft_ycc[..., 1:] * color_strength
        out = torch.cat([y, cbcr], dim=-1) @ torch.linalg.inv(m).T
        return (out.clamp(0, 1),)


class PetMatchColors:
    """Gives a newly painted animal the coat colours of the photo. The pose differs, so pixel-wise
    colour restore is impossible; instead the colour statistics (mean and spread of the chroma, and
    optionally brightness) of the animal are matched to the photo's animal: a ginger-tinted grey cat
    turns grey again, while the brushwork and markings stay."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "painting": ("IMAGE",),
            "painting_mask": ("MASK",),
            "photo": ("IMAGE",),
            "photo_mask": ("MASK",),
            "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            "brightness": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                     "tooltip": "How much of the photo's overall brightness and contrast to take as well."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match"
    CATEGORY = "image/postprocessing"

    def match(self, painting, painting_mask, photo, photo_mask, strength, brightness):
        m = _TO_YCBCR.to(painting)
        p = painting[0] @ m.T
        q = photo[0].to(painting) @ m.T
        pm = _fit_mask(painting_mask, p.shape[:2]).to(p) > 0.5
        qm = _fit_mask(photo_mask, q.shape[:2]).to(p) > 0.5
        if pm.sum() < 16 or qm.sum() < 16:
            return (painting,)
        ps, qs = p[pm], q[qm]
        mean_p, std_p = ps.mean(0), ps.std(0) + 1e-4
        mean_q, std_q = qs.mean(0), qs.std(0) + 1e-4
        matched = (p - mean_p) / std_p * std_q + mean_q
        k = torch.tensor([brightness, 1.0, 1.0]).to(p) * strength
        out = p + (matched - p) * k
        a = pm.float()[..., None]
        out = out * a + p * (1 - a)
        return ((out @ torch.linalg.inv(m).T).clamp(0, 1)[None],)


class PetCropToBox:
    """Crops the image to a Qwen3-VL box (0-1000 coordinates) with some margin and scales it to a
    square, e.g. a close-up of the face as an extra reference for the painter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "box": ("STRING", {"forceInput": True, "tooltip": "Text containing [x1, y1, x2, y2] in 0-1000 coordinates."}),
            "pad": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            "size": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 16}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "image/transform"

    def crop(self, image, box, pad, size):
        h, w = image.shape[1:3]
        found = re.search(r"\[([^\]]*)\]", box)
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", found.group(1))[:4]] if found else []
        if len(nums) == 4 and nums[2] > nums[0] and nums[3] > nums[1]:
            x0, y0, x1, y1 = nums[0] * w / 1000, nums[1] * h / 1000, nums[2] * w / 1000, nums[3] * h / 1000
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        crop = image[:1, y0:y1, x0:x1].movedim(-1, 1)
        return (F.interpolate(crop, size=(size, size), mode="bicubic", antialias=True).movedim(1, -1).clamp(0, 1),)


def _fit_mask(mask, size):
    m = mask[:1].float()
    if m.shape[1:] != tuple(size):
        m = F.interpolate(m[None], size=size, mode="bilinear")[0]
    return m[0]


# Gradient maps from shadows to highlights. The animal keeps its own brightness (all the brushwork),
# only the hues are taken from the palette, so the result reads as one colour scheme with the background.
PALETTES = {
    "original colors": None,
    "match background": None,  # built from the canvas colour
    "warm sepia": [0x3B2A20, 0x8A6A4F, 0xD9BF9C, 0xFBF3E6],
    "terracotta": [0x3A1F1A, 0x9A4E3A, 0xE0A084, 0xFBEEE4],
    "dusty rose": [0x3A2630, 0x8E5A68, 0xD9A8B0, 0xFBF0F0],
    "sage olive": [0x262B20, 0x5F6B4A, 0xB5BF9B, 0xF4F6EC],
    "blue hour": [0x1E2433, 0x4F5F80, 0xA9B6CF, 0xF2F4FA],
    "ochre charcoal": [0x262321, 0x6B5A3E, 0xD2A95A, 0xFAF2DF],
    "lavender dusk": [0x2A2433, 0x6E5F86, 0xBDB0D4, 0xF6F2FA],
}


def _rgb(c):
    return [((c >> 16) & 0xFF) / 255.0, ((c >> 8) & 0xFF) / 255.0, (c & 0xFF) / 255.0]


class PetColorPalette:
    """Recolours the animal with a palette (a gradient map from dark to light) while keeping its
    brightness, so fur detail survives; strength 0 leaves the colours untouched."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "canvas": ("IMAGE", {"tooltip": "The background; used by \"match background\"."}),
            "palette": (list(PALETTES),),
            "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "How much of the palette's hue replaces the animal's own colours."}),
            "tone": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                               "tooltip": "How much of the palette's light and dark values is mixed in as well."}),
        }, "optional": {
            "mask": ("MASK", {"tooltip": "Only this area is recoloured (the background stays exact)."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/postprocessing"

    def apply(self, image, canvas, palette, strength, tone, mask=None):
        if palette == "original colors" or strength == 0:
            return (image,)
        if palette == "match background":
            bg = canvas[0, 0, 0].float().cpu()
            ycc = bg @ _TO_YCBCR.T
            # same hue as the background, more saturated towards the shadows
            stops = [torch.cat([y, ycc[1:] * k]) @ torch.linalg.inv(_TO_YCBCR).T
                     for y, k in ((torch.tensor([0.16]), 3.0), (torch.tensor([0.45]), 3.0), (torch.tensor([0.78]), 2.0), (torch.tensor([0.97]), 0.6))]
            stops = torch.stack(stops).clamp(0, 1)
        else:
            stops = torch.tensor([_rgb(c) for c in PALETTES[palette]])
        stops = stops.to(image)

        m = _TO_YCBCR.to(image)
        ycc = image @ m.T
        y = ycc[..., 0].clamp(0, 1)
        # piecewise-linear gradient map over the stops, evenly spaced in brightness
        pos = y * (len(stops) - 1)
        i = pos.floor().clamp(max=len(stops) - 2).long()
        f = (pos - i)[..., None]
        mapped = stops[i] * (1 - f) + stops[i + 1] * f
        mycc = mapped @ m.T
        out_y = ycc[..., :1] * (1 - tone) + mycc[..., :1] * tone
        out_c = ycc[..., 1:] * (1 - strength) + mycc[..., 1:] * strength
        out = (torch.cat([out_y, out_c], dim=-1) @ torch.linalg.inv(m).T).clamp(0, 1)
        if mask is not None:
            a = mask.to(image)[:, None]
            if a.shape[2:] != image.shape[1:3]:
                a = F.interpolate(a, size=image.shape[1:3], mode="bilinear")
            a = a.movedim(1, -1)
            out = out * a + image * (1 - a)
        return (out,)


PASTELS = {
    "pale warm cream": 0xF3E6CC,
    "soft dusty pink": 0xEBD5D0,
    "pale sage green": 0xDCE3D2,
    "soft powder blue": 0xD6E0E6,
    "warm pale apricot": 0xF2DCC8,
    "pale lavender": 0xE2DCE8,
    "light butter yellow": 0xF3EBCB,
    "soft greige": 0xE6E1DA,
}


class PetPastelBackground:
    """Solid pastel canvas plus its name, so one dropdown drives both the prompt and the fill."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "color": (list(PASTELS),),
            "width": ("INT", {"default": 896, "min": 64, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 1120, "min": 64, "max": 4096, "step": 16}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("canvas", "color_name")
    FUNCTION = "make"
    CATEGORY = "image"

    def make(self, color, width, height):
        rgb = PASTELS[color]
        pixel = torch.tensor([(rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF], dtype=torch.float32) / 255.0
        return (pixel.expand(1, height, width, 3).clone(), color)


# Extended background palette, grouped; the names go into the prompt, so they describe the colour.
PASTEL_GROUPS = {
    "warm": {
        "pale warm cream": 0xF3E6CC,
        "pale vanilla": 0xF6EFD9,
        "light butter yellow": 0xF3EBCB,
        "warm pale apricot": 0xF2DCC8,
        "soft peach": 0xF5D9C6,
        "pale coral pink": 0xF4D4CC,
        "soft blush pink": 0xF2DADA,
        "soft dusty pink": 0xEBD5D0,
        "light terracotta blush": 0xE9CDBF,
    },
    "cool": {
        "pale pistachio": 0xE4EAD0,
        "pale sage green": 0xDCE3D2,
        "soft mint": 0xE0EDE4,
        "soft duck egg blue": 0xD5E5E2,
        "pale sky blue": 0xDCE8F0,
        "soft powder blue": 0xD6E0E6,
        "pale periwinkle": 0xDCDFEE,
        "pale lavender": 0xE2DCE8,
        "soft lilac": 0xE8DDEB,
    },
    "neutral": {
        "soft linen": 0xF1ECE3,
        "pale oat beige": 0xEDE6DA,
        "soft greige": 0xE6E1DA,
        "warm light stone": 0xE3DDD3,
        "pale mushroom": 0xDCD3CC,
        "soft dove grey": 0xDFDFDD,
    },
}
PASTEL_PALETTE = {n: c for g in PASTEL_GROUPS.values() for n, c in g.items()}
AUTO = "auto (по окрасу)"
RANDOM = {"random": list(PASTEL_PALETTE), "random warm": list(PASTEL_GROUPS["warm"]),
          "random cool": list(PASTEL_GROUPS["cool"]), "random neutral": list(PASTEL_GROUPS["neutral"])}


def _hue_dist(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1 - d) * 360


def _auto_pastel(rgb, seed=0, top=3):
    """Background that sets off the coat: a colour roughly opposite on the colour wheel for coloured
    coats, warm pastels for grey and black ones, and deeper tones for white ones so they don't vanish.
    The seed picks one of the `top` best tones, so repeated runs vary but stay harmonious."""
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    scored = []
    for name, c in PASTEL_PALETTE.items():
        ch, cs, cv = colorsys.rgb_to_hsv(*_rgb(c))
        if cs < 0.06:
            score = 0.45  # neutral tones go with anything, but are never the most flattering
        elif s > 0.2 and v > 0.15:
            score = 1 - abs(_hue_dist(h, ch) - 150) / 150
        else:
            score = 1 - _hue_dist(ch, 20 / 360) / 180
        if v > 0.75 and s < 0.25:
            score += (1 - cv) * 3 + cs * 2
        scored.append((score, name))
    scored.sort(reverse=True)
    return scored[seed % min(top, len(scored))][1]


class PetPastelPalette:
    """Pastel background from an extended palette (24 tones in warm, cool and neutral groups).
    "auto" picks the tone that suits the animal's coat; "random ..." picks by seed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "color": ([AUTO] + list(RANDOM) + list(PASTEL_PALETTE),),
            "width": ("INT", {"default": 896, "min": 64, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 1120, "min": 64, "max": 4096, "step": 16}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                             "tooltip": "Random choices use it; auto uses it to pick one of the 3 best tones."}),
        }, "optional": {
            "image": ("IMAGE", {"tooltip": "Photo of the animal, for the automatic choice."}),
            "mask": ("MASK", {"tooltip": "Mask of the animal on that photo."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("canvas", "color_name")
    FUNCTION = "make"
    CATEGORY = "image"

    def make(self, color, width, height, seed, image=None, mask=None):
        if color == AUTO:
            if image is None:
                color = "pale warm cream"
            else:
                img = image[0]
                m = torch.ones(img.shape[:2]) if mask is None else mask[0]
                if m.shape != img.shape[:2]:
                    m = F.interpolate(m[None, None].float(), size=img.shape[:2], mode="bilinear")[0, 0]
                px = img[m.to(img.device) > 0.5]
                coat = (px.median(dim=0).values if len(px) else img.reshape(-1, 3).median(dim=0).values).tolist()
                color = _auto_pastel(coat, seed)
        elif color in RANDOM:
            names = RANDOM[color]
            color = names[torch.randint(len(names), (1,), generator=torch.Generator().manual_seed(seed)).item()]
        pixel = torch.tensor(_rgb(PASTEL_PALETTE[color]), dtype=torch.float32)
        return (pixel.expand(1, height, width, 3).clone(), color)


class PetFaceMask:
    """Per-pixel strength for the painterly pass (Differential Diffusion + Set Latent Noise Mask):
    1 on fur and body, `face_strength` on the face box, so the eyes and nose stay as painted by FLUX."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Sets the mask size."}),
            "face_box": ("STRING", {"forceInput": True, "tooltip": 'Text with one or more [x1, y1, x2, y2] in 0-1000 coordinates.'}),
            "face_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Share of the pass's denoise applied to the face: 0.5 with denoise 0.48 repaints the face at ~0.24."}),
            "grow": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                               "tooltip": "Enlarge the face box by this share of its size."}),
            "feather": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "Width of the transition to full strength, as a share of the face size."}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "make"
    CATEGORY = "mask"

    def make(self, image, face_box, face_strength, grow, feather):
        h, w = image.shape[1:3]
        mask = torch.ones(1, h, w)
        xs = torch.arange(w, dtype=torch.float32) + 0.5
        ys = torch.arange(h, dtype=torch.float32) + 0.5
        # one box per face, so a group portrait protects every animal's face
        for box in re.findall(r"\[([^\[\]]*)\]", face_box):
            nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", box)[:4]]
            if len(nums) != 4 or nums[2] <= nums[0] or nums[3] <= nums[1]:
                continue
            x0, y0, x1, y1 = nums[0] * w / 1000, nums[1] * h / 1000, nums[2] * w / 1000, nums[3] * h / 1000
            gx, gy = (x1 - x0) * grow / 2, (y1 - y0) * grow / 2
            x0, y0, x1, y1 = x0 - gx, y0 - gy, x1 + gx, y1 + gy
            f = max(1.0, feather * min(x1 - x0, y1 - y0))
            # distance outside the box, 0 inside; full strength once it reaches the feather width
            dx = (x0 - xs).clamp(min=0) + (xs - x1).clamp(min=0)
            dy = (y0 - ys).clamp(min=0) + (ys - y1).clamp(min=0)
            t = (torch.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2) / f).clamp(0, 1)
            t = t * t * (3 - 2 * t)
            mask[0] = torch.minimum(mask[0], face_strength + (1 - face_strength) * t)
        return (mask,)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class PetSplitAnimals:
    """One crop per animal, from Qwen3-VL boxes ("bbox_2d" in 0-1000 coordinates). The output is a
    list, so every node after it runs once per animal and the workflow saves one portrait each.
    With one animal (or none found) the photo passes through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "boxes": ("STRING", {"forceInput": True, "tooltip": 'Text with one or more "bbox_2d": [x1, y1, x2, y2].'}),
            "pad": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01,
                              "tooltip": "Margin added around each box, as a share of the box size."}),
            "min_size": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "Skip animals smaller than this share of the largest one (background pets, reflections)."}),
            "max_animals": ("INT", {"default": 6, "min": 1, "max": 20}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("animals",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split"
    CATEGORY = "image/transform"

    def split(self, image, boxes, pad, min_size, max_animals):
        img = image[:1]
        h, w = img.shape[1:3]
        found = []
        for m in re.finditer(r'"bbox_2d"\s*:\s*\[([^\]]*)\]', boxes):
            nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))[:4]]
            if len(nums) == 4 and nums[2] > nums[0] and nums[3] > nums[1]:
                found.append((nums[0] * w / 1000, nums[1] * h / 1000, nums[2] * w / 1000, nums[3] * h / 1000))
        found.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        kept = []
        for b in found:
            if all(_iou(b, k) < 0.6 for k in kept):  # the model sometimes lists one animal twice
                kept.append(b)
        if kept:
            largest = (kept[0][2] - kept[0][0]) * (kept[0][3] - kept[0][1])
            kept = [b for b in kept if (b[2] - b[0]) * (b[3] - b[1]) >= min_size * largest][:max_animals]
        if len(kept) < 2:
            return ([img],)
        kept.sort(key=lambda b: b[0])  # left to right, so file numbers follow the photo
        crops = []
        for x0, y0, x1, y1 in kept:
            px, py = (x1 - x0) * pad, (y1 - y0) * pad
            cx0, cy0 = max(0, int(x0 - px)), max(0, int(y0 - py))
            cx1, cy1 = min(w, int(x1 + px + 0.5)), min(h, int(y1 + py + 0.5))
            crops.append(img[:, cy0:cy1, cx0:cx1].clone())
        return (crops,)


NODE_CLASS_MAPPINGS = {"PetPortraitLayout": PetPortraitLayout, "PetHeadPortraitLayout": PetHeadPortraitLayout,
                       "PetRestoreColors": PetRestoreColors, "PetFlattenBackground": PetFlattenBackground,
                       "PetPastelBackground": PetPastelBackground, "PetPastelPalette": PetPastelPalette,
                       "PetPortraitLayoutPadded": PetPortraitLayoutPadded, "PetColorPalette": PetColorPalette,
                       "PetSplitAnimals": PetSplitAnimals, "PetFaceMask": PetFaceMask, "PetGroupLayout": PetGroupLayout,
                       "PetMatchColors": PetMatchColors, "PetCropToBox": PetCropToBox}
NODE_DISPLAY_NAME_MAPPINGS = {"PetPortraitLayout": "Pet Portrait Layout", "PetHeadPortraitLayout": "Pet Head Portrait Layout",
                              "PetRestoreColors": "Pet Restore Colors", "PetFlattenBackground": "Pet Flatten Background",
                              "PetPastelBackground": "Pet Pastel Background",
                              "PetPastelPalette": "Pet Pastel Palette (experimental)",
                              "PetPortraitLayoutPadded": "Pet Portrait Layout (padded, experimental)",
                              "PetColorPalette": "Pet Color Palette (experimental)",
                              "PetSplitAnimals": "Pet Split Animals (experimental)",
                              "PetFaceMask": "Pet Face Mask (experimental)",
                              "PetGroupLayout": "Pet Group Layout (experimental)",
                              "PetMatchColors": "Pet Match Colors (experimental)",
                              "PetCropToBox": "Pet Crop To Box (experimental)"}
